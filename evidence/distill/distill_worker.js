// One distillation worker: distills an assigned set of visitors, each against its own mirror, and writes its
// per-visitor batches. Visitor is the parallelism unit — within a visitor replay is serial (one stateful
// mirror, reset per slice), across visitors it is independent (own mirror, own writes; the DB is the only
// shared resource, serialized by WAL). Reports the compute time of the pipeline itself (flatten + mirror +
// projectMarkdown + diffLines), separate from the DB write.
import { Database } from "bun:sqlite";
import {
	addResidue,
	DISTILLED_COLUMNS,
	distilledValues,
	distillVisitor,
	newResidue,
} from "./distill.js";
import { CANONICAL_ORDER } from "./raw.js";
import {
	EventTypeNames,
	IncrementalSourceNames,
	MediaInteractionsNames,
	MouseInteractionsNames,
} from "./rrweb_constants.js";

// Every type_str the generated schema can produce; anything else is a numeric code the
// schema doesn't cover — the generated constants drifting behind the recorder's rrweb.
const KNOWN_KINDS = new Set([
	...Object.values(EventTypeNames),
	...Object.values(IncrementalSourceNames),
	...Object.values(MouseInteractionsNames),
	...Object.values(MediaInteractionsNames),
]);

let db, update, select;

// Two-phase: the open is its own message, acknowledged, so the parent can serialize opens across
// workers — simultaneous bun:sqlite opens of one file from sibling Workers intermittently read an
// empty schema ("no such table", Bun 1.3.14). Compute stays fully parallel.
self.onmessage = (e) => {
	if (e.data.open) {
		db = new Database(e.data.open);
		// The other workers must WAIT for a held write lock, not error out (SQLITE_BUSY). busy_timeout
		// makes every locked operation block up to the timeout instead of throwing — writes serialize
		// on the DB (the one shared resource) while the compute stays parallel across visitors.
		db.exec("PRAGMA busy_timeout = 120000");
		// Built from the one declaration of the distilled columns, and never guarded: a prepare that fails
		// here is a real failure — a locked db, a failing disk — and it takes the whole distillation down
		// rather than quietly writing nothing.
		update = db.prepare(
			`UPDATE events SET ${DISTILLED_COLUMNS.map((c) => `"${c}"=?`).join(", ")} WHERE id=?`,
		);
		select = db.query(
			`SELECT id, slice_id, raw_json FROM events WHERE visitor_id = ? ${CANONICAL_ORDER}`,
		);
		postMessage({ opened: true });
		return;
	}
	const { visitors } = e.data;
	let events = 0;
	let unknownKinds = 0;
	let computeNs = 0;
	const residue = newResidue();

	for (const visitor_id of visitors) {
		const rows = select.all(visitor_id);
		const t = Bun.nanoseconds();
		const { updates, residue: vr } = distillVisitor(rows);
		computeNs += Bun.nanoseconds() - t;
		addResidue(residue, vr);
		events += updates.length;
		for (const u of updates)
			if (!KNOWN_KINDS.has(u.cols.type_str)) unknownKinds += 1;
		// Recorded text can carry lone surrogates (raw_json holds them faithfully,
		// as \uXXXX escapes; JSON.parse turns them back into real surrogates).
		// The flat columns must be well-formed UTF-8 — a strict reader (Python's
		// sqlite3) refuses ill-formed text — so every string is sanitized (lone
		// surrogates → U+FFFD) at this one write chokepoint.
		const wf = (v) => (typeof v === "string" ? v.toWellFormed() : v);
		db.transaction(() => {
			for (const u of updates) {
				// wf sanitizes every string and passes everything else through.
				update.run(...distilledValues(u).map(wf), u.id);
			}
		})();
	}
	db.close();
	postMessage({ events, computeMs: computeNs / 1e6, unknownKinds, residue });
};
