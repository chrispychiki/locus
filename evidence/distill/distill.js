// Distillation pass over the keystone DB: flattens each raw event into the queryable columns, projects each
// FullSnapshot's DOM to markdown, and collapses each contiguous mutation run into a literal diff against the
// prior projection. Mechanical only — no salience, no labels. Run:
//   bun distill/distill.js <events.db>
//
// The slice is the unit: each replayable slice is self-contained (its own FullSnapshot head) and is rendered
// fresh in isolation by the screenshot renderer, so distillation mirrors that — a fresh LightweightMirror per slice,
// never bleeding DOM state across slices.

import { diffLines } from "./diff.js";
import { flatten } from "./flatten.js";
import { LightweightMirror } from "./mirror.js";
import { projectMarkdown } from "./project.js";
import { parseRaw } from "./raw.js";
import { EventType, IncrementalSource } from "./rrweb_constants.js";

// The columns distillation writes — the one declaration of them. The worker builds its UPDATE from this list and
// binds the values distilledValues yields in this order.
export const DISTILLED_COLUMNS = [
	"type_str",
	"url",
	"page_url",
	"tag",
	"class",
	"text",
	"x",
	"y",
	"input",
	"href",
	"title",
	"referrer",
	"pointer_type",
	"extra",
	"hidden",
	"md",
	"diff",
];

// One update's value for each distilled column, in DISTILLED_COLUMNS order — the one statement of
// which field feeds which column, beside the declaration so the UPDATE's positional bind can never
// drift from it. Three columns are the update's own (the forward-filled page, the projection, the
// run's diff); everything else is the flattened column of the same name.
export function distilledValues(update) {
	return DISTILLED_COLUMNS.map((column) => {
		if (column === "page_url") return update.pageUrl;
		if (column === "md") return update.md;
		if (column === "diff") return update.diff;
		return update.cols[column];
	});
}

// Mutation residue: events the replayer (and so the mirror) aborts partway, drops, or skips on unresolvable
// ids. Not data loss — the screenshots show the same — but a mechanical fact, counted, never silent. wrongTyped is
// flatten's count of wire values its columns have no honest projection for (open-write store; raw_json keeps
// them) — dropped to null rather than crashing the DB write. One shape, so a new counter reaches every total.
export const newResidue = () => ({
	aborted: 0,
	droppedAdds: 0,
	unresolvedRefs: 0,
	wrongTyped: 0,
});

export function addResidue(total, part) {
	for (const k of Object.keys(part)) total[k] = (total[k] ?? 0) + part[k];
	return total;
}

// Distills one slice's events against a single fresh mirror. step() returns the per-event update: the
// flattened columns for every event, the markdown projection at each FullSnapshot, and — when a contiguous
// run of mutations ends — the diff of the projection across that run.
export class SliceDistiller {
	constructor() {
		this.mirror = new LightweightMirror();
		this.rootId = null;
		this.priorProjection = "";
		this.run = null;
		this.lastUrl = null;
		this.residue = newResidue();
	}

	flushRun() {
		if (!this.run) return;
		const projection =
			this.rootId != null ? projectMarkdown(this.mirror, this.rootId) : "";
		this.run.diff = diffLines(this.priorProjection, projection);
		this.priorProjection = projection;
		this.run = null;
	}

	step(event, id) {
		const isMutation =
			event.type === EventType.IncrementalSnapshot &&
			event.data?.source === IncrementalSource.Mutation;
		if (!isMutation) this.flushRun();

		const cols = flatten(event, this.mirror, this.residue);
		if (cols.url) this.lastUrl = cols.url;
		const update = { id, cols, pageUrl: this.lastUrl, md: null, diff: null };

		const report = this.mirror.applyEvent(event);
		if (report) {
			if (report.aborted) this.residue.aborted += 1;
			this.residue.droppedAdds += report.droppedAdds.length;
			this.residue.unresolvedRefs +=
				report.missingRemoves.length +
				report.missingTexts.length +
				report.missingAttributes.length;
		}

		if (event.type === EventType.FullSnapshot) {
			this.rootId = event.data?.node?.id ?? null;
			const md =
				this.rootId != null ? projectMarkdown(this.mirror, this.rootId) : "";
			this.priorProjection = md;
			update.md = md;
		} else if (isMutation) {
			this.run = update;
		}

		return update;
	}

	finish() {
		this.flushRun();
	}
}

// Distill all of a visitor's events (canonical order), each slice against its own distiller. A visitor's
// concurrent page contexts interleave their slices in the canonical stream, so a slice's events need not be
// contiguous there — each event steps the distiller of the slice it is stamped with, which holds that slice's
// mirror and projection state across the interleaving, and a distiller folds away after its slice's last
// event. Events before the first slice and events in discarded slices flow through the same way, keyed by
// their (null or discarded) slice — an empty mirror, their non-DOM columns, no projection. Returns the
// per-event updates in input order plus the summed mutation residue.
export function distillVisitor(rows) {
	const updates = [];
	const residue = newResidue();
	const lastIndex = new Map();
	rows.forEach((row, i) => {
		lastIndex.set(row.slice_id, i);
	});
	const distillers = new Map();
	rows.forEach((row, i) => {
		let distiller = distillers.get(row.slice_id);
		if (!distiller) {
			distiller = new SliceDistiller();
			distillers.set(row.slice_id, distiller);
		}
		updates.push(distiller.step(parseRaw(row.raw_json), row.id));
		if (lastIndex.get(row.slice_id) === i) {
			distiller.finish();
			addResidue(residue, distiller.residue);
			distillers.delete(row.slice_id);
		}
	});
	return { updates, residue };
}

if (import.meta.main) {
	const { Database } = await import("bun:sqlite");
	const dbPath = process.argv[2];
	if (!dbPath) {
		console.error("usage: bun distill/distill.js <events.db>");
		process.exit(1);
	}

	// Incremental by default: distillation always writes type_str, so a visitor holding any type_str-NULL event
	// has rows a load just landed and not yet distilled — distill only those. `--all` re-distills the whole store,
	// for when the projection logic itself changed and every visitor's output is now stale.
	const all = process.argv.includes("--all");

	// Visitor is the parallelism unit: each worker owns a disjoint set of visitors, its own mirror, and writes
	// its own batches; the DB (WAL) is the only shared resource. A visitor is atomic to its worker, so nothing
	// splits a heavy one — round-robin only keeps neighbouring visitors (adjacent in id, often adjacent in
	// weight) from clustering onto the same shard.
	const probe = new Database(dbPath);
	const visitors = probe
		.query(
			all
				? "SELECT DISTINCT visitor_id FROM events"
				: "SELECT DISTINCT visitor_id FROM events WHERE type_str IS NULL",
		)
		.all()
		.map((r) => r.visitor_id);
	probe.close();

	// The fallback covers a host that does not report hardwareConcurrency, at a shard count small
	// enough to oversubscribe no small machine.
	const workerCount = Math.max(
		1,
		Math.min(visitors.length, navigator.hardwareConcurrency || 4),
	);
	const shards = Array.from({ length: workerCount }, (_, i) =>
		visitors.filter((_, j) => j % workerCount === i),
	).filter((s) => s.length);

	const t0 = Bun.nanoseconds();
	// One worker failing means the distillation is incomplete: every sibling is torn down and the process dies
	// loud, rather than reporting totals for a pass that half ran.
	const workers = shards.map(
		() => new Worker(new URL("./distill_worker.js", import.meta.url).href),
	);
	// Opens are serialized (each worker acknowledges its open before the next begins): simultaneous
	// bun:sqlite opens of one file from sibling Workers intermittently read an empty schema
	// ("no such table", Bun 1.3.14). Compute below stays fully parallel.
	for (const w of workers) {
		await new Promise((resolve, reject) => {
			w.onmessage = (e) => resolve(e.data);
			w.onerror = (e) =>
				reject(new Error(`distillation worker failed: ${e?.message ?? e}`));
			w.postMessage({ open: dbPath });
		});
	}
	const results = await Promise.all(
		workers.map(
			(w, i) =>
				new Promise((resolve, reject) => {
					w.onmessage = (e) => resolve(e.data);
					w.onerror = (e) =>
						reject(new Error(`distillation worker failed: ${e?.message ?? e}`));
					w.postMessage({ visitors: shards[i] });
				}),
		),
	).finally(() => {
		for (const w of workers) w.terminate();
	});
	const wallMs = (Bun.nanoseconds() - t0) / 1e6;

	let events = 0,
		computeMs = 0,
		unknownKinds = 0;
	const residue = newResidue();
	for (const r of results) {
		events += r.events;
		computeMs += r.computeMs;
		unknownKinds += r.unknownKinds;
		addResidue(residue, r.residue);
	}

	console.log(
		`distilled ${events} events across ${visitors.length} visitors on ${shards.length} workers`,
	);
	console.log(
		`wall ${(wallMs / 1000).toFixed(1)}s | pipeline compute ${(computeMs / 1000).toFixed(1)}s | ${((computeMs * 1000) / events).toFixed(1)} us/event`,
	);
	if (residue.aborted || residue.droppedAdds || residue.unresolvedRefs) {
		console.log(
			`mutation residue (replayer-faithful, not data loss): ${residue.aborted} aborted mutations, ` +
				`${residue.droppedAdds} dropped adds, ${residue.unresolvedRefs} ops on unresolvable ids`,
		);
	}
	if (residue.wrongTyped) {
		console.log(
			`WARNING: dropped ${residue.wrongTyped} wire values their columns cannot hold ` +
				`(open-write store; columns null, raw_json intact)`,
		);
	}
	if (unknownKinds > 0) {
		console.log(
			`WARNING: ${unknownKinds} events carry type codes outside the generated schema — extend extract_rrweb_constants.js`,
		);
	}
}
