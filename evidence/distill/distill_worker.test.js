// The worker is the one write chokepoint: every distilled string lands well-formed (recorded text can
// carry lone surrogates, and a strict reader — Python's sqlite3 — refuses ill-formed text), and a type
// code outside the generated schema counts as unknown. Driven as a real Worker against a real db file,
// the same seam the distillation CLI uses.
import { Database } from "bun:sqlite";
import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deflateSync } from "node:zlib";
import { DISTILLED_COLUMNS } from "./distill.js";
import { EventType, IncrementalSource } from "./rrweb_constants.js";

// A type code the generated schema does not name, computed from the schema itself so no rrweb
// numeric is ever hand-written here.
const UNKNOWN_TYPE = Math.max(...Object.values(EventType)) + 1;

const send = (worker, message) =>
	new Promise((resolve, reject) => {
		worker.onmessage = (e) => resolve(e.data);
		worker.onerror = (e) => reject(new Error(String(e?.message ?? e)));
		worker.postMessage(message);
	});

test("the worker writes well-formed columns and counts schema-unknown kinds", async () => {
	const dir = mkdtempSync(join(tmpdir(), "locus-distill-"));
	try {
		const path = join(dir, "events.db");
		const db = new Database(path);
		db.exec(
			`CREATE TABLE events (id INTEGER PRIMARY KEY, visitor_id TEXT, slice_id TEXT, timestamp INTEGER, counter INTEGER, raw_json BLOB, ${DISTILLED_COLUMNS.map((c) => `"${c}"`).join(", ")})`,
		);
		const insert = db.prepare(
			"INSERT INTO events (id, visitor_id, slice_id, timestamp, counter, raw_json) VALUES (?, ?, ?, ?, ?, ?)",
		);
		const pack = (event) => deflateSync(JSON.stringify(event));
		insert.run(
			1,
			"v1",
			"s1",
			1000,
			0,
			pack({
				type: EventType.Meta,
				timestamp: 1000,
				data: { href: "https://x.test/\ud83d?q=1" },
			}),
		);
		insert.run(
			2,
			"v1",
			"s1",
			2000,
			0,
			pack({
				type: EventType.IncrementalSnapshot,
				timestamp: 2000,
				data: { source: IncrementalSource.Input, id: 7, text: "ok \ud800 cut" },
			}),
		);
		insert.run(
			3,
			"v1",
			"s1",
			3000,
			0,
			pack({ type: UNKNOWN_TYPE, timestamp: 3000, data: {} }),
		);
		db.close();

		const worker = new Worker(
			new URL("./distill_worker.js", import.meta.url).href,
		);
		try {
			expect(await send(worker, { open: path })).toEqual({ opened: true });
			const result = await send(worker, { visitors: ["v1"] });
			expect(result.events).toBe(3);
			expect(result.unknownKinds).toBe(1);
		} finally {
			worker.terminate();
		}

		const check = new Database(path);
		const rows = check
			.query("SELECT id, type_str, url, input FROM events ORDER BY id")
			.all();
		check.close();
		expect(rows[0].type_str).toBe("Meta");
		expect(rows[0].url).toBe("https://x.test/�?q=1");
		expect(rows[1].input).toBe("ok � cut");
		expect(rows[2].type_str).toBe(String(UNKNOWN_TYPE));
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
