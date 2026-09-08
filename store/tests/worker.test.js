import { describe, expect, test } from "bun:test";
import { gunzipSync, gzipSync, zlibSync } from "fflate";

import { bodySniff, chunkTarget, MAX_CHUNK_BYTES } from "../src/keys.js";
import { AE_BLOB_BYTES, fitted } from "../src/telemetry.js";
import worker from "../src/worker.js";

const SNIPPET = "abc123";
const VISITOR = "0f6e2a7c-9b1d-4e5f-8a3b-2c4d6e8f0a1b";
const SLICE = "01749600000000-x7k2";
const COUNTER = "1749600000000000001";
const DATE = "2025-06-11";
const PATH = `/chunks/${SNIPPET}/${VISITOR}/${SLICE}/${COUNTER}`;

function fakeEnv() {
	const objects = new Map();
	const points = [];
	return {
		objects,
		points,
		CHUNKS: {
			put: async (key, bytes) => {
				objects.set(key, bytes);
			},
		},
		CAPTURE: {
			writeDataPoint: (p) => points.push(p),
		},
	};
}

function put(path, body, method = "PUT") {
	return new Request(`https://store.test${path}`, { method, body });
}

const GZIP_BODY = gzipSync(new TextEncoder().encode('{"events":[]}'));

// Every datapoint's blobs are padded to the fixed own-blob width and then carry the three
// request facts (Origin, User-Agent, AS organization) at uniform tail positions — empty here
// because a bare test Request attests none of them.
const stamped = (blobs, facts = ["", "", ""]) => [
	...blobs,
	...Array(7 - blobs.length).fill(""),
	...facts,
];

describe("chunkTarget", () => {
	test("maps the upload path to the snippet-first key layout, with its segments", () => {
		expect(chunkTarget(PATH)).toEqual({
			key: `${SNIPPET}/${DATE}/${VISITOR}/${SLICE}/${COUNTER}.json.gz`,
			snippetId: SNIPPET,
			visitorId: VISITOR,
			sliceId: SLICE,
		});
	});

	test("a chunk stamped after midnight files under its slice's start day, never its own", () => {
		// The date partition is the slice's start, so every chunk of a slice lands under one
		// prefix — a slice straddling midnight is not split, and a late arrival self-places by the
		// slice id it was stamped with however long after it shows up.
		const openMs = 1749686370000; // 2025-06-11T23:59:30Z
		const slice = `${String(openMs).padStart(14, "0")}-x7k2`;
		const counter = "1749688860000000123"; // stamped 2025-06-12T00:41:00Z
		expect(
			chunkTarget(`/chunks/${SNIPPET}/${VISITOR}/${slice}/${counter}`).key,
		).toBe(`${SNIPPET}/2025-06-11/${VISITOR}/${slice}/${counter}.json.gz`);
	});

	test("the date partition is the UTC day, whatever timezone the worker's host reads as", () => {
		// Every derived day in Locus is the UTC day. A host-local derivation would file the same
		// slice under two different prefixes depending on where the code ran, and `locus ls`'s
		// date-prefix enumeration would miss it.
		const openMs = 1749686370000; // 2025-06-11T23:59:30Z — Kiritimati (+14) is already on the
		// 12th at this instant, so a host-local derivation there would file it a day forward.
		const slice = `${String(openMs).padStart(14, "0")}-x7k2`;
		const before = process.env.TZ;
		try {
			const dates = ["America/Los_Angeles", "Pacific/Kiritimati", "UTC"].map(
				(tz) => {
					process.env.TZ = tz;
					return chunkTarget(
						`/chunks/${SNIPPET}/${VISITOR}/${slice}/${COUNTER}`,
					).key.split("/")[1];
				},
			);
			expect(dates).toEqual(["2025-06-11", "2025-06-11", "2025-06-11"]);
		} finally {
			if (before === undefined) delete process.env.TZ;
			else process.env.TZ = before;
		}
	});

	test("bodySniff reads the magic bytes, not the length", () => {
		expect(bodySniff(GZIP_BODY)).toBe("gzip");
		expect(bodySniff(new Uint8Array([0x1f, 0x8b]))).toBe("other"); // magic but no body
		expect(bodySniff(new TextEncoder().encode('{"events":[]}'))).toBe("json");
		expect(bodySniff(new TextEncoder().encode("  \n[1,2]"))).toBe("json");
		expect(bodySniff(zlibSync(new TextEncoder().encode("{}")))).toBe("deflate");
		expect(bodySniff(new Uint8Array(MAX_CHUNK_BYTES))).toBe("other"); // big, but zeros
	});

	test("rejects traversal, malformed segments, and foreign paths", () => {
		expect(chunkTarget("/chunks/../secret/x/1/2")).toBeNull();
		expect(chunkTarget(`/chunks/${VISITOR}/${SLICE}/${COUNTER}`)).toBeNull();
		expect(
			chunkTarget(`/chunks/BAD!/${VISITOR}/${SLICE}/${COUNTER}`),
		).toBeNull();
		expect(
			chunkTarget(`/chunks/${SNIPPET}/${VISITOR}/not-a-slice/${COUNTER}`),
		).toBeNull();
		expect(
			chunkTarget(`/chunks/${SNIPPET}/${VISITOR}/${SLICE}/abc`),
		).toBeNull();
		// A bare ms, which the recorder's counter stamp never is.
		expect(
			chunkTarget(`/chunks/${SNIPPET}/${VISITOR}/${SLICE}/1749600000000`),
		).toBeNull();
		expect(chunkTarget(`/chunks/${SNIPPET}/v/${SLICE}/${COUNTER}`)).toBeNull();
		expect(chunkTarget("/locus-recorder.min.js")).toBeNull();
		expect(chunkTarget("/")).toBeNull();
	});
});

describe("worker", () => {
	test("stores a valid PUT under the derived key", async () => {
		const env = fakeEnv();
		const response = await worker.fetch(put(PATH, GZIP_BODY), env);
		expect(response.status).toBe(204);
		expect(response.headers.get("Access-Control-Allow-Origin")).toBe("*");
		expect([...env.objects.keys()]).toEqual([
			`${SNIPPET}/${DATE}/${VISITOR}/${SLICE}/${COUNTER}.json.gz`,
		]);
		// A clean gzip arrival writes no AE row — the observability plane carries exceptions, not traffic.
		expect(env.points).toEqual([]);
	});

	test("every answer carries the store's stamp, exposed cross-origin — how the recorder tells a refusal of this worker's from a middlebox's (recorder/src/sink.js)", async () => {
		const env = fakeEnv();
		for (const request of [
			put(PATH, GZIP_BODY),
			put("/admin", GZIP_BODY),
			new Request(`https://store.test${PATH}`, { method: "GET" }),
		]) {
			const response = await worker.fetch(request, env);
			expect(response.headers.get("Locus-Store")).toBe("1");
			expect(response.headers.get("Access-Control-Expose-Headers")).toBe(
				"Locus-Store",
			);
		}
	});

	test("a gzip arrival is stored as the bytes that arrived — never wrapped a second time", async () => {
		// The canonical form at rest is one gzip layer: `locus load` ungzips an object once and
		// expects the recorder's payload. A second wrapper stores something that ungzips to gzip,
		// and every consumer reads it as a corrupt chunk.
		const env = fakeEnv();
		await worker.fetch(put(PATH, GZIP_BODY), env);
		const stored = env.objects.get(
			`${SNIPPET}/${DATE}/${VISITOR}/${SLICE}/${COUNTER}.json.gz`,
		);
		expect([...stored]).toEqual([...GZIP_BODY]);
		expect(new TextDecoder().decode(gunzipSync(stored))).toBe('{"events":[]}');
	});

	test("a body declaring itself over the cap is refused on the declaration, without reading it", async () => {
		// The endpoint is open-write: a sender that announces a gigabyte is answered from the
		// header rather than streamed and counted first. The declaration is what bounces here —
		// the bytes behind it would sail under the cap.
		const env = fakeEnv();
		const body = new ReadableStream({
			start(controller) {
				controller.enqueue(GZIP_BODY);
				controller.close();
			},
		});
		const res = await worker.fetch(
			new Request(`https://store.test${PATH}`, {
				method: "POST",
				body,
				headers: { "Content-Length": String(MAX_CHUNK_BYTES + 1) },
				duplex: "half",
			}),
			env,
		);
		expect(res.status).toBe(413);
		expect(env.objects.size).toBe(0);
		expect(env.points.map((p) => p.blobs[0])).toEqual(["upload_rejected"]);
	});

	test("accepts POST identically — sendBeacon's verb", async () => {
		const env = fakeEnv();
		const response = await worker.fetch(put(PATH, GZIP_BODY, "POST"), env);
		expect(response.status).toBe(204);
		expect(env.objects.size).toBe(1);
		expect(env.points).toEqual([]);
	});

	test("a same-key concurrent-write 10058 whose object landed is answered as success — the race's loser is redundant, not failed", async () => {
		const env = fakeEnv();
		env.CHUNKS.head = async (key) => (env.objects.has(key) ? {} : null);
		env.CHUNKS.put = async (key) => {
			env.objects.set(key, GZIP_BODY); // the winner's write is already there
			throw new Error(
				"put: Reduce your concurrent request rate for the same object. (10058)",
			);
		};
		const response = await worker.fetch(put(PATH, GZIP_BODY), env);
		expect(response.status).toBe(204);
		expect(env.points).toEqual([]); // no worker_error — nothing failed
	});

	test("a 10058 with no landed object stays a real failure", async () => {
		const env = fakeEnv();
		env.CHUNKS.head = async () => null;
		env.CHUNKS.put = async () => {
			throw new Error(
				"put: Reduce your concurrent request rate for the same object. (10058)",
			);
		};
		const response = await worker.fetch(put(PATH, GZIP_BODY), env);
		expect(response.status).toBe(500);
		expect(env.points.map((p) => p.blobs[0])).toEqual(["worker_error"]);
	});

	test("answers preflight without touching the bucket", async () => {
		const env = fakeEnv();
		const response = await worker.fetch(
			new Request(`https://store.test${PATH}`, { method: "OPTIONS" }),
			env,
		);
		expect(response.status).toBe(204);
		expect(response.headers.get("Access-Control-Allow-Methods")).toContain(
			"PUT",
		);
		expect(env.objects.size).toBe(0);
	});

	test("bounces oversized, unknown paths, and GET", async () => {
		const oversizeEnv = fakeEnv();
		const big = new Uint8Array(MAX_CHUNK_BYTES + 1);
		big[0] = 0x1f;
		big[1] = 0x8b;
		const oversized = await worker.fetch(put(PATH, big), oversizeEnv);
		expect(oversized.status).toBe(413);
		expect(oversizeEnv.objects.size).toBe(0);

		// A foreign path never parsed, so there is no snippet to attest under: no object, no AE row.
		const foreignEnv = fakeEnv();
		const foreign = await worker.fetch(put("/admin", GZIP_BODY), foreignEnv);
		expect(foreign.status).toBe(404);
		expect(foreignEnv.objects.size).toBe(0);
		expect(foreignEnv.points).toEqual([]);

		const getEnv = fakeEnv();
		const get = await worker.fetch(
			new Request(`https://store.test${PATH}`, { method: "GET" }),
			getEnv,
		);
		expect(get.status).toBe(405);
		expect(getEnv.objects.size).toBe(0);
		expect(getEnv.points).toEqual([]);
	});

	test("a body at exactly the cap is accepted; one past it bounces whatever its shape", async () => {
		const atCapEnv = fakeEnv();
		const atCap = new Uint8Array(MAX_CHUNK_BYTES);
		atCap[0] = 0x1f;
		atCap[1] = 0x8b;
		const accepted = await worker.fetch(put(PATH, atCap), atCapEnv);
		expect(accepted.status).toBe(204);
		expect(atCapEnv.objects.size).toBe(1);
		expect(atCapEnv.points).toEqual([]);

		// The cap gates the received bytes, before any sniff or transcode — a non-gzip oversize
		// body bounces as itself, never ballooning through the gzip wrapper first.
		const jsonEnv = fakeEnv();
		const bigJson = new Uint8Array(MAX_CHUNK_BYTES + 1).fill(0x20);
		bigJson[0] = 0x7b;
		const rejected = await worker.fetch(put(PATH, bigJson), jsonEnv);
		expect(rejected.status).toBe(413);
		expect(jsonEnv.objects.size).toBe(0);
		expect(jsonEnv.points.map((p) => p.blobs[0])).toEqual(["upload_rejected"]);
	});

	test("a non-gzip body is accepted byte-exact under a gzip wrapper and stamped transcoded", async () => {
		const raw = new TextEncoder().encode('{"events":[]}');
		for (const [body, sniff] of [
			[raw, "json"],
			[zlibSync(raw), "deflate"],
			[new Uint8Array([0x1f, 0xef, 0xbf, 0xbd]), "other"],
		]) {
			const env = fakeEnv();
			const response = await worker.fetch(put(PATH, body), env);
			expect(response.status).toBe(204);
			const stored = env.objects.get(
				`${SNIPPET}/${DATE}/${VISITOR}/${SLICE}/${COUNTER}.json.gz`,
			);
			expect(bodySniff(stored)).toBe("gzip");
			expect([...gunzipSync(stored)]).toEqual([...body]);
			expect(env.points).toEqual([
				{
					indexes: [SNIPPET],
					blobs: stamped(["upload_transcoded", VISITOR, SLICE, "", sniff]),
				},
			]);
		}
	});

	test("a slice_started ping writes a snippet-indexed AE datapoint and 204s", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "slice_started",
					sliceId: SLICE,
					recorderVersion: "locus-recorder/0.2.0",
					url: "https://site/p",
					first_slice: true,
					visitor_source: "written",
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"slice_started",
					VISITOR,
					SLICE,
					"locus-recorder/0.2.0",
					"https://site/p",
					"1",
					"written",
				]),
				doubles: [1749600000000],
			},
		]);
		expect(env.objects.size).toBe(0);
	});

	test("a transport_probe — the recorder's channel-assessment shot — lands, on a response whose Timing-Allow-Origin lets the sender read the shot's fate", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "transport_probe",
					sliceId: SLICE,
					recorderVersion: "locus-recorder/0.3.0",
					ts: 1749600000123,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(res.headers.get("Timing-Allow-Origin")).toBe("*");
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"transport_probe",
					VISITOR,
					SLICE,
					"locus-recorder/0.3.0",
				]),
				doubles: [1749600000123],
			},
		]);
	});

	// An upgrade reaches one device at a time, so the worker always outruns some recorder it is
	// replacing. Refusing that recorder's births would drop them out of every count for the length
	// of the upgrade.
	test("a birth from a bundle predating the visitor source lands with it empty", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "slice_started",
					sliceId: SLICE,
					recorderVersion: "locus-recorder/0.2.0",
					url: "https://site/p",
					first_slice: true,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points[0].blobs).toEqual(
			stamped([
				"slice_started",
				VISITOR,
				SLICE,
				"locus-recorder/0.2.0",
				"https://site/p",
				"1",
				"",
			]),
		);
	});

	test("a ping that is not what the recorder emits bounces — no datapoint padded with invented zeros", async () => {
		const env = fakeEnv();
		const bodies = [
			{ metric: "slice_started" }, // fields missing
			{
				metric: "slice_started",
				sliceId: "not-a-slice", // malformed slice id
				url: "https://site/p",
				first_slice: true,
			},
			{
				metric: "cost_sample",
				sliceId: SLICE,
				mainThreadMs: "12", // wrong type
				uploadBytes: 1,
				deliveredBytes: 1,
				heapBytesMax: 1,
				heapBytesLimit: 1,
				backlogBytesMax: 1,
			},
			{ metric: "recorder_fault", sliceId: SLICE, reason: "x" }, // error missing
			{
				metric: "slice_started",
				sliceId: SLICE,
				url: "https://site/p",
				first_slice: true,
				recorderVersion: 5,
			}, // version wrong type
			{
				metric: "slice_started",
				sliceId: SLICE,
				url: "https://site/p",
				first_slice: true,
				visitor_source: 7,
			}, // visitor source wrong type
			{ metric: "transport_probe", sliceId: "not-a-slice" }, // malformed slice id
			{ metric: "transport_probe", sliceId: SLICE, ts: "12" }, // wrong-typed ts
			{
				metric: "page_load",
				sliceId: SLICE,
				url: "https://site/p",
				spaRoute: true,
			}, // referrer missing
			{
				metric: "page_load",
				sliceId: SLICE,
				url: "https://site/p",
				referrer: "",
				spaRoute: "yes",
			}, // wrong-typed spaRoute
			{
				metric: "chunk_oversize",
				sliceId: SLICE,
				gzippedBytes: 1234,
				count: "7",
			}, // wrong-typed count
			{ metric: "snapshot_fatal", sliceId: SLICE, count: 7 }, // gzippedBytes missing
			{ metric: "capture_gated", reason: "bot" }, // url missing
			{ metric: "capture_gated", reason: 7, url: "https://site/p" }, // wrong-typed reason
			{
				metric: "capture_gated",
				reason: "bot",
				url: "https://site/p",
				detail: 7,
			}, // wrong-typed detail
			{ metric: "made_up_metric" }, // unknown metric
			"just a string",
			null,
		];
		for (const body of bodies) {
			const res = await worker.fetch(
				new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
					method: "POST",
					body: JSON.stringify(body),
				}),
				env,
			);
			expect(res.status).toBe(400);
		}
		const notJson = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: "{oops",
			}),
			env,
		);
		expect(notJson.status).toBe(400);
		// Every bounce is attested — a rejection is never traceless.
		expect(env.points.map((p) => p.blobs[0])).toEqual(
			[...bodies, "{oops"].map(() => "telemetry_rejected"),
		);
		// The reason names what bounced: a metric that failed its shape, by name, and a body with
		// no metric to name as exactly that.
		expect(env.points.map((p) => p.blobs[4])).toEqual([
			"shape:slice_started",
			"shape:slice_started",
			"shape:cost_sample",
			"shape:recorder_fault",
			"shape:slice_started",
			"shape:slice_started",
			"shape:transport_probe",
			"shape:transport_probe",
			"shape:page_load",
			"shape:page_load",
			"shape:chunk_oversize",
			"shape:snapshot_fatal",
			"shape:capture_gated",
			"shape:capture_gated",
			"shape:capture_gated",
			"shape:made_up_metric",
			"not_an_object",
			"not_an_object",
			"not_json",
		]);
	});

	test("an oversized blob is fitted, never allowed to take the whole datapoint down", async () => {
		const env = fakeEnv();
		const url = `https://site/${"q".repeat(9000)}`;
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "slice_started",
					sliceId: SLICE,
					url,
					first_slice: true,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		const [point] = env.points;
		expect(point.blobs[0]).toBe("slice_started");
		// The identifying blobs survive whole; only the free text is cut.
		expect(point.blobs[1]).toBe(VISITOR);
		expect(point.blobs[2]).toBe(SLICE);
		expect(point.blobs[4].length).toBeLessThan(url.length);
		const bytes = point.blobs.reduce(
			(n, b) => n + new TextEncoder().encode(b).length,
			0,
		);
		expect(bytes).toBeLessThanOrEqual(AE_BLOB_BYTES);
	});

	test("the blob budget stays byte-exact through a multibyte cut — never a character overrun", () => {
		const enc = new TextEncoder();
		const totalBytes = (blobs) =>
			blobs.reduce((n, b) => n + enc.encode(b).length, 0);
		// Sweep the cut across every UTF-8 alignment: 3-byte characters at three offsets, and
		// 4-byte (astral) characters whose raw cut would mint a lone surrogate.
		for (const [pad, char] of [
			[0, "☃"],
			[1, "☃"],
			[2, "☃"],
			[0, "🙂"],
			[3, "🙂"],
		]) {
			const blobs = fitted(["x".repeat(pad) + char.repeat(AE_BLOB_BYTES)]);
			expect(totalBytes(blobs)).toBeLessThanOrEqual(AE_BLOB_BYTES);
			for (const b of blobs) {
				expect(b).toBe(b.toWellFormed());
				expect(b).not.toContain("�");
			}
		}
	});

	test("a multibyte URL cut at the boundary keeps the whole datapoint inside the budget", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "slice_started",
					sliceId: SLICE,
					url: `https://site/${"☃".repeat(4000)}`,
					first_slice: true,
					visitor_source: "written",
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		const [point] = env.points;
		expect(point.blobs[0]).toBe("slice_started");
		const bytes = point.blobs.reduce(
			(n, b) => n + new TextEncoder().encode(b).length,
			0,
		);
		expect(bytes).toBeLessThanOrEqual(AE_BLOB_BYTES);
		expect(point.blobs[4]).not.toContain("�");
	});

	test("a cost_sample from a heap-less platform carries nulls, encoded as the -1 sentinel, never a fake 0", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "cost_sample",
					sliceId: null,
					mainThreadMs: 12.5,
					uploadBytes: 3400,
					deliveredBytes: 3400,
					heapBytesMax: null,
					heapBytesLimit: null,
					backlogBytesMax: 51200,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped(["cost_sample", VISITOR, "", ""]),
				doubles: [12.5, 3400, -1, -1, 51200, 3400, -1, -1],
			},
		]);
	});

	test("a cost sample from the recorder being replaced still lands, its newer figures unmeasured", async () => {
		// An upgrade reaches one device at a time, so the worker always outruns the bundle. A gate that
		// refuses the pings of the recorder it is replacing goes blind through exactly the window it
		// exists to watch — and a figure the old bundle never measured is unmeasured, not zero.
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "cost_sample",
					sliceId: SLICE,
					mainThreadMs: 12.5,
					uploadBytes: 3400,
					heapBytesMax: 9000000,
					heapBytesLimit: 2147483648,
					backlogBytesMax: 51200,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points[0].doubles).toEqual([
			12.5, 3400, 9000000, 2147483648, 51200, -1, -1, -1,
		]);
	});

	test("a backlog nobody ever read rides as unmeasured — a device holding a whole recording cannot report the same 0 as an idle one", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "cost_sample",
					sliceId: SLICE,
					mainThreadMs: 4,
					uploadBytes: 0,
					deliveredBytes: 0,
					heapBytesMax: 9000000,
					heapBytesLimit: 2147483648,
					backlogBytesMax: null,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points[0].doubles).toEqual([
			4, 0, 9000000, 2147483648, -1, 0, -1, -1,
		]);
	});

	test("a page_load ping writes a snippet-indexed AE datapoint and 204s", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "page_load",
					sliceId: SLICE,
					url: "https://site/p2",
					referrer: "https://ref/x",
					spaRoute: true,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"page_load",
					VISITOR,
					SLICE,
					"",
					"https://site/p2",
					"https://ref/x",
					"1",
				]),
				doubles: [-1],
			},
		]);
		expect(env.objects.size).toBe(0);
	});

	test("a cost_sample ping writes blobs plus doubles", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "cost_sample",
					sliceId: SLICE,
					mainThreadMs: 12.5,
					uploadBytes: 3400,
					deliveredBytes: 1200,
					heapBytesMax: 9000000,
					heapBytesLimit: 2147483648,
					backlogBytesMax: 51200,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped(["cost_sample", VISITOR, SLICE, ""]),
				doubles: [12.5, 3400, 9000000, 2147483648, 51200, 1200, -1, -1],
			},
		]);
	});

	test("recorder-health pings (oversize/fatal) write AE datapoints with bytes+count doubles", async () => {
		const env = fakeEnv();
		for (const metric of ["chunk_oversize", "snapshot_fatal"]) {
			await worker.fetch(
				new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
					method: "POST",
					body: JSON.stringify({
						metric,
						sliceId: SLICE,
						gzippedBytes: 1234,
						count: 7,
					}),
				}),
				env,
			);
		}
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped(["chunk_oversize", VISITOR, SLICE, ""]),
				doubles: [1234, 7, -1],
			},
			{
				indexes: [SNIPPET],
				blobs: stamped(["snapshot_fatal", VISITOR, SLICE, ""]),
				doubles: [1234, 7, -1],
			},
		]);
	});

	test("upload rejections write an AE upload_rejected datapoint by reason", async () => {
		const env = fakeEnv();
		const big = new Uint8Array(MAX_CHUNK_BYTES + 1);
		big[0] = 0x1f;
		big[1] = 0x8b;
		await worker.fetch(put(PATH, big), env);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped(["upload_rejected", VISITOR, SLICE, "", "oversize"]),
			},
		]);
	});

	test("a worker-originated point carries this deploy's version from the metadata binding", async () => {
		const env = fakeEnv();
		env.CF_VERSION_METADATA = { id: "3c0e7f2a-deadbeef" };
		await worker.fetch(
			put(PATH, new TextEncoder().encode('{"events":[]}')),
			env,
		);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"upload_transcoded",
					VISITOR,
					SLICE,
					"3c0e7f2a-deadbeef",
					"json",
				]),
			},
		]);
	});

	test("a recorder_fault ping writes the reason and the recorder's verbatim error", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "recorder_fault",
					sliceId: SLICE,
					reason: "terminated",
					error: "boom",
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"recorder_fault",
					VISITOR,
					SLICE,
					"",
					"terminated",
					"boom",
					"",
				]),
				doubles: [-1],
			},
		]);
	});

	test("a ping's client-minted ts rides at its declared double position; a beacon can land long after the moment it reports", async () => {
		const env = fakeEnv();
		const ts = 1752900000123;
		await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "page_load",
					ts,
					sliceId: SLICE,
					url: "https://site/p2",
					referrer: "",
					spaRoute: false,
				}),
			}),
			env,
		);
		await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "recorder_fault",
					ts,
					sliceId: null,
					reason: "terminated",
					error: "boom",
				}),
			}),
			env,
		);
		expect(env.points[0].doubles).toEqual([ts]);
		expect(env.points[1].doubles).toEqual([ts]);
	});

	test("the request facts ride every datapoint at the uniform tail positions, attested by the transport", async () => {
		const env = fakeEnv();
		const headers = {
			"User-Agent": "Mozilla/5.0 (X11; TestKit)",
			Origin: "https://site.example",
		};
		const ping = new Request(
			`https://store.test/telemetry/${SNIPPET}/${VISITOR}`,
			{
				method: "POST",
				headers,
				body: JSON.stringify({
					metric: "page_load",
					sliceId: SLICE,
					url: "https://site/p",
					referrer: "",
					spaRoute: false,
				}),
			},
		);
		Object.defineProperty(ping, "cf", {
			value: { asOrganization: "Test Cloud Inc" },
		});
		await worker.fetch(ping, env);

		const transcoded = new Request(`https://store.test${PATH}`, {
			method: "PUT",
			headers,
			body: "plain json",
		});
		Object.defineProperty(transcoded, "cf", {
			value: { asOrganization: "Test Cloud Inc" },
		});
		await worker.fetch(transcoded, env);

		const facts = [
			"https://site.example",
			"Mozilla/5.0 (X11; TestKit)",
			"Test Cloud Inc",
		];
		expect(env.points.map((p) => p.blobs.slice(7))).toEqual([facts, facts]);
		expect(env.points.map((p) => p.blobs[0])).toEqual([
			"page_load",
			"upload_transcoded",
		]);
	});

	test("a capture_gated ping lands with its reason, url, and the gate's detail — a policy exclusion is never silent", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "capture_gated",
					reason: "bot",
					url: "https://site/p",
					detail: "detectNotificationPermissions,detectWindowSize",
					ts: 1752900000123,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"capture_gated",
					VISITOR,
					"",
					"",
					"bot",
					"https://site/p",
					"detectNotificationPermissions,detectWindowSize",
				]),
				doubles: [1752900000123],
			},
		]);
	});

	test("a capture_gated ping without detail lands with the slot empty — a gate ping carries only what its author's gate could say", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "capture_gated",
					reason: "bot",
					url: "https://site/p",
					ts: 1752900000123,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"capture_gated",
					VISITOR,
					"",
					"",
					"bot",
					"https://site/p",
				]),
				doubles: [1752900000123],
			},
		]);
	});

	test("a recorder_fault's url rides the row — a start-path fault has no slice or birth to join to", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "recorder_fault",
					sliceId: null,
					reason: "config_unresolved",
					error: "TypeError: failed to fetch",
					url: "https://site/landing",
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points[0].blobs.slice(0, 7)).toEqual([
			"recorder_fault",
			VISITOR,
			"",
			"",
			"config_unresolved",
			"TypeError: failed to fetch",
			"https://site/landing",
		]);
	});

	test("a cost_sample's visibleMs rides after ts — the context's own engaged-time testimony", async () => {
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "cost_sample",
					sliceId: SLICE,
					mainThreadMs: 4,
					uploadBytes: 10,
					deliveredBytes: 10,
					heapBytesMax: null,
					heapBytesLimit: null,
					backlogBytesMax: null,
					ts: 1752900000123,
					visibleMs: 45250,
				}),
			}),
			env,
		);
		expect(res.status).toBe(204);
		expect(env.points[0].doubles).toEqual([
			4, 10, -1, -1, -1, 10, 1752900000123, 45250,
		]);
	});

	test("a ping omitting its optional fields still lands — absent sentinels, never a bounce", async () => {
		// Some device is always running a bundle older than the worker — a deploy reaches the worker
		// before the last cached bundle turns over — and a custom telemetry sink carries what
		// its author gave it.
		const env = fakeEnv();
		const fault = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "recorder_fault",
					sliceId: null,
					reason: "drain_failed",
					error: "Error: sink down",
				}),
			}),
			env,
		);
		expect(fault.status).toBe(204);
		expect(env.points[0].blobs.slice(0, 7)).toEqual([
			"recorder_fault",
			VISITOR,
			"",
			"",
			"drain_failed",
			"Error: sink down",
			"",
		]);

		const cost = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "cost_sample",
					sliceId: SLICE,
					mainThreadMs: 4,
					uploadBytes: 10,
					deliveredBytes: 10,
					heapBytesMax: null,
					heapBytesLimit: null,
					backlogBytesMax: null,
					ts: 1752900000123,
				}),
			}),
			env,
		);
		expect(cost.status).toBe(204);
		expect(env.points[1].doubles).toEqual([
			4, 10, -1, -1, -1, 10, 1752900000123, -1,
		]);
	});

	test("an uncaught throw records a worker_error datapoint and answers 500, not a dead request", async () => {
		const env = fakeEnv();
		env.CHUNKS.put = async () => {
			throw new Error("R2 unavailable");
		};
		const res = await worker.fetch(put(PATH, GZIP_BODY), env);
		expect(res.status).toBe(500);
		expect(res.headers.get("Access-Control-Allow-Origin")).toBe("*");
		// Blobs 2–3 are visitor and slice on every metric — empty here, because the
		// worker's own failure knows neither — and blob 4 is the attestor's version,
		// empty without the metadata binding; a position cannot mean two things.
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped([
					"worker_error",
					"",
					"",
					"",
					"Error: R2 unavailable",
					PATH,
				]),
			},
		]);
	});

	test("the failure-report path is itself guarded — a broken AE binding still yields the 500", async () => {
		const env = fakeEnv();
		env.CHUNKS.put = async () => {
			throw new Error("R2 unavailable");
		};
		env.CAPTURE.writeDataPoint = () => {
			throw new Error("AE down");
		};
		const res = await worker.fetch(put(PATH, GZIP_BODY), env);
		expect(res.status).toBe(500);
	});

	test("an oversized telemetry body bounces under the same bound as a chunk, attested by reason", async () => {
		// The ping path is open-write too, and nothing about a ping's declared size is trusted: a
		// body past the store's one bound is refused rather than buffered, and the refusal is a row
		// like any other so the plane never goes quietly blind.
		const env = fakeEnv();
		const res = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: new Uint8Array(MAX_CHUNK_BYTES + 1).fill(0x20),
			}),
			env,
		);
		expect(res.status).toBe(413);
		expect(env.points).toEqual([
			{
				indexes: [SNIPPET],
				blobs: stamped(["telemetry_rejected", VISITOR, "", "", "oversize"]),
			},
		]);
	});

	test("a deployment without the Analytics Engine binding still ingests — the rates plane is optional", async () => {
		// AE is documented as an optional binding (store/README.md): a deploy without it loses the
		// rates plane and nothing else. Every path that would have written a datapoint must still
		// answer as it does with the binding present.
		const env = fakeEnv();
		env.CAPTURE = undefined;

		const clean = await worker.fetch(put(PATH, GZIP_BODY), env);
		expect(clean.status).toBe(204);
		expect(env.objects.size).toBe(1);

		const transcoded = await worker.fetch(
			put(PATH, new TextEncoder().encode('{"events":[]}')),
			env,
		);
		expect(transcoded.status).toBe(204);

		const ping = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: JSON.stringify({
					metric: "slice_started",
					sliceId: SLICE,
					url: "https://site/p",
					first_slice: true,
				}),
			}),
			env,
		);
		expect(ping.status).toBe(204);

		const bounced = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "POST",
				body: "{oops",
			}),
			env,
		);
		expect(bounced.status).toBe(400);

		const thrower = fakeEnv();
		thrower.CAPTURE = undefined;
		thrower.CHUNKS.put = async () => {
			throw new Error("R2 unavailable");
		};
		expect((await worker.fetch(put(PATH, GZIP_BODY), thrower)).status).toBe(
			500,
		);
	});

	test("telemetry bounces a bad path and a non-POST", async () => {
		const env = fakeEnv();
		const badPath = await worker.fetch(
			new Request("https://store.test/telemetry/BAD!/v", {
				method: "POST",
				body: "{}",
			}),
			env,
		);
		expect(badPath.status).toBe(404);
		const getPing = await worker.fetch(
			new Request(`https://store.test/telemetry/${SNIPPET}/${VISITOR}`, {
				method: "GET",
			}),
			env,
		);
		expect(getPing.status).toBe(405);
		expect(env.points).toEqual([]);
	});
});
