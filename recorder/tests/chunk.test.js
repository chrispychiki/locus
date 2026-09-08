import { describe, expect, test } from "bun:test";
import { gunzipSync } from "fflate";

import {
	buildChunkPayloads,
	buildChunks,
	collectEnvelope,
	gzipBytes,
	serializePayload,
} from "../src/chunk.js";
import { EventType } from "../src/rrweb_constants.js";

const record = (sliceId, ts, counter, recorderVersion = "0.1.0") => ({
	sliceId,
	recorderVersion,
	event: {
		type: EventType.IncrementalSnapshot,
		data: {},
		timestamp: ts,
		counter,
	},
});

const CONTEXT = {
	visitorId: "v-1",
	recorderVersion: "0.1.0",
	envelope: { userAgent: "test-ua" },
	errors: ["boom"],
};

const unzip = (chunk) =>
	JSON.parse(new TextDecoder().decode(gunzipSync(chunk.bytes)));

describe("buildChunks", () => {
	test("one self-describing gzipped chunk per slice, order preserved", () => {
		const chunks = buildChunks(
			[
				record("00000000001000-aaaa", 1000, "1000000001"),
				record("00000000001000-aaaa", 1500, "1500000002"),
				record("00000000002000-bbbb", 2000, "2000000003"),
			],
			CONTEXT,
		);

		expect(chunks.length).toBe(2);
		const [first, second] = chunks;
		expect(first.descriptor).toEqual({
			visitorId: "v-1",
			sliceId: "00000000001000-aaaa",
			chunkKey: "1000000001",
			count: 2,
			hasFullSnapshot: false,
		});
		expect(second.descriptor.sliceId).toBe("00000000002000-bbbb");

		const payload = unzip(first);
		expect(payload.visitorId).toBe("v-1");
		expect(payload.envelope.userAgent).toBe("test-ua");
		expect(payload.events.map((e) => e.counter)).toEqual([
			"1000000001",
			"1500000002",
		]);

		expect(payload.errors).toEqual(["boom"]);
		expect(unzip(second).errors).toEqual([]);
	});

	test("hasFullSnapshot flags the slice that carries a FullSnapshot", () => {
		const chunks = buildChunks(
			[
				{
					sliceId: "s-1",
					recorderVersion: "0.1.0",
					event: {
						type: EventType.FullSnapshot,
						data: {},
						timestamp: 1,
						counter: "1000001",
					},
				},
				record("s-2", 2, "2000002"),
			],
			CONTEXT,
		);
		expect(chunks.map((c) => c.descriptor.hasFullSnapshot)).toEqual([
			true,
			false,
		]);
	});

	test("a chunk carries the version of the recorder that captured it, not the one shipping it", () => {
		const chunks = buildChunkPayloads(
			[
				record("s-old", 1, "1000001", "locus-recorder/0.9.0"),
				record("s-new", 2, "2000002", "locus-recorder/1.0.0"),
			],
			{ ...CONTEXT, recorderVersion: "locus-recorder/1.0.0" },
		);

		expect(chunks.map((c) => c.payload.recorderVersion)).toEqual([
			"locus-recorder/0.9.0",
			"locus-recorder/1.0.0",
		]);
	});

	test("an unstamped record falls back to the couriering context's version", () => {
		const [{ payload }] = buildChunkPayloads(
			[
				{
					sliceId: "s",
					event: {
						type: EventType.IncrementalSnapshot,
						data: {},
						timestamp: 1,
						counter: "1000001",
					},
				},
			],
			{ ...CONTEXT, recorderVersion: "locus-recorder/1.0.0" },
		);
		expect(payload.recorderVersion).toBe("locus-recorder/1.0.0");
	});

	// The beacon path and the drain path must be interchangeable on the wire: a store cannot tell
	// which one produced a chunk.
	test("buildChunks and buildChunkPayloads produce identical descriptors", () => {
		const records = [
			record("00000000001000-aaaa", 1000, "1000000001"),
			record("00000000001000-aaaa", 1500, "1500000002"),
			record("00000000002000-bbbb", 2000, "2000000003"),
		];
		const sync = buildChunks(records, CONTEXT);
		const payloads = buildChunkPayloads(records, CONTEXT);
		expect(sync.map((c) => c.descriptor)).toEqual(
			payloads.map((p) => p.descriptor),
		);
	});

	test("the async gzip path decodes to the same payload as the sync path", async () => {
		const records = [record("00000000001000-aaaa", 1000, "1000000001")];
		const [{ payload }] = buildChunkPayloads(records, CONTEXT);
		const syncChunk = buildChunks(records, CONTEXT)[0];
		const asyncBytes = await gzipBytes(serializePayload(payload));
		expect(
			JSON.parse(new TextDecoder().decode(gunzipSync(asyncBytes))),
		).toEqual(
			JSON.parse(new TextDecoder().decode(gunzipSync(syncChunk.bytes))),
		);
	});

	// chunkKey stays, because emission order is not something the timestamps can answer: the
	// first-emitted event is not the earliest one when a re-homed move keeps its flush timestamp.
	test("nothing that the events or the slice id already say is copied beside them", () => {
		const [{ descriptor, payload }] = buildChunkPayloads(
			[
				record("00000000009999-zzzz", 5000, "5000000001"),
				record("00000000009999-zzzz", 1000, "1000000002"),
			],
			CONTEXT,
		);
		expect(payload).not.toHaveProperty("range");
		expect(descriptor).not.toHaveProperty("range");
		expect(descriptor).not.toHaveProperty("sliceStartMs");
		expect(descriptor.chunkKey).toBe("5000000001");
	});

	test("the envelope is carried verbatim into every chunk payload", () => {
		const envelope = {
			userAgent: "Mozilla/5.0 test",
			language: "en-US",
			timeZone: "America/New_York",
			screen: { width: 1920, height: 1080 },
		};
		const chunks = buildChunks(
			[record("s-1", 1, "1000001"), record("s-2", 2, "2000002")],
			{ ...CONTEXT, envelope },
		);
		for (const chunk of chunks) {
			expect(unzip(chunk).envelope).toEqual(envelope);
		}
	});

	test("a chunk states the capture's visitor and envelope, not the courier's", () => {
		const capture = {
			visitorId: "v-writer",
			envelope: { userAgent: "writer-ua", language: "de" },
		};
		const [{ descriptor, payload }] = buildChunkPayloads(
			[record("s-stamped", 1, "1000001")],
			CONTEXT,
			{
				"s-stamped": capture,
			},
		);

		expect(payload.visitorId).toBe("v-writer");
		expect(payload.envelope).toEqual(capture.envelope);
		expect(descriptor.visitorId).toBe("v-writer");
	});

	test("a slice without a capture context falls back to the couriering context", () => {
		const chunks = buildChunkPayloads(
			[record("s-stamped", 1, "1000001"), record("s-bare", 2, "2000002")],
			CONTEXT,
			{
				"s-stamped": {
					visitorId: "v-writer",
					envelope: { userAgent: "writer-ua" },
				},
			},
		);

		const bySlice = Object.fromEntries(
			chunks.map((c) => [c.payload.sliceId, c.payload]),
		);
		expect(bySlice["s-stamped"].visitorId).toBe("v-writer");
		expect(bySlice["s-bare"].visitorId).toBe("v-1");
		expect(bySlice["s-bare"].envelope).toEqual(CONTEXT.envelope);
	});

	test("the error log stays the courier's whatever identity the first chunk carries", () => {
		// Evictions and poison drops happen at shipping, so the courier is their honest witness;
		// the strings ride the first chunk regardless of whose capture context it states.
		const [first] = buildChunkPayloads(
			[record("s-stamped", 1, "1000001")],
			{ ...CONTEXT, errors: ["courier: evicted 9 bytes"] },
			{ "s-stamped": { visitorId: "v-writer", envelope: {} } },
		);
		expect(first.payload.visitorId).toBe("v-writer");
		expect(first.payload.errors).toEqual(["courier: evicted 9 bytes"]);
	});

	test("collectEnvelope reads the device facts, each behind its own guard", () => {
		const envelope = collectEnvelope({
			navigator: { userAgent: "ua-x", language: "fr-CA", maxTouchPoints: 5 },
			screen: { width: 800, height: 600 },
		});
		expect(envelope.userAgent).toBe("ua-x");
		expect(envelope.language).toBe("fr-CA");
		expect(envelope.screen).toEqual({ width: 800, height: 600 });
		expect(envelope.maxTouchPoints).toBe(5);
		// The time zone comes from ambient Intl, the one witness with no global to hand in.
		expect(typeof envelope.timeZone).toBe("string");
	});

	test("a hostile global yields a partial envelope, never a throw", () => {
		const envelope = collectEnvelope({
			get navigator() {
				throw new Error("locked down");
			},
			screen: { width: 1, height: 1 },
		});
		expect(envelope.userAgent).toBeUndefined();
		expect(envelope.language).toBeUndefined();
		expect(envelope.maxTouchPoints).toBeUndefined();
		expect(envelope.screen).toEqual({ width: 1, height: 1 });
	});

	test("errors attach to the first chunk only and a single-slice build still carries them", () => {
		const single = buildChunks([record("only", 1, "1000001")], {
			...CONTEXT,
			errors: ["lonely-error"],
		});
		expect(unzip(single[0]).errors).toEqual(["lonely-error"]);

		const multi = buildChunks(
			[
				record("a", 1, "1000001"),
				record("b", 2, "2000002"),
				record("c", 3, "3000003"),
			],
			{
				...CONTEXT,
				errors: ["first-only"],
			},
		);
		expect(multi.map((c) => unzip(c).errors)).toEqual([["first-only"], [], []]);
	});
});
