/**
 * The recorder carries no surrogate-sanitization walk: it does not need one to emit valid wire
 * bytes. JSON.stringify (ES2019+) escapes a lone surrogate to its \uXXXX ASCII form, so by the
 * time serializePayload's TextEncoder runs there is no lone surrogate left to mangle — the gzipped
 * chunk is valid UTF-8 regardless. The lone surrogate survives, escaped, into the parsed payload.
 */
import { describe, expect, test } from "bun:test";

import { gunzipSync } from "fflate";

import { buildChunks, gzipBytes, serializePayload } from "../src/chunk.js";
import { EventType } from "../src/rrweb_constants.js";

describe("recorder emits valid escaped UTF-8 for lone surrogates without a sanitization walk", () => {
	test("a lone high surrogate serializes to valid UTF-8 bytes, escaped not replaced", () => {
		const payload = {
			events: [
				{
					type: EventType.IncrementalSnapshot,
					data: { text: "bad \uD83D end", nested: { deep: "\uD800" } },
				},
			],
		};
		const bytes = serializePayload(payload);

		const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
		expect(text).toContain("bad \\ud83d end");
		expect(text).toContain("\\ud800");
		expect(text).not.toContain("�");

		const parsed = JSON.parse(text);
		expect(parsed.events[0].data.text).toBe("bad \uD83D end");
		expect(parsed.events[0].data.nested.deep).toBe("\uD800");
	});

	test("a lone low surrogate serializes to valid UTF-8 bytes, escaped not replaced", () => {
		const bytes = serializePayload({
			events: [{ data: { text: "x \uDC00 y" } }],
		});
		const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
		expect(text).toContain("\\udc00");
		expect(text).not.toContain("�");
		expect(JSON.parse(text).events[0].data.text).toBe("x \uDC00 y");
	});

	test("a well-formed surrogate pair round-trips intact", () => {
		const bytes = serializePayload({
			events: [{ data: { text: "ok 😀 emoji" } }],
		});
		const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
		expect(JSON.parse(text).events[0].data.text).toBe("ok 😀 emoji");
	});

	// The bytes that actually ship: a full gzipped chunk, not serializePayload in isolation.
	test("the gzipped chunk that ships is valid UTF-8 — surrogate survives compress/decompress", () => {
		const record = {
			sliceId: "00000000001000-aaaa",
			event: {
				type: EventType.IncrementalSnapshot,
				data: { text: "lone \uD83D here" },
				timestamp: 1000,
				counter: "1000000001",
			},
		};
		const [chunk] = buildChunks([record], {
			visitorId: "v-1",
			recorderVersion: "0.1.0",
			envelope: {},
		});
		// gunzip yields raw bytes; decoding them strictly must not throw and must keep the escape.
		const text = new TextDecoder("utf-8", { fatal: true }).decode(
			gunzipSync(chunk.bytes),
		);
		expect(text).toContain("\\ud83d");
		expect(text).not.toContain("�");
		expect(JSON.parse(text).events[0].data.text).toBe("lone \uD83D here");
	});

	// The drain path — the common case, so it cannot be weaker than the beacon's.
	test("the async gzip path preserves escaped surrogates on the wire", async () => {
		const bytes = serializePayload({
			events: [{ data: { a: "\uD800", b: "\uDFFF" } }],
		});
		const gz = await gzipBytes(bytes);
		const text = new TextDecoder("utf-8", { fatal: true }).decode(
			gunzipSync(gz),
		);
		expect(text).toContain("\\ud800");
		expect(text).toContain("\\udfff");
		expect(JSON.parse(text).events[0].data.a).toBe("\uD800");
		expect(JSON.parse(text).events[0].data.b).toBe("\uDFFF");
	});

	// The guarantee is per-codepoint, not first-one-only; a real masked-text node can hold several.
	test("several lone surrogates in one string all escape, none replaced", () => {
		const bytes = serializePayload({
			events: [{ data: { text: "\uD800a\uDC00b\uDBFFc\uDFFF" } }],
		});
		const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
		for (const esc of ["\\ud800", "\\udc00", "\\udbff", "\\udfff"]) {
			expect(text).toContain(esc);
		}
		expect(text).not.toContain("�");
		expect(JSON.parse(text).events[0].data.text).toBe(
			"\uD800a\uDC00b\uDBFFc\uDFFF",
		);
	});
});
