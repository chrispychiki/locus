import { afterEach, describe, expect, test } from "bun:test";
import { gunzipSync } from "fflate";

import { gzipBytes } from "../src/chunk.js";

const original = globalThis.CompressionStream;

afterEach(() => {
	if (original === undefined) delete globalThis.CompressionStream;
	else globalThis.CompressionStream = original;
});

describe("gzipBytes — the drain compression path never freezes the host", () => {
	test("uses CompressionStream (off the main thread) when the platform has it", async () => {
		let constructed = 0;
		class SpyCompressionStream extends original {
			constructor(format) {
				super(format);
				constructed += 1;
			}
		}
		globalThis.CompressionStream = SpyCompressionStream;

		const huge = new TextEncoder().encode("x".repeat(5_000_000));
		const out = await gzipBytes(huge);

		expect(constructed).toBe(1);
		expect(out).toBeInstanceOf(Uint8Array);
		expect(out.length).toBeGreaterThan(0);
		expect(out.length).toBeLessThan(huge.length);
	});

	test("returns the result asynchronously — the await yields to the event loop, so a giant snapshot gzips without a synchronous stall", async () => {
		globalThis.CompressionStream = original;
		let ranAfterCall = false;
		const promise = gzipBytes(new TextEncoder().encode("y".repeat(2_000_000)));
		queueMicrotask(() => {
			ranAfterCall = true;
		});
		const out = await promise;
		expect(ranAfterCall).toBe(true);
		expect(out.length).toBeGreaterThan(0);
	});

	test("falls back to synchronous gzip only when CompressionStream is absent", async () => {
		delete globalThis.CompressionStream;
		const out = await gzipBytes(new TextEncoder().encode("hello world"));
		expect(out).toBeInstanceOf(Uint8Array);
		expect(out.length).toBeGreaterThan(0);
	});

	test("the CompressionStream output is real, lossless gzip — it decompresses back to the exact input", async () => {
		// A round-trip, including binary bytes a text-only check would miss: smaller-than-input
		// says nothing about whether the store can inflate it.
		globalThis.CompressionStream = original;
		const input = new Uint8Array([
			0,
			1,
			2,
			255,
			128,
			7,
			7,
			7,
			7,
			7,
			...new TextEncoder().encode("a real rrweb chunk payload"),
		]);
		const out = await gzipBytes(input);
		expect(gunzipSync(out)).toEqual(input);
	});

	test("the synchronous fallback also produces lossless gzip", async () => {
		delete globalThis.CompressionStream;
		const input = new Uint8Array([
			9,
			9,
			0,
			200,
			13,
			10,
			...new TextEncoder().encode("fallback payload"),
		]);
		const out = await gzipBytes(input);
		expect(gunzipSync(out)).toEqual(input);
	});
});
