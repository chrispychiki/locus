/**
 * The terminate path, pinned end-to-end at start(): which oversize snapshot kills the recorder and
 * which one is merely dropped. The discriminator is the page-load slice, captured when the first
 * slice opens (uploader.js). Delivery failures sit on the other side of the same line, and are
 * pinned here too — none of them may terminate.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

import {
	fullSnapshot,
	installBrowserGlobals,
	meta,
	settle,
} from "./_support.js";

let capturedEmit = null;
let stopRecordingCalls = 0;

mock.module("@rrweb/record", () => ({
	record: (options) => {
		capturedEmit = options.emit;
		return () => {
			stopRecordingCalls += 1;
		};
	},
}));
mock.module("@fingerprintjs/botd", () => ({
	load: async () => ({ detect: () => ({ bot: false }) }),
}));

const { start } = await import("../src/index.js");
const { EventType } = await import("../src/rrweb_constants.js");

let teardown = () => {};

beforeEach(() => {
	capturedEmit = null;
	stopRecordingCalls = 0;
	teardown = installBrowserGlobals();
});
afterEach(() => teardown());

// An incompressible payload. A snapshot of repeated chars gzips to near-nothing, so forcing a chunk
// genuinely over a small cap takes bytes gzip cannot crush.
const noisy = (n) => {
	let s = "";
	for (let i = 0; i < n; i += 1)
		s += String.fromCharCode(33 + (((i * 2654435761) >>> 0) % 90));
	return s;
};

describe("catastrophic page-load snapshot terminates the recorder", () => {
	test("an over-cap page-load FullSnapshot kills the recorder and fires a catastrophic ping", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			maxGzippedChunkBytes: 1,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		await r.flush();

		expect(stopRecordingCalls).toBe(1);
		expect(telemetry.pings.some((e) => e.metric === "snapshot_fatal")).toBe(
			true,
		);
	});

	// Only the page-load slice's chunk is fatal: the checkout chunk behind it, over the cap in the
	// same doomed pass, takes the ordinary oversize path. One ping per doomed chunk would read as a
	// fleet-wide fault.
	test("a doomed recording fires exactly one catastrophic ping, not one per over-cap chunk", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			maxGzippedChunkBytes: 1,
		});
		capturedEmit(meta(1000)); // page-load slice opens
		capturedEmit(fullSnapshot(1001)); // page-load snapshot — over cap, and reached first
		capturedEmit(meta(2000)); // checkout: a second slice opens
		capturedEmit(fullSnapshot(2001)); // checkout snapshot — also over cap, never reached
		await settle();

		await r.flush();

		expect(stopRecordingCalls).toBe(1);
		expect(
			telemetry.pings.filter((e) => e.metric === "snapshot_fatal"),
		).toHaveLength(1);
		expect(
			telemetry.pings.filter((e) => e.metric === "chunk_oversize"),
		).toHaveLength(1);
	});

	// The fatal chunk alone is undeliverable: a later slice in the same batch carries its own
	// covering snapshot and ships before the recorder stops.
	test("the batch's deliverable chunks ship before a catastrophic termination", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const sent = [];
		const r = await start({
			sink: {
				send: async (_bytes, descriptor) => {
					sent.push(descriptor.sliceId);
				},
			},
			telemetry,
			botDetection: false,
			// 800 fits the tiny checkout chunk but not the noisy page-load chunk.
			maxGzippedChunkBytes: 800,
		});
		capturedEmit(meta(1000)); // page-load slice opens
		capturedEmit({
			type: EventType.FullSnapshot,
			data: { node: { big: noisy(40_000) } },
			timestamp: 1001,
		}); // page-load snapshot — over the cap, fatal
		capturedEmit(meta(2000)); // checkout: a second slice opens
		capturedEmit(fullSnapshot(2001)); // tiny checkout snapshot — under the cap
		await settle();

		await r.flush();

		expect(stopRecordingCalls).toBe(1);
		expect(
			telemetry.pings.filter((e) => e.metric === "snapshot_fatal"),
		).toHaveLength(1);
		const fatalSlice = telemetry.pings.find(
			(e) => e.metric === "snapshot_fatal",
		).sliceId;
		expect(sent).not.toContain(fatalSlice);
		expect(sent.length).toBeGreaterThan(0);
	});

	// A recorder that survives its own broken sink keeps capturing, so a later page context can drain
	// the durable buffer.
	test("a sink that fails every delivery never self-terminates the recorder", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			sink: {
				send: async () => {
					throw new Error("network down");
				},
			},
			telemetry,
			botDetection: false,
			errorThreshold: 1, // the tightest bound there is
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		// A whole outage's worth of drains, every one of them failing to reach the store.
		for (let i = 0; i < 12; i += 1) await r.flush();
		await settle();

		expect(stopRecordingCalls).toBe(0);
		expect(telemetry.pings.some((e) => e.metric === "snapshot_fatal")).toBe(
			false,
		);
	});

	// The page-load chunk stays under the cap here, so only the later checkout chunk is oversize.
	test("an over-cap checkout snapshot alone is dropped but does not terminate", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			// 800 fits the tiny page-load chunk (~350 gzipped bytes) but not the noisy checkout chunk below.
			maxGzippedChunkBytes: 800,
		});
		capturedEmit(meta(1000)); // page-load slice opens
		capturedEmit(fullSnapshot(1001)); // tiny page-load snapshot — under the cap
		await settle();
		await r.flush(); // ship the page-load slice without terminating
		expect(stopRecordingCalls).toBe(0);
		telemetry.pings.length = 0;

		// A second slice (checkout) whose snapshot is incompressibly large → its gzipped chunk exceeds the cap.
		capturedEmit(meta(2000));
		capturedEmit({
			type: EventType.FullSnapshot,
			data: { node: { big: noisy(40_000) } },
			timestamp: 2001,
		});
		await settle();
		await r.flush();

		expect(stopRecordingCalls).toBe(0);
		expect(telemetry.pings.some((e) => e.metric === "snapshot_fatal")).toBe(
			false,
		);
		expect(telemetry.pings.some((e) => e.metric === "chunk_oversize")).toBe(
			true,
		);
	});
});
