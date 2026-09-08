/**
 * The cost_sample telemetry emit, pinned at start(): what a sample carries, and when one fires.
 *
 * The figures have to tell a broken recorder from an idle one — a device holding a recording it
 * cannot ship must not look like a visitor who did nothing — so the assertions here are about the
 * distinctions that carry that: the heap high-water mark rather than a point reading, an unmeasured
 * null rather than a claimed-empty 0, a backlog sampled wherever it moves rather than inside the
 * drain loop, and bytes pushed counted separately from bytes acknowledged.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

import {
	fullSnapshot,
	installBrowserGlobals,
	meta,
	settle,
} from "./_support.js";

let capturedEmit = null;
let visibilityHandler = null;

mock.module("@rrweb/record", () => ({
	record: (options) => {
		capturedEmit = options.emit;
		return () => {};
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
	visibilityHandler = null;
	teardown = installBrowserGlobals({
		document: {
			addEventListener: (type, fn) => {
				if (type === "visibilitychange") visibilityHandler = fn;
			},
		},
		// performance.memory is Chromium-only; most of this suite is about the case where it exists.
		performance: {
			now: () => 0,
			memory: { usedJSHeapSize: 0, jsHeapSizeLimit: 2_000_000 },
		},
	});
});
afterEach(() => teardown());

describe("cost_sample emit", () => {
	test("emits the page-load slice id, the three cost figures, and no single-point heapBytes", async () => {
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
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		document.visibilityState = "hidden";
		visibilityHandler();

		const born = telemetry.pings.find((e) => e.metric === "slice_started");
		const cost = telemetry.pings.find((e) => e.metric === "cost_sample");
		expect(cost).toBeDefined();
		expect(cost.sliceId).toBe(born.sliceId); // the page-load slice, 1:1 with this start() lifetime
		expect(cost.sliceId).not.toBeNull();
		expect(typeof cost.mainThreadMs).toBe("number");
		expect(typeof cost.uploadBytes).toBe("number");
		expect(typeof cost.heapBytesMax).toBe("number");
		expect(cost.heapBytesLimit).toBe(2_000_000); // the OOM ceiling — heap means nothing without it
		expect(typeof cost.backlogBytesMax).toBe("number");
		expect(cost).not.toHaveProperty("heapBytes");
		r.stop();
	});

	test("heapBytesMax carries the running per-context peak, not the latest sample", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		performance.memory.usedJSHeapSize = 100;
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		document.visibilityState = "hidden";
		visibilityHandler(); // sample at heap = 100
		document.visibilityState = "visible";
		visibilityHandler(); // visible → no cost emit
		performance.memory.usedJSHeapSize = 50; // heap fell back
		document.visibilityState = "hidden";
		visibilityHandler(); // sample at heap = 50 — the max must still report 100

		const costs = telemetry.pings.filter((e) => e.metric === "cost_sample");
		expect(costs).toHaveLength(2);
		expect(costs[0].heapBytesMax).toBe(100);
		expect(costs[1].heapBytesMax).toBe(100); // the peak survives a lower later sample
		r.stop();
	});

	test("without performance.memory the heap figures are null — unmeasured, never a claimed-empty 0", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		delete globalThis.performance.memory; // Safari/Firefox: the API does not exist
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		document.visibilityState = "hidden";
		visibilityHandler();

		const cost = telemetry.pings.find((e) => e.metric === "cost_sample");
		expect(cost.heapBytesMax).toBeNull();
		expect(cost.heapBytesLimit).toBeNull();
		expect(typeof cost.mainThreadMs).toBe("number"); // the measured figures still ride
		r.stop();
	});

	test("backlogBytesMax captures the on-device buffer backlog the uploader is holding", async () => {
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
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush(); // a drain samples the standing backlog before it claims it

		document.visibilityState = "hidden";
		visibilityHandler();

		const cost = telemetry.pings.find((e) => e.metric === "cost_sample");
		expect(cost.backlogBytesMax).toBeGreaterThan(0);
		r.stop();
	});

	test("a page context that never drained still reports what it is holding", async () => {
		// The context buffers and never drains — a page that died before its first drain tick, or one
		// whose drain loop is itself what broke.
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
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		document.visibilityState = "hidden";
		visibilityHandler();

		const cost = telemetry.pings.find((e) => e.metric === "cost_sample");
		expect(cost.backlogBytesMax).toBeGreaterThan(0);
		expect(cost.deliveredBytes).toBe(0);
		r.stop();
	});

	test("storage that refuses every write falls to memory — the sample reads the memory backlog and the downgrade is the fault", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		globalThis.indexedDB = {
			open: () => {
				const request = {};
				queueMicrotask(() => {
					request.result = {
						onclose: null,
						close: () => {},
						transaction: () => {
							throw new Error("storage is gone");
						},
					};
					request.onsuccess?.();
				});
				return request;
			},
		};
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		document.visibilityState = "hidden";
		visibilityHandler();

		const cost = telemetry.pings.find((e) => e.metric === "cost_sample");
		expect(cost.backlogBytesMax).toBeGreaterThan(0); // the memory buffer is holding the recording
		const fault = telemetry.pings.find((e) => e.metric === "recorder_fault");
		expect(fault.reason).toBe("buffer_not_durable");
		r.stop();
	});

	test("a buffer that never answered reports no reading, not an empty one", async () => {
		// Storage whose transactions hang: at sample time nothing has committed and nothing has been
		// diagnosed, so there is no backlog figure to give, and none is given.
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		globalThis.indexedDB = {
			open: () => {
				const request = {};
				queueMicrotask(() => {
					request.result = {
						onclose: null,
						close: () => {},
						transaction: () => ({
							abort() {},
							objectStore: () => ({ get: () => ({}), put: () => ({}) }),
						}),
					};
					request.onsuccess?.();
				});
				return request;
			},
		};
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		document.visibilityState = "hidden";
		visibilityHandler();

		const cost = telemetry.pings.find((e) => e.metric === "cost_sample");
		expect(cost.backlogBytesMax).toBeNull();
		r.stop();
	});

	test("uploadBytes is what the device spent; deliveredBytes is what the sink took", async () => {
		// A sink that accepts nothing still costs the visitor their bandwidth.
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const sink = {
			send: async () => {
				throw new Error("sink down");
			},
		};
		const r = await start({ sink, telemetry, botDetection: false });
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush();

		document.visibilityState = "hidden";
		visibilityHandler();

		const cost = telemetry.pings.find((e) => e.metric === "cost_sample");
		expect(cost.uploadBytes).toBeGreaterThan(0);
		expect(cost.deliveredBytes).toBe(0);
		r.stop();
	});

	test("visibleMs accrues only while the page is foreground — the running engaged-time total", async () => {
		let now = 0;
		globalThis.performance.now = () => now;
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
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();

		now = 5000;
		document.visibilityState = "hidden";
		visibilityHandler(); // stint 1 ends: 5000ms foreground
		now = 8000; // 3000ms hidden — must not count
		document.visibilityState = "visible";
		visibilityHandler();
		now = 9000;
		document.visibilityState = "hidden";
		visibilityHandler(); // stint 2 ends: +1000ms

		const costs = telemetry.pings.filter((e) => e.metric === "cost_sample");
		expect(costs.map((c) => c.visibleMs)).toEqual([5000, 6000]);
		r.stop();
	});

	test("the periodic tick emits only when work happened since the last sample; hidden emits unconditionally", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		let tick = null;
		const realSetInterval = globalThis.setInterval;
		globalThis.setInterval = (fn) => {
			tick = fn;
			return 0;
		};
		try {
			const r = await start({
				sink: { send: async () => {} },
				telemetry,
				botDetection: false,
			});
			capturedEmit(meta(1000));
			capturedEmit(fullSnapshot(1001));
			await settle();
			const costs = () =>
				telemetry.pings.filter((e) => e.metric === "cost_sample").length;

			tick();
			expect(costs()).toBe(1); // work accrued since start → the tick samples
			tick();
			tick();
			expect(costs()).toBe(1); // no work since → idle ticks are suppressed
			capturedEmit({
				type: EventType.IncrementalSnapshot,
				data: {},
				timestamp: 1002,
			});
			tick();
			expect(costs()).toBe(2); // new work → the next tick samples again
			tick();
			expect(costs()).toBe(2);

			document.visibilityState = "hidden";
			visibilityHandler();
			expect(costs()).toBe(3); // the terminal at hidden never gates
			r.stop();
		} finally {
			globalThis.setInterval = realSetInterval;
		}
	});
});
