/**
 * SPA route-awareness (index.js): a PageLoad on each real route change through every channel the
 * recorder hooks — pushState, replaceState, popstate, hashchange — forcing no snapshot and opening
 * no slice, plus the non-navigations that must emit nothing, the host's navigation surviving a
 * throwing emit, and terminate restoring the history methods.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { EventType } from "../src/rrweb_constants.js";
import {
	capturingSink,
	installBrowserGlobals,
	makeWindow,
	pageLoads,
	sliceCount,
} from "./_support.js";

let capturedEmit = null;
let snapshotCalls = 0;
let recording = false;

// A faithful @rrweb/record stand-in. record() opens recording and lays down the record-start
// Meta+FullSnapshot, so the EventStream opens a slice. takeFullSnapshot(isCheckout) emits the same
// pair carrying the live window.location.href, and throws "after start recording" when recording is
// not active, as rrweb's does. snapshotCalls counts only explicit checkouts, so it stays 0 across
// these tests: a route change must force none.
function emitCheckout(isCheckout) {
	capturedEmit({
		type: EventType.Meta,
		data: { href: window.location.href, width: 1, height: 1 },
		timestamp: Date.now(),
	});
	capturedEmit({
		type: EventType.FullSnapshot,
		data: { node: {}, initialOffset: { left: 0, top: 0 } },
		timestamp: Date.now(),
		isCheckout,
	});
}

mock.module("@rrweb/record", () => {
	const record = (options) => {
		capturedEmit = options.emit;
		recording = true;
		emitCheckout(false);
		return () => {
			recording = false;
		};
	};
	record.takeFullSnapshot = (isCheckout) => {
		if (!recording)
			throw new Error("please take full snapshot after start recording");
		snapshotCalls += 1;
		emitCheckout(isCheckout);
	};
	return { record };
});
mock.module("@fingerprintjs/botd", () => ({
	load: async () => ({ detect: () => ({ bot: false }) }),
}));

const { start } = await import("../src/index.js");

let titleThrows = false;
let teardown = () => {};

beforeEach(() => {
	capturedEmit = null;
	snapshotCalls = 0;
	titleThrows = false;
	recording = false;
	teardown = installBrowserGlobals({
		window: makeWindow({ href: "https://client.example.com/home" }),
		document: {
			get title() {
				if (titleThrows) throw new Error("title boom");
				return "t";
			},
			referrer: "https://ref.example.com/",
		},
	});
});
afterEach(() => teardown());

const startWithSink = (sink) =>
	start({ visitorId: "v1", botDetection: false, sink });

describe("SPA route-awareness emits a PageLoad on each real route change", () => {
	test("a real route change emits a PageLoad for the new URL, forces no checkout, opens no slice", async () => {
		const sink = capturingSink();
		const r = await startWithSink(sink);
		await r.flush();
		const slicesBefore = sliceCount(sink);
		const loadsBefore = pageLoads(sink).length; // the record-start PageLoad

		window.history.pushState(null, "", "https://client.example.com/page2");
		await r.flush();

		expect(snapshotCalls).toBe(0); // no forced checkout
		expect(sliceCount(sink)).toBe(slicesBefore); // no new slice
		const loads = pageLoads(sink);
		expect(loads.length).toBe(loadsBefore + 1);
		expect(loads.at(-1).data.url).toBe("https://client.example.com/page2");
		expect(window.location.href).toBe("https://client.example.com/page2"); // host navigation still happened
	});

	test("replaceState, popstate, and hashchange each emit a PageLoad on a real change", async () => {
		const sink = capturingSink();
		const r = await startWithSink(sink);
		await r.flush();
		const loadsBefore = pageLoads(sink).length;

		window.history.replaceState(null, "", "https://client.example.com/a");
		window.location.href = "https://client.example.com/b";
		window.dispatch("popstate");
		window.location.href = "https://client.example.com/b#section";
		window.dispatch("hashchange");
		await r.flush();

		expect(snapshotCalls).toBe(0);
		const urls = pageLoads(sink)
			.slice(loadsBefore)
			.map((e) => e.data.url);
		expect(urls).toEqual([
			"https://client.example.com/a",
			"https://client.example.com/b",
			"https://client.example.com/b#section",
		]);
	});

	test("a pushState to the same route emits no PageLoad and opens no slice", async () => {
		const sink = capturingSink();
		const r = await startWithSink(sink);
		await r.flush();
		const loadsBefore = pageLoads(sink).length;
		const slicesBefore = sliceCount(sink);

		window.history.pushState(null, "", window.location.href);
		await r.flush();

		expect(pageLoads(sink).length).toBe(loadsBefore);
		expect(sliceCount(sink)).toBe(slicesBefore);
	});

	test("a query-only rewrite is not a navigation — no PageLoad", async () => {
		const sink = capturingSink();
		const r = await startWithSink(sink);
		await r.flush();
		const loadsBefore = pageLoads(sink).length;

		// same origin+path+hash, different query — analytics/state-sync churn
		window.history.replaceState(
			null,
			"",
			"https://client.example.com/home?utm_source=google&_gl=1",
		);
		await r.flush();

		expect(pageLoads(sink).length).toBe(loadsBefore);
	});

	test("a fragment-query-only rewrite is not a navigation either", async () => {
		const sink = capturingSink();
		const r = await startWithSink(sink);
		await r.flush();

		window.location.href = "https://client.example.com/app#/checkout?step=1";
		window.dispatch("hashchange");
		await r.flush();
		const loadsBefore = pageLoads(sink).length;

		// same origin+path+fragment route, different fragment query — the
		// hash-router's own state churn
		window.location.href = "https://client.example.com/app#/checkout?step=2";
		window.dispatch("hashchange");
		await r.flush();

		expect(pageLoads(sink).length).toBe(loadsBefore);
	});

	test("the route PageLoad carries the page's title and referrer", async () => {
		const sink = capturingSink();
		const r = await startWithSink(sink);
		window.history.pushState(null, "", "https://client.example.com/page2");
		await r.flush();

		const load = pageLoads(sink).at(-1);
		expect(load.data.title).toBe("t");
		expect(load.data.referrer).toBe("https://ref.example.com/");
	});

	test("a route change emits a page_load ping on the rates plane, spaRoute true, in the same slice", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const sink = capturingSink();
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			telemetry,
		});
		await r.flush();

		// The record-start page-load ping is a page load, not a route change.
		const startPings = telemetry.pings.filter((e) => e.metric === "page_load");
		expect(startPings).toHaveLength(1);
		expect(startPings[0].spaRoute).toBe(false);

		window.history.pushState(null, "", "https://client.example.com/page2");
		await r.flush();

		const born = telemetry.pings.find((e) => e.metric === "slice_started");
		const loads = telemetry.pings.filter((e) => e.metric === "page_load");
		expect(loads).toHaveLength(2);
		expect(loads[1]).toMatchObject({
			visitorId: born.visitorId,
			url: "https://client.example.com/page2",
			referrer: "https://ref.example.com/",
			spaRoute: true,
		});
		expect(loads[1].sliceId).toBe(born.sliceId); // a route change opens no new slice
		r.stop();
	});

	test("a throwing emit never propagates into host navigation, and never self-terminates", async () => {
		const sink = capturingSink();
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			errorThreshold: 1,
			sink,
		});
		titleThrows = true;

		for (let i = 0; i < 10; i += 1) {
			expect(() =>
				window.history.pushState(null, "", `https://client.example.com/n${i}`),
			).not.toThrow();
		}

		// Still recording: a later non-throwing route change is captured, so ten straight throws left
		// a live recorder rather than a corpse that merely did not throw.
		titleThrows = false;
		window.history.pushState(null, "", "https://client.example.com/live");
		await r.flush();
		expect(pageLoads(sink).map((e) => e.data.url)).toContain(
			"https://client.example.com/live",
		);
		r.stop();
	});

	test("even the fault path throwing never reaches the host's pushState call stack", async () => {
		// The deepest failure: the route change throws AND reporting it throws — a telemetry sink
		// that dies synchronously inside onError's own fault emit — so the inner catch rethrows and
		// only the outer guard stands between that and the host app's navigation.
		const telemetry = {
			broken: false,
			emit() {
				if (this.broken) throw new Error("telemetry sink down");
			},
		};
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink: capturingSink(),
			telemetry,
		});
		telemetry.broken = true;
		titleThrows = true;

		expect(() =>
			window.history.pushState(null, "", "https://client.example.com/deep"),
		).not.toThrow();
		expect(window.location.href).toBe("https://client.example.com/deep");
		r.stop();
	});

	test("route detection that stops working says so — it does not leave a recording that merely looks whole", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink: capturingSink(),
			telemetry,
		});
		titleThrows = true;

		window.history.pushState(null, "", "https://client.example.com/dead-route");

		const fault = telemetry.pings.find((e) => e.metric === "recorder_fault");
		expect(fault.reason).toBe("route_detection_failed");
		expect(fault.error).toContain("title boom");
		r.stop();
	});

	test("terminate restores the original history methods and removes listeners", async () => {
		const originalPush = window.history.pushState;
		const originalReplace = window.history.replaceState;
		const r = await startWithSink(capturingSink());
		expect(window.history.pushState).not.toBe(originalPush);
		expect(window.history.replaceState).not.toBe(originalReplace);

		r.stop();
		expect(window.history.pushState).toBe(originalPush);
		expect(window.history.replaceState).toBe(originalReplace);
		expect(window.listenerCount("popstate")).toBe(0);
		expect(window.listenerCount("hashchange")).toBe(0);

		window.history.pushState(null, "", "https://client.example.com/after");
		expect(snapshotCalls).toBe(0);
	});
});
