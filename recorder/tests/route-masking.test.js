/**
 * Per-route masking (index.js, masking.js): rrwebRules re-resolved on each SPA route change, the
 * restart a changed result forces, and the in-place PageLoad an unchanged one keeps — over both
 * path routes and hash routes.
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
let recordOptions = [];

// A faithful @rrweb/record stand-in: record() opens recording and lays down a record-start
// Meta+FullSnapshot (so the EventStream opens a slice), and its stop fn closes recording.
// takeFullSnapshot is rrweb's periodic-checkout entry, which the recorder never calls, so
// snapshotCalls stays 0: a restart is a fresh record(), not a checkout.
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
		recordOptions.push(options);
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

let teardown = () => {};

beforeEach(() => {
	capturedEmit = null;
	snapshotCalls = 0;
	recording = false;
	recordOptions = [];
	teardown = installBrowserGlobals({
		window: makeWindow({ href: "https://client.example.com/home" }),
		document: { referrer: "https://ref.example.com/" },
	});
});
afterEach(() => teardown());

describe("rrwebRules re-resolve per route and restart rrweb only when options differ", () => {
	const rules = [
		{
			pattern: "client.example.com",
			options: { maskInputOptions: { password: true } },
		},
		{
			pattern: "client.example.com/checkout",
			options: { maskInputOptions: { email: true } },
		},
	];

	test("a route whose masking differs restarts rrweb with the new options and opens a new slice", async () => {
		const sink = capturingSink();
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			rrwebRules: rules,
		});
		await r.flush();
		expect(recordOptions.length).toBe(1);
		expect(recordOptions[0].maskInputOptions).toMatchObject({ password: true });
		expect(recordOptions[0].maskInputOptions.email).toBe(true); // routing, not rule intent (masking.js)
		const slicesBefore = sliceCount(sink);
		const loadsBefore = pageLoads(sink).length;

		window.history.pushState(null, "", "https://client.example.com/checkout");
		await r.flush();

		expect(recordOptions.length).toBe(2); // restarted
		expect(recordOptions[1].maskInputOptions).toMatchObject({
			password: true,
			email: true,
		});
		expect(sliceCount(sink)).toBe(slicesBefore + 1); // fresh snapshot = new slice
		const loads = pageLoads(sink);
		expect(loads.length).toBe(loadsBefore + 1);
		expect(loads.at(-1).data.url).toBe("https://client.example.com/checkout");
		expect(snapshotCalls).toBe(0); // a restart is a fresh record(), not a checkout
		r.stop();
	});

	test("a masking-restart route's page_load ping carries spa=true — a client-side navigation, never a full load", async () => {
		const sink = capturingSink();
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			telemetry,
			rrwebRules: rules,
		});
		await r.flush();
		const initial = telemetry.pings.filter((e) => e.metric === "page_load");
		expect(initial.at(-1).spaRoute).toBe(false); // the genuine full load

		window.history.pushState(null, "", "https://client.example.com/checkout");
		await r.flush();

		const loads = telemetry.pings.filter((e) => e.metric === "page_load");
		expect(loads.length).toBe(initial.length + 1);
		expect(loads.at(-1).url).toBe("https://client.example.com/checkout");
		expect(loads.at(-1).spaRoute).toBe(true); // restart-driven, still a route change
		r.stop();
	});

	test("a route with the same masking does not restart — it emits the in-place PageLoad marker", async () => {
		const sink = capturingSink();
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			rrwebRules: rules,
		});
		await r.flush();
		const slicesBefore = sliceCount(sink);
		const loadsBefore = pageLoads(sink).length;

		window.history.pushState(null, "", "https://client.example.com/about");
		await r.flush();

		expect(recordOptions.length).toBe(1); // no restart — same masking
		expect(sliceCount(sink)).toBe(slicesBefore); // no new slice
		const loads = pageLoads(sink);
		expect(loads.length).toBe(loadsBefore + 1);
		expect(loads.at(-1).data.url).toBe("https://client.example.com/about");
		r.stop();
	});

	test("password masking survives the per-route restart", async () => {
		const sink = capturingSink();
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			rrwebRules: [
				{
					pattern: "client.example.com/checkout",
					options: { maskInputOptions: { password: false, email: true } },
				},
			],
		});
		await r.flush();

		window.history.pushState(null, "", "https://client.example.com/checkout");
		await r.flush();

		expect(recordOptions.at(-1).maskInputOptions.password).toBe(true);
		expect(recordOptions.at(-1).maskInputOptions.email).toBe(true);
		r.stop();
	});

	test("a route differing only by a masking function restarts — the change-key sees functions", async () => {
		const maskFn = (t) => t;
		const sink = capturingSink();
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			rrwebRules: [
				{
					pattern: "client.example.com",
					options: { maskTextSelector: "body" },
				},
				{
					pattern: "client.example.com/contact",
					options: { maskTextSelector: "body", maskTextFn: maskFn },
				},
			],
		});
		await r.flush();
		expect(recordOptions.length).toBe(1);
		expect(recordOptions[0].maskTextFn).toBeUndefined();

		window.history.pushState(null, "", "https://client.example.com/contact");
		await r.flush();

		expect(recordOptions.length).toBe(2); // restarted despite ONLY maskTextFn differing
		expect(recordOptions[1].maskTextFn).toBe(maskFn);
		r.stop();
	});

	test("a single universal rule never restarts rrweb on a route change — options stay fixed at start", async () => {
		const sink = capturingSink();
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			rrwebRules: [
				{ pattern: "*", options: { maskInputOptions: { email: true } } },
			],
		});
		await r.flush();

		window.history.pushState(null, "", "https://client.example.com/checkout");
		await r.flush();

		expect(recordOptions.length).toBe(1); // fixed options — no restart
		expect(sliceCount(sink)).toBe(1);
		expect(pageLoads(sink).at(-1).data.url).toBe(
			"https://client.example.com/checkout",
		); // in-place marker
		r.stop();
	});
});

describe("rrwebRules match hash-routed SPA routes via the fragment", () => {
	const hashRules = [
		{
			pattern: "client.example.com",
			options: { maskInputOptions: { password: true } },
		},
		{
			pattern: "client.example.com/#/checkout",
			options: { maskInputOptions: { email: true } },
		},
	];

	test("a hashchange into a differently-masked route restarts rrweb with the new options and opens a new slice", async () => {
		const sink = capturingSink();
		globalThis.window.location.href = "https://client.example.com/#/home";
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			rrwebRules: hashRules,
		});
		await r.flush();
		expect(recordOptions[0].maskInputOptions).toMatchObject({ password: true });
		const slicesBefore = sliceCount(sink);

		window.location.href = "https://client.example.com/#/checkout";
		window.dispatch("hashchange");
		await r.flush();

		expect(recordOptions.length).toBe(2);
		expect(recordOptions[1].maskInputOptions).toMatchObject({
			password: true,
			email: true,
		});
		expect(sliceCount(sink)).toBe(slicesBefore + 1);
		expect(pageLoads(sink).at(-1).data.url).toBe(
			"https://client.example.com/#/checkout",
		);
		expect(snapshotCalls).toBe(0);
		r.stop();
	});

	test("a hashchange within the same masking does not restart — it emits the in-place PageLoad marker", async () => {
		const sink = capturingSink();
		globalThis.window.location.href = "https://client.example.com/#/home";
		const r = await start({
			visitorId: "v1",
			botDetection: false,
			sink,
			rrwebRules: hashRules,
		});
		await r.flush();
		const slicesBefore = sliceCount(sink);
		const loadsBefore = pageLoads(sink).length;

		window.location.href = "https://client.example.com/#/about";
		window.dispatch("hashchange");
		await r.flush();

		expect(recordOptions.length).toBe(1);
		expect(sliceCount(sink)).toBe(slicesBefore);
		expect(pageLoads(sink).length).toBe(loadsBefore + 1);
		expect(pageLoads(sink).at(-1).data.url).toBe(
			"https://client.example.com/#/about",
		);
		r.stop();
	});
});
