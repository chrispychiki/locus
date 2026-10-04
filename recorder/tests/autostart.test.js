/**
 * The facade's own failure: a page context that loads the bundle, passes every gate, and then dies
 * inside start(), which on both planes is identical to a visitor who never arrived.
 *
 * Driven through the real telemetry sink, so what is asserted is the ping that actually leaves the
 * page, not a double's record of one.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { gunzipSync } from "fflate";

import { installBrowserGlobals, makeWindow } from "./_support.js";

let recordThrows = false;

mock.module("@rrweb/record", () => ({
	record: () => {
		if (recordThrows) throw new Error("rrweb could not record this document");
		return () => {};
	},
}));
mock.module("@fingerprintjs/botd", () => ({
	load: async () => ({ detect: () => ({ bot: false }) }),
}));

const { autostart } = await import("../src/global.js");
const { EventType } = await import("../src/rrweb_constants.js");

const TAG = { src: "https://store.test/locus.min.js?id=abc123" };

let shots = [];
let cookieThrows = false;
let teardown = () => {};
const realFetch = globalThis.fetch;

// `target` is the shot's destination; the body's own `url` field (the page the ping is
// about) spreads in beside it. The telemetry sink picks its transport itself (sink.js), so both
// are captured — which channel a ping rode is not this suite's business.
const pings = () =>
	shots.map(({ url, body }) => ({ target: url, ...JSON.parse(body) }));

// The user ids the uploaded chunks recorded, in order.
const identified = () =>
	shots
		.filter(({ url }) => url.includes("/chunks/"))
		.flatMap(
			({ body }) =>
				JSON.parse(new TextDecoder().decode(gunzipSync(body))).events,
		)
		.filter((e) => e.type === EventType.Identify)
		.map((e) => e.data.userId);

beforeEach(() => {
	shots = [];
	recordThrows = false;
	cookieThrows = false;
	teardown = installBrowserGlobals({
		window: makeWindow(),
		document: {
			get cookie() {
				if (cookieThrows)
					throw new Error("SecurityError: cookies are unavailable");
				return "";
			},
			set cookie(_value) {
				/* accepted */
			},
		},
	});
	globalThis.navigator.sendBeacon = (url, body) => {
		shots.push({ url, body });
		return true;
	};
	globalThis.fetch = (url, opts) => {
		shots.push({ url, body: opts.body });
		return Promise.resolve({ ok: true });
	};
});
afterEach(() => {
	teardown();
	globalThis.fetch = realFetch;
});

describe("autostart", () => {
	test("a start() that dies faults on the telemetry plane, not only into the console", async () => {
		recordThrows = true;

		await autostart(TAG);

		const [fault] = pings().filter((p) => p.metric === "recorder_fault");
		expect(fault.reason).toBe("start_failed");
		expect(fault.error).toContain("rrweb could not record this document");
		expect(fault.target).toMatch(/^https:\/\/store\.test\/telemetry\/abc123\//);
		expect(fault.url).toBe(window.location.href);
	});

	test("the fault names the page by address — the query string never reaches the wire", async () => {
		recordThrows = true;
		globalThis.window.location.href =
			"https://client.example.com/reset?token=secret";

		await autostart(TAG);

		const [fault] = pings().filter((p) => p.metric === "recorder_fault");
		expect(fault.url).toBe("https://client.example.com/reset");
		expect(shots.map((s) => s.body).join("")).not.toContain("secret");
	});

	test("a device that cannot even be identified still reports that it failed", async () => {
		// Reading document.cookie is itself a way to die (visitor.js), so the fault has to survive
		// the loss of the very thing it would be keyed by.
		cookieThrows = true;

		await autostart(TAG);

		const [fault] = pings().filter((p) => p.metric === "recorder_fault");
		expect(fault.reason).toBe("start_failed");
		expect(fault.error).toContain("SecurityError");
		expect(fault.target).toBe(
			"https://store.test/telemetry/abc123/unidentified",
		);
	});

	test("a recorder that starts says nothing — a fault is a failure, never a lifecycle event", async () => {
		await autostart(TAG);

		expect(pings().filter((p) => p.metric === "recorder_fault")).toEqual([]);
		expect(typeof window.LocusRecorder.flush).toBe("function");
		expect(typeof window.LocusRecorder.identify).toBe("function");
	});
});

describe("identify on the marker", () => {
	test("an id given before start() finishes is held, the latest winning, and recorded once it does", async () => {
		const running = autostart(TAG);
		const early = window.LocusRecorder.identify;
		window.LocusRecorder.identify("first-guess");
		window.LocusRecorder.identify("user-42");
		expect(window.LocusRecorder.flush).toBeUndefined();
		await running;

		expect(window.LocusRecorder.identify).toBe(early);
		await window.LocusRecorder.flush();
		expect(identified()).toEqual(["user-42"]);
	});

	test("once recording, each call records its own Identify", async () => {
		await autostart(TAG);
		window.LocusRecorder.identify("user-42");
		window.LocusRecorder.identify("user-43");
		await window.LocusRecorder.flush();
		expect(identified()).toEqual(["user-42", "user-43"]);
	});

	test("a page context that never records keeps identify and drops every id, before and after", async () => {
		for (const setup of [
			() => {
				globalThis.window.location.hostname = "localhost";
			},
			() => {
				recordThrows = true;
			},
		]) {
			delete globalThis.window.LocusRecorder;
			setup();
			const running = autostart(TAG);
			window.LocusRecorder.identify("held-then-dropped");
			await running;
			expect(Object.keys(window.LocusRecorder)).toEqual(["identify"]);
			expect(() => window.LocusRecorder.identify("user-42")).not.toThrow();
		}
		expect(identified()).toEqual([]);
	});

	test("a bad id throws into the caller whatever the recorder's state", async () => {
		globalThis.window.location.hostname = "localhost";
		const running = autostart(TAG);
		expect(() => window.LocusRecorder.identify(42)).toThrow(/must be a string/);
		await running;
		expect(() => window.LocusRecorder.identify("")).toThrow(/1–256/);
	});
});
