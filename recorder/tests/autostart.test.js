/**
 * The facade's own failure: a page context that loads the bundle, passes every gate, and then dies
 * inside start(), which on both planes is identical to a visitor who never arrived.
 *
 * Driven through the real telemetry sink, so what is asserted is the ping that actually leaves the
 * page, not a double's record of one.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

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
	});
});
