/**
 * The operator's user id attached to the recording: the Identify event start() emits when handed a
 * userId, the one identify() emits mid-life, and the gate both share.
 *
 * Observed through the real buffer and the real chunk path, like the orchestration suite.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

import {
	capturingSink,
	fullSnapshot,
	installBrowserGlobals,
	meta,
	settle,
} from "./_support.js";

let capturedEmit = null;

mock.module("@rrweb/record", () => {
	const record = (options) => {
		capturedEmit = options.emit;
		return () => {};
	};
	record.takeFullSnapshot = () => {};
	return { record };
});
mock.module("@fingerprintjs/botd", () => ({
	load: async () => ({
		detect: () => ({ bot: false }),
		getDetections: () => ({}),
	}),
}));

const { start } = await import("../src/index.js");
const { assertUserId, MAX_USER_ID_LENGTH } = await import("../src/identify.js");
const { EventType } = await import("../src/rrweb_constants.js");

let teardown = () => {};
beforeEach(() => {
	capturedEmit = null;
	teardown = installBrowserGlobals();
});
afterEach(() => teardown());

const identifies = (sink) =>
	sink.chunks.flatMap((c) =>
		c.payload.events
			.filter((e) => e.type === EventType.Identify)
			.map((event) => ({ sliceId: c.payload.sliceId, event })),
	);

const telemetryLog = () => ({
	pings: [],
	emit(e) {
		this.pings.push(e);
	},
});

const begin = (extra = {}) => {
	const sink = capturingSink();
	const telemetry = telemetryLog();
	return start({
		sink,
		telemetry,
		botDetection: false,
		intervalMs: 1_000_000,
		...extra,
	}).then((recorder) => ({ recorder, sink, telemetry }));
};

describe("Identify at start()", () => {
	test("a userId becomes one Identify carrying it, stamped like any recorded event", async () => {
		const { recorder, sink } = await begin({ userId: "user-42" });
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await recorder.flush();

		const found = identifies(sink);
		expect(found).toHaveLength(1);
		expect(found[0].event.data).toEqual({ userId: "user-42" });
		expect(typeof found[0].event.counter).toBe("string");
		const snapshotSlice = sink.chunks.find((c) =>
			c.payload.events.some((e) => e.type === EventType.FullSnapshot),
		).payload.sliceId;
		expect(found[0].sliceId).toBe(snapshotSlice);
		await recorder.stop();
	});

	test("no userId, or null, records no Identify", async () => {
		for (const extra of [{}, { userId: null }]) {
			const { recorder, sink } = await begin(extra);
			capturedEmit(meta(1000));
			capturedEmit(fullSnapshot(1001));
			await settle();
			await recorder.flush();
			expect(identifies(sink)).toEqual([]);
			await recorder.stop();
		}
	});

	test("a bad userId throws before anything runs — no capture, no telemetry, even where nothing would record", async () => {
		for (const hostname of ["client.example.com", "localhost"]) {
			globalThis.window.location.hostname = hostname;
			const telemetry = telemetryLog();
			await expect(
				start({
					sink: capturingSink(),
					telemetry,
					botDetection: false,
					userId: 42,
				}),
			).rejects.toThrow(/must be a string/);
			expect(capturedEmit).toBeNull();
			expect(telemetry.pings).toEqual([]);
		}
	});

	test("the id never rides the telemetry channel", async () => {
		const { recorder, telemetry } = await begin({ userId: "user-42" });
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await recorder.flush();
		expect(JSON.stringify(telemetry.pings)).not.toContain("user-42");
		await recorder.stop();
	});
});

describe("identify() mid-life", () => {
	test("records an Identify in whatever slice is current", async () => {
		const { recorder, sink } = await begin();
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		capturedEmit(meta(5000)); // a checkout opens a second slice
		capturedEmit(fullSnapshot(5001));
		recorder.identify("after-login");
		await settle();
		await recorder.flush();

		const found = identifies(sink);
		expect(found).toHaveLength(1);
		expect(found[0].event.data).toEqual({ userId: "after-login" });
		const secondSlice = sink.chunks.find((c) =>
			c.payload.events.some(
				(e) => e.type === EventType.Meta && e.timestamp === 5000,
			),
		).payload.sliceId;
		expect(found[0].sliceId).toBe(secondSlice);
		await recorder.stop();
	});

	test("each call records its own Identify — a switched account testifies both ids", async () => {
		const { recorder, sink } = await begin({ userId: "first" });
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		recorder.identify("second");
		await settle();
		await recorder.flush();
		expect(identifies(sink).map((f) => f.event.data.userId)).toEqual([
			"first",
			"second",
		]);
		await recorder.stop();
	});

	test("a bad id throws into the caller and records nothing", async () => {
		const { recorder, sink } = await begin();
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		expect(() => recorder.identify(null)).toThrow(/must be a string/);
		expect(() => recorder.identify("")).toThrow(/1–256 characters/);
		await settle();
		await recorder.flush();
		expect(identifies(sink)).toEqual([]);
		await recorder.stop();
	});
});

describe("the user-id gate", () => {
	test("admits any string within the cap, whatever its characters", () => {
		for (const id of [
			"a",
			"auth0|5f7c8ec7c33c6c004bbafe82",
			"名前@example.com",
			"x".repeat(MAX_USER_ID_LENGTH),
		]) {
			expect(() => assertUserId(id)).not.toThrow();
		}
	});

	test("admits the longest email address a mail path can carry", () => {
		const local = "l".repeat(64);
		const domain = `${"d".repeat(63)}.${"e".repeat(63)}.${"f".repeat(61)}`;
		const email = `${local}@${domain}`;
		expect(email.length).toBe(254);
		expect(() => assertUserId(email)).not.toThrow();
	});

	test.each([
		["a number", 42, /must be a string/],
		["undefined", undefined, /must be a string/],
		["an object", { id: "u" }, /must be a string/],
		["an empty string", "", /1–256 characters; got 0/],
		["one past the cap", "x".repeat(MAX_USER_ID_LENGTH + 1), /got 257/],
	])("refuses %s", (_name, id, message) => {
		expect(() => assertUserId(id)).toThrow(message);
	});

	test("the refusal never quotes the value — start()'s failure is reported on telemetry, which nothing masks", () => {
		const token = `eyJ${"a".repeat(400)}`;
		try {
			assertUserId(token);
			throw new Error("expected a refusal");
		} catch (error) {
			expect(error.message).not.toContain("eyJ");
		}
	});
});
