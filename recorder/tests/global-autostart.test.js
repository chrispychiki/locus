/**
 * The refusal branches of the script-tag entry — the identification contract. A capture proving the
 * bundle was not loaded as its own classic <script src> tag fails loud and starts nothing; a tag
 * without a valid id sets the loaded marker and refuses to start. The happy path through start() is
 * the recorder↔store integration suite's and the drive check's.
 */
import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import { autostart } from "../src/global.js";

let errors;
let errorSpy;
let fetchTargets;
let realFetch;
let realNavigator;

beforeEach(() => {
	errors = [];
	errorSpy = spyOn(console, "error").mockImplementation((msg) =>
		errors.push(msg),
	);
	globalThis.window = {};
	// The failure path emits a real fault ping through storeSink's carrier: without a stub that is
	// live network egress from a unit test, and with one it is an assertable seam.
	fetchTargets = [];
	realFetch = globalThis.fetch;
	globalThis.fetch = async (url) => {
		fetchTargets.push(String(url));
		return new Response(null, { status: 204 });
	};
	realNavigator = globalThis.navigator;
	globalThis.navigator = { userAgent: "ua", language: "en" };
});

afterEach(() => {
	errorSpy.mockRestore();
	globalThis.fetch = realFetch;
	globalThis.navigator = realNavigator;
	delete globalThis.window;
});

describe("autostart tag binding", () => {
	test("a null capture fails loud and starts nothing — the bundle was not loaded as its own tag", async () => {
		await autostart(null);
		expect(errors).toHaveLength(1);
		expect(errors[0]).toContain(
			"not loaded as its own classic <script src> tag",
		);
		expect(globalThis.window.LocusRecorder).toBeUndefined();
	});

	test("an inline script's empty src fails the same way", async () => {
		await autostart({ src: "" });
		expect(errors).toHaveLength(1);
		expect(errors[0]).toContain(
			"not loaded as its own classic <script src> tag",
		);
		expect(globalThis.window.LocusRecorder).toBeUndefined();
	});

	test("a tag without ?id= sets the loaded marker but refuses to start, naming the missing param", async () => {
		await autostart({ src: "https://store.example.com/locus-recorder.min.js" });
		expect(errors).toHaveLength(1);
		expect(errors[0]).toContain("?id=");
		expect(errors[0]).toContain("Not starting");
		expect(globalThis.window.LocusRecorder).toEqual({});
	});

	test("an invalid id is refused whatever the tag's filename or origin — identification is not a name match", async () => {
		await autostart({
			src: "https://www.operator.example/assets/a.js?id=NOT-VALID",
		});
		expect(errors).toHaveLength(1);
		expect(errors[0]).toContain("snippet id");
		expect(globalThis.window.LocusRecorder).toEqual({});
	});

	test("a second tag on the page stays quiet and changes nothing — one page, one recording", async () => {
		const claimed = { flush: async () => {} };
		globalThis.window.LocusRecorder = claimed;
		await autostart({
			src: "https://store.example.com/locus-recorder.min.js?id=abc123",
		});
		expect(errors).toEqual([]);
		expect(globalThis.window.LocusRecorder).toBe(claimed);
	});

	test("a throwing start() lands as a console.error, never an unhandled rejection", async () => {
		// This rig's bare window has no location, so start() throws synchronously once autostart
		// hands off — the deterministic stand-in for any start()-time failure.
		await autostart({
			src: "https://store.example.com/locus-recorder.min.js?id=abc123",
		});
		expect(errors).toHaveLength(1);
		expect(errors[0]).toContain("failed to start");
		expect(globalThis.window.LocusRecorder).toEqual({});
		// The page context is otherwise invisible on both planes, so the fault must reach the tag's
		// own store — and nothing else: the one shot the failure path fires.
		expect(fetchTargets).toHaveLength(1);
		expect(fetchTargets[0]).toStartWith(
			"https://store.example.com/telemetry/abc123/",
		);
	});
});
