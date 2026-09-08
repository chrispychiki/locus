/**
 * The rig every start()-level suite drives the recorder through: one browser-globals scaffold, one
 * window double, one set of event helpers.
 *
 * The rrweb and BotD mocks stay per-file: bun's mock.module is file-scoped, and each suite needs a
 * different rrweb (some count checkouts, some capture options, some just hold the emit).
 */

import { IDBFactory } from "fake-indexeddb";
import { gunzipSync } from "fflate";

import { EventType } from "../src/rrweb_constants.js";

/**
 * A window double carrying the whole surface the recorder touches: location, the history methods it
 * patches, and the listener table it registers on.
 *
 * `dispatch(type)` fires the listeners the recorder registered (popstate, hashchange), which take no
 * event object; `listenerCount(type)` is how a test sees that terminate() unregistered them.
 */
export function makeWindow({
	hostname = "client.example.com",
	href = "https://client.example.com/p",
} = {}) {
	const listeners = {};
	const location = { hostname, href };
	return {
		location,
		history: {
			pushState(_state, _title, url) {
				if (url) location.href = url;
			},
			replaceState(_state, _title, url) {
				if (url) location.href = url;
			},
		},
		addEventListener: (type, fn) => {
			listeners[type] ||= [];
			listeners[type].push(fn);
		},
		removeEventListener: (type, fn) => {
			listeners[type] = (listeners[type] || []).filter((f) => f !== fn);
		},
		dispatch: (type) => {
			(listeners[type] || []).slice().forEach((fn) => {
				fn();
			});
		},
		listenerCount: (type) => (listeners[type] || []).length,
	};
}

/**
 * The browser globals start() reads, installed on globalThis. Returns a teardown.
 *
 * `document` is the piece each suite varies — a title getter that throws or counts reads, a listener
 * table that captures visibilitychange — so it is merged over the plain default rather than
 * replacing it, and a suite that only wants a title getter need not restate the rest.
 */
export function installBrowserGlobals({
	window = makeWindow(),
	document = {},
	performance = null,
} = {}) {
	const realPerformance = globalThis.performance;
	globalThis.indexedDB = new IDBFactory();
	globalThis.window = window;
	// Descriptors, never a spread: a suite that hands over `{ get title() { reads += 1; ... } }` is
	// counting the recorder's reads, and spreading would invoke the getter once here and install its
	// return value, leaving the suite measuring nothing and passing.
	globalThis.document = Object.defineProperties(
		{
			title: "t",
			referrer: "",
			visibilityState: "visible",
			cookie: "",
			addEventListener: () => {},
			removeEventListener: () => {},
		},
		Object.getOwnPropertyDescriptors(document),
	);
	globalThis.performance = performance ?? realPerformance ?? { now: () => 0 };
	globalThis.navigator = {
		userAgent: "ua",
		language: "en",
		sendBeacon: () => true,
	};
	globalThis.screen = { width: 1, height: 1 };

	return () => {
		globalThis.performance = realPerformance;
		delete globalThis.indexedDB;
		delete globalThis.window;
		delete globalThis.document;
		delete globalThis.navigator;
		delete globalThis.screen;
	};
}

/**
 * A sink that retains every uploaded chunk as its decoded payload. sliceId is a property of the
 * per-slice chunk payload (one slice per chunk — chunk.js groups by slice), NOT of each event inside
 * payload.events, so slice membership must be read at the payload level.
 */
export function capturingSink() {
	const chunks = [];
	return {
		chunks,
		send: async (bytes, descriptor) => {
			chunks.push({
				descriptor,
				payload: JSON.parse(new TextDecoder().decode(gunzipSync(bytes))),
			});
		},
	};
}

export const pageLoads = (sink) =>
	sink.chunks
		.flatMap((c) => c.payload.events)
		.filter((e) => e.type === EventType.PageLoad);

export const sliceCount = (sink) =>
	new Set(sink.chunks.map((c) => c.payload.sliceId)).size;

export const meta = (ts, href = "https://x") => ({
	type: EventType.Meta,
	data: { href, width: 1, height: 1 },
	timestamp: ts,
});

export const fullSnapshot = (ts) => ({
	type: EventType.FullSnapshot,
	data: { node: {} },
	timestamp: ts,
});

/** Let the recorder's own microtasks and its drain timer run. */
export const settle = () => new Promise((r) => setTimeout(r, 50));
