/**
 * The recorder must refuse exactly the ids the worker's path gate refuses, and cap chunks at exactly
 * the size the worker stores. Both ids ride the chunk key, so a value one side accepts and the other
 * rejects 404s every upload and captures nothing — and the 404 fires before the path is parsed, so
 * it writes no reject telemetry either: the worst blast radius with the least observability.
 *
 * Both sides are driven here, so a change to either one fails this suite.
 */
import { describe, expect, test } from "bun:test";

import { EventType } from "../../recorder/src/rrweb_constants.js";
import { isValidSnippetId } from "../../recorder/src/snippet.js";
import { EventStream, makeSliceId } from "../../recorder/src/stream.js";
import { MAX_GZIPPED_CHUNK_BYTES } from "../../recorder/src/uploader.js";
import {
	isValidVisitorId,
	UNIDENTIFIED_VISITOR,
} from "../../recorder/src/visitor.js";
import {
	chunkTarget,
	MAX_CHUNK_BYTES,
	SLICE_ID_SHAPE,
	telemetryTarget,
} from "../src/keys.js";

const SNIPPET = "abc123";
const VISITOR = "0f6e2a7c-9b1d-4e5f-8a3b-2c4d6e8f0a1b";
const SLICE = "01749600000000-x7k2";
const COUNTER = "1749600000000000001";

const workerAcceptsSnippet = (id) =>
	chunkTarget(`/chunks/${id}/${VISITOR}/${SLICE}/${COUNTER}`) !== null;
const workerAcceptsVisitor = (id) =>
	chunkTarget(`/chunks/${SNIPPET}/${id}/${SLICE}/${COUNTER}`) !== null;

describe("snippet id — recorder validator agrees with the worker path gate", () => {
	for (const id of ["abc123", "123456", "a1b2c3d4e5", "x".repeat(64), "aaa"]) {
		test(`both accept ${JSON.stringify(id)}`, () => {
			expect(isValidSnippetId(id)).toBe(true);
			expect(workerAcceptsSnippet(id)).toBe(true);
		});
	}

	for (const id of [
		"MyApp123",
		"my-app",
		"my_app",
		"ab",
		"",
		"x".repeat(65),
		"abc!",
	]) {
		test(`both reject ${JSON.stringify(id)}`, () => {
			expect(isValidSnippetId(id)).toBe(false);
			expect(workerAcceptsSnippet(id)).toBe(false);
		});
	}
});

// A recorder cap above the store's ships chunks the store refuses forever: a poison batch retried
// to its attempt bound and then dropped, losing exactly the heaviest pages. Below it, the recorder
// is leaving room on the table. They are one number.
test("the recorder's chunk cap is the store's", () => {
	expect(MAX_GZIPPED_CHUNK_BYTES).toBe(MAX_CHUNK_BYTES);
});

// The slice id and counter shapes are written on both sides of the wire — the recorder mints them
// (stream.js), the worker's path gate matches them (keys.js) — so the pin is the recorder's own
// mints driven through the gate. A drift on either side 404s every upload of every affected
// visitor, before the path parses, so it writes no reject telemetry either.
describe("slice id and counter — the recorder's mints pass the worker path gate", () => {
	test("a freshly minted slice id and stamped counter form an accepted upload path", () => {
		const now = Date.now();
		const sliceId = makeSliceId(now);
		const stream = new EventStream(sliceId, "locus-recorder/0.0.0", () => {});
		const [{ event }] = stream.stamp({
			type: EventType.Meta,
			timestamp: now,
			data: {},
		});
		expect(SLICE_ID_SHAPE.test(sliceId)).toBe(true);
		expect(
			chunkTarget(`/chunks/${SNIPPET}/${VISITOR}/${sliceId}/${event.counter}`),
		).not.toBeNull();
	});

	test("a same-millisecond burst mints counters the gate accepts, in lexicographic order", () => {
		const now = Date.now();
		const stream = new EventStream(makeSliceId(now), "v", () => {});
		const counters = [];
		for (let i = 0; i < 3; i += 1) {
			const [{ event }] = stream.stamp({
				type: EventType.Meta,
				timestamp: now,
				data: {},
			});
			counters.push(event.counter);
		}
		expect([...counters].sort()).toEqual(counters);
		for (const counter of counters) {
			expect(
				chunkTarget(
					`/chunks/${SNIPPET}/${VISITOR}/${makeSliceId(now)}/${counter}`,
				),
			).not.toBeNull();
		}
	});
});

describe("visitor id — recorder validator agrees with the worker path gate", () => {
	for (const id of [VISITOR, "abcdefgh", "A-Z_az09", "x".repeat(64)]) {
		test(`both accept ${JSON.stringify(id)}`, () => {
			expect(isValidVisitorId(id)).toBe(true);
			expect(workerAcceptsVisitor(id)).toBe(true);
		});
	}

	// The fault a start() raises before it could learn who the visitor was is a telemetry ping, and
	// the ping is keyed by the visitor: a report of the failure that fails the gate is the failure
	// going unreported.
	test("the sentinel a start-failure is reported under passes the worker's gate", () => {
		expect(
			telemetryTarget(`/telemetry/${SNIPPET}/${UNIDENTIFIED_VISITOR}`),
		).toEqual({
			snippetId: SNIPPET,
			visitorId: UNIDENTIFIED_VISITOR,
		});
	});

	// The values a hostile or careless script on the host page plants in the cookie.
	for (const id of [
		"short",
		"",
		"x".repeat(65),
		"has space",
		"has/slash",
		"has.dot",
		"☃nowman",
	]) {
		test(`both reject ${JSON.stringify(id)}`, () => {
			expect(isValidVisitorId(id)).toBe(false);
			expect(workerAcceptsVisitor(id)).toBe(false);
		});
	}
});
