/**
 * isValidSnippetId's own behavior — what it accepts and what it refuses.
 *
 * That it agrees with the store is pinned where both sides can be driven, in
 * store/tests/snippet-id.test.js, which runs this validator and the worker's path gate over the
 * same ids.
 */
import { describe, expect, test } from "bun:test";

import { isValidSnippetId } from "../src/snippet.js";

describe("isValidSnippetId", () => {
	test("accepts lowercase alphanumeric ids within the length bounds", () => {
		expect(isValidSnippetId("k4p2ma")).toBe(true);
		expect(isValidSnippetId("abc")).toBe(true); // min length 3
		expect(isValidSnippetId("a".repeat(64))).toBe(true); // max length 64
		expect(isValidSnippetId("9z8y7x")).toBe(true);
	});

	test("rejects ids outside the length bounds", () => {
		expect(isValidSnippetId("ab")).toBe(false);
		expect(isValidSnippetId("a".repeat(65))).toBe(false);
		expect(isValidSnippetId("")).toBe(false);
	});

	test("rejects anything with non [a-z0-9] characters", () => {
		expect(isValidSnippetId("MyApp")).toBe(false); // uppercase
		expect(isValidSnippetId("ab_cd")).toBe(false); // underscore
		expect(isValidSnippetId("ab-cd")).toBe(false); // dash
		expect(isValidSnippetId("ab cd")).toBe(false); // space
		expect(isValidSnippetId("ab/cd")).toBe(false); // path separator
	});

	test("rejects non-strings", () => {
		expect(isValidSnippetId(null)).toBe(false);
		expect(isValidSnippetId(undefined)).toBe(false);
		expect(isValidSnippetId(123456)).toBe(false);
		expect(isValidSnippetId({})).toBe(false);
	});

	test("rejects whitespace-padded ids — the store gate anchors and would 404 them", () => {
		// Padding sneaks past a non-anchored or trimming check, and the store's gate rejects it.
		expect(isValidSnippetId(" k4p2ma")).toBe(false);
		expect(isValidSnippetId("k4p2ma ")).toBe(false);
		expect(isValidSnippetId("k4p2ma\n")).toBe(false);
		expect(isValidSnippetId("\tk4p2ma")).toBe(false);
	});
});
