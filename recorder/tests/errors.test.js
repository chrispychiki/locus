/**
 * describeError: every error report carries real diagnostic content — name and message where they
 * exist, an explicit account of what was thrown where they do not. The observed failure it pins
 * against: IndexedDB rejecting null, which String() collapses to the four characters "null".
 */
import { describe, expect, test } from "bun:test";

import { describeError } from "../src/errors.js";

describe("describeError", () => {
	test.each([
		[null, "null was thrown (no error object)"],
		[undefined, "undefined was thrown (no error object)"],
		["", "an empty string was thrown"],
		["   ", "an empty string was thrown"],
		["plain message", "plain message"],
		[42, "non-error thrown: 42"],
		[false, "non-error thrown: false"],
		[{ foo: 1 }, 'non-error thrown: {"foo":1}'],
	])("%p → %s", (thrown, described) => {
		expect(describeError(thrown)).toBe(described);
	});

	test("an Error carries name and message", () => {
		expect(describeError(new Error("boom"))).toBe("Error: boom");
		expect(describeError(new TypeError("bad"))).toBe("TypeError: bad");
	});

	test("a message-less Error still carries its name", () => {
		expect(describeError(new Error())).toBe("Error");
	});

	test("a DOMException-shaped object carries name and message without being an Error instance", () => {
		expect(
			describeError({ name: "QuotaExceededError", message: "storage full" }),
		).toBe("QuotaExceededError: storage full");
	});

	test("an unserializable object still produces something", () => {
		const circular = {};
		circular.self = circular;
		expect(describeError(circular)).toBe("non-error thrown: [object Object]");
	});
});
