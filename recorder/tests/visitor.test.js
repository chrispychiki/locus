import { beforeEach, describe, expect, test } from "bun:test";

import { getOrCreateVisitorId, getVisitorIdentity } from "../src/visitor.js";

let jar;
beforeEach(() => {
	jar = "";
	globalThis.document = {
		get cookie() {
			return jar;
		},
		set cookie(value) {
			jar = value.split(";")[0];
		},
	};
});

describe("getOrCreateVisitorId", () => {
	test("returns a valid existing cookie unchanged", () => {
		jar = "locusVisitorId=abcd1234-ef56-7890-abcd-ef1234567890";
		expect(getOrCreateVisitorId()).toBe("abcd1234-ef56-7890-abcd-ef1234567890");
	});

	test("replaces a cookie outside the key charset with a fresh id", () => {
		jar = "locusVisitorId=bad id/with spaces";
		const id = getOrCreateVisitorId();
		expect(id).not.toBe("bad id/with spaces");
		expect(id).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
	});

	test("mints an id when no cookie is set", () => {
		expect(getOrCreateVisitorId()).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
	});

	// An id one character short or one over bounces every upload at ingest, so both edges matter.
	test("accepts an existing cookie at the exact length bounds unchanged", () => {
		jar = `locusVisitorId=${"a".repeat(8)}`;
		expect(getOrCreateVisitorId()).toBe("a".repeat(8));
		jar = `locusVisitorId=${"a".repeat(64)}`;
		expect(getOrCreateVisitorId()).toBe("a".repeat(64));
	});

	test("replaces a cookie just outside the length bounds", () => {
		jar = `locusVisitorId=${"a".repeat(7)}`;
		expect(getOrCreateVisitorId()).not.toBe("a".repeat(7));
		jar = `locusVisitorId=${"a".repeat(65)}`;
		expect(getOrCreateVisitorId()).not.toBe("a".repeat(65));
	});

	test("a valid id is stable across repeated calls — never re-minted", () => {
		jar = "locusVisitorId=Stable_Visitor-01";
		const first = getOrCreateVisitorId();
		const second = getOrCreateVisitorId();
		expect(first).toBe("Stable_Visitor-01");
		expect(second).toBe(first);
	});

	// Without the write-back the next page mints yet another id, and the visitor fragments into a
	// string of one-page sessions.
	test("a freshly minted id is persisted so the next page reads it back", () => {
		const minted = getOrCreateVisitorId();
		expect(jar).toBe(`locusVisitorId=${minted}`);
		const next = getOrCreateVisitorId();
		expect(next).toBe(minted);
	});

	// randomUUID exists only in secure contexts, so a plain-HTTP deployment mints every id
	// through the fallback — it must produce the same well-formed v4 shape from getRandomValues.
	test("minting without randomUUID yields a well-formed v4 uuid", () => {
		const realCrypto = globalThis.crypto;
		globalThis.crypto = {
			getRandomValues: (array) => realCrypto.getRandomValues(array),
		};
		try {
			const id = getOrCreateVisitorId();
			expect(id).toMatch(
				/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
			);
		} finally {
			globalThis.crypto = realCrypto;
		}
	});

	// The bad value is overwritten in the jar, so a later page cannot re-read it.
	test("replacing an invalid cookie overwrites it in the jar", () => {
		jar = "locusVisitorId=bad id/with spaces";
		const id = getOrCreateVisitorId();
		expect(jar).toBe(`locusVisitorId=${id}`);
		expect(id).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
	});
});

describe("getVisitorIdentity", () => {
	test("an id already in the jar is reported as coming from the cookie", () => {
		jar = "locusVisitorId=Stable_Visitor-01";
		expect(getVisitorIdentity()).toEqual({
			id: "Stable_Visitor-01",
			source: "cookie",
		});
	});

	test("a minted id the jar accepts is reported as written", () => {
		const { id, source } = getVisitorIdentity();
		expect(source).toBe("written");
		expect(jar).toBe(`locusVisitorId=${id}`);
	});

	// A jar that drops the write is silent about it, so every page mints again and one visitor
	// arrives as many. Reading back is the only way the page finds out.
	test("a jar that silently drops the write reports the identity as unpersisted", () => {
		globalThis.document = {
			get cookie() {
				return "";
			},
			set cookie(_value) {
				/* how a blocked jar refuses: no throw, no effect */
			},
		};
		const { id, source } = getVisitorIdentity();
		expect(source).toBe("unpersisted");
		expect(id).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
	});

	// The read-back compares the value rather than merely finding the key: a jar that keeps
	// something other than what was written leaves the next page reading an id this one never
	// issued.
	test("a jar that stores a different value than was written reports unpersisted", () => {
		globalThis.document = {
			get cookie() {
				return jar;
			},
			set cookie(_value) {
				jar = "locusVisitorId=not_what_was_written";
			},
		};
		expect(getVisitorIdentity().source).toBe("unpersisted");
	});
});
