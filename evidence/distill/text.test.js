// The shared text rules. Both consumers — the flat columns and the markdown projection — read
// through these, so a value can never appear two ways depending on which one the reader looks at.
import { expect, test } from "bun:test";

import {
	collapse,
	cutNote,
	describeUrl,
	LABEL_CAP,
	TEXT_CAP,
	truncate,
	truncateWords,
	URL_CAP,
	VALUE_CAP,
} from "./text.js";

test("the role ceilings order by what each role can legitimately be", () => {
	expect(LABEL_CAP).toBeLessThan(TEXT_CAP);
	expect(TEXT_CAP).toBeLessThan(VALUE_CAP);
});

test("cutNote returns the value whole with an empty note below the ceiling", () => {
	expect(cutNote("hello", 10)).toEqual(["hello", ""]);
	expect(cutNote("hello", 5)).toEqual(["hello", ""]);
});

test("cutNote cuts visibly and states first-of-total past the ceiling", () => {
	const [shown, note] = cutNote("x".repeat(600), 200);
	expect(shown.length).toBe(200);
	expect(shown.endsWith("…")).toBe(true);
	expect(note).toBe(" (first 199 of 600 chars)");
});

test("whitespace collapses the way the browser lays it out", () => {
	expect(collapse("  Add   to\n\tcart \n")).toBe("Add to cart");
	expect(collapse("\n\n")).toBe("");
});

test("a short string is returned unchanged, and a non-string is passed through", () => {
	expect(truncate("hello", 10)).toBe("hello");
	expect(truncate("hello", 5)).toBe("hello");
	expect(truncate(null, 5)).toBe(null);
	expect(truncate(42, 5)).toBe(42);
});

test("a cut is visible and never grows the cap", () => {
	expect(truncate("abcdef", 4)).toBe("abc…");
	expect(truncate("abcdef", 4).length).toBe(4);
});

test("a cut never splits a surrogate pair", () => {
	// Every astral character is two code units, so a naive slice at an odd offset mints a lone
	// surrogate — an unpaired half that is not a character, and that SQLite and JSON both reject.
	const s = "ab\u{1F600}cd"; // a b <hi> <lo> c d
	expect(truncate(s, 3)).toBe("ab…"); // the cut lands inside the emoji: drop it whole
	expect(truncate(s, 5)).toBe("ab\u{1F600}…"); // the emoji fits ahead of the marker: keep it
	for (let n = 1; n <= s.length; n += 1) {
		const cut = truncate(s, n);
		expect(cut).toBe(cut.toWellFormed()); // no lone surrogate at any cut point
		expect(cut.length).toBeLessThanOrEqual(n); // the marker rides inside the cap
	}
});

test("a combining mark the cut reached is kept — it is real text, not a fragment", () => {
	const s = "café latte"; // e + combining acute: 5 code units through the accent
	expect(truncate(s, 4)).toBe("caf…"); // the cut stops before the mark
	expect(truncate(s, 6)).toBe("café…");
	// A cut can never orphan a mark: a prefix holding one holds the base that precedes it.
	expect(truncate(s, 6).normalize("NFC")).toBe("café…");
});

test("a URL under the ceiling rides whole — semantic queries are content, not churn", () => {
	const searchy = `https://shop.test/results?q=${"flannel+overshirt+".repeat(20)}`;
	expect(describeUrl(searchy)).toBe(searchy);
	expect(describeUrl("x".repeat(URL_CAP)).length).toBe(URL_CAP);
});

test("past the ceiling the query sheds whole and testifies its size", () => {
	const long = `https://shop.test/product?${"x".repeat(3000)}`;
	expect(describeUrl(long)).toBe("https://shop.test/product?… (3kB query)");
});

test("a shed query keeps its fragment — a hash-routed SPA's route lives there", () => {
	const tracked = `https://school.test/?_gl=1*${"x".repeat(3000)}#/recording/42`;
	expect(describeUrl(tracked)).toBe(
		"https://school.test/?… (3kB query)#/recording/42",
	);
});

test("a URL whose path alone exceeds the ceiling cuts visibly after the shed falls short", () => {
	const long = `https://shop.test/${"p".repeat(URL_CAP + 100)}?q=term`;
	const shown = describeUrl(long);
	expect(shown.length).toBe(URL_CAP);
	expect(shown.endsWith("…")).toBe(true);
});

test("a long URL with no query to shed is cut visibly at the cap", () => {
	const long = `https://shop.test/${"p/".repeat(URL_CAP)}`;
	const shown = describeUrl(long);
	expect(shown.length).toBe(URL_CAP);
	expect(shown.endsWith("…")).toBe(true);
	expect(long.startsWith(shown.slice(0, -1))).toBe(true);
});

test("an inlined data URI is summarized, never carried", () => {
	// The testimony is that an image is present, not its bytes — a 2MB base64 blob in a src
	// attribute would otherwise be the largest thing in the event stream and say nothing.
	const uri = `data:image/png;base64,${"A".repeat(4096)}`;
	const shown = describeUrl(uri);
	expect(shown).toContain("data:image/png;base64,");
	expect(shown).toContain("inlined");
	expect(shown).not.toContain("AAAA");
	expect(shown.length).toBeLessThan(80);
});

test("a data URI's size is reported in units a reader can weigh", () => {
	expect(describeUrl("data:text/plain,hello")).toContain("21 chars");
	expect(describeUrl(`data:image/png;base64,${"A".repeat(2048)}`)).toContain(
		"2kB",
	);
});

test("a headerless data URI is still summarized rather than dumped", () => {
	const shown = describeUrl(`data:${"A".repeat(500)}`);
	expect(shown).toContain("inlined");
	expect(shown.length).toBeLessThan(80);
});

test("a non-string url is passed through untouched", () => {
	expect(describeUrl(undefined)).toBe(undefined);
	expect(describeUrl(null)).toBe(null);
});

test("a word-boundary cut lands after a whole word, visibly", () => {
	const s = "Premium heavyweight flannel overshirt in forest green";
	const cut = truncateWords(s, 30);
	expect(cut.endsWith("…")).toBe(true);
	expect(cut.length).toBeLessThanOrEqual(30);
	expect(s.startsWith(cut.slice(0, -1))).toBe(true);
	expect(cut.slice(0, -1).endsWith(" ")).toBe(false);
	expect(s[cut.length - 1]).toBe(" ");
});

test("unspaced text still caps — a word boundary is used where one exists, never required", () => {
	const cjk = "商品説明".repeat(40);
	const cut = truncateWords(cjk, 30);
	expect(cut.length).toBeLessThanOrEqual(30);
	expect(cut.endsWith("…")).toBe(true);
});

test("a short string passes truncateWords unchanged", () => {
	expect(truncateWords("Add to cart", 80)).toBe("Add to cart");
	expect(truncateWords(null, 80)).toBe(null);
});
