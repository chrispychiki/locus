/**
 * pageAddress: what a page's url becomes before it reaches the telemetry plane.
 *
 * The plane is unmasked by construction, so the assertions here are about what never leaves the
 * page — the query string in every place a url can hold one — as much as about what does.
 */
import { describe, expect, test } from "bun:test";

import { pageAddress } from "../src/address.js";

describe("pageAddress", () => {
	test("drops the query string and keeps what locates the page", () => {
		expect(
			pageAddress("https://shop.example.com/checkout?email=a%40b.com&token=t"),
		).toBe("https://shop.example.com/checkout");
		expect(
			pageAddress("https://shop.example.com/land?gclid=xyz&utm_source=ads"),
		).toBe("https://shop.example.com/land");
		expect(pageAddress("https://shop.example.com:8443/p?x=1")).toBe(
			"https://shop.example.com:8443/p",
		);
	});

	test("leaves a query-less address alone", () => {
		expect(pageAddress("https://shop.example.com/checkout")).toBe(
			"https://shop.example.com/checkout",
		);
	});

	test("keeps the fragment — a hash-routed SPA's route lives there — and strips its query too", () => {
		expect(pageAddress("https://app.example.com/?a=1#/checkout")).toBe(
			"https://app.example.com/#/checkout",
		);
		expect(pageAddress("https://app.example.com/#/checkout?token=secret")).toBe(
			"https://app.example.com/#/checkout",
		);
	});

	test("userinfo does not survive the origin", () => {
		expect(pageAddress("https://user:secret@example.com/p?x=1")).toBe(
			"https://example.com/p",
		);
	});

	test("an address no parser accepts is still never carried whole", () => {
		expect(pageAddress("about:blank?x=1")).toBe("about:blank");
		expect(pageAddress("/relative/path?token=t")).toBe("/relative/path");
		expect(pageAddress("/relative/path#/r?token=t")).toBe("/relative/path#/r");
		// Every `?`-run, not just the first: an unparseable address can hold one before the
		// fragment and another inside it, and a survivor carries the secret whole.
		expect(pageAddress("/relative/path?email=a@b.com#/r?token=t")).toBe(
			"/relative/path#/r",
		);
		expect(pageAddress("")).toBe("");
		expect(pageAddress(undefined)).toBe("");
	});
});
