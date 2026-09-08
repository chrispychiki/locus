import { describe, expect, test } from "bun:test";

import { optionsForUrl } from "../src/masking.js";

const RULES = [
	{ pattern: "*", options: { maskInputOptions: { password: true } } },
	{
		pattern: "shop.example.com",
		options: { maskInputOptions: { password: true, email: true } },
	},
	{
		pattern: "shop.example.com/checkout",
		options: { maskAllInputs: true },
	},
];

describe("optionsForUrl", () => {
	test("matching rules layer — last wins per key, broader masking preserved", () => {
		expect(
			optionsForUrl(RULES, "https://shop.example.com/checkout/pay"),
		).toEqual({
			maskAllInputs: true,
			maskInputOptions: { password: true, email: true },
		});
		expect(optionsForUrl(RULES, "https://shop.example.com/catalog")).toEqual({
			maskInputOptions: { password: true, email: true },
		});
	});

	test("the catch-all applies when nothing more specific matches", () => {
		expect(optionsForUrl(RULES, "https://other.example.org/")).toEqual({
			maskInputOptions: { password: true },
		});
	});

	test("matching ignores protocol and www", () => {
		expect(
			optionsForUrl(RULES, "http://www.shop.example.com/checkout"),
		).toEqual({
			maskAllInputs: true,
			maskInputOptions: { password: true, email: true },
		});
	});

	test("no rules and no match both leave the base posture in force", () => {
		expect(optionsForUrl([], "https://anything.test/")).toEqual({});
		expect(
			optionsForUrl(
				[{ pattern: "only.test", options: { maskAllInputs: true } }],
				"https://elsewhere.test/",
			),
		).toEqual({});
	});

	test("a host pattern matches the host and its subdomains, at a label boundary", () => {
		const rules = [
			{ pattern: "example.com", options: { maskAllInputs: true } },
		];
		const masked = { maskAllInputs: true };
		expect(optionsForUrl(rules, "https://example.com/")).toEqual(masked);
		expect(optionsForUrl(rules, "https://example.com/page?x=1#h")).toEqual(
			masked,
		);
		expect(optionsForUrl(rules, "https://app.example.com/account")).toEqual(
			masked,
		); // subdomain
		expect(optionsForUrl(rules, "https://notexample.com/")).toEqual({}); // prefix lookalike
		expect(optionsForUrl(rules, "https://example.com.evil.com/")).toEqual({}); // suffix attack
		expect(optionsForUrl(rules, "https://other.test/p/example.com/x")).toEqual(
			{},
		); // substring in path
	});

	test("a path-prefix matches at a segment boundary; query dropped, fragment kept", () => {
		const rules = [
			{ pattern: "site.test/checkout", options: { maskAllInputs: true } },
		];
		const masked = { maskAllInputs: true };
		expect(optionsForUrl(rules, "https://site.test/checkout")).toEqual(masked);
		expect(optionsForUrl(rules, "https://site.test/checkout/pay")).toEqual(
			masked,
		);
		expect(optionsForUrl(rules, "https://site.test/checkout?step=2")).toEqual(
			masked,
		);
		expect(optionsForUrl(rules, "https://site.test/checkoutX")).toEqual({}); // not a boundary
		expect(optionsForUrl(rules, "https://site.test/?to=/checkout")).toEqual({}); // query, not path
	});

	test("a hash-routed path matches via the kept fragment", () => {
		const rules = [
			{ pattern: "app.test/#/checkout", options: { maskAllInputs: true } },
		];
		expect(optionsForUrl(rules, "https://app.test/#/checkout")).toEqual({
			maskAllInputs: true,
		});
		expect(optionsForUrl(rules, "https://app.test/#/checkout/pay")).toEqual({
			maskAllInputs: true,
		});
		expect(optionsForUrl(rules, "https://app.test/#/home")).toEqual({});
	});

	test("a hash-router's own query is not route identity either", () => {
		const rules = [
			{ pattern: "app.test/#/checkout", options: { maskAllInputs: true } },
		];
		const masked = { maskAllInputs: true };
		expect(
			optionsForUrl(rules, "https://app.test/#/checkout?token=secret"),
		).toEqual(masked);
		// a real query ahead of the fragment must not shelter the fragment's own
		expect(
			optionsForUrl(rules, "https://app.test/?ref=ads#/checkout?token=secret"),
		).toEqual(masked);
		expect(
			optionsForUrl(rules, "https://app.test/#/home?to=/checkout"),
		).toEqual({});
	});

	// An implementation that compiled the pattern to an unescaped regex would over-match here.
	test("the pattern is matched literally, never as a regex", () => {
		const rules = [
			{ pattern: "a.b.test/p+q", options: { maskAllInputs: true } },
		];
		expect(optionsForUrl(rules, "https://a.b.test/p+q")).toEqual({
			maskAllInputs: true,
		});
		expect(optionsForUrl(rules, "https://axb.test/p+q")).toEqual({});
		expect(optionsForUrl(rules, "https://a.b.test/pXq")).toEqual({});
	});

	// The rule a deployment writes is often copied out of the address bar, protocol and www and all.
	test("protocol and www are stripped from the pattern, not only the url", () => {
		const rules = [
			{
				pattern: "https://www.shop.example.com",
				options: { maskAllInputs: true },
			},
		];
		expect(optionsForUrl(rules, "https://shop.example.com/checkout")).toEqual({
			maskAllInputs: true,
		});
		expect(optionsForUrl(rules, "http://www.shop.example.com/")).toEqual({
			maskAllInputs: true,
		});
	});

	test("rule order decides the winner per top-level key", () => {
		const broadThenNarrow = [
			{ pattern: "site.test", options: { maskAllInputs: false } },
			{ pattern: "site.test/admin", options: { maskAllInputs: true } },
		];
		expect(optionsForUrl(broadThenNarrow, "https://site.test/admin")).toEqual({
			maskAllInputs: true,
		});

		const narrowThenBroad = [
			{ pattern: "site.test/admin", options: { maskAllInputs: true } },
			{ pattern: "site.test", options: { maskAllInputs: false } },
		];
		expect(optionsForUrl(narrowThenBroad, "https://site.test/admin")).toEqual({
			maskAllInputs: false,
		});
	});

	test("maskInputOptions deep-merges across layers, never replacing the whole object", () => {
		const rules = [
			{ pattern: "*", options: { maskInputOptions: { password: true } } },
			{ pattern: "forms.test", options: { maskInputOptions: { email: true } } },
			{
				pattern: "forms.test/ssn",
				options: { maskInputOptions: { text: true } },
			},
		];
		expect(optionsForUrl(rules, "https://forms.test/ssn/step1")).toEqual({
			maskInputOptions: { password: true, email: true, text: true },
		});
	});

	// Deep-merge is last-write-wins per key, not a monotonic union of trues.
	test("a later maskInputOptions key overrides the same key from an earlier layer", () => {
		const rules = [
			{
				pattern: "*",
				options: { maskInputOptions: { password: true, email: true } },
			},
			{
				pattern: "public.test",
				options: { maskInputOptions: { email: false } },
			},
		];
		expect(optionsForUrl(rules, "https://public.test/page")).toEqual({
			maskInputOptions: { password: true, email: false },
		});
	});

	// A differing resolution is what the recorder keys its per-route restart on (index.js).
	test("two routes under route-specific rules resolve to different option sets", () => {
		const rules = [
			{
				pattern: "app.test",
				options: { maskInputOptions: { password: true } },
			},
			{ pattern: "app.test/billing", options: { maskAllInputs: true } },
		];
		expect(optionsForUrl(rules, "https://app.test/home")).toEqual({
			maskInputOptions: { password: true },
		});
		expect(optionsForUrl(rules, "https://app.test/billing")).toEqual({
			maskAllInputs: true,
			maskInputOptions: { password: true },
		});
	});

	// The other side of it: identical resolutions leave the recorder nothing to restart for.
	test("one rule spanning multiple routes masks each identically", () => {
		const rules = [
			{
				pattern: "app.test",
				options: { maskAllInputs: true, maskInputOptions: { password: true } },
			},
		];
		const strict = {
			maskAllInputs: true,
			maskInputOptions: { password: true },
		};
		expect(optionsForUrl(rules, "https://app.test/home")).toEqual(strict);
		expect(optionsForUrl(rules, "https://app.test/billing")).toEqual(strict);
	});

	test("deep-merge holds when only one layer carries maskInputOptions", () => {
		const onlyLaterNested = [
			{ pattern: "site.test", options: { maskAllInputs: true } },
			{
				pattern: "site.test/in",
				options: { maskInputOptions: { email: true } },
			},
		];
		expect(optionsForUrl(onlyLaterNested, "https://site.test/in")).toEqual({
			maskAllInputs: true,
			maskInputOptions: { email: true },
		});

		const onlyEarlierNested = [
			{ pattern: "site.test", options: { maskInputOptions: { email: true } } },
			{ pattern: "site.test/in", options: { maskAllInputs: true } },
		];
		expect(optionsForUrl(onlyEarlierNested, "https://site.test/in")).toEqual({
			maskAllInputs: true,
			maskInputOptions: { email: true },
		});
	});

	test("a fragment anchor never lets a page escape its path rule", () => {
		const rules = [
			{ pattern: "site.test/checkouts", options: { maskAllInputs: true } },
		];
		expect(optionsForUrl(rules, "https://site.test/checkouts#payment")).toEqual(
			{ maskAllInputs: true },
		);
		expect(optionsForUrl(rules, "https://site.test/checkouts/pay#cc")).toEqual({
			maskAllInputs: true,
		});
		expect(optionsForUrl(rules, "https://site.test/checkoutsX")).toEqual({});
	});

	// One rule set is resolved for every URL for the recording's life, so a merge that wrote back
	// into rule.options would let one URL's resolution poison the next.
	test("resolution does not mutate the input rule options", () => {
		const innerOptions = { maskInputOptions: { email: true } };
		const rules = [
			{ pattern: "*", options: { maskInputOptions: { password: true } } },
			{ pattern: "site.test", options: innerOptions },
		];
		optionsForUrl(rules, "https://site.test/a");
		optionsForUrl(rules, "https://site.test/b");
		expect(innerOptions).toEqual({ maskInputOptions: { email: true } });
		expect(rules[0].options).toEqual({ maskInputOptions: { password: true } });
	});
});
