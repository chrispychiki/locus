/**
 * Every invariant provision and doctor assert is a field inside Cloudflare's response envelope
 * (scripts/cf.js), so the failure that matters is not a wrong value but a missing one — a proxy's
 * HTML error page, a captive portal, a truncated body. These pin the bodies that must not count as
 * answers.
 */
import { describe, expect, test } from "bun:test";

process.env.CLOUDFLARE_ACCOUNT_ID ??= "test-account";
process.env.CLOUDFLARE_API_TOKEN ??= "test-token";

const { readEnvelope, errMsg } = await import("../scripts/cf.js");

describe("a Cloudflare response is an answer only when it says it is", () => {
	test("the envelope Cloudflare actually sends is an answer", () => {
		const { body, answered } = readEnvelope(
			'{"success":true,"errors":[],"result":{"enabled":false}}',
		);
		expect(answered).toBe(true);
		expect(body.result.enabled).toBe(false);
	});

	test("an HTML error page is not an answer — least of all a reassuring one", () => {
		// Read as an answer, an HTML 502 would hand every script the invariant absent — the shape
		// each of them reads as the safe case. It must fail, and say what it actually was.
		const { body, answered } = readEnvelope(
			"<!doctype html><title>502 Bad Gateway</title>",
		);
		expect(answered).toBe(false);
		expect(errMsg({ body })).toContain("not a Cloudflare API envelope");
	});

	test("an empty body is not an answer", () => {
		expect(readEnvelope("").answered).toBe(false);
	});

	test("valid JSON that is not an object is not an answer, and never a crash", () => {
		// `JSON.parse("null")` parses clean — a verdict must come back, not a TypeError.
		const { body, answered } = readEnvelope("null");
		expect(answered).toBe(false);
		expect(errMsg({ body })).toBe("the response carried no success");
	});

	test("a parsed envelope with no success field says so", () => {
		const { body, answered } = readEnvelope('{"result":{}}');
		expect(answered).toBe(false);
		expect(errMsg({ body })).toBe("the response carried no success");
	});

	test("a well-formed failure is not an answer, and says why", () => {
		const { body, answered } = readEnvelope(
			'{"success":false,"errors":[{"code":10004,"message":"bucket not found"}]}',
		);
		expect(answered).toBe(false);
		expect(errMsg({ body })).toBe("bucket not found [10004]");
	});
});
