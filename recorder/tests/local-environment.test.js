/**
 * The local-environment gate (environment.js): whole hosts and whole addresses, never substrings —
 * loopback, RFC 1918 and link-local v4, ULA and link-local v6, localhost and .local names.
 */
import { describe, expect, test } from "bun:test";

import { isLocalEnvironment } from "../src/environment.js";

describe("isLocalEnvironment", () => {
	test("local hosts and addresses match", () => {
		for (const host of [
			"localhost",
			"app.localhost",
			"local",
			"myapp.local",
			"127.0.0.1",
			"127.1.2.3",
			"10.0.0.5",
			"192.168.1.20",
			"172.16.0.1",
			"172.31.255.255",
			"169.254.169.254",
			"::1",
			"[::1]",
			"fd00::1",
			"[fd12:3456:789a::1]",
			"fc00::1",
			"fe80::1",
			"[fe80::abcd:ef01]",
		]) {
			expect(isLocalEnvironment(host)).toBe(true);
		}
	});

	test("public hosts and addresses do not", () => {
		for (const host of [
			"example.com",
			"localhost.tools.acme.com",
			"10.example.com",
			"mylocal.dev",
			"172.32.0.1",
			"172.15.0.1",
			"192.169.0.1",
			"169.253.0.1",
			"8.8.8.8",
			"2606:4700::1111",
			"fe00::1",
			"febf.example.com",
			"",
		]) {
			expect(isLocalEnvironment(host)).toBe(false);
		}
	});
});
