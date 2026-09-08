/**
 * The credential mask (masking.js): maskingPosture routes every input value through its composed
 * maskInputFn, which masks credential elements unconditionally — by live type, by type history,
 * by autocomplete tokens, by name/id credential tokens — and otherwise applies exactly the
 * resolved rule's own intent.
 */
import { describe, expect, test } from "bun:test";

import {
	isCredentialInput,
	maskingPosture,
	ROUTE_ALL_INPUTS,
} from "../src/masking.js";

function el({
	tag = "INPUT",
	type,
	autocomplete,
	name,
	id,
	rrPassword = false,
} = {}) {
	const attrs = {};
	if (autocomplete !== undefined) attrs.autocomplete = autocomplete;
	if (name !== undefined) attrs.name = name;
	if (id !== undefined) attrs.id = id;
	if (rrPassword) attrs["data-rr-is-password"] = "";
	return {
		tagName: tag,
		type: type ?? (tag === "INPUT" ? "text" : undefined),
		hasAttribute: (name) => name in attrs,
		getAttribute: (name) => (name in attrs ? attrs[name] : null),
	};
}

describe("isCredentialInput", () => {
	test("live type=password is a credential", () => {
		expect(isCredentialInput(el({ type: "password" }))).toBe(true);
	});

	test("an element once seen as password stays a credential after a show-password toggle re-types it", () => {
		const field = el({ type: "password" });
		expect(isCredentialInput(field)).toBe(true);
		field.type = "text";
		expect(isCredentialInput(field)).toBe(true);
	});

	test("a field that was never password and carries no credential autocomplete is not a credential, whatever it holds", () => {
		expect(isCredentialInput(el({ type: "text" }))).toBe(false);
		expect(isCredentialInput(el({ type: "email" }))).toBe(false);
	});

	test("credential autocomplete tokens decide regardless of type", () => {
		for (const token of [
			"current-password",
			"new-password",
			"one-time-code",
			"cc-number",
			"cc-csc",
			"cc-exp",
			"cc-exp-month",
			"cc-exp-year",
		]) {
			expect(isCredentialInput(el({ type: "text", autocomplete: token }))).toBe(
				true,
			);
		}
	});

	test("autocomplete is matched per token, case-insensitively", () => {
		expect(isCredentialInput(el({ autocomplete: "billing CC-Number" }))).toBe(
			true,
		);
		expect(isCredentialInput(el({ autocomplete: "email" }))).toBe(false);
		expect(isCredentialInput(el({ autocomplete: "" }))).toBe(false);
	});

	test("rrweb's data-rr-is-password escape hatch counts as password", () => {
		expect(isCredentialInput(el({ rrPassword: true }))).toBe(true);
	});

	test("a re-rendered login field is a credential by its name/id, history-less and autocomplete=off (the show-password leak shape)", () => {
		expect(
			isCredentialInput(
				el({
					type: "text",
					autocomplete: "off",
					name: "password",
					id: "password",
				}),
			),
		).toBe(true);
	});

	test("name/id tokens match on the normalized form — separators stripped, case folded, decorations kept", () => {
		expect(isCredentialInput(el({ name: "current-password" }))).toBe(true);
		expect(isCredentialInput(el({ name: "user_pw" }))).toBe(true);
		expect(isCredentialInput(el({ id: "Password2" }))).toBe(true);
		expect(isCredentialInput(el({ name: "confirm password" }))).toBe(true);
	});

	test("id alone decides when name is absent", () => {
		expect(isCredentialInput(el({ id: "passwd" }))).toBe(true);
	});

	test("multilingual and payment name tokens are credentials", () => {
		for (const name of [
			"contraseña",
			"密码",
			"jelszó",
			"mật_khẩu",
			"wachtwoord",
			"card-number",
			"cvv",
			"iban",
			"routing_number",
		]) {
			expect(isCredentialInput(el({ name }))).toBe(true);
		}
	});

	test("a decomposed (NFD) name matches the same as its precomposed form", () => {
		// Latin diacritics decompose into combining marks, which the name collapse strips —
		// katakana dakuten and Hangul syllables decompose into characters that are letters, so
		// they survive the collapse as a different word and only composition reunites them with
		// the token lists.
		for (const name of [
			"contraseña",
			"hasło",
			"mật_khẩu",
			"jelszó",
			"パスワード",
			"암호",
			"비밀번호",
		]) {
			expect(isCredentialInput(el({ name: name.normalize("NFD") }))).toBe(true);
		}
	});

	test("ambiguous English words fire only as the whole name, so their tokened uses stay clear", () => {
		expect(isCredentialInput(el({ name: "pass" }))).toBe(true);
		expect(isCredentialInput(el({ name: "pin" }))).toBe(true);
		expect(isCredentialInput(el({ name: "swift" }))).toBe(true);
		expect(isCredentialInput(el({ id: "bic" }))).toBe(true);
		for (const name of [
			"passenger-count",
			"passport",
			"shipping",
			"pinboard",
			"pin-code",
			"otc-medications",
			"swiftDelivery",
			"swift-delivery",
			"swiftShipping",
			"bicColor",
		]) {
			expect(isCredentialInput(el({ name }))).toBe(false);
		}
	});

	test("the bank compounds still fire as token runs where the bare word cannot", () => {
		for (const name of ["swift-code", "swiftCode", "bic_code", "bicCode"]) {
			expect(isCredentialInput(el({ name }))).toBe(true);
		}
	});

	test("camelCase humps are token boundaries, so collapse collisions stay clear", () => {
		for (const name of [
			"hasLocation",
			"hasLoaded",
			"hasLoginError",
			"theSlot",
			"passOrder",
			"chosenHandle",
			"resortCode",
		]) {
			expect(isCredentialInput(el({ name }))).toBe(false);
		}
	});

	test("exact tokens fire inside a token sequence — as one token or a separated compound run", () => {
		for (const name of [
			"moje_haslo",
			"nove-heslo",
			"senha",
			"mfa-code",
			"auth-code",
			"cc-exp",
			"customer-sort-code",
			"pin2",
		]) {
			expect(isCredentialInput(el({ name }))).toBe(true);
		}
	});

	test("bank compounds match as substrings, so decorated forms catch", () => {
		for (const name of [
			"bank-account-number",
			"bankaccountnumber",
			"aba-routing-number",
		]) {
			expect(isCredentialInput(el({ name }))).toBe(true);
		}
	});

	test("a bare account number is ordinary content, not a credential", () => {
		expect(isCredentialInput(el({ name: "accountNumber" }))).toBe(false);
		expect(isCredentialInput(el({ id: "loyalty-account-number" }))).toBe(false);
	});

	test("ordinary field names are not credentials", () => {
		for (const name of [
			"email",
			"search",
			"username",
			"address-line1",
			"promo-code",
			"zipcode",
		]) {
			expect(isCredentialInput(el({ name }))).toBe(false);
		}
	});
});

describe("maskingPosture's composed maskInputFn", () => {
	test("credentials mask to length-preserving asterisks with no rules at all", () => {
		const { maskInputFn } = maskingPosture({});
		expect(maskInputFn("hunter2!", el({ type: "password" }))).toBe("********");
		expect(maskInputFn("424242", el({ autocomplete: "cc-number" }))).toBe(
			"******",
		);
	});

	test("non-credential values pass through verbatim with no rules", () => {
		const { maskInputFn } = maskingPosture({});
		expect(maskInputFn("hello", el({ type: "text" }))).toBe("hello");
		expect(maskInputFn("a@b.co", el({ type: "email" }))).toBe("a@b.co");
		expect(maskInputFn("notes", el({ tag: "TEXTAREA" }))).toBe("notes");
	});

	test("a hidden input with a credential name or role never records its value — the server-rendered credential shape", () => {
		const { maskInputFn } = maskingPosture({});
		expect(
			maskInputFn("hunter2!", el({ type: "hidden", name: "password" })),
		).toBe("********");
		expect(
			maskInputFn(
				"424242",
				el({ type: "hidden", autocomplete: "current-password" }),
			),
		).toBe("******");
	});

	test("a non-credential hidden value survives verbatim", () => {
		const { maskInputFn } = maskingPosture({});
		expect(maskInputFn("dark", el({ type: "hidden", name: "theme" }))).toBe(
			"dark",
		);
		expect(
			maskInputFn("step-2", el({ type: "hidden", id: "wizardStep" })),
		).toBe("step-2");
	});

	test("a rule's maskInputOptions gate masks its own fields and nothing else", () => {
		const { maskInputFn } = maskingPosture({
			maskInputOptions: { email: true },
		});
		expect(maskInputFn("a@b.co", el({ type: "email" }))).toBe("******");
		expect(maskInputFn("hello", el({ type: "text" }))).toBe("hello");
	});

	test("a rule's tag-name gate (textarea) is honored like rrweb's own", () => {
		const { maskInputFn } = maskingPosture({
			maskInputOptions: { textarea: true },
		});
		expect(maskInputFn("notes", el({ tag: "TEXTAREA" }))).toBe("*****");
	});

	test("a rule's maskAllInputs masks every non-credential value", () => {
		const { maskInputFn } = maskingPosture({ maskAllInputs: true });
		expect(maskInputFn("hello", el({ type: "text" }))).toBe("*****");
		expect(maskInputFn("hunter2!", el({ type: "password" }))).toBe("********");
	});

	test("a rule's maskInputFn runs for the fields its gate matches", () => {
		const { maskInputFn } = maskingPosture({
			maskInputOptions: { email: true },
			maskInputFn: (text) => text.replace(/./g, "#"),
		});
		expect(maskInputFn("a@b.co", el({ type: "email" }))).toBe("######");
		expect(maskInputFn("hello", el({ type: "text" }))).toBe("hello");
	});

	test("password history survives a capture restart — a later posture still masks the re-typed field", () => {
		// The type history is per element, not per posture: a masking route change restarts capture
		// with a freshly composed maskingPosture, and the same DOM node a show-password toggle
		// re-typed must stay masked under it.
		const field = el({ type: "password" });
		maskingPosture({}).maskInputFn("x", field);
		field.type = "text";
		const { maskInputFn } = maskingPosture({});
		expect(maskInputFn("hunter2!", field)).toBe("********");
	});

	test("a rule's maskInputFn never reaches a credential — pass-through included", () => {
		const { maskInputFn } = maskingPosture({
			maskAllInputs: true,
			maskInputFn: (text) => text,
		});
		expect(maskInputFn("hunter2!", el({ type: "password" }))).toBe("********");
		const toggled = el({ type: "password" });
		maskInputFn("x", toggled);
		toggled.type = "text";
		expect(maskInputFn("hunter2!", toggled)).toBe("********");
	});

	test("maskInputOptions routes every input type and keeps a rule's select key", () => {
		const bare = maskingPosture({}).maskInputOptions;
		for (const key of [
			"color",
			"date",
			"datetime-local",
			"email",
			"month",
			"number",
			"range",
			"search",
			"tel",
			"text",
			"time",
			"url",
			"week",
			"textarea",
			"password",
			"hidden",
		]) {
			expect(bare[key]).toBe(true);
		}
		expect(bare.select).toBeUndefined();
		const withSelect = maskingPosture({
			maskInputOptions: { select: true },
		}).maskInputOptions;
		expect(withSelect.select).toBe(true);
	});

	test("a gated select masks through the rule path", () => {
		const { maskInputFn } = maskingPosture({
			maskInputOptions: { select: true },
		});
		expect(maskInputFn("visa", el({ tag: "SELECT", type: "select-one" }))).toBe(
			"****",
		);
	});

	test("ROUTE_ALL_INPUTS is the exported gate a per-element rule fn rides — asked on any type, select stays out", () => {
		// The role a fn judges (an autocomplete token) rides any input type, so a rule whose fn
		// decides gates every typed input with the exported list.
		expect(maskingPosture({}).maskInputOptions).toEqual({
			...ROUTE_ALL_INPUTS,
		});
		const roleFn = (text, element) =>
			element.getAttribute("autocomplete") === "email"
				? "*".repeat(text.length)
				: text;
		const { maskInputFn } = maskingPosture({
			maskInputOptions: ROUTE_ALL_INPUTS,
			maskInputFn: roleFn,
		});
		expect(
			maskInputFn("a@b.co", el({ type: "url", autocomplete: "email" })),
		).toBe("******");
		expect(maskInputFn("https://x.co", el({ type: "url" }))).toBe(
			"https://x.co",
		);
		expect(maskInputFn("visa", el({ tag: "SELECT", type: "select-one" }))).toBe(
			"visa",
		);
	});
});
