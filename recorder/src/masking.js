/**
 * Per-URL rrweb options — the pattern grammar, and what a pattern matches.
 *
 * A rule is `{pattern, options}`. A pattern is `*` (every page), a host, or a `host/path-prefix`,
 * matched scheme-, www-, and port-insensitively, with every query dropped from both sides — the
 * fragment's own included — and the fragment kept, so a hash-routed SPA's `/#/checkout` is
 * matchable. The host matches
 * case-insensitively, as DNS is; the path matches as written, as URL paths are. A host matches
 * that domain and any subdomain of it at a label boundary — `example.com` matches `example.com`
 * and `app.example.com`, never `myexample.com` or `example.com.evil.com` — and a path-prefix
 * matches at a path-segment boundary (`example.com/checkouts` matches `/checkouts` and
 * `/checkouts/pay`, never `/checkoutsX`). Matching rules layer in order, each merging over the
 * last and deep-merging maskInputOptions, so a deployment lists host-wide rules first and layers
 * page-specific ones after without dropping the broader masking. No match returns `{}`, leaving
 * the recorder's base posture (the credential mask below) in force.
 *
 * rrweb fixes options once at record start — record() captures them into the observer closures it
 * installs (rrweb packages/rrweb/src/record/index.ts)
 * — so a single record() lifetime carries one masking posture; per-route masking is start()'s.
 */
import { stripQueries } from "./address.js";

function splitHostPath(urlOrPattern) {
	const noScheme = urlOrPattern.replace(/^[a-z][a-z0-9+.-]*:\/\//i, "");
	const cut = noScheme.search(/[/?#]/);
	const host = (cut === -1 ? noScheme : noScheme.slice(0, cut))
		.replace(/^www\./i, "")
		.replace(/:\d+$/, "")
		.toLowerCase();
	const afterHost = cut === -1 ? "" : noScheme.slice(cut);
	// The query is not route identity — a `?ref=` must never carry a match — so every query is
	// dropped, a hash-router's own (`#/checkout?ref=…`) included; address.js states the one rule
	// (stripQueries), shared with the route-change detector, so resolution and detection see the
	// same identity. The fragment itself is kept: a hash-routed SPA's route lives there.
	const path = stripQueries(afterHost) || "/";
	return { host, path };
}

function hostMatches(patternHost, host) {
	if (patternHost === "*") return true;
	return host === patternHost || host.endsWith(`.${patternHost}`);
}

function pathMatches(patternPath, path) {
	if (patternPath === "/") return true;
	const prefix = patternPath.replace(/\/+$/, "");
	// "#" is a segment boundary too, so /checkouts matches /checkouts#payment: a fragment
	// anchor never lets a matched page escape its masking.
	return (
		path === prefix ||
		path.startsWith(`${prefix}/`) ||
		path.startsWith(`${prefix}#`)
	);
}

function mergeOptions(base, overlay) {
	const merged = { ...base, ...overlay };
	if (base.maskInputOptions || overlay.maskInputOptions) {
		merged.maskInputOptions = {
			...base.maskInputOptions,
			...overlay.maskInputOptions,
		};
	}
	return merged;
}

// The recorder's own mechanical mask. Credential content — secrets and payment-instrument
// fields — is invariant across every deployment and every question ever asked of a recording:
// zero analytical value, pure liability. So it is masked below the rules layer, decided by what
// the element is, never by per-site judgment; everything non-credential stays the deployment's
// call through the rules.
//
// No single fact about an element identifies a credential field through its whole life, so the
// verdict is any-of four arms, each covering what the others cannot see:
//
// - Live `type=password` — definitional when present, and gone the moment a show-password
//   toggle flips the field to `text` with the secret still in it.
// - Type history — an element once seen as a password is a credential for its lifetime. The set
//   is a fact about the element, so it is module state, surviving the capture restarts a route
//   change drives. But the memory is keyed to the element instance, and a toggle that
//   re-renders the field mints a fresh `type=text` node the history has never seen — with the
//   secret riding in as its serialized `value` attribute.
// - `autocomplete` role tokens — markup, so a re-render regenerates them with the node. Honored
//   whatever the type; the attribute is a space-separated token list ("shipping cc-number"), so
//   membership is per token.
// - `name`/`id` credential tokens — the same replacement-surviving property as autocomplete,
//   and the arm that holds when a site sets `autocomplete="off"` on its login field. A name is
//   a convention, not a declaration, so this arm is
//   probabilistic — the token lists below are tuned so a false positive stars one benign field
//   (cheap) while a miss records a credential (never cheap). Cheap is still a cost, and the
//   mask sits below every rule, so a false positive here is one no deployment can undo — which
//   is what splits the lists: each token matches by its own collision risk, never wider.
//
// Residue: a re-rendered field whose markup carries no credential token at all — no
// autocomplete role, no recognizable name or id — records verbatim. No stateless read can catch
// it; a deployment expecting that shape masks the page's inputs by rule.
const CREDENTIAL_AUTOCOMPLETE = new Set([
	"current-password",
	"new-password",
	"one-time-code",
	"cc-number",
	"cc-csc",
	"cc-exp",
	"cc-exp-month",
	"cc-exp-year",
]);
const everPassword = new WeakSet();

// Matched as substrings of the whole collapsed name/id — the only test that catches an
// unseparated compound ("newpassword", "userPassword2") — so a token earns this list solely by
// being collision-proof under collapse: long enough, or written in a script whose characters
// never appear inside a collapsed Latin name. Collapse erases camelCase boundaries, so a short
// Latin word here would fire inside ordinary names ("hasLoaded" collapses onto "haslo",
// "chosenHandle" onto "senha") — those tokens live in the exact set instead.
const CREDENTIAL_NAME_SUBSTRINGS = [
	"password",
	"passcode",
	"passphrase",
	"passwort",
	"motdepasse",
	"contrasena",
	"contraseña",
	"hasło",
	"wachtwoord",
	"salasana",
	"lösenord",
	"şifre",
	"jelszo",
	"jelszó",
	"lozinka",
	"matkhau",
	"mậtkhẩu",
	"пароль",
	"パスワード",
	"密码",
	"密碼",
	"암호",
	"비밀번호",
	"cardnumber",
	"ccnumber",
	"ccnum",
	"cardnum",
	"creditcard",
	"securitycode",
	"verificationcode",
	"onetimecode",
	"onetimepassword",
	"bankaccount",
	"routingnumber",
];

// Matched only against whole token runs (below): tokens a substring test would fire inside
// ordinary names — `pass` inside "passenger", `passord` inside "passOrder", `sortcode` inside
// "resortCode" — but that carry credential meaning wherever they stand as a token.
const CREDENTIAL_NAME_EXACT = new Set([
	"passwd",
	"pwd",
	"pw",
	"pword",
	"pwrd",
	"userpw",
	"otp",
	"totp",
	"mfacode",
	"2facode",
	"authcode",
	"cvv",
	"cvv2",
	"cvc",
	"csc",
	"cvn",
	"ccexp",
	"iban",
	"swiftcode",
	"biccode",
	"sortcode",
	"haslo",
	"heslo",
	"senha",
	"sifre",
	"passord",
	"losenord",
]);

// Matched only against the entire collapsed name: ordinary English words that mean credential
// solely when they are the whole of what a field is called — as a mere token they ride inside
// benign names (`pass` in "passOrder", `pin` in India's postal "pin-code", `otc` in
// "otc-medications", `swift` in "swiftDelivery", `bic` in "bicColor"). The bank compounds
// (`swiftcode`, `biccode`) stay in the exact set, so "swift-code" and "bicCode" still mask.
const CREDENTIAL_NAME_WHOLE = new Set(["pass", "pin", "otc", "swift", "bic"]);

// Compose to NFC first — a decomposed "hasło" or "mật khẩu" carries its diacritics as combining
// marks, which are \p{M}, not \p{L}, and stripping them would compare a different word than the
// precomposed token lists hold — then lowercase and strip everything that is not a letter or
// digit, so `current-password`, `user_pw`, and `Password 2` compare as `currentpassword`,
// `userpw`, `password2`.
function normalizeFieldName(raw) {
	return String(raw)
		.normalize("NFC")
		.toLowerCase()
		.replace(/[^\p{L}\p{N}]+/gu, "");
}

// A field name is a token sequence: separators and camelCase humps are boundaries, digits stay
// attached to their token ("2facode" is one token, not a split). NFC first, as normalizeFieldName
// composes, so a decomposed name yields the same tokens the exact sets hold.
function fieldNameTokens(raw) {
	return String(raw)
		.normalize("NFC")
		.replace(/(\p{Ll}|\p{Nd})(\p{Lu})/gu, "$1 $2")
		.replace(/(\p{Lu})(\p{Lu}\p{Ll})/gu, "$1 $2")
		.toLowerCase()
		.split(/[^\p{L}\p{N}]+/u)
		.filter(Boolean);
}

// The exact set is tested against every contiguous token run's concatenation — a single token
// ("moje_haslo" → haslo), a separated compound ("mfa-code", "customer-sort-code" → mfacode,
// sortcode), and the whole name as the full run — with trailing digits stripped per run
// ("otp2" → otp). A token match never crosses a boundary, so "hasLocation" (has|location) and
// "resortCode" (resort|code) stay clear where a substring test would fire.
function exactTokenRunMatches(tokens) {
	for (let i = 0; i < tokens.length; i++) {
		let run = "";
		for (let j = i; j < tokens.length; j++) {
			run += tokens[j];
			if (CREDENTIAL_NAME_EXACT.has(run)) return true;
			const stripped = run.replace(/\p{Nd}+$/u, "");
			if (stripped !== run && CREDENTIAL_NAME_EXACT.has(stripped)) return true;
		}
	}
	return false;
}

function nameDeclaresCredential(element) {
	for (const attr of ["name", "id"]) {
		const raw = element.getAttribute?.(attr);
		if (!raw) continue;
		const collapsed = normalizeFieldName(raw);
		if (!collapsed) continue;
		if (CREDENTIAL_NAME_SUBSTRINGS.some((t) => collapsed.includes(t)))
			return true;
		if (CREDENTIAL_NAME_WHOLE.has(collapsed.replace(/\p{Nd}+$/u, "")))
			return true;
		if (exactTokenRunMatches(fieldNameTokens(raw))) return true;
	}
	return false;
}

// rrweb's own type read (its getInputType): the data-rr-is-password escape hatch wins, else the
// live `type`, lowercased — `element.type` so an attribute-less input reads "text".
function inputType(element) {
	if (element.hasAttribute?.("data-rr-is-password")) return "password";
	return element.type ? String(element.type).toLowerCase() : null;
}

export function isCredentialInput(element) {
	if (!element) return false;
	if (inputType(element) === "password") {
		everPassword.add(element);
		return true;
	}
	if (everPassword.has(element)) return true;
	const autocomplete = element.getAttribute?.("autocomplete");
	if (
		autocomplete
			?.toLowerCase()
			.split(/\s+/)
			.some((token) => CREDENTIAL_AUTOCOMPLETE.has(token))
	) {
		return true;
	}
	return nameDeclaresCredential(element);
}

/**
 * Every input type rrweb's maskAllInputs enables, minus `select`, plus `hidden` — the gate that
 * routes every value-bearing input to a maskInputFn. rrweb only calls a maskInputFn on values its
 * maskInputOptions gate matches, so a fn that judges per element — an autocomplete role rides any
 * input type — is only as wide as its gate: a rule whose fn decides sets this as its
 * maskInputOptions, and the composition below uses it the same way to see every value. It is a
 * routing gate, not a mask-all switch: gating everything with no fn is rrweb's "asterisk
 * everything".
 *
 * `hidden` is in: rrweb serializes a hidden input's value like any visible field's but leaves the
 * type out of its own maskAllInputs set, so a credential a server renders into a hidden field —
 * a `name="password"` echo, an `autocomplete` role — would record verbatim with no fn ever asked.
 * Routing it costs nothing: nobody types into a hidden field, and a non-credential hidden value
 * passes the composed fn untouched unless a rule gates it.
 *
 * `select` stays out: rrweb's select masking also deletes the `selected` attribute from option
 * nodes, and rebuild shows a select's state only through that attribute (its serialized value
 * lands as a dead `value` attribute), so enabling it site-wide trades every select's replay
 * state for dropdowns whose only credential use is a card expiry — not a payment instrument
 * without the number and csc, which are typed fields and covered. A select credential is a
 * rule's to mask.
 */
export const ROUTE_ALL_INPUTS = Object.freeze({
	color: true,
	date: true,
	"datetime-local": true,
	email: true,
	month: true,
	number: true,
	range: true,
	search: true,
	tel: true,
	text: true,
	time: true,
	url: true,
	week: true,
	textarea: true,
	password: true,
	hidden: true,
});

/**
 * The masking options handed to rrweb.record(): every input value routes through the returned
 * maskInputFn, which masks credentials unconditionally and otherwise applies exactly the resolved
 * rule's own intent — its maskInputOptions gate (replicated from rrweb's: tag name, then type),
 * its maskAllInputs, its maskInputFn where it supplied one. rrweb calls a supplied maskInputFn
 * INSTEAD of asterisking, so handing a rule's fn straight to record() would let it take password
 * masking away; composing it under the credential check is what keeps the mask unconditional.
 */
export function maskingPosture(resolved) {
	const ruleGate = { ...resolved.maskInputOptions };
	const ruleMasksAll = resolved.maskAllInputs === true;
	const ruleFn = resolved.maskInputFn;
	return {
		maskInputOptions: { ...ruleGate, ...ROUTE_ALL_INPUTS },
		maskInputFn: (text, element) => {
			if (isCredentialInput(element)) return "*".repeat(text.length);
			const tagName = element?.tagName?.toLowerCase();
			const type = element ? inputType(element) : null;
			if (ruleMasksAll || ruleGate[tagName] || (type && ruleGate[type])) {
				return ruleFn ? ruleFn(text, element) : "*".repeat(text.length);
			}
			return text;
		},
	};
}

export function optionsForUrl(rules, url) {
	const { host, path } = splitHostPath(url);
	let resolved = {};
	for (const { pattern, options } of rules) {
		const p = splitHostPath(pattern);
		if (hostMatches(p.host, host) && pathMatches(p.path, path)) {
			resolved = mergeOptions(resolved, options);
		}
	}
	return resolved;
}
