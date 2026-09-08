#!/usr/bin/env bun
/**
 * Generate the projection's HTML semantics from the standards that define them — the only legitimate source for what a browser presents by default. Hand-typed lists of block elements, hidden elements, emphasis tags, or ARIA states anywhere in this repo are a defect: they are always incomplete, and the incompleteness is invisible until a page exercises the missing entry.
 *
 *   bun distill/extract_html_semantics.js    regenerate from the live specs
 *
 * Output (DO NOT EDIT, regenerate deliberately; the header stamps each source's publication date and content hash):
 *   distill/html_semantics.js
 *
 * Three sources, each read for one thing:
 *   - The WHATWG HTML rendering section's user-agent stylesheet: every `display`, `visibility`, `white-space`, `content-visibility`, `text-decoration`, `font-weight`, `font-style`, `font-family`, `vertical-align`, `background`, and `content` declaration, with its selectors, source order, and `!important`. What is laid out on its own line, what is not rendered at all, what keeps its whitespace, and what inline markup means to the eye all come from here. Rules under `@media (scripting)` are kept — the recorded pages ran scripts.
 *   - The WAI-ARIA specification's list of states — the aria-* attributes whose values change with interaction, as distinct from properties.
 *   - The HTML-AAM attribute mappings: the native HTML attributes whose unconditional meaning is one of those ARIA states (checked, disabled, selected, open on details, ...), so native and ARIA state read as one vocabulary.
 */
import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import postcss from "postcss";

const SOURCES = {
	rendering: "https://html.spec.whatwg.org/multipage/rendering.html",
	aria: "https://w3c.github.io/aria/",
	htmlAam: "https://w3c.github.io/html-aam/",
};

const CONSUMED = new Set([
	"display",
	"visibility",
	"white-space",
	"content-visibility",
	"text-decoration",
	"font-weight",
	"font-style",
	"font-family",
	"vertical-align",
	"background",
	"content",
]);

const untag = (s) =>
	s
		.replace(/<[^>]+>/g, "")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#39;|&apos;/g, "'")
		.replace(/&nbsp;/g, " ")
		.replace(/&amp;/g, "&");

async function fetchText(url) {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
	return res.text();
}

function uaRules(page) {
	const blocks = [
		...page.matchAll(/<pre><code class='css'>(.*?)<\/code><\/pre>/gs),
	].map((m) => untag(m[1]));
	if (blocks.length === 0)
		throw new Error("rendering section: no CSS blocks found");
	const css = blocks.join("\n");
	const root = postcss.parse(css);
	const rules = [];
	root.walkRules((rule) => {
		const media = rule.parent?.type === "atrule" ? rule.parent.name : null;
		if (
			media &&
			!(rule.parent.name === "media" && rule.parent.params === "(scripting)")
		) {
			throw new Error(
				`unexpected at-rule context: @${media} ${rule.parent.params}`,
			);
		}
		const declarations = [];
		rule.walkDecls((decl) => {
			if (CONSUMED.has(decl.prop)) {
				declarations.push({
					prop: decl.prop,
					value: decl.value,
					important: !!decl.important,
				});
			}
		});
		if (declarations.length)
			rules.push({ selectors: rule.selectors, declarations });
	});
	const updated = page.match(/Last Updated <span class=pubdate>([^<]+)/)?.[1];
	if (!updated) throw new Error("rendering section: no Last Updated stamp");
	return { rules, updated, hash: sha(css) };
}

function ariaStates(page) {
	const states = [
		...page.matchAll(/<section class="state notoc" id="(aria-[a-z]+)"/g),
	].map((m) => m[1]);
	if (states.length === 0) throw new Error("aria: no state sections found");
	const updated = page.match(
		/<time class="dt-published" datetime="([^"]+)"/,
	)?.[1];
	if (!updated) throw new Error("aria: no publication date");
	return { states, updated, hash: sha(page) };
}

function nativeStates(page, states) {
	const sections = [
		...page.matchAll(
			/<h4 id="(att-[a-z-]+)"[^>]*>(.*?)<\/h4>.*?<table class="data"[^>]*>(.*?)<\/table>/gs,
		),
	];
	if (sections.length === 0)
		throw new Error("html-aam: no attribute sections found");
	const out = [];
	for (const [, , heading, table] of sections) {
		const attribute = untag(heading)
			.replace(/^[\d.]+\s*/, "")
			.replace(/\s*\(.*$/, "")
			.trim();
		const cells = {};
		for (const [, th, td] of table.matchAll(
			/<tr>\s*<th>(.*?)<\/th>\s*<td>(.*?)<\/td>/gs,
		)) {
			cells[untag(th).replace(/\s+/g, " ").trim()] = untag(td)
				.replace(/\s+/g, " ")
				.trim();
		}
		const elements = cells["Element(s)"] ?? "";
		const mapping =
			Object.entries(cells).find(([k]) => /^\[WAI-ARIA/.test(k))?.[1] ?? "";
		const unconditional = mapping.match(
			/^(aria-[a-z]+)="([a-z]+)(?: \| [a-z]+)?"$/,
		);
		if (!unconditional) continue;
		const [, state, value] = unconditional;
		if (!states.includes(state)) continue;
		if (/\(if absent\)/.test(untag(heading))) continue;
		out.push({ attribute, elements, state, value });
	}
	if (out.length === 0)
		throw new Error("html-aam: no unconditional state mappings found");
	const updated = page.match(
		/<time class="dt-published" datetime="([^"]+)"/,
	)?.[1];
	if (!updated) throw new Error("html-aam: no publication date");
	return { native: out, updated, hash: sha(page) };
}

const sha = (s) => createHash("sha256").update(s).digest("hex").slice(0, 16);

const [rendering, aria, htmlAam] = await Promise.all(
	Object.values(SOURCES).map(fetchText),
);
const ua = uaRules(rendering);
const states = ariaStates(aria);
const native = nativeStates(htmlAam, states.states);

const header = `// GENERATED by distill/extract_html_semantics.js — DO NOT EDIT. Regenerate deliberately.
//
// Sources, each stamped with its publication date and a hash of what was read:
//   ${SOURCES.rendering}  (${ua.updated}, ${ua.hash})
//   ${SOURCES.aria}  (${states.updated}, ${states.hash})
//   ${SOURCES.htmlAam}  (${native.updated}, ${native.hash})
//
// UA_RULES: the user-agent stylesheet's rules carrying a consumed property, in source order, each
// with its selector list and declarations ({prop, value, important}). ARIA_STATES: the aria-*
// attributes the ARIA spec classes as states. NATIVE_STATES: HTML attributes whose unconditional
// HTML-AAM mapping is one of those states ({attribute, elements, state, value}).
`;
const body = `${header}
export const UA_RULES = ${JSON.stringify(ua.rules, null, "\t")};

export const ARIA_STATES = ${JSON.stringify(states.states, null, "\t")};

export const NATIVE_STATES = ${JSON.stringify(native.native, null, "\t")};
`;
const out = new URL("./html_semantics.js", import.meta.url);
writeFileSync(out, body);
console.log(
	`wrote ${out.pathname}: ${ua.rules.length} UA rules, ${states.states.length} ARIA states, ${native.native.length} native state attributes`,
);
