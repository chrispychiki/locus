#!/usr/bin/env bun
/**
 * Generate rrweb schema constants from the installed @rrweb/types package — the ONLY legitimate source for numeric↔name mappings. Hand-typed rrweb constants anywhere in this repo are a defect: the nested details are always misremembered. SVGTagMap (lowercase→camelCase SVG tag-case restoration) is not in @rrweb/types, so it is parsed out of the installed rrweb dist instead — still derived from rrweb's own source, never hand-typed — and emitted into the JS outputs only (Python never serializes SVG).
 *
 *   bun distill/extract_rrweb_constants.js            regenerate
 *   bun distill/extract_rrweb_constants.js --check    verify the checked-in
 *       outputs are byte-identical to what the generator would emit (exit 1
 *       on stale or hand-edited files) — run by the test suite, so the
 *       canonical files are provably derived, not just labeled DO NOT EDIT
 *
 * Outputs (DO NOT EDIT, regenerate after any rrweb upgrade):
 *   distill/rrweb_constants.js
 *   src/locus/evidence/rrweb_constants.py
 *   ../recorder/src/rrweb_constants.js
 *
 * Locus custom event types come from the recorder, not rrweb, and are appended here as the single declaration site: 69 PageLoad (arrival context), 70 PageVisible / 71 PageHidden (visibility transitions — "the visitor left" is hidden-then-silence, inferred downstream, never asserted by the recorder).
 */
import { readFileSync, writeFileSync } from "node:fs";
import {
	EventType,
	IncrementalSource,
	MediaInteractions,
	MouseInteractions,
	NodeType,
	PointerTypes,
} from "@rrweb/types";
import { version } from "@rrweb/types/package.json";

const CUSTOM_EVENT_TYPES = { PageLoad: 69, PageVisible: 70, PageHidden: 71 };

const numeric = (enumObject) =>
	Object.entries(enumObject).filter(([, value]) => typeof value === "number");

const enums = {
	EventType: [...numeric(EventType), ...Object.entries(CUSTOM_EVENT_TYPES)],
	IncrementalSource: numeric(IncrementalSource),
	MediaInteractions: numeric(MediaInteractions),
	MouseInteractions: numeric(MouseInteractions),
	PointerTypes: numeric(PointerTypes),
	NodeType: numeric(NodeType),
};

const header = (
	comment,
) => `${comment} Auto-generated from @rrweb/types@${version} by distill/extract_rrweb_constants.js
${comment} DO NOT EDIT — regenerate with: bun distill/extract_rrweb_constants.js
`;

const js =
	header("//") +
	Object.entries(enums)
		.map(
			([name, entries]) =>
				`\nexport const ${name} = {\n` +
				entries.map(([key, value]) => `  ${key}: ${value},`).join("\n") +
				`\n};\n\nexport const ${name}Names = {\n` +
				entries.map(([key, value]) => `  ${value}: "${key}",`).join("\n") +
				"\n};\n",
		)
		.join("");

const py =
	header("#") +
	Object.entries(enums)
		.map(
			([name, entries]) =>
				`\n\nclass ${name}:\n` +
				entries.map(([key, value]) => `    ${key} = ${value}`).join("\n") +
				`\n\n\n${name.toUpperCase()}_NAMES = {\n` +
				entries.map(([key, value]) => `    ${value}: "${key}",`).join("\n") +
				"\n}",
		)
		.join("") +
	"\n";

// SVGTagMap lives in rrweb's dist, not @rrweb/types, and isn't exported — parse it out of the
// installed rrweb (a declared dependency of this package, pinned alongside @rrweb/types). A flat
// object of `identifier: "string"` entries, so a literal-bounded regex is sufficient and stays derived.
const svgTagMap = (() => {
	const src = readFileSync(new URL(import.meta.resolve("rrweb")), "utf8");
	const at = src.indexOf("const SVGTagMap = {");
	if (at === -1)
		throw new Error(
			"SVGTagMap not found in the installed rrweb dist — extend this generator",
		);
	const block = src.slice(at, src.indexOf("};", at));
	const map = {};
	for (const m of block.matchAll(/(\w+):\s*"([^"]+)"/g)) map[m[1]] = m[2];
	if (!Object.keys(map).length)
		throw new Error("SVGTagMap parsed empty from rrweb dist");
	return map;
})();
const svgMapJs =
	"\nexport const SVGTagMap = {\n" +
	Object.entries(svgTagMap)
		.map(([k, v]) => `  ${k}: "${v}",`)
		.join("\n") +
	"\n};\n";

const outputs = [
	[new URL("rrweb_constants.js", import.meta.url), js + svgMapJs],
	[new URL("../src/locus/evidence/rrweb_constants.py", import.meta.url), py],
	[
		new URL("../../recorder/src/rrweb_constants.js", import.meta.url),
		js + svgMapJs,
	],
];

if (process.argv.includes("--check")) {
	const stale = outputs.filter(([url, content]) => {
		try {
			return readFileSync(url, "utf8") !== content;
		} catch {
			return true;
		}
	});
	for (const [url] of stale) {
		console.error(
			`${url.pathname} differs from what @rrweb/types@${version} ` +
				"generates — stale or hand-edited; rerun the generator",
		);
	}
	process.exit(stale.length ? 1 : 0);
}

for (const [url, content] of outputs) writeFileSync(url, content);
console.log(
	`generated constants from @rrweb/types@${version}: ` +
		Object.entries(enums)
			.map(([n, e]) => `${n}=${e.length}`)
			.join(" ") +
		` SVGTagMap=${Object.keys(svgTagMap).length}`,
);
