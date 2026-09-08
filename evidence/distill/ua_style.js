// The browser's default presentation of an element, computed from the user-agent stylesheet the HTML
// standard specifies (html_semantics.js, generated from the spec) plus the element's own style
// attribute — the two style origins a recorded DOM carries without its stylesheets. What this answers
// is what the projection needs: whether the element is rendered at all, whether
// it starts a new line, whether its whitespace is kept, and what its markup means to the eye
// (bold, italic, struck, monospace, quoted, sub/superscript). Page stylesheets are not applied —
// that is the projection's declared residual — so a class that hides or restyles is not seen here.
//
// The matcher covers the selector grammar the UA stylesheet actually uses, and refuses at load on
// anything else, so a spec revision that introduces new syntax fails the build instead of silently
// matching nothing. A pseudo-class the DOM cannot answer (:popover-open, :link, :visited, :host)
// makes its whole selector inert, inside :not() as much as outside: the unresolvable case falls to
// "rendered", the recoverable error direction, never to "hidden".
import { UA_RULES } from "./html_semantics.js";
import { NodeType } from "./rrweb_constants.js";

const CONSUMED = [
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
];

const UNANSWERABLE = new Set(["popover-open", "link", "visited", "host"]);
const HEADING = /^h[1-6]$/;

function parseCompound(src) {
	let s = src;
	const c = {
		type: null,
		attrs: [],
		pseudos: [],
		pseudoElement: null,
		specificity: [0, 0],
	};
	const m = s.match(/^[a-z][a-z0-9]*/);
	if (m) {
		c.type = m[0];
		c.specificity[1]++;
		s = s.slice(m[0].length);
	}
	while (s.length) {
		const attr = s.match(/^\[([a-z-]+)(?:=("?)([^\]"\s]*)\2( i)?)?\]/);
		const pseudoElement = attr ? null : s.match(/^::([a-z-]+)/);
		const functional = attr || pseudoElement ? null : s.match(/^:(not|is)\(/);
		const pseudo =
			attr || pseudoElement || functional ? null : s.match(/^:([a-z-]+)/);
		if (attr) {
			c.attrs.push({ name: attr[1], value: attr[3], ci: !!attr[4] });
			c.specificity[0]++;
			s = s.slice(attr[0].length);
		} else if (pseudoElement) {
			c.pseudoElement = pseudoElement[1];
			s = s.slice(pseudoElement[0].length);
		} else if (functional) {
			const close = matchParen(s, functional[0].length - 1);
			const inner = s.slice(functional[0].length, close);
			const list = splitTop(inner).map(parseSelector);
			c.pseudos.push({ name: functional[1], list });
			const most = Math.max(
				...list.map((sel) => sel.specificity[0] * 1000 + sel.specificity[1]),
			);
			c.specificity[0] += Math.floor(most / 1000);
			c.specificity[1] += most % 1000;
			s = s.slice(close + 1);
		} else if (pseudo) {
			c.pseudos.push({ name: pseudo[1] });
			c.specificity[0]++;
			s = s.slice(pseudo[0].length);
		} else {
			throw new Error(
				`ua_style: unsupported selector syntax at "${s}" in "${src}"`,
			);
		}
	}
	return c;
}

function matchParen(s, open) {
	let depth = 0;
	for (let i = open; i < s.length; i++) {
		if (s[i] === "(") depth++;
		else if (s[i] === ")" && --depth === 0) return i;
	}
	throw new Error(`ua_style: unbalanced parentheses in "${s}"`);
}

function splitTop(s) {
	const out = [];
	let depth = 0;
	let start = 0;
	for (let i = 0; i < s.length; i++) {
		if (s[i] === "(") depth++;
		else if (s[i] === ")") depth--;
		else if (s[i] === "," && depth === 0) {
			out.push(s.slice(start, i).trim());
			start = i + 1;
		}
	}
	out.push(s.slice(start).trim());
	return out;
}

// A selector is compounds joined by combinators, kept right-to-left for matching.
function parseSelector(src) {
	const parts = [];
	let depth = 0;
	let buf = "";
	let left = null;
	let pending = null;
	const flush = () => {
		if (buf.trim()) {
			parts.push({ compound: parseCompound(buf.trim()), combinator: left });
			left = pending;
			pending = null;
		}
		buf = "";
	};
	for (const ch of src.trim()) {
		if (ch === "[" || ch === "(") depth++;
		else if (ch === "]" || ch === ")") depth--;
		if (depth === 0 && (ch === " " || ch === ">")) {
			if (buf.trim()) {
				pending = ch;
				flush();
			} else if (ch === ">") left = ">";
			continue;
		}
		buf += ch;
	}
	flush();
	const specificity = parts.reduce(
		(acc, p) => [
			acc[0] + p.compound.specificity[0],
			acc[1] + p.compound.specificity[1],
		],
		[0, 0],
	);
	return { parts: parts.reverse(), specificity };
}

const tagOf = (node) => (node?.tagName || "").toLowerCase();

function matchesCompound(c, node) {
	if (node?.type !== NodeType.Element) return false;
	const tag = tagOf(node);
	const attrs = node.attributes || {};
	if (c.type && c.type !== tag) return false;
	for (const a of c.attrs) {
		if (!(a.name in attrs)) return false;
		if (a.value !== undefined) {
			const have = String(attrs[a.name]);
			if (
				a.ci ? have.toLowerCase() !== a.value.toLowerCase() : have !== a.value
			)
				return false;
		}
	}
	for (const p of c.pseudos) {
		if (p.name === "not") {
			if (p.list.some((sel) => matchesSelector(sel, node))) return false;
		} else if (p.name === "is") {
			if (!p.list.some((sel) => matchesSelector(sel, node))) return false;
		} else if (p.name === "heading") {
			if (!HEADING.test(tag)) return false;
		} else if (p.name === "first-of-type") {
			const siblings = (node.parentNode?.childNodes || []).filter(
				(n) => tagOf(n) === tag,
			);
			if (siblings[0] !== node) return false;
		} else if (p.name === "first-child") {
			const siblings = (node.parentNode?.childNodes || []).filter(
				(n) => n.tagName,
			);
			if (siblings[0] !== node) return false;
		} else if (UNANSWERABLE.has(p.name)) {
			return false;
		} else {
			throw new Error(`ua_style: unsupported pseudo-class :${p.name}`);
		}
	}
	return true;
}

function matchesSelector(sel, node) {
	const [subject, ...ancestors] = sel.parts;
	if (subject.compound.pseudoElement) return false;
	if (!matchesCompound(subject.compound, node)) return false;
	let combinator = subject.combinator;
	let current = node;
	for (const part of ancestors) {
		if (combinator === ">") {
			current = current.parentNode;
			if (!current || !matchesCompound(part.compound, current)) return false;
		} else {
			current = current.parentNode;
			while (current && !matchesCompound(part.compound, current))
				current = current.parentNode;
			if (!current) return false;
		}
		combinator = part.combinator;
	}
	return true;
}

const inertSelector = (sel) =>
	sel.parts.some((p) =>
		p.compound.pseudos.some(
			(ps) => UNANSWERABLE.has(ps.name) || ps.list?.some(inertSelector),
		),
	);

const RULES = UA_RULES.flatMap((rule, order) =>
	rule.selectors
		.map((src) => ({
			sel: parseSelector(src),
			order,
			declarations: rule.declarations,
		}))
		.filter((r) => !inertSelector(r.sel)),
);

const PSEUDO_CONTENT = RULES.filter(
	(r) => r.sel.parts[0].compound.pseudoElement,
);
const ELEMENT_RULES = RULES.filter(
	(r) => !r.sel.parts[0].compound.pseudoElement,
);

// Candidate rules per subject type, so an element is tested only against the rules that could name
// it: the handful whose subject has no type (the [hidden] family, :heading) plus its own tag's.
const UNTYPED = ELEMENT_RULES.filter((r) => !r.sel.parts[0].compound.type);
const BY_TAG = new Map();
for (const r of ELEMENT_RULES) {
	const type = r.sel.parts[0].compound.type;
	if (!type) continue;
	if (!BY_TAG.has(type)) BY_TAG.set(type, []);
	BY_TAG.get(type).push(r);
}
const candidates = (node) => {
	const typed = BY_TAG.get(tagOf(node));
	return typed ? [...UNTYPED, ...typed] : UNTYPED;
};

function inlineDeclarations(style) {
	if (typeof style !== "string") return [];
	return style
		.split(";")
		.map((d) => d.split(":"))
		.filter((kv) => kv.length >= 2)
		.map(([k, ...v]) => ({
			prop: k.trim().toLowerCase(),
			value: v.join(":").trim(),
		}))
		.filter((d) => CONSUMED.includes(d.prop))
		.map((d) => ({
			...d,
			important: /!important/.test(d.value),
			value: d.value.replace(/\s*!important/, ""),
		}));
}

const beats = (a, b) =>
	a.important !== b.important
		? a.important
		: a.origin !== b.origin
			? a.origin > b.origin
			: a.spec[0] !== b.spec[0]
				? a.spec[0] > b.spec[0]
				: a.spec[1] !== b.spec[1]
					? a.spec[1] > b.spec[1]
					: a.order >= b.order;

// The computed values of the consumed properties for one element: {display, visibility,
// whiteSpace, contentVisibility, textDecoration, fontWeight, fontStyle, fontFamily,
// verticalAlign, before, after} — undefined where nothing declares one. Only the element's own
// declarations; the projection walks inheritance itself where a property inherits.
export function uaStyle(node) {
	const winner = {};
	const consider = (decl, meta) => {
		const cand = { ...meta, value: decl.value, important: decl.important };
		const have = winner[decl.prop];
		if (!have || beats(cand, have)) winner[decl.prop] = cand;
	};
	for (const r of candidates(node)) {
		if (!matchesSelector(r.sel, node)) continue;
		for (const d of r.declarations)
			consider(d, { origin: 0, spec: r.sel.specificity, order: r.order });
	}
	inlineDeclarations(node.attributes?.style).forEach((d, i) => {
		consider(d, { origin: 1, spec: [1000, 0], order: i });
	});
	const out = {};
	for (const [prop, w] of Object.entries(winner)) {
		out[prop.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = w.value;
	}
	for (const r of PSEUDO_CONTENT) {
		const which = r.sel.parts[0].compound.pseudoElement;
		if (which !== "before" && which !== "after") continue;
		const probe = {
			...r.sel,
			parts: [
				{
					...r.sel.parts[0],
					compound: { ...r.sel.parts[0].compound, pseudoElement: null },
				},
				...r.sel.parts.slice(1),
			],
		};
		if (!matchesSelector(probe, node)) continue;
		const content = r.declarations.find((d) => d.prop === "content");
		if (content) out[which] = content.value;
	}
	return out;
}

export const parsedSelectorCount = RULES.length;
