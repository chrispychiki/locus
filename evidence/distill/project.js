// The projection is the page as a person at it would take it in, as markdown: what was readable, how it was grouped, what could be acted on and what it was called, what state its parts declared, and what its markup meant to the eye — and nothing else. Testimony, never summary: what was there, not what mattered. Each of those comes from the DOM's own statement of it, read from the standards that define the statement (html_semantics.js, generated from the specs; ua_style.js applies the user-agent stylesheet):
//   - Rendered at all: the user-agent stylesheet's display:none and visibility rules, the style attribute, and a closed details' body. What a page stylesheet hides — a class, a media query, off-screen placement — is the declared residual: such content reads as if presented, and the recording's screenshots settle what was on screen.
//   - Grouping: a block boundary wherever the user-agent stylesheet lays an element out on its own line, a list item where it is a list item, a row and cells where it is a table row. The grouping survives inside every construct — a link, a heading, a cell, a list item, a button, an emphasis — as lines under it.
//   - Affordances and their names: links with target, buttons, form controls with kind, current value, and checked state, images with their alt; an element that shows no text of its own is named by its accessible name in the accname order (aria-labelledby, aria-label, the native label, title), printed as label="…" so it never reads as visible text. An element declaring a widget role carries it.
//   - State: the ARIA state attributes, and the native attributes HTML-AAM maps onto them (checked, disabled, selected, a details' open), printed as {…} after the element's first line. aria-hidden is left out: it declares exposure to assistive technology, not presentation.
//   - Inline meaning: the user-agent stylesheet's bold, italic, struck, monospace, highlighted, and quoted text as markdown marks, on inline elements; preserved whitespace (pre) as a fenced block.
// Whitespace follows the browser's normal-flow rule: runs of whitespace collapse to one space unless the element keeps it, so line breaks come only from block boundaries — source indentation between sibling text nodes never asserts a break the visitor didn't see. Every string carried from the page passes one of the shared ceilings by its role (text.js). URLs are affordance metadata, capped by the shared rule. A shadow host renders its shadow tree in place of its light children; <slot> re-projection of light children is residual. Sub- and superscript, and list-style variants, are residual: markdown has no mark for them.
//
// Reads the tree the mirror reconstructs (LightweightMirror), walking node objects: type for kind (the
// generated rrweb NodeType, never a numeric literal), childNodes/parentNode as object references, tagName
// lowercased here, attributes a plain object, live input state on props, character-data text via textContent.
import { roles } from "aria-query";
import { ARIA_STATES, NATIVE_STATES } from "./html_semantics.js";
import { NodeType } from "./rrweb_constants.js";
import {
	collapse,
	cutNote,
	describeUrl,
	escapeValue,
	LABEL_CAP,
	TEXT_CAP,
	VALUE_CAP,
} from "./text.js";
import { uaStyle } from "./ua_style.js";

const shown = (s, cap) => cutNote(s, cap).join("");
const quoted = (s, cap) => {
	const [kept, note] = cutNote(s, cap);
	return `"${kept}"${note}`;
};
// A field's value: cut at the value ceiling on the raw text — the same cut the Input event's
// value takes at distillation, so the page and the stream state one value identically — then
// written with its line structure escaped (escapeValue).
const fieldValue = (value) => {
	const [kept, note] = cutNote(String(value), VALUE_CAP);
	return `"${escapeValue(kept)}"${note}`;
};

// svg renders (its text — chart labels, in-svg copy — is presented content), but its tooltip-only
// (desc) and definition (defs) children do not; its title is the svg's accessible name, read by
// accessibleName and never as text.
const SVG_MACHINERY = new Set(["defs", "desc", "title"]);

const HEADING = /^h([1-6])$/;
const WIDGET_ROLES = new Set(
	[...roles.entries()]
		.filter(
			([, def]) =>
				!def.abstract &&
				def.superClass.some((chain) => chain.includes("widget")),
		)
		.map(([name]) => name),
);
const STATE_ATTRS = new Set(ARIA_STATES.filter((s) => s !== "aria-hidden"));
const NATIVE_STATE_ATTRS = new Set(NATIVE_STATES.map((n) => n.attribute));
const BLOCK_DISPLAYS =
	/^(block|list-item|table|table-.*|flex|grid|flow-root|inline-block)$/;

const tagOf = (node) => (node?.tagName || "").toLowerCase();

// Indentation under a construct is carried as a marker while rendering, so the browser's rule that
// whitespace at the start of a line collapses away can be applied to source whitespace without
// eating the indentation; the marker becomes two spaces at the end.
const INDENT = "\u0001";

// Rendered, by what the DOM itself states: the user-agent stylesheet (with the style attribute), and a
// closed details' body, which the HTML standard describes as not rendered rather than styling away.
function unrendered(node, style) {
	if (style.display === "none") return true;
	if (style.visibility === "hidden" || style.visibility === "collapse")
		return true;
	if (style.contentVisibility === "hidden") return true;
	const parent = node.parentNode;
	if (tagOf(parent) === "details" && !("open" in (parent.attributes || {}))) {
		const summaries = (parent.childNodes || []).filter(
			(c) => tagOf(c) === "summary",
		);
		return summaries[0] !== node;
	}
	return false;
}

// The element's declared state, as the DOM states it: ARIA states present as attributes, and the native
// attributes HTML-AAM maps onto ARIA states, scoped to the elements the mapping names.
function stateMarks(node) {
	const attrs = node.attributes || {};
	const tag = tagOf(node);
	const marks = [];
	const role = attrs.role?.split(/\s+/)[0];
	if (role && WIDGET_ROLES.has(role)) marks.push(`role=${role}`);
	for (const [name, value] of Object.entries(attrs)) {
		if (STATE_ATTRS.has(name)) marks.push(`${name}=${String(value)}`);
	}
	for (const attribute of NATIVE_STATE_ATTRS) {
		if (!(attribute in attrs) || attribute === "checked") continue;
		const scoped = NATIVE_STATES.some(
			(n) =>
				n.attribute === attribute &&
				n.elements.split(/[;,]\s*/).some((e) => e.split(/\s+/)[0] === tag),
		);
		if (scoped) marks.push(attribute);
	}
	return marks.length ? `{${marks.join(" ")}}` : "";
}

const withMarks = (rendered, marks) => {
	if (!marks) return rendered;
	const lines = rendered.split("\n");
	const i = lines.findIndex((l) => l.trim());
	if (i < 0) return `${marks}`;
	lines[i] = `${lines[i]} ${marks}`;
	return lines.join("\n");
};

const textOf = (node) => {
	if (!node) return "";
	if (node.type === NodeType.Text) return node.textContent || "";
	return ((node.shadowRoot || node).childNodes || []).map(textOf).join("");
};

// The accessible name in the accname order, for an element whose rendering shows no text of its
// own: aria-labelledby, aria-label, the native label (an input's value for button types, an image's
// alt, an svg's title), title, then an input's placeholder. Content is what the rendering already
// shows, so it is never the name here; a label element renders as text in place, so it is not
// repeated as a name either. One derivation for every surface that names an element — the
// projection's brackets and links, and the event line's target — so the same control carries the
// same name wherever it appears. The id index behind aria-labelledby is built once per resolver,
// on first need.
export function nameResolver(mirror) {
	let idIndex = null;
	const byHtmlId = (id) => {
		if (!idIndex) {
			idIndex = new Map();
			const walk = (n) => {
				if (!n) return;
				if (n.type === NodeType.Element && n.attributes?.id !== undefined) {
					idIndex.set(String(n.attributes.id), n);
				}
				for (const c of (n.shadowRoot || n).childNodes || []) walk(c);
				if (n.contentDocument) walk(n.contentDocument);
			};
			walk(mirror.rootDoc);
		}
		return idIndex.get(id);
	};

	const accessibleName = (node) => {
		const attrs = node.attributes || {};
		const tag = tagOf(node);
		if (typeof attrs["aria-labelledby"] === "string") {
			const text = attrs["aria-labelledby"]
				.split(/\s+/)
				.map((id) => collapse(textOf(byHtmlId(id))))
				.filter(Boolean)
				.join(" ");
			if (text) return text;
		}
		if (typeof attrs["aria-label"] === "string" && attrs["aria-label"].trim()) {
			return collapse(attrs["aria-label"]);
		}
		if (tag === "input" && /^(button|submit|reset)$/i.test(attrs.type || "")) {
			if (attrs.value) return collapse(String(attrs.value));
		}
		if (
			(tag === "input" && /^image$/i.test(attrs.type || "")) ||
			tag === "img"
		) {
			if (attrs.alt) return collapse(attrs.alt);
		}
		if (tag === "svg") {
			const title = (node.childNodes || []).find((c) => tagOf(c) === "title");
			if (title) return collapse(textOf(title));
		}
		for (const c of node.childNodes || []) {
			if (tagOf(c) === "svg") {
				const inner = accessibleName(c);
				if (inner) return inner;
			}
		}
		if (typeof attrs.title === "string" && attrs.title.trim())
			return collapse(attrs.title);
		if (
			(tag === "input" || tag === "textarea") &&
			typeof attrs.placeholder === "string" &&
			attrs.placeholder.trim()
		)
			return collapse(attrs.placeholder);
		return "";
	};
	return accessibleName;
}

export function projectMarkdown(mirror, rootId) {
	const accessibleName = nameResolver(mirror);

	// A field's placeholder is printed on its own line as the text the field showed, so a name that
	// is only the placeholder would state it twice.
	const label = (node) => {
		const name = accessibleName(node);
		if (!name || name === collapse(node.attributes?.placeholder || ""))
			return "";
		return ` label=${quoted(name, LABEL_CAP)}`;
	};

	// Lines under a construct: the first line stays in place, the rest indent beneath it.
	const split = (inner) => {
		const lines = inner.split("\n").filter((l) => l.trim());
		const head = (lines[0] || "").trim();
		const rest = lines.slice(1).map((l) => l.trim());
		return [head, rest];
	};
	const nest = (head, rest) =>
		rest.length
			? `${head}\n${rest.map((l) => `${INDENT}${l}`).join("\n")}`
			: head;

	const render = (node, ctx) => {
		if (!node) return "";
		if (node.type === NodeType.Text) {
			const raw = node.textContent || "";
			if (ctx.pre) return shown(raw, TEXT_CAP);
			return shown(raw.replace(/\s+/g, " "), TEXT_CAP);
		}
		if (node.type === NodeType.Document) {
			return (node.childNodes || []).map((c) => render(c, ctx)).join("");
		}
		if (node.type !== NodeType.Element) return "";

		const tag = tagOf(node);
		if (SVG_MACHINERY.has(tag)) return "";
		const attrs = node.attributes || {};
		const style = uaStyle(node);
		if (unrendered(node, style)) return "";
		const marks = stateMarks(node);
		const pre = ctx.pre || /^pre/.test(style.whiteSpace || "");
		const kids = () => {
			const cn = (node.shadowRoot || node).childNodes || [];
			return cn.map((c) => render(c, { pre })).join("");
		};
		const block = (text) => {
			const inner = text.replace(/^\n+|\n+$/g, "");
			return inner.trim()
				? `\n${withMarks(inner, marks)}\n`
				: marks
					? `\n${marks}\n`
					: "";
		};

		if (tag === "iframe") {
			return node.contentDocument ? render(node.contentDocument, ctx) : "";
		}
		const heading = tag.match(HEADING);
		if (heading) {
			const [head, rest] = split(kids());
			return block(nest(`${"#".repeat(Number(heading[1]))} ${head}`, rest));
		}
		if (tag === "a") {
			const [head, rest] = split(kids());
			const link = (text) =>
				attrs.href ? `[${text}](${describeUrl(attrs.href)})` : text;
			const named = head || rest.length ? head : label(node).trim();
			const out = nest(link(named), rest.map(link));
			return rest.length ? block(out) : withMarks(out, marks);
		}
		if (tag === "button") {
			const [head, rest] = split(kids());
			const out = nest(
				`[button${head || rest.length ? "" : label(node)}] ${head}`.trimEnd(),
				rest,
			);
			return rest.length ? block(out) : withMarks(out, marks);
		}
		if (tag === "li" || (style.display === "list-item" && tag !== "summary")) {
			const [head, rest] = split(kids());
			const parent = node.parentNode;
			let marker = "-";
			if (tagOf(parent) === "ol") {
				const siblings = (parent.childNodes || []).filter(
					(c) => tagOf(c) === "li",
				);
				marker = `${siblings.indexOf(node) + 1}.`;
			}
			return `\n${withMarks(nest(`${marker} ${head}`, rest), marks)}`;
		}
		if (tag === "tr") {
			const below = [];
			const cells = (node.childNodes || [])
				.filter((c) => /^(td|th)$/.test(tagOf(c)))
				.map((c) => {
					const cellStyle = uaStyle(c);
					if (unrendered(c, cellStyle)) return null;
					const rendered = render(c, { pre, cell: true });
					const [head, rest] = /^\n/.test(rendered)
						? ["", split(rendered).flat()]
						: split(rendered);
					below.push(...rest);
					return withMarks(head, stateMarks(c));
				})
				.filter((c) => c !== null);
			if (!cells.length) return "";
			return `\n${withMarks(nest(`| ${cells.join(" | ")} |`, below), marks)}`;
		}
		if (tag === "td" || tag === "th") {
			return ctx.cell ? kids() : block(kids());
		}
		if (tag === "br") return "\n";
		if (tag === "hr") return "\n---\n";
		if (tag === "img") {
			return attrs.src
				? `\n![${shown(attrs.alt || "", LABEL_CAP)}](${describeUrl(attrs.src)})${marks ? ` ${marks}` : ""}\n`
				: "";
		}
		// Live input state (what the visitor typed or toggled — rendered by the replayer from DOM properties)
		// outranks the snapshot-time value attribute.
		if (tag === "input") {
			const parts = [`type=${attrs.type || "text"}`];
			if (attrs.name) parts.push(`name=${shown(attrs.name, LABEL_CAP)}`);
			if (attrs.placeholder)
				parts.push(`placeholder=${quoted(attrs.placeholder, LABEL_CAP)}`);
			const value = node.props?.value ?? attrs.value;
			if (value !== undefined)
				parts.push(`value=${fieldValue(value)}`);
			const checkable = /^(checkbox|radio)$/i.test(attrs.type || "");
			const checked =
				node.props?.checked ?? ("checked" in attrs ? true : undefined);
			if (checkable && checked !== undefined) parts.push(`checked=${checked}`);
			const named = label(node);
			if (named && !/^(button|submit|reset)$/i.test(attrs.type || ""))
				parts.push(named.trim());
			return withMarks(`[input ${parts.join(" ")}]`, marks);
		}
		if (tag === "textarea") {
			const parts = ["textarea"];
			if (attrs.placeholder)
				parts.push(`placeholder=${quoted(attrs.placeholder, LABEL_CAP)}`);
			// A textarea's text children are its value, not prose: read raw, whitespace kept, under
			// the value ceiling. The field's value is the API value — the raw text with its newlines
			// normalized to LF (HTML's textarea rule), the one form the Input event also reports.
			const recorded = (node.childNodes || [])
				.map((c) => c.textContent || "")
				.join("")
				.replace(/\r\n?/g, "\n");
			const value = node.props?.value ?? recorded;
			if (value) parts.push(`value=${fieldValue(value)}`);
			const named = label(node);
			if (named) parts.push(named.trim());
			return withMarks(`[${parts.join(" ")}]`, marks);
		}
		if (tag === "select") {
			const parts = ["select"];
			if (attrs.name) parts.push(`name=${shown(attrs.name, LABEL_CAP)}`);
			const value = node.props?.value ?? attrs.value;
			if (value !== undefined)
				parts.push(`value=${fieldValue(value)}`);
			const named = label(node);
			if (named) parts.push(named.trim());
			return `${withMarks(`[${parts.join(" ")}]`, marks)}${kids()}\n`;
		}
		if (tag === "option") return `\n- ${withMarks(collapse(kids()), marks)}`;
		if (tag === "summary") return block(kids());

		const inner = kids();
		if (
			BLOCK_DISPLAYS.test(style.display || "") &&
			style.display !== "inline-block"
		) {
			if (pre && style.fontFamily === "monospace" && !ctx.pre) {
				const body = inner.replace(/^\n+|\n+$/g, "");
				return body.trim()
					? `\n\`\`\`\n${withMarks(body, marks)}\n\`\`\`\n`
					: "";
			}
			return block(inner);
		}
		return withMarks(inlineMarks(inner, style), marks);
	};

	return finish(render(mirror.getNode(rootId), { pre: false }));
}

// The browser's whitespace rules applied to the assembled text, fenced (whitespace-preserving)
// blocks excepted: runs of spaces collapse to one, a line's leading source whitespace collapses away,
// and a bracketed construct is set off from adjoining text by a space — that space is the
// projection's syntax, not a claim about the page. Indentation markers become their two spaces last.
function finish(text) {
	return text
		.split(/(\n```\n[\s\S]*?\n```\n)/)
		.map((chunk, i) =>
			i % 2
				? chunk
				: chunk
						.replace(/[ \t]+\n/g, "\n")
						.replace(/\n[ \t]+/g, "\n")
						.replace(/ {2,}/g, " ")
						.replace(
							/([^\s[])(\[(?:(?:button|input|select|textarea)\b|label=))/g,
							"$1 $2",
						)
						.replace(/\}(?=\[)/g, "} "),
		)
		.join("")
		.replace(/\n{3,}/g, "\n\n")
		.split(INDENT)
		.join("  ")
		.trim();
}

// Inline meaning as markdown marks, from the element's own user-agent style: bold, italic, struck,
// monospace, highlighted, quoted. Applied per line so grouping inside the element survives.
function inlineMarks(inner, style) {
	const wraps = [];
	if (/^(bold|bolder)$/.test(style.fontWeight || "")) wraps.push(["**", "**"]);
	if (style.fontStyle === "italic") wraps.push(["*", "*"]);
	if (/line-through/.test(style.textDecoration || "")) wraps.push(["~~", "~~"]);
	if (style.fontFamily === "monospace") wraps.push(["`", "`"]);
	if (
		style.background &&
		style.background !== "transparent" &&
		style.background !== "none"
	)
		wraps.push(["==", "=="]);
	if (style.before === "open-quote" && style.after === "close-quote")
		wraps.push(["“", "”"]);
	if (!wraps.length) return inner;
	const lead = /^\s/.test(inner) ? " " : "";
	const trail = /\s$/.test(inner) ? " " : "";
	const wrapped = inner
		.split("\n")
		.map((line) => {
			const core = collapse(line);
			if (!core) return line;
			return wraps.reduce((s, [o, c]) => `${o}${s}${c}`, core);
		})
		.join("\n");
	if (!wrapped.trim()) return inner ? " " : "";
	return `${lead}${wrapped}${trail}`;
}

// Whether the node is inside what the projection renders, by the projection's own criterion walked upward:
// false when the node or any ancestor is unrendered by the DOM's own statement, or is svg machinery, and
// false when the walk tops out anywhere but the mirror's root document — a detached subtree is outside the
// page's testimony. Shadow content hops to its host and iframe content to its host iframe, the same
// in-place rendering projectMarkdown does, so an embedded page's visibility is its host element's. Same
// residual blindness as the projection: cascade-based hiding is not resolved, so this answers "outside
// the projection", never "invisible to the eye".
// The text an element shows, by the projection's own criterion: text nodes under it, skipping the
// subtrees the projection never renders — svg definitions and anything the DOM states unrendered,
// scripts and noscript fallbacks included — so those never read as words on screen.
export function shownText(node) {
	if (!node) return "";
	if (node.type === NodeType.Text) return node.textContent || "";
	if (node.type === NodeType.Element) {
		const tag = tagOf(node);
		if (SVG_MACHINERY.has(tag) || unrendered(node, uaStyle(node))) return "";
	}
	return ((node.shadowRoot || node).childNodes || []).map(shownText).join("");
}

export function isProjected(mirror, id) {
	let node = mirror.getNode(id);
	while (node) {
		if (node.type === NodeType.Document) {
			if (node.host) {
				node = node.host;
				continue;
			}
			return node === mirror.rootDoc;
		}
		if (node.type === NodeType.Element) {
			const tag = tagOf(node);
			if (tag === "shadowroot") {
				node = node.host;
				continue;
			}
			if (SVG_MACHINERY.has(tag) || unrendered(node, uaStyle(node)))
				return false;
		}
		node = node.parentNode;
	}
	return false;
}
