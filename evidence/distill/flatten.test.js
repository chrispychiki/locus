import { expect, test } from "bun:test";
import { flatten, TARGET_TEXT_CAP } from "./flatten.js";
import { VALUE_CAP } from "./text.js";
import { LightweightMirror } from "./mirror.js";
import {
	EventType,
	EventTypeNames,
	IncrementalSource,
	IncrementalSourceNames,
	MediaInteractions,
	MediaInteractionsNames,
	MouseInteractions,
	MouseInteractionsNames,
	NodeType,
	PointerTypes,
} from "./rrweb_constants.js";

function mirrorWithButton() {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "DIV",
		id: 1,
		attributes: {},
		childNodes: [
			{
				type: NodeType.Element,
				tagName: "BUTTON",
				id: 2,
				attributes: { class: "cta primary" },
				childNodes: [
					{ type: NodeType.Text, id: 3, textContent: "  Add to  cart " },
				],
			},
		],
	});
	return mirror;
}

test("click resolves node columns from the mirror", () => {
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			timestamp: 1,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 2,
				x: 10,
				y: 20,
			},
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(cols).toMatchObject({
		type_str: "Click",
		x: 10,
		y: 20,
		tag: "button",
		class: "cta primary",
		text: "Add to cart",
	});
});

test("meta carries the url", () => {
	const cols = flatten(
		{ type: EventType.Meta, data: { href: "https://x.test/p" } },
		new LightweightMirror(),
		{ wrongTyped: 0 },
	);
	expect(cols.type_str).toBe("Meta");
	expect(cols.url).toBe("https://x.test/p");
});

test("custom recorder events translate to readable names", () => {
	const pageload = flatten(
		{ type: EventType.PageLoad, data: { url: "https://x.test/" } },
		new LightweightMirror(),
		{ wrongTyped: 0 },
	);
	expect(pageload.type_str).toBe("PageLoad");
	expect(pageload.url).toBe("https://x.test/");
	const hidden = flatten(
		{ type: EventType.PageHidden, data: { url: "https://x.test/" } },
		new LightweightMirror(),
		{ wrongTyped: 0 },
	);
	expect(hidden.type_str).toBe("PageHidden");
	expect(hidden.url).toBe("https://x.test/");
});

test("input events carry the text, mousemove the last position", () => {
	const input = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 2, text: "ab" },
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(input.type_str).toBe("Input");
	expect(input.input).toBe("ab");

	const move = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MouseMove,
				positions: [
					{ x: 1, y: 2 },
					{ x: 9, y: 8 },
				],
			},
		},
		new LightweightMirror(),
		{ wrongTyped: 0 },
	);
	expect(move.type_str).toBe("MouseMove");
	expect(move.x).toBe(9);
	expect(move.y).toBe(8);
});

test("a pointer type translates to its name", () => {
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 2,
				x: 1,
				y: 2,
				pointerType: PointerTypes.Touch,
			},
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(cols.pointer_type).toBe("Touch");
});

test("an anchor's href rides its own column, never extra", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "A",
		id: 4,
		attributes: { href: "/pricing", class: "nav" },
		childNodes: [{ type: NodeType.Text, id: 5, textContent: "Pricing" }],
	});
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 4,
				x: 1,
				y: 2,
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(cols.href).toBe("/pricing");
	expect(cols.class).toBe("nav");
	expect(cols.text).toBe("Pricing");
	expect(cols.extra).toBeNull();
});

test("a click on an element inside a link carries the link's destination", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "A",
		id: 4,
		attributes: { href: "https://linktr.ee/x" },
		childNodes: [
			{
				type: NodeType.Element,
				tagName: "EM",
				id: 5,
				attributes: { class: "title" },
				childNodes: [{ type: NodeType.Text, id: 6, textContent: "Buy" }],
			},
		],
	});
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 5,
				x: 1,
				y: 2,
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(cols.tag).toBe("em");
	expect(cols.href).toBe("https://linktr.ee/x");
	expect(cols.text).toBe("Buy");
});

const clickOn = (id) => ({
	type: EventType.IncrementalSnapshot,
	data: {
		source: IncrementalSource.MouseInteraction,
		type: MouseInteractions.Click,
		id,
		x: 1,
		y: 2,
	},
});

test("a scroll inside a link carries no destination — only a click follows the anchor", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "A",
		id: 4,
		attributes: { href: "https://linktr.ee/x" },
		childNodes: [
			{
				type: NodeType.Element,
				tagName: "DIV",
				id: 5,
				attributes: {},
				childNodes: [],
			},
		],
	});
	const scroll = {
		type: EventType.IncrementalSnapshot,
		data: { source: IncrementalSource.Scroll, id: 5, x: 0, y: 300 },
	};
	expect(flatten(scroll, mirror, { wrongTyped: 0 }).href).toBeNull();
	expect(flatten(clickOn(5), mirror, { wrongTyped: 0 }).href).toBe(
		"https://linktr.ee/x",
	);
});

test("a click inside a shadow tree carries its host's enclosing link", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "A",
			id: 4,
			attributes: { href: "https://example.com/go" },
			childNodes: [
				{
					type: NodeType.Element,
					tagName: "MY-ICON",
					id: 5,
					attributes: {},
					isShadowHost: true,
					childNodes: [
						{
							type: NodeType.Element,
							tagName: "SVG",
							id: 6,
							attributes: {},
							isShadow: true,
							childNodes: [],
						},
					],
				},
			],
		},
	]);
	expect(flatten(clickOn(6), mirror, { wrongTyped: 0 }).href).toBe(
		"https://example.com/go",
	);
});

test("a click inside an iframe never carries the parent page's link — the browser follows none", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "A",
			id: 4,
			attributes: { href: "https://example.com/outer" },
			childNodes: [
				{ type: NodeType.Element, tagName: "IFRAME", id: 5, attributes: {} },
			],
		},
	]);
	mirror.applyMutation({
		adds: [
			{
				parentId: 5,
				nextId: null,
				node: {
					type: NodeType.Document,
					id: 10,
					childNodes: [
						{
							type: NodeType.Element,
							tagName: "HTML",
							id: 11,
							attributes: {},
							childNodes: [
								{
									type: NodeType.Element,
									tagName: "BUTTON",
									id: 12,
									attributes: {},
									childNodes: [],
								},
							],
						},
					],
				},
			},
		],
	});
	expect(flatten(clickOn(12), mirror, { wrongTyped: 0 }).href).toBeNull();
});

test("a checkable input carries its toggle as the value; the wire fields never leak into extra", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "INPUT",
		id: 4,
		attributes: { type: "checkbox", name: "tos" },
		childNodes: [],
	});
	const toggled = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 4, isChecked: true },
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(toggled.input).toBe("true");
	const extra = toggled.extra ? JSON.parse(toggled.extra) : {};
	expect(extra.isChecked).toBeUndefined();
	expect(extra.text).toBeUndefined();

	const counters = { wrongTyped: 0 };
	const wrong = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 4, isChecked: "yes" },
		},
		mirror,
		counters,
	);
	expect(wrong.input).toBeNull();
	expect(counters.wrongTyped).toBe(1);
});

test("a selection whose start node never resolved carries no text", () => {
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.Selection,
				ranges: [{ start: 99, startOffset: 0, end: 99, endOffset: 5 }],
			},
		},
		new LightweightMirror(),
		{ wrongTyped: 0 },
	);
	expect(cols.type_str).toBe("Selection");
	expect(cols.text).toBeNull();
});

test("input values are captured verbatim — masking is the recorder's job, not the flattener's", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "DIV",
		id: 1,
		attributes: {},
		childNodes: [
			{
				type: NodeType.Element,
				tagName: "INPUT",
				id: 2,
				attributes: { type: "hidden", name: "cf-turnstile-response" },
			},
			{
				type: NodeType.Element,
				tagName: "INPUT",
				id: 5,
				attributes: { type: "text" },
			},
		],
	});
	const token = `0.${"x".repeat(98)}`;
	const hidden = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 2, text: token },
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(hidden.input).toBe(token);

	const visible = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 5, text: "hello" },
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(visible.input).toBe("hello");
});

test("an input value under the ceiling rides whole — the value is the payload", () => {
	const pasted = "word ".repeat(1000).trim(); // a real paste, ~5k chars
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 2, text: pasted },
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(cols.input).toBe(pasted);
	expect(
		cols.extra ? JSON.parse(cols.extra).input_chars : undefined,
	).toBeUndefined();
});

test("an input value past the ceiling is cut visibly and testifies its true size", () => {
	const blob = "x".repeat(60000); // script-set pathology scale
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 2, text: blob },
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(cols.input.length).toBe(VALUE_CAP);
	expect(cols.input.endsWith("…")).toBe(true);
	expect(JSON.parse(cols.extra).input_chars).toBe(60000);
});

test("a selection past the ceiling keeps value-role width and testifies its size", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "P",
		id: 1,
		attributes: {},
		childNodes: [
			{ type: NodeType.Text, id: 2, textContent: "y".repeat(60000) },
		],
	});
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.Selection,
				ranges: [{ start: 2, startOffset: 0, end: 2, endOffset: 60000 }],
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(cols.text.length).toBe(VALUE_CAP);
	expect(cols.text.endsWith("…")).toBe(true);
	expect(JSON.parse(cols.extra).text_chars).toBe(60000);
});

test("interaction target text caps at label length on a word boundary", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "DIV",
		id: 1,
		attributes: { class: "card" },
		childNodes: [
			{
				type: NodeType.Text,
				id: 2,
				textContent:
					"Premium heavyweight flannel overshirt woven from brushed cotton twill with a two-pocket chest and corozo buttons throughout",
			},
		],
	});
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 1,
				x: 5,
				y: 5,
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(cols.text.length).toBeLessThanOrEqual(TARGET_TEXT_CAP);
	expect(cols.text.endsWith("…")).toBe(true);
	expect(cols.text.slice(0, -1).endsWith(" ")).toBe(false);
	expect(
		"Premium heavyweight flannel overshirt woven from brushed cotton twill with a two-pocket chest and corozo buttons throughout".startsWith(
			cols.text.slice(0, -1),
		),
	).toBe(true);
});

test("verbose attrs blacklisted; data URIs summarized; points kept", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "IMG",
		id: 2,
		childNodes: [],
		attributes: {
			srcset: "a 1x, b 2x",
			src: `data:image/png;base64,${"Q".repeat(5000)}`,
			points: "0,0 1,1",
		},
	});
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 2,
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	const extra = JSON.parse(cols.extra);
	expect(extra.srcset).toBeUndefined();
	expect(extra.src).toMatch(/^data:image\/png;base64,… \d+kB inlined$/);
	expect(extra.points).toBe("0,0 1,1");
});

test("media interactions carry their verb", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "VIDEO",
		id: 9,
		attributes: {},
		childNodes: [],
	});
	const play = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MediaInteraction,
				type: MediaInteractions.Play,
				id: 9,
				currentTime: 0.03,
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(play.type_str).toBe("Play");
	expect(play.tag).toBe("video");
	const pause = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MediaInteraction,
				type: MediaInteractions.Pause,
				id: 9,
				currentTime: 14.16,
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(pause.type_str).toBe("Pause");
});

test("selection events carry the selected text", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "P",
		id: 1,
		attributes: {},
		childNodes: [
			{
				type: NodeType.Text,
				id: 2,
				textContent: "The paper demonstrates results.",
			},
			{ type: NodeType.Text, id: 3, textContent: "A second sentence here." },
		],
	});
	const single = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.Selection,
				ranges: [{ start: 2, startOffset: 4, end: 2, endOffset: 9 }],
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(single.type_str).toBe("Selection");
	expect(single.text).toBe("paper");

	const spanning = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.Selection,
				ranges: [{ start: 2, startOffset: 4, end: 3, endOffset: 8 }],
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(spanning.text).toBe("paper demonstrates results. … A second");
});

test("unknown numeric types surface as opaque numbers, never crash", () => {
	const unknownType = 50;
	expect(EventTypeNames[unknownType]).toBeUndefined();
	const cols = flatten(
		{ type: unknownType, data: {} },
		new LightweightMirror(),
		{ wrongTyped: 0 },
	);
	expect(cols.type_str).toBe("50");
});

test("every schema enum member translates to a name — models never see codes", () => {
	const mirror = new LightweightMirror();
	for (const [value, name] of Object.entries(EventTypeNames)) {
		if (Number(value) === EventType.IncrementalSnapshot) continue;
		const cols = flatten({ type: Number(value), data: {} }, mirror, {
			wrongTyped: 0,
		});
		expect(cols.type_str).toBe(name);
	}
	for (const value of Object.keys(IncrementalSourceNames)) {
		const cols = flatten(
			{
				type: EventType.IncrementalSnapshot,
				data: { source: Number(value) },
			},
			mirror,
			{ wrongTyped: 0 },
		);
		expect(cols.type_str).not.toMatch(/^\d+$/);
	}
	for (const [value, name] of Object.entries(MouseInteractionsNames)) {
		const cols = flatten(
			{
				type: EventType.IncrementalSnapshot,
				data: {
					source: IncrementalSource.MouseInteraction,
					type: Number(value),
				},
			},
			mirror,
			{ wrongTyped: 0 },
		);
		expect(cols.type_str).toBe(name);
	}
	for (const [value, name] of Object.entries(MediaInteractionsNames)) {
		const cols = flatten(
			{
				type: EventType.IncrementalSnapshot,
				data: {
					source: IncrementalSource.MediaInteraction,
					type: Number(value),
				},
			},
			mirror,
			{ wrongTyped: 0 },
		);
		expect(cols.type_str).toBe(name);
	}
});

test("wire values a column cannot hold drop to null and are counted, never bound", () => {
	const counters = { wrongTyped: 0 };
	const cols = flatten(
		{
			type: EventType.Meta,
			timestamp: 1,
			data: { href: { evil: true }, width: 800, height: 600 },
		},
		new LightweightMirror(),
		counters,
	);
	expect(cols.url).toBeNull();
	expect(counters.wrongTyped).toBe(1);

	const click = flatten(
		{
			type: EventType.IncrementalSnapshot,
			timestamp: 1,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 2,
				x: "12",
				y: [3],
				pointerType: {},
			},
		},
		mirrorWithButton(),
		counters,
	);
	expect(click).toMatchObject({
		type_str: "Click",
		x: null,
		y: null,
		pointer_type: null,
		tag: "button",
	});
	expect(counters.wrongTyped).toBe(4);

	const input = flatten(
		{
			type: EventType.IncrementalSnapshot,
			timestamp: 1,
			data: { source: IncrementalSource.Input, id: 2, text: { nested: "x" } },
		},
		mirrorWithButton(),
		counters,
	);
	expect(input.input).toBeNull();
	expect(counters.wrongTyped).toBe(5);

	const move = flatten(
		{
			type: EventType.IncrementalSnapshot,
			timestamp: 1,
			data: {
				source: IncrementalSource.MouseMove,
				positions: [{ x: "a", y: 2, id: 1, timeOffset: -10 }],
			},
		},
		new LightweightMirror(),
		counters,
	);
	expect(move.x).toBeNull();
	expect(move.y).toBe(2);
	expect(counters.wrongTyped).toBe(6);
});

// A code the generated schema has no name for — derived from the schema itself, so it is unknown by
// construction and stays unknown the day rrweb adds one more.
const unnamed = (names) => Math.max(...Object.keys(names).map(Number)) + 1;

// The unknown-kind counter is the only signal that says the generated constants have drifted from
// the recorder's rrweb. An interaction subtype the schema cannot name must not fall back to the
// source's own name — "MouseInteraction", a kind the schema knows — or the counter never fires and
// an unnameable interaction reads downstream as an ordinary one for as long as the drift lasts.
test("an interaction subtype the schema cannot name says so, instead of passing as a known kind", () => {
	const code = unnamed(MouseInteractionsNames);
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			timestamp: 1,
			data: { source: IncrementalSource.MouseInteraction, type: code, id: 2 },
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(cols.type_str).toBe(`MouseInteraction:${code}`);
	expect(Object.values(MouseInteractionsNames)).not.toContain(cols.type_str);
	expect(Object.values(IncrementalSourceNames)).not.toContain(cols.type_str);
});

test("a media subtype the schema cannot name says so too", () => {
	const code = unnamed(MediaInteractionsNames);
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			timestamp: 1,
			data: { source: IncrementalSource.MediaInteraction, type: code, id: 2 },
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(cols.type_str).toBe(`MediaInteraction:${code}`);
	expect(Object.values(MediaInteractionsNames)).not.toContain(cols.type_str);
});

// The projection's visibility boundary, stamped per target-carrying event: what the page never
// displays is machinery, and the hidden column is how the event stream knows to keep it out.
function documentMirror(bodyChildren) {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Document,
		id: 1,
		childNodes: [
			{
				type: NodeType.Element,
				tagName: "HTML",
				id: 2,
				attributes: {},
				childNodes: [
					{
						type: NodeType.Element,
						tagName: "BODY",
						id: 3,
						attributes: {},
						childNodes: bodyChildren,
					},
				],
			},
		],
	});
	return mirror;
}

const inputEvent = (id) => ({
	type: EventType.IncrementalSnapshot,
	data: { source: IncrementalSource.Input, id, text: "v" },
});

test("an input the page displays is stamped visible", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "INPUT",
			id: 4,
			attributes: { type: "text" },
		},
	]);
	expect(flatten(inputEvent(4), mirror, { wrongTyped: 0 }).hidden).toBe(0);
});

test("an input under a display:none ancestor is stamped hidden", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "DIV",
			id: 4,
			attributes: { style: "display: none;" },
			childNodes: [
				{ type: NodeType.Element, tagName: "INPUT", id: 5, attributes: {} },
			],
		},
	]);
	expect(flatten(inputEvent(5), mirror, { wrongTyped: 0 }).hidden).toBe(1);
});

test("a type=hidden input is stamped hidden", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "INPUT",
			id: 4,
			attributes: { type: "hidden" },
		},
	]);
	expect(flatten(inputEvent(4), mirror, { wrongTyped: 0 }).hidden).toBe(1);
});

test("an input inside a same-origin iframe is visible — the embedded page is presented content", () => {
	const mirror = documentMirror([
		{ type: NodeType.Element, tagName: "IFRAME", id: 4, attributes: {} },
	]);
	mirror.applyMutation({
		adds: [
			{
				parentId: 4,
				nextId: null,
				node: {
					type: NodeType.Document,
					id: 10,
					childNodes: [
						{
							type: NodeType.Element,
							tagName: "HTML",
							id: 11,
							attributes: {},
							childNodes: [
								{
									type: NodeType.Element,
									tagName: "INPUT",
									id: 12,
									attributes: {},
								},
							],
						},
					],
				},
			},
		],
	});
	expect(flatten(inputEvent(12), mirror, { wrongTyped: 0 }).hidden).toBe(0);
});

test("an input inside a hidden iframe stays hidden — the host's visibility governs its content", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "IFRAME",
			id: 4,
			attributes: { style: "display: none;" },
		},
	]);
	mirror.applyMutation({
		adds: [
			{
				parentId: 4,
				nextId: null,
				node: {
					type: NodeType.Document,
					id: 10,
					childNodes: [
						{
							type: NodeType.Element,
							tagName: "HTML",
							id: 11,
							attributes: {},
							childNodes: [
								{
									type: NodeType.Element,
									tagName: "INPUT",
									id: 12,
									attributes: {},
								},
							],
						},
					],
				},
			},
		],
	});
	expect(flatten(inputEvent(12), mirror, { wrongTyped: 0 }).hidden).toBe(1);
});

test("shadow content hops to its host — projected shadow inputs are visible", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "MY-WIDGET",
			id: 4,
			attributes: {},
			isShadowHost: true,
			childNodes: [
				{
					type: NodeType.Element,
					tagName: "INPUT",
					id: 5,
					attributes: {},
					isShadow: true,
				},
			],
		},
	]);
	expect(flatten(inputEvent(5), mirror, { wrongTyped: 0 }).hidden).toBe(0);
});

test("a target that never resolved stays unjudged", () => {
	expect(
		flatten(inputEvent(99), documentMirror([]), { wrongTyped: 0 }).hidden,
	).toBe(null);
});

test("an input inside a SKIP_TAGS subtree is stamped hidden — template content is never displayed", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "TEMPLATE",
			id: 4,
			attributes: {},
			childNodes: [
				{ type: NodeType.Element, tagName: "INPUT", id: 5, attributes: {} },
			],
		},
	]);
	expect(flatten(inputEvent(5), mirror, { wrongTyped: 0 }).hidden).toBe(1);
});

test("a still-addressable node whose subtree was detached is stamped hidden — the page no longer paints it", () => {
	// Shadow children survive their host's removal in the mirror's map, exactly as the replayer
	// leaves them — a mapped node whose ancestor walk tops out off the root document.
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "MY-WIDGET",
			id: 4,
			attributes: {},
			isShadowHost: true,
			childNodes: [
				{
					type: NodeType.Element,
					tagName: "INPUT",
					id: 5,
					attributes: {},
					isShadow: true,
				},
			],
		},
	]);
	mirror.applyMutation({
		adds: [],
		removes: [{ id: 4, parentId: 3 }],
		texts: [],
		attributes: [],
	});
	expect(mirror.getNode(5)).not.toBeNull();
	expect(flatten(inputEvent(5), mirror, { wrongTyped: 0 }).hidden).toBe(1);
});

test("coordinates land as whole pixels — sub-pixel float jitter is not testimony", () => {
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			timestamp: 1,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 2,
				x: 370.34283447265625,
				y: 174.61144256591797,
			},
		},
		mirrorWithButton(),
		{ wrongTyped: 0 },
	);
	expect(cols.x).toBe(370);
	expect(cols.y).toBe(175);
});

test("a typed input's element text is not quoted — it is the value itself, one event stale", () => {
	const mirror = documentMirror([
		{
			type: NodeType.Element,
			tagName: "TEXTAREA",
			id: 4,
			attributes: {},
			childNodes: [{ type: NodeType.Text, id: 5, textContent: "old value" }],
		},
	]);
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Input, id: 4, text: "new value" },
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(cols.input).toBe("new value");
	expect(cols.text).toBe(null);
});

test("a textless target is stamped with the page's name for it, by the projection's derivation", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "BUTTON",
		id: 1,
		attributes: { "aria-label": "Open  cart", name: "cart" },
		childNodes: [],
	});
	mirror.addNode({
		type: NodeType.Element,
		tagName: "INPUT",
		id: 2,
		attributes: { type: "search", placeholder: "Search products" },
		childNodes: [],
	});
	mirror.addNode({
		type: NodeType.Element,
		tagName: "A",
		id: 3,
		attributes: {
			href: "https://x.test/more",
			"aria-label": "Read more about it",
		},
		childNodes: [{ type: NodeType.Text, id: 4, textContent: "Read more" }],
	});
	const click = (id) =>
		flatten(
			{
				type: EventType.IncrementalSnapshot,
				data: {
					source: IncrementalSource.MouseInteraction,
					type: MouseInteractions.Click,
					id,
					x: 1,
					y: 1,
				},
			},
			mirror,
			{ wrongTyped: 0 },
		);
	expect(JSON.parse(click(1).extra).label).toBe("Open cart");
	expect(JSON.parse(click(2).extra).label).toBe("Search products");
	const named = click(3);
	expect(named.text).toBe("Read more");
	expect(JSON.parse(named.extra).label).toBeUndefined();
});

test("a target's text is what it shows — a noscript fallback or an unrendered helper never reads as words on screen", () => {
	const mirror = new LightweightMirror();
	mirror.addNode({
		type: NodeType.Element,
		tagName: "DIV",
		id: 1,
		attributes: { class: "carousel" },
		childNodes: [
			{
				type: NodeType.Element,
				tagName: "NOSCRIPT",
				id: 2,
				attributes: {},
				childNodes: [
					{ type: NodeType.Text, id: 3, textContent: '<img src="a.jpg">' },
				],
			},
			{
				type: NodeType.Element,
				tagName: "SPAN",
				id: 4,
				attributes: { style: "display:none" },
				childNodes: [{ type: NodeType.Text, id: 5, textContent: "helper" }],
			},
			{ type: NodeType.Text, id: 6, textContent: "Slide 1 of 4" },
		],
	});
	const cols = flatten(
		{
			type: EventType.IncrementalSnapshot,
			data: {
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 1,
				x: 1,
				y: 1,
			},
		},
		mirror,
		{ wrongTyped: 0 },
	);
	expect(cols.text).toBe("Slide 1 of 4");
});
