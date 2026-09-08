import { expect, test } from "bun:test";
import { LightweightMirror } from "./mirror.js";
import { projectMarkdown } from "./project.js";
import { LABEL_CAP, TEXT_CAP, VALUE_CAP } from "./text.js";
import { EventType, NodeType } from "./rrweb_constants.js";

const rec = await Bun.file(
	new URL("../tests/fixtures/demo_recording.json", import.meta.url),
).json();
const firstFull = rec.events.filter(
	(e) => e.type === EventType.FullSnapshot,
)[0];

function project() {
	const mirror = new LightweightMirror();
	mirror.addNode(firstFull.data.node);
	return projectMarkdown(mirror, firstFull.data.node.id);
}

test("projects a captured FullSnapshot DOM to readable markdown", () => {
	const md = project();
	expect(md.length).toBeGreaterThan(4000);
	expect(md).toContain("Meridian Coffee");
	expect(md).toContain("View roast");
	expect(md).toMatch(/\]\(http/);
});

test("drops non-informational subtrees", () => {
	const md = project();
	expect(md).not.toMatch(/function\s*\(|var \w+\s*=/);
	expect(md).not.toContain("<svg");
});

test("projection is a large reduction over the raw DOM", () => {
	const md = project();
	const rawSize = JSON.stringify(firstFull.data.node).length;
	expect(md.length).toBeLessThan(rawSize / 5);
});

const el = (id, tagName, attributes = {}, childNodes = [], extra = {}) => ({
	type: NodeType.Element,
	id,
	tagName,
	attributes,
	childNodes,
	...extra,
});
const txt = (id, textContent) => ({ type: NodeType.Text, id, textContent });

// Real rrweb mutation events always carry all four arrays, so tests supply the full shape — the
// realistic input, not a reduced convenience form the wire never produces.
const MUT = (d) => ({ adds: [], removes: [], texts: [], attributes: [], ...d });

// Synthetic test DOMs are real FullSnapshots fed to the real mirror: wrap the test root in a document so
// the snapshot builds the way a recording's would, then project from the inner id — so the html/body
// scaffold never enters the asserted output.
const DOC = (kids) => ({
	type: NodeType.Document,
	id: 90000,
	childNodes: kids,
});
const HTML = (kids) => ({
	type: NodeType.Element,
	id: 90001,
	tagName: "html",
	attributes: {},
	childNodes: kids,
});
const BODY = (kids) => ({
	type: NodeType.Element,
	id: 90002,
	tagName: "body",
	attributes: {},
	childNodes: kids,
});

function mirrorOf(root) {
	const mirror = new LightweightMirror();
	const wrapped =
		root.tagName === "body" ? DOC([HTML([root])]) : DOC([HTML([BODY([root])])]);
	mirror.addNode(wrapped);
	return mirror;
}

function projectNode(node) {
	return projectMarkdown(mirrorOf(node), node.id);
}

test("hidden subtrees are excluded — the projection is testimony", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "div", { style: "display:none" }, [txt(3, "toast text")]),
			el(4, "div", { style: "visibility: hidden" }, [txt(5, "tooltip")]),
			el(6, "div", { hidden: "" }, [txt(7, "drawer")]),
			el(8, "input", { type: "hidden", name: "csrf", value: "tok" }),
			el(9, "p", {}, [txt(10, "visible paragraph")]),
		]),
	);
	expect(md).toBe("visible paragraph");
});

test("data: URIs are summarized to header and size, never inlined", () => {
	const photo = `data:image/png;base64,${"A".repeat(700 * 1024)}`;
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "img", { src: photo, alt: "First captured photo" }),
			el(3, "a", { href: photo }, [txt(4, "download")]),
			el(5, "img", { src: "https://cdn.example.com/logo.png" }),
		]),
	);
	expect(md.length).toBeLessThan(500);
	expect(md).toContain("First captured photo");
	expect(md).toContain("data:image/png;base64,");
	expect(md).toContain("700kB inlined");
	expect(md).toContain("https://cdn.example.com/logo.png");
});

test("monster URLs shed their query — serialized-state blobs never dominate the projection", () => {
	const tracked = `https://school.example.com/?_gl=1*${"x".repeat(3000)}#/recording/42`;
	const md = projectNode(
		el(1, "div", {}, [el(2, "a", { href: tracked }, [txt(3, "3")])]),
	);
	expect(md.length).toBeLessThan(160);
	expect(md).toContain(
		"https://school.example.com/?… (3kB query)#/recording/42",
	);
});

test("a reveal mutation changes the projection", () => {
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "div", { style: "display:none" }, [txt(3, "Invalid credentials!")]),
			el(4, "p", {}, [txt(5, "Sign In")]),
		]),
	);
	const before = projectMarkdown(mirror, 1);
	mirror.applyMutation(
		MUT({ attributes: [{ id: 2, attributes: { style: "" } }] }),
	);
	const after = projectMarkdown(mirror, 1);
	expect(before).not.toContain("Invalid credentials!");
	expect(after).toContain("Invalid credentials!");
});

test("inserted subtrees appear in the projection (sonner-toast shape)", () => {
	const mirror = mirrorOf(
		el(1, "body", {}, [el(64, "div", {}, [txt(65, "Sign In")])]),
	);
	const before = projectMarkdown(mirror, 1);
	mirror.applyMutation(
		MUT({
			adds: [
				{ parentId: 64, nextId: null, node: el(150, "ol", {}) },
				{ parentId: 150, nextId: null, node: el(151, "li", {}) },
				{ parentId: 151, nextId: null, node: txt(152, "Invalid credentials!") },
			],
		}),
	);
	const after = projectMarkdown(mirror, 1);
	expect(before).not.toContain("Invalid credentials!");
	expect(after).toContain("Invalid credentials!");
});

test("nextId places siblings in document order", () => {
	// "A" is added before "B", so the two text nodes project adjacent and in that order — the
	// browser's inline flow puts no space between them.
	const mirror = mirrorOf(el(1, "body", {}, [txt(2, "B")]));
	mirror.applyMutation(
		MUT({ adds: [{ parentId: 1, nextId: 2, node: txt(3, "A") }] }),
	);
	expect(projectMarkdown(mirror, 1)).toBe("AB");
});

test("removed nodes leave the projection; moves do not duplicate", () => {
	const mirror = mirrorOf(
		el(1, "body", {}, [
			el(2, "p", {}, [txt(3, "gone")]),
			el(4, "p", {}, [txt(5, "kept")]),
		]),
	);
	mirror.applyMutation(MUT({ removes: [{ id: 2, parentId: 1 }] }));
	expect(projectMarkdown(mirror, 1)).toBe("kept");
});

test("headings render with their markdown level", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "h1", {}, [txt(3, "Big")]),
			el(4, "h3", {}, [txt(5, "Sub")]),
			el(6, "p", {}, [txt(7, "body copy")]),
		]),
	);
	expect(md).toBe("# Big\n\n### Sub\n\nbody copy");
});

test("block boundaries assert line breaks — sibling blocks never run together", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "div", {}, [txt(3, "first block")]),
			el(4, "div", {}, [txt(5, "second block")]),
		]),
	);
	expect(md).toBe("first block\n\nsecond block");
});

test("source whitespace collapses like the browser's normal flow", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "span", {}, [txt(3, "11 /")]),
			txt(4, "\n                             13"),
		]),
	);
	expect(md).toBe("11 / 13");
});

test("select options carry explicit boundaries", () => {
	const md = projectNode(
		el(1, "select", { name: "dept", value: "RD" }, [
			el(2, "option", {}, [txt(3, "INTERN")]),
			el(4, "option", {}, [txt(5, "RD")]),
			el(6, "option", {}, [txt(7, "Sale")]),
		]),
	);
	expect(md).toBe('[select name=dept value="RD"]\n- INTERN\n- RD\n- Sale');
});

test("table rows render cells with separators", () => {
	const md = projectNode(
		el(1, "table", {}, [
			el(2, "tr", {}, [
				el(3, "th", {}, [txt(4, "Name")]),
				el(5, "th", {}, [txt(6, "Score")]),
			]),
			el(7, "tr", {}, [
				el(8, "td", {}, [txt(9, "Kim")]),
				el(10, "td", {}, [txt(11, "92")]),
			]),
		]),
	);
	expect(md).toBe("| Name | Score |\n| Kim | 92 |");
});

test("emphasis renders as markdown markers", () => {
	const md = projectNode(
		el(1, "p", {}, [
			txt(2, "a "),
			el(3, "strong", {}, [txt(4, "bold")]),
			txt(5, " and"),
			el(6, "em", {}, [txt(7, " italic ")]),
			txt(8, "word"),
		]),
	);
	expect(md).toBe("a **bold** and *italic* word");
});

test("typed input state outranks the snapshot-time value attribute", () => {
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "input", {
				type: "email",
				name: "email",
				value: "stale@snapshot.test",
			}),
		]),
	);
	mirror.applyInput({ id: 2, text: "typed@live.test", isChecked: false });
	const md = projectMarkdown(mirror, 1);
	expect(md).toContain('value="typed@live.test"');
	expect(md).not.toContain("stale@snapshot.test");
	expect(md).not.toContain("checked");
});

test("checked state testifies on checkables only", () => {
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "input", { type: "checkbox", name: "tos" }),
			el(3, "input", { type: "text", name: "q" }),
		]),
	);
	mirror.applyInput({ id: 2, text: "on", isChecked: true });
	mirror.applyInput({ id: 3, text: "hello", isChecked: false });
	const md = projectMarkdown(mirror, 1);
	expect(md).toContain("checked=true");
	expect(md.match(/checked/g).length).toBe(1);
});

test("textarea content testifies: recorded value, then typed state", () => {
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "textarea", { placeholder: "Notes", value: "recorded draft" }),
		]),
	);
	expect(projectMarkdown(mirror, 1)).toContain('value="recorded draft"');
	mirror.applyInput({ id: 2, text: "typed over it", isChecked: false });
	expect(projectMarkdown(mirror, 1)).toContain('value="typed over it"');
});

test("a field's value past the value ceiling is cut and testifies its size", () => {
	const paste = "p".repeat(VALUE_CAP + 9000);
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "textarea", { placeholder: "Notes" }),
			el(3, "input", { type: "text", name: "q", value: paste }),
		]),
	);
	mirror.applyInput({ id: 2, text: paste, isChecked: false });
	const md = projectMarkdown(mirror, 1);
	const note = ` (first ${VALUE_CAP - 1} of ${paste.length} chars)`;
	expect(md).toContain(`value="${"p".repeat(VALUE_CAP - 1)}…"${note}`);
	expect(md.split(note).length).toBe(3);
	expect(md).not.toContain(paste);
});

test("a field's value keeps its line structure on one line, cut on the raw text before escaping", () => {
	const typed = "first line\n\tsecond \\ line\r\nthird";
	const mirror = mirrorOf(
		el(1, "div", {}, [el(2, "textarea", {}, [txt(3, "a\nb")])]),
	);
	expect(projectMarkdown(mirror, 1)).toBe('[textarea value="a\\nb"]');
	mirror.applyInput({ id: 2, text: typed, isChecked: false });
	expect(projectMarkdown(mirror, 1)).toBe(
		'[textarea value="first line\\n\\tsecond \\\\ line\\r\\nthird"]',
	);
	const paste = `${"p".repeat(VALUE_CAP - 5)}\n\n\n\n\n\n\n\n\n\n`;
	mirror.applyInput({ id: 2, text: paste, isChecked: false });
	const md = projectMarkdown(mirror, 1);
	expect(md).toContain(`value="${"p".repeat(VALUE_CAP - 5)}\\n\\n\\n\\n…" (first ${VALUE_CAP - 1} of ${paste.length} chars)`);
});

test("a textarea's recorded text is its API value: newlines normalized to LF, as an Input reports them", () => {
	const md = projectNode(
		el(1, "div", {}, [el(2, "textarea", {}, [txt(3, "a\r\nb\rc\nd")])]),
	);
	expect(md).toBe('[textarea value="a\\nb\\nc\\nd"]');
});

test("a textarea's recorded text is its value: the value ceiling applies, not the text node's", () => {
	const draft = "d".repeat(TEXT_CAP + 2000);
	const md = projectNode(
		el(1, "div", {}, [el(2, "textarea", {}, [txt(3, draft)])]),
	);
	expect(md).toBe(`[textarea value="${draft}"]`);
});

test("a text node past the text ceiling is cut and testifies its size; prose below rides whole", () => {
	const blob = `{"k":${"1".repeat(TEXT_CAP + 500)}}`;
	const prose = "w".repeat(TEXT_CAP - 1);
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "pre", {}, [txt(3, blob)]),
			el(4, "p", {}, [txt(5, prose)]),
		]),
	);
	const md = projectMarkdown(mirror, 1);
	expect(md).toContain(`… (first ${TEXT_CAP - 1} of ${blob.length} chars)`);
	expect(md).not.toContain(blob);
	expect(md).toContain(prose);
});

test("label attributes past the label ceiling are cut and testify their size", () => {
	const dump = "d".repeat(LABEL_CAP + 300);
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "img", { src: "https://x.test/a.png", alt: dump }),
			el(3, "input", { type: "text", placeholder: dump, name: dump }),
		]),
	);
	const md = projectMarkdown(mirror, 1);
	const note = ` (first ${LABEL_CAP - 1} of ${dump.length} chars)`;
	expect(md).toContain(
		`![${"d".repeat(LABEL_CAP - 1)}…${note}](https://x.test/a.png)`,
	);
	expect(md).toContain(`name=${"d".repeat(LABEL_CAP - 1)}…${note}`);
	expect(md).toContain(`placeholder="${"d".repeat(LABEL_CAP - 1)}…"${note}`);
	expect(md).not.toContain(dump);
});

test("shadow-DOM content renders in place of the host's light children", () => {
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(
				2,
				"cookie-banner",
				{},
				[
					el(3, "p", {}, [txt(4, "We use cookies")], { isShadow: true }),
					txt(5, "unslotted light text"),
				],
				{ isShadowHost: true },
			),
		]),
	);
	const md = projectMarkdown(mirror, 1);
	expect(md).toContain("We use cookies");
	expect(md).not.toContain("unslotted light text");
});

test("svg text projects — chart labels and in-svg copy are presented content", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "svg", {}, [
				el(3, "title", {}, [txt(4, "Revenue chart")]),
				el(5, "desc", {}, [txt(6, "Quarterly revenue by region")]),
				el(7, "defs", {}, [el(8, "text", {}, [txt(9, "gradient stop label")])]),
				el(10, "text", {}, [
					txt(11, "Q3 revenue "),
					el(12, "tspan", {}, [txt(13, "$1.2M")]),
				]),
				el(14, "path", { d: "M0 0L10 10" }),
			]),
			el(15, "p", {}, [txt(16, "beside the chart")]),
		]),
	);
	expect(md).toContain("Q3 revenue");
	expect(md).toContain("$1.2M");
	expect(md).toContain("beside the chart");
	// Tooltip-only and machinery subtrees present nothing on screen.
	expect(md).not.toContain("Revenue chart");
	expect(md).not.toContain("Quarterly revenue by region");
	expect(md).not.toContain("gradient stop label");
});

test("same-origin iframe content projects in place — the embedded page was on screen", () => {
	const mirror = mirrorOf(
		el(1, "div", {}, [
			el(2, "p", {}, [txt(3, "host page text")]),
			el(4, "iframe", {}),
		]),
	);
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
									tagName: "BODY",
									id: 12,
									attributes: {},
									childNodes: [
										{
											type: NodeType.Element,
											tagName: "BUTTON",
											id: 13,
											attributes: {},
											childNodes: [
												{ type: NodeType.Text, id: 14, textContent: "Pay now" },
											],
										},
									],
								},
							],
						},
					],
				},
			},
		],
	});
	const md = projectMarkdown(mirror, 1);
	expect(md).toContain("host page text");
	expect(md).toContain("[button] Pay now");
});

test("an unrecorded (cross-origin) iframe projects nothing", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "iframe", { src: "https://other.example/embed" }),
			el(3, "p", {}, [txt(4, "around it")]),
		]),
	);
	expect(md).toBe("around it");
});

test("nested lists indent; ordered lists number their items", () => {
	const md = projectNode(
		el(1, "ol", {}, [
			el(2, "li", {}, [txt(3, "first")]),
			el(4, "li", {}, [
				txt(5, "second"),
				el(6, "ul", {}, [
					el(7, "li", {}, [txt(8, "child a")]),
					el(9, "li", {}, [txt(10, "child b")]),
				]),
			]),
		]),
	);
	expect(md).toBe("1. first\n2. second\n  - child a\n  - child b");
});

test("what the user-agent stylesheet does not render is excluded — a closed dialog, a hidden row, a form inside a table", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "dialog", {}, [txt(3, "closed dialog")]),
			el(4, "dialog", { open: "" }, [txt(5, "open dialog")]),
			el(6, "table", {}, [
				el(7, "tr", { hidden: "" }, [el(8, "td", {}, [txt(9, "hidden row")])]),
				el(10, "tr", {}, [el(11, "td", {}, [txt(12, "shown row")])]),
				el(13, "form", {}, [txt(14, "stray form")]),
			]),
			el(15, "noscript", {}, [txt(16, "no script")]),
			el(17, "div", { hidden: "until-found" }, [txt(18, "until found")]),
		]),
	);
	expect(md).toBe("open dialog\n\n| shown row |");
});

test("a closed details shows only its summary; open, its body too, with the state", () => {
	const closed = projectNode(
		el(1, "details", {}, [
			el(2, "summary", {}, [txt(3, "Shipping")]),
			el(4, "p", {}, [txt(5, "Free over $100")]),
		]),
	);
	expect(closed).toBe("Shipping");
	const open = projectNode(
		el(1, "details", { open: "" }, [
			el(2, "summary", {}, [txt(3, "Shipping")]),
			el(4, "p", {}, [txt(5, "Free over $100")]),
		]),
	);
	expect(open).toBe("Shipping {open}\n\nFree over $100");
});

test("block boundaries come from the whole default-display set, not a hand list", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "dl", {}, [
				el(3, "dt", {}, [txt(4, "Term")]),
				el(5, "dd", {}, [txt(6, "Definition")]),
			]),
			el(7, "fieldset", {}, [
				el(8, "legend", {}, [txt(9, "Legend")]),
				txt(10, "field"),
			]),
			el(11, "search", {}, [txt(12, "search box")]),
			el(13, "address", {}, [txt(14, "1 Main St")]),
		]),
	);
	expect(md).toBe(
		"Term\n\nDefinition\n\nLegend\nfield\n\nsearch box\n\n1 Main St",
	);
});

test("grouping survives inside a cell, a link, a list item, and a button", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "table", {}, [
				el(3, "tr", {}, [
					el(4, "td", {}, [txt(5, "Teacher")]),
					el(6, "td", {}, [
						el(7, "table", {}, [
							el(8, "tr", {}, [
								el(9, "td", {}, [txt(10, "Course A")]),
								el(11, "td", {}, [txt(12, "8")]),
							]),
							el(13, "tr", {}, [
								el(14, "td", {}, [txt(15, "Course B")]),
								el(16, "td", {}, [txt(17, "7")]),
							]),
						]),
					]),
				]),
			]),
			el(18, "a", { href: "https://x.test/p" }, [
				el(19, "div", {}, [txt(20, "Petra Dress")]),
				el(21, "div", {}, [txt(22, "$595")]),
			]),
			el(23, "ul", {}, [
				el(24, "li", {}, [
					el(25, "p", {}, [txt(26, "first paragraph")]),
					el(27, "p", {}, [txt(28, "second paragraph")]),
				]),
			]),
			el(29, "button", {}, [
				el(30, "div", {}, [txt(31, "Light")]),
				el(32, "div", {}, [txt(33, "recommended")]),
			]),
		]),
	);
	expect(md).toBe(
		[
			"| Teacher | |",
			"  | Course A | 8 |",
			"  | Course B | 7 |",
			"",
			"[Petra Dress](https://x.test/p)",
			"  [$595](https://x.test/p)",
			"",
			"- first paragraph",
			"  second paragraph",
			"",
			"[button] Light",
			"  recommended",
		].join("\n"),
	);
});

test("a textless control is named by its accessible name, in the accname order, never as visible text", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "span", { id: "cart-lbl" }, [txt(3, "Cart")]),
			el(4, "button", {
				"aria-labelledby": "cart-lbl",
				"aria-label": "ignored",
			}),
			el(5, "button", { "aria-label": "Close navigation" }),
			el(6, "button", {}, [
				el(7, "svg", {}, [el(8, "title", {}, [txt(9, "Search")])]),
			]),
			el(10, "button", { title: "Open menu" }),
			el(11, "a", {
				href: "https://x.test/ig",
				"aria-label": "Visit our Instagram",
			}),
			el(12, "button", { "aria-label": "not needed" }, [txt(13, "Save")]),
			el(14, "input", { type: "text", "aria-label": "Search products" }),
		]),
	);
	expect(md).toBe(
		'Cart [button label="Cart"] [button label="Close navigation"] [button label="Search"] [button label="Open menu"] [label="Visit our Instagram"](https://x.test/ig) [button] Save [input type=text label="Search products"]',
	);
});

test("declared state rides the element: ARIA states, widget roles, and the native attributes mapped onto them", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "button", { "aria-pressed": "true" }, [txt(3, "Medium")]),
			el(
				4,
				"a",
				{ href: "https://x.test/t", role: "tab", "aria-selected": "true" },
				[txt(5, "Students")],
			),
			el(6, "button", { disabled: "" }, [txt(7, "Submit")]),
			el(8, "div", { role: "navigation", "aria-hidden": "true" }, [
				txt(9, "decor"),
			]),
			el(10, "select", {}, [
				el(11, "option", {}, [txt(12, "One")]),
				el(13, "option", { selected: "" }, [txt(14, "Two")]),
			]),
			el(15, "p", { "aria-busy": "true" }, [txt(16, "loading list")]),
		]),
	);
	expect(md).toBe(
		[
			"[button] Medium {aria-pressed=true} [Students](https://x.test/t) {role=tab aria-selected=true} [button] Submit {disabled}",
			"decor",
			"[select]",
			"- One",
			"- Two {selected}",
			"",
			"loading list {aria-busy=true}",
		].join("\n"),
	);
});

test("inline meaning from the user-agent stylesheet: struck, monospace, highlighted, quoted, and preserved whitespace", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "p", {}, [
				el(3, "s", {}, [txt(4, "$645")]),
				txt(5, " $387 "),
				el(6, "code", {}, [txt(7, "npm i")]),
				txt(8, " "),
				el(9, "mark", {}, [txt(10, "new")]),
				txt(11, " "),
				el(12, "q", {}, [txt(13, "hello")]),
				txt(14, " "),
				el(15, "del", {}, [txt(16, "old")]),
				el(17, "ins", {}, [txt(18, "new")]),
			]),
			el(19, "pre", {}, [txt(20, "line 1\n  line 2")]),
		]),
	);
	expect(md).toBe(
		"~~$645~~ $387 `npm i` ==new== “hello” ~~old~~new\n\n```\nline 1\n  line 2\n```",
	);
});

test("aria-hidden content still renders — it declares exposure to assistive technology, not presentation", () => {
	const md = projectNode(
		el(1, "div", {}, [
			el(2, "span", { "aria-hidden": "true" }, [txt(3, "★")]),
			el(4, "span", {}, [txt(5, " rated")]),
		]),
	);
	expect(md).toBe("★ rated");
});
