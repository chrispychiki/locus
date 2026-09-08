import { expect, test } from "bun:test";
import { NodeType } from "./rrweb_constants.js";
import { parsedSelectorCount, uaStyle } from "./ua_style.js";

const el = (tagName, attributes = {}, childNodes = []) => {
	const node = { type: NodeType.Element, tagName, attributes, childNodes };
	for (const c of childNodes) c.parentNode = node;
	return node;
};

test("every selector the user-agent stylesheet uses parses", () => {
	expect(parsedSelectorCount).toBeGreaterThan(150);
});

test("display comes from the spec: blocks, list items, table parts, and what is not rendered", () => {
	expect(uaStyle(el("search")).display).toBe("block");
	expect(uaStyle(el("summary")).display).toBe("block");
	expect(uaStyle(el("li")).display).toBe("list-item");
	expect(uaStyle(el("td")).display).toBe("table-cell");
	expect(uaStyle(el("span")).display).toBeUndefined();
	expect(uaStyle(el("template")).display).toBe("none");
	expect(uaStyle(el("dialog")).display).toBe("none");
	expect(uaStyle(el("dialog", { open: "" })).display).toBe("block");
	expect(uaStyle(el("input", { type: "HIDDEN" })).display).toBe("none");
	expect(uaStyle(el("div", { hidden: "" })).display).toBe("none");
	expect(uaStyle(el("div", { hidden: "until-found" })).contentVisibility).toBe(
		"hidden",
	);
});

test("combinators and functional pseudo-classes resolve against the tree", () => {
	const table = el("table", {}, [el("form"), el("tr", { hidden: "" })]);
	expect(uaStyle(table.childNodes[0]).display).toBe("none");
	expect(uaStyle(table.childNodes[1]).visibility).toBe("collapse");
	const details = el("details", {}, [el("summary"), el("summary")]);
	expect(uaStyle(details.childNodes[0]).display).toBe("list-item");
	expect(uaStyle(details.childNodes[1]).display).toBe("block");
});

test("a pseudo-class the DOM cannot answer leaves its selector inert, so a popover stays rendered", () => {
	expect(uaStyle(el("div", { popover: "" })).display).toBe("block");
});

test("the style attribute is the author origin and beats the user-agent rule", () => {
	expect(uaStyle(el("div", { style: "display: none" })).display).toBe("none");
	expect(uaStyle(el("span", { style: "visibility:hidden" })).visibility).toBe(
		"hidden",
	);
	expect(uaStyle(el("dialog", { style: "display: block" })).display).toBe(
		"block",
	);
});

test("inline meaning and whitespace come from the same sheet", () => {
	expect(uaStyle(el("s")).textDecoration).toBe("line-through");
	expect(uaStyle(el("strong")).fontWeight).toBe("bolder");
	expect(uaStyle(el("cite")).fontStyle).toBe("italic");
	expect(uaStyle(el("kbd")).fontFamily).toBe("monospace");
	expect(uaStyle(el("pre")).whiteSpace).toBe("pre");
	expect(uaStyle(el("mark")).background).toBe("yellow");
	expect(uaStyle(el("q")).before).toBe("open-quote");
});
