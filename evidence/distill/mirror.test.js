// The mirror's contract is the vendored replayer's own reconstruction behavior under the screenshot renderer's
// configuration (useVirtualDom) — snapshot builds per buildNodeWithSN, mutations per rrdom, failure modes
// included. Every case here is transcribed from that source, not from what any recording happened to contain.
import { expect, test } from "bun:test";
import { LightweightMirror } from "./mirror.js";
import { EventType, IncrementalSource, NodeType } from "./rrweb_constants.js";

const el = (id, tagName, attributes = {}, childNodes = [], extra = {}) => ({
	type: NodeType.Element,
	id,
	tagName,
	attributes,
	childNodes,
	...extra,
});
const txt = (id, textContent, extra = {}) => ({
	type: NodeType.Text,
	id,
	textContent,
	...extra,
});
const doc = (id, kids, extra = {}) => ({
	type: NodeType.Document,
	id,
	childNodes: kids,
	...extra,
});
const doctype = (id, name = "html", publicId = "", systemId = "") => ({
	type: NodeType.DocumentType,
	id,
	name,
	publicId,
	systemId,
});

const MUT = (d) => ({ adds: [], removes: [], texts: [], attributes: [], ...d });

function snapshot(bodyKids) {
	return doc(1, [
		el(2, "html", {}, [el(3, "head", {}), el(4, "body", {}, bodyKids)]),
	]);
}

function mirrorWith(bodyKids) {
	const m = new LightweightMirror();
	m.addNode(snapshot(bodyKids));
	return m;
}

const tags = (node) => node.childNodes.map((c) => c.tagName ?? "#text");

test("builds the tree with id addressability and parent/child links", () => {
	const m = mirrorWith([el(10, "div", { class: "a" }, [txt(11, "hi")])]);
	const div = m.getNode(10);
	expect(div.tagName).toBe("div");
	expect(div.attributes.class).toBe("a");
	expect(div.parentNode).toBe(m.getNode(4));
	expect(div.childNodes[0]).toBe(m.getNode(11));
	expect(m.getNode(1).type).toBe(NodeType.Document);
	expect(m.rootDoc).toBe(m.getNode(1));
});

test("decodes serialized attributes: true → empty, null dropped, unselected option dropped, non-strings dropped", () => {
	const m = mirrorWith([
		el(10, "input", { disabled: true, gone: null, "data-n": 42 }),
		el(11, "select", {}, [
			el(12, "option", { selected: false }),
			el(13, "option", { selected: true }),
		]),
	]);
	expect(m.getNode(10).attributes).toEqual({ disabled: "" });
	expect(m.getNode(12).attributes).toEqual({});
	expect(m.getNode(13).attributes).toEqual({ selected: "" });
});

test("textarea value materializes as its only text child; serialized children are dropped", () => {
	const m = mirrorWith([
		el(10, "textarea", { value: "draft" }, [txt(11, "stale serialized child")]),
	]);
	const ta = m.getNode(10);
	expect(ta.attributes.value).toBeUndefined();
	expect(ta.childNodes.length).toBe(1);
	expect(ta.childNodes[0].textContent).toBe("draft");
	expect(m.getNode(11)).toBeNull();
});

test("style _cssText materializes as one unserialized text child, recorded CSS verbatim", () => {
	const css =
		".a:hover { color: red; } @media (max-device-width: 300px) { .b { top: 0; } }";
	const m = mirrorWith([el(10, "style", { _cssText: css })]);
	const style = m.getNode(10);
	expect(style.childNodes.length).toBe(1);
	expect(style.childNodes[0].textContent).toBe(css);
	expect(style.childNodes[0].id).toBeNull();
	expect(style.attributes._cssText).toBeUndefined();
});

test("style _cssText re-splits across serialized text children on rr_split markers", () => {
	const m = mirrorWith([
		el(10, "style", { _cssText: ".a{}/* rr_split */.b{}/* rr_split */.c{}" }, [
			txt(11, ""),
			txt(12, ""),
		]),
	]);
	const kids = m.getNode(10).childNodes;
	expect(kids.length).toBe(2);
	expect(kids[0].textContent).toBe(".a{}");
	expect(kids[1].textContent).toBe(".b{}.c{}");
});

test("a link carrying _cssText materializes as a style element", () => {
	const m = mirrorWith([
		el(10, "link", { rel: "stylesheet", href: "/x.css", _cssText: ".a{}" }),
	]);
	const node = m.getNode(10);
	expect(node.tagName).toBe("style");
	expect(node.childNodes[0].textContent).toBe(".a{}");
});

test("BackCompat documents get a quirks doctype injected; standards documents do not", () => {
	const quirks = new LightweightMirror();
	quirks.addNode(doc(1, [el(2, "html", {}, [])], { compatMode: "BackCompat" }));
	expect(quirks.getNode(1).childNodes[0].type).toBe(NodeType.DocumentType);
	expect(quirks.getNode(1).childNodes[0].publicId).toContain(
		"HTML 4.0 Transitional",
	);

	const std = new LightweightMirror();
	std.addNode(doc(1, [el(2, "html", {}, [])]));
	expect(std.getNode(1).childNodes[0].type).toBe(NodeType.Element);

	const declared = new LightweightMirror();
	declared.addNode(
		doc(1, [doctype(9), el(2, "html", {}, [])], { compatMode: "BackCompat" }),
	);
	expect(
		declared
			.getNode(1)
			.childNodes.filter((c) => c.type === NodeType.DocumentType).length,
	).toBe(1);

	const xhtml = new LightweightMirror();
	xhtml.addNode(
		doc(1, [el(2, "html", { xmlns: "http://www.w3.org/1999/xhtml" }, [])], {
			compatMode: "BackCompat",
		}),
	);
	expect(xhtml.getNode(1).childNodes[0].publicId).toBe(
		"-//W3C//DTD XHTML 1.0 Transitional//EN",
	);
});

test("body is forced to be html's last child even when the snapshot serialized trailing siblings", () => {
	const m = new LightweightMirror();
	m.addNode(
		doc(1, [
			el(2, "html", {}, [
				el(3, "head", {}),
				el(4, "body", {}, []),
				el(5, "extension-junk", {}),
			]),
		]),
	);
	expect(tags(m.getNode(2))).toEqual(["head", "extension-junk", "body"]);
});

test("shadow children build under the host's shadow root, off the light tree", () => {
	const m = mirrorWith([
		el(
			10,
			"my-widget",
			{},
			[
				el(11, "span", {}, [txt(12, "shadow text")], { isShadow: true }),
				el(13, "span", {}, [txt(14, "light text")]),
			],
			{ isShadowHost: true },
		),
	]);
	const host = m.getNode(10);
	expect(tags(host)).toEqual(["span"]);
	expect(host.childNodes[0]).toBe(m.getNode(13));
	expect(host.shadowRoot.childNodes[0]).toBe(m.getNode(11));
	expect(m.getNode(11).parentNode).toBe(host.shadowRoot);
});

test("SVG tag case is restored from the serialized lowercase form", () => {
	const m = mirrorWith([
		el(10, "svg", {}, [el(11, "clippath", {}, [], { isSVG: true })], {
			isSVG: true,
		}),
	]);
	expect(m.getNode(11).tagName).toBe("clipPath");
});

test("snapshot-built img with rr_dataURL swaps src to the recorded bytes", () => {
	const m = mirrorWith([
		el(10, "img", {
			src: "https://x.test/a.png",
			rr_dataURL: "data:image/png;base64,AA",
		}),
	]);
	expect(m.getNode(10).attributes.src).toBe("data:image/png;base64,AA");
	expect(m.getNode(10).attributes["rrweb-original-src"]).toBe(
		"https://x.test/a.png",
	);
});

test("img with srcset and rr_dataURL keeps only the replayer's surviving attributes", () => {
	const m = mirrorWith([
		el(10, "img", {
			src: "https://x.test/a.png",
			srcset: "a 1x, b 2x",
			alt: "gone",
			rr_dataURL: "data:image/png;base64,AA",
		}),
	]);
	expect(m.getNode(10).attributes).toEqual({
		"rrweb-original-srcset": "a 1x, b 2x",
		"rrweb-original-src": "https://x.test/a.png",
		src: "data:image/png;base64,AA",
	});
});

test("rr_width/rr_height fold into the style attribute", () => {
	const m = mirrorWith([
		el(10, "div", { rr_width: "100px", rr_height: "50px" }),
	]);
	expect(m.getNode(10).attributes.style).toBe("width: 100px; height: 50px;");
});

test("media state and timing land on props under the replayer's type guards", () => {
	const m = mirrorWith([
		el(10, "video", {
			rr_mediaState: "played",
			rr_mediaCurrentTime: 3.5,
			rr_mediaPlaybackRate: 2,
			rr_mediaMuted: false,
			rr_mediaLoop: true,
			rr_mediaVolume: 0.5,
		}),
		el(11, "audio", { rr_mediaState: "paused", rr_mediaCurrentTime: "3.5" }),
	]);
	const video = m.getNode(10);
	expect(video.props.paused).toBe(false);
	expect(video.props.currentTime).toBe(3.5);
	expect(video.props.playbackRate).toBe(2);
	expect(video.props.muted).toBe(false);
	expect(video.props.volume).toBe(0.5);
	// The replayer decodes serialized `true` to "" before collecting rr_* specials, so a boolean-true
	// media special fails its typeof guard and never applies — transcribed, not corrected.
	expect(video.props.loop).toBeUndefined();
	expect(video.attributes).toEqual({});
	const audio = m.getNode(11);
	expect(audio.props.paused).toBe(true);
	expect(audio.props.currentTime).toBeUndefined();
});

test("canvas rr_dataURL lands on props — bitmap state, not markup", () => {
	const m = mirrorWith([
		el(10, "canvas", { rr_dataURL: "data:image/png;base64,AA", width: "300" }),
	]);
	expect(m.getNode(10).props.rr_dataURL).toBe("data:image/png;base64,AA");
	expect(m.getNode(10).attributes).toEqual({ width: "300" });
});

test("a dialog's rr_open_mode is kept as the attribute the replayer sets", () => {
	const m = mirrorWith([el(10, "dialog", { open: "", rr_open_mode: "modal" })]);
	expect(m.getNode(10).attributes).toEqual({
		open: "",
		rr_open_mode: "modal",
	});
});

test("getTextContent concatenates descendant text", () => {
	const m = mirrorWith([
		el(10, "p", {}, [txt(11, "a "), el(12, "b", {}, [txt(13, "bold")])]),
	]);
	expect(m.getTextContent(10)).toBe("a bold");
});

test("nextId places an added sibling in document order", () => {
	const m = mirrorWith([el(10, "p", {}, [txt(11, "B")])]);
	m.applyMutation(
		MUT({ adds: [{ parentId: 10, nextId: 11, node: txt(12, "A") }] }),
	);
	expect(m.getNode(10).childNodes.map((c) => c.textContent)).toEqual([
		"A",
		"B",
	]);
});

test("adds whose parents arrive later in the same batch resolve out of order", () => {
	const m = mirrorWith([el(10, "div", {})]);
	const report = m.applyMutation(
		MUT({
			adds: [
				{ parentId: 20, nextId: null, node: txt(21, "leaf") },
				{ parentId: 10, nextId: null, node: el(20, "section", {}) },
			],
		}),
	);
	expect(report.droppedAdds).toEqual([]);
	expect(m.getNode(21).parentNode).toBe(m.getNode(20));
	expect(m.getNode(20).parentNode).toBe(m.getNode(10));
});

test("adds whose parent never arrives are dropped and reported", () => {
	const m = mirrorWith([el(10, "div", {})]);
	const report = m.applyMutation(
		MUT({ adds: [{ parentId: 999, nextId: null, node: el(20, "p", {}) }] }),
	);
	expect(report.droppedAdds).toEqual([20]);
	expect(m.getNode(20)).toBeNull();
});

test("an add for a known id with equal meta moves the existing node", () => {
	const m = mirrorWith([
		el(10, "div", {}, [el(11, "span", { k: "v" })]),
		el(12, "aside", {}),
	]);
	const span = m.getNode(11);
	m.applyMutation(
		MUT({
			adds: [{ parentId: 12, nextId: null, node: el(11, "span", { k: "v" }) }],
		}),
	);
	expect(m.getNode(11)).toBe(span);
	expect(span.parentNode).toBe(m.getNode(12));
	expect(m.getNode(10).childNodes).toEqual([]);
});

test("an add for a known id with changed meta displaces it: fresh node mapped, old node still painted", () => {
	const m = mirrorWith([
		el(10, "div", {}, [el(11, "span", { k: "old" })]),
		el(12, "aside", {}),
	]);
	const old = m.getNode(11);
	m.applyMutation(
		MUT({
			adds: [
				{ parentId: 12, nextId: null, node: el(11, "span", { k: "new" }) },
			],
		}),
	);
	expect(m.getNode(11)).not.toBe(old);
	expect(m.getNode(11).attributes.k).toBe("new");
	expect(old.parentNode).toBe(m.getNode(10));
	expect(m.getNode(10).childNodes[0]).toBe(old);
});

test("legacy -1 sibling adds wait in the retry map and resolve against their anchor", () => {
	const m = mirrorWith([el(10, "div", {})]);
	m.applyMutation(
		MUT({
			adds: [
				{ parentId: 10, previousId: -1, nextId: null, node: el(20, "em", {}) },
				{
					parentId: 10,
					previousId: 20,
					nextId: null,
					node: el(21, "strong", {}),
				},
			],
		}),
	);
	expect(tags(m.getNode(10))).toEqual(["em", "strong"]);
	expect(m.legacyMissingNodeRetryMap).toEqual({});
});

test("an add inserted after a previous whose next sibling lives under another parent aborts the mutation", () => {
	const m = mirrorWith([
		el(10, "div", {}, [el(11, "span", {})]),
		el(12, "aside", {}, [el(13, "b", {}), el(14, "i", {})]),
	]);
	const report = m.applyMutation(
		MUT({
			adds: [
				// previous 13 and its next sibling 14 sit under 12; the declared parent is 10 — the replayer calls
				// insertBefore(target, 14) on 10, which throws.
				{ parentId: 10, previousId: 13, nextId: null, node: el(20, "p", {}) },
			],
			texts: [{ id: 11, value: "never applied" }],
		}),
	);
	expect(report.aborted).toContain("10");
	expect(m.getTextContent(11)).toBe("");
});

test("a next that is a deep descendant aborts; a next outside the parent appends", () => {
	const deep = mirrorWith([
		el(10, "div", {}, [el(11, "span", {}, [txt(12, "x")])]),
	]);
	const aborted = deep.applyMutation(
		MUT({ adds: [{ parentId: 10, nextId: 12, node: el(20, "p", {}) }] }),
	);
	expect(aborted.aborted).toContain("10");

	const outside = mirrorWith([
		el(10, "div", {}),
		el(13, "aside", {}, [txt(14, "y")]),
	]);
	const ok = outside.applyMutation(
		MUT({ adds: [{ parentId: 10, nextId: 14, node: el(20, "p", {}) }] }),
	);
	expect(ok.aborted).toBeNull();
	expect(tags(outside.getNode(10))).toEqual(["p"]);
});

test("a mutation-built img carrying rr_dataURL aborts the mutation (replayer TypeError)", () => {
	const m = mirrorWith([el(10, "div", {})]);
	const report = m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: el(20, "img", {
						src: "https://x.test/a.png",
						rr_dataURL: "data:image/png;base64,AA",
					}),
				},
			],
		}),
	);
	expect(report.aborted).toContain("rr_dataURL");
	expect(m.getNode(20)).toBeNull();
});

test("an add whose rootId is unknown is dropped; a rootId resolving to a non-document aborts", () => {
	const m = mirrorWith([el(10, "div", {})]);
	const dropped = m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: { ...el(20, "p", {}), rootId: 999 },
				},
			],
		}),
	);
	expect(dropped.droppedAdds).toEqual([20]);

	const aborted = m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: { ...el(21, "p", {}), rootId: 10 },
				},
			],
		}),
	);
	expect(aborted.aborted).toContain("10");
});

test("a second documentElement is refused (one element per document), html re-adds replace", () => {
	const m = mirrorWith([]);
	const refused = m.applyMutation(
		MUT({ adds: [{ parentId: 1, nextId: null, node: el(20, "div", {}) }] }),
	);
	expect(refused.aborted).toContain("documentElement");

	const readd = m.applyMutation(
		MUT({
			adds: [{ parentId: 1, nextId: null, node: el(30, "html", {}, []) }],
		}),
	);
	expect(readd.aborted).toBeNull();
	expect(
		m.getNode(1).childNodes.filter((c) => c.type === NodeType.Element).length,
	).toBe(1);
	expect(m.getNode(1).childNodes.find((c) => c.type === NodeType.Element)).toBe(
		m.getNode(30),
	);
	expect(m.getNode(2)).not.toBeNull();
});

test("an incoming doctype replaces a leading doctype; without one to yield, a second doctype is refused", () => {
	const m = new LightweightMirror();
	m.addNode(doc(1, [doctype(9), el(2, "html", {}, [])]));
	const old = m.getNode(9);
	const replaced = m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 1,
					nextId: 2,
					node: doctype(30, "html", "-//W3C//DTD HTML 4.01//EN"),
				},
			],
		}),
	);
	expect(replaced.aborted).toBeNull();
	expect(m.getNode(1).childNodes[0]).toBe(m.getNode(30));
	// The displaced doctype detaches but keeps its id, exactly as the replayer's map does.
	expect(m.getNode(9)).toBe(old);
	expect(old.parentNode).toBeNull();

	const bare = new LightweightMirror();
	bare.addNode(doc(1, [el(2, "html", {}, [])]));
	const trailing = bare.applyMutation(
		MUT({ adds: [{ parentId: 1, nextId: null, node: doctype(30) }] }),
	);
	// No leading doctype to replace: the add appends after <html>, as the replayer's appendChild does.
	expect(trailing.aborted).toBeNull();
	expect(bare.getNode(1).childNodes[1]).toBe(bare.getNode(30));
	const refused = bare.applyMutation(
		MUT({ adds: [{ parentId: 1, nextId: null, node: doctype(31) }] }),
	);
	expect(refused.aborted).toContain("doctype");
});

test("legacy -1 anchors park across mutations and resolve as the named next sibling", () => {
	const m = mirrorWith([el(10, "div", {})]);
	m.applyMutation(
		MUT({
			adds: [
				{ parentId: 10, previousId: null, nextId: -1, node: el(20, "em", {}) },
			],
		}),
	);
	expect(m.getNode(20)).not.toBeNull();
	expect(m.getNode(20).parentNode).toBeNull();
	expect(m.legacyMissingNodeRetryMap[20]).toBeDefined();
	m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					previousId: null,
					nextId: 20,
					node: el(21, "strong", {}),
				},
			],
		}),
	);
	expect(tags(m.getNode(10))).toEqual(["strong", "em"]);
	expect(m.legacyMissingNodeRetryMap).toEqual({});
});

test("an add whose named next sibling never arrives is dropped and reported", () => {
	const m = mirrorWith([el(10, "div", {})]);
	const report = m.applyMutation(
		MUT({ adds: [{ parentId: 10, nextId: 999, node: el(20, "p", {}) }] }),
	);
	expect(report.droppedAdds).toEqual([20]);
	expect(m.getNode(20)).toBeNull();
	expect(m.getNode(10).childNodes).toEqual([]);
});

test("isShadow adds attach under the host's shadow root, creating it on demand", () => {
	const m = mirrorWith([el(10, "my-widget", {})]);
	m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: { ...txt(20, "shadow text"), isShadow: true },
				},
			],
		}),
	);
	const host = m.getNode(10);
	expect(host.childNodes).toEqual([]);
	expect(host.shadowRoot.childNodes[0]).toBe(m.getNode(20));
});

test("a text added to a style holding its materialized sheet inherits the sheet's text", () => {
	const m = mirrorWith([el(10, "style", { _cssText: ".a{}" })]);
	m.applyMutation(
		MUT({ adds: [{ parentId: 10, nextId: null, node: txt(20, "") }] }),
	);
	const style = m.getNode(10);
	expect(style.childNodes.length).toBe(1);
	expect(style.childNodes[0]).toBe(m.getNode(20));
	expect(m.getNode(20).textContent).toBe(".a{}");
});

test("a text added to a textarea clears its previous text children first", () => {
	const m = mirrorWith([el(10, "textarea", { value: "old" })]);
	m.applyMutation(
		MUT({ adds: [{ parentId: 10, nextId: null, node: txt(20, "new") }] }),
	);
	expect(m.getNode(10).childNodes.length).toBe(1);
	expect(m.getTextContent(10)).toBe("new");
});

test("a remove detaches the subtree and purges its ids", () => {
	const m = mirrorWith([
		el(10, "div", {}, [el(11, "span", {}, [txt(12, "x")])]),
	]);
	const report = m.applyMutation(MUT({ removes: [{ id: 11, parentId: 10 }] }));
	expect(report.aborted).toBeNull();
	expect(m.getNode(10).childNodes).toEqual([]);
	expect(m.getNode(11)).toBeNull();
	expect(m.getNode(12)).toBeNull();
});

test("a remove of an unknown node is skipped and reported", () => {
	const m = mirrorWith([el(10, "div", {})]);
	const report = m.applyMutation(MUT({ removes: [{ id: 999, parentId: 10 }] }));
	expect(report.aborted).toBeNull();
	expect(report.missingRemoves).toEqual([999]);
});

test("a remove under the wrong declared parent aborts after unmapping: subtree stays painted, unaddressable", () => {
	const m = mirrorWith([
		el(10, "div", {}, [el(11, "span", {}, [txt(12, "x")])]),
		el(13, "aside", {}),
	]);
	const span = m.getNode(11);
	const report = m.applyMutation(
		MUT({
			removes: [{ id: 11, parentId: 13 }],
			texts: [{ id: 12, value: "never applied" }],
		}),
	);
	expect(report.aborted).toContain("13");
	expect(m.getNode(11)).toBeNull();
	expect(m.getNode(12)).toBeNull();
	expect(m.getNode(10).childNodes[0]).toBe(span);
	expect(span.childNodes[0].textContent).toBe("x");
});

test("an isShadow remove routes through the shadow root", () => {
	const m = mirrorWith([
		el(10, "my-widget", {}, [el(11, "span", {}, [], { isShadow: true })], {
			isShadowHost: true,
		}),
	]);
	const report = m.applyMutation(
		MUT({ removes: [{ id: 11, parentId: 10, isShadow: true }] }),
	);
	expect(report.aborted).toBeNull();
	expect(m.getNode(10).shadowRoot.childNodes).toEqual([]);
	expect(m.getNode(11)).toBeNull();
});

test("shadow subtrees keep their ids when their host is removed, as the replayer's map does", () => {
	const m = mirrorWith([
		el(10, "my-widget", {}, [el(11, "span", {}, [], { isShadow: true })], {
			isShadowHost: true,
		}),
	]);
	m.applyMutation(MUT({ removes: [{ id: 10, parentId: 4 }] }));
	expect(m.getNode(10)).toBeNull();
	expect(m.getNode(11)).not.toBeNull();
});

test("a text mutation on an element replaces its children with one text node, empty value included", () => {
	const m = mirrorWith([
		el(10, "div", {}, [el(11, "span", {}), txt(12, "old")]),
	]);
	m.applyMutation(MUT({ texts: [{ id: 10, value: "" }] }));
	const div = m.getNode(10);
	expect(div.childNodes.length).toBe(1);
	expect(div.childNodes[0].type).toBe(NodeType.Text);
	expect(div.childNodes[0].textContent).toBe("");
	m.applyMutation(
		MUT({ adds: [{ parentId: 10, nextId: null, node: el(20, "b", {}) }] }),
	);
	expect(div.childNodes.length).toBe(2);
});

test("only the last text mutation per node in a batch applies", () => {
	const m = mirrorWith([txt(10, "start")]);
	m.applyMutation(
		MUT({
			texts: [
				{ id: 10, value: "first" },
				{ id: 10, value: "last" },
			],
		}),
	);
	expect(m.getTextContent(10)).toBe("last");
});

test("text/attribute mutations on nodes removed in the same batch stay silent; unknown ids are reported", () => {
	const m = mirrorWith([el(10, "div", {}, [txt(11, "x")])]);
	const report = m.applyMutation(
		MUT({
			removes: [{ id: 11, parentId: 10 }],
			texts: [
				{ id: 11, value: "gone" },
				{ id: 998, value: "?" },
			],
			attributes: [
				{ id: 11, attributes: { k: "v" } },
				{ id: 999, attributes: { k: "v" } },
			],
		}),
	);
	expect(report.missingTexts).toEqual([998]);
	expect(report.missingAttributes).toEqual([999]);
});

test("attribute mutations set, null-remove, and apply style diffs with the replayer's canonical serialization", () => {
	const m = mirrorWith([
		el(10, "div", { style: "color: red; top: 1px;", title: "t" }),
	]);
	m.applyMutation(
		MUT({
			attributes: [
				{
					id: 10,
					attributes: {
						title: null,
						"data-k": "v",
						style: {
							color: "blue",
							top: false,
							"--x": ["1px", "important"],
							badName: "ignored",
						},
					},
				},
			],
		}),
	);
	const attrs = m.getNode(10).attributes;
	expect(attrs.title).toBeUndefined();
	expect(attrs["data-k"]).toBe("v");
	expect(attrs.style).toBe("color: blue; --x: 1px !important;");
});

test("every camelCase style name in one diff is skipped, as the replayer skips each — the check is stateless across consecutive names", () => {
	const m = mirrorWith([el(10, "div", { style: "color: red;" })]);
	m.applyMutation(
		MUT({
			attributes: [
				{
					id: 10,
					attributes: {
						style: { badOne: "1px", badTwo: "2px", top: "3px" },
					},
				},
			],
		}),
	);
	expect(m.getNode(10).attributes.style).toBe("color: red; top: 3px;");
});

test("a value attribute mutation on a textarea materializes as its text child", () => {
	const m = mirrorWith([el(10, "textarea", {}, [txt(11, "old")])]);
	m.applyMutation(
		MUT({ attributes: [{ id: 10, attributes: { value: "new" } }] }),
	);
	expect(m.getNode(10).childNodes.length).toBe(1);
	expect(m.getTextContent(10)).toBe("new");
});

test("a _cssText attribute mutation rebuilds the node in place: link becomes style, id remaps, batch folds in", () => {
	const m = mirrorWith([
		el(9, "i", {}),
		el(10, "link", { rel: "stylesheet", href: "/x.css" }),
		el(11, "i", {}),
	]);
	m.applyMutation(
		MUT({
			attributes: [
				{ id: 10, attributes: { _cssText: ".a{}", media: "screen" } },
			],
		}),
	);
	const rebuilt = m.getNode(10);
	expect(rebuilt.tagName).toBe("style");
	expect(rebuilt.childNodes[0].textContent).toBe(".a{}");
	expect(rebuilt.attributes.media).toBe("screen");
	expect(tags(m.getNode(4))).toEqual(["i", "style", "i"]);
});

test("a _cssText mutation on a detached node remaps the id to the rebuilt style but leaves the tree alone", () => {
	const m = mirrorWith([el(10, "div", {})]);
	m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					previousId: -1,
					nextId: null,
					node: el(20, "link", { rel: "stylesheet", href: "/x.css" }),
				},
			],
		}),
	);
	const parked = m.getNode(20);
	expect(parked.parentNode).toBeNull();
	const report = m.applyMutation(
		MUT({ attributes: [{ id: 20, attributes: { _cssText: ".a{}" } }] }),
	);
	expect(report.aborted).toBeNull();
	const rebuilt = m.getNode(20);
	expect(rebuilt).not.toBe(parked);
	expect(rebuilt.tagName).toBe("style");
	expect(rebuilt.childNodes[0].textContent).toBe(".a{}");
	expect(rebuilt.attributes._cssText).toBeUndefined();
	expect(rebuilt.parentNode).toBeNull();
	// The replayer's fallthrough then sets the attribute plainly on the node it held.
	expect(parked.tagName).toBe("link");
	expect(parked.attributes._cssText).toBe(".a{}");
	expect(m.getNode(10).childNodes).toEqual([]);
});

test("a document added under an iframe becomes its content document, addressable with its subtree", () => {
	const m = mirrorWith([el(10, "iframe", {})]);
	const report = m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: doc(50, [
						el(51, "html", {}, [el(52, "body", {}, [txt(53, "inner")])]),
					]),
				},
			],
		}),
	);
	expect(report.aborted).toBeNull();
	expect(m.getNode(50)).toBe(m.getNode(10).contentDocument);
	expect(m.getTextContent(52)).toBe("inner");
	expect(m.getNode(10).childNodes).toEqual([]);
});

test("a document arriving before its iframe waits in the new-document queue across mutations", () => {
	const m = mirrorWith([]);
	m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: doc(50, [
						el(51, "html", {}, [el(52, "body", {}, [txt(53, "inner")])]),
					]),
				},
			],
		}),
	);
	expect(m.getNode(50)).toBeNull();
	m.applyMutation(
		MUT({ adds: [{ parentId: 4, nextId: null, node: el(10, "iframe", {}) }] }),
	);
	expect(m.getNode(50)).toBe(m.getNode(10).contentDocument);
	expect(m.getTextContent(53)).toBe("inner");
});

test("re-attaching a document to an iframe clears the old content; the old children stay addressable", () => {
	const m = mirrorWith([el(10, "iframe", {})]);
	m.applyMutation(
		MUT({
			adds: [
				{ parentId: 10, nextId: null, node: doc(50, [el(51, "html", {}, [])]) },
			],
		}),
	);
	m.applyMutation(
		MUT({
			adds: [
				{ parentId: 10, nextId: null, node: doc(60, [el(61, "html", {}, [])]) },
			],
		}),
	);
	const content = m.getNode(10).contentDocument;
	expect(m.getNode(60)).toBe(content);
	expect(m.getNode(50)).toBe(content);
	expect(content.childNodes[0]).toBe(m.getNode(61));
	expect(m.getNode(51)).not.toBeNull();
	expect(m.getNode(51).parentNode).toBeNull();
});

test("input events land on props (last wins) and never touch attributes", () => {
	const m = mirrorWith([el(10, "input", { type: "text", value: "initial" })]);
	m.applyInput({ id: 10, text: "typed", isChecked: false });
	m.applyInput({ id: 10, text: "typed more", isChecked: false });
	const node = m.getNode(10);
	expect(node.props.value).toBe("typed more");
	expect(node.attributes.value).toBe("initial");
	const size = m.size;
	m.applyInput({ id: -1, text: "x", isChecked: false });
	m.applyInput({ id: 999, text: "x", isChecked: false });
	expect(m.size).toBe(size);
});

test("scroll events land on props", () => {
	const m = mirrorWith([el(10, "div", {})]);
	m.applyScroll({ id: 10, x: 3, y: 400 });
	expect(m.getNode(10).props.scrollTop).toBe(400);
});

test("applyEvent steps raw events: FullSnapshot resets, mutations report, others no-op", () => {
	const m = new LightweightMirror();
	m.applyEvent({
		type: EventType.FullSnapshot,
		timestamp: 1,
		data: { node: snapshot([txt(10, "one")]) },
	});
	expect(m.getTextContent(10)).toBe("one");
	const report = m.applyEvent({
		type: EventType.IncrementalSnapshot,
		timestamp: 2,
		data: {
			source: IncrementalSource.Mutation,
			...MUT({ texts: [{ id: 10, value: "two" }] }),
		},
	});
	expect(report.aborted).toBeNull();
	expect(m.getTextContent(10)).toBe("two");
	expect(
		m.applyEvent({
			type: EventType.Meta,
			timestamp: 3,
			data: { href: "https://x.test" },
		}),
	).toBeNull();
	m.applyEvent({
		type: EventType.FullSnapshot,
		timestamp: 4,
		data: { node: snapshot([txt(10, "fresh")]) },
	});
	expect(m.getTextContent(10)).toBe("fresh");
});

// The report is the whole of what the rescue gate sees. An add that builds nothing — a node type
// outside the serialization format — must be reported: dropped silently, the node would be missing
// from the mirror AND missing from the report, and the slice it belonged to could be certified
// continuous while carrying content that was never applied.
test("an add of a node type the format does not define is dropped and reported", () => {
	// Derived from the format itself, so it is outside it by construction — and stays outside it the
	// day the format grows a type.
	const undefinedType = Math.max(...Object.values(NodeType)) + 1;
	const m = mirrorWith([el(10, "div", {})]);
	const report = m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: { type: undefinedType, id: 20, childNodes: [] },
				},
			],
		}),
	);
	expect(report.droppedAdds).toEqual([20]);
	expect(m.getNode(20)).toBeNull();
});

// The projection walks downward into an iframe's content document in place; a visibility walk needs
// the way back up — the same host inverse a shadow root carries.
test("an iframe's content document links back to its host element", () => {
	const m = mirrorWith([el(10, "iframe", {})]);
	const host = m.getNode(10);
	expect(host.contentDocument.host).toBe(host);
	m.applyMutation(
		MUT({
			adds: [
				{
					parentId: 10,
					nextId: null,
					node: {
						type: NodeType.Document,
						id: 20,
						childNodes: [el(21, "html", {}, [txt(22, "inside")])],
					},
				},
			],
		}),
	);
	// Attachment re-registers the same document under the incoming id — the backlink survives.
	expect(m.getNode(20)).toBe(host.contentDocument);
	expect(m.getNode(20).host).toBe(host);
});
