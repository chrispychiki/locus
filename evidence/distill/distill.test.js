// The orchestration above the leaves: what the slice boundary resets, when a mutation run flushes its diff,
// what page_url carries, and that the residue every consumer reads is the sum of what actually happened.
import { expect, test } from "bun:test";
import { deflateSync } from "node:zlib";
import {
	addResidue,
	distillVisitor,
	newResidue,
	SliceDistiller,
} from "./distill.js";
import {
	EventType,
	IncrementalSource,
	MouseInteractions,
	NodeType,
} from "./rrweb_constants.js";

const snapshot = (bodyText, id = 1) => ({
	type: EventType.FullSnapshot,
	timestamp: 1000,
	data: {
		node: {
			type: NodeType.Document,
			id,
			childNodes: [
				{
					type: NodeType.Element,
					tagName: "HTML",
					id: id + 1,
					attributes: {},
					childNodes: [
						{
							type: NodeType.Element,
							tagName: "BODY",
							id: id + 2,
							attributes: {},
							childNodes: [
								{
									type: NodeType.Element,
									tagName: "P",
									id: id + 3,
									attributes: {},
									childNodes: [
										{ type: NodeType.Text, id: id + 4, textContent: bodyText },
									],
								},
							],
						},
					],
				},
			],
		},
	},
});

const textMutation = (id, text) => ({
	type: EventType.IncrementalSnapshot,
	timestamp: 2000,
	data: {
		source: IncrementalSource.Mutation,
		adds: [],
		removes: [],
		attributes: [],
		texts: [{ id, value: text }],
	},
});

const click = (id) => ({
	type: EventType.IncrementalSnapshot,
	timestamp: 3000,
	data: {
		source: IncrementalSource.MouseInteraction,
		type: MouseInteractions.Click,
		id,
		x: 5,
		y: 6,
	},
});

const meta = (url) => ({
	type: EventType.Meta,
	timestamp: 900,
	data: { href: url, width: 800, height: 600 },
});

const rows = (events, sliceId) =>
	events.map((e, i) => ({
		id: i + 1,
		slice_id: sliceId,
		raw_json: deflateSync(JSON.stringify(e)),
	}));

test("a FullSnapshot carries the projection; the mutation run carries the diff on its last event", () => {
	const events = [
		snapshot("before"),
		textMutation(5, "after"),
		textMutation(5, "after again"),
		click(4),
	];
	const { updates } = distillVisitor(rows(events, "s1"));

	expect(updates[0].md).toContain("before");
	expect(updates[0].diff).toBe(null);
	// Mid-run events carry nothing — the run collapses to one delta, on the event that closed it.
	expect(updates[1].diff).toBe(null);
	expect(updates[2].diff).toContain("- before");
	expect(updates[2].diff).toContain("+ after again");
	expect(updates[3].diff).toBe(null);
});

test("an unterminated mutation run still flushes at the end of the visitor's events", () => {
	const { updates } = distillVisitor(
		rows([snapshot("before"), textMutation(5, "after")], "s1"),
	);
	expect(updates[1].diff).toContain("+ after");
});

test("the mirror resets at the slice boundary — no DOM state bleeds across slices", () => {
	const first = rows([snapshot("first page"), click(4)], "s1");
	const second = rows([snapshot("second page", 100), click(103)], "s2");
	second.forEach((r, i) => {
		r.id = 10 + i;
	});

	const { updates } = distillVisitor([...first, ...second]);

	// The second slice's click resolves against its own snapshot's ids...
	expect(updates[3].cols.tag).toBe("p");
	// ...and its projection is a fresh document, never a diff against the previous slice's.
	expect(updates[2].md).toContain("second page");
	expect(updates[2].md).not.toContain("first page");
});

test("a node id the new slice never introduced does not resolve from the old slice's DOM", () => {
	const first = rows([snapshot("first page")], "s1");
	const second = rows([snapshot("second page", 100), click(4)], "s2");
	second.forEach((r, i) => {
		r.id = 10 + i;
	});

	const { updates } = distillVisitor([...first, ...second]);
	expect(updates.at(-1).cols.tag).toBe(null);
});

// Every slice opens with its own Meta (rrweb takes a Meta+FullSnapshot at record start, at each page load,
// and at each checkout alike), so each slice re-attests its url rather than inheriting the last one seen.
test("page_url carries forward within a slice, and each slice attests its own", () => {
	const first = rows(
		[meta("https://site.test/a"), snapshot("a"), click(4)],
		"s1",
	);
	const second = rows(
		[meta("https://site.test/b"), snapshot("b", 100), click(103)],
		"s2",
	);
	second.forEach((r, i) => {
		r.id = 10 + i;
	});

	const { updates } = distillVisitor([...first, ...second]);
	expect(updates[0].cols.url).toBe("https://site.test/a");
	expect(updates.slice(0, 3).map((u) => u.pageUrl)).toEqual(
		Array(3).fill("https://site.test/a"),
	);
	expect(updates.slice(3).map((u) => u.pageUrl)).toEqual(
		Array(3).fill("https://site.test/b"),
	);
});

// A visitor's concurrent page contexts (tabs) interleave their slices in the canonical stream, so a
// slice's events are not contiguous there; each event must step its own slice's mirror and projection
// state, held across the interleaving.
test("interleaved slices keep their own DOM state and diff against their own projections", () => {
	const a = rows(
		[snapshot("page a"), textMutation(5, "a changed"), click(4)],
		"sA",
	);
	const b = rows([snapshot("page b", 100), click(103)], "sB");
	b.forEach((r, i) => {
		r.id = 10 + i;
	});
	// Canonical order interleaves: A opens, B opens, A mutates and clicks, B clicks.
	const stream = [a[0], b[0], a[1], a[2], b[1]];

	const { updates, residue } = distillVisitor(stream);
	const byId = new Map(updates.map((u) => [u.id, u]));

	// A's click resolves against A's DOM even though B's snapshot intervened...
	expect(byId.get(3).cols.tag).toBe("p");
	// ...its mutation run diffs against A's own projection, not B's...
	expect(byId.get(2).diff).toContain("- page a");
	expect(byId.get(2).diff).toContain("+ a changed");
	// ...and B's click resolves against B's DOM.
	expect(byId.get(11).cols.tag).toBe("p");
	expect(residue.unresolvedRefs).toBe(0);
});

test("events before the first slice flow through with columns and no projection", () => {
	const { updates } = distillVisitor(
		rows([meta("https://site.test/a"), click(4)], null),
	);
	expect(updates[0].cols.url).toBe("https://site.test/a");
	expect(updates[0].md).toBe(null);
	expect(updates[1].cols.tag).toBe(null);
});

test("the visitor's residue is the sum of its slices'", () => {
	const orphanMutation = {
		type: EventType.IncrementalSnapshot,
		timestamp: 2500,
		data: {
			source: IncrementalSource.Mutation,
			adds: [],
			removes: [{ parentId: 999, id: 998 }],
			attributes: [],
			texts: [],
		},
	};
	const first = rows([snapshot("a"), orphanMutation], "s1");
	const second = rows([snapshot("b", 100), orphanMutation], "s2");
	second.forEach((r, i) => {
		r.id = 10 + i;
	});

	const { residue } = distillVisitor([...first, ...second]);
	expect(residue.unresolvedRefs).toBe(2);
});

test("aborted mutations and dropped adds count in the residue", () => {
	const abortingRemove = {
		type: EventType.IncrementalSnapshot,
		timestamp: 2000,
		data: {
			source: IncrementalSource.Mutation,
			adds: [],
			// Node 4's real parent is 3; the wrong declared parent aborts the mutation partway.
			removes: [{ id: 4, parentId: 2 }],
			attributes: [],
			texts: [],
		},
	};
	const droppedAdd = {
		type: EventType.IncrementalSnapshot,
		timestamp: 2500,
		data: {
			source: IncrementalSource.Mutation,
			adds: [
				{
					parentId: 999,
					nextId: null,
					node: {
						type: NodeType.Element,
						tagName: "b",
						id: 50,
						childNodes: [],
					},
				},
			],
			removes: [],
			attributes: [],
			texts: [],
		},
	};
	const { residue } = distillVisitor(
		rows([snapshot("a"), abortingRemove, droppedAdd], "s1"),
	);
	expect(residue.aborted).toBe(1);
	expect(residue.droppedAdds).toBe(1);
});

test("wrongTyped counts the wire values the columns cannot hold", () => {
	const badUrl = {
		type: EventType.Meta,
		timestamp: 900,
		data: { href: { not: "a string" } },
	};
	const { residue } = distillVisitor(rows([badUrl], "s1"));
	expect(residue.wrongTyped).toBe(1);
});

test("the residue shape has one home, so a fold reaches every counter", () => {
	const total = newResidue();
	const distiller = new SliceDistiller();
	expect(Object.keys(distiller.residue)).toEqual(Object.keys(total));

	addResidue(total, { ...newResidue(), aborted: 2 });
	addResidue(total, { ...newResidue(), aborted: 3, wrongTyped: 1 });
	expect(total).toEqual({
		aborted: 5,
		droppedAdds: 0,
		unresolvedRefs: 0,
		wrongTyped: 1,
	});
});
