// The gate decides whether a snapshot-less slice may be welded onto the prior slice — a correctness decision
// slice materialization takes on trust. Strict is the whole point: a miss must refuse with its reason, never pass quietly.
import { expect, test } from "bun:test";
import { gate } from "./rescue_gate.js";
import {
	EventType,
	IncrementalSource,
	MouseInteractions,
	NodeType,
} from "./rrweb_constants.js";

const snapshot = () => ({
	type: EventType.FullSnapshot,
	timestamp: 1000,
	data: {
		node: {
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
							childNodes: [
								{
									type: NodeType.Element,
									tagName: "P",
									id: 4,
									attributes: {},
									childNodes: [
										{ type: NodeType.Text, id: 5, textContent: "hello" },
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

const incremental = (data, timestamp = 2000) => ({
	type: EventType.IncrementalSnapshot,
	timestamp,
	data,
});

const mutation = (data) =>
	incremental({
		source: IncrementalSource.Mutation,
		adds: [],
		removes: [],
		texts: [],
		attributes: [],
		...data,
	});

test("an orphan whose every reference resolves against the prior slice's end state is rescued", () => {
	const verdict = gate(
		[snapshot()],
		[
			incremental({
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 4,
				x: 1,
				y: 2,
			}),
			mutation({ texts: [{ id: 5, value: "goodbye" }] }),
		],
	);
	expect(verdict).toEqual({ rescued: true, reason: null });
});

test("an event referencing a node the prior slice never had is refused, by node and moment", () => {
	const verdict = gate(
		[snapshot()],
		[
			incremental(
				{
					source: IncrementalSource.MouseInteraction,
					type: MouseInteractions.Click,
					id: 77,
					x: 1,
					y: 2,
				},
				2200,
			),
		],
	);
	expect(verdict.rescued).toBe(false);
	expect(verdict.reason).toContain("2200");
	expect(verdict.reason).toContain("77");
});

test("a mouse move is judged by the ids inside its positions", () => {
	const move = (id) =>
		incremental({
			source: IncrementalSource.MouseMove,
			positions: [{ x: 1, y: 2, id, timeOffset: 0 }],
		});
	expect(gate([snapshot()], [move(4)]).rescued).toBe(true);
	expect(gate([snapshot()], [move(77)]).rescued).toBe(false);
});

test("a selection is judged by the ids inside its ranges", () => {
	const selection = (start, end) =>
		incremental({
			source: IncrementalSource.Selection,
			ranges: [{ start, startOffset: 0, end, endOffset: 1 }],
		});
	expect(gate([snapshot()], [selection(4, 5)]).rescued).toBe(true);
	expect(gate([snapshot()], [selection(4, 77)]).rescued).toBe(false);
});

test("a document-level reference (id -1) is not a node the mirror must know", () => {
	const scroll = incremental({
		source: IncrementalSource.Scroll,
		id: -1,
		x: 0,
		y: 400,
	});
	expect(gate([snapshot()], [scroll]).rescued).toBe(true);
});

test("a mutation the replayer would not survive refuses the orphan", () => {
	const verdicts = [
		mutation({ removes: [{ parentId: 3, id: 999 }] }),
		mutation({ texts: [{ id: 999, value: "x" }] }),
		mutation({ attributes: [{ id: 999, attributes: { class: "x" } }] }),
	].map((m) => gate([snapshot()], [m]));

	expect(verdicts.map((v) => v.rescued)).toEqual([false, false, false]);
	for (const v of verdicts) expect(v.reason).toContain("999");
});

test("a mutation the replayer aborts partway refuses the orphan with the abort", () => {
	// Node 4's real parent is 3; declaring 2 makes the replayer's removeChild throw mid-mutation.
	const verdict = gate(
		[snapshot()],
		[mutation({ removes: [{ id: 4, parentId: 2 }] })],
	);
	expect(verdict.rescued).toBe(false);
	expect(verdict.reason).toContain("2000");
});

test("references resolve against the state the orphan's own mutations built", () => {
	const verdict = gate(
		[snapshot()],
		[
			mutation({
				adds: [
					{
						parentId: 3,
						nextId: null,
						node: {
							type: NodeType.Element,
							tagName: "button",
							id: 50,
							attributes: {},
							childNodes: [],
						},
					},
				],
			}),
			incremental(
				{
					source: IncrementalSource.MouseInteraction,
					type: MouseInteractions.Click,
					id: 50,
					x: 1,
					y: 2,
				},
				2100,
			),
		],
	);
	expect(verdict).toEqual({ rescued: true, reason: null });
});

test("an add whose parent never resolves refuses the orphan", () => {
	const orphanAdd = mutation({
		adds: [
			{
				parentId: 999,
				nextId: null,
				node: { type: NodeType.Text, id: 50, textContent: "x" },
			},
		],
	});
	const verdict = gate([snapshot()], [orphanAdd]);
	expect(verdict.rescued).toBe(false);
	expect(verdict.reason).toContain("50");
});

test("a prior slice that reconstructs no DOM cannot cover anything", () => {
	const verdict = gate(
		[],
		[
			incremental({
				source: IncrementalSource.MouseInteraction,
				type: MouseInteractions.Click,
				id: 4,
				x: 1,
				y: 2,
			}),
		],
	);
	expect(verdict.rescued).toBe(false);
	expect(verdict.reason).toBeTruthy();
});
