// Node-id continuity gate for slice rescue (slices.rescue_orphan_slices): a discarded "no covering FullSnapshot after Meta" slice may be appended to the prior replayable slice only if every node id its events reference resolves against the DOM state at the end of that prior slice — the merged slice must replay forward from the prior slice's head snapshot with no unresolved reference and no aborted mutation. Strict where distillation is tolerant: any miss is a refusal with the reason, never a silent skip.
//
//   bun distill/rescue_gate.js <events.db> <prior-slice-id> <orphan-slice-id>
//   stdout: {"rescued", "reason"}
//
// One candidate per invocation, structurally: a rescue rewrites the prior slice's events and end
// time, and the next candidate's verdict must be taken against that new state — so there is no
// batch to offer.
import { Database } from "bun:sqlite";
import { LightweightMirror } from "./mirror.js";
import { CANONICAL_ORDER, parseRaw } from "./raw.js";
import { EventType, IncrementalSource } from "./rrweb_constants.js";

function mutationFailure(report) {
	if (report.aborted) return `mutation aborted: ${report.aborted}`;
	if (report.missingRemoves.length)
		return `remove of unknown node ${report.missingRemoves[0]}`;
	if (report.missingTexts.length)
		return `text change on unknown node ${report.missingTexts[0]}`;
	if (report.missingAttributes.length)
		return `attribute change on unknown node ${report.missingAttributes[0]}`;
	if (report.droppedAdds.length)
		return `add of node ${report.droppedAdds[0]} never resolved`;
	return null;
}

function referencedIds(data) {
	const ids = [];
	if (typeof data.id === "number") ids.push(data.id);
	for (const position of data.positions || []) {
		if (typeof position.id === "number") ids.push(position.id);
	}
	for (const range of data.ranges || []) ids.push(range.start, range.end);
	return ids.filter((id) => id >= 1);
}

// The gate itself, over the two slices' events: replay the prior slice to its end state, then walk the
// orphan's events against it. Returns the verdict, never throws on a miss.
export function gate(priorEvents, orphanEvents) {
	const mirror = new LightweightMirror();
	for (const event of priorEvents) mirror.applyEvent(event);
	if (mirror.size === 0) {
		return { rescued: false, reason: "prior slice has no DOM state" };
	}
	for (const event of orphanEvents) {
		if (event.type !== EventType.IncrementalSnapshot) continue;
		const data = event.data || {};
		if (data.source === IncrementalSource.Mutation) {
			const failure = mutationFailure(mirror.applyMutation(data));
			if (failure) {
				return { rescued: false, reason: `at ${event.timestamp}: ${failure}` };
			}
			continue;
		}
		for (const id of referencedIds(data)) {
			if (!mirror.has(id)) {
				return {
					rescued: false,
					reason: `at ${event.timestamp}: event references unknown node ${id}`,
				};
			}
		}
	}
	return { rescued: true, reason: null };
}

if (import.meta.main) {
	const [dbPath, priorArg, orphanArg] = process.argv.slice(2);
	const prior = Number(priorArg);
	const orphan = Number(orphanArg);
	if (!dbPath || !Number.isInteger(prior) || !Number.isInteger(orphan)) {
		console.error(
			"usage: bun distill/rescue_gate.js <events.db> <prior-slice-id> <orphan-slice-id>",
		);
		process.exit(1);
	}

	const db = new Database(dbPath);
	const SLICE_EVENTS = db.prepare(
		`SELECT raw_json FROM events WHERE slice_id = ? ${CANONICAL_ORDER}`,
	);
	const sliceEvents = (sliceId) =>
		SLICE_EVENTS.all(sliceId).map((row) => parseRaw(row.raw_json));

	console.log(JSON.stringify(gate(sliceEvents(prior), sliceEvents(orphan))));
}
