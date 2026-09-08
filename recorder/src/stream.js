/**
 * The event stream: counter stamping and slice stamping.
 *
 * Counter — `${timestamp}${seq6}` — orders same-millisecond events by emission. The sequence is zero-padded to six digits so counters compare lexicographically inside a millisecond (an unpadded seq would sort "10" before "9"), and wraps at one million. seq is a monotonic in-memory counter for the page context's lifetime, never derived from buffer length: the buffer drains, so same-ms events across a drain would collide.
 *
 * Slice stamping. Every slice is stored self-covering, so a store GET needs no covering-snapshot reach-back. Slice id = `${zero-padded-open-ms}-${disambiguator}`: time-ordered, and monotonic across page loads because it is timestamp-derived — no shared counter survives a fresh JS context. The random disambiguator separates two contexts opening slices in the same millisecond.
 *
 * The page context opens its own head slice, at start(), and hands it here, so a slice exists from the context's first instant: every event has somewhere to go and every ping has a slice to name, including from a page that dies before rrweb ever produces a snapshot. rrweb emits Meta then FullSnapshot at record-start and on each periodic checkout (takeFullSnapshot: rrweb packages/rrweb/src/record/index.ts): the record-start Meta *fills* the head slice — the snapshot it heralds is the head slice's covering snapshot — and every Meta after it opens a new slice, exactly as a checkout does. A head slice whose snapshot never arrives is a snapshotless slice, which the reject rule downstream drops.
 *
 * Move straddle. A mousemove/touch/drag batch buffers its motion and flushes later (initMoveObserver subtracts the total offset at flush, so positions[0].timeOffset is negative: rrweb packages/rrweb/src/record/observer.ts), so a batch can flush after a checkout Meta while carrying positions from before it: its flush timestamp lands in the new slice, but the motion happened in the prior one. When the batch's earliest position (flush + positions[0].timeOffset) predates the current slice's start, it re-homes onto the retained prior slice — bounded on both sides (prior.start ≤ true-start ≤ prior.end), so a batch contained by neither slice is reported loud rather than silently mis-keyed. Only the slice id moves; the raw timestamp stays the flush.
 */
import { EventType, IncrementalSource } from "./rrweb_constants.js";

const MOVE_SOURCES = new Set([
	IncrementalSource.MouseMove,
	IncrementalSource.TouchMove,
	IncrementalSource.Drag,
]);

export function makeSliceId(openMetaMs, random = Math.random) {
	const disambiguator = Math.floor(random() * 36 ** 4)
		.toString(36)
		.padStart(4, "0");
	return `${String(openMetaMs).padStart(14, "0")}-${disambiguator}`;
}

export function sliceStartMs(sliceId) {
	return Number(sliceId.split("-")[0]);
}

function moveTrueStart(event) {
	if (event.type !== EventType.IncrementalSnapshot) return null;
	const data = event.data;
	if (!data || !MOVE_SOURCES.has(data.source)) return null;
	const positions = data.positions;
	if (!positions?.length) return null;
	return event.timestamp + (positions[0].timeOffset || 0);
}

export class EventStream {
	/**
	 * @param {string} headSliceId  the page context's own slice, opened at start()
	 * @param {string} recorderVersion  the version of the recorder that captured these events
	 * @param {(sliceId: string) => void} onSliceOpen  fires for each slice this stream opens
	 * @param {(message: string) => void} onError  loud channel for an out-of-bounds straddle
	 */
	constructor(headSliceId, recorderVersion, onSliceOpen, onError = () => {}) {
		this.currentSliceId = headSliceId;
		this.recorderVersion = recorderVersion;
		this.headUnfilled = true;
		this.onSliceOpen = onSliceOpen;
		this.onError = onError;
		this.seq = 0;
		this.priorSliceId = null;
		this.priorSliceEnd = 0;
		this.lastTs = sliceStartMs(headSliceId);
	}

	/**
	 * Stamp one emitted event. Returns the {sliceId, recorderVersion, event} records ready for the buffer — always exactly one, since a slice exists from the context's first instant.
	 *
	 * The version is stamped here, on the event, not on the chunk at drain: the buffer outlives the page context, so any page context may courier another's records, and across a deploy those are different bundles. script_version is what downstream reads to know which recorder's shape it is looking at, so a chunk stamped by its courier would be read under the wrong rules.
	 */
	stamp(event) {
		this.seq += 1;
		event.counter = `${event.timestamp}${String(this.seq % 1_000_000).padStart(6, "0")}`;

		const record = (sliceId) => [
			{ sliceId, recorderVersion: this.recorderVersion, event },
		];

		if (event.type === EventType.Meta) {
			if (this.headUnfilled) {
				this.headUnfilled = false;
			} else {
				this.priorSliceId = this.currentSliceId;
				this.priorSliceEnd = this.lastTs;
				this.currentSliceId = makeSliceId(event.timestamp);
				this.onSliceOpen(this.currentSliceId);
			}
			this.lastTs = event.timestamp;
			return record(this.currentSliceId);
		}

		const trueStart = moveTrueStart(event);
		if (trueStart !== null && trueStart < sliceStartMs(this.currentSliceId)) {
			if (
				this.priorSliceId !== null &&
				sliceStartMs(this.priorSliceId) <= trueStart &&
				trueStart <= this.priorSliceEnd
			) {
				return record(this.priorSliceId);
			}
			this.onError(
				`locus-recorder: move batch straddles out of bounds — true start ${trueStart} ` +
					`predates slice ${this.currentSliceId} but is not contained by prior ` +
					`[${this.priorSliceId}, end ${this.priorSliceEnd}]; left in current slice`,
			);
		}

		this.lastTs = event.timestamp;
		return record(this.currentSliceId);
	}
}
