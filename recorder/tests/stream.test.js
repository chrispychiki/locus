import { describe, expect, test } from "bun:test";

import { EventType, IncrementalSource } from "../src/rrweb_constants.js";
import { EventStream, makeSliceId, sliceStartMs } from "../src/stream.js";

const move = (ts, timeOffset, source = IncrementalSource.MouseMove) => ({
	type: EventType.IncrementalSnapshot,
	timestamp: ts,
	data: {
		source,
		positions: [{ x: 1, y: 1, timeOffset }],
	},
});

const meta = (ts) => ({
	type: EventType.Meta,
	data: { href: "x" },
	timestamp: ts,
});
const snapshot = (ts) => ({
	type: EventType.FullSnapshot,
	data: {},
	timestamp: ts,
});
const incremental = (ts) => ({
	type: EventType.IncrementalSnapshot,
	data: {},
	timestamp: ts,
});
const pageLoad = (ts) => ({
	type: EventType.PageLoad,
	data: {},
	timestamp: ts,
});

// The head slice the page context opens for itself at start(), before rrweb has emitted anything.
const head = (ts = 1000) => makeSliceId(ts);

const VERSION = "locus-recorder/1.2.3";

describe("slice ids", () => {
	test("zero-padded, time-ordered, disambiguated", () => {
		const id = makeSliceId(1774012655339, () => 0.5);
		expect(id).toMatch(/^01774012655339-[0-9a-z]{4}$/);
		expect(sliceStartMs(id)).toBe(1774012655339);
		expect(makeSliceId(1774012655339) < makeSliceId(1774012655340)).toBe(true);
	});

	// Two tabs racing start() is how this happens in the wild.
	test("two slices opened the same millisecond get distinct ids via the disambiguator", () => {
		const id1 = makeSliceId(1774012655339, () => 0.1);
		const id2 = makeSliceId(1774012655339, () => 0.9);
		expect(sliceStartMs(id1)).toBe(sliceStartMs(id2));
		expect(id1).not.toBe(id2);
	});
});

describe("EventStream — the head slice", () => {
	test("an event before the record-start Meta stamps onto the context's own head slice", () => {
		const HEAD = head();
		const opened = [];
		const stream = new EventStream(HEAD, VERSION, (id) => opened.push(id));

		const [early] = stream.stamp(pageLoad(1200));

		expect(early.sliceId).toBe(HEAD);
		expect(opened).toEqual([]);
	});

	test("the record-start Meta fills the head slice — it does not open a new one", () => {
		const HEAD = head();
		const opened = [];
		const stream = new EventStream(HEAD, VERSION, (id) => opened.push(id));

		const [opening] = stream.stamp(meta(1000));
		const [snap] = stream.stamp(snapshot(1001));

		expect(opening.sliceId).toBe(HEAD);
		expect(snap.sliceId).toBe(HEAD);
		expect(opened).toEqual([]);
	});

	// A periodic checkout, or a capture restart on a masking route change.
	test("every Meta after the record-start one opens a new slice", () => {
		const HEAD = head();
		const opened = [];
		const stream = new EventStream(HEAD, VERSION, (id) => opened.push(id));
		stream.stamp(meta(1000));
		stream.stamp(snapshot(1001));

		const checkout = stream.stamp(meta(3_601_000));
		const second = stream.stamp(meta(7_201_000));

		expect(opened.length).toBe(2);
		expect(checkout[0].sliceId).toBe(opened[0]);
		expect(second[0].sliceId).toBe(opened[1]);
		expect(sliceStartMs(opened[0])).toBe(3_601_000);
	});

	test("an opening Meta lands in the slice it opens, not the prior slice", () => {
		const HEAD = head();
		const opened = [];
		const stream = new EventStream(HEAD, VERSION, (id) => opened.push(id));
		stream.stamp(meta(1000));
		stream.stamp(snapshot(1001));

		const [checkoutMeta] = stream.stamp(meta(3_601_000));

		expect(checkoutMeta.sliceId).toBe(opened[0]);
		expect(checkoutMeta.sliceId).not.toBe(HEAD);
	});

	test("every stamped record carries the producing recorder's version — never left for a courier to guess", () => {
		// The buffer outlives the page context, so any context may courier another's records; a chunk
		// falls back to the courier's version only for records that carry none (chunk.js), which is
		// exactly why every record stamped here must carry its own.
		const stream = new EventStream(head(), VERSION, () => {});
		const records = [
			...stream.stamp(meta(1000)),
			...stream.stamp(snapshot(1001)),
			...stream.stamp(pageLoad(1002)),
		];
		expect(records).toHaveLength(3);
		for (const record of records) {
			expect(record.recorderVersion).toBe(VERSION);
		}
	});
});

describe("EventStream — counters", () => {
	test("counters are monotonic and order same-ms events", () => {
		const stream = new EventStream(head(), VERSION, () => {});
		const ms = 1774012655339;
		const first = stream.stamp(incremental(ms))[0];
		const second = stream.stamp(incremental(ms))[0];
		expect(first.event.counter).toBe("1774012655339000001");
		expect(second.event.counter).toBe("1774012655339000002");
		expect(second.event.counter > first.event.counter).toBe(true);
	});

	test("seq never resets across a simulated buffer drain — same-ms events stay distinct", () => {
		const stream = new EventStream(head(), VERSION, () => {});
		const ms = 1774012655339;
		const a = stream.stamp(incremental(ms))[0];
		const b = stream.stamp(incremental(ms))[0];
		// A drain would empty an external buffer here. The seq must keep climbing regardless,
		// not restart at a post-drain length of 0.
		const c = stream.stamp(incremental(ms))[0];
		const counters = [a, b, c].map((r) => r.event.counter);
		expect(new Set(counters).size).toBe(3);
		expect(counters[0] < counters[1] && counters[1] < counters[2]).toBe(true);
		expect(counters).toEqual([
			"1774012655339000001",
			"1774012655339000002",
			"1774012655339000003",
		]);
	});

	test("every event consumes one seq — the record-start Meta included", () => {
		const stream = new EventStream(head(), VERSION, () => {});
		const early = pageLoad(900);
		stream.stamp(early); // seq 1
		const [opening] = stream.stamp(meta(1000)); // seq 2
		const [next] = stream.stamp(incremental(1001)); // seq 3
		const seqOf = (counter) => Number(counter.slice(-6));
		expect(seqOf(early.counter)).toBe(1);
		expect(seqOf(opening.event.counter)).toBe(2);
		expect(seqOf(next.event.counter)).toBe(3);
	});

	test("seq is zero-padded to six digits and wraps at one million", () => {
		const stream = new EventStream(head(), VERSION, () => {});
		stream.seq = 999_998;
		const ts = 1774012655339;
		const a = stream.stamp(incremental(ts))[0]; // seq 999_999
		const b = stream.stamp(incremental(ts))[0]; // seq 1_000_000 -> 000000
		const c = stream.stamp(incremental(ts))[0]; // seq 1_000_001 -> 000001
		expect(a.event.counter).toBe(`${ts}999999`);
		expect(b.event.counter).toBe(`${ts}000000`);
		expect(c.event.counter).toBe(`${ts}000001`);
	});
});

describe("EventStream — move straddle", () => {
	const checkedOut = () => {
		const HEAD = head();
		const opened = [];
		const errors = [];
		const stream = new EventStream(
			HEAD,
			VERSION,
			(id) => opened.push(id),
			(m) => errors.push(m),
		);
		stream.stamp(meta(1000)); // fills the head slice
		stream.stamp(incremental(2000)); // head slice's last event -> its end
		stream.stamp(meta(3_601_000)); // checkout: opens opened[0]
		return { HEAD, opened, errors, stream };
	};

	test("a move batch whose motion predates the checkout re-homes to the prior slice", () => {
		const { HEAD, errors, stream } = checkedOut();
		// true start = 3_601_050 - 3_599_850 = 1200, inside the head slice [1000, 2000]
		const [stamped] = stream.stamp(move(3_601_050, -3_599_850));
		expect(errors).toEqual([]);
		expect(stamped.sliceId).toBe(HEAD);
		expect(stamped.event.timestamp).toBe(3_601_050);
	});

	// The motion's true time is recoverable at hydration from positions[0].timeOffset, so the raw
	// timestamp has no reason to move — and the counter is assigned before any re-home branch.
	test("re-home moves only the slice id — timestamp and counter stay the flush values", () => {
		const { HEAD, stream } = checkedOut();
		const [stamped] = stream.stamp(move(3_601_050, -3_599_850)); // the 4th stamped event
		expect(stamped.sliceId).toBe(HEAD);
		expect(stamped.event.timestamp).toBe(3_601_050);
		expect(stamped.event.counter).toBe(`3601050${String(4).padStart(6, "0")}`);
	});

	test("a move whose true start equals the prior slice start re-homes (inclusive lower bound)", () => {
		const { HEAD, errors, stream } = checkedOut();
		const [stamped] = stream.stamp(move(3_601_050, 1000 - 3_601_050));
		expect(errors).toEqual([]);
		expect(stamped.sliceId).toBe(HEAD);
	});

	// The prior slice's end is the last timestamp stamped into it before the checkout Meta.
	test("a move whose true start equals the prior slice end re-homes (inclusive upper bound)", () => {
		const { HEAD, errors, stream } = checkedOut();
		const [stamped] = stream.stamp(move(3_601_050, 2000 - 3_601_050));
		expect(errors).toEqual([]);
		expect(stamped.sliceId).toBe(HEAD);
	});

	test("a straddling batch contained by no slice is reported, left in current", () => {
		const { opened, errors, stream } = checkedOut();
		// true start = 0 — before the head slice itself
		const [stamped] = stream.stamp(move(3_601_050, -3_601_050));
		expect(errors.length).toBe(1);
		expect(stamped.sliceId).toBe(opened[0]);
	});

	test("a move whose true start predates the prior slice start is reported, left in current", () => {
		const { opened, errors, stream } = checkedOut();
		const [stamped] = stream.stamp(move(3_601_050, 500 - 3_601_050));
		expect(errors.length).toBe(1);
		expect(stamped.sliceId).toBe(opened[0]);
	});

	// Inside the head slice the context has opened only one slice, so there is no prior to re-home to.
	test("a predating move with no retained prior is reported, left in the head slice", () => {
		const HEAD = head();
		const errors = [];
		const stream = new EventStream(
			HEAD,
			VERSION,
			() => {},
			(m) => errors.push(m),
		);
		stream.stamp(meta(1000));
		// true start = 1050 - 200 = 850, before the head slice's start of 1000
		const [stamped] = stream.stamp(move(1050, -200));
		expect(errors.length).toBe(1);
		expect(stamped.sliceId).toBe(HEAD);
	});

	// rrweb batches motion the same way for all three of them (initMoveObserver serves mousemove,
	// touchmove, and drag), so every one can flush across a checkout carrying motion from before it.
	test("every move-family source straddles — touch and drag re-home exactly as the mouse does", () => {
		for (const source of [
			IncrementalSource.MouseMove,
			IncrementalSource.TouchMove,
			IncrementalSource.Drag,
		]) {
			const { HEAD, opened, errors, stream } = checkedOut();
			const [rehomed] = stream.stamp(move(3_601_050, -3_599_850, source));
			expect(rehomed.sliceId).toBe(HEAD);
			expect(errors).toEqual([]);
			const [outOfBounds] = stream.stamp(move(3_601_060, -3_601_060, source));
			expect(outOfBounds.sliceId).toBe(opened[0]);
			expect(errors.length).toBe(1);
		}
	});

	test("an ordinary move stamps to the current slice", () => {
		const HEAD = head();
		const stream = new EventStream(HEAD, VERSION, () => {});
		stream.stamp(meta(1000));
		const [stamped] = stream.stamp(move(1500, -40));
		expect(stamped.sliceId).toBe(HEAD);
	});

	// A non-move incremental, and a move with no positions, both have no true start to compute.
	test("only move-family events with positions can re-home — others stamp current", () => {
		const { opened, errors, stream } = checkedOut();
		const emptyMove = {
			type: EventType.IncrementalSnapshot,
			timestamp: 3_601_050,
			data: { source: IncrementalSource.MouseMove, positions: [] },
		};
		const [plain] = stream.stamp(incremental(3_601_010));
		const [movedEmpty] = stream.stamp(emptyMove);
		expect(plain.sliceId).toBe(opened[0]);
		expect(movedEmpty.sliceId).toBe(opened[0]);
		expect(errors).toEqual([]);
	});
});
