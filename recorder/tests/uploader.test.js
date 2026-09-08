import { describe, expect, test } from "bun:test";

import { BufferClosedError, openBuffer } from "../src/buffer.js";
import { EventType } from "../src/rrweb_constants.js";
import { storeRejection } from "../src/sink.js";
import { Uploader } from "../src/uploader.js";

const CONTEXT = () => ({
	visitorId: "v-1",
	recorderVersion: "0.1.0",
	envelope: {},
	errors: [],
});

const record = (sliceId, n) => ({
	sliceId,
	event: {
		type: EventType.IncrementalSnapshot,
		timestamp: n,
		counter: `${n}000001`,
	},
});

function collectingSink(failures = 0) {
	const sink = { sends: [], failuresLeft: failures };
	sink.send = async (_bytes, descriptor) => {
		if (sink.failuresLeft > 0) {
			sink.failuresLeft -= 1;
			throw new Error("sink down");
		}
		sink.sends.push(descriptor);
	};
	return sink;
}

describe("Uploader", () => {
	test("a pass that never returns cannot end the loop, and cannot report health while it hangs", async () => {
		// A browser under storage pressure can drop an IndexedDB request on the floor rather than
		// erroring it, so a pass can await something that never settles.
		let release;
		const stuck = new Promise((resolve) => {
			release = resolve;
		});
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		let hang = true;
		const realClaimed = buffer.claimed.bind(buffer);
		buffer.claimed = () => (hang ? stuck : realClaimed());

		const sink = collectingSink();
		const errors = [];
		const cost = {
			mainThreadMs: 0,
			uploadBytes: 0,
			deliveredBytes: 0,
			heapBytesMax: 0,
			backlogBytesMax: null,
			dirty: false,
		};
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			cost,
			onError: (e) => errors.push(String(e)),
			intervalMs: 10,
		});
		uploader.start();
		await new Promise((r) => setTimeout(r, 120)); // ~12 ticks against a hung pass
		expect(sink.sends).toEqual([]);
		expect(cost.deliveredBytes).toBe(0);
		expect(errors).toEqual([]);
		// Not a healthy zero: the device is holding a recording and shipping none of it.
		expect(cost.backlogBytesMax).toBeGreaterThan(0);

		// The loop kept ticking, skipping the stuck pass rather than waiting on it, so the moment
		// the buffer answers again it delivers.
		hang = false;
		release(null);
		await new Promise((r) => setTimeout(r, 60));
		expect(sink.sends.length).toBeGreaterThan(0);
		expect(cost.deliveredBytes).toBeGreaterThan(0);
		uploader.stop();
	});

	test("a failing pass faults — it does not merely queue a string onto the next chunk", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		buffer.claimed = async () => {
			throw new Error("indexeddb is gone");
		};
		const reported = [];
		const uploader = new Uploader({
			buffer,
			sink: collectingSink(),
			context: CONTEXT,
			onError: (e, opts = {}) => reported.push({ error: String(e), ...opts }),
		});

		await uploader.drain();

		expect(reported).toEqual([
			{
				error: "Error: indexeddb is gone",
				counts: false,
				fault: "drain_failed",
			},
		]);
	});

	test("a buffer closed by teardown ends the pass quietly — nothing is lost, so nothing is reported or fail-counted", async () => {
		const accounting = [];
		const buffer = {
			claimed: async () => {
				throw new BufferClosedError("claimed");
			},
			claim: async () => null,
			fail: async () => {
				accounting.push("fail");
				return 1;
			},
			resolve: async () => {
				accounting.push("resolve");
			},
		};
		const reported = [];
		const uploader = new Uploader({
			buffer,
			sink: collectingSink(),
			context: CONTEXT,
			onError: (e) => reported.push(String(e)),
		});

		expect(await uploader.drain()).toBe(false);

		expect(reported).toEqual([]); // a claimed batch stays inflight for a successor context
		expect(accounting).toEqual([]); // and the closed buffer is not asked to account a failure
	});

	test("error strings survive a failed delivery and ride the batch's own successful retry", async () => {
		const { gunzipSync } = await import("fflate");
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const queue = ["earlier: oversize note"];
		const payloads = [];
		const sink = {
			failuresLeft: 1,
			send: async (bytes) => {
				if (sink.failuresLeft-- > 0) throw new Error("sink down");
				payloads.push(JSON.parse(new TextDecoder().decode(gunzipSync(bytes))));
			},
		};
		const uploader = new Uploader({
			buffer,
			sink,
			onError: (e) => queue.push(String(e)),
			context: () => ({
				visitorId: "v-1",
				recorderVersion: "0.1.0",
				envelope: {},
				errors: queue.slice(),
			}),
			onErrorsDelivered: (n) => queue.splice(0, n),
		});

		await uploader.drain(); // fails — the strings must NOT evaporate
		expect(queue[0]).toBe("earlier: oversize note");

		await uploader.drain(); // the retry carries them out
		expect(payloads).toHaveLength(1);
		expect(payloads[0].errors[0]).toBe("earlier: oversize note");
		expect(payloads[0].errors.some((s) => s.includes("sink down"))).toBe(true);
		expect(queue).toEqual([]); // cleared only after the carrier landed
	});

	test("error strings outlive a dropped oversize carrier and ride the next delivered first chunk", async () => {
		// The strings ride the batch's FIRST chunk. When that chunk is dropped oversize, it was
		// never delivered — treating it as shipped would silently lose the whole error log.
		const { gunzipSync } = await import("fflate");
		const buffer = await openBuffer(null);
		const noise = new Uint8Array(6000);
		crypto.getRandomValues(noise);
		await buffer.append({
			sliceId: "s-1",
			event: {
				type: EventType.FullSnapshot,
				timestamp: 1,
				counter: "1000001",
				data: Array.from(noise, (b) => b.toString(16).padStart(2, "0")).join(
					"",
				),
			},
		});
		await buffer.append(record("s-2", 2));
		const queue = ["queued before the drain"];
		const payloads = [];
		const sink = {
			send: async (bytes) => {
				payloads.push(JSON.parse(new TextDecoder().decode(gunzipSync(bytes))));
			},
		};
		const uploader = new Uploader({
			buffer,
			sink,
			onError: (e) => queue.push(String(e)),
			context: () => ({
				visitorId: "v-1",
				recorderVersion: "0.1.0",
				envelope: {},
				errors: queue.slice(),
			}),
			onErrorsDelivered: (n) => queue.splice(0, n),
			onOversize: () => {},
			maxGzippedChunkBytes: 2000,
		});

		await uploader.drain();
		expect(payloads).toHaveLength(1); // s-2 delivered; the carrier (s-1) dropped oversize
		expect(payloads[0].errors).toEqual([]); // the strings never smuggle onto a later chunk
		expect(queue[0]).toBe("queued before the drain"); // and never evaporate

		await buffer.append(record("s-3", 3));
		await uploader.drain();
		const carried = payloads.find((p) => p.errors.length > 0);
		expect(carried.errors[0]).toBe("queued before the drain");
		expect(
			carried.errors.some((s) => s.includes("dropped oversize chunk")),
		).toBe(true);
		expect(queue).toEqual([]); // cleared only once a first chunk truly landed
	});

	test("the undeliverable-drop record names what died — slices, counter ranges, counts", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		await buffer.append(record("s-1", 2));
		await buffer.append(record("s-2", 3));
		const sink = {
			send: async () => {
				throw storeRejection("400 malformed chunk");
			},
		};
		const strings = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e) => strings.push(String(e)),
			maxAttempts: 1,
		});
		await uploader.drain();

		const drop = strings.find((s) => s.includes("undeliverable"));
		expect(drop).toContain("slice s-1: 2 events (counters 1000001–2000001)");
		expect(drop).toContain("slice s-2: 1 events (counters 3000001–3000001)");
		expect(drop).toContain("400 malformed chunk");
	});

	test("drains one batch into per-slice chunks and resolves on ack", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		await buffer.append(record("s-1", 2));
		await buffer.append(record("s-2", 3));
		const sink = collectingSink();
		const errors = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e) => errors.push(e),
		});

		await uploader.drain();
		expect(sink.sends.map((d) => [d.sliceId, d.count])).toEqual([
			["s-1", 2],
			["s-2", 1],
		]);
		expect(await buffer.claimed()).toBeNull();
		expect(errors).toEqual([]);
	});

	test("flush drains the whole outbox now — batch after batch, not one tick", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		await buffer.append(record("s-1", 2));
		await buffer.append(record("s-2", 3));
		const sink = collectingSink();
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
			maxUncompressedBatchBytes: 1,
		});
		await uploader.flush();
		expect(sink.sends.map((d) => [d.sliceId, d.count])).toEqual([
			["s-1", 1],
			["s-1", 1],
			["s-2", 1],
		]);
		expect(await buffer.claimed()).toBeNull();
		expect(await buffer.claim(1_000_000)).toBeNull();
	});

	test("flush clears inherited pending batches before new work, then the outbox", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		await buffer.append(record("s-1", 2));
		await buffer.claim(1); // two inflight batches, as an
		await buffer.claim(1); // earlier page context leaves them
		await buffer.append(record("s-2", 3)); // plus fresh outbox work
		const sink = collectingSink();
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});
		await uploader.flush();
		expect(sink.sends.map((d) => [d.sliceId, d.count])).toEqual([
			["s-1", 1],
			["s-1", 1],
			["s-2", 1],
		]);
		expect(await buffer.claimed()).toBeNull();
		expect(await buffer.claim(1_000_000)).toBeNull();
	});

	test("flush on an empty buffer resolves without sending anything", async () => {
		const buffer = await openBuffer(null);
		const sink = collectingSink();
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});
		await uploader.flush();
		expect(sink.sends).toEqual([]);
	});

	test("flush stops at a failing batch instead of spinning — the residue stays for the timer loop", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		await buffer.append(record("s-2", 2));
		const sink = {
			send: async () => {
				throw new Error("sink down");
			},
		};
		const errors = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e) => errors.push(String(e)),
			maxUncompressedBatchBytes: 1,
		});
		await uploader.flush();
		expect(await buffer.claimed()).not.toBeNull();
		expect(await buffer.claim(1_000_000)).not.toBeNull();
		expect(errors.some((s) => s.includes("sink down"))).toBe(true);
	});

	test("a drain landing mid-pass joins the in-flight pass — one request ever", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		let release;
		const gate = new Promise((r) => {
			release = r;
		});
		const sends = [];
		const sink = {
			send: async (_bytes, d) => {
				await gate;
				sends.push(d);
			},
		};
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});
		const first = uploader.drain();
		const second = uploader.drain();
		// Joining is the contract, not merely "one send happened": the buffer's own claim would
		// serialize two independent passes into one send all by itself, so the assertion that can
		// only hold under single-flight is that the second call got the very pass already running.
		expect(second).toBe(first);
		release();
		await Promise.all([first, second]);
		expect(sends).toHaveLength(1);
		expect(await buffer.claimed()).toBeNull();
	});

	test("failed batches stay inflight and retry before new work", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const sink = collectingSink(1);
		const errors = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e) => errors.push(e),
		});

		await uploader.drain();
		expect(errors.length).toBe(1);
		expect(sink.sends).toEqual([]);

		await buffer.append(record("s-1", 2));
		await uploader.drain();
		expect(sink.sends.map((d) => d.count)).toEqual([1]);

		await uploader.drain();
		expect(sink.sends.map((d) => d.count)).toEqual([1, 1]);
		expect(await buffer.claimed()).toBeNull();
	});

	test("a deterministically failing batch is dropped after the cap, sparing the queue and not counting toward termination", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const sink = {
			sends: [],
			send: async () => {
				throw storeRejection("413 too large");
			},
		};
		const counted = [];
		const onError = (e, opts = {}) => {
			if (opts.counts !== false) counted.push(e);
		};
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError,
			maxAttempts: 3,
		});

		for (let i = 0; i < 3; i++) await uploader.drain();

		expect(await buffer.claimed()).toBeNull(); // given up on and dropped
		expect(counted.length).toBe(0); // no delivery failure counts toward terminate

		await buffer.append(record("s-2", 2));
		const good = collectingSink();
		await new Uploader({
			buffer,
			sink: good,
			context: CONTEXT,
			onError: () => {},
		}).drain();
		expect(good.sends.map((d) => d.sliceId)).toEqual(["s-2"]);
	});

	test("an outage never poisons: sends the store never answered retry past the cap and land when the network returns", async () => {
		// The offline device. Nothing about the batch failed — the environment did — so no attempt
		// is spent on it and it is still queued whenever delivery starts working again.
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const failCalls = [];
		const realFail = buffer.fail.bind(buffer);
		buffer.fail = (batchId) => {
			failCalls.push(batchId);
			return realFail(batchId);
		};
		const sink = {
			sends: [],
			offline: true,
			send: async (_bytes, descriptor) => {
				if (sink.offline) throw new TypeError("Failed to fetch");
				sink.sends.push(descriptor);
			},
		};
		const dropped = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
			onUndeliverable: (e) => dropped.push(e),
			maxAttempts: 3,
		});

		for (let i = 0; i < 10; i++) await uploader.drain(); // far past the cap

		expect(failCalls).toEqual([]); // not one attempt spent on the environment
		expect(dropped).toEqual([]);
		expect(await buffer.claimed()).not.toBeNull(); // still queued, nothing lost

		sink.offline = false;
		await uploader.drain();
		expect(sink.sends.map((d) => d.sliceId)).toEqual(["s-1"]);
		expect(await buffer.claimed()).toBeNull();
	});

	test("only the store's refusals spend attempts — no-answer failures interleaved between them spend none", async () => {
		// A flapping network around a batch the store really does refuse: the cap is reached on the
		// refusals alone, however many times the send never got an answer in between.
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const failures = [
			() => storeRejection("400 malformed chunk"),
			() => new TypeError("Failed to fetch"),
			() => new TypeError("Failed to fetch"),
			() => storeRejection("400 malformed chunk"),
		];
		const sink = {
			send: async () => {
				throw failures.shift()();
			},
		};
		const dropped = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
			onUndeliverable: (e) => dropped.push(String(e)),
			maxAttempts: 2,
		});

		await uploader.drain(); // refusal: attempt 1
		await uploader.drain(); // no answer
		await uploader.drain(); // no answer
		expect(dropped).toEqual([]);
		expect(await buffer.claimed()).not.toBeNull();

		await uploader.drain(); // refusal: attempt 2 = the cap
		expect(dropped).toHaveLength(1);
		expect(await buffer.claimed()).toBeNull();
	});

	test("an oversize chunk is dropped client-side, reported, never uploaded or counted", async () => {
		const buffer = await openBuffer(null);
		await buffer.append({
			sliceId: "s-1",
			event: {
				type: EventType.FullSnapshot,
				timestamp: 1,
				counter: "1000001",
				data: {},
			},
		});
		const sink = collectingSink();
		const counted = [];
		const oversize = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e, opts = {}) => {
				if (opts.counts !== false) counted.push(e);
			},
			onOversize: (d, n) => oversize.push([d.sliceId, n]),
			maxGzippedChunkBytes: 1,
		});

		await uploader.drain();

		expect(sink.sends).toEqual([]); // never uploaded
		expect(oversize.map((o) => o[0])).toEqual(["s-1"]);
		expect(counted).toEqual([]); // not counted toward terminate
		expect(await buffer.claimed()).toBeNull(); // batch resolved, queue advances
	});

	test("an oversize page-load snapshot is catastrophic — signalled, never uploaded, batch resolved, not the plain oversize path", async () => {
		const buffer = await openBuffer(null);
		await buffer.append({
			sliceId: "s-1",
			event: {
				type: EventType.FullSnapshot,
				timestamp: 1,
				counter: "1000001",
				data: {},
			},
		});
		const sink = collectingSink();
		const oversize = [];
		const catastrophic = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
			onOversize: (d) => oversize.push(d.sliceId),
			onCatastrophic: (d, n) => catastrophic.push([d.sliceId, n]),
			isPageLoadSlice: (sliceId) => sliceId === "s-1",
			maxGzippedChunkBytes: 1,
		});

		await uploader.drain();

		expect(sink.sends).toEqual([]); // never uploaded
		expect(catastrophic.map((c) => c[0])).toEqual(["s-1"]); // catastrophic fired
		expect(oversize).toEqual([]); // NOT the plain oversize-drop path
		expect(await buffer.claimed()).toBeNull(); // batch resolved, never retries
	});

	test("an oversize checkout snapshot is a plain drop, not catastrophic — queue advances", async () => {
		const buffer = await openBuffer(null);
		await buffer.append({
			sliceId: "s-2",
			event: {
				type: EventType.FullSnapshot,
				timestamp: 1,
				counter: "1000001",
				data: {},
			},
		});
		const sink = collectingSink();
		const oversize = [];
		const catastrophic = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
			onOversize: (d) => oversize.push(d.sliceId),
			onCatastrophic: (d) => catastrophic.push(d.sliceId),
			isPageLoadSlice: (sliceId) => sliceId === "s-1", // s-2 is a checkout, not page-load
			maxGzippedChunkBytes: 1,
		});

		await uploader.drain();

		expect(sink.sends).toEqual([]); // not uploaded
		expect(oversize).toEqual(["s-2"]); // plain oversize drop
		expect(catastrophic).toEqual([]); // not catastrophic
		expect(await buffer.claimed()).toBeNull(); // dropped, queue advances
	});

	test("every per-attempt delivery failure before the cap reports counts:false — a poison batch can never self-terminate the recorder", async () => {
		// Every retry of a deterministically failing batch, not merely the final give-up.
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const sink = {
			sends: [],
			send: async () => {
				throw storeRejection("413 too large");
			},
		};
		const counted = [];
		const onError = (e, opts = {}) => {
			if (opts.counts !== false) counted.push(e);
		};
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError,
			maxAttempts: 5,
		});

		for (let i = 0; i < 5; i++) await uploader.drain();

		expect(counted).toEqual([]); // not one attempt counted toward terminate
	});

	test("onUndeliverable fires exactly once, at the drop — not on the attempts before it", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const sink = {
			sends: [],
			send: async () => {
				throw storeRejection("413 too large");
			},
		};
		const dropped = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
			onUndeliverable: (e) => dropped.push(String(e)),
			maxAttempts: 3,
		});

		await uploader.drain();
		await uploader.drain();
		expect(dropped).toEqual([]); // attempts 1..cap-1: no ping

		await uploader.drain(); // attempt 3 = the cap: dropped
		expect(dropped).toHaveLength(1);
		expect(dropped[0]).toContain("413");

		await buffer.append(record("s-2", 2));
		await uploader.drain(); // later failures start their own count
		expect(dropped).toHaveLength(1);
	});

	test("a backlog drained under a new identity still ships as the visitor who was recorded", async () => {
		// The cleared-cookie case: the writing page dies undrained, the cookie expires, and the
		// next page load — a different visitor id — drains the backlog. The chunks must state the
		// writer's identity and envelope, or the old visitor's slices file under the new id.
		const { IDBFactory } = await import("fake-indexeddb");
		const { gunzipSync } = await import("fflate");
		const factory = new IDBFactory();
		const writer = await openBuffer(factory);
		writer.captureContext = {
			visitorId: "v-old",
			envelope: { userAgent: "old-ua" },
		};
		await writer.append(record("s-old", 1));
		writer.close();

		const courier = await openBuffer(factory);
		courier.captureContext = {
			visitorId: "v-new",
			envelope: { userAgent: "new-ua" },
		};
		const payloads = [];
		const sink = {
			sends: [],
			send: async (bytes, descriptor) => {
				sink.sends.push(descriptor);
				payloads.push(JSON.parse(new TextDecoder().decode(gunzipSync(bytes))));
			},
		};
		const uploader = new Uploader({
			buffer: courier,
			sink,
			context: () => ({
				visitorId: "v-new",
				recorderVersion: "0.1.0",
				envelope: { userAgent: "new-ua" },
				errors: [],
			}),
			onError: () => {},
		});

		await uploader.drain();
		expect(sink.sends.map((d) => d.visitorId)).toEqual(["v-old"]);
		expect(payloads[0].visitorId).toBe("v-old");
		expect(payloads[0].envelope).toEqual({ userAgent: "old-ua" });
	});

	test("a batch claimed in one page context retries first in a fresh uploader (later page context)", async () => {
		// A brand-new Uploader over the same buffer stands in for the later page context.
		const buffer = await openBuffer(null);
		await buffer.append(record("s-old", 1));
		const downSink = {
			send: async () => {
				throw new Error("offline");
			},
		};
		await new Uploader({
			buffer,
			sink: downSink,
			context: CONTEXT,
			onError: () => {},
		}).drain(); // leaves s-old inflight
		expect(await buffer.claimed()).not.toBeNull();

		await buffer.append(record("s-new", 2)); // newer work arrives later
		const sink = collectingSink();
		const fresh = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});

		await fresh.drain();
		expect(sink.sends.map((d) => d.sliceId)).toEqual(["s-old"]); // pending shipped first
	});

	test("a clean drain with nothing left behind does NOT reset the ramp (only backpressure does)", async () => {
		// A steady trickle of events would otherwise pin the interval at base forever.
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const sink = collectingSink();
		const u = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});
		u.startedAt = 1;

		await u.drain();
		expect(u.startedAt).toBe(1); // ramp untouched on a clean, backlog-free drain
	});

	test("a pending retry resets the ramp even when the claim left no work behind", async () => {
		// A drain that ships one pending batch and leaves no `more` behind is still a buffer
		// that fell behind.
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const downSink = {
			send: async () => {
				throw new Error("offline");
			},
		};
		await new Uploader({
			buffer,
			sink: downSink,
			context: CONTEXT,
			onError: () => {},
		}).drain(); // one batch left inflight

		const sink = collectingSink();
		const u = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});
		u.startedAt = 1;
		await u.drain(); // ships the pending batch
		expect(u.startedAt).toBeGreaterThan(1); // ramp reset by the retry
	});

	test("single-flight: a drain in progress skips re-entry", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		let release;
		const gate = new Promise((resolve) => {
			release = resolve;
		});
		const sink = {
			sends: [],
			send: async (_bytes, descriptor) => {
				await gate;
				sink.sends.push(descriptor);
			},
		};
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});

		const first = uploader.drain();
		const second = uploader.drain();
		release();
		await Promise.all([first, second]);
		expect(sink.sends.length).toBe(1);
	});

	test("backpressure resets the ramp when a claim leaves work behind", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		await buffer.append(record("s-2", 2));
		const sink = collectingSink();
		const u = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
			maxUncompressedBatchBytes: 1,
		});
		u.startedAt = 0;
		await u.drain();
		expect(u.startedAt).toBeGreaterThan(0);
	});

	test("drain interval ramps with the page context's age, capped", () => {
		const u = new Uploader({
			buffer: null,
			sink: null,
			context: () => ({}),
			onError: () => {},
			intervalMs: 3_000,
			maxIntervalMs: 30_000,
			rampPeriodMs: 120_000,
		});
		expect(u.nextInterval(0)).toBe(3_000);
		expect(u.nextInterval(60_000)).toBe(3_000);
		expect(u.nextInterval(120_000)).toBe(6_000);
		expect(u.nextInterval(240_000)).toBe(12_000);
		expect(u.nextInterval(360_000)).toBe(24_000);
		expect(u.nextInterval(600_000)).toBe(30_000);
	});
});

describe("Uploader against untrusted persisted state", () => {
	test("one malformed record costs itself, never the batch", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		await buffer.append(null);
		await buffer.append({ sliceId: "s-1", event: undefined });
		await buffer.append({ sliceId: "s-1", event: { timestamp: 9 } }); // no counter — would POST to .../undefined
		await buffer.append(record("s-1", 2));
		const sink = collectingSink();
		const errors = [];
		const counted = [];
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e, opts = {}) => {
				errors.push(String(e));
				if (opts.counts !== false) counted.push(e);
			},
		});

		await uploader.drain();
		expect(sink.sends.map((d) => [d.sliceId, d.count])).toEqual([["s-1", 2]]);
		expect(await buffer.claimed()).toBeNull();
		expect(errors.some((e) => e.includes("3 malformed"))).toBe(true);
		expect(counted).toEqual([]);
	});

	test("a batch of nothing but malformed records resolves away instead of wedging", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(null);
		const sink = collectingSink();
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: () => {},
		});

		await uploader.drain();
		expect(sink.sends).toEqual([]);
		expect(await buffer.claimed()).toBeNull();
	});

	test("a batch resolved by a concurrent context mid-flight is not poison", async () => {
		// The sink stands in for the other context: it resolves the batch, then fails this send,
		// so fail() reports the batch already gone.
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		const errors = [];
		const dropped = [];
		const sink = {
			send: async () => {
				const pending = await buffer.claimed();
				await buffer.resolve(pending.batchId);
				throw storeRejection("400 malformed chunk");
			},
		};
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e) => errors.push(String(e)),
			onUndeliverable: (e) => dropped.push(e),
			maxAttempts: 1,
		});

		await uploader.drain();
		expect(dropped).toEqual([]);
		expect(errors.some((e) => e.includes("undeliverable"))).toBe(false);
	});

	test("drain never rejects, even when the buffer dies inside the failure path", async () => {
		const buffer = await openBuffer(null);
		await buffer.append(record("s-1", 1));
		buffer.fail = async () => {
			throw new Error("connection gone");
		};
		const errors = [];
		const sink = {
			send: async () => {
				throw storeRejection("400 malformed chunk");
			},
		};
		const uploader = new Uploader({
			buffer,
			sink,
			context: CONTEXT,
			onError: (e) => errors.push(String(e)),
		});

		await expect(uploader.drain()).resolves.toBe(false);
		expect(errors.some((e) => e.includes("connection gone"))).toBe(true);
	});
});

describe("Uploader over a corrupt inflight row", () => {
	test("an unreadable inflight batch is resolved away loudly, not silently", async () => {
		const { IDBFactory } = await import("fake-indexeddb");
		const factory = new IDBFactory();
		const writer = await openBuffer(factory);
		await new Promise((resolve, reject) => {
			const tx = writer.db.transaction(["inflight"], "readwrite");
			tx.objectStore("inflight").add(
				{ garbage: true },
				"00000000000001-zzzz-000001",
			);
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});
		const errors = [];
		const sink = collectingSink();
		const uploader = new Uploader({
			buffer: writer,
			sink,
			context: CONTEXT,
			onError: (e) => errors.push(String(e)),
		});

		await uploader.drain();
		expect(await writer.claimed()).toBeNull();
		expect(sink.sends).toEqual([]);
		expect(errors.some((e) => e.includes("unreadable inflight batch"))).toBe(
			true,
		);
	});

	test("flush resolves the unreadable batch away and still ships the real work behind it", async () => {
		const { IDBFactory } = await import("fake-indexeddb");
		const factory = new IDBFactory();
		const writer = await openBuffer(factory);
		await new Promise((resolve, reject) => {
			const tx = writer.db.transaction(["inflight"], "readwrite");
			tx.objectStore("inflight").add(
				{ garbage: true },
				"00000000000001-zzzz-000001",
			);
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});
		await writer.append(record("s-1", 1));
		const errors = [];
		const sink = collectingSink();
		const uploader = new Uploader({
			buffer: writer,
			sink,
			context: CONTEXT,
			onError: (e) => errors.push(String(e)),
		});

		await uploader.flush();
		expect(errors.some((e) => e.includes("unreadable inflight batch"))).toBe(
			true,
		);
		expect(sink.sends.map((d) => [d.sliceId, d.count])).toEqual([["s-1", 1]]);
		expect(await writer.claimed()).toBeNull();
		expect(await writer.claim(1_000_000)).toBeNull();
	});
});
