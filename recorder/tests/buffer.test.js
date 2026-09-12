import { describe, expect, test } from "bun:test";
import { IDBFactory } from "fake-indexeddb";

import {
	BUFFER_ABSENT,
	BUFFER_NOT_DURABLE,
	BYTES_KEY,
	openBuffer,
	recordBytes,
} from "../src/buffer.js";
import { EventType } from "../src/rrweb_constants.js";

const record = (n) => ({
	sliceId: "s-1",
	event: {
		type: EventType.IncrementalSnapshot,
		timestamp: n,
		counter: `${n}000001`,
	},
});

// The shared durable counter, read as a new page context would read it: a fresh open's seed reads
// the counter — or recounts a database carrying none — and leaves the total on `bytes`
// (buffer.js seedCounter). The cross-context reading, taken through real surface only.
const sharedBytes = async (factory) => {
	const reader = await openBuffer(factory);
	await reader.seeded;
	const total = reader.bytes;
	reader.close();
	return total;
};

function eachBackend(name, body) {
	test(`${name} [indexeddb]`, () => body(() => openBuffer(new IDBFactory())));
	test(`${name} [memory]`, () => body(() => openBuffer(null)));
}

// A connection whose transactions hang forever — requests answer nothing, so only the deadline
// ends them — while recording every transaction and whether it was aborted.
const hangingDb = (txs = []) => ({
	transaction: () => {
		const tx = {
			aborted: false,
			abort() {
				this.aborted = true;
			},
			objectStore: () => ({ get: () => ({}) }),
		};
		txs.push(tx);
		return tx;
	},
	close: () => {},
});

// A factory whose opens succeed promptly — onto whatever connection the maker builds. The
// wedged-database shape and its kin: opening is never the problem, what the connection does is.
const factoryOf = (makeDb) => ({
	open: () => {
		const request = {};
		queueMicrotask(() => {
			request.result = makeDb();
			request.onsuccess?.();
		});
		return request;
	},
});

const hangingFactory = (txs) => factoryOf(() => hangingDb(txs));

describe("buffer", () => {
	eachBackend("claim is exclusive and order-preserving", async (open) => {
		const buffer = await open();
		for (const n of [1, 2, 3]) await buffer.append(record(n));

		const batch = await buffer.claim(1_000_000);
		expect(batch.records.map((r) => r.event.timestamp)).toEqual([1, 2, 3]);
		expect(await buffer.claim(1_000_000)).toBe(null);

		await buffer.resolve(batch.batchId);
		expect(await buffer.claimed()).toBeNull();
	});

	eachBackend(
		"the oldest unresolved batch surfaces for retry before new claims",
		async (open) => {
			const buffer = await open();
			await buffer.append(record(1));
			const failed = await buffer.claim(1_000_000);
			await buffer.append(record(2));

			const pending = await buffer.claimed();
			expect(pending.batchId).toBe(failed.batchId);
			expect(pending.records.map((r) => r.event.timestamp)).toEqual([1]);
		},
	);

	eachBackend(
		"claim respects the byte cap but always takes one",
		async (open) => {
			const buffer = await open();
			for (const n of [1, 2, 3]) await buffer.append(record(n));

			const tiny = await buffer.claim(1);
			expect(tiny.records.length).toBe(1);

			const capped = await buffer.claim(recordBytes(record(2)));
			expect(capped.records.length).toBe(1);
			expect(capped.records[0].event.timestamp).toBe(2);
		},
	);

	eachBackend(
		"claim signals leftover work via more, so the uploader can apply backpressure",
		async (open) => {
			const buffer = await open();
			for (const n of [1, 2, 3]) await buffer.append(record(n));

			const partial = await buffer.claim(recordBytes(record(1)));
			expect(partial.records.length).toBe(1);
			expect(partial.more).toBe(true); // two records still waiting

			const rest = await buffer.claim(1_000_000);
			expect(rest.more).toBe(false); // drained the outbox in this claim
		},
	);

	eachBackend(
		"claim of an empty outbox returns null, not an empty batch",
		async (open) => {
			const buffer = await open();
			expect(await buffer.claim(1_000_000)).toBe(null);
		},
	);

	eachBackend(
		"an unresolved claimed batch keeps surfacing until resolve — double-delivery is by design",
		async (open) => {
			const buffer = await open();
			await buffer.append(record(1));
			const batch = await buffer.claim(1_000_000);

			expect((await buffer.claimed()).batchId).toBe(batch.batchId);
			expect((await buffer.claimed()).batchId).toBe(batch.batchId);

			await buffer.resolve(batch.batchId);
			expect(await buffer.claimed()).toBeNull();
		},
	);

	eachBackend(
		"resolving an already-gone batch is a harmless no-op",
		async (open) => {
			const buffer = await open();
			await expect(buffer.resolve(123456)).resolves.toBeUndefined();
		},
	);

	test("buffered data survives a new page context (same idb)", async () => {
		const factory = new IDBFactory();
		const firstContext = await openBuffer(factory);
		await firstContext.append(record(1));
		const inflight = await firstContext.claim(1_000_000);
		await firstContext.append(record(2));

		const secondContext = await openBuffer(factory);
		const pending = await secondContext.claimed();
		expect(pending.batchId).toBe(inflight.batchId);
		const fresh = await secondContext.claim(1_000_000);
		expect(fresh.records.map((r) => r.event.timestamp)).toEqual([2]);
	});

	test("falls back to memory when indexeddb is broken", async () => {
		const buffer = await openBuffer({
			open: () => {
				throw new Error("denied");
			},
		});
		await buffer.append(record(1));
		expect((await buffer.claim(1_000_000)).records.length).toBe(1);
	});

	eachBackend(
		"fail counts deliveries and reports Infinity once the batch is gone",
		async (open) => {
			const buffer = await open();
			await buffer.append(record(1));
			const batch = await buffer.claim(1_000_000);
			expect(await buffer.fail(batch.batchId)).toBe(1);
			expect(await buffer.fail(batch.batchId)).toBe(2);
			await buffer.resolve(batch.batchId);
			expect(await buffer.fail(batch.batchId)).toBe(Infinity);
		},
	);

	test("the fail attempt count survives a new page context (poison batch can't reset its budget by reloading)", async () => {
		const factory = new IDBFactory();
		const first = await openBuffer(factory);
		await first.append(record(1));
		const batch = await first.claim(1_000_000);
		expect(await first.fail(batch.batchId)).toBe(1);

		const second = await openBuffer(factory);
		expect(await second.fail(batch.batchId)).toBe(2);
	});
});

describe("capture context", () => {
	const WRITER = {
		visitorId: "v-writer",
		envelope: { userAgent: "writer-ua" },
	};

	eachBackend(
		"a claimed batch carries each slice's capture context, stamped at write",
		async (open) => {
			const buffer = await open();
			buffer.captureContext = WRITER;
			await buffer.append(record(1));
			await buffer.append({ ...record(2), sliceId: "s-2" });

			const batch = await buffer.claim(1_000_000);
			expect(batch.contexts["s-1"]).toEqual(WRITER);
			expect(batch.contexts["s-2"]).toEqual(WRITER);
		},
	);

	eachBackend(
		"a writer that stamps nothing leaves the batch contextless — the courier-fallback case",
		async (open) => {
			const buffer = await open();
			await buffer.append(record(1));
			expect((await buffer.claim(1_000_000)).contexts).toEqual({});
		},
	);

	eachBackend("the retry path carries the contexts too", async (open) => {
		const buffer = await open();
		buffer.captureContext = WRITER;
		await buffer.append(record(1));
		await buffer.claim(1_000_000); // claimed, never resolved
		expect((await buffer.claimed()).contexts["s-1"]).toEqual(WRITER);
	});

	test("a courier draining another context's backlog gets the writer's context, not its own", async () => {
		const factory = new IDBFactory();
		const writer = await openBuffer(factory);
		writer.captureContext = WRITER;
		await writer.append(record(1));
		writer.close(); // the writing page died undrained

		const courier = await openBuffer(factory);
		courier.captureContext = {
			visitorId: "v-courier",
			envelope: { userAgent: "courier-ua" },
		};
		await courier.append({ ...record(2), sliceId: "s-courier" });

		const batch = await courier.claim(1_000_000);
		expect(batch.contexts["s-1"]).toEqual(WRITER);
		expect(batch.contexts["s-courier"].visitorId).toBe("v-courier");
	});

	test("context rows are wiped at the drained-empty moment and re-stamped by the next append", async () => {
		const factory = new IDBFactory();
		const buffer = await openBuffer(factory);
		buffer.captureContext = WRITER;
		await buffer.append(record(1));
		const batch = await buffer.claim(1_000_000);
		await buffer.resolve(batch.batchId);
		expect(await buffer.claim(1_000_000)).toBeNull(); // the drained-empty claim: rows wiped

		const rows = await new Promise((resolve, reject) => {
			const tx = buffer.db.transaction(["meta"], "readonly");
			const request = tx.objectStore("meta").getAllKeys();
			request.onsuccess = () => resolve(request.result);
			tx.onerror = () => reject(tx.error);
		});
		expect(rows.filter((key) => String(key).startsWith("context-"))).toEqual(
			[],
		);

		await buffer.append(record(2)); // the row rides the append itself
		expect((await buffer.claim(1_000_000)).contexts["s-1"]).toEqual(WRITER);
	});

	test("the capture context survives the fall to memory", async () => {
		const buffer = await openBuffer(new IDBFactory(), { timeoutMs: 30 });
		await buffer.seeded;
		buffer.captureContext = WRITER;
		buffer.db = hangingDb([]);
		buffer.factory = { open: () => ({}) };

		await buffer.append(record(1)); // falls to memory
		expect((await buffer.claim(1_000_000)).contexts["s-1"]).toEqual(WRITER);
	});
});

describe("backlog ring", () => {
	eachBackend(
		"at the ceiling the oldest outbox records evict to admit new capture, loudly",
		async (open) => {
			const buffer = await open();
			const evictions = [];
			buffer.onEvict = (records, bytes) => evictions.push({ records, bytes });
			const size = recordBytes(record(1));
			buffer.maxBacklogBytes = size * 3 + 10;

			for (const n of [1, 2, 3, 4, 5]) await buffer.append(record(n));

			const kept = (await buffer.claim(1_000_000)).records.map(
				(r) => r.event.timestamp,
			);
			expect(kept).toEqual([3, 4, 5]); // newest win; oldest evicted
			expect(
				evictions.flatMap((e) => e.records.map((r) => r.event.timestamp)),
			).toEqual([1, 2]);
			expect(evictions.every((e) => e.bytes > 0)).toBe(true);
		},
	);

	eachBackend(
		"eviction touches only the outbox — inflight batches belong to the poison cap",
		async (open) => {
			const buffer = await open();
			const evictions = [];
			buffer.onEvict = (records, bytes) => evictions.push({ records, bytes });
			await buffer.append(record(1));
			await buffer.append(record(2));
			const inflight = await buffer.claim(1_000_000); // both records now inflight
			buffer.maxBacklogBytes = 1; // everything is over the ceiling

			await buffer.append(record(3)); // nothing evictable in the outbox but itself's admission

			const pending = await buffer.claimed();
			expect(pending.batchId).toBe(inflight.batchId);
			expect(pending.records.map((r) => r.event.timestamp)).toEqual([1, 2]);
			const fresh = await buffer.claim(1_000_000);
			expect(fresh.records.map((r) => r.event.timestamp)).toEqual([3]);
		},
	);
});

describe("durable byte counter", () => {
	test("the buffer says what it holds without being asked, the open-time seed being the first reading", async () => {
		const factory = new IDBFactory();
		const size = recordBytes(record(1));
		const buffer = await openBuffer(factory);
		await buffer.seeded;
		expect(buffer.bytes).toBe(0); // observed empty at open — a reading, not a guess

		await buffer.append(record(1));
		expect(buffer.bytes).toBe(size); // the write knows the new total; no read needed
		await buffer.append(record(2));
		expect(buffer.bytes).toBe(size * 2);

		const batch = await buffer.claim(1_000_000);
		await buffer.resolve(batch.batchId);
		expect(buffer.bytes).toBe(0); // delivered, and the device is holding nothing
	});

	test("the total is exact across contexts and shrinks on resolve — no scan, one shared counter", async () => {
		const factory = new IDBFactory();
		const size = recordBytes(record(1));
		const tabA = await openBuffer(factory);
		await tabA.append(record(1));
		await tabA.append(record(2));

		const tabB = await openBuffer(factory);
		await tabB.append(record(3));
		expect(await sharedBytes(factory)).toBe(size * 3); // one counter sees A's and B's writes

		const batch = await tabA.claim(1_000_000);
		expect(await sharedBytes(factory)).toBe(size * 3); // claim is net zero
		await tabA.resolve(batch.batchId);
		expect(await sharedBytes(factory)).toBe(0); // delivery shrinks the shared total
	});

	test("a database carrying no counter seeds by one recount at open, then stays counted", async () => {
		const factory = new IDBFactory();
		const size = recordBytes(record(1));
		const old = await openBuffer(factory);
		await old.append(record(1));
		await old.append(record(2));
		await new Promise((resolve, reject) => {
			// erase the counter: a database carrying none
			const tx = old.db.transaction(["meta"], "readwrite");
			tx.objectStore("meta").delete("backlogBytes");
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});

		const fresh = await openBuffer(factory);
		await fresh.seeded;
		expect(fresh.bytes).toBe(size * 2); // seeded at open, before any operation
		await fresh.append(record(3));
		expect(fresh.bytes).toBe(size * 3); // maintained from there
	});

	test("an append that finds no counter still lands, and the next recount prices its row", async () => {
		const factory = new IDBFactory();
		const size = recordBytes(record(1));
		const buffer = await openBuffer(factory);
		await buffer.append(record(1));
		await new Promise((resolve, reject) => {
			// the counter vanishes under a live context
			const tx = buffer.db.transaction(["meta"], "readwrite");
			tx.objectStore("meta").delete("backlogBytes");
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});

		await buffer.append(record(2)); // admitted unaccounted — and it kicks a reseed
		await buffer.reseeding;
		expect(buffer.bytes).toBe(size * 2); // the reseed's recount found both rows
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([1, 2]);
	});

	test("a row stored without a size stamp is priced by recount, and the ledger stays exact", async () => {
		const factory = new IDBFactory();
		const size = recordBytes(record(999));
		const buffer = await openBuffer(factory);
		await buffer.seeded;
		await new Promise((resolve, reject) => {
			// another bundle's write: no size in the key, and no ledger entry
			const tx = buffer.db.transaction(["outbox", "meta"], "readwrite");
			tx.objectStore("outbox").add(record(999), 7);
			tx.objectStore("meta").delete("backlogBytes");
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});

		expect(await sharedBytes(factory)).toBe(size); // recounted from the value
		const batch = await buffer.claim(1_000_000);
		expect(batch.records.map((r) => r.event.timestamp)).toEqual([999]);
		await buffer.resolve(batch.batchId);
		expect(await sharedBytes(factory)).toBe(0);
	});

	test("positive drift (an uncounted resolve) heals at the first empty claim", async () => {
		const factory = new IDBFactory();
		const buffer = await openBuffer(factory);
		await buffer.append(record(1));
		const batch = await buffer.claim(1_000_000);
		await new Promise((resolve, reject) => {
			// a tab that doesn't keep the counter resolves the batch: bare delete, no ledger entry
			const tx = buffer.db.transaction(["inflight"], "readwrite");
			tx.objectStore("inflight").delete(batch.batchId);
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});
		expect(await sharedBytes(factory)).toBe(recordBytes(record(1))); // the stranded bytes

		expect(await buffer.claim(1_000_000)).toBeNull(); // both stores empty — the heal moment
		expect(await sharedBytes(factory)).toBe(0);
	});

	test("a phantom-full counter exits through one delivery cycle instead of evicting forever", async () => {
		const factory = new IDBFactory();
		const buffer = await openBuffer(factory);
		const evictions = [];
		buffer.onEvict = (_records, bytes) => evictions.push(bytes);
		await new Promise((resolve, reject) => {
			// maximal drift: counter at the ceiling over empty stores
			const tx = buffer.db.transaction(["meta"], "readwrite");
			tx.objectStore("meta").put(buffer.maxBacklogBytes, "backlogBytes");
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});

		await buffer.append(record(1)); // nothing to evict — capture is admitted regardless
		const batch = await buffer.claim(1_000_000);
		expect(batch.records).toHaveLength(1);
		await buffer.resolve(batch.batchId);
		expect(await buffer.claim(1_000_000)).toBeNull(); // empty claim heals the residue
		expect(await sharedBytes(factory)).toBe(0);

		await buffer.append(record(2)); // the ring is sane again
		expect(evictions).toEqual([]);
		expect((await buffer.claim(1_000_000)).records).toHaveLength(1);
	});

	test("no heal while a batch is still inflight — empty outbox alone proves nothing", async () => {
		const factory = new IDBFactory();
		const size = recordBytes(record(1));
		const buffer = await openBuffer(factory);
		await buffer.append(record(1));
		const batch = await buffer.claim(1_000_000); // outbox now empty, batch inflight
		expect(await buffer.claim(1_000_000)).toBeNull(); // empty claim, but not a heal moment
		expect(await sharedBytes(factory)).toBe(size); // inflight bytes still owed
		await buffer.resolve(batch.batchId);
		expect(await sharedBytes(factory)).toBe(0);
	});

	test("a backend that never answers leaves bytes null — no reading taken is not an empty buffer", async () => {
		const buffer = await openBuffer(hangingFactory([]), { timeoutMs: 30 });
		await buffer.seeded; // the seed timed out and was swallowed
		expect(buffer.bytes).toBeNull();
	});
});

describe("the buffer never touches permission-bearing APIs", () => {
	test("opening asks the browser for nothing — no permission query, no persist", async () => {
		// navigator.storage.persist() shows Firefox visitors a permission doorhanger, so the
		// buffer must never call it: storage stays best-effort, evictable under pressure.
		const original = globalThis.navigator;
		const calls = [];
		globalThis.navigator = {
			permissions: {
				query: async (d) => {
					calls.push(d);
					return { state: "granted" };
				},
			},
			storage: {
				persist: async () => {
					calls.push("persist");
					return true;
				},
			},
		};
		try {
			await openBuffer(new IDBFactory());
			await new Promise((r) => setTimeout(r, 0));
			expect(calls).toEqual([]);
		} finally {
			globalThis.navigator = original;
		}
	});
});

describe("buffer under concurrency and death", () => {
	test("concurrent page contexts with colliding counters never collide on keys", async () => {
		// Two tabs share one visitor DB and both start their event sequence at 1, so identical
		// counters across contexts are routine.
		const factory = new IDBFactory();
		const tabA = await openBuffer(factory);
		const tabB = await openBuffer(factory);
		for (const n of [1, 2]) {
			await tabA.append(record(n));
			await tabB.append(record(n));
		}

		const batch = await tabA.claim(1_000_000);
		expect(batch.records.map((r) => r.event.timestamp).sort()).toEqual([
			1, 1, 2, 2,
		]);
	});

	test("generator-keyed records persisted by an older recorder drain first", async () => {
		// A visitor DB may hold numeric autoIncrement keys from a bundle that used the generator.
		// IndexedDB orders numbers before strings, so that backlog keeps the head of the queue.
		const factory = new IDBFactory();
		const buffer = await openBuffer(factory);
		await new Promise((resolve, reject) => {
			const tx = buffer.db.transaction(["outbox"], "readwrite");
			tx.objectStore("outbox").add(record(999), 7);
			tx.oncomplete = resolve;
			tx.onerror = () => reject(tx.error);
		});
		await buffer.append(record(1));

		const batch = await buffer.claim(1_000_000);
		expect(batch.records.map((r) => r.event.timestamp)).toEqual([999, 1]);
	});

	test("a malformed inflight row surfaces as an empty batch, never a throw", async () => {
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
		await writer.append(record(1));
		await writer.claim(1_000_000);

		const reader = await openBuffer(factory);
		const pending = await reader.claimed(); // the malformed row is oldest
		expect(pending.records).toEqual([]);
		await reader.resolve(pending.batchId);
		expect(
			(await reader.claimed()).records.map((r) => r.event.timestamp),
		).toEqual([1]);
	});

	test("a lost connection heals once and the operation retries", async () => {
		const factory = new IDBFactory();
		const buffer = await openBuffer(factory);
		await buffer.append(record(1));

		const dead = {
			transaction: () => {
				const error = new Error("The database connection is closing.");
				error.name = "InvalidStateError";
				throw error;
			},
		};
		buffer.db = dead;
		await buffer.append(record(2));

		expect(buffer.db).not.toBe(dead);
		const batch = await buffer.claim(1_000_000);
		expect(batch.records.map((r) => r.event.timestamp)).toEqual([1, 2]);
	});

	test("a connection that stops answering is a failure, not a wait — the deadline aborts its transaction and the buffer falls to memory", async () => {
		// The stand-in for a browser that force-closes the connection and drops the in-flight
		// requests, firing neither success nor error.
		const factory = new IDBFactory();
		const unavailable = [];
		const buffer = await openBuffer(factory, {
			timeoutMs: 30,
			onUnavailable: (e) => unavailable.push(e),
		});
		await buffer.seeded; // the open-time seed settles against the real database
		const txs = [];
		buffer.db = hangingDb(txs);
		buffer.factory = { open: () => ({}) }; // the reopen goes silent too

		await buffer.append(record(1)); // resolves — into memory, not into nowhere
		expect(unavailable.map((e) => e.name)).toEqual(["BufferTimeoutError"]);
		expect(txs.length).toBe(1);
		expect(txs[0].aborted).toBe(true); // aborted at the deadline, never left pending
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([1]);
	});

	test("a wedged database — reopen succeeds, the retry queues behind the same lock — falls to memory with every transaction aborted", async () => {
		// The fleet signature: a foreign context's pending transaction holds the store lock, so
		// this context's first operation times out, the heal-once reopen succeeds instantly (opens
		// need no store lock), and the retry times out behind the same lock.
		const factory = new IDBFactory();
		const unavailable = [];
		const buffer = await openBuffer(factory, {
			timeoutMs: 30,
			onUnavailable: (e) => unavailable.push(e),
		});
		await buffer.seeded; // the open-time seed settles against the real database
		const txs = [];
		buffer.db = hangingDb(txs);
		buffer.factory = hangingFactory(txs); // a fresh connection to the same wedged database

		await buffer.append(record(1));
		expect(txs.length).toBe(2); // birth attempt + healed retry
		expect(txs.every((tx) => tx.aborted)).toBe(true); // no pending transaction outlives its deadline
		expect(unavailable.map((e) => e.name)).toEqual(["BufferTimeoutError"]);

		// Dead is dead for this context: later operations run in memory, with no new deadlines.
		await buffer.append(record(2));
		expect(txs.length).toBe(2);
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([1, 2]);
	});

	test("the fallen buffer keeps the ring ceiling and the loud evictions", async () => {
		const unavailable = [];
		const buffer = await openBuffer(new IDBFactory(), {
			timeoutMs: 30,
			onUnavailable: (e) => unavailable.push(e),
		});
		const evictions = [];
		buffer.onEvict = (records) =>
			evictions.push(records.map((r) => r.event.timestamp));
		const size = recordBytes(record(1));
		buffer.maxBacklogBytes = size * 2 + 10;
		buffer.db = hangingDb();
		buffer.factory = { open: () => ({}) };

		for (const n of [1, 2, 3, 4]) await buffer.append(record(n));
		expect(unavailable.length).toBe(1); // the fall is reported once, not per append
		expect(buffer.bytes).toBe(size * 2); // `bytes` reads the memory ledger now
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([3, 4]);
		expect(evictions.flat()).toEqual([1, 2]);
	});

	test("close() aborts in-flight work and refuses new work; resume() accepts work again", async () => {
		const factory = new IDBFactory();
		const unavailable = [];
		const buffer = await openBuffer(factory, {
			onUnavailable: (e) => unavailable.push(e),
		});
		await buffer.append(record(1)); // durable before the teardown
		await buffer.seeded;

		const txs = [];
		buffer.db = hangingDb(txs); // an append that will still be in flight
		const inflight = buffer.append(record(2));
		buffer.close();
		await expect(inflight).rejects.toThrow(/buffer closed during append/);
		expect(txs[0].aborted).toBe(true); // aborted now, not left for the deadline
		await expect(buffer.claim(1_000_000)).rejects.toThrow(
			/buffer closed during claim/,
		);

		buffer.resume(); // pageshow / lifecycle resume
		const batch = await buffer.claim(1_000_000); // reopens lazily via the real factory
		expect(batch.records.map((r) => r.event.timestamp)).toEqual([1]);
		expect(unavailable).toEqual([]); // teardown is not unavailability
	});

	test("an open answered after its deadline is closed on arrival, never leaked", async () => {
		// Observed in Chromium: an open queued behind another context's stalled versionchange open
		// delivers its connection long after any watchdog gave up on it.
		let request = null;
		const factory = {
			open: () => {
				request = {};
				return request;
			},
		};
		const unavailable = [];
		const buffer = await openBuffer(factory, {
			timeoutMs: 20,
			onUnavailable: (e) => unavailable.push(e),
		});
		expect(unavailable.map((e) => e.name)).toEqual(["BufferTimeoutError"]);

		let closed = false;
		request.result = {
			close: () => {
				closed = true;
			},
		};
		request.onsuccess();
		expect(closed).toBe(true);

		await buffer.append(record(1)); // and the memory fallback buffers on
		expect((await buffer.claim(1_000_000)).records.length).toBe(1);
	});

	test("an append whose key is already stored is its own earlier attempt's commit — durable, counted once, never a fall", async () => {
		// The fleet shape: the first attempt's transaction committed after its deadline (or completed
		// without delivering the success event), the buffer reopened and reran the same append, and
		// the add collided with the row it had already written. Plant that committed row — the
		// record under the key this instance will mint next, counted — and append: the ConstraintError
		// is the commit's witness, so the append resolves stored, IndexedDB stays the buffer, the
		// counter holds one copy, and the outbox holds one record.
		const unavailable = [];
		const buffer = await openBuffer(new IDBFactory(), {
			onUnavailable: (e) => unavailable.push(e),
		});
		const rec = record(1);
		const size = new TextEncoder().encode(JSON.stringify(rec)).length;
		const key = `${rec.event.counter}-${buffer.tag}-${String(buffer.seq + 1).padStart(6, "0")}-s${size}`;
		await new Promise((resolve, reject) => {
			const tx = buffer.db.transaction(["outbox", "meta"], "readwrite");
			tx.objectStore("outbox").add(rec, key);
			tx.objectStore("meta").put(size, BYTES_KEY);
			tx.oncomplete = resolve;
			tx.onabort = () => reject(tx.error);
		});

		await buffer.append(rec);
		expect(unavailable.length).toBe(0);
		expect(buffer.fallen).toBeNull();
		expect(buffer.bytes).toBe(size);
		const batch = await buffer.claim(1_000_000);
		expect(batch.records.length).toBe(1);
	});

	test("an append whose key is already stored makes no room for itself — at the ceiling, nothing older is evicted", async () => {
		// Same committed-then-retried shape, with the backlog exactly at the ceiling once the row is
		// counted: an older record plus the committed row fill it to the byte. The row already holds
		// its space, so the retry must evict nothing and leave both records for the claim.
		const evictions = [];
		const buffer = await openBuffer(new IDBFactory());
		buffer.onEvict = (records, bytes) => evictions.push({ records, bytes });
		const older = record(1);
		const rec = record(2);
		const size = new TextEncoder().encode(JSON.stringify(rec)).length;
		await buffer.append(older);
		buffer.maxBacklogBytes = buffer.bytes + size;
		const key = `${rec.event.counter}-${buffer.tag}-${String(buffer.seq + 1).padStart(6, "0")}-s${size}`;
		await new Promise((resolve, reject) => {
			const tx = buffer.db.transaction(["outbox", "meta"], "readwrite");
			tx.objectStore("outbox").add(rec, key);
			tx.objectStore("meta").put(buffer.maxBacklogBytes, BYTES_KEY);
			tx.oncomplete = resolve;
			tx.onabort = () => reject(tx.error);
		});

		await buffer.append(rec);
		expect(buffer.fallen).toBeNull();
		expect(evictions).toEqual([]);
		expect(buffer.bytes).toBe(buffer.maxBacklogBytes);
		const batch = await buffer.claim(1_000_000);
		expect(batch.records.map((r) => r.event.counter)).toEqual([
			older.event.counter,
			rec.event.counter,
		]);
	});

	test("a failed request surfaces its own error — read at abort, where IndexedDB has set it", async () => {
		// The spec fires the transaction's error event before it sets the transaction's error slot;
		// the abort that follows is where the request's error is readable. A request error that is
		// not the append's own key collision stays unhandled, the transaction aborts, and the fall
		// reports what the request failed with.
		const failing = () => ({
			transaction: () => {
				const error = Object.assign(new Error("data error"), {
					name: "DataError",
				});
				const tx = {
					error: null,
					abort() {},
					objectStore: () => ({
						get: () => {
							const request = {};
							queueMicrotask(() => {
								request.result = undefined;
								request.onsuccess?.();
							});
							return request;
						},
						put: () => ({}),
						add: () => {
							const request = { error };
							queueMicrotask(() => {
								request.onerror?.({ preventDefault() {} });
								tx.error = error;
								tx.onabort?.();
							});
							return request;
						},
					}),
				};
				return tx;
			},
			close: () => {},
		});
		const unavailable = [];
		const buffer = await openBuffer(new IDBFactory(), {
			onUnavailable: (e) => unavailable.push(e),
		});
		buffer.db = failing();
		buffer.factory = factoryOf(failing);

		await buffer.append(record(1)); // resolves — into memory
		expect(unavailable.length).toBe(1);
		expect(unavailable[0]?.name).toBe("DataError");
	});

	test("a transaction aborted with no error object surfaces an honest error, never null", async () => {
		const nullAbortDb = () => ({
			transaction: () => {
				const tx = {
					error: null,
					abort() {},
					objectStore: () => ({ get: () => ({}) }),
				};
				queueMicrotask(() => tx.onabort());
				return tx;
			},
			close: () => {},
		});
		const unavailable = [];
		const buffer = await openBuffer(new IDBFactory(), {
			onUnavailable: (e) => unavailable.push(e),
		});
		buffer.db = nullAbortDb();
		buffer.factory = factoryOf(nullAbortDb);

		await buffer.append(record(1)); // resolves — into memory
		expect(unavailable.length).toBe(1);
		expect(unavailable[0]).not.toBeNull();
		expect(String(unavailable[0])).toContain(
			"append transaction aborted with no error object",
		);
	});

	test("a transaction that completes without ever answering a request is a failure, never a value", async () => {
		// The fleet shape: the backend runs the transaction to completion but no request's success
		// handler is ever called, so the body never reports. Reading that as a result crashed the
		// caller on an undefined summary; it is the connection answering nothing, and it falls the
		// same way the deadline does — into memory, reported, the append still landing.
		const silentDb = () => ({
			transaction: () => {
				const tx = {
					error: null,
					abort() {},
					objectStore: () => ({
						get: () => ({}),
						put: () => ({}),
						add: () => ({}),
					}),
				};
				queueMicrotask(() => tx.oncomplete());
				return tx;
			},
			close: () => {},
		});
		const unavailable = [];
		const buffer = await openBuffer(new IDBFactory(), {
			onUnavailable: (e) => unavailable.push(e),
		});
		await buffer.seeded;
		buffer.db = silentDb();
		buffer.factory = factoryOf(silentDb);

		await buffer.append(record(1)); // resolves — into memory
		expect(unavailable.length).toBe(1);
		expect(String(unavailable[0])).toContain("completed without");
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([1]);
	});

	test("a one-off failure of any name heals — retried against a fresh connection, no fall", async () => {
		const factory = new IDBFactory();
		const unavailable = [];
		const buffer = await openBuffer(factory, {
			onUnavailable: (e) => unavailable.push(e),
		});
		buffer.db = {
			transaction: () => {
				const error = new Error("disk hiccup");
				error.name = "UnknownError";
				throw error;
			},
			close: () => {},
		};

		await buffer.append(record(1)); // the retry runs against the real reopened backend
		expect(unavailable).toEqual([]);
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([1]);
	});

	test("a quota-shaped death falls to memory too — any persistent failure degrades, never errors forever", async () => {
		const quotaDb = () => ({
			transaction: () => {
				const error = new Error("quota exceeded");
				error.name = "QuotaExceededError";
				throw error;
			},
			close: () => {},
		});
		const unavailable = [];
		const buffer = await openBuffer(new IDBFactory(), {
			onUnavailable: (e) => unavailable.push(e),
		});
		buffer.db = quotaDb();
		buffer.factory = factoryOf(quotaDb);

		await buffer.append(record(1)); // resolves — into memory
		expect(unavailable.map((e) => e.name)).toEqual(["QuotaExceededError"]);
		await buffer.append(record(2)); // dead is dead: memory from here on
		expect(unavailable.length).toBe(1);
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([1, 2]);
	});

	test("an open nobody answers is a failure too", async () => {
		// indexedDB.open fires `blocked` — not success, not error — when another connection holds
		// it up, and nothing at all when storage has stopped responding.
		const silent = { open: () => ({}) };
		const errors = [];
		const buffer = await openBuffer(silent, {
			timeoutMs: 30,
			onUnavailable: (e) => errors.push(e),
		});
		expect(errors.map((e) => e.name)).toEqual(["BufferTimeoutError"]);
		// and it degrades to memory rather than to nothing
		await buffer.append(record(1));
		expect((await buffer.claim(1_000_000)).records.length).toBe(1);
	});

	test("the in-memory fallback is reported, never silent", async () => {
		const errors = [];
		const reasons = [];
		await openBuffer(null, {
			onUnavailable: (e, reason) => {
				errors.push(e);
				reasons.push(reason);
			},
		});
		expect(errors.length).toBe(1);
		expect(String(errors[0])).toContain("no indexedDB");
		// A platform with no IndexedDB is a fact about the device, and the one case that may say so:
		// nothing this context does will ever be durable, and no teardown produces this.
		expect(reasons).toEqual([BUFFER_ABSENT]);
	});

	test("every connection loss heals — a reopen is never cached", async () => {
		const factory = new IDBFactory();
		const buffer = await openBuffer(factory);
		const kill = () => {
			buffer.db = {
				transaction: () => {
					const error = new Error("The database connection is closing.");
					error.name = "InvalidStateError";
					throw error;
				},
			};
		};
		for (const ts of [1, 2, 3]) {
			kill();
			await buffer.append(record(ts));
		}
		const batch = await buffer.claim(1_000_000);
		expect(batch.records.map((r) => r.event.timestamp)).toEqual([1, 2, 3]);
	});

	test("a failed reopen ends IndexedDB for this context — it falls to memory loudly, and the durable backlog stays for the next context", async () => {
		const factory = new IDBFactory();
		const unavailable = [];
		const reasons = [];
		const buffer = await openBuffer(factory, {
			onUnavailable: (e, reason) => {
				unavailable.push(e);
				reasons.push(reason);
			},
		});
		await buffer.append(record(1)); // durable before the fall

		buffer.db = {
			transaction: () => {
				const error = new Error("The database connection is closing.");
				error.name = "InvalidStateError";
				throw error;
			},
			close: () => {},
		};
		buffer.factory = {
			open: () => {
				const request = {};
				queueMicrotask(() => {
					request.error = new Error("storage is gone");
					request.onerror?.();
				});
				return request;
			},
		};

		await buffer.append(record(2)); // resolves — into memory
		expect(unavailable.map((e) => String(e))).toEqual([
			"Error: storage is gone",
		]);
		// An operation that did not complete, and nothing more: a page torn down mid-write reaches
		// this same path, so the report names the downgrade rather than the device.
		expect(reasons).toEqual([BUFFER_NOT_DURABLE]);
		expect(
			(await buffer.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([2]);

		// Nothing durable was touched by falling: a healthy successor context drains record 1.
		const successor = await openBuffer(factory);
		expect(
			(await successor.claim(1_000_000)).records.map((r) => r.event.timestamp),
		).toEqual([1]);
	});
});
