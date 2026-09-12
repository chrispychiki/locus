/**
 * Local-first event buffer: IndexedDB, with an in-memory fallback where IndexedDB is unavailable or broken.
 *
 * Each event is one async structured write; an abrupt crash loses only the not-yet-committed write.
 *
 * Delivery is two-phase: claim() moves a batch outbox → inflight in one IndexedDB transaction, which the spec requires to be atomic (all writes commit together or none do: https://www.w3.org/TR/IndexedDB/#transaction-construct), the uploader sends it, resolve() deletes it on ack. A batch that never resolves is retried first on the next drain — including a drain in a later page context, which is what makes delayed delivery work at all. Each inflight batch carries a durable attempt count (fail() increments it), which the uploader spends against its poison cap (uploader.js). The count lives in the record because a poison batch survives page reloads, so an in-memory counter would reset before it ever tripped. Double-delivery is possible by design; hydration's content-hash dedup absorbs it.
 *
 * **A transaction body never awaits.** IndexedDB deactivates a transaction once control returns to the event loop, and the only sanctioned way to chain is to issue the next request from inside the previous one's success handler — same task, transaction still live. Awaiting the request itself survives that on some engines and not others; awaiting anything *around* it (an async helper, a Promise.all) survives on none of them by spec. So every body here is synchronous and callback-chained: it issues requests, hands results forward through `then`, and reports its value through `done`, and the transaction's own `oncomplete` settles the promise.
 *
 * **Every operation has a deadline, and a dead operation aborts its transaction.** A browser that force-closes the connection under storage pressure or page suspension may reject the in-flight requests, or drop them, firing neither success nor error — a promise that never settles. So silence past the deadline rejects like any other failure: the connection is presumed dead and the buffer reopens. The deadline aborts the transaction rather than abandoning it, because readwrite transactions on one store serialize across connections: a pending transaction — including one still queued behind the lock, and one whose page froze mid-flight — holds or waits on the store lock against every sibling and successor context of this visitor until it finishes, and abort releases that hold immediately even for a transaction that never started (verified in Chromium).
 *
 * Keys are minted by the writer, never by IndexedDB's key generator: several page contexts (tabs, rapid renavigations) share this one per-visitor database over separate connections, and the generator's state is cached per connection, so concurrent connections can be handed colliding keys (a ConstraintError on a plain add — observed in Chromium). Every key is a string ordered timestamp-major and suffixed with a per-context tag plus a per-instance sequence, so keys are collision-free by construction and cursor order stays arrival order. The one add that can collide is an append's own retry, when a first attempt whose outcome was hidden — deadline passed, or completion without the request's success — had committed. That ConstraintError is the commit's witness and resolves the append as stored (addOwn).
 *
 * An outbox key also carries the record's serialized byte size, in a marked final segment, so measuring a stored record never re-serializes it; a row whose key carries no size (another bundle's write) is recounted where its size is needed. The size lives in the key and nowhere else because the key is the one part of the row a sibling context never interprets: the record value must stay the verbatim record — it flows into chunks, and the contexts sharing this database need not all run this bundle, so a changed value shape would be unreadable to the others — while keys are only ordered by and deleted by.
 *
 * **Any operation failure is treated as the connection's death.** Whatever the error's name — a close event, an InvalidStateError, a deadline, a quota refusal, an internal abort — the buffer drops the connection (closing whatever remains of it, so it never leaks open against another context's versionchange), reopens, and retries the operation once against the fresh one. Reopens are deduplicated only while one is in flight; a failed reopen is never cached, or the first dead connection would poison every operation for the rest of the page's life.
 *
 * **A buffer that fails twice degrades to memory.** One retry is the whole allowance, because the reopen heals only a lost connection: a wedged database hands the fresh connection's transaction to the same lock queue (verified in Chromium), a full quota stays full, an I/O fault stays faulty. So a second failure of any kind — or a failed reopen — ends this context's use of IndexedDB: the buffer falls to a MemoryBuffer for the rest of the page's life, reports the downgrade through onUnavailable, and reruns the interrupted operation against memory. Whatever already sits in IndexedDB stays for a later healthy context to drain. BufferClosedError is exempt.
 *
 * **Teardown closes the buffer.** A frozen page's event loop stops and its deadline timers stop with it, so the deadline cannot release what a leaving page left pending. close() (stop, pagehide, freeze) aborts the in-flight transactions and the connection, and operations arriving while closed reject as BufferClosedError without touching IndexedDB. resume() (pageshow, lifecycle resume) reopens; the connection itself reopens lazily on the next operation.
 *
 * **A BufferClosedError proves teardown; nothing else disproves it.** The browser kills the connection at navigation commit on its own schedule, and a pending operation's rejection can reach its handler before this context's pagehide listener has run — so close() is what this page does about teardown, never how it recognizes one. Every other failure is an operation that did not complete, for reasons a page cannot distinguish from inside itself, and nothing here may name one.
 *
 * The backlog is a ring. At maxBacklogBytes an append evicts the oldest outbox records to admit the new, and reports them through onEvict. Eviction touches only the outbox; inflight batches belong to the uploader's poison cap.
 *
 * The byte total is a durable counter in the meta store, updated inside the same atomic transaction as every mutation it accounts (append and eviction adjust it, resolve subtracts the batch size claim() stamped on the inflight row, claim itself is net zero) — exact across concurrent contexts by construction, and left on `bytes` by every transaction that moves it, so counting never deserializes the buffer. A database that carries no counter yet is seeded by one full recount at open, off every operation's deadlined path — a recount is the one buffer job whose cost scales with the backlog, and riding it on the first operation would spend that operation's deadline on exactly the slow-device/big-backlog combination durability exists for. A failed seed just leaves the counter absent: the next operation that reads the counter under full-store scope recounts inline, and an append that finds no counter admits its record unaccounted — every row counts the same whenever the recount runs, so the ledger stays exact — then kicks one reseed, because an unaccounted append is also an unenforced ring and nothing else is guaranteed to recount while delivery is failing, which is exactly when the backlog grows.
 *
 * Drift is still possible: one visitor's tabs may run different recorder bundles over this one database, and any of them couriers any batch, so a context that does not keep the counter can move bytes the ledger never sees. The dangerous direction is such a context resolving a counted batch — its bytes strand in the counter and ratchet toward a phantom-full ring that evicts everything an append admits. The heal lives in claim(): finding both stores empty is the one moment the true total is provably zero, so a nonzero counter there is overwritten with 0.
 *
 * **The buffer carries each slice's capture context beside its records.** A chunk's visitor id and envelope are facts about the moment of capture, but chunks are assembled at ship time by whichever context drains the backlog — possibly days later, possibly under a different identity. So a writer sets `captureContext` ({visitorId, envelope}) once, and every append puts it into the meta store keyed by the record's slice id, inside the append's own transaction — the row and its records commit or vanish together, so a stored record's context row is present by construction. The put is unconditional per append rather than once per slice because the drained-empty wipe below can remove a live slice's row, and only a re-put on the next append restores it. claim() and claimed() hand the rows for their batch's slices back as `contexts`, read in the same transaction as the records; a slice without a row (another bundle's write, a pre-stamp backlog) is the courier-reconstruction fallback, chunk.js's business. The rows are wiped at the drained-empty moment claim() already owns: with both record stores empty every context row is provably an orphan, and without the wipe a long-lived visitor database would accrete one row per slice forever.
 *
 * Every transaction that moves the total leaves it on `bytes`, readable synchronously at the moment it changes. On the IndexedDB buffer `bytes` is null until a transaction first observes the total — no reading taken, which is a different fact from an empty buffer; the open-time seed is such an observation, so the null window normally lasts only until it settles (`seeded`). The memory buffer is its own ledger, so it opens at 0.
 */

const DB_NAME = "locus-recorder";
export const BYTES_KEY = "backlogBytes";
const STORES = ["outbox", "inflight", "meta"];

// Capture-context rows share the meta store with the byte counter, under a key prefix no slice id
// can collide with (BYTES_KEY carries no dash and no prefix).
const CONTEXT_KEY_PREFIX = "context-";
const contextKey = (sliceId) => CONTEXT_KEY_PREFIX + sliceId;

/** Collect the capture-context rows for a set of records' slices, inside the transaction that read the records. Chained gets, one per distinct slice — a batch rarely spans more than a few. Slices without a row are absent from the result. */
function attachContexts(records, tx, done) {
	const sliceIds = [
		...new Set(
			records
				.map((record) => record?.sliceId)
				.filter((sliceId) => typeof sliceId === "string"),
		),
	];
	const contexts = {};
	const step = (i) => {
		if (i === sliceIds.length) return done(contexts);
		then(tx.objectStore("meta").get(contextKey(sliceIds[i])), (row) => {
			if (row !== undefined) contexts[sliceIds[i]] = row;
			step(i + 1);
		});
	};
	step(0);
}

/**
 * What onUnavailable is reporting, and the whole of what it may claim. Only one of the two is a fact about the device.
 *
 * BUFFER_ABSENT: the platform exposes no indexedDB at all, so nothing this context captures will ever be durable. Knowable, permanent, and reached by no other route.
 *
 * BUFFER_NOT_DURABLE: an IndexedDB operation did not complete and capture has moved to memory. A broken database and a page torn down mid-operation both arrive here (see the teardown note above), so the name goes no further than the downgrade itself. Telling the two apart belongs to the rates plane, which holds the evidence a page cannot: a broken database keeps reporting while its context goes on living, and a teardown's report is the last thing that visitor ever sends.
 */
export const BUFFER_ABSENT = "buffer_unavailable";
export const BUFFER_NOT_DURABLE = "buffer_not_durable";

/** The ring's ceiling — the footprint a broken delivery path may hold on a visitor's device. Both buffers open at it; a deployment raises or lowers it through start()'s maxBacklogBytes. */
export const MAX_BACKLOG_BYTES = 50 * 1024 * 1024;

/** How long any one buffer operation may take before the connection is presumed dead. Generous against a slow device under load, finite against a connection that has stopped answering. */
export const OPERATION_TIMEOUT_MS = 10_000;

/** A buffer operation that did not answer before its deadline. Silence is all that was observed — a wedged database, a device under load, and a page being torn down all produce it. The deadline is a timer, throttled in background tabs and stopped under freeze, so how long the operation actually had is unknown here. */
export class BufferTimeoutError extends Error {
	constructor(what) {
		super(
			`locus-recorder: IndexedDB ${what} did not answer before its deadline — the connection is presumed dead`,
		);
		this.name = "BufferTimeoutError";
	}
}

/** An operation stopped by page teardown. A transaction that had already committed but not yet delivered its completion event is unaffected by the abort — its write is durable — so this error means "not known to have committed", never "known lost". */
export class BufferClosedError extends Error {
	constructor(what) {
		super(
			`locus-recorder: buffer closed during ${what} — page teardown aborts still-pending ` +
				"IndexedDB work rather than leave a transaction holding the store lock",
		);
		this.name = "BufferClosedError";
	}
}

const contextTag = () =>
	Math.floor(Math.random() * 36 ** 4)
		.toString(36)
		.padStart(4, "0");

/** Issue a request and hand its result to the next step — the sanctioned chaining point, because the step runs inside the request's own success handler, where the transaction is still live. A request that errors goes unhandled on purpose: IndexedDB aborts the transaction, and the transaction's onabort surfaces it. */
function then(request, step) {
	request.onsuccess = () => step(request.result);
}

/** Add under a key only this instance can mint, and tell the next step whether the row was already there. A ConstraintError is this append's earlier attempt having committed: consumed (preventDefault keeps the transaction alive) and reported as stored(true). Any other request error stays unhandled, so the transaction aborts and onabort surfaces it. */
function addOwn(store, value, key, stored) {
	const request = store.add(value, key);
	request.onsuccess = () => stored(false);
	request.onerror = (event) => {
		if (request.error?.name !== "ConstraintError") return;
		event.preventDefault();
		stored(true);
	};
}

export function recordBytes(record) {
	return new TextEncoder().encode(JSON.stringify(record)).length;
}

// The marked size segment an outbox key carries (see the key note at the top). The marker is what
// keeps parsing unambiguous: a key without a size ends in the bare numeric sequence, which a bare
// numeric tail could not be told apart from.
const KEY_SIZE_SUFFIX = /-s(\d+)$/;

/** A stored outbox row's byte size: read off the key's size segment where the writer stamped one, recounted from the value where it didn't — a generator-numbered or differently-shaped key is another bundle's write, priced the slow way. */
function rowBytes(key, record) {
	const match = typeof key === "string" ? KEY_SIZE_SUFFIX.exec(key) : null;
	return match ? Number(match[1]) : recordBytes(record);
}

/** The durable byte total, read inside an open transaction spanning meta and both record stores (the latter for the recount of a database whose counter is absent). The recount prices each outbox row off its key stamp and each inflight batch off its row stamp, deserializing only rows a pre-stamp context wrote. */
function countedBytes(tx, done) {
	then(tx.objectStore("meta").get(BYTES_KEY), (existing) => {
		if (typeof existing === "number") return done(existing);
		let total = 0;
		const cursor = tx.objectStore("outbox").openCursor();
		cursor.onsuccess = () => {
			const current = cursor.result;
			if (current) {
				total += rowBytes(current.key, current.value);
				return current.continue();
			}
			then(tx.objectStore("inflight").getAll(), (inflight) => {
				for (const batch of inflight) {
					if (typeof batch?.bytes === "number") {
						total += batch.bytes;
						continue;
					}
					for (const record of batch?.records ?? [])
						total += recordBytes(record);
				}
				done(total);
			});
		};
	});
}

class IdbBuffer {
	constructor(
		db,
		factory,
		timeoutMs = OPERATION_TIMEOUT_MS,
		onUnavailable = null,
	) {
		this.db = db;
		this.factory = factory;
		this.timeoutMs = timeoutMs;
		this.onUnavailable = onUnavailable;
		this.tag = contextTag();
		this.seq = 0;
		this.connectionLost = false;
		this.reopening = null;
		this.closed = false;
		this.fallen = null;
		this.liveTransactions = new Set();
		this.maxBacklogBytes = MAX_BACKLOG_BYTES;
		this.onEvict = null;
		this.captureContext = null;
		this.observedBytes = null;
		this.watchConnection(db);
		this.reseeding = null;
		this.seeded = this.seedCounter();
	}

	/**
	 * The open-time counter seed. A cheap read-only probe answers the common case (the counter exists); only a database that carries none pays the full recount, in its own transaction here rather than inside a deadlined operation. Failure is swallowed whole — never a fall, never a report — because absence of the counter is a state every counter-reading operation already handles, and a page torn down mid-seed would otherwise demote a healthy buffer.
	 */
	async seedCounter() {
		try {
			const existing = await this.transact(
				"seed",
				["meta"],
				"readonly",
				(tx, done) => {
					then(tx.objectStore("meta").get(BYTES_KEY), done);
				},
			);
			if (typeof existing === "number") {
				this.bytes = existing;
				return;
			}
			const total = await this.transact(
				"seed",
				STORES,
				"readwrite",
				(tx, done) => {
					countedBytes(tx, (counted) => {
						tx.objectStore("meta").put(counted, BYTES_KEY);
						done(counted);
					});
				},
			);
			this.bytes = total;
		} catch {
			/* the counter stays absent; a full-scope operation seeds inline */
		}
	}

	/** The last observed backlog total — the fallen buffer's live ledger once the buffer has fallen. A write landing after the fall is a straggling IndexedDB reading, stale by definition once the backend is memory, and is discarded. */
	get bytes() {
		return this.fallen ? this.fallen.bytes : this.observedBytes;
	}

	set bytes(total) {
		if (!this.fallen) this.observedBytes = total;
	}

	watchConnection(db) {
		db.onclose = () => {
			this.connectionLost = true;
		};
	}

	/** A connection arriving after close() is shut on arrival — teardown must not resurrect it. The outgoing connection is closed before the swap: where the fault was not the connection's own death it is still open, and left unreferenced it would sit on the database for the rest of the page's life. */
	reopen() {
		this.reopening ??= openDb(this.factory, this.timeoutMs)
			.then((db) => {
				if (this.closed) {
					try {
						db.close();
					} catch {
						/* nothing held; nothing to release */
					}
					throw new BufferClosedError("reopen");
				}
				try {
					this.db.close();
				} catch {
					/* often already dead; harmless when not */
				}
				this.db = db;
				this.connectionLost = false;
				this.watchConnection(db);
			})
			.finally(() => {
				this.reopening = null;
			});
		return this.reopening;
	}

	async run(what, op) {
		if (this.closed) throw new BufferClosedError(what);
		if (!this.connectionLost) {
			try {
				return await op();
			} catch (error) {
				if (this.closed || this.fallen || error?.name === "BufferClosedError")
					throw error;
				this.connectionLost = true;
			}
		}
		try {
			await this.reopen();
		} catch (error) {
			if (!this.closed && !this.fallen) this.fall(error);
			throw error;
		}
		try {
			return await op();
		} catch (error) {
			if (!this.closed && !this.fallen && error?.name !== "BufferClosedError") {
				this.fall(error);
			}
			throw error;
		}
	}

	/**
	 * Run one transaction. `body(tx, done)` is synchronous: it issues its requests, chains them through `then`, and calls `done(value)` with what the transaction is worth. Nothing awaits — see the note at the top of this file.
	 *
	 * The transaction runs under this buffer's deadline and is registered for teardown: past the deadline it is aborted — releasing the store lock, or its place in line for it — and the operation rejects without waiting for onabort, because a backend hung enough to answer nothing may never deliver the abort event either. A failed request is read at onabort: the transaction's error event fires before its `error` slot is set, and the abort that follows is where the error is readable. Rejections carry a real error even where IndexedDB offers none: an aborted transaction's `error` is legitimately null, and null must never reach an error report.
	 *
	 * A transaction that completes without the body ever reaching `done` is a failure, not a value: every body reports through the success handler of its last request, so completion with no report means the backend ran the transaction without delivering a request's success event to its handler — a backend answering nothing, the same condition as the deadline, and it is rejected as one. `done(undefined)` is a legitimate report and stays one.
	 */
	transact(what, stores, mode, body) {
		return new Promise((resolve, reject) => {
			let tx;
			try {
				tx = this.db.transaction(stores, mode);
			} catch (error) {
				reject(error);
				return;
			}
			let result;
			let reported = false;
			let settled = false;
			let timer = null;
			const settle = (act) => {
				if (settled) return;
				settled = true;
				clearTimeout(timer);
				this.liveTransactions.delete(live);
				act();
			};
			const live = {
				what,
				abort: (reason) => {
					settle(() => reject(reason));
					try {
						tx.abort();
					} catch {
						/* already committing or aborted */
					}
				},
			};
			this.liveTransactions.add(live);
			timer = setTimeout(
				() => live.abort(new BufferTimeoutError(what)),
				this.timeoutMs,
			);
			tx.oncomplete = () =>
				settle(() =>
					reported
						? resolve(result)
						: reject(
								new Error(
									`locus-recorder: IndexedDB ${what} transaction completed without delivering a request result to its handler — the connection is presumed dead`,
								),
							),
				);
			tx.onabort = () =>
				settle(() =>
					reject(
						tx.error ??
							new Error(
								`locus-recorder: IndexedDB ${what} transaction aborted with no error object`,
							),
					),
				);
			try {
				body(tx, (value) => {
					reported = true;
					result = value;
				});
			} catch (error) {
				settle(() => reject(error));
				try {
					tx.abort();
				} catch {
					/* already dead; the rejection above stands */
				}
			}
		});
	}

	fall(error) {
		if (this.fallen) return;
		const fallen = new MemoryBuffer();
		fallen.maxBacklogBytes = this.maxBacklogBytes;
		fallen.onEvict = this.onEvict;
		fallen.captureContext = this.captureContext;
		this.fallen = fallen;
		for (const live of [...this.liveTransactions]) live.abort(error);
		try {
			this.db.close();
		} catch {
			/* may already be shut; nothing left to release */
		}
		this.onUnavailable?.(error, BUFFER_NOT_DURABLE);
	}

	async dispatch(idbOp, memoryOp) {
		if (this.fallen) return memoryOp(this.fallen);
		try {
			return await idbOp();
		} catch (error) {
			if (this.fallen) return memoryOp(this.fallen);
			throw error;
		}
	}

	close() {
		this.closed = true;
		this.connectionLost = true;
		for (const live of [...this.liveTransactions]) {
			live.abort(new BufferClosedError(live.what));
		}
		try {
			this.db.close();
		} catch {
			/* already gone */
		}
	}

	resume() {
		this.closed = false;
	}

	nextKey(prefix) {
		this.seq += 1;
		return `${prefix}-${this.tag}-${String(this.seq).padStart(6, "0")}`;
	}

	append(record) {
		return this.dispatch(
			async () => {
				const size = recordBytes(record);
				const key = `${this.nextKey(String(record.event.counter))}-s${size}`;
				// The eviction tally is built inside the transaction body: a lost-connection retry
				// reruns the body against a rolled-back store, so a tally hoisted outside it would
				// double-count the first attempt's undone deletions.
				const { evicted, evictedBytes, total } = await this.run("append", () =>
					this.transact(
						"append",
						["outbox", "meta"],
						"readwrite",
						(tx, done) => {
							const outbox = tx.objectStore("outbox");
							// The slice's capture context commits with the record or not at all, so a stored
							// record's context row is never missing by a race.
							if (this.captureContext && typeof record?.sliceId === "string") {
								tx.objectStore("meta").put(
									this.captureContext,
									contextKey(record.sliceId),
								);
							}
							then(tx.objectStore("meta").get(BYTES_KEY), (base) => {
								// No counter to account against — and inflight, which a recount needs, is
								// deliberately outside this transaction's scope. Admit the record and leave the
								// counter absent: the next recount counts this row like any other, so the ledger
								// stays exact, and committing here keeps appends ordered ahead of any claim
								// issued after them. The ring check sits out this one append — no base to
								// measure against.
								if (typeof base !== "number") {
									return addOwn(outbox, record, key, () =>
										done({ evicted: [], evictedBytes: 0, total: null }),
									);
								}
								// The add goes first, because whether the row was already there decides what the
								// ring must make room for: a row already stored was counted when it was stored, so
								// only a new row adds its size, and only this attempt's evictions move the counter.
								addOwn(outbox, record, key, (already) => {
									const added = already ? 0 : size;
									const evicted = [];
									let evictedBytes = 0;
									const settle = () => {
										const total = base - evictedBytes + added;
										tx.objectStore("meta").put(total, BYTES_KEY);
										done({ evicted, evictedBytes, total });
									};
									const room = () =>
										base - evictedBytes + added <= this.maxBacklogBytes;
									if (room()) return settle();
									const cursor = outbox.openCursor();
									cursor.onsuccess = () => {
										const current = cursor.result;
										if (!current || room()) return settle();
										if (current.key === key) return current.continue();
										evicted.push(current.value);
										evictedBytes += rowBytes(current.key, current.value);
										current.delete();
										current.continue();
									};
								});
							});
						},
					),
				);
				if (total !== null) this.bytes = total;
				// An unaccounted append means the ring went unenforced for that row, and nothing else is
				// guaranteed to recount soon: resolve() recounts only when delivery succeeds, and delivery
				// failing is exactly when the backlog grows. Reseeding here keeps the unenforced state
				// transient — appends, not the page's life.
				else this.reseed();
				if (evicted.length > 0) this.onEvict?.(evicted, evictedBytes);
			},
			(fallen) => fallen.append(record),
		);
	}

	/** One reseed in flight at a time; failure is swallowed like the open-time seed's, and the next unaccounted append just asks again. */
	reseed() {
		this.reseeding ??= this.seedCounter().finally(() => {
			this.reseeding = null;
		});
	}

	/** Atomically move up to maxBytes of outbox records into one inflight batch. Returns {batchId, records, more} (more = outbox work remained past the cap) or null when the outbox is empty. */
	claim(maxBytes) {
		return this.dispatch(
			async () => {
				let drained = false;
				const batch = await this.run("claim", () =>
					this.transact("claim", STORES, "readwrite", (tx, done) => {
						// Minted per attempt. A retry after a hidden commit finds the outbox drained and
						// returns null, the committed batch waiting in inflight for the next drain — unless
						// a sibling context appended meanwhile, in which case the retry claims those records
						// as a new batch; under the first attempt's key that add would collide with the
						// committed row and demote the page to memory.
						const batchKey = this.nextKey(String(Date.now()).padStart(14, "0"));
						const outbox = tx.objectStore("outbox");
						const records = [];
						const keys = [];
						let bytes = 0;
						let more = false;
						drained = false;

						const empty = () => {
							// Both stores empty is the one moment the true total is provably zero, so any
							// other counter reading here is drift and is overwritten. It is equally the one
							// moment every capture-context row is provably an orphan, so they are wiped here
							// too; a live slice's next append re-puts its row in the same transaction as the
							// record, so nothing stored ever sits rowless.
							then(tx.objectStore("inflight").count(), (inflightCount) => {
								if (inflightCount !== 0) return done(null);
								const meta = tx.objectStore("meta");
								then(meta.get(BYTES_KEY), (counted) => {
									if (counted !== 0) meta.put(0, BYTES_KEY);
									const rows = meta.openCursor();
									rows.onsuccess = () => {
										const current = rows.result;
										if (!current) {
											drained = true;
											return done(null);
										}
										if (
											typeof current.key === "string" &&
											current.key.startsWith(CONTEXT_KEY_PREFIX)
										) {
											current.delete();
										}
										current.continue();
									};
								});
							});
						};

						const commit = () => {
							for (const key of keys) outbox.delete(key);
							// The batch's byte size rides the row so resolve() can subtract it from the
							// durable counter without re-serializing the records. The move itself is net
							// zero — outbox bytes become inflight bytes.
							then(
								tx
									.objectStore("inflight")
									.add({ records, attempts: 0, bytes }, batchKey),
								(batchId) =>
									attachContexts(records, tx, (contexts) =>
										done({ batchId, records, contexts, more }),
									),
							);
						};

						const cursor = outbox.openCursor();
						cursor.onsuccess = () => {
							const current = cursor.result;
							if (!current) return records.length === 0 ? empty() : commit();
							const size = rowBytes(current.key, current.value);
							if (records.length > 0 && bytes + size > maxBytes) {
								more = true;
								return commit();
							}
							records.push(current.value);
							keys.push(current.key);
							bytes += size;
							current.continue();
						};
					}),
				);
				if (drained) this.bytes = 0;
				return batch;
			},
			(fallen) => fallen.claim(maxBytes),
		);
	}

	/** The oldest unresolved batch, or null — retried before new claims, one at a time, because a drain retries exactly one delivery. A batch row is persisted state another (possibly dying, possibly buggy) context wrote, so its shape is untrusted: a row without a records array surfaces as an empty batch for the drain to resolve away, never a throw, which would wedge every batch behind it. */
	claimed() {
		return this.dispatch(
			() =>
				this.run("claimed", () =>
					this.transact(
						"claimed",
						["inflight", "meta"],
						"readonly",
						(tx, done) => {
							then(tx.objectStore("inflight").openCursor(), (cursor) => {
								if (!cursor) return done(null);
								const records = Array.isArray(cursor.value?.records)
									? cursor.value.records
									: [];
								attachContexts(records, tx, (contexts) =>
									done({ batchId: cursor.key, records, contexts }),
								);
							});
						},
					),
				),
			(fallen) => fallen.claimed(),
		);
	}

	resolve(batchId) {
		return this.dispatch(
			async () => {
				const total = await this.run("resolve", () =>
					this.transact("resolve", STORES, "readwrite", (tx, done) => {
						const store = tx.objectStore("inflight");
						then(store.get(batchId), (row) => {
							if (!row) return done(null); // already resolved elsewhere — the counter was adjusted there
							let bytes = row.bytes;
							if (typeof bytes !== "number") {
								// The row is persisted state and its size stamp may be missing; recount rather
								// than subtract a number that isn't there.
								bytes = 0;
								for (const record of row.records ?? [])
									bytes += recordBytes(record);
							}
							countedBytes(tx, (base) => {
								const remaining = Math.max(0, base - bytes);
								tx.objectStore("meta").put(remaining, BYTES_KEY);
								store.delete(batchId);
								done(remaining);
							});
						});
					}),
				);
				if (total !== null) this.bytes = total;
			},
			(fallen) => fallen.resolve(batchId),
		);
	}

	/** Count one more failed delivery of a batch; returns its new cumulative attempt count (Infinity if it is already gone — a resolve may have landed from another page context sharing this buffer). Durable, so the give-up decision survives the reloads a poison batch is otherwise retried across. */
	fail(batchId) {
		return this.dispatch(
			() =>
				this.run("fail", () =>
					this.transact("fail", ["inflight"], "readwrite", (tx, done) => {
						const store = tx.objectStore("inflight");
						then(store.get(batchId), (record) => {
							if (!record) return done(Infinity);
							record.attempts = (record.attempts ?? 0) + 1;
							then(store.put(record, batchId), () => done(record.attempts));
						});
					}),
				),
			(fallen) => fallen.fail(batchId),
		);
	}
}

class MemoryBuffer {
	constructor() {
		this.outbox = [];
		this.inflight = new Map();
		this.nextBatchId = 1;
		this.maxBacklogBytes = MAX_BACKLOG_BYTES;
		this.onEvict = null;
		this.captureContext = null;
		this.contexts = new Map();
		this.bytes = 0;
		this.seeded = Promise.resolve(); // its ledger is itself; there is nothing to read in
	}

	sliceContexts(records) {
		const contexts = {};
		for (const record of records) {
			const sliceId = record?.sliceId;
			if (typeof sliceId === "string" && this.contexts.has(sliceId)) {
				contexts[sliceId] = this.contexts.get(sliceId);
			}
		}
		return contexts;
	}

	async append(record) {
		const size = recordBytes(record);
		const evicted = [];
		let evictedBytes = 0;
		while (
			this.outbox.length > 0 &&
			this.bytes - evictedBytes + size > this.maxBacklogBytes
		) {
			const victim = this.outbox.shift();
			evicted.push(victim);
			evictedBytes += recordBytes(victim);
		}
		this.outbox.push(record);
		if (this.captureContext && typeof record?.sliceId === "string") {
			this.contexts.set(record.sliceId, this.captureContext);
		}
		this.bytes += size - evictedBytes;
		if (evicted.length > 0) this.onEvict?.(evicted, evictedBytes);
	}

	async claim(maxBytes) {
		if (this.outbox.length === 0) {
			if (this.inflight.size === 0) this.contexts.clear();
			return null;
		}
		const records = [];
		let bytes = 0;
		while (this.outbox.length > 0) {
			const size = recordBytes(this.outbox[0]);
			if (records.length > 0 && bytes + size > maxBytes) break;
			records.push(this.outbox.shift());
			bytes += size;
		}
		const batchId = this.nextBatchId++;
		this.inflight.set(batchId, { records, attempts: 0 });
		return {
			batchId,
			records,
			contexts: this.sliceContexts(records),
			more: this.outbox.length > 0,
		};
	}

	async claimed() {
		const first = this.inflight.entries().next().value;
		return first
			? {
					batchId: first[0],
					records: first[1].records,
					contexts: this.sliceContexts(first[1].records),
				}
			: null;
	}

	async resolve(batchId) {
		const batch = this.inflight.get(batchId);
		if (batch) {
			for (const record of batch.records) this.bytes -= recordBytes(record);
		}
		this.inflight.delete(batchId);
	}

	async fail(batchId) {
		const batch = this.inflight.get(batchId);
		if (!batch) return Infinity;
		batch.attempts = (batch.attempts ?? 0) + 1;
		return batch.attempts;
	}

	/** Teardown and resume are part of the buffer contract; memory holds no lock and outlives nothing, so both are moot here. */
	close() {}

	resume() {}
}

/** Open the database, under a deadline like every other operation. `blocked` — an open another connection is holding up — answers immediately rather than waiting, since without that it is one more promise that never settles. An open request cannot be aborted, so one that outlives its deadline is not abandoned bare: a connection the backend delivers late (observed in Chromium when the open queues behind another context's stalled versionchange open) is closed on arrival rather than leaked to sit on the database for the rest of the page's life. */
function openDb(idbFactory, timeoutMs = OPERATION_TIMEOUT_MS) {
	return new Promise((resolve, reject) => {
		let settled = false;
		let timer = null;
		const settle = (act) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			act();
		};
		const open = idbFactory.open(DB_NAME, 1);
		timer = setTimeout(
			() => settle(() => reject(new BufferTimeoutError("open"))),
			timeoutMs,
		);
		open.onupgradeneeded = () => {
			// The generator is never used — every key is writer-minted, and an explicit key
			// overrides it — but the stores are created with one anyway: a store without a key
			// generator DataErrors on any keyless add, and this database is shared by whatever
			// recorder bundles the visitor's tabs are running, which need not all mint keys.
			open.result.createObjectStore("outbox", { autoIncrement: true });
			open.result.createObjectStore("inflight", { autoIncrement: true });
			open.result.createObjectStore("meta");
		};
		open.onsuccess = () => {
			if (settled) {
				try {
					open.result.close();
				} catch {
					/* delivered dead; nothing to release */
				}
				return;
			}
			settle(() => resolve(open.result));
		};
		open.onerror = () =>
			settle(() =>
				reject(
					open.error ??
						new Error(
							"locus-recorder: IndexedDB open failed with no error object",
						),
				),
			);
		open.onblocked = () =>
			settle(() => reject(new Error("locus-recorder: IndexedDB open blocked")));
	});
}

/**
 * The buffer for this page context: IndexedDB where it works, memory where it does not. Memory dies with the page, so a visitor buffering to it loses everything they recorded unless it ships before they leave.
 *
 * `onUnavailable(error, reason)` is called wherever that happens — at open, and again mid-life if a working buffer later falls to memory. The reason separates the platform having no IndexedDB from an operation that did not complete; see BUFFER_ABSENT / BUFFER_NOT_DURABLE above.
 */
export async function openBuffer(
	idbFactory = globalThis.indexedDB,
	{ onUnavailable = null, timeoutMs = OPERATION_TIMEOUT_MS } = {},
) {
	if (!idbFactory) {
		onUnavailable?.(
			new Error("locus-recorder: no indexedDB on this platform"),
			BUFFER_ABSENT,
		);
		return new MemoryBuffer();
	}
	try {
		return new IdbBuffer(
			await openDb(idbFactory, timeoutMs),
			idbFactory,
			timeoutMs,
			onUnavailable,
		);
	} catch (error) {
		onUnavailable?.(error, BUFFER_NOT_DURABLE);
		return new MemoryBuffer();
	}
}
