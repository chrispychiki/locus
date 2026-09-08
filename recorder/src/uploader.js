/**
 * The drain loop: periodically move one claimed batch from the buffer through the sink, one gzipped chunk per slice. Single-flight (one request in progress, ever) and pending-first (an unresolved inflight batch from this or any earlier page context retries before new work is claimed). Compression runs off the main thread where the platform allows it (CompressionStream), so a large snapshot cannot freeze the host page on gzip.
 *
 * The drain interval ramps with the page context's age: base early, doubling each rampPeriodMs toward a cap. Backpressure overrides the ramp — a claim that leaves work behind, or any pending retry, resets it to base. The ramp relaxes delivery cadence only, never capture: every event commits to the buffer on emit (durable in IndexedDB while it is healthy; buffer.js), so a longer interval costs delivery latency and a wider crash-and-never-return window, nothing else.
 *
 * Oversize guard: the one chunk too big to store is unsplittable — a single FullSnapshot of a huge DOM (every other chunk is bounded below the store cap by the batch claim). It is caught client-side before upload, when its gzipped size exceeds maxGzippedChunkBytes, dropped via onOversize plus a non-counting error, and never retried: downstream, the rescue rule recovers a checkout snapshot's slice from prior continuity. The page-load slice's own snapshot is the exception — nothing downstream can rescue a slice that never had a covering snapshot, so the recorder stops rather than keep capturing what no one can replay: the fatal chunk alone is dropped, the batch's remaining chunks ship in this same pass, and onCatastrophic fires on the way out of the pass, whichever way it exits.
 *
 * Poison-batch guard: a batch the store refuses is retried first on every drain, so nothing behind it ships until it does. After maxAttempts cumulative refusals — the count durable in the buffer, surviving the reloads the batch is retried across — the uploader drops it and the queue advances. Only a store rejection spends an attempt (sink.js classifies): a send that got no answer indicts the network the page is on, not the batch, and it retries at the normal cadence for as long as the outage lasts — nothing else can ship during one either, so there is no queue to starve, and the buffer's bounded ring is what limits what an outage costs. No delivery failure counts toward the terminate threshold: a failing upload is not the host-page harm termination guards against.
 *
 * A claimed batch is untrusted input, persisted state a dying or buggy browser context may have written malformed. Records without the stamped shape are dropped loudly (a non-counting error, riding the payload error channel) and the rest of the batch ships. A fail() that reports the batch already gone means another page context sharing the buffer delivered and resolved it mid-flight — double-delivery working, not poison, so it neither drops nor faults. drain() never rejects; every failure lands in onError, so no scheduling path can throw into the host page.
 *
 * Two byte counters: uploadBytes is what this page context handed to the sink — the visitor's bandwidth, spent again by every retry and for nothing by every failure — and deliveredBytes is what the sink acknowledged.
 */
import { buildChunkPayloads, gzipBytes, serializePayload } from "./chunk.js";
import { sampleBacklog, sampleHeap } from "./cost.js";
import { describeError } from "./errors.js";
import { isStoreRejection } from "./sink.js";

/** One line naming what a dropped set of records was: per-slice counter ranges and counts, so a loss record carries the identity of the loss, not just its size. */
export function describeRecords(records) {
	const bySlice = new Map();
	for (const record of records) {
		const sliceId = record?.sliceId ?? "(unstamped)";
		const counter = record?.event?.counter;
		const s = bySlice.get(sliceId) ?? { count: 0, min: null, max: null };
		s.count += 1;
		if (typeof counter === "string") {
			if (s.min === null || counter < s.min) s.min = counter;
			if (s.max === null || counter > s.max) s.max = counter;
		}
		bySlice.set(sliceId, s);
	}
	return [...bySlice]
		.map(
			([sliceId, s]) =>
				`slice ${sliceId}: ${s.count} events${s.min !== null ? ` (counters ${s.min}–${s.max})` : ""}`,
		)
		.join("; ");
}

// Two caps, two jobs, two units. The batch cap bounds how much uncompressed event JSON one drain
// claims from the buffer — a working-set knob, decimal because nothing downstream reads it. The
// chunk cap bounds the gzipped bytes a store will accept, so it is binary, and it must equal the
// store's own cap (store/src/keys.js MAX_CHUNK_BYTES) or the recorder ships chunks the store
// refuses forever.
const MAX_UNCOMPRESSED_BATCH_BYTES = 10_000_000;
export const MAX_GZIPPED_CHUNK_BYTES = 10 * 1024 * 1024;

export class Uploader {
	/**
	 * @param {object} options
	 * @param {object} options.buffer        openBuffer() result
	 * @param {object} options.sink          sink contract (sink.js)
	 * @param {() => object} options.context this context's own facts per drain — the courier fallback for slices the buffer holds no capture context for, and the ship-time loss witness (chunk.js)
	 * @param {(error: unknown, opts?: {counts?: boolean}) => void} options.onError
	 * @param {object} [options.cost]        the cost record this drain loop samples into (heap, bytes pushed, bytes acknowledged, backlog high-water)
	 * @param {(descriptor: object, gzippedBytes: number) => void} [options.onOversize]  a chunk exceeded maxGzippedChunkBytes and can never be stored
	 * @param {(error: unknown) => void} [options.onUndeliverable]  a poison batch was dropped after maxAttempts store rejections — permanent data loss
	 * @param {(count: number) => void} [options.onErrorsDelivered]  the error records riding a chunk reached the store, so the recorder may forget them
	 * @param {(sliceId: string) => boolean} [options.isPageLoadSlice]  whether a slice is a page-load slice — an oversize FullSnapshot there is fatal to the recording, not merely a dropped chunk
	 * @param {(descriptor: object, gzippedBytes: number) => void} [options.onCatastrophic]  a page-load slice's FullSnapshot can never be stored, so the recording can never be replayable — the recorder stops rather than capture what it can't deliver
	 * @param {number} [options.intervalMs=3000]        base drain cadence
	 * @param {number} [options.maxIntervalMs=30000]    cadence cap as the page context ages
	 * @param {number} [options.rampPeriodMs=120000]    doubling period for the cadence; backpressure resets it
	 * @param {number} [options.maxUncompressedBatchBytes]  how much the uploader claims from the buffer per drain (default: MAX_UNCOMPRESSED_BATCH_BYTES)
	 * @param {number} [options.maxGzippedChunkBytes]  per-chunk gzipped cap; must equal the store's (default: MAX_GZIPPED_CHUNK_BYTES)
	 * @param {number} [options.maxAttempts=5]          durable bound on store rejections of one batch before it is dropped as poison
	 */
	constructor({
		buffer,
		sink,
		context,
		onError,
		cost = null,
		onOversize = null,
		onUndeliverable = null,
		onErrorsDelivered = null,
		isPageLoadSlice = null,
		onCatastrophic = null,
		intervalMs = 3_000,
		maxIntervalMs = 30_000,
		rampPeriodMs = 120_000,
		maxUncompressedBatchBytes = MAX_UNCOMPRESSED_BATCH_BYTES,
		maxGzippedChunkBytes = MAX_GZIPPED_CHUNK_BYTES,
		maxAttempts = 5,
	}) {
		this.buffer = buffer;
		this.sink = sink;
		this.context = context;
		this.onError = onError;
		this.cost = cost;
		this.onOversize = onOversize;
		this.onUndeliverable = onUndeliverable;
		this.onErrorsDelivered = onErrorsDelivered;
		this.isPageLoadSlice = isPageLoadSlice;
		this.onCatastrophic = onCatastrophic;
		this.intervalMs = intervalMs;
		this.maxIntervalMs = maxIntervalMs;
		this.rampPeriodMs = rampPeriodMs;
		this.maxUncompressedBatchBytes = maxUncompressedBatchBytes;
		this.maxGzippedChunkBytes = maxGzippedChunkBytes;
		this.maxAttempts = maxAttempts;
		this.inflight = null;
		this.running = false;
		this.timeoutId = null;
		this.startedAt = 0;
	}

	/** Drain delay for a given elapsed recording time: base interval, doubling each rampPeriodMs, capped at maxIntervalMs. */
	nextInterval(elapsedMs) {
		const grown =
			this.intervalMs * 2 ** Math.floor(elapsedMs / this.rampPeriodMs);
		return Math.min(this.maxIntervalMs, grown);
	}

	/** Drop the ramp back to base — called when the buffer is behind, so backpressure overrides the age-based slowdown. */
	resetRamp() {
		this.startedAt = Date.now();
	}

	sampleHeap() {
		if (this.cost) sampleHeap(this.cost);
	}

	sampleBacklog() {
		if (this.cost) sampleBacklog(this.cost, this.buffer);
	}

	start() {
		this.running = true;
		this.startedAt = Date.now();
		this.scheduleNext();
	}

	// The timer owns the cadence and re-arms before the pass it fires runs. Chaining the next tick
	// off the current pass's completion would make the loop only as durable as its most fragile
	// await: one pass that never settles would end delivery for the page. A tick landing on a pass
	// still in flight skips rather than stacks.
	scheduleNext() {
		if (!this.running) return;
		this.timeoutId = setTimeout(
			() => {
				this.scheduleNext();
				if (this.inflight) return;
				this.drain().catch((error) =>
					this.onError(error, { counts: false, fault: "drain_failed" }),
				);
			},
			this.nextInterval(Date.now() - this.startedAt),
		);
	}

	stop() {
		this.running = false;
		if (this.timeoutId !== null) clearTimeout(this.timeoutId);
		this.timeoutId = null;
	}

	/** Single-flight join point: one delivery pass runs at a time, and a call landing mid-pass awaits that pass instead of starting another. Resolves true when the pass delivered a batch and the buffer may still owe more — a pending retry cleared, or a claim that left work behind — which is the keep-going signal flush() loops on; false on an empty buffer or a failed delivery. Never rejects; every failure lands in onError. */
	drain() {
		if (!this.inflight) {
			this.inflight = this.drainOnce().finally(() => {
				this.inflight = null;
			});
		}
		return this.inflight;
	}

	/** Drain to empty now instead of waiting out the timer: repeated passes while each one delivers and leaves work behind. A pass that fails or finds nothing ends the loop, so flush never outruns the poison-batch discipline; the timer loop owns any residue. */
	async flush() {
		while (await this.drain()) {
			/* deliver until a pass comes back empty */
		}
	}

	async drainOnce() {
		let batch = null;
		// The fatal page-load-snapshot verdict, latched where it is detected and acted on in the
		// finally below: onCatastrophic must fire however the pass exits, and on the success path it
		// must fire after the resolve, because termination closes the buffer.
		let fatal = null;
		try {
			this.sampleBacklog();
			const pending = await this.buffer.claimed();
			const retrying = pending !== null;
			batch =
				pending ?? (await this.buffer.claim(this.maxUncompressedBatchBytes));
			if (!batch) return false;
			this.sampleHeap();
			const usable = [];
			const malformed = [];
			for (const record of batch.records) {
				const ok =
					typeof record?.sliceId === "string" &&
					typeof record?.event?.timestamp === "number" &&
					typeof record?.event?.counter === "string";
				(ok ? usable : malformed).push(record);
			}
			if (malformed.length > 0) {
				this.onError(
					new Error(
						`dropped ${malformed.length} malformed buffered records ` +
							`(batch ${batch.batchId}; ${describeRecords(malformed)})`,
					),
					{ counts: false },
				);
			} else if (retrying && batch.records.length === 0) {
				this.onError(
					new Error(`resolved unreadable inflight batch ${batch.batchId}`),
					{ counts: false },
				);
			}
			const t0 = performance.now();
			const ctx = this.context();
			const carriedErrors = ctx.errors?.length ?? 0;
			const chunks = buildChunkPayloads(usable, ctx, batch.contexts).map(
				({ descriptor, payload }) => ({
					descriptor,
					raw: serializePayload(payload),
				}),
			);
			if (this.cost) this.cost.mainThreadMs += performance.now() - t0;
			// The error strings ride the first chunk (chunk.js) and are cleared from the queue only
			// once that chunk is delivered; a failed or dropped carrier leaves them queued for the
			// next attempt.
			let errorsShipped = false;
			for (const { descriptor, raw } of chunks) {
				const bytes = await gzipBytes(raw);
				if (bytes.length > this.maxGzippedChunkBytes) {
					if (
						descriptor.hasFullSnapshot &&
						this.isPageLoadSlice?.(descriptor.sliceId)
					) {
						// Page-load snapshot over the cap: it is the slice's only covering snapshot, so
						// every event on this page is unreplayable and the recorder must stop — but only
						// this one chunk is undeliverable, so it alone is dropped (its accounting is the
						// catastrophic ping, not the ordinary oversize pair) and the batch's remaining
						// chunks ship: later slices carry their own covering snapshots. The fatal slice's
						// increments cannot mislead downstream — slice materialization discards a snapshotless slice,
						// and its rescue cannot mis-attribute them to another document because the slice id
						// is minted before its own Meta, which is what its new-document wall reads — and
						// they keep arriving through the outbox residue a successor context couriers
						// regardless, so suppressing this batch's would protect nothing.
						fatal = { descriptor, gzippedBytes: bytes.length };
						continue;
					}
					this.onOversize?.(descriptor, bytes.length);
					this.onError(
						new Error(
							`dropped oversize chunk: ${bytes.length} gzipped bytes exceed ` +
								`${this.maxGzippedChunkBytes} (slice ${descriptor.sliceId}, ` +
								`${descriptor.count} events, ` +
								`fullSnapshot=${descriptor.hasFullSnapshot})`,
						),
						{ counts: false },
					);
					continue;
				}
				if (this.cost) {
					this.cost.uploadBytes += bytes.length;
					this.cost.dirty = true;
				}
				await this.sink.send(bytes, descriptor);
				if (this.cost) this.cost.deliveredBytes += bytes.length;
				if (chunks[0].descriptor === descriptor) errorsShipped = true;
			}
			await this.buffer.resolve(batch.batchId);
			if (errorsShipped && carriedErrors > 0) {
				this.onErrorsDelivered?.(carriedErrors);
			}
			if (fatal) return false;
			if (retrying || batch.more) this.resetRamp();
			return retrying || batch.more;
		} catch (error) {
			if (error?.name === "BufferClosedError") {
				// Teardown closed the buffer mid-drain. Nothing is lost: a claimed batch stays
				// inflight for this or a successor context to retry, and the failure-accounting
				// writes below would themselves be refused by the closed buffer.
				return false;
			}
			try {
				// An attempt is spent only by a failure that indicts the batch. Anything else — a send
				// that never reached the store, a claim or a resolve the buffer refused — is the
				// environment failing around a batch that may well be fine, and counting it would
				// convert an outage into permanent loss.
				const attempts =
					batch && isStoreRejection(error)
						? await this.buffer.fail(batch.batchId)
						: 0;
				if (batch && attempts === Infinity) {
					// The batch is not in the store this asked. A concurrent page context delivering and
					// resolving it produces that, and so does a buffer that fell to memory mid-flight,
					// whose inflight map never held it. Neither loses records and neither makes the batch
					// poison, so the two need no telling apart here. Stay quiet.
				} else if (batch && attempts >= this.maxAttempts) {
					await this.buffer.resolve(batch.batchId);
					this.onUndeliverable?.(error);
					this.onError(
						new Error(
							`dropped undeliverable batch after ${this.maxAttempts} ` +
								`attempts (${describeRecords(batch.records)}): ` +
								describeError(error),
						),
						{ counts: false },
					);
				} else {
					this.onError(error, { counts: false, fault: "drain_failed" });
				}
			} catch (inner) {
				this.onError(inner, { counts: false, fault: "drain_failed" });
			}
			return false;
		} finally {
			// However the pass ended — remainder delivered and batch resolved, a send that failed, the
			// buffer closing under it — a latched fatal terminates the recorder. A batch a failure left
			// inflight is a successor context's to salvage: its own page-load slice is a different one,
			// so it drops the fatal chunk as ordinary oversize and delivers the rest.
			if (fatal) this.onCatastrophic?.(fatal.descriptor, fatal.gzippedBytes);
		}
	}
}
