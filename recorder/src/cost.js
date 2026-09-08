/**
 * The recorder's cost accounting: what this page context spends of the host device — main-thread
 * time, upload bandwidth, heap, on-device backlog — kept as high-water marks and running totals,
 * and emitted as the cost_sample telemetry ping. The figures have to tell a broken recorder from an
 * idle one: a device holding a recording it cannot ship must never look like a visitor who did
 * nothing.
 */

export const COST_PING_INTERVAL_MS = 60_000;

/** Raise the cost record's heap high-water mark. performance.memory is Chromium-only; where it is absent nothing is written and the caller's unmeasured value stands. */
export function sampleHeap(cost) {
	const heap = globalThis.performance?.memory?.usedJSHeapSize;
	if (heap !== undefined && heap > (cost.heapBytesMax ?? 0))
		cost.heapBytesMax = heap;
}

/** Raise the cost record's backlog high-water mark from what the buffer last observed itself holding (buffer.js `bytes`). Free and synchronous, so it is sampled wherever the backlog moves — every write, every emit — not only in the drain loop, since a stalled drain is exactly when a device holds an undelivered recording. A buffer that has never observed its total leaves the mark unmeasured. */
export function sampleBacklog(cost, buffer) {
	const bytes = buffer?.bytes;
	if (typeof bytes === "number" && bytes > (cost.backlogBytesMax ?? -1)) {
		cost.backlogBytesMax = bytes;
	}
}

/**
 * One page context's cost meter. `cost` is the live record the capture and drain paths write into
 * (record() adds main-thread time, the uploader adds bytes; both mark it dirty); emit(buffer)
 * samples and sends one cost_sample ping; onVisibility(isVisible) keeps the foreground-visible
 * clock. sliceId is the context's page-load slice, fixed by the time the meter exists.
 */
export function costMeter({ telemetry, visitorId, recorderVersion, sliceId }) {
	// An unmeasured figure is null, never 0: backlogBytesMax opens null because nothing has looked
	// at the buffer yet, and the heap figures go out null wherever performance.memory does not
	// exist, which is everywhere outside Chromium. The store encodes null as an explicit unmeasured
	// sentinel (store/src/telemetry.js).
	const cost = {
		mainThreadMs: 0,
		uploadBytes: 0,
		deliveredBytes: 0,
		heapBytesMax: 0,
		backlogBytesMax: null,
		dirty: false,
	};

	// Cumulative foreground-visible time. A running total like mainThreadMs: each stint accrues
	// at its hidden transition, and a read adds the open stint, so the unconditional sample at
	// hidden always carries the stint that just ended.
	const visible = {
		accumMs: 0,
		since: document.visibilityState === "visible" ? performance.now() : null,
	};
	const visibleMs = () =>
		visible.accumMs +
		(visible.since === null ? 0 : performance.now() - visible.since);

	const emit = (buffer) => {
		sampleHeap(cost);
		sampleBacklog(cost, buffer);
		const memory = globalThis.performance?.memory;
		telemetry?.emit({
			metric: "cost_sample",
			visitorId,
			recorderVersion,
			sliceId,
			mainThreadMs: cost.mainThreadMs,
			uploadBytes: cost.uploadBytes,
			deliveredBytes: cost.deliveredBytes,
			heapBytesMax: memory ? cost.heapBytesMax : null,
			heapBytesLimit: memory?.jsHeapSizeLimit ?? null,
			backlogBytesMax: cost.backlogBytesMax,
			visibleMs: visibleMs(),
		});
		cost.dirty = false;
	};

	// Periodic sampling gates on work since the last sample: record() and a chunk upload mark the
	// cost dirty, any emit clears it, so an idle or backgrounded tab stops sampling rather than
	// pinging on a bare clock. The marker is explicit rather than a mainThreadMs delta because
	// coarsened performance.now() measures small work as 0ms, so mainThreadMs can sit flat across
	// real activity. The terminal sample at hidden is unconditional.
	const emitIfActive = (buffer) => {
		if (!cost.dirty) return;
		emit(buffer);
	};

	const onVisibility = (isVisible) => {
		if (isVisible) {
			visible.since ??= performance.now();
		} else if (visible.since !== null) {
			visible.accumMs += performance.now() - visible.since;
			visible.since = null;
		}
	};

	return { cost, emit, emitIfActive, onVisibility };
}
