// How Locus puts rrweb at a moment — the one statement of it, loaded by the replay
// component (concatenated ahead of its contract) and by the screenshot renderer's harness.
// A moment shows every event stamped at or before it, and a recording shows a moment only
// when it holds one; a moment it cannot show is refused by whichever surface was asked.
window.LocusSeek = {
	// rrweb applies the events stamped strictly before the offset it is given, so a moment's
	// own events ride only on the millisecond after it.
	offsetOf(ms, startTs) {
		return ms - startTs + 1;
	},
	// Why a set of slices — {id, start_ts, end_ts, snapshot_ts} each, time-disjoint — holds
	// no page at the moment, or null when it holds one. Three shapes: outside the set; a
	// gap between two of its slices, when no context was recording; a slice's lead, after
	// it opened but before its page was first captured.
	unshowable(slices, ms) {
		const sorted = [...slices].sort((a, b) => a.start_ts - b.start_ts);
		const first = sorted[0],
			last = sorted[sorted.length - 1];
		if (ms < first.start_ts || ms > last.end_ts) {
			return `nothing recorded at t=${ms}; the recording spans t=${first.start_ts} to t=${last.end_ts}`;
		}
		for (let i = 0; i < sorted.length; i++) {
			const sl = sorted[i];
			if (ms < sl.start_ts) {
				const prev = sorted[i - 1];
				return `nothing recorded at t=${ms}; slice ${prev.id} ended at t=${prev.end_ts} and slice ${sl.id} opened at t=${sl.start_ts}`;
			}
			if (ms <= sl.end_ts) {
				if (ms < sl.snapshot_ts) {
					return `nothing captured at t=${ms}; slice ${sl.id} opened at t=${sl.start_ts} and its page was first captured at t=${sl.snapshot_ts}`;
				}
				return null;
			}
		}
		return null;
	},
	// The replayed document laid out complete: its fonts in and every image decoded or
	// failed, then a layout forced over them. A recorded scroll applied before this lands
	// where the page's height allowed at that instant, not where the visitor was. Layout is
	// read, never a painted frame awaited: a window behind others, or on a sleeping display,
	// paints no frames, and a frame awaited there never comes.
	async settled(doc) {
		if (doc.fonts) await doc.fonts.ready;
		await Promise.all(
			[...doc.images].map((img) => img.decode().catch(() => {})),
		);
		await new Promise((tick) => setTimeout(tick, 0));
		void doc.documentElement.offsetHeight;
	},
	// A document moved to a moment restarts every animation and transition the moment's
	// events set off — a drawer that slid in at the recorded moment is mid-slide again, at a
	// different offset on every seek. The moment wants their end states, so each finite one
	// is finished; one that repeats forever — a spinner — has no end state and keeps running.
	// Layout is forced first, since a transition exists only once style has been recalculated.
	finishAnimations(doc) {
		void doc.documentElement.offsetHeight;
		for (const a of doc.getAnimations()) {
			const timing = a.effect && a.effect.getTiming();
			if (timing && timing.iterations !== Infinity) a.finish();
		}
	},
	// Injected into the replayed document on every rebuild. A seeked scroll defers to the
	// page's own CSS scroll-behavior, and a page that declares smooth animates it — a frame
	// read before the animation ends holds an earlier scroll position; auto lands every
	// seeked scroll instantly, while playback passes its own smooth explicitly and keeps it.
	// Scroll anchoring moves the scroll position whenever content above the viewport
	// changes height, and in a replay that happens as the page's fonts and images arrive
	// after the recorded scroll was applied — a position no event ever recorded; off, the
	// document holds the scroll the events said.
	STYLE_RULES: [
		"* { scroll-behavior: auto !important; }",
		"* { overflow-anchor: none !important; }",
	],
};
