// The Locus replay contract — how a page composes a replay. The distributed component
// file carries the seek module (replay_seek.js, ahead of this block) and the vendored
// rrweb-player and its CSS (below it), so one <script src> is the whole dependency and
// the page works from file:// with nothing to serve.
//
// A payload script (materialized beside this file by `locus browse open`) registers its
// slice set on window.LOCUS_REPLAYS in load order; the page loads this component, then
// its payloads, then mounts one player per set it wants to show — a minimal page,
// composed beside the material so the relative srcs resolve:
//
//   <script src="./locus-replay.js"></script>
//   <script src="./PAYLOAD.js"></script>
//   <div id="p" style="height: 100vh"></div>
//   <script>LocusReplay.mount(document.getElementById('p'), LOCUS_REPLAYS[0])</script>
//
// Any composition works from there — side by side, a summary page with an embedded moment —
// under the page author's own name; the default page `locus browse open` composes is
// refreshed on every open. A payload plays one visitor's time-disjoint slice timeline,
// snapshot to snapshot, on one player; concurrent slices (two tabs, two visitors)
// are separate payloads on separate mounts. Every mount has a name — the composer's
// (opts.name; an analysis page passes its slice labels) or m1, m2, … in mount order.
//
// Every mount plays on the recording's own absolute epoch-ms clock — the clock citations
// use. A moment shows every event stamped at or before it, and a mount shows a moment
// only when its recording holds one: a moment outside the set, in a gap between two of
// its slices, or in a slice's lead — after the slice opened but before its page was
// first captured — is refused with an error naming the mount and what it can show.
// Seeking several mounts is one request: every mount is checked first, and one refusal
// refuses the whole seek before any mount moves. A seek shows the moment and nothing of
// the route to it: a player that has been played or seeked before is replaced by a fresh
// one opened at the moment, so the mount's player is whichever is current. A player is
// built, its snapshot page's fonts and images are waited for, and only then is it moved to
// the moment, so the recorded scrolls land on a laid-out page; the animations and transitions
// the move sets off are finished, so the moment shows their end states — a seek is checked
// at once and refused at once, and moves once the page is ready.
//
// The page's URL fragment addresses the clock: #t=<epoch-ms> opens paused at that
// moment, #t=<start>-<end> additionally marks the range on the timeline and pauses
// playback at its end, and &m=<name> (repeatable) narrows the seek to those mounts —
// without it every mount seeks. A fragment carrying either key is a seek request and is
// refused whole when it is not exactly that grammar; a fragment carrying neither is the
// page's own. The opening fragment is applied once the document has finished parsing,
// to every mount up by then, so a refusal is one error over a whole page; a hashchange
// re-seeks in place, no reload. The drive surface for a scripting browser is
// window.locus = { player, seek, moved }: the first mount's current player;
// seek(startMs, endMs, names), which drives every mount or the named ones, throws a
// refusal synchronously and returns a promise that settles once every mount has moved;
// and moved, that promise for the latest seek, the opening fragment's included.
// mount() sizes the player to its target's box (falling back to the viewport) and returns
// { name, player, seek, replay } for per-mount control.
(() => {
	const CONTROLLER_H = 80;
	const style = document.createElement("style");
	style.textContent =
		".rr-progress { position: relative; } " +
		".locus-range { position: absolute; top: 0; bottom: 0; " +
		"background: rgba(255,196,0,.35); border-left: 2px solid #ffc400; " +
		"border-right: 2px solid #ffc400; pointer-events: none; }";
	document.head.appendChild(style);

	const mounts = [];
	// The latest seek's movement — resolved once every mount it addressed has moved.
	let moved = Promise.resolve([]);

	const FRAGMENT_GRAMMAR =
		"#t=<epoch-ms> or #t=<start>-<end>, then &m=<mount> any number of times";

	function parseFragment() {
		const hash = location.hash.replace(/^#/, "");
		if (!hash) return null;
		const pairs = hash.split("&").map((p) => {
			const eq = p.indexOf("=");
			return eq < 0 ? [p, null] : [p.slice(0, eq), p.slice(eq + 1)];
		});
		if (!pairs.some(([k]) => k === "t" || k === "m")) return null;
		const malformed = (why) =>
			new Error(
				`LocusReplay: fragment #${hash} ${why} — expected ${FRAGMENT_GRAMMAR}`,
			);
		const ts = pairs.filter(([k]) => k === "t");
		if (ts.length !== 1)
			throw malformed(ts.length ? "names t more than once" : "names no t");
		const t = ts[0][1] ?? "";
		const m = t.match(/^(\d+)(?:-(\d+))?$/);
		if (!m) throw malformed(`has t=${t}, not an epoch-ms moment or range`);
		const names = [];
		for (const [k, v] of pairs) {
			if (k === "t") continue;
			if (k !== "m") throw malformed(`names ${k}, which is not a key`);
			if (!v) throw malformed("has an empty m");
			names.push(decodeURIComponent(v));
		}
		return [+m[1], m[2] ? +m[2] : null, names.length ? names : null];
	}

	function chosen(names) {
		if (names == null) return mounts;
		if (
			!Array.isArray(names) ||
			!names.length ||
			!names.every((n) => typeof n === "string" && n)
		) {
			throw new Error(
				`LocusReplay: mount names are an array of one or more names, got ${JSON.stringify(names)}`,
			);
		}
		const known = mounts.map((h) => h.name);
		const unknown = names.filter((n) => !known.includes(n));
		if (unknown.length) {
			throw new Error(
				`LocusReplay: no mount named ${unknown.join(", ")} — mounted: ${known.join(", ")}`,
			);
		}
		return mounts.filter((h) => names.includes(h.name));
	}

	function seekAll(startMs, endMs, names) {
		const targets = chosen(names);
		const refusals = targets
			.map((h) => h.refusal(startMs, endMs))
			.filter(Boolean);
		if (refusals.length) throw new Error(`LocusReplay: ${refusals.join("; ")}`);
		moved = Promise.all(targets.map((h) => h.seek(startMs, endMs)));
		return moved;
	}

	function mount(target, replay, opts = {}) {
		const events = replay.events;
		const startTs = events[0].timestamp;
		const endTs = events[events.length - 1].timestamp;
		const name = opts.name ?? `m${mounts.length + 1}`;
		if (mounts.some((h) => h.name === name)) {
			throw new Error(`LocusReplay: a mount named ${name} already exists`);
		}
		const recW = replay.viewport.width,
			recH = replay.viewport.height;
		const box = target.getBoundingClientRect();
		const availW = opts.width ?? (box.width || innerWidth);
		const availH =
			(opts.height ?? (box.height || innerHeight - box.top)) - CONTROLLER_H;
		const scale = Math.min(availW / recW, availH / recH);
		const Player = rrwebPlayer.Player || rrwebPlayer.default || rrwebPlayer;
		const band = document.createElement("div");
		band.className = "locus-range";
		let rangeEndOffset = null;
		let player = null;
		// A player is pristine until anything moves it — a seek, the controller, playback.
		// Only a pristine player shows a moment as a fresh one would; any other is replaced.
		let pristine = false;
		// The player's snapshot page built and its fonts and images settled.
		let ready = null;

		// rrweb writes into the events it is given — a mutation it applies twice keeps only
		// the removes still in its mirror, on the event object itself — so every player
		// replays its own copy and the payload stays what the recording said.
		function build() {
			const p = new Player({
				target,
				props: {
					events: structuredClone(events),
					width: Math.floor(recW * scale),
					height: Math.floor(recH * scale),
					autoPlay: false,
					skipInactive: true,
					showController: true,
					pauseAnimation: false,
					insertStyleRules: LocusSeek.STYLE_RULES,
				},
			});
			// The replayer rebuilds the iframe document with document.open() and never issues the
			// final close(), so the tab spinner runs over a fully rendered replay. Leave it: a
			// close() issued after the rebuild strands stylesheet rules Chrome had already applied
			// — a 200 KB <style> kept its early rules and lost its late ones, and only a re-parse
			// of the element brought them back — so the spinner is the honest state.
			p.addEventListener("ui-update-current-time", (e) => {
				pristine = false;
				if (rangeEndOffset != null && e.payload >= rangeEndOffset) p.pause();
			});
			p.addEventListener("ui-update-player-state", () => {
				pristine = false;
			});
			pristine = true;
			const replayer = p.getReplayer();
			ready = new Promise((built) =>
				replayer.on("fullsnapshot-rebuilded", built),
			).then(() => LocusSeek.settled(replayer.iframe.contentDocument));
			return p;
		}
		player = build();
		const totalMs = player.getMetaData().totalTime;

		// Why this mount cannot honor the seek — the moment, or the range's end — or null.
		function refusal(startMs, endMs) {
			if (!Number.isInteger(startMs))
				return `${name}: t=${startMs} is not an epoch-ms moment`;
			const gone = LocusSeek.unshowable(replay.slices, startMs);
			if (gone) return `${name} cannot show t=${startMs} — ${gone}`;
			if (endMs != null) {
				if (!Number.isInteger(endMs))
					return `${name}: the range end t=${endMs} is not an epoch-ms moment`;
				if (endMs < startMs)
					return `${name}: the range ends at t=${endMs}, before it starts at t=${startMs}`;
				if (endMs > endTs)
					return `${name}: the range ends at t=${endMs}, past its recording's end at t=${endTs}`;
			}
			return null;
		}

		function seek(startMs, endMs) {
			const why = refusal(startMs, endMs);
			if (why) throw new Error(`LocusReplay: ${why}`);
			if (!pristine) {
				player.getReplayer().destroy();
				player.$destroy();
				player = build();
			}
			const own = player;
			return ready.then(() => {
				if (player !== own) return;
				const s = startMs - startTs;
				rangeEndOffset = endMs == null ? null : endMs - startTs;
				const bar = target.querySelector(".rr-progress");
				if (bar && rangeEndOffset != null && rangeEndOffset > s) {
					band.style.left = (s / totalMs) * 100 + "%";
					band.style.width = ((rangeEndOffset - s) / totalMs) * 100 + "%";
					bar.appendChild(band);
				} else {
					band.remove();
				}
				const offset = LocusSeek.offsetOf(startMs, startTs);
				player.goto(offset, false);
				const doc = player.getReplayer().iframe.contentDocument;
				return LocusSeek.settled(doc).then(() => {
					if (player !== own) return;
					LocusSeek.finishAnimations(doc);
				});
			});
		}

		const handle = {
			name,
			get player() {
				return player;
			},
			seek,
			refusal,
			replay,
		};
		mounts.push(handle);
		if (mounts.length === 1) {
			window.locus = {
				get player() {
					return handle.player;
				},
				seek: seekAll,
				get moved() {
					return moved;
				},
			};
		}
		return handle;
	}

	function applyFragment() {
		const fragment = parseFragment();
		if (fragment) seekAll(...fragment);
	}
	if (document.readyState === "loading") {
		addEventListener("DOMContentLoaded", applyFragment);
	} else {
		queueMicrotask(applyFragment);
	}
	addEventListener("hashchange", applyFragment);

	window.LocusReplay = {
		mount,
		seek: seekAll,
		mounts,
		get moved() {
			return moved;
		},
	};
})();
