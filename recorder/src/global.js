/**
 * The script-tag entry, and the deployment's one configuration point.
 *
 * The bundle reads its own snippet id and store origin off its own script tag —
 * document.currentScript, captured while the classic script evaluates. The spec pins currentScript
 * to the executing script for the whole synchronous evaluation, however the tag was injected:
 * parser, createElement, GTM
 * (https://html.spec.whatwg.org/multipage/dom.html#dom-document-currentscript). A null or src-less
 * capture means the bundle was not loaded as its own classic script tag — module-wrapped, inlined,
 * eval'd — so the id and origin exist nowhere on the page, and it reports that and stops rather
 * than guess at other scripts.
 *
 * It wires two channels back to that origin through storeSink (sink.js) — the chunk sink and a
 * fire-and-forget telemetry ping — operational facts, never page content, on a channel that
 * survives a recorder whose uploads are failing. Drop the `telemetry` option below and no ping is
 * ever sent. The key layout lives in storeSink rather than pasted into the page: a tag cannot be
 * changed once it is on the operator's site, so the id is all it carries.
 *
 * Capture config — masking (`rrwebRules`), drain cadence, a fractional-rollout gate — is written
 * inline here, config as code. Options go on the start() call below; a rollout gate is a few lines
 * before it (bucket getOrCreateVisitorId() against your fraction and return early when out — the
 * marker still claims the tag, so a sampled-out page context is loaded-but-not-recording, same as a
 * bot-gated one). The store's deploy rebuilds this facade from source and vends it, so an edit
 * ships on `bun run deploy`.
 * start()'s jsdoc (index.js) is the option surface, and masking.js declares what a masking pattern
 * matches. Nothing here ships configured: the untouched facade records verbatim except credentials (masking.js declares the always-masked set).
 *
 * Once the facade runs it sets `window.LocusRecorder`. Its presence means the bundle loaded and
 * ran, whether or not this page context is recorded; it carries `flush()` only while recording. flush
 * drains the buffer to empty at once — batch after batch until nothing deliverable remains, a
 * delivery failure ending it early into the error accounting rather than thrown into the page.
 */
import {
	faultPing,
	getVisitorIdentity,
	isValidSnippetId,
	SNIPPET_ID_PATTERN,
	start,
	storeSink,
} from "./index.js";

export async function autostart(tag) {
	// A second tag on the page changes nothing: the first claimed it (marker set synchronously
	// below, before any await), and one page is one recording — the dupe stays inert, saying so
	// once, because a duplicate install (a tag manager plus a hardcoded tag) is otherwise
	// invisible.
	if (window.LocusRecorder) {
		console.warn(
			"locus-recorder: window.LocusRecorder is already set — an earlier tag claimed this " +
				"page and one page is one recording, so this tag stays inert.",
		);
		return;
	}
	if (!tag?.src) {
		console.error(
			"locus-recorder: document.currentScript carried no src — the bundle was not loaded as its " +
				"own classic <script src> tag (it was module-wrapped, inlined, or eval'd), so the snippet " +
				"id and store origin it must read off that tag exist nowhere. Not starting.",
		);
		return;
	}
	window.LocusRecorder = {};
	const url = new URL(tag.src);
	const id = url.searchParams.get("id");
	if (id === null) {
		console.error(
			"locus-recorder: the script tag's src carries no ?id=<snippet-id> — the id is the first " +
				"segment of every store key, so without it nothing can be captured. Paste the tag " +
				"with its id intact. Not starting.",
		);
		return;
	}
	if (!isValidSnippetId(id)) {
		console.error(
			`locus-recorder: snippet id ${JSON.stringify(id)} does not match ${SNIPPET_ID_PATTERN} — ` +
				"the store rejects every upload under it, capturing nothing. Not starting.",
		);
		return;
	}
	// storeSink closes the store's URL contract over this tag's origin and id, and shares one
	// carrier between the telemetry pings and the chunk sink's hidden-marker shots — the
	// fire-and-forget transport verdict is a fact about the environment, not about either channel.
	const { sink, telemetry } = storeSink({ origin: url.origin, snippetId: id });
	let visitorId = null;
	try {
		// Minted here rather than inside start(), so that a start() which dies on this very call —
		// reading document.cookie throws on an opaque origin (visitor.js) — still has a channel to
		// say so.
		const identity = getVisitorIdentity();
		visitorId = identity.id;
		const instance = await start({
			visitorId,
			visitorSource: identity.source,
			telemetry,
			sink,
		});
		if (instance) window.LocusRecorder = { flush: instance.flush };
	} catch (error) {
		// A page context that loaded the bundle, passed every gate, and then died on the way up is
		// invisible on both planes — no birth, no chunk, no error — and indistinguishable from a
		// visitor who never came. The telemetry sink exists independently of start(), so the fault
		// still goes out, its shape faultPing's (sink.js) — where the identity itself is what failed,
		// visitorId is null there and the ping is keyed to the sentinel: this visitor could not be
		// named.
		console.error("locus-recorder: failed to start —", error);
		telemetry.emit(faultPing("start_failed", error, { visitorId }));
	}
}

// Importing this module IS the install — so it must be inert wherever there is no page to record
// (a bundler's SSR pass, a test importing autostart to drive it directly). currentScript is read at
// the module's own synchronous evaluation; it is null once that ends.
if (typeof document !== "undefined") autostart(document.currentScript);
