/**
 * The sink contract: the recorder hands a self-describing chunk to a sink that owns the key layout and the transport. Nothing here is hardcoded to any delivery; the deployment injects the sink.
 *
 *   sink.send(bytes: Uint8Array, descriptor) -> Promise<void>   reject = retry
 *   sink.beacon?(bytes, descriptor) -> boolean                 fire-and-forget; false = refused oversize
 *
 * descriptor: {visitorId, sliceId, chunkKey, count, hasFullSnapshot}. chunkKey is the first event's counter — timestamp+sequence, so it orders the slice's chunks and dedups them for free. Nothing the chunk's own events already carry is copied into it.
 *
 * hasFullSnapshot flags the chunks a store's size cap can ever reject: a FullSnapshot is the one indivisible record that can exceed it.
 *
 * **A rejection must say which of the two things failed, and only the sink can tell.** The uploader retries a rejected batch ahead of all new work and gives up on it only when the batch itself is at fault (uploader.js), so a send that fails rejects with storeRejection() when — and only when — the store answered and refused these bytes: an answer says the store saw the payload and said no, which no retry of the same bytes changes. Every other failure has no answer behind it — a dropped connection, a DNS failure, an offline device, a send the abort deadline cut off — and indicts the environment, saying nothing about the batch; it rejects plainly and is retried at the normal cadence for as long as the outage lasts, so a bounce or an offline stretch costs latency, never a recording. A sink that marks nothing is read as an environment failing throughout, the reading that never drops a batch.
 *
 * httpSink reads that off the response, and first checks who is speaking: the store stamps every response it mints with the Locus-Store header (store/src/worker.js declares and exposes it), and a status code alone never identifies the speaker — a filtering proxy's 403 and a captive portal's 404 wear the same codes the store's refusals do. A stamped 4xx names this upload (too large, malformed, a path the store does not serve), so it is a store rejection; an unstamped one is a middlebox answering in the store's place and retries like a silent network. 408 and 429 name the moment rather than the payload, and every 5xx names the store's own failure to process it, so both retry the same way.
 *
 * httpSink is the reference implementation for any HTTP store — the deployment supplies the descriptor→URL mapping, so the key layout belongs to the store, never to the recorder. storeSink is this deployment's mapping: given the store origin and snippet id it closes over the path shapes store/src/keys.js gates on ingest (chunk and telemetry), so the emit-side contract has one home.
 *
 * Every request is a CORS *simple* request — a POST with no author-set headers — so none can preflight; sendBeacon cannot preflight at all, and a beacon that would need one is dropped while still returning true (https://fetch.spec.whatwg.org/#cors-preflight-fetch, safelist at https://fetch.spec.whatwg.org/#cors-safelisted-request-header). A gzip body has no safelisted Content-Type to declare it, so the store reads its magic bytes. The single-flight drain reads its response, which is what the store's Access-Control-Allow-Origin is for.
 *
 * Every awaited send carries a 30s abort, so a hung request cannot wedge the single-flight uploader. Fire-and-forget shots ride the transport carrier below; both of its carriers set the keepalive flag, which the Fetch standard size-limits (https://fetch.spec.whatwg.org/#http-network-or-cache-fetch), so MAX_BEACON_BYTES is checked before any shot.
 */

import { pageAddress } from "./address.js";
import { describeError } from "./errors.js";
import { RECORDER_VERSION } from "./version.js";
import { UNIDENTIFIED_VISITOR } from "./visitor.js";

export const MAX_BEACON_BYTES = 64 * 1024;

// How long a transport assessment may keep watching for its verdict's evidence. The witness fetch
// pings it waits on ride the cost cadence, so the window spans several of those beats.
const ASSESS_WINDOW_MS = 5 * 60_000;

// The mark rides the error's name rather than an instance identity: a deployment's sink is often
// bundled separately from the uploader that reads its rejections, and a string survives that where
// an instanceof check would silently read every rejection as unmarked.
const STORE_REJECTED = "StoreRejectedError";

/** Mint the rejection for a send the store answered and refused — the one failure that indicts the batch rather than the environment. */
export function storeRejection(message) {
	const error = new Error(message);
	error.name = STORE_REJECTED;
	return error;
}

/** Whether a failed send was refused by the store itself, as opposed to never reaching it. */
export function isStoreRejection(error) {
	return error?.name === STORE_REJECTED;
}

// The store's mark on every response it mints — one half of a two-sided fact whose writer is
// store/src/worker.js (the CORS block declares the header and exposes it cross-origin).
const STORE_STAMP = "locus-store";

function refusesPayload(response) {
	const { status } = response;
	if (status < 400 || status >= 500 || status === 408 || status === 429)
		return false;
	return response.headers?.get(STORE_STAMP) != null;
}

/**
 * The fire-and-forget transport, assessed once per environment — one carrier object per page context, shared by every sink that fires shots without reading results (telemetry pings, the hidden-marker chunk shots), so one verdict governs them all.
 *
 *   carrier.shoot(target, body)  -> void   fire on the standing channel, every failure swallowed
 *   carrier.assess(target, body) -> void   fire the channel-assessment shot (at most once)
 *
 * Two transports can carry a shot, and neither is safe everywhere. navigator.sendBeacon is the one whose post-document dispatch the user agent owns by contract — WebKit kills queued keepalive fetches at tab teardown while dispatching queued beacons — but content blockers type it as a trackable "ping" and can eat it while it still returns true, and hardened browsers delete the API outright. A keepalive fetch is typed as ordinary fetch, so ping-blocking rules never match it, and Chromium and Firefox deliver it through teardown — but that survival is empirical rather than contractual, and WebKit reports the option supported while breaking it. So the carrier assesses the environment it woke in and picks one standing channel for everything.
 *
 * assess() fires a dedicated throwaway shot over sendBeacon — to the same URL real pings ride, so no path rule can split probe from payload — and watches its fate through Resource Timing, which the store's Timing-Allow-Origin header opens: a delivered shot's entry carries a nonzero transferSize, a blocked one's carries zero. A nonzero probe entry verdicts the beacon outright. A zero one is ambiguous — a block, or an engine that never fills the field — and resolves only on the witness the fetch pings provide on the same URL: a nonzero fetch entry proves the field readable, so the probe's zero was a real block → keepalive fetch, which delivers through teardown on exactly the engines whose blockers eat beacons; a zero entry for a fetch that provably delivered (its promise resolved) proves the field dead → the beacon — WebKit's shape, the engine with no ping-blocker ecosystem and the one that needs the beacon. Resource Timing absent → the beacon, the default where nothing can be known. sendBeacon absent → keepalive fetch, and no probe fires: a probe row in the store means exactly one thing, a beacon from this context arrived.
 *
 * Until a verdict lands, shots ride keepalive fetch — the one channel that delivers everywhere blind: its worst case during the wait is a WebKit tab-close race, where an unproven beacon's is an environment silently eating every shot. The verdict is always evidence, never a default adopted while waiting, so the first pings go out the door immediately and the verdict only decides the channel from then on. On a beacon verdict, a shot the API refuses (returns false or throws) falls back to a keepalive fetch, that shot only. A deployment that never assesses (telemetry dropped, so no ping URL exists to probe) rides the blind-safe channel for good.
 */
export function transportCarrier({ beaconFn, fetchFn } = {}) {
	const doBeacon =
		beaconFn !== undefined
			? beaconFn
			: typeof navigator !== "undefined" && navigator.sendBeacon
				? (target, body) => navigator.sendBeacon(target, body)
				: null;
	const doFetch = fetchFn ?? ((...args) => fetch(...args));
	let channel = "fetch";
	let assessed = false;
	// The open assessment: the probed URL plus the evidence gathered so far. A zero probe entry is
	// ambiguous — a block, or an engine that never fills transferSize — and only an entry on the
	// probed URL itself can disambiguate: WebKit fills the field for same-origin entries while
	// reading zero for every cross-origin one, TAO or not, so a nonzero elsewhere on the page
	// proves nothing. The fetch pings ride the same URL and are the witness in both directions —
	// a nonzero fetch entry proves the field (the zero was a block), a zero entry for a fetch
	// whose promise resolved proves it dead (delivery without a readable size). Until one form of
	// evidence lands, the channel stays the blind-safe fetch; no verdict is ever a default.
	let open = null;

	const settle = () => {
		if (!open?.probeZero) return;
		if (open.fieldProven) channel = "fetch";
		else if (open.fetchEntryZero && open.fetchDelivered) channel = "beacon";
		else return;
		open.observer.disconnect();
		clearTimeout(open.deadline);
		open = null;
	};

	const overFetch = (target, body) => {
		// keepalive lets the shot outlive its document on the engines that honor it; an engine that
		// does not know the option ignores it, leaving a plain fetch.
		Promise.resolve(
			doFetch(target, { method: "POST", body, keepalive: true }),
		).then(
			() => {
				if (open && open.target === target) {
					open.fetchDelivered = true;
					settle();
				}
			},
			() => {},
		);
	};

	const watch = (target) => {
		const state = {
			target,
			probeZero: false,
			fieldProven: false,
			fetchEntryZero: false,
			fetchDelivered: false,
		};
		state.observer = new PerformanceObserver((list) => {
			if (open !== state) return;
			for (const entry of list.getEntries()) {
				if (entry.name !== target) continue;
				if (entry.initiatorType === "fetch") {
					if (entry.transferSize > 0) state.fieldProven = true;
					else state.fetchEntryZero = true;
				} else if (entry.transferSize > 0) {
					channel = "beacon";
					state.observer.disconnect();
					clearTimeout(state.deadline);
					open = null;
					return;
				} else {
					state.probeZero = true;
				}
			}
			settle();
		});
		// The assessment is bounded: evidence that has not arrived within the window never will
		// matter enough to keep an observer running on the host page for its whole life — every
		// resource entry would keep paying the callback. Past the deadline the verdict stays
		// unproven and shots keep riding the blind-safe channel.
		state.deadline = setTimeout(() => {
			if (open !== state) return;
			state.observer.disconnect();
			open = null;
		}, ASSESS_WINDOW_MS);
		open = state;
		state.observer.observe({ type: "resource", buffered: true });
	};

	return {
		shoot(target, body) {
			if (channel === "beacon" && doBeacon) {
				try {
					if (doBeacon(target, body)) return;
				} catch {
					/* fall through to the fetch shot */
				}
			}
			overFetch(target, body);
		},
		assess(target, body) {
			if (assessed) return;
			assessed = true;
			if (!doBeacon) return;
			try {
				watch(target);
			} catch {
				// Nothing can watch the shot's fate; the beacon is the default where nothing is known.
				open = null;
				channel = "beacon";
			}
			try {
				doBeacon(target, body);
			} catch {
				/* the assessment is best-effort */
			}
		},
	};
}

export function httpSink({ url, timeoutMs = 30_000, fetchFn, carrier }) {
	const doFetch = fetchFn ?? ((...args) => fetch(...args));
	const ride = carrier ?? transportCarrier();

	return {
		async send(bytes, descriptor) {
			const controller = new AbortController();
			let timeout = setTimeout(() => controller.abort(), timeoutMs);
			// A frozen page suspends the request but not the wall clock: the abort deadline expires
			// during the freeze, and on unfreeze the stale timer fires before the request gets a
			// moment of runtime, killing an upload at exactly the moment delivery works again. The
			// Page Lifecycle resume event re-arms a fresh window instead.
			const onResume = () => {
				clearTimeout(timeout);
				timeout = setTimeout(() => controller.abort(), timeoutMs);
			};
			const doc = typeof document === "undefined" ? null : document;
			doc?.addEventListener("resume", onResume);
			try {
				const response = await doFetch(url(descriptor), {
					method: "POST",
					body: bytes,
					signal: controller.signal,
				});
				if (!response.ok) {
					const message = `sink upload failed: ${response.status}`;
					throw refusesPayload(response)
						? storeRejection(message)
						: new Error(message);
				}
			} finally {
				clearTimeout(timeout);
				doc?.removeEventListener("resume", onResume);
			}
		},

		beacon(bytes, descriptor) {
			if (bytes.length > MAX_BEACON_BYTES) return false;
			try {
				ride.shoot(url(descriptor), bytes);
			} catch {
				return false;
			}
			return true;
		},
	};
}

/**
 * Fire-and-forget telemetry sink — a channel independent of the chunk upload, so a slice-birth ping or cost sample is recorded even when uploads are failing or the recorder dies before its data ships. That independence is what lets the store tell a slice that was born but never arrived from one that was never started.
 *
 *   telemetry.emit(ping)  -> void   deliver on the carrier's standing channel
 *   telemetry.probe(ping) -> void   hand the carrier its channel-assessment shot
 *
 * The transport is the carrier's (transportCarrier above): probe() feeds it the one sacrificial ping its assessment fires, and every emit rides its standing channel. There is no result to read, and every failure is swallowed — telemetry never blocks capture or counts toward the error threshold. The JSON rides as a plain string, so the shot stays a CORS simple request on either transport (see the file header) and the store parses the body without consulting its type. The deployment supplies the ping→URL mapping, as it does for chunks.
 *
 * emit and probe stamp `ts` — the queue moment, epoch ms on the device's clock — onto every ping. A ping can dispatch well after it is queued (a hidden tab, a dying page), so the store's receive time is not the ping's time; the stamp is what lets a reader window pings on the same client-minted clock the recording itself is stamped with.
 *
 * Where a ping names a page it names it by address (address.js): this channel is unmasked by construction, so its door strips `url` and `referrer` before anything is on the wire, whatever composed the payload. The composers below already write addresses, so the door's strip is idempotent over them; a replacement telemetry channel owns the same duty for its own wire.
 */
export function httpTelemetry({ url, carrier }) {
	const ride = carrier ?? transportCarrier();
	const stamped = (ping) => {
		const addressed = { ts: Date.now(), ...ping };
		if ("url" in addressed) addressed.url = pageAddress(addressed.url);
		if ("referrer" in addressed)
			addressed.referrer = pageAddress(addressed.referrer);
		return JSON.stringify(addressed);
	};
	return {
		emit(ping) {
			try {
				ride.shoot(url(ping), stamped(ping));
			} catch {
				/* telemetry is best-effort */
			}
		},
		probe(ping) {
			try {
				ride.assess(url(ping), stamped(ping));
			} catch {
				/* telemetry is best-effort */
			}
		},
	};
}

/**
 * The two ping shapes with more than one author. Every other metric is composed at exactly one site inside start(), but a fault or a gate verdict is emitted by facades too — the deployment code start() cannot see — and a hand-written payload re-derives the worker's wire contract and drifts. So the shape's invariants live here, once: the metric name the worker's gate accepts, the error line and its truncation, the sentinel identity where the visitor could not be named, the page named by address. A facade states only the judgment — which reason, when, and any identity it holds.
 *
 * recorderVersion defaults to this build's provenance and is overridden where the deployment overrides the stamped provenance (start()'s recorderVersion option).
 */
export function faultPing(
	reason,
	error,
	{ visitorId, sliceId = null, recorderVersion = RECORDER_VERSION } = {},
) {
	return {
		metric: "recorder_fault",
		visitorId: visitorId ?? UNIDENTIFIED_VISITOR,
		recorderVersion,
		sliceId,
		reason,
		error: describeError(error).slice(0, 256),
		// Optional-chained: a fault composer is often the page's last resort and must not itself
		// throw, even where what broke is the window being less than a browser's.
		url: pageAddress(window.location?.href ?? ""),
	};
}

/** A page context a policy gate declined, attesting itself: the recorded corpus is definitionally only what passed the gates, so without this ping the exclusion has no witness on any plane. detail is the gate's own evidence for the verdict — for the bot gate, the firing BotD detector names (detectBot in index.js) — because a bare verdict over a page context that recorded nothing can never be audited; empty where the gate has none. */
export function gatedPing(
	reason,
	{ visitorId, recorderVersion = RECORDER_VERSION, detail = "" } = {},
) {
	return {
		metric: "capture_gated",
		visitorId: visitorId ?? UNIDENTIFIED_VISITOR,
		recorderVersion,
		reason,
		detail,
		url: pageAddress(window.location?.href ?? ""),
	};
}

/**
 * The Locus store's emit-side URL contract, closed over origin and snippet id. One home for the path shapes that store/src/keys.js gates on ingest:
 *
 *   {origin}/chunks/{snippetId}/{visitorId}/{sliceId}/{chunkKey}
 *   {origin}/telemetry/{snippetId}/{visitorId}
 *
 * Returns the chunk sink and the telemetry channel wired to that contract and sharing one carrier, so the fire-and-forget transport verdict is one fact about the environment. An injected carrier is honored; otherwise one is minted for the pair. timeoutMs and fetchFn pass through to the chunk sink only (telemetry has no result to wait on).
 */
export function storeSink({ origin, snippetId, carrier, timeoutMs, fetchFn }) {
	const ride = carrier ?? transportCarrier();
	return {
		sink: httpSink({
			url: (d) =>
				`${origin}/chunks/${snippetId}/${d.visitorId}/${d.sliceId}/${d.chunkKey}`,
			carrier: ride,
			timeoutMs,
			fetchFn,
		}),
		telemetry: httpTelemetry({
			url: (ping) => `${origin}/telemetry/${snippetId}/${ping.visitorId}`,
			carrier: ride,
		}),
	};
}
