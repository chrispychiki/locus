/**
 * The open Locus recorder: rrweb capture → stamped events → IndexedDB buffer → periodic drain to a deployment-injected sink, as self-describing gzipped chunks keyed by visitor/slice/counter.
 *
 *   import { start, httpSink } from "./index.js";
 *   const recorder = await start({ sink: httpSink({ url: d => ... }) });
 *
 * Two rrweb options are set explicitly (rrweb packages/rrweb/src/record/index.ts): recordDOM, without which takeFullSnapshot returns early and no snapshot is ever captured, and checkoutEveryNms, the periodic Meta+FullSnapshot cadence that keeps a stream self-healing (it drives takeFullSnapshot(true)).
 *
 * Privacy posture: credential content — passwords, one-time codes, payment card fields — is always masked, decided per element below the rules layer (masking.js); no rule can unmask it. Everything else records verbatim, and rrweb's full masking API is open to the deployment through rrwebRules (record options: rrweb guide.md).
 *
 * Synthetic events, because no rrweb event holds a page's arrival context — its union has no visibility type, and its Load/DomContentLoaded events carry no data at all, no url/title/referrer (rrweb packages/types/src/index.ts):
 * PageLoad marks an arrival, carrying url/title/referrer. It fires on the page's first FullSnapshot — inside that snapshot's slice, right behind the covering snapshot — and again on every SPA route change, which rrweb itself leaves invisible. A route whose masking differs restarts capture, so its PageLoad rides the restart's first snapshot; any other route emits it in place. spaRoute is set either way.
 * PageVisible/PageHidden fire on visibility transitions and carry url/referrer. document.referrer is fixed when the Document is created and constant for its lifetime (https://html.spec.whatwg.org/multipage/dom.html#dom-document-referrer), so a page that bounces before rrweb has a DOM to capture still testifies its arrival facts through the hidden marker, which the context's head slice carries. hidden is the last moment a page can reliably observe — unload and pagehide do not fire in many mobile teardowns (https://developer.chrome.com/docs/web-platform/page-lifecycle-api) — so a fire-and-forget shot on the transport carrier (sink.js) carries the hidden marker and nothing else; IndexedDB is the durability, and the next page context drains it.
 */

import { load as loadBotd } from "@fingerprintjs/botd";
import * as rrweb from "@rrweb/record";

import { pageAddress } from "./address.js";
import { openBuffer } from "./buffer.js";
import { buildChunks, collectEnvelope } from "./chunk.js";
import { COST_PING_INTERVAL_MS, costMeter, sampleBacklog } from "./cost.js";
import { isLocalEnvironment } from "./environment.js";
import { describeError } from "./errors.js";
import { maskingPosture, optionsForUrl } from "./masking.js";
import { watchRoutes } from "./route.js";
import { EventType } from "./rrweb_constants.js";
import { faultPing, gatedPing } from "./sink.js";
import { EventStream, makeSliceId } from "./stream.js";
import { describeRecords, Uploader } from "./uploader.js";
import { RECORDER_VERSION } from "./version.js";
import { getVisitorIdentity } from "./visitor.js";

export { isLocalEnvironment } from "./environment.js";
export { describeError } from "./errors.js";
export { ROUTE_ALL_INPUTS } from "./masking.js";
export {
	faultPing,
	gatedPing,
	httpSink,
	httpTelemetry,
	MAX_BEACON_BYTES,
	storeRejection,
	storeSink,
	transportCarrier,
} from "./sink.js";
export { isValidSnippetId, SNIPPET_ID_PATTERN } from "./snippet.js";
export { RECORDER_VERSION } from "./version.js";
export {
	getOrCreateVisitorId,
	getVisitorIdentity,
	isValidVisitorId,
	UNIDENTIFIED_VISITOR,
} from "./visitor.js";

// The error-string queue is cleared only when a chunk carrying it lands (uploader.js), so a
// context that can never deliver would grow it without bound; it is capped at the newest
// MAX_ERROR_RECORDS, oldest dropped. The queue is best-effort testimony by design — the fault
// plane (recorder_fault) announces the same failures independently, and neither witness is a
// census — so a trim drops old records silently rather than landing a count-less marker among
// them.
const MAX_ERROR_RECORDS = 100;

// A BotD verdict resting on one of these detectors alone is not a bot: each fires on a real
// browser by that browser's own design, while automation trips other detectors beside it.
//  - detectMimeTypesConsistent reads prototype identity on navigator.mimeTypes. Facebook's Android
//    in-app browser patches navigator and fails it, on real phones, by app release.
//  - detectPluginsLengthInconsistency reads navigator.plugins.length === 0 on a Chromium BotD did
//    not judge Android. Chromium empties the list whenever its PDF viewer is unavailable (the
//    download-PDFs setting, a PDF extension, Vivaldi), and Android Chrome always reports 0, so
//    every Android device BotD's own android heuristic misses lands here too. Headless Chrome's new
//    mode reports the same plugin entries as headed, so alone the check catches only an old-mode
//    headless that spoofed its user agent and webdriver flag but left plugins untouched.
const LONE_DETECTOR_EXEMPT = new Set([
	"detectMimeTypesConsistent",
	"detectPluginsLengthInconsistency",
]);

export async function detectBot() {
	const botd = await loadBotd({ monitoring: false });
	const { bot } = botd.detect();
	if (!bot) return { isBot: false, detail: "" };
	// The verdict alone can't be interrogated — which check fired is what separates a real
	// automation signature from an environment quirk tripping a heuristic — so the firing
	// detector names ride the verdict, verbatim as BotD keys them.
	const detections = botd.getDetections();
	const firing = Object.keys(detections).filter((name) => detections[name].bot);
	const detail = firing.join(",");
	if (firing.length === 1 && LONE_DETECTOR_EXEMPT.has(firing[0])) {
		return { isBot: false, detail };
	}
	return { isBot: true, detail };
}

/**
 * @param {object} config
 * @param {object} config.sink                    required; see sink.js
 * @param {object} [config.telemetry]             optional fire-and-forget liveness/cost ping sink; see httpTelemetry in sink.js. A replacement channel needs emit; probe is optional — a channel that runs no transport assessment omits it
 * @param {Array<{pattern:string,options:object}>} [config.rrwebRules]  ordered per-URL rrweb options, resolved at each record() and re-resolved on route change; a changed result stops/restarts rrweb (new slice), an unchanged one keeps the in-place PageLoad. Omit for the credential-mask-only default. recordDOM, the event sink, and the credential mask (masking.js) are locked (a rule may add masking, never take credential masking away — a rule's own maskInputFn still runs, beneath the credential check; a rule's maskAllInputs masks every input, `select` included unless its maskInputOptions says `select: false`); the snapshot cadence (checkoutEveryNms below) is a default any rule can override
 * @param {string} [config.visitorId]             override the cookie identity; honored verbatim, never reformatted or rejected, so a deployment supplying its own id owns keeping it inside its store's visitor charset
 * @param {string} [config.visitorSource]         how that supplied id was come by; rides every birth. Absent, births carry "undeclared"
 * @param {boolean} [config.recordLocalEnvironment=false]
 * @param {boolean} [config.botDetection=true]    BotD gate
 * @param {{isBot:boolean, detail:string}} [config.botVerdict]  a verdict the deployment already took from detectBot() ahead of start(); honored as the gate's own, so BotD runs once, and its exempted detector rides the births
 * @param {number} [config.intervalMs]            base drain cadence (defaults in uploader.js)
 * @param {number} [config.maxIntervalMs]         drain cadence cap as the page context ages (defaults in uploader.js)
 * @param {number} [config.rampPeriodMs]          doubling period for the drain cadence (defaults in uploader.js)
 * @param {number} [config.errorThreshold=5]      self-termination bound
 * @param {number} [config.maxGzippedChunkBytes]  per-chunk gzipped cap; match the store's
 * @param {number} [config.maxBacklogBytes]       on-device buffer ceiling (default: MAX_BACKLOG_BYTES in buffer.js), at which the buffer evicts its oldest undelivered records to admit new capture
 * @returns {Promise<{stop, flush} | null>}       null when gated off
 */
export async function start(config) {
	if (!config?.sink) throw new Error("locus-recorder: a sink is required");

	// A prerendered page runs this script before any human sees it (Chromium speculation-rules and
	// omnibox prerendering: https://wicg.github.io/nav-speculation/prerendering.html), so recording
	// from here would mint recordings, births, and uploads for pages nobody viewed. Everything waits
	// for activation: prerenderingchange fires exactly once at activation (the flag is already false
	// inside the handler, by spec order), and a prerender that is never activated is discarded
	// silently, so nothing below ever runs for it. A browser without the flag does not prerender —
	// the flag and the prerendering machinery are defined by the same spec.
	if (document.prerendering) {
		await new Promise((resolve) =>
			document.addEventListener("prerenderingchange", resolve, { once: true }),
		);
	}

	if (
		!config.recordLocalEnvironment &&
		isLocalEnvironment(window.location.hostname)
	) {
		return null;
	}

	// Only the minter knows how an id was come by, so a supplied one without a stated source leaves
	// the question open.
	const identity =
		config.visitorId == null
			? getVisitorIdentity()
			: { id: config.visitorId, source: config.visitorSource ?? "undeclared" };
	const visitorId = identity.id;
	const recorderVersion = RECORDER_VERSION;
	// The capture-moment device facts, collected once: every envelope field is constant for the
	// page context's life, and this one reading serves everything the context stamps or ships.
	const envelope = collectEnvelope();

	// The bot gate attests itself: BotD's verdict is per page context and knowable no other way, and
	// the recorded corpus is definitionally only what passed, so without this ping "how much of my
	// traffic is bots" has no witness at all. The local-environment gate above stays silent — the
	// operator's own dev machine is not their traffic.
	const verdict =
		config.botVerdict ??
		(config.botDetection !== false ? await detectBot() : null);
	if (verdict?.isBot) {
		config.telemetry?.emit(
			gatedPing("bot", {
				visitorId,
				recorderVersion,
				detail: verdict.detail,
			}),
		);
		return null;
	}
	// A verdict the gate exempted leaves no gate row, so the births carry the detector it rested
	// on — the admitted cohort's only witness on either plane.
	const gateExempt = verdict?.detail ?? "";

	let firstSlice = true;
	let pageLoadSliceId = null;
	const onSliceOpen = (sliceId) => {
		if (firstSlice) pageLoadSliceId = sliceId;
		config.telemetry?.emit({
			metric: "slice_started",
			visitorId,
			recorderVersion,
			sliceId,
			url: pageAddress(window.location.href),
			// snake_case, alone on a camelCase wire: the store's telemetry gate accepts this exact
			// field name (store/src/telemetry.js), and it rejects any ping whose shape it doesn't know.
			first_slice: firstSlice,
			// How this visitor's id was come by. A page load whose id was never persisted
			// records normally and lands under an identity nothing later joins to, so nothing but the
			// birth distinguishes it.
			visitor_source: identity.source,
			gate_exempt: gateExempt,
		});
		firstSlice = false;
	};

	// The page context opens its own slice here, before anything that can await. Everything below
	// takes time a visitor can leave inside — the buffer open, and rrweb's own wait for a DOM worth
	// snapshotting — and until a slice exists there is nothing to stamp an event onto and nothing to
	// name in a ping. rrweb's record-start Meta fills this slice rather than opening one (stream.js).
	const headSliceId = makeSliceId(Date.now());
	// The channel-assessment shot precedes the birth: the shared carrier reads the standing
	// transport for every fire-and-forget shot — pings and the hidden-marker chunk shots alike —
	// off this ping's observed fate (sink.js); the birth follows immediately on the blind-safe
	// channel.
	config.telemetry?.probe?.({
		metric: "transport_probe",
		visitorId,
		recorderVersion,
		sliceId: headSliceId,
	});
	onSliceOpen(headSliceId);

	const errorThreshold = config.errorThreshold ?? 5;
	const errors = [];
	// How many entries the cap trim has removed since the last drain snapshot. The trim eats the
	// queue's front — the same entries a delivery clears — so the delivered-count forget below
	// subtracts what the trim already took, never removing entries that have not shipped.
	let errorsTrimmed = 0;
	let errorCount = 0;
	let terminated = false;
	let stopRecording = null;
	let pageLoadEmitted = false;
	let activeOptionsKey = null;
	let costTimer = null;
	let restoreRouteAwareness = null;
	// The context's cost meter (cost.js): the live cost record the capture and drain paths write
	// into, the foreground-visible clock, and the cost_sample emit.
	const meter = costMeter({
		telemetry: config.telemetry,
		visitorId,
		recorderVersion,
		sliceId: pageLoadSliceId,
	});
	const cost = meter.cost;

	const terminate = () => {
		if (terminated) return;
		terminated = true;
		if (costTimer !== null) clearInterval(costTimer);
		stopRecording?.();
		uploader.stop();
		document.removeEventListener("visibilitychange", onVisibilityChange);
		window.removeEventListener("pagehide", onTeardown);
		window.removeEventListener("pageshow", onRestore);
		document.removeEventListener("freeze", onTeardown);
		document.removeEventListener("resume", onRestore);
		restoreRouteAwareness?.();
		buffer.close();
	};

	// Once per reason per page context: a broken buffer fails on every tick, and the fault names the
	// failure, not its rate. The channel is fire-and-forget, so a failing emission neither throws
	// into the page nor counts. The shape is faultPing's (sink.js): the page address rides the fault
	// itself, because a fault exists to be acted on and acting starts at the page it happened on.
	const faulted = new Set();
	let captureFailed = false;
	const emitFault = (reason, error) => {
		if (faulted.has(reason)) return;
		faulted.add(reason);
		config.telemetry?.emit(
			faultPing(reason, error, {
				visitorId,
				sliceId: pageLoadSliceId,
				recorderVersion,
			}),
		);
	};

	// Two independent questions. `counts` asks whether the error is a recorder malfunction it should
	// stop rather than keep running through — a broken buffer, an internal error, a slicing anomaly
	// it cannot reconcile all count; a failing upload does not, being expected and retryable rather
	// than the recorder failing at its job. `fault` asks whether anyone is told, and on which plane:
	// the error strings ride the next chunk, which reaches nobody on a recorder whose chunks are
	// exactly what is failing, so a fault also goes out on telemetry.
	const onError = (error, { counts = true, fault = null } = {}) => {
		errors.push(describeError(error));
		if (errors.length > MAX_ERROR_RECORDS) {
			const excess = errors.length - MAX_ERROR_RECORDS;
			errors.splice(0, excess);
			errorsTrimmed += excess;
		}
		if (fault) emitFault(fault, error);
		if (!counts) return;
		errorCount += 1;
		if (errorCount >= errorThreshold && !terminated) {
			emitFault("terminated", error);
			terminate();
		}
	};

	// A memory buffer dies with the page, so a visitor on one loses their whole recording unless it
	// ships before they leave. Capture continues either way, so the fault is the only witness. The
	// buffer names which fact it is reporting (buffer.js); this forwards that name rather than
	// deciding one here, because only one of the two is knowable from inside the page.
	const buffer = await openBuffer(undefined, {
		onUnavailable: (error, reason) =>
			onError(error, { counts: false, fault: reason }),
	});
	if (config.maxBacklogBytes !== undefined) {
		buffer.maxBacklogBytes = config.maxBacklogBytes;
	}
	// The capture context every buffered record's slice is stamped with at write, so a chunk
	// assembled by a later, possibly different-identity context still states the capture's
	// visitor and envelope (buffer.js, chunk.js).
	buffer.captureContext = { visitorId, envelope };
	buffer.onEvict = (records, bytes) =>
		onError(
			new Error(
				`evicted ${bytes} buffered bytes at the backlog ceiling ` +
					`(${describeRecords(records)})`,
			),
			{
				counts: false,
				fault: "backlog_evicted",
			},
		);

	const emitCost = () => meter.emit(buffer);
	const emitCostIfActive = () => meter.emitIfActive(buffer);

	// The freeze handler is the last code the page runs before suspension, so the buffer's teardown
	// close happens here or never.
	const onTeardown = () => buffer.close();
	const onRestore = () => buffer.resume();

	const stream = new EventStream(
		headSliceId,
		recorderVersion,
		onSliceOpen,
		onError,
	);

	const record = (event) => {
		if (terminated) return [];
		const t0 = performance.now();
		const stamped = stream.stamp(event);
		// A write that fails takes the event with it, so it counts toward self-termination as
		// host-device harm — and it faults, because a buffer that cannot be written to is otherwise
		// indistinguishable from a page context with nothing to record. A write the buffer itself
		// refused after close() is the exception: teardown is deliberate there, so it is recorded in
		// the errors channel and neither counted nor faulted. That exception reaches only the writes
		// close() got to first; one the browser killed ahead of it counts, which is the single place
		// left where a teardown can read as device harm.
		for (const item of stamped) {
			buffer.append(item).then(
				() => sampleBacklog(cost, buffer),
				(error) =>
					error?.name === "BufferClosedError"
						? onError(error, { counts: false })
						: onError(error, { fault: "buffer_write_failed" }),
			);
		}
		cost.mainThreadMs += performance.now() - t0;
		cost.dirty = true;
		return stamped;
	};

	// The recorded PageLoad carries the arrival verbatim — it is the recording, masked at capture and
	// read back from the operator's own bucket. The ping carries the same arrival as an address
	// (address.js), because the telemetry plane is operational facts and nothing masks it.
	const recordPageLoad = (data, spaRoute) => {
		const stamped = record({
			type: EventType.PageLoad,
			data,
			timestamp: Date.now(),
		});
		if (stamped.length === 0) return;
		config.telemetry?.emit({
			metric: "page_load",
			visitorId,
			recorderVersion,
			sliceId: stamped[stamped.length - 1].sliceId,
			url: pageAddress(data.url),
			referrer: pageAddress(data.referrer),
			spaRoute,
		});
	};

	// The error strings go out as a copy, never drained: the uploader reports back
	// (onErrorsDelivered) once the chunk carrying them lands, and only then are they removed.
	// Snapshotting opens a fresh trim count — the drain is single-flight, so exactly one
	// snapshot is ever in flight against it.
	const context = () => {
		errorsTrimmed = 0;
		return {
			visitorId,
			recorderVersion,
			envelope,
			errors: errors.slice(),
		};
	};

	const uploader = new Uploader({
		buffer,
		sink: config.sink,
		context,
		onError,
		cost,
		onErrorsDelivered: (count) =>
			errors.splice(0, Math.max(0, count - errorsTrimmed)),
		intervalMs: config.intervalMs,
		maxIntervalMs: config.maxIntervalMs,
		rampPeriodMs: config.rampPeriodMs,
		maxGzippedChunkBytes: config.maxGzippedChunkBytes,
		onOversize: (descriptor, gzippedBytes) =>
			config.telemetry?.emit({
				metric: "chunk_oversize",
				visitorId,
				recorderVersion,
				sliceId: descriptor.sliceId,
				gzippedBytes,
				count: descriptor.count,
			}),
		onUndeliverable: (error) => emitFault("poison_dropped", error),
		isPageLoadSlice: (sliceId) => sliceId === pageLoadSliceId,
		onCatastrophic: (descriptor, gzippedBytes) => {
			config.telemetry?.emit({
				metric: "snapshot_fatal",
				visitorId,
				recorderVersion,
				sliceId: descriptor.sliceId,
				gzippedBytes,
				count: descriptor.count,
			});
			terminate();
		},
	});

	const onVisibilityChange = () => {
		const isVisible = document.visibilityState === "visible";
		meter.onVisibility(isVisible);
		const marker = {
			type: isVisible ? EventType.PageVisible : EventType.PageHidden,
			data: {
				url: window.location.href,
				referrer: document.referrer,
			},
			timestamp: Date.now(),
		};
		const stamped = record(marker);
		if (!isVisible) emitCost();
		if (!isVisible && config.sink.beacon && stamped.length > 0) {
			const fireAndForget = {
				visitorId,
				recorderVersion,
				envelope,
				errors: [],
			};
			for (const chunk of buildChunks(stamped, fireAndForget)) {
				// The shot is redundancy — the buffered copy is durable and the next drain or context
				// ships it — so a refusal is recorded in the errors channel, never counted.
				if (!config.sink.beacon(chunk.bytes, chunk.descriptor)) {
					onError(
						new Error(
							`hidden-marker beacon refused (slice ${chunk.descriptor.sliceId}, ` +
								`${chunk.bytes.length} gzipped bytes)`,
						),
						{ counts: false },
					);
				}
			}
		}
	};

	const resolveOptions = (href) =>
		config.rrwebRules ? optionsForUrl(config.rrwebRules, href) : {};
	// The restart-decision key must see masking functions (maskTextFn and the like): JSON.stringify
	// drops functions, so two routes differing only by a function would compare equal and skip the
	// restart, leaving the new route under the old masking. Functions serialize by source.
	const optionsKey = (options) =>
		JSON.stringify(options, (_key, value) =>
			typeof value === "function" ? `fn:${value.toString()}` : value,
		);
	// A capture restart driven by a route change is still a client-side navigation, so its
	// record-start PageLoad carries spa=true rather than reading downstream as a full page load.
	// Set by the restart branch, consumed by the one snapshot that follows.
	let restartIsSpaRoute = false;
	const onRrwebEmit = (event) => {
		record(event);
		if (!pageLoadEmitted && event.type === EventType.FullSnapshot) {
			pageLoadEmitted = true;
			recordPageLoad(
				{
					url: window.location.href,
					title: document.title,
					referrer: document.referrer,
				},
				restartIsSpaRoute,
			);
			restartIsSpaRoute = false;
		}
	};
	const startCapture = (href) => {
		pageLoadEmitted = false;
		const resolved = resolveOptions(href);
		activeOptionsKey = optionsKey(resolved);
		stopRecording = rrweb.record({
			checkoutEveryNms: 1_800_000,
			// Whether an input came from a human or a script is knowable only at capture: with this
			// set, every recorded input carries a userTriggered flag, so a tracking pixel writing a
			// hidden field never reads downstream as the visitor typing. A default, not locked — a
			// rule may override it.
			userTriggeredOnInput: true,
			...resolved,
			emit: onRrwebEmit,
			recordDOM: true,
			// The credential mask (masking.js): its keys are set after the spread so a rule can
			// neither drop the routing, displace it with maskAllInputs, nor hand rrweb a maskInputFn
			// of its own — the rule's masking intent, its fn included, is honored inside the composed
			// fn.
			...maskingPosture(resolved),
			// rrweb wraps its observer callbacks only when a handler is given; without one, a throw
			// inside capture reaches only the page's global error reporting and the recorder never
			// learns it is failing. It is not swallowed here (rrweb suppresses only on `true`): an
			// uncaught error on the page is the page's own error tracking's to see; ours is only to
			// know. Every throw still reaches it, including the ones after the first.
			//
			// Each wrapped callback belongs to one observer, so a throw costs that observer's events
			// while the rest keep recording — partial capture loss, not the whole-recorder failure
			// the stop threshold exists to end, and it does not count toward it. It is recorded once
			// per context, like the fault it raises: these callbacks run per event, so a throwing
			// observer throws for as long as the page churns, and describing each repeat would spend
			// main-thread time per event and crowd the chunk-borne error log with one string.
			errorHandler: (error) => {
				if (captureFailed) return;
				captureFailed = true;
				onError(error, { counts: false, fault: "capture_failed" });
			},
		});
	};

	startCapture(window.location.href);

	document.addEventListener("visibilitychange", onVisibilityChange);
	window.addEventListener("pagehide", onTeardown);
	window.addEventListener("pageshow", onRestore);
	document.addEventListener("freeze", onTeardown);
	document.addEventListener("resume", onRestore);

	// SPA route-awareness (route.js hooks the channels): a PageLoad — the same navigation marker a
	// full page load emits — goes out on each real route change, carrying the route's
	// url/title/referrer, without forcing a fresh DOM snapshot: rrweb already records the route's
	// DOM swap as mutations, so it stays replayable from the slice's covering snapshot. The patch
	// is restored on terminate.
	restoreRouteAwareness = watchRoutes({
		isActive: () => !terminated,
		onRoute: (href) => {
			// A route whose resolved options differ restarts capture: rrweb fixes its options at record()
			// (masking.js), so a restart is the only way to change them. The restart takes a fresh
			// snapshot — a new slice — whose record-start PageLoad attests the route. A route with
			// unchanged options emits the in-place PageLoad marker.
			if (optionsKey(resolveOptions(href)) !== activeOptionsKey) {
				restartIsSpaRoute = true;
				stopRecording?.();
				startCapture(href);
				return;
			}
			recordPageLoad(
				{
					url: href,
					title: document.title,
					referrer: document.referrer,
				},
				true,
			);
		},
		onRouteError: (error) =>
			onError(error, { counts: false, fault: "route_detection_failed" }),
	});

	uploader.start();
	costTimer = config.telemetry
		? setInterval(emitCostIfActive, COST_PING_INTERVAL_MS)
		: null;

	return { stop: terminate, flush: () => uploader.flush() };
}
