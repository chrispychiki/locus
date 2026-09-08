/**
 * The worker's Analytics Engine half: every AE datapoint the deployment writes, declared and gated in one place.
 *
 * AE is the rates plane; R2 is the exact one, and arrivals are R2 keys rather than an AE datapoint. writeDataPoint is fire-and-forget and the CAPTURE binding is optional, so a deploy or test without it records nothing. One AE dataset, indexed by snippet id — the segment common to every datapoint — and discriminated by the first blob, the metric name. Every metric's blob layout lives in telemetry-schema.json; `telemetryPoint` gates and builds the recorder's pings, and the builders beside it build the datapoints the worker originates about itself. Finer segments ride as blobs.
 */
import { SLICE_ID_SHAPE } from "./keys.js";
import SCHEMA from "./telemetry-schema.json";

// The row layout is declared once, in telemetry-schema.json — this module writes by it and
// `locus ae` serves it back to readers — so a layout edit lands on writer and reader together. The
// writer's own contract is pinned here: the uniform head every builder passes positionally, and
// the request-fact tail requestFacts() stamps in header order.
const declaredAt = (positions) =>
	positions.map((p) => SCHEMA.blobs[p]).join(",");
if (declaredAt(["1", "2", "3", "4"]) !== "metric,visitor,slice,version") {
	throw new Error(
		"telemetry-schema.json blobs 1-4 drifted from the writer's contract",
	);
}
if (declaredAt(["8", "9", "10"]) !== "origin,user_agent,as_organization") {
	throw new Error(
		"telemetry-schema.json blobs 8-10 drifted from the writer's contract",
	);
}

/** A metric declaration's positional map ({"5": "url", ...}) as its ordered name list, loud on a gap — a hole would silently shift every later value off its declared position. */
function declaredNames(map, first, what) {
	const positions = Object.keys(map)
		.map(Number)
		.sort((a, b) => a - b);
	positions.forEach((p, i) => {
		if (p !== first + i)
			throw new Error(
				`telemetry-schema.json ${what} positions must run contiguously from ${first}; got ${positions.join(",")}`,
			);
	});
	return positions.map((p) => map[String(p)]);
}

// AE caps a datapoint's blobs at this many bytes in total and throws past it. The free-text
// blobs — a url, a recorder's error string — are open-write input with no length bound, so an
// oversized one would take the whole write down and drop a valid birth out of the completeness
// denominator. Every blob is fitted to the remaining budget in order, so the identifying blobs
// (metric, visitor, slice) are never the ones cut, and the request facts — appended last — are
// the first.
export const AE_BLOB_BYTES = 5120;

// Request facts: what the platform attests about the sender, stamped onto every datapoint the
// worker writes — the recorder's pings and its own points alike. The Origin header locates the
// site even on a row whose body cannot name a page (a rejected ping, a thrown request), and its
// absence marks a non-browser client; the User-Agent header is the one device fact that exists
// for a page context that never delivered a chunk; the AS organization from Cloudflare's request
// metadata is the network's own name for itself, which for datacenter traffic is the bot tell.
// All three are attested by the transport rather than claimed by the body.
//
// They ride the uniform tail positions the schema declares, on every row: each metric's own
// blobs are padded out to the slot before the tail, then the facts append. AE has no joins and
// each metric samples independently, so a cut by sender must name one position that holds on
// every row; a metric that needs more own blobs than the region holds appends them
// after the request facts. HEAD_AND_OWN_BLOBS is the width of everything ahead of the facts —
// the four-blob uniform head plus the own-blob region.
const HEAD_AND_OWN_BLOBS =
	Math.min(
		...Object.keys(SCHEMA.blobs)
			.filter((k) => /^\d+$/.test(k))
			.map(Number)
			.filter((p) => p > 4),
	) - 1;
const requestFacts = (request) => [
	request.headers.get("Origin") ?? "",
	request.headers.get("User-Agent") ?? "",
	request.cf?.asOrganization ?? "",
];

/** One datapoint ordered by the declared layout: the uniform head, then the metric's own blobs and doubles by name — a value for every declared name and no undeclared ones, loud otherwise, so a write cannot drift off the layout readers are served. */
function declaredPoint(
	metric,
	visitorId,
	sliceId,
	version,
	own = {},
	doubles = {},
) {
	const decl = SCHEMA.metrics[metric];
	if (!decl)
		throw new Error(
			`metric ${metric} has no declared layout in telemetry-schema.json`,
		);
	const named = (names, values, kind) => {
		const extra = Object.keys(values).filter((k) => !names.includes(k));
		if (extra.length) {
			throw new Error(
				`${metric} ${kind} not in the declared layout: ${extra.join(", ")}`,
			);
		}
		return names.map((name) => {
			if (!(name in values))
				throw new Error(`${metric} ${kind} missing declared ${name}`);
			return values[name];
		});
	};
	const point = {
		blobs: [
			metric,
			visitorId,
			sliceId,
			version,
			...named(declaredNames(decl.blobs, 5, `${metric} blobs`), own, "blobs"),
		],
	};
	const orderedDoubles = named(
		declaredNames(decl.doubles, 1, `${metric} doubles`),
		doubles,
		"doubles",
	);
	if (orderedDoubles.length) point.doubles = orderedDoubles;
	return point;
}

export function fitted(blobs) {
	const encoder = new TextEncoder();
	const decoder = new TextDecoder();
	let budget = AE_BLOB_BYTES;
	return blobs.map((blob) => {
		const bytes = encoder.encode(String(blob ?? ""));
		let kept = bytes;
		if (bytes.length > budget) {
			// Never cut mid-sequence: a raw byte cut through a multibyte character decodes to U+FFFD,
			// which re-encodes larger than the bytes it replaced — the write would overrun the budget
			// the accounting swears by. Backing off to the sequence boundary keeps the decoded string
			// re-encoding to exactly the bytes kept.
			let end = budget;
			while (end > 0 && (bytes[end] & 0xc0) === 0x80) end -= 1;
			kept = bytes.slice(0, end);
		}
		budget -= kept.length;
		return decoder.decode(kept);
	});
}

/** Fire one datapoint onto AE, the request facts stamped at their uniform positions. The binding is optional — absent in tests and bare deploys, where this no-ops. */
function record(env, request, snippetId, blobs, doubles) {
	if (blobs.length > HEAD_AND_OWN_BLOBS) {
		throw new Error(
			`datapoint carries ${blobs.length} blobs ahead of the request facts, over HEAD_AND_OWN_BLOBS — the overflow belongs after the facts, or they shift off their uniform positions`,
		);
	}
	const padded =
		blobs.length < HEAD_AND_OWN_BLOBS
			? blobs.concat(Array(HEAD_AND_OWN_BLOBS - blobs.length).fill(""))
			: blobs;
	const point = {
		indexes: [snippetId],
		blobs: fitted([...padded, ...requestFacts(request)]),
	};
	if (doubles?.length) point.doubles = doubles;
	env.CAPTURE?.writeDataPoint(point);
}

const isString = (v) => typeof v === "string";
const isBoolean = (v) => typeof v === "boolean";
const isNumber = (v) => typeof v === "number" && Number.isFinite(v);
const isSliceId = (v) => isString(v) && SLICE_ID_SHAPE.test(v);
const isStringOrAbsent = (v) => v === undefined || v === null || isString(v);

// AE doubles are positional numbers with no null, so an unmeasured figure encodes as this
// sentinel. The recorder sends null for a heap reading on any non-Chromium engine, and for a
// backlog nothing has looked at yet; a reader must never take the sentinel for a measured 0.
const UNMEASURED = SCHEMA.unmeasured_sentinel;

// A figure is either measured or absent, and absence covers three cases that read the same
// downstream: the platform has no such API (heap is Chromium-only), the recorder took no reading (a
// buffer that would not answer holds an unknown backlog, not an empty one), and the device's bundle
// predates the field — a rolling upgrade means the worker outruns the recorder it is replacing,
// and refusing that recorder's older pings would go blind for the window of the upgrade. All three
// ride the sentinel.
const measuredOrAbsent = (v) =>
	v === undefined ||
	v === null ||
	(typeof v === "number" && Number.isFinite(v));

/**
 * One ping body → its AE blob/double arrays, or null when the body is not what
 * the recorder emits. The body is open-write input and the defense is shape, as
 * on the chunk path: an unknown metric, a missing field, a wrong type all bounce,
 * rather than landing a datapoint padded with invented zeros that would read
 * downstream as measurements never taken. cost_sample and recorder_fault may carry
 * sliceId null — both can fire before the visitor's first slice opens — and null
 * lands as an empty blob. recorderVersion may be absent — an upgrade reaches one device
 * at a time, so the worker always outruns some recorder it is replacing — and an
 * absent version lands as an empty blob, never a bounce.
 *
 * Every ping carries `ts`, the recorder's queue moment in epoch ms (recorder/src/sink.js): a
 * ping can dispatch long after it is queued, so the row's server-receive timestamp is not the
 * ping's time, and windowing pings against the recording's own client-minted clock needs the
 * client's stamp. It rides at the double position each metric declares below — absent (a custom
 * telemetry sink that does not stamp it) as the UNMEASURED sentinel — except slice_started,
 * whose slice-start double is already that instant on that clock. A field added to a metric
 * sits after `ts`: a double position, once written, means that figure forever.
 */
export function telemetryPoint(ping, visitorId) {
	if (typeof ping !== "object" || ping === null) return null;
	const metric = ping.metric;
	if (!isStringOrAbsent(ping.recorderVersion)) return null;
	const version = ping.recorderVersion ?? "";

	if (metric === "transport_probe") {
		// The recorder's channel-assessment shot, fired blind over sendBeacon at context open
		// (recorder/src/sink.js). A row here attests that a beacon from this context arrived —
		// presence only: this dataset samples per row, so a missing probe row says nothing about
		// one context, and the blocked-beacon share is read in aggregate, probe rows against births.
		if (!isSliceId(ping.sliceId) || !measuredOrAbsent(ping.ts)) return null;
		return declaredPoint(
			metric,
			visitorId,
			ping.sliceId,
			version,
			{},
			{ ts: ping.ts ?? UNMEASURED },
		);
	}
	if (metric === "slice_started") {
		if (
			!isSliceId(ping.sliceId) ||
			!isString(ping.url) ||
			!isBoolean(ping.first_slice) ||
			!isStringOrAbsent(ping.visitor_source)
		)
			return null;
		// The slice-start ms (the slice id's leading segment) rides as a double so reads can
		// window and bucket births on the same client-minted clock the R2 key's date partition
		// is derived from. AE SQL cannot parse it out of the blob — there is no string→number
		// conversion — and the row's own timestamp is server receive, which files
		// midnight-straddling and clock-skewed slices into the wrong day.
		//
		// The visitor source says how the recorder came by this visitor's id.
		return declaredPoint(
			metric,
			visitorId,
			ping.sliceId,
			version,
			{
				url: ping.url,
				first_slice: ping.first_slice ? "1" : "0",
				visitor_source: ping.visitor_source ?? "",
			},
			{ slice_open_ms: Number(ping.sliceId.split("-")[0]) },
		);
	}
	if (metric === "page_load") {
		if (
			!isSliceId(ping.sliceId) ||
			!isString(ping.url) ||
			!isString(ping.referrer) ||
			!isBoolean(ping.spaRoute) ||
			!measuredOrAbsent(ping.ts)
		)
			return null;
		return declaredPoint(
			metric,
			visitorId,
			ping.sliceId,
			version,
			{
				url: ping.url,
				referrer: ping.referrer,
				spa_route: ping.spaRoute ? "1" : "0",
			},
			{ ts: ping.ts ?? UNMEASURED },
		);
	}
	if (metric === "cost_sample") {
		if (
			!(ping.sliceId === null || isSliceId(ping.sliceId)) ||
			!isNumber(ping.mainThreadMs) ||
			!isNumber(ping.uploadBytes) ||
			!measuredOrAbsent(ping.deliveredBytes) ||
			!measuredOrAbsent(ping.heapBytesMax) ||
			!measuredOrAbsent(ping.heapBytesLimit) ||
			!measuredOrAbsent(ping.backlogBytesMax) ||
			!measuredOrAbsent(ping.ts) ||
			!measuredOrAbsent(ping.visibleMs)
		)
			return null;
		// uploadBytes is what the page context pushed; deliveredBytes is what the sink acknowledged.
		// visibleMs is the context's cumulative foreground-visible time. mainThreadMs, visibleMs,
		// and both byte figures are running totals over the context's life, re-reported by every
		// sample — a window read takes each context's max; summing samples multiply-counts.
		return declaredPoint(
			metric,
			visitorId,
			ping.sliceId ?? "",
			version,
			{},
			{
				main_thread_ms: ping.mainThreadMs,
				upload_bytes: ping.uploadBytes,
				heap_bytes_max: ping.heapBytesMax ?? UNMEASURED,
				heap_bytes_limit: ping.heapBytesLimit ?? UNMEASURED,
				backlog_bytes_max: ping.backlogBytesMax ?? UNMEASURED,
				delivered_bytes: ping.deliveredBytes ?? UNMEASURED,
				ts: ping.ts ?? UNMEASURED,
				visible_ms: ping.visibleMs ?? UNMEASURED,
			},
		);
	}
	if (metric === "chunk_oversize" || metric === "snapshot_fatal") {
		if (
			!isSliceId(ping.sliceId) ||
			!isNumber(ping.gzippedBytes) ||
			!isNumber(ping.count) ||
			!measuredOrAbsent(ping.ts)
		)
			return null;
		return declaredPoint(
			metric,
			visitorId,
			ping.sliceId,
			version,
			{},
			{
				gzipped_bytes: ping.gzippedBytes,
				count: ping.count,
				ts: ping.ts ?? UNMEASURED,
			},
		);
	}
	if (metric === "recorder_fault") {
		if (
			!(ping.sliceId === null || isSliceId(ping.sliceId)) ||
			!isString(ping.reason) ||
			!isString(ping.error) ||
			!isStringOrAbsent(ping.url) ||
			!measuredOrAbsent(ping.ts)
		)
			return null;
		// A fault exists to be acted on, and acting starts at the page it happened on — carried on
		// the row itself because the start-path faults precede any slice or birth to join to.
		return declaredPoint(
			metric,
			visitorId,
			ping.sliceId ?? "",
			version,
			{ reason: ping.reason, error: ping.error, url: ping.url ?? "" },
			{ ts: ping.ts ?? UNMEASURED },
		);
	}
	if (metric === "capture_gated") {
		// A page context the recorder declined by policy, the reason naming which gate, detail the
		// gate's own evidence for the verdict (the bot gate's firing detector names). The recorded
		// corpus is definitionally only what passed, and a per-context verdict like BotD's is
		// knowable no other way — without this row "how much of my traffic is bots" has no
		// witness. No slice ever exists for one. detail may be absent: a gate ping carries only
		// what its author's gate could say.
		if (
			!isString(ping.reason) ||
			!isString(ping.url) ||
			!isStringOrAbsent(ping.detail) ||
			!measuredOrAbsent(ping.ts)
		)
			return null;
		return declaredPoint(
			metric,
			visitorId,
			"",
			version,
			{ reason: ping.reason, url: ping.url, detail: ping.detail ?? "" },
			{ ts: ping.ts ?? UNMEASURED },
		);
	}
	return null;
}

/**
 * The datapoints the worker originates about itself, beside the recorder's pings
 * above, so every datapoint shape is declared in one place. The uniform head rides
 * every metric, known or empty — version is the attestor's: the recorder's bundle
 * version on a recorder ping, this worker's deploy version on a worker-originated
 * point — because a position that means different things per metric cannot be read
 * across metrics. The same rule puts the request facts at their uniform tail
 * positions past every metric's own blobs (record(), above).
 */
export const uploadRejected = (visitorId, sliceId, version, reason) =>
	declaredPoint("upload_rejected", visitorId, sliceId, version, { reason });

// An accepted chunk whose body did not arrive as gzip: the sniff blob names what the first
// bytes looked like (bodySniff, keys.js), which is the one observable that can tell a transcoding
// proxy (json) from a wrong-format compressor (deflate) from in-flight mangling (other).
export const uploadTranscoded = (visitorId, sliceId, version, sniff) =>
	declaredPoint("upload_transcoded", visitorId, sliceId, version, { sniff });

export const telemetryRejected = (visitorId, version, reason) =>
	declaredPoint("telemetry_rejected", visitorId, "", version, { reason });

export const workerError = (version, error, pathname) =>
	declaredPoint("worker_error", "", "", version, { error, pathname });

export const recordPoint = (env, request, snippetId, point) =>
	record(env, request, snippetId, point.blobs, point.doubles);

/** This deploy's version, from the version-metadata binding — stamped by the platform at deploy, so there is no hand-maintained constant to drift. Absent in tests and bare deploys, where it lands as an empty blob. */
export const workerVersion = (env) => env.CF_VERSION_METADATA?.id ?? "";
