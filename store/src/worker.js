/**
 * The R2 store worker — the deployment's whole server side.
 *
 * Three jobs: ingest chunks into R2 (keys.js gates the path and derives the key), accept recorder telemetry onto Workers Analytics Engine, and originate its own datapoints when a request's body fails a gate or the worker itself throws. A path that fails the shape gate answers 404 with no datapoint: an unparseable path names no snippet to file one under, and the internet's scanners would write the noise.
 *
 * An upload lands identically whether it arrives as PUT or POST — both are accepted (the recorder itself always POSTs; PUT serves any non-browser client of the open-write path), and the object is named entirely by its key — which lets the recorder send every request as a CORS simple request and never pay a preflight (recorder/src/sink.js). Nothing here reads Content-Type either: a chunk body is read by its magic bytes, and a telemetry body is parsed as JSON regardless of what it claims to be. Every accepted chunk is stored as gzip — a body that arrives as anything else is wrapped byte-exact and stamped as transcoded (keys.js states why the bytes are not a gate). Telemetry rides a path independent of the chunk channel, so a slice that was born but never arrived is still attested. CORS is wide open, because the recorder runs on the operator's site rather than this worker's origin.
 *
 * A serve yields no count here: a static-asset serve never runs the worker, and the workers.dev origin exposes no request analytics. Completeness is measured as arrivals against births — the birth ping is the denominator.
 *
 * The Analytics Engine half — every datapoint's declaration, gate, and write — is telemetry.js's; this file owns the request handling that feeds it.
 *
 * Reads are not served here: `locus` reads straight from R2 with bucket credentials.
 */
import {
	bodySniff,
	chunkTarget,
	MAX_CHUNK_BYTES,
	telemetryTarget,
} from "./keys.js";
import {
	recordPoint,
	telemetryPoint,
	telemetryRejected,
	uploadRejected,
	uploadTranscoded,
	workerError,
	workerVersion,
} from "./telemetry.js";

const CORS = {
	"Access-Control-Allow-Origin": "*",
	"Access-Control-Allow-Methods": "PUT, POST, OPTIONS",
	"Access-Control-Allow-Headers": "Content-Type",
	// Opens transferSize on this response's Resource Timing entry to the sending page — how the
	// recorder's transport probe reads whether its shot arrived (recorder/src/sink.js).
	"Timing-Allow-Origin": "*",
	"Access-Control-Max-Age": "86400",
	// The stamp that marks a response as this store's own answer, exposed so the recorder can
	// read it cross-origin: only a stamped refusal is the store's verdict on an upload — a 4xx
	// without it is a middlebox (a filtering proxy, a captive portal) answering in the store's
	// place, which the recorder retries rather than dropping (recorder/src/sink.js).
	"Locus-Store": "1",
	"Access-Control-Expose-Headers": "Locus-Store",
};

// The answer for a path outside the shape gate — doctor matches this exact body at the worker's
// root as proof that the origin's answerer is this worker.
export const NOT_FOUND = "not found";

function respond(status, body = null) {
	return new Response(body, { status, headers: CORS });
}

/** Read a request body without ever materializing more than the cap: a stated Content-Length over it rejects before a byte is read, and a stream that runs past it is abandoned mid-read (leaving the loop cancels it) — an open-write endpoint must not buffer an arbitrarily large body just to refuse it. Returns the bytes, or null for an over-cap body. */
async function readBodyCapped(request, maxBytes) {
	const declared = Number(request.headers.get("Content-Length"));
	if (Number.isFinite(declared) && declared > maxBytes) return null;
	if (request.body === null) return new Uint8Array(0);
	const parts = [];
	let total = 0;
	for await (const part of request.body) {
		total += part.length;
		if (total > maxBytes) return null;
		parts.push(part);
	}
	const bytes = new Uint8Array(total);
	let offset = 0;
	for (const part of parts) {
		bytes.set(part, offset);
		offset += part.length;
	}
	return bytes;
}

/** The received bytes under a gzip wrapper, byte-exact — the canonicalize half of accepting a non-gzip body (handle(), below). */
async function gzipBytes(bytes) {
	const stream = new Blob([bytes])
		.stream()
		.pipeThrough(new CompressionStream("gzip"));
	return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function handleTelemetry(request, env, pathname) {
	const target = telemetryTarget(pathname);
	if (target === null) return respond(404, NOT_FOUND);
	if (request.method !== "POST") return respond(405, "method not allowed");

	// The same capped read as the chunk path, under the store's one declared body bound: a ping is
	// a few hundred bytes, but the path is open-write and must not buffer an arbitrary body.
	const bytes = await readBodyCapped(request, MAX_CHUNK_BYTES);
	if (bytes === null) {
		recordPoint(
			env,
			request,
			target.snippetId,
			telemetryRejected(target.visitorId, workerVersion(env), "oversize"),
		);
		return respond(413, "telemetry too large");
	}
	let ping;
	try {
		ping = JSON.parse(new TextDecoder().decode(bytes));
	} catch {
		recordPoint(
			env,
			request,
			target.snippetId,
			telemetryRejected(target.visitorId, workerVersion(env), "not_json"),
		);
		return respond(400, "bad telemetry");
	}

	const point = telemetryPoint(ping, target.visitorId);
	if (point === null) {
		// A body that is JSON but not an object has no metric to name, and is reported as
		// exactly that — a metric of "undefined" would read as a named ping that failed.
		const named =
			typeof ping === "object" && ping !== null
				? `shape:${String(ping.metric).slice(0, 32)}`
				: "not_an_object";
		recordPoint(
			env,
			request,
			target.snippetId,
			telemetryRejected(target.visitorId, workerVersion(env), named),
		);
		return respond(400, "bad telemetry");
	}
	recordPoint(env, request, target.snippetId, point);
	return respond(204);
}

async function handle(request, env) {
	if (request.method === "OPTIONS") return respond(204);
	const { pathname } = new URL(request.url);

	if (pathname.startsWith("/telemetry/")) {
		return handleTelemetry(request, env, pathname);
	}

	const target = chunkTarget(pathname);
	if (target === null) return respond(404, NOT_FOUND);
	if (request.method !== "PUT" && request.method !== "POST") {
		return respond(405, "method not allowed");
	}

	const { key, snippetId, visitorId, sliceId } = target;
	const bytes = await readBodyCapped(request, MAX_CHUNK_BYTES);
	if (bytes === null) {
		recordPoint(
			env,
			request,
			snippetId,
			uploadRejected(visitorId, sliceId, workerVersion(env), "oversize"),
		);
		return respond(413, "chunk too large");
	}
	// gzip is lossless over whatever arrived, so the wrapper canonicalizes storage without
	// bounding recovery: ungzip an object and you hold exactly the wire's bytes — a bare JSON
	// payload parses, a deflate body inflates (hydration tries both), mangled bytes stay
	// mangled but preserved. The transcoded stamp lands after the put, so a row attests a
	// stored chunk, once per landed upload rather than per retry.
	const sniff = bodySniff(bytes);
	const body = sniff === "gzip" ? bytes : await gzipBytes(bytes);

	try {
		await env.CHUNKS.put(key, body);
	} catch (error) {
		// Two page contexts couriering the same batch write the same key concurrently — delivery
		// is at-least-once, and the key is derived from the records themselves — and R2
		// rate-limits same-object writes with error 10058. The loser's write is redundant rather
		// than failed: if the object is there, the data is delivered, so the answer is success and
		// the uploader resolves instead of burning a poison attempt on a race the winner settled. A
		// 10058 with no landed object is a real failure.
		if (!String(error).includes("(10058)")) throw error;
		if ((await env.CHUNKS.head(key)) === null) throw error;
	}
	if (sniff !== "gzip") {
		recordPoint(
			env,
			request,
			snippetId,
			uploadTranscoded(visitorId, sliceId, workerVersion(env), sniff),
		);
	}
	return respond(204);
}

export default {
	async fetch(request, env) {
		try {
			return await handle(request, env);
		} catch (error) {
			// An uncaught throw would otherwise be invisible: the request dies as a bare exception
			// before any datapoint, no logs are retained, and the failure reaches no plane the
			// deployment reads. It becomes its own AE datapoint, and the answer is a 500, which
			// the uploader retries like any non-ok. Writing that datapoint is itself guarded, so the
			// catch is total.
			try {
				const { pathname } = new URL(request.url);
				recordPoint(
					env,
					request,
					pathname.split("/")[2] || "unknown",
					workerError(workerVersion(env), String(error), pathname),
				);
			} catch {
				/* a failing failure-report must not re-throw */
			}
			return respond(500, "internal error");
		}
	},
};
