/**
 * Chunk assembly: stamped records → self-describing gzipped chunks, one per slice, ready for any sink.
 *
 * The chunk is exact raw rrweb plus the envelope the events cannot carry themselves: visitor id, slice id, the producing recorder's version, and the device facts — userAgent, language, time zone, screen, touch points (maxTouchPoints: the one fact that contradicts iPadOS's desktop-mode UA, which is byte-identical to macOS Safari's). rrweb events hold none of those: its Meta data is only {href, width, height}, the *viewport* (rrweb packages/types/src/index.ts), and device/os/browser derivation is evidence-side, at hydration. Nothing the events already carry rides beside them — a chunk's time range is the min and max of the very events inside it.
 *
 * Every fact in the chunk is stated by its honest witness. The visitor id and envelope are facts about the capture moment, and the couriering context — whichever page context drains the shared backlog, possibly days later, possibly under a different identity — can only testify to its own; so each slice's chunk takes them from the capture context the writer stamped beside its records (buffer.js), and the courier's own context stands in only where no stamp exists (another bundle's backlog), the same fallback shape as the per-record recorderVersion. The error log (`errors`) is the one ship-time fact: evictions and poison drops happen at shipping, so the courier is their honest witness.
 *
 * The descriptor (sink.js) carries what a sink needs to build its key and what the uploader must decide with; the key layout itself is the sink's, never the recorder's.
 *
 * Two compression paths share one payload builder. buildChunkPayloads is the cheap synchronous step (group by slice, build descriptor + payload). buildChunks gzips synchronously for the hidden-marker beacon, which cannot await a page that may already be unloading. The drain instead serializes then awaits gzipBytes, which compresses off the main thread (CompressionStream) so a large snapshot never freezes the host page; gzipSync is the fallback where CompressionStream is absent.
 */
import { gzipSync } from "fflate";

import { EventType } from "./rrweb_constants.js";

export function collectEnvelope(global = globalThis) {
	const envelope = {};
	try {
		envelope.userAgent = global.navigator.userAgent;
	} catch {
		/* the field stays absent */
	}
	try {
		envelope.language = global.navigator.language;
	} catch {
		/* the field stays absent */
	}
	try {
		envelope.timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
	} catch {
		/* the field stays absent */
	}
	try {
		envelope.screen = {
			width: global.screen.width,
			height: global.screen.height,
		};
	} catch {
		/* the field stays absent */
	}
	try {
		envelope.maxTouchPoints = global.navigator.maxTouchPoints;
	} catch {
		/* the field stays absent */
	}
	return envelope;
}

/**
 * @param {{sliceId: string, recorderVersion?: string, event: object}[]} records  in emission order
 * @param {{visitorId: string, recorderVersion: string, envelope: object, errors?: string[]}} context  the couriering context's own facts — the fallback witness for unstamped slices, and the sole witness for `errors`
 * @param {Object<string, {visitorId: string, envelope: object}>} [captureContexts]  per-slice capture contexts the buffer stored beside the records
 * @returns {{descriptor: object, payload: object}[]}
 */
export function buildChunkPayloads(records, context, captureContexts = {}) {
	const bySlice = new Map();
	for (const { sliceId, recorderVersion, event } of records) {
		if (!bySlice.has(sliceId)) {
			bySlice.set(sliceId, { recorderVersion, events: [] });
		}
		bySlice.get(sliceId).events.push(event);
	}

	const chunks = [];
	let errors = context.errors ?? [];
	for (const [sliceId, slice] of bySlice) {
		const events = slice.events;
		const capture = captureContexts?.[sliceId];
		const visitorId = capture?.visitorId ?? context.visitorId;
		const payload = {
			visitorId,
			sliceId,
			// The producing recorder, taken from the records — a slice is one page context's, so
			// every record in it carries the same version.
			recorderVersion: slice.recorderVersion ?? context.recorderVersion,
			envelope: capture?.envelope ?? context.envelope,
			events,
			errors,
		};
		errors = [];
		chunks.push({
			descriptor: {
				visitorId,
				sliceId,
				chunkKey: events[0].counter,
				count: events.length,
				hasFullSnapshot: events.some(
					(event) => event.type === EventType.FullSnapshot,
				),
			},
			payload,
		});
	}
	return chunks;
}

export function serializePayload(payload) {
	return new TextEncoder().encode(JSON.stringify(payload));
}

/** Gzip off the main thread where the platform allows, falling back to synchronous gzip. */
export async function gzipBytes(bytes) {
	if (typeof CompressionStream !== "undefined") {
		const compressed = new Blob([bytes])
			.stream()
			.pipeThrough(new CompressionStream("gzip"));
		return new Uint8Array(await new Response(compressed).arrayBuffer());
	}
	return gzipSync(bytes);
}

/** Synchronous build + gzip, for the hidden-marker beacon that cannot await unload. The records were captured moments ago in this very context, so the courier context IS the capture context and no stamped blocks are needed. */
export function buildChunks(records, context) {
	return buildChunkPayloads(records, context).map(
		({ descriptor, payload }) => ({
			descriptor,
			bytes: gzipSync(serializePayload(payload)),
		}),
	);
}
