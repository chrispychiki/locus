/**
 * Ingest validation: upload path → R2 object key. This is the store's whole defense.
 *
 * The recorder's sink contract leaves the key layout to the deployment. This deployment's layout is `{snippetId}/{slice-start-date}/{visitor}/{slice}/{chunkKey}.json.gz`: snippet id first, so one operator's sites never comingle in a shared bucket; then the slice's start date, derived here from the slice-id's leading ms rather than carried in the upload path, so a time-window enumeration scopes to date prefixes instead of scanning the whole snippet id; then visitor; then slice ids, zero-padded and time-ordered, so a slice's chunks list in recording order. Every chunk of a slice files under its slice's start date, late arrivals included — a delayed chunk self-places by its stamped slice-id whenever it arrives, never split across a date boundary.
 *
 * The endpoint is open-write, as any analytics collector is: there is no upload secret to design against, and the defense is shape. The four path segments must look like what the recorder emits — snippet id (the site's id in this deployment), visitor id (cookie UUID or a deployment override), slice id, chunk key (the first event's counter) — and the body must fit the size cap. Anything else bounces. The body's bytes are never a gate: the recorder always sends gzip, but a visitor's environment can rewrite the body in flight — a scanning proxy, an antivirus fetch wrapper, a broken CompressionStream polyfill — and bouncing on that rewrite punishes only the visitor's recording, forever, while enforcing nothing on anyone who could comply. The worker canonicalizes every accepted body to gzip at rest and stamps the non-gzip arrivals (worker.js); bodySniff names what the first bytes look like.
 */

export const MAX_CHUNK_BYTES = 10 * 1024 * 1024;

// The one declaration of each identifier's shape; the path gates and the telemetry
// body gate are all built from these, so they can never disagree.
const SNIPPET_ID = "[a-z0-9]{3,64}";
const VISITOR_ID = "[A-Za-z0-9_-]{8,64}";
const SLICE_ID = "\\d{14}-[a-z0-9]{4}";
// The recorder's counter stamp (stream.js): epoch-ms then a 6-digit sequence.
const CHUNK_COUNTER = "\\d{19}";

const CHUNK_PATH = new RegExp(
	`^\\/chunks\\/(${SNIPPET_ID})\\/(${VISITOR_ID})\\/(${SLICE_ID})\\/(${CHUNK_COUNTER})$`,
);

const TELEMETRY_PATH = new RegExp(
	`^\\/telemetry\\/(${SNIPPET_ID})\\/(${VISITOR_ID})$`,
);

/** The slice id's shape, for validating slice ids that arrive in a body rather than a path. */
export const SLICE_ID_SHAPE = new RegExp(`^${SLICE_ID}$`);

/** Upload path → {key, snippetId, visitorId, sliceId}, or null when the path is not what the recorder emits. The segments come back alongside the key so the worker never re-splits a string it has already parsed. The layout is not private to this file: `locus ls` reads the key back positionally on the way out, so moving it is a change on both sides of the store. */
export function chunkTarget(pathname) {
	const match = CHUNK_PATH.exec(pathname);
	if (!match) return null;
	const [, snippetId, visitorId, sliceId, counter] = match;
	const date = new Date(Number(sliceId.split("-")[0]))
		.toISOString()
		.slice(0, 10);
	return {
		key: `${snippetId}/${date}/${visitorId}/${sliceId}/${counter}.json.gz`,
		snippetId,
		visitorId,
		sliceId,
	};
}

/** Telemetry ping path → {snippetId, visitorId}, or null. A ping's path carries the snippet id and the visitor; the ping's metric and slice ride in the JSON body. */
export function telemetryTarget(pathname) {
	const match = TELEMETRY_PATH.exec(pathname);
	if (!match) return null;
	return { snippetId: match[1], visitorId: match[2] };
}

/**
 * What an upload body's first bytes look like: "gzip" is stored as-is; anything else is
 * canonicalized and stamped with this verdict. The verdict is also the recoverability read for
 * an `upload_transcoded` row: "deflate" is zlib's CMF byte and "json" a bare payload — an
 * environment that decompressed or re-compressed the body in flight, so the stored object is a
 * readable chunk; "other" is in-flight byte mangling — the stored object is preserved evidence,
 * permanently unreadable, and hydration skips it as corrupt.
 */
export function bodySniff(bytes) {
	if (bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) return "gzip";
	if (bytes.length > 2 && bytes[0] === 0x78) return "deflate";
	let i = 0;
	while (
		i < bytes.length &&
		(bytes[i] === 0x20 ||
			bytes[i] === 0x09 ||
			bytes[i] === 0x0a ||
			bytes[i] === 0x0d)
	)
		i++;
	if (i < bytes.length && (bytes[i] === 0x7b || bytes[i] === 0x5b))
		return "json";
	return "other";
}
