/**
 * A page's route identity: origin + path + fragment, every query dropped — the fragment's own included.
 *
 * A query string is the web's standard place for content that has nothing to do with which page this is — an email in a prefilled form link, a session or reset token, the click ids and tracking params an ad network appends. A hash-routed SPA keeps its route in the fragment, so the fragment is kept, and its own query (`#/checkout?token=…`) is a query all the same. That one rule — a `?`-run, ending at the next `#` or the end, is never identity — is stated here once (stripQueries) and read by every consumer that asks "which page is this": the telemetry plane's address (pageAddress below), mask-rule matching (masking.js resolves rules against the same strip), and route-change detection (index.js keys routes by pageAddress, so a query-only rewrite is not a navigation).
 *
 * pageAddress is what the telemetry plane carries. That plane is operational facts about the recorder, never page content, and it is written unmasked by construction: no rule reaches it, and nothing downstream can un-carry what a ping already delivered — so every ping's url is composed through here (each metric's one composition site: start()'s emits, sink.js's ping composers), and the shipped channel's door strips `url`/`referrer` again before the wire (httpTelemetry). Userinfo (`https://user:secret@host/`) does not survive the origin. A string no URL parser accepts — an empty referrer, an opaque or relative address — falls back to the bare strip, so a value that cannot be parsed is still never carried whole.
 */
export function stripQueries(address) {
	return address.replace(/\?[^#]*/g, "");
}

export function pageAddress(href) {
	const raw = typeof href === "string" ? href : "";
	try {
		const u = new URL(raw);
		if (u.origin && u.origin !== "null") {
			return u.origin + u.pathname + stripQueries(u.hash);
		}
	} catch {
		/* not a parseable absolute URL; the textual strip below still answers */
	}
	return stripQueries(raw);
}
