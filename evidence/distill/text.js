// The text rules the distilled columns and the projection share: how whitespace collapses, how a string is cut
// without breaking a character, and how a URL is rendered. One home, so the same value never appears two ways
// depending on which column the reader is looking at.

export const collapse = (s) => s.replace(/\s+/g, " ").trim();

// Never land a cut boundary between the halves of a surrogate pair: a lone half is not a
// character, and it is what a naive slice mints every time it lands mid-astral. The one statement
// of the rule — every cut here and in the diff's elision (diff.js) goes through it.
export const safeCut = (s, i) => {
	const c = s.charCodeAt(i - 1);
	return c >= 0xd800 && c <= 0xdbff ? i - 1 : i;
};

// Cut at n code units with the cut kept visible — the ellipsis rides inside the cap, so a capped
// value never reads as complete.
export function truncate(s, n) {
	if (typeof s !== "string" || s.length <= n) return s;
	return `${s.slice(0, safeCut(s, n - 1))}…`;
}

// Cut at n code units keeping whole words where the text has them: the cut backtracks to the last
// space inside the cap when one sits close enough to matter, and cuts hard otherwise (unspaced
// text — CJK, tokens — still caps). Same visible-ellipsis and surrogate rules as truncate.
export function truncateWords(s, n) {
	const hard = truncate(s, n);
	if (hard === s || typeof hard !== "string") return hard;
	const kept = hard.slice(0, -1);
	const space = kept.lastIndexOf(" ");
	return space > 0 ? `${kept.slice(0, space).trimEnd()}…` : hard;
}

// Every ceiling here is a pathology ceiling, not a fit to real content: a value below it rides
// whole, and the cut exists against the blobs that sit far above anything of its role. Each role
// has its own, bounded by what that role can legitimately be.
//
// A value by a person's hand — an Input's text, a field's value, a selection — is bounded by a
// paste, and a paste can be a whole document; the ceiling sits above a document of pages and
// below the serialized state a script sets into a field. A text node is bounded by what one block
// of prose is — writing breaks into paragraphs, and even legal boilerplate runs a couple thousand
// characters per block — and sits below the minified JSON, log, or data dump that a text node
// past that is. A label attribute — placeholder, alt, name — is a phrase: accessibility guidance
// tops alt text around 125 characters, and anything past a few hundred is a description dump.
export const VALUE_CAP = 20000;
export const TEXT_CAP = 5000;
export const LABEL_CAP = 200;

// A cut value testifies its true size beside the visible cut, so a huge value is itself evidence:
// the shown text (the ellipsis riding inside the cap) and a note stating the count — an empty note
// when nothing was cut, so the caller composes both without branching.
export function cutNote(s, cap) {
	const shown = truncate(s, cap);
	if (shown === s) return [s, ""];
	return [shown, ` (first ${shown.length - 1} of ${s.length} chars)`];
}

// A field's value rides on one line with its line structure kept: a textarea renders its
// whitespace (the user-agent stylesheet keeps it), so a newline typed there is testimony, shown
// as \n rather than folded into a space. Backslashes escape too, so the written form maps back to
// exactly one value and two values never read the same. The event stream keys a repeated value
// on this same written form (locus/evidence/text.py's twin; test_text.py pins the two).
export const escapeValue = (s) =>
	s
		.replace(/\\/g, "\\\\")
		.replace(/\n/g, "\\n")
		.replace(/\r/g, "\\r")
		.replace(/\t/g, "\\t");

// A URL below the ceiling rides whole — its query can be semantic state (search terms, filter
// sets) and is content — while the multi-kB serialized-state and tracking monsters the cut exists
// for sit far above. At ~4 chars/token a ceiling-length URL costs a few hundred tokens on the
// handful of lines that carry one.
export const URL_CAP = 2000;

const magnitude = (n) =>
	n < 1024 ? `${n} chars` : `${Math.round(n / 1024)}kB`;

// A URL is affordance metadata: an inlined data: URI is summarized to its header and size — the
// testimony is that an image is present and roughly what it is, never its bytes. Past the ceiling
// the cut is structural, never mid-string: the query is shed whole and testifies its size
// ("?… (8kB query)"), because the path and fragment are the URL's identity — a hash-routed SPA's
// route lives in the fragment — and a shed URL still matches and follows by its path.
// The event stream and context blocks state this same rule in Python (src/locus/evidence/text.py);
// test_text.py pins the two together.
export function describeUrl(url) {
	if (typeof url !== "string") return url;
	if (url.startsWith("data:")) {
		const comma = url.indexOf(",");
		const header = url.slice(0, comma > 0 ? comma + 1 : 40).slice(0, 40);
		return `${header}… ${magnitude(url.length)} inlined`;
	}
	if (url.length <= URL_CAP) return url;
	const hash = url.indexOf("#");
	const frag = hash >= 0 ? url.slice(hash) : "";
	const base = hash >= 0 ? url.slice(0, hash) : url;
	const q = base.indexOf("?");
	if (q >= 0) {
		const kept = `${base.slice(0, q)}?… (${magnitude(base.length - q - 1)} query)${frag}`;
		if (kept.length <= URL_CAP) return kept;
	}
	return truncate(url, URL_CAP);
}
