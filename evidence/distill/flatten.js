import { isProjected, nameResolver, shownText } from "./project.js";
import {
	EventType,
	EventTypeNames,
	IncrementalSource,
	IncrementalSourceNames,
	MediaInteractionsNames,
	MouseInteractions,
	MouseInteractionsNames,
	NodeType,
	PointerTypesNames,
} from "./rrweb_constants.js";
import {
	collapse,
	describeUrl,
	truncate,
	truncateWords,
	VALUE_CAP,
} from "./text.js";

// A target's text identifies the element acted on — a label's worth, cut at a word boundary; the
// page content a large clickable container holds is the projection's testimony, not the click
// line's. A value (an Input's text, a Selection) IS the content, cut only at the shared value
// ceiling; a cut value testifies its true size beside the visible cut (input_chars / text_chars
// in extra).
export const TARGET_TEXT_CAP = 80;
const EXTRA_CAP = 200;

const SKIP_DATA = new Set([
	"source",
	"type",
	"id",
	"positions",
	"adds",
	"removes",
	"texts",
	"attributes",
	"ranges",
	"node",
	"initialOffset",
	"styles",
	"styleIds",
	"define",
	"commands",
	"isAttachIframe",
]);

const SKIP_ATTR = new Set(["_cssText", "d", "srcset", "data-srcset"]);

// The nearest anchor above a node in its own document: a shadow root is crossed to its host,
// since the host's anchor wraps what the shadow tree renders, while an iframe's document ends
// the walk — the browser follows no anchor of the parent page for a click inside the frame.
function enclosingHref(node) {
	let n = node.parentNode;
	while (n && n.type === NodeType.Element) {
		if (n.tagName === "shadowroot") {
			n = n.host;
			continue;
		}
		const tag = n.tagName.toLowerCase();
		if ((tag === "a" || tag === "area") && n.attributes?.href != null)
			return n.attributes.href;
		n = n.parentNode;
	}
	return null;
}

export function flatten(event, mirror, counters) {
	const cols = {
		type_str: null,
		url: null,
		tag: null,
		class: null,
		text: null,
		x: null,
		y: null,
		input: null,
		href: null,
		title: null,
		referrer: null,
		pointer_type: null,
		extra: null,
		hidden: null,
	};
	const data = event.data || {};
	const extra = {};
	const put = (k, v) => {
		if (v === undefined || v === null || typeof v === "object") return;
		if (SKIP_ATTR.has(k)) return;
		if (typeof v !== "string") {
			extra[k] = v;
			return;
		}
		extra[k] = v.startsWith("data:") ? describeUrl(v) : truncate(v, EXTRA_CAP);
	};
	// The wire is open-write, so a field can arrive with a type its column has
	// no honest projection for (a url that is an object, an x that is a string).
	// Such a value drops to null, counted via counters.wrongTyped — raw_json
	// keeps the bytes — instead of reaching the DB write, where binding it would
	// kill the whole distillation pass.
	const drop = () => {
		counters.wrongTyped += 1;
		return null;
	};
	const str = (v) => (typeof v === "string" ? v : v == null ? null : drop());
	const num = (v) =>
		typeof v === "number" && Number.isFinite(v) ? v : v == null ? null : drop();

	if (event.type === EventType.IncrementalSnapshot) {
		cols.type_str = IncrementalSourceNames[data.source] ?? String(data.source);
		// A subtype the generated schema cannot name must not fall back to the source's own name:
		// "MouseInteraction" is a kind the schema knows, so the unknown-kind counter — the one signal
		// that says the constants are stale against the recorder's rrweb — would never fire, and an
		// interaction nobody can name would read as an ordinary one for as long as the drift lasted.
		if (data.source === IncrementalSource.MouseInteraction) {
			cols.type_str =
				MouseInteractionsNames[data.type] ?? `${cols.type_str}:${data.type}`;
		} else if (data.source === IncrementalSource.MediaInteraction) {
			cols.type_str =
				MediaInteractionsNames[data.type] ?? `${cols.type_str}:${data.type}`;
		}
	} else {
		cols.type_str = EventTypeNames[event.type] ?? String(event.type);
	}

	for (const [k, v] of Object.entries(data)) {
		if (SKIP_DATA.has(k)) continue;
		if (k === "url" || k === "href") cols.url = str(v);
		else if (k === "title") cols.title = str(v);
		else if (k === "referrer") cols.referrer = str(v);
		else if (k === "x") cols.x = num(v);
		else if (k === "y") cols.y = num(v);
		else if (k === "pointerType") {
			cols.pointer_type =
				typeof v === "number"
					? (PointerTypesNames[v] ?? String(v))
					: v == null
						? null
						: drop();
		} else put(k, v);
	}

	// A move batch flattens to its endpoint: rrweb batches pointer samples on a sub-second cadence,
	// so endpoint-per-batch preserves motion at the scale behavior reads at — trajectory, dwell,
	// circling that spans batches — while the intra-batch samples stay in raw_json and the replay.
	const isMove =
		event.type === EventType.IncrementalSnapshot &&
		(data.source === IncrementalSource.MouseMove ||
			data.source === IncrementalSource.TouchMove ||
			data.source === IncrementalSource.Drag);
	if (isMove) {
		const last = data.positions?.[data.positions.length - 1];
		if (last) {
			cols.x = num(last.x);
			cols.y = num(last.y);
		}
	}

	if (
		event.type === EventType.IncrementalSnapshot &&
		data.source === IncrementalSource.Input
	) {
		if (typeof data.text === "string") {
			cols.input = truncate(data.text, VALUE_CAP);
			if (cols.input !== data.text) extra.input_chars = data.text.length;
		} else if (data.text != null) cols.input = drop();
		else if (typeof data.isChecked === "boolean")
			cols.input = String(data.isChecked);
		else cols.input = data.isChecked === undefined ? null : drop();
		delete extra.isChecked;
		delete extra.text;
	}

	if (
		event.type === EventType.IncrementalSnapshot &&
		data.source === IncrementalSource.Selection
	) {
		const range = data.ranges?.[0];
		if (
			range &&
			typeof range.start === "number" &&
			typeof range.end === "number" &&
			typeof range.startOffset === "number" &&
			typeof range.endOffset === "number" &&
			mirror.getNode(range.start)
		) {
			const startText = mirror.getTextContent(range.start);
			const selected =
				range.start === range.end
					? startText.slice(range.startOffset, range.endOffset)
					: `${startText.slice(range.startOffset)} … ${mirror.getTextContent(range.end).slice(0, range.endOffset)}`;
			const whole = collapse(selected);
			cols.text = truncate(whole, VALUE_CAP) || null;
			if (cols.text !== null && cols.text !== whole)
				extra.text_chars = whole.length;
		}
	}

	// A typed-value Input's target text is its own previous value (a textarea's textContent IS the value),
	// so quoting it would state every value twice, one event stale. The value rides in `input`; the element
	// stays identified by tag and class. A checkable Input keeps its text.
	const typedInput =
		cols.input !== null &&
		event.type === EventType.IncrementalSnapshot &&
		data.source === IncrementalSource.Input &&
		typeof data.text === "string";

	if (typeof data.id === "number") {
		const node = mirror.getNode(data.id);
		if (node?.tagName) {
			// The projection's visibility boundary, stamped per event: 1 when the target sits outside what the
			// page displays (a hidden field, a tracking pixel's sandbox frame), 0 when it is projected content,
			// NULL when the target never resolved.
			cols.hidden = isProjected(mirror, data.id) ? 0 : 1;
			cols.tag = node.tagName.toLowerCase();
			const attrs = node.attributes || {};
			cols.class = str(attrs.class);
			if (!cols.text && !typedInput) {
				const t = collapse(shownText(node));
				if (t) cols.text = truncateWords(t, TARGET_TEXT_CAP);
			}
			// A target showing no text of its own carries the page's name for it, by the projection's
			// derivation, so the event line and the projection call the control the same thing.
			if (!cols.text) {
				const name = nameResolver(mirror)(node);
				if (name) extra.label = truncate(name, EXTRA_CAP);
			}
			for (const [k, v] of Object.entries(attrs)) {
				if (k === "class") continue;
				if (k === "href") cols.href = str(v);
				else put(k, v);
			}
			// A click lands on whatever element sits under the pointer — the <em> inside the link, the
			// <svg> icon in it — while the browser follows the enclosing anchor, so a click's destination
			// is the nearest anchor's when the target has none of its own. Only a click: a scroll or an
			// input inside a link acts on nothing the anchor points at.
			const click =
				data.source === IncrementalSource.MouseInteraction &&
				data.type === MouseInteractions.Click;
			if (click && cols.href === null) cols.href = str(enclosingHref(node));
		}
	}

	// Whole pixels: the wire carries sub-pixel float coordinates, but a screen has no sub-pixel touch —
	// the fraction is jitter, not testimony, and the columns declare INTEGER.
	if (cols.x !== null) cols.x = Math.round(cols.x);
	if (cols.y !== null) cols.y = Math.round(cols.y);
	if (Object.keys(extra).length) cols.extra = JSON.stringify(extra);
	return cols;
}
