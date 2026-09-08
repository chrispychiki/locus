// LightweightMirror — a browserless reconstruction of the DOM state rrweb's replayer holds at each event,
// on a flat id→node map. It is a transcription of the vendored replayer's own reconstruction machinery under the
// screenshot renderer's configuration (useVirtualDom: true, where every seek applies mutations through rrdom):
// FullSnapshots build per rrweb-snapshot's buildNodeWithSN against a real document, and mutations apply
// with rrdom's exact tree semantics — including its failure modes. A mutation the replayer aborts partway
// aborts identically here, in the same order (id-map removals that preceded the abort stay removed), because
// the screenshots the model grounds on come from that replayer: content it never painted must never enter the
// projection, and content it left painted must.
//
// The line the mirror draws: it decodes the rrweb serialization format completely — rr_* specials, textarea value materialization, _cssText stylesheet materialization and rr_split re-splitting, the link-carrying-_cssText style retag, boolean/null attribute decoding, SVG tag-case restoration, shadow/iframe routing, quirks-mode doctype injection, the body-forced-last document rule — but it does not perform the replayer's browser-environment adaptations, which exist to make a live replay safe or paintable and change no content: script→noscript renaming, on*→_on* handler neutering, CSP/preload defusing, the postcss :hover/device-media CSS rewrites, and seek-time chrome (hover classes, rrweb-paused, injected style rules). Element state the replayer holds as DOM properties rather than attributes (input value/checked, scroll offsets, media state, canvas bitmaps) lives on node.props.
//
// Not mirrored, by scope: CSSOM mutations (StyleSheetRule/StyleDeclaration/AdoptedStyleSheet — the style
// *elements'* text is mirrored; constructed-sheet state is not), canvas drawing commands, media playback
// timing, and font loading. These change paint through channels a tree projection does not read.
//
// Determinism: rrweb bails out of its add-resolution loop on a 500ms wall clock; the mirror iterates the same
// loop to a fixed point instead, so identical input always yields identical state.
import {
	EventType,
	IncrementalSource,
	NodeType,
	SVGTagMap,
} from "./rrweb_constants.js";

// --- CSS declaration text, transcribed from rrdom's style proxy (parseCSSText/toCSSText/camelize/hyphenate).
// A style-object attribute mutation is applied by parsing the current style attribute to a camelCase property
// map, mutating it with setProperty/removeProperty semantics, and re-serializing — which canonicalizes the
// whole declaration (hyphenated names, "name: value;" joined by spaces). The replayer paints that canonical
// form, so the mirror must produce it.
const CAMELIZE_RE = /-([a-z])/g;
const CUSTOM_PROPERTY_RE = /^--[a-zA-Z0-9-]+$/;
const UPPERCASE_RE = /\B([A-Z])/g;
// The skip check is stateless on purpose: upstream rrdom mints its /\B([A-Z])/g fresh per style-getter
// call, so every property name is tested from index 0. Reusing the global UPPERCASE_RE with .test()
// would advance lastIndex and let alternate camelCase names through — content the replayer never paints.
const HAS_UPPERCASE_RE = /\B[A-Z]/;
const camelize = (str) =>
	CUSTOM_PROPERTY_RE.test(str)
		? str
		: str.replace(CAMELIZE_RE, (_, c) => (c ? c.toUpperCase() : ""));
const hyphenate = (str) => str.replace(UPPERCASE_RE, "-$1").toLowerCase();

function parseCSSText(cssText) {
	const res = {};
	const listDelimiter = /;(?![^(]*\))/g;
	const propertyDelimiter = /:(.+)/;
	const comment = /\/\*.*?\*\//g;
	cssText
		.replace(comment, "")
		.split(listDelimiter)
		.forEach((item) => {
			if (item) {
				const tmp = item.split(propertyDelimiter);
				if (tmp.length > 1) res[camelize(tmp[0].trim())] = tmp[1].trim();
			}
		});
	return res;
}

function toCSSText(style) {
	const properties = [];
	for (const name in style) {
		const value = style[name];
		if (typeof value !== "string") continue;
		properties.push(`${hyphenate(name)}: ${value};`);
	}
	return properties.join(" ");
}

function cssSetProperty(map, name, value, priority) {
	if (HAS_UPPERCASE_RE.test(name)) return;
	const normalized = camelize(name);
	if (!value) delete map[normalized];
	else map[normalized] = value;
	if (priority === "important") map[normalized] += " !important";
}

function cssRemoveProperty(map, name) {
	if (HAS_UPPERCASE_RE.test(name)) return;
	delete map[camelize(name)];
}

function applyStyleDiff(current, diff) {
	const map = parseCSSText(current || "");
	for (const prop in diff) {
		const v = diff[prop];
		if (v === false) cssRemoveProperty(map, prop);
		else if (Array.isArray(v)) cssSetProperty(map, prop, v[0], v[1]);
		else cssSetProperty(map, prop, v);
	}
	return toCSSText(map);
}

// rrweb's getTagName minus the script→noscript execution defusing: SVG tag names are restored to their
// case-sensitive forms, and a <link> carrying an inlined stylesheet (_cssText) materializes as a <style>.
function getTagName(sn) {
	let tagName = SVGTagMap[sn.tagName] || sn.tagName;
	if (tagName === "link" && sn.attributes && sn.attributes._cssText)
		tagName = "style";
	return tagName;
}

// rrweb-snapshot's isNodeMetaEqual: decides whether an add for an already-known id reuses the existing node
// (a move) or displaces it with a fresh build.
function isNodeMetaEqual(a, b) {
	if (!a || !b || a.type !== b.type) return false;
	if (a.type === NodeType.Document) return a.compatMode === b.compatMode;
	if (a.type === NodeType.DocumentType)
		return (
			a.name === b.name &&
			a.publicId === b.publicId &&
			a.systemId === b.systemId
		);
	if (
		a.type === NodeType.Comment ||
		a.type === NodeType.Text ||
		a.type === NodeType.CDATA
	)
		return a.textContent === b.textContent;
	if (a.type === NodeType.Element)
		return (
			a.tagName === b.tagName &&
			JSON.stringify(a.attributes) === JSON.stringify(b.attributes) &&
			a.isSVG === b.isSVG &&
			a.needBlock === b.needBlock
		);
	return false;
}

// rrweb applies only the LAST text mutation per node in a batch.
function uniqueTextMutations(mutations) {
	const seen = new Set();
	const out = [];
	for (let i = mutations.length; i--; ) {
		const m = mutations[i];
		if (!seen.has(m.id)) {
			out.push(m);
			seen.add(m.id);
		}
	}
	return out;
}

// rrweb's out-of-order add resolution: queued adds are arranged into parent/sibling trees and replayed
// top-down once their anchors exist.
function queueToResolveTrees(queue) {
	const map = {};
	const put = (m, parent) => {
		const t = { value: m, parent, children: [] };
		map[m.node.id] = t;
		return t;
	};
	const roots = [];
	for (const m of queue) {
		const { nextId, parentId } = m;
		if (nextId && nextId in map) {
			const nextT = map[nextId];
			if (nextT.parent) {
				const idx = nextT.parent.children.indexOf(nextT);
				nextT.parent.children.splice(idx, 0, put(m, nextT.parent));
			} else {
				roots.splice(roots.indexOf(nextT), 0, put(m, null));
			}
			continue;
		}
		if (parentId in map) {
			map[parentId].children.push(put(m, map[parentId]));
			continue;
		}
		roots.push(put(m, null));
	}
	return roots;
}

function iterateResolveTree(tree, cb) {
	cb(tree.value);
	for (let i = tree.children.length - 1; i >= 0; i--) {
		iterateResolveTree(tree.children[i], cb);
	}
}

// A mutation the replayer aborts partway: rrdom's tree ops throw plain Errors (not DOMExceptions), which
// escape the replayer's DOMException-only catch and unwind the rest of the mutation event; replay then
// continues with the next event.
class MutationAbort extends Error {}

const isContainerType = (t) =>
	t === NodeType.Element || t === NodeType.Document;

export class LightweightMirror {
	constructor() {
		this.nodes = new Map();
		this.rootDoc = null;
		this.legacyMissingNodeRetryMap = {};
		// A Document add whose iframe host hasn't arrived yet waits here across mutations (Replayer state).
		this.newDocumentQueue = [];
		// The serialized node whose children the build has already consumed (a textarea whose value attribute
		// materialized as its text): the replayer drops them from the DOM it builds, and the mirror does the
		// same. The mirror never writes to the event it is handed, so the fact is held here.
		this.childrenConsumed = null;
	}

	clear() {
		this.nodes.clear();
		this.rootDoc = null;
		this.legacyMissingNodeRetryMap = {};
		this.newDocumentQueue = [];
		this.childrenConsumed = null;
	}

	get size() {
		return this.nodes.size;
	}

	getNode(id) {
		return id == null ? null : (this.nodes.get(id) ?? null);
	}

	has(id) {
		return this.nodes.has(id);
	}

	getTextContent(id) {
		const node = this.getNode(id);
		return node ? this._textOf(node) : "";
	}

	_textOf(node) {
		if (
			node.type === NodeType.Text ||
			node.type === NodeType.Comment ||
			node.type === NodeType.CDATA
		) {
			return node.textContent || "";
		}
		return node.childNodes.map((c) => this._textOf(c)).join("");
	}

	// The single stepping entry: consume one raw rrweb event. Returns the mutation report for Mutation
	// events, null otherwise.
	applyEvent(event) {
		if (event.type === EventType.FullSnapshot) {
			this.clear();
			if (event.data?.node) this.addNode(event.data.node);
			return null;
		}
		if (event.type !== EventType.IncrementalSnapshot) return null;
		const d = event.data || {};
		if (d.source === IncrementalSource.Mutation) return this.applyMutation(d);
		if (d.source === IncrementalSource.Input) this.applyInput(d);
		else if (d.source === IncrementalSource.Scroll) this.applyScroll(d);
		return null;
	}

	// Build a serialized subtree (a FullSnapshot's node, or any detached serialized tree). Snapshot realm:
	// the replayer rebuilds FullSnapshots against a real document, then hands the tree to rrdom, so
	// realm-dependent behavior (img rr_dataURL) takes the real-DOM path here.
	addNode(sn) {
		const iframes = [];
		const node = this._buildWithSN(sn, {
			doc: null,
			skipChild: false,
			realm: "snapshot",
			iframes,
		});
		if (node && node.type === NodeType.Document && !this.rootDoc)
			this.rootDoc = node;
		for (const el of iframes) this._drainNewDocumentQueue(el);
		return node;
	}

	applyMutation(d) {
		const report = {
			aborted: null,
			missingRemoves: [],
			missingTexts: [],
			missingAttributes: [],
			droppedAdds: [],
		};
		try {
			this._applyMutation(d, report);
		} catch (e) {
			if (e instanceof MutationAbort) report.aborted = e.message;
			else throw e;
		}
		return report;
	}

	// rrweb's applyInput sets DOM properties, not attributes; successive inputs overwrite (last wins).
	applyInput(d) {
		if (d.id === -1) return;
		const node = this.nodes.get(d.id);
		if (!node) return;
		node.props.checked = d.isChecked;
		node.props.value = d.text;
	}

	applyScroll(d) {
		if (d.id === -1) return;
		const node = this.nodes.get(d.id);
		if (!node) return;
		node.props.scrollLeft = d.x;
		node.props.scrollTop = d.y;
	}

	_newNode(type, doc) {
		return {
			id: null,
			type,
			tagName: undefined,
			isSVG: false,
			attributes: {},
			props: {},
			textContent: null,
			childNodes: [],
			parentNode: null,
			ownerDocument: doc,
			shadowRoot: null,
			contentDocument: null,
			meta: null,
		};
	}

	_newDocument() {
		const doc = this._newNode(NodeType.Document, null);
		doc.ownerDocument = doc;
		doc.compatMode = "CSS1Compat";
		return doc;
	}

	_newShadowRoot(host) {
		const sr = this._newNode(NodeType.Element, host.ownerDocument);
		sr.tagName = "shadowroot";
		// Upward inverse of the host→shadowRoot link, for consumers walking a node toward the top (the
		// projection walks downward and renders shadow trees in place; a visibility walk needs the way back).
		sr.host = host;
		return sr;
	}

	// rrweb-snapshot's buildNodeWithSN: reuse a mapped meta-equal node, else build fresh and (re)map the id —
	// a displaced predecessor stays in the tree, no longer id-addressable, exactly as the replayer leaves it.
	_buildWithSN(sn, opts) {
		const { doc, skipChild, realm, iframes } = opts;
		const existing = this.nodes.get(sn.id);
		if (existing && isNodeMetaEqual(existing.meta, sn)) return existing;

		let node = this._buildNode(sn, doc, realm);
		if (!node) return null;

		if (sn.type === NodeType.Document) {
			// The replayer reuses the target document: doc.open() clears it, and a BackCompat snapshot whose
			// children carry no doctype gets one written in to force quirks mode (unserialized, never mapped).
			node = doc && doc.type === NodeType.Document ? doc : this._newDocument();
			for (const c of node.childNodes) c.parentNode = null;
			node.childNodes = [];
			node.compatMode = sn.compatMode || "CSS1Compat";
			if (
				sn.compatMode === "BackCompat" &&
				Array.isArray(sn.childNodes) &&
				sn.childNodes.length &&
				sn.childNodes[0].type !== NodeType.DocumentType
			) {
				const xhtml =
					sn.childNodes[0].type === NodeType.Element &&
					sn.childNodes[0].attributes &&
					sn.childNodes[0].attributes.xmlns === "http://www.w3.org/1999/xhtml";
				const doctype = this._newNode(NodeType.DocumentType, node);
				doctype.name = "html";
				doctype.publicId = xhtml
					? "-//W3C//DTD XHTML 1.0 Transitional//EN"
					: "-//W3C//DTD HTML 4.0 Transitional//EN";
				doctype.systemId = "";
				this._append(node, doctype);
			}
		}

		node.id = sn.id;
		node.meta = sn;
		this.nodes.set(sn.id, node);

		const consumed = this.childrenConsumed === sn;
		this.childrenConsumed = null;

		if (
			isContainerType(sn.type) &&
			!skipChild &&
			!consumed &&
			Array.isArray(sn.childNodes)
		) {
			const targetDoc = node.type === NodeType.Document ? node : doc;
			for (const childSn of sn.childNodes) {
				const child = this._buildWithSN(childSn, { ...opts, doc: targetDoc });
				if (!child) continue;
				if (
					childSn.isShadow &&
					node.type === NodeType.Element &&
					node.shadowRoot
				) {
					this._append(node.shadowRoot, child);
				} else if (
					sn.type === NodeType.Document &&
					childSn.type === NodeType.Element
				) {
					// The replayer appends <html> to its document by detaching <body>, appending, and re-appending
					// body — forcing body to be html's last child even when the snapshot serialized trailing siblings.
					const body = child.childNodes.find(
						(c) => (c.tagName || "").toLowerCase() === "body",
					);
					if (body) this._detach(body);
					this._append(node, child);
					if (body) this._append(child, body);
				} else {
					this._append(node, child);
				}
				if ((child.tagName || "").toLowerCase() === "iframe" && iframes)
					iframes.push(child);
			}
		}
		return node;
	}

	// rrweb-snapshot's buildNode. Attribute decoding is the serialization format's, minus the environment
	// adaptations (see header); rr_* specials materialize per the replayer.
	_buildNode(sn, doc, realm) {
		switch (sn.type) {
			case NodeType.Document:
				return this._newDocument();
			case NodeType.DocumentType: {
				const node = this._newNode(NodeType.DocumentType, doc);
				node.name = sn.name || "html";
				node.publicId = sn.publicId;
				node.systemId = sn.systemId;
				return node;
			}
			case NodeType.Element:
				return this._buildElement(sn, doc, realm);
			case NodeType.Text: {
				const node = this._newNode(NodeType.Text, doc);
				node.textContent = sn.textContent;
				return node;
			}
			case NodeType.CDATA: {
				const node = this._newNode(NodeType.CDATA, doc);
				node.textContent = sn.textContent;
				return node;
			}
			case NodeType.Comment: {
				const node = this._newNode(NodeType.Comment, doc);
				node.textContent = sn.textContent;
				return node;
			}
			default:
				return null;
		}
	}

	_buildElement(sn, doc, realm) {
		// Building through a non-document target (an add whose rootId resolves to an element) makes the
		// replayer call createElement on a non-document, which throws and aborts the mutation.
		if (realm === "mutation" && doc && doc.type !== NodeType.Document) {
			throw new MutationAbort(
				`add of node ${sn.id} targets non-document root ${doc.id}`,
			);
		}
		const tagName = getTagName(sn);
		const node = this._newNode(NodeType.Element, doc);
		node.tagName = tagName;
		node.isSVG = !!sn.isSVG;
		if (tagName.toLowerCase() === "iframe") {
			node.contentDocument = this._newDocument();
			// Upward inverse of the iframe→content link, same as a shadow root's host: the projection
			// renders the content document in place, and a visibility walk needs the way back up.
			node.contentDocument.host = node;
		}

		const src = sn.attributes || {};
		const special = {};
		let textareaValued = false;
		for (const name in src) {
			if (!Object.hasOwn(src, name)) continue;
			let value = src[name];
			if (tagName === "option" && name === "selected" && value === false)
				continue;
			if (value === null) continue;
			if (value === true) value = "";
			if (name.startsWith("rr_")) {
				special[name] = value;
				continue;
			}
			if (typeof value !== "string") continue;
			if (tagName === "style" && name === "_cssText") {
				this._buildStyleText(node, sn, value);
				continue;
			}
			if (tagName === "textarea" && name === "value") {
				this._append(node, this._syntheticText(value, doc));
				textareaValued = true;
				continue;
			}
			if (tagName === "img" && src.srcset && src.rr_dataURL) {
				// The replayer's attribute chain routes EVERY remaining attribute of such an image into this
				// branch (no name check in the source), so the image keeps only rrweb-original-srcset plus the
				// rr_dataURL materialization below.
				node.attributes["rrweb-original-srcset"] = String(src.srcset);
				continue;
			}
			node.attributes[name] = String(value);
		}
		// The replayer drops a textarea's serialized children once the value attribute materialized as text.
		if (textareaValued) this.childrenConsumed = sn;

		for (const name in special) {
			const value = special[name];
			if (tagName === "canvas" && name === "rr_dataURL") {
				node.props.rr_dataURL = String(value);
			} else if (tagName === "img" && name === "rr_dataURL") {
				if (realm === "mutation") {
					// rrdom elements carry no currentSrc, so the replayer's rr_dataURL swap throws a TypeError on
					// every mutation-built image and aborts the mutation. Snapshot builds run on a real document
					// and swap cleanly.
					throw new MutationAbort(
						`mutation-built img ${sn.id} carries rr_dataURL (replayer TypeError)`,
					);
				}
				node.attributes["rrweb-original-src"] = String(src.src);
				node.attributes.src = String(value);
			}
			if (name === "rr_width" || name === "rr_height") {
				const map = parseCSSText(node.attributes.style || "");
				cssSetProperty(
					map,
					name === "rr_width" ? "width" : "height",
					String(value),
				);
				node.attributes.style = toCSSText(map);
			} else if (name === "rr_mediaCurrentTime" && typeof value === "number") {
				node.props.currentTime = value;
			} else if (name === "rr_mediaState") {
				if (value === "played") node.props.paused = false;
				else if (value === "paused") node.props.paused = true;
			} else if (name === "rr_mediaPlaybackRate" && typeof value === "number") {
				node.props.playbackRate = value;
			} else if (name === "rr_mediaMuted" && typeof value === "boolean") {
				node.props.muted = value;
			} else if (name === "rr_mediaLoop" && typeof value === "boolean") {
				node.props.loop = value;
			} else if (name === "rr_mediaVolume" && typeof value === "number") {
				node.props.volume = value;
			} else if (name === "rr_open_mode") {
				node.attributes.rr_open_mode = value;
			}
		}

		if (sn.isShadowHost) {
			if (!node.shadowRoot) node.shadowRoot = this._newShadowRoot(node);
			else for (const c of node.shadowRoot.childNodes.slice()) this._detach(c);
		}
		return node;
	}

	_syntheticText(value, doc) {
		const node = this._newNode(NodeType.Text, doc);
		node.textContent = value;
		return node;
	}

	// rrweb's buildStyleNode: an inlined stylesheet (_cssText) re-splits across the serialized text children
	// when there are any (writing onto the serialized nodes, which the child build then materializes), else
	// materializes as one unserialized text child. The mirror carries the recorded CSS verbatim — the
	// replayer's postcss adaptation rewrites selectors for hover/media simulation, not content.
	_buildStyleText(node, sn, cssText) {
		const textKids = (sn.childNodes || []).filter(
			(c) => c.type === NodeType.Text,
		);
		if (textKids.length) {
			const splits = cssText.split("/* rr_split */");
			while (splits.length > 1 && splits.length > textKids.length) {
				splits.splice(-2, 2, splits.slice(-2).join(""));
			}
			for (let i = 0; i < textKids.length; i++) {
				if (i === splits.length) break;
				textKids[i].textContent = splits[i];
			}
		} else {
			this._append(node, this._syntheticText(cssText, node.ownerDocument));
		}
	}

	// Tree operations follow rrdom's semantics.

	_append(parent, child) {
		if (
			parent.type === NodeType.Document &&
			(child.type === NodeType.Element || child.type === NodeType.DocumentType)
		) {
			if (parent.childNodes.some((c) => c.type === child.type)) {
				throw new MutationAbort(
					`document ${parent.id} already has a ${child.type === NodeType.Element ? "documentElement" : "doctype"}`,
				);
			}
		}
		if (child.parentNode) this._detach(child);
		parent.childNodes.push(child);
		child.parentNode = parent;
		child.ownerDocument = parent.ownerDocument;
		return child;
	}

	_insertBefore(parent, child, ref) {
		if (
			parent.type === NodeType.Document &&
			(child.type === NodeType.Element || child.type === NodeType.DocumentType)
		) {
			if (parent.childNodes.some((c) => c.type === child.type)) {
				throw new MutationAbort(
					`document ${parent.id} already has a ${child.type === NodeType.Element ? "documentElement" : "doctype"}`,
				);
			}
		}
		if (!ref) return this._append(parent, child);
		if (ref.parentNode !== parent) {
			throw new MutationAbort(
				`insert of node ${child.id} before a node that is not a child of parent ${parent.id}`,
			);
		}
		if (child === ref) return child;
		if (child.parentNode) this._detach(child);
		parent.childNodes.splice(parent.childNodes.indexOf(ref), 0, child);
		child.parentNode = parent;
		child.ownerDocument = parent.ownerDocument;
		return child;
	}

	_detach(child) {
		const parent = child.parentNode;
		if (!parent) return;
		const i = parent.childNodes.indexOf(child);
		if (i >= 0) parent.childNodes.splice(i, 1);
		child.parentNode = null;
	}

	_nextSibling(node) {
		const parent = node.parentNode;
		if (!parent) return null;
		const i = parent.childNodes.indexOf(node);
		return i >= 0 && i + 1 < parent.childNodes.length
			? parent.childNodes[i + 1]
			: null;
	}

	// rrdom's contains: same-document only, then a parentNode walk (which stops at shadow boundaries).
	_contains(ancestor, node) {
		if (node.ownerDocument !== ancestor.ownerDocument) return false;
		if (node === ancestor) return true;
		while (node.parentNode) {
			if (node.parentNode === ancestor) return true;
			node = node.parentNode;
		}
		return false;
	}

	// The replayer's Mirror.removeNodeFromMap: drop the node's CURRENT id mapping and recurse over its
	// light children — shadow roots and iframe content documents are not childNodes, so their subtrees stay
	// addressable after their host is removed, exactly as the replayer leaves them.
	_removeNodeFromMap(node) {
		if (node.id != null) this.nodes.delete(node.id);
		for (const c of node.childNodes) this._removeNodeFromMap(c);
	}

	// Mutation application transcribes Replayer.applyMutation.

	_applyMutation(d, report) {
		const removes = (d.removes || []).filter((m) => {
			if (!this.nodes.has(m.id)) {
				report.missingRemoves.push(m.id);
				return false;
			}
			return true;
		});
		for (const m of removes) {
			const target = this.nodes.get(m.id);
			if (!target) continue;
			let parent = this.nodes.get(m.parentId);
			if (!parent) {
				report.missingRemoves.push(m.parentId);
				continue;
			}
			if (m.isShadow && parent.shadowRoot) parent = parent.shadowRoot;
			// Id-map removal precedes the detach, so a remove that then aborts (declared parent isn't the real
			// one) leaves the subtree attached and painted but no longer id-addressable.
			this._removeNodeFromMap(target);
			if (target.parentNode !== parent) {
				throw new MutationAbort(
					`remove of node ${m.id} whose declared parent ${m.parentId} is not its parent`,
				);
			}
			this._detach(target);
		}

		this._applyAdds(d.adds || [], report);

		for (const m of uniqueTextMutations(d.texts || [])) {
			const target = this.nodes.get(m.id);
			if (!target) {
				if (!(d.removes || []).some((r) => r.id === m.id))
					report.missingTexts.push(m.id);
				continue;
			}
			if (target.type === NodeType.Element) {
				// rrdom's textContent setter: children replaced by exactly one text node, empty value included —
				// a later add appends alongside it and both render.
				for (const c of target.childNodes) c.parentNode = null;
				target.childNodes = [];
				this._append(
					target,
					this._syntheticText(m.value, target.ownerDocument),
				);
			} else {
				target.textContent = m.value;
			}
		}

		for (const m of d.attributes || []) {
			const target = this.nodes.get(m.id);
			if (!target) {
				if (!(d.removes || []).some((r) => r.id === m.id))
					report.missingAttributes.push(m.id);
				continue;
			}
			for (const name in m.attributes) {
				const value = m.attributes[name];
				if (value === null) {
					delete target.attributes[name];
				} else if (typeof value === "string") {
					const tag = (target.tagName || "").toLowerCase();
					if (name === "_cssText" && (tag === "link" || tag === "style")) {
						if (this._rebuildCssTextNode(target, m.attributes)) break;
						target.attributes[name] = value;
					} else if (name === "value" && tag === "textarea") {
						for (const c of target.childNodes) c.parentNode = null;
						target.childNodes = [];
						this._append(
							target,
							this._syntheticText(value, target.ownerDocument),
						);
					} else {
						target.attributes[name] = value;
					}
				} else if (name === "style" && value && typeof value === "object") {
					target.attributes.style = applyStyleDiff(
						target.attributes.style,
						value,
					);
				}
			}
		}
	}

	// The replayer's _cssText attribute path rebuilds the whole node from its stored meta merged with the
	// mutation's attribute batch (re-deriving the tag, so a <link> gaining _cssText becomes a <style>),
	// maps the id to the fresh node, and swaps it into the tree; the batch is folded into the stored meta
	// and the remaining attribute names are skipped (they rode the rebuild). A detached target gets the
	// fresh build and the id remap but no tree swap; the caller then falls through to a plain set.
	_rebuildCssTextNode(target, mutAttrs) {
		if (!target.meta) return false;
		let newNode = null;
		try {
			const newSn = {
				...target.meta,
				attributes: { ...target.meta.attributes, ...mutAttrs },
			};
			newNode = this._buildWithSN(newSn, {
				doc: target.ownerDocument,
				skipChild: true,
				realm: "mutation",
			});
			Object.assign(target.meta.attributes, mutAttrs);
			if (newNode && target.parentNode) {
				const parent = target.parentNode;
				const sibling = this._nextSibling(target);
				this._detach(target);
				this._insertBefore(parent, newNode, sibling);
				return true;
			}
		} catch {
			// The replayer swallows rebuild failures and falls through to a plain attribute set.
		}
		return false;
	}

	_applyAdds(adds, report) {
		const legacyMap = { ...this.legacyMissingNodeRetryMap };
		const queue = [];

		const appendNode = (m) => {
			let parent = this.nodes.get(m.parentId);
			if (!parent) {
				if (m.node.type === NodeType.Document) {
					this.newDocumentQueue.push(m);
					return;
				}
				queue.push(m);
				return;
			}
			if (m.node.isShadow) {
				if (!parent.shadowRoot) parent.shadowRoot = this._newShadowRoot(parent);
				parent = parent.shadowRoot;
			}
			const previous = m.previousId ? this.nodes.get(m.previousId) : null;
			const next = m.nextId ? this.nodes.get(m.nextId) : null;
			if (m.nextId != null && m.nextId !== -1 && !next) {
				queue.push(m);
				return;
			}
			if (m.node.rootId && !this.nodes.has(m.node.rootId)) {
				report.droppedAdds.push(m.node.id);
				return;
			}
			const targetDoc = m.node.rootId
				? this.nodes.get(m.node.rootId)
				: this.rootDoc;
			if ((parent.tagName || "").toLowerCase() === "iframe" && parent.meta) {
				this._attachDocumentToIframe(m, parent);
				return;
			}

			const target = this._buildWithSN(m.node, {
				doc: targetDoc,
				skipChild: true,
				realm: "mutation",
			});
			if (!target) {
				// A node type the serialization format doesn't define builds nothing. Returning quietly
				// would leave the add missing from the mirror AND missing from the report — and the report
				// is the whole of what the rescue gate sees, so the slice would be certified continuous
				// while carrying content that was never applied.
				report.droppedAdds.push(m.node.id);
				return;
			}

			if (m.previousId === -1 || m.nextId === -1) {
				legacyMap[m.node.id] = { node: target, mutation: m };
				return;
			}

			const parentSn = parent.meta;
			if (
				parentSn &&
				parentSn.type === NodeType.Element &&
				m.node.type === NodeType.Text
			) {
				if (parentSn.tagName === "textarea") {
					for (const c of parent.childNodes.slice()) {
						if (c.type === NodeType.Text) this._detach(c);
					}
				} else if (
					parentSn.tagName === "style" &&
					parent.childNodes.length === 1
				) {
					// The stream's placeholder text for a materialized stylesheet inherits the sheet: the
					// unserialized cssText child's text moves onto the mirrored node.
					const only = parent.childNodes[0];
					if (only.type === NodeType.Text && only.meta == null) {
						target.textContent = only.textContent;
						this._detach(only);
					}
				}
			} else if (parentSn && parentSn.type === NodeType.Document) {
				// Checkout-style re-adds: an incoming doctype replaces a leading doctype, an incoming <html>
				// replaces the documentElement — detached from the tree but still id-addressable.
				if (
					m.node.type === NodeType.DocumentType &&
					parent.childNodes[0] &&
					parent.childNodes[0].type === NodeType.DocumentType
				) {
					this._detach(parent.childNodes[0]);
				}
				if ((target.tagName || "").toLowerCase() === "html") {
					const docEl = parent.childNodes.find(
						(c) =>
							c.type === NodeType.Element &&
							(c.tagName || "").toLowerCase() === "html",
					);
					if (docEl) this._detach(docEl);
				}
			}

			// rrweb inserts before previous.nextSibling without checking it belongs to the declared parent;
			// when previous sits in another subtree the insert throws and the mutation aborts.
			const prevNext = previous ? this._nextSibling(previous) : null;
			if (previous && prevNext) {
				this._insertBefore(parent, target, prevNext);
			} else if (next?.parentNode) {
				if (this._contains(parent, next))
					this._insertBefore(parent, target, next);
				else this._append(parent, target);
			} else {
				this._append(parent, target);
			}

			if ((target.tagName || "").toLowerCase() === "iframe" && target.meta) {
				this._drainNewDocumentQueue(target);
			}
			if (m.previousId || m.nextId) {
				this._legacyResolveMissingNode(legacyMap, parent, target, m);
			}
		};

		for (const m of adds) appendNode(m);

		// rrweb loops the unresolved queue under a 500ms wall clock; the mirror loops to a fixed point.
		while (queue.length) {
			const before = queue.length;
			const trees = queueToResolveTrees(queue.slice());
			queue.length = 0;
			for (const tree of trees) {
				const parent = this.nodes.get(tree.value.parentId);
				if (!parent) {
					iterateResolveTree(tree, (m) => report.droppedAdds.push(m.node.id));
					continue;
				}
				iterateResolveTree(tree, (m) => appendNode(m));
			}
			if (queue.length >= before) {
				for (const m of queue) report.droppedAdds.push(m.node.id);
				break;
			}
		}

		this.legacyMissingNodeRetryMap = legacyMap;
	}

	// A serialized-document add under a serialized iframe becomes its content document: the existing content
	// document is cleared (doc.open) and re-registered under the incoming node's id, and the subtree builds
	// into it — children of a previously attached document stay id-addressable, exactly as the replayer's
	// id map does.
	_attachDocumentToIframe(m, iframeEl) {
		const iframes = [];
		this._buildWithSN(m.node, {
			doc: iframeEl.contentDocument,
			skipChild: false,
			realm: "mutation",
			iframes,
		});
		for (const el of iframes) this._drainNewDocumentQueue(el);
	}

	_drainNewDocumentQueue(iframeEl) {
		const queued = this.newDocumentQueue.find(
			(m) => m.parentId === iframeEl.id,
		);
		if (!queued) return;
		this.newDocumentQueue = this.newDocumentQueue.filter((m) => m !== queued);
		this._attachDocumentToIframe(queued, iframeEl);
	}

	_legacyResolveMissingNode(map, parent, target, targetMutation) {
		const { previousId, nextId } = targetMutation;
		const previousInMap = previousId && map[previousId];
		const nextInMap = nextId && map[nextId];
		if (previousInMap) {
			const { node, mutation } = previousInMap;
			this._insertBefore(parent, node, target);
			delete map[mutation.node.id];
			delete this.legacyMissingNodeRetryMap[mutation.node.id];
			if (mutation.previousId || mutation.nextId) {
				this._legacyResolveMissingNode(map, parent, node, mutation);
			}
		}
		if (nextInMap) {
			const { node, mutation } = nextInMap;
			this._insertBefore(parent, node, this._nextSibling(target));
			delete map[mutation.node.id];
			delete this.legacyMissingNodeRetryMap[mutation.node.id];
			if (mutation.previousId || mutation.nextId) {
				this._legacyResolveMissingNode(map, parent, node, mutation);
			}
		}
	}
}
