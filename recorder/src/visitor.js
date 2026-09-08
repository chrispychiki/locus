/**
 * Persistent visitor identity: a first-party cookie, one year.
 *
 * The cookie is untrusted, externally-writable input — any script on the page, a sibling subdomain, or an attacker can set it — and it flows straight into the chunk key, where a value outside the store's accepted charset bounces every upload and silently loses the visitor. So a cookie that does not match VALID_VISITOR_ID (the store's visitor path slot, store/src/keys.js) is treated as absent and replaced with a fresh id.
 */

const COOKIE = "locusVisitorId";
const VALID_VISITOR_ID = /^[A-Za-z0-9_-]{8,64}$/;

/** The visitor a fault belongs to when the identity itself is what failed: reading the cookie throws on an opaque origin, and a sandboxed iframe has no cookie jar at all. The sentinel satisfies the store's key charset — a report of the failure that fails the gate is the failure going unreported — and is not an id any visitor can hold. */
export const UNIDENTIFIED_VISITOR = "unidentified";

/** Whether an id can survive the trip: the store keys chunks by it, so an id its gate refuses loses every upload. Exported so the store's gate is pinned against this rather than the two agreeing by hand. */
export function isValidVisitorId(id) {
	return typeof id === "string" && VALID_VISITOR_ID.test(id);
}

function getCookie(name) {
	for (const part of document.cookie.split(";")) {
		const [key, ...rest] = part.trim().split("=");
		if (key === name) return rest.join("=");
	}
	return null;
}

function setCookie(name, value, days) {
	const expires = new Date(Date.now() + days * 86_400_000).toUTCString();
	document.cookie = `${name}=${value};expires=${expires};path=/;SameSite=Lax`;
}

function uuid() {
	// randomUUID exists only in secure contexts, so a plain-HTTP deployment mints every id
	// through the fallback; getRandomValues has no such gate, and the id is the deployment's
	// primary join key, so its entropy is never Math.random's.
	if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
	const bytes = crypto.getRandomValues(new Uint8Array(16));
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join(
		"",
	);
	return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/**
 * The visitor id, and how this page came by it. Assigning document.cookie is a silent no-op wherever the browser declines it, so a refused write reads downstream as a first-time visitor and the same visitor arrives under a new id on every page.
 *
 * "cookie" — a valid id was already in the jar when this page looked. Who wrote it and when is not visible from here; a server or another script on the origin can set the same cookie.
 * "written" — minted here and read back, so the jar took it. A browser may still cap a script-written cookie's lifetime, which no read at write time can see.
 * "unpersisted" — minted here and not read back. This page load joins to nothing, and the next load mints again.
 */
export function getVisitorIdentity() {
	const existing = getCookie(COOKIE);
	if (isValidVisitorId(existing)) return { id: existing, source: "cookie" };
	const id = uuid();
	setCookie(COOKIE, id, 365);
	return { id, source: getCookie(COOKIE) === id ? "written" : "unpersisted" };
}

export function getOrCreateVisitorId() {
	return getVisitorIdentity().id;
}
