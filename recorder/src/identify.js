/**
 * The operator's own user id, attached to the recording. The cookie visitor stays the recording's key; an Identify event carries the user id in band, in whatever slice is current, and the join from a user to their recordings runs through the visitor id downstream. Nothing is persisted: every page context of the visitor already shares the cookie, so one Identify anywhere in a visitor's recordings names them all.
 *
 * The id never enters a store key, so no charset constrains it. Its length is capped: an email address is the longest shape a real user id takes (254 characters at most, RFC 5321's path limit less its brackets), and anything past that is not an id at all — a serialized object, a session token — that the operator did not mean to record.
 */

export const MAX_USER_ID_LENGTH = 256;

/**
 * Throws on anything that is not a usable user id. A number is refused rather than stringified: a 64-bit database id past 2^53 has already lost digits by the time it is a JS number, so only the operator's own string form of it is the id. The message states the value's type and length, never the value: a start() that throws here reports its error on the telemetry channel, which nothing masks.
 */
export function assertUserId(id) {
	if (typeof id !== "string") {
		throw new TypeError(
			`locus-recorder: a user id must be a string; got ${id === null ? "null" : typeof id} ` +
				"(a numeric id goes as the string your backend holds it as — a JS number past 2^53 " +
				"has already lost digits)",
		);
	}
	if (id.length === 0 || id.length > MAX_USER_ID_LENGTH) {
		throw new RangeError(
			`locus-recorder: a user id must be 1–${MAX_USER_ID_LENGTH} characters; got ${id.length}`,
		);
	}
}
