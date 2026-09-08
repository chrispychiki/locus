/**
 * The local-environment gate: whether a hostname is the operator's own machine or private network,
 * where recording would capture their own dev traffic rather than their visitors'.
 */

// The gate is total and silent: a page it matches records nothing at all, and says nothing about
// why. So it matches whole hosts and whole addresses, never substrings — `10.example.com` is a
// public site and `localhost.tools.acme.com` is a public host. Loopback and private-range
// literals are matched as addresses; a hostname is matched at a label boundary, as masking.js
// matches its own hosts.
const LOOPBACK_V4 = /^127(\.\d{1,3}){3}$/;
const PRIVATE_V4 =
	/^(10(\.\d{1,3}){3}|192\.168(\.\d{1,3}){2}|172\.(1[6-9]|2\d|3[01])(\.\d{1,3}){2}|169\.254(\.\d{1,3}){2})$/;
// fc00::/7 is the ULA range (RFC 4193), fe80::/10 link-local; a link-local address in a URL
// carries a zone id (fe80::1%25en0), which the suffix-tolerant prefix match absorbs.
const PRIVATE_V6 = /^f([cd][0-9a-f]{2}|e[89ab][0-9a-f])(:|$)/;

export function isLocalEnvironment(hostname) {
	if (!hostname) return false;
	const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
	return (
		host === "localhost" ||
		host.endsWith(".localhost") ||
		host === "::1" ||
		host === "local" ||
		host.endsWith(".local") ||
		LOOPBACK_V4.test(host) ||
		PRIVATE_V4.test(host) ||
		PRIVATE_V6.test(host)
	);
}
