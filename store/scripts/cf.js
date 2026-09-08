/**
 * Authenticated Cloudflare API access for the store's provision/doctor scripts. The account id and token are the deployment's standard CLOUDFLARE_* credentials, auto-loaded by bun from store/.env — the same single credential the worker deploy and `locus` use.
 */
const ACCOUNT = process.env.CLOUDFLARE_ACCOUNT_ID;
const TOKEN = process.env.CLOUDFLARE_API_TOKEN;
if (!ACCOUNT || !TOKEN) {
	console.error(
		"CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set (store/.env)",
	);
	process.exit(1);
}

export const account = ACCOUNT;
const API = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT}`;

/**
 * A Cloudflare API response, and whether it is one at all. Every endpoint answers in the same envelope and always carries `success`, so a body without it is not a negative answer — it is no answer: an HTML error page from a proxy, a captive portal, a truncated response. Read as a negative it becomes "the rule is absent", "the bucket is missing", "public access is disabled" — every invariant reading as its safe case. So a response counts as an answer only when the envelope says it succeeded; everything else is a failure, with the body in hand to say why.
 */
export function readEnvelope(text) {
	let body;
	try {
		body = text ? JSON.parse(text) : {};
	} catch {
		body = { raw: text };
	}
	// `body?.` because valid JSON need not be an object: a bare `null` body is no answer either.
	return { body, answered: body?.success === true };
}

export async function cf(path, init = {}) {
	const res = await fetch(`${API}${path}`, {
		...init,
		headers: {
			Authorization: `Bearer ${TOKEN}`,
			"Content-Type": "application/json",
			...init.headers,
		},
	});
	const { body, answered } = readEnvelope(await res.text());
	return { ok: res.ok && answered, status: res.status, body };
}

export function errMsg(r) {
	const errors = (r.body?.errors || [])
		.map((e) => `${e.message}${e.code ? ` [${e.code}]` : ""}`)
		.join("; ");
	if (errors) return errors;
	if (typeof r.body?.raw === "string") {
		return `the response was not a Cloudflare API envelope: ${r.body.raw.slice(0, 200)}`;
	}
	if (r.body?.success !== true) return "the response carried no success";
	return "";
}
