/**
 * Runs a store script (`bun tests/fake-cf.js <provision|doctor>`) against a faked Cloudflare
 * account: a stateful stand-in installed behind global fetch, so scripts/cf.js — envelope reading
 * included — runs for real while no request leaves the process. The account's state loads from the
 * JSON file named by FAKE_CF_STATE, every mutation writes it back synchronously (the scripts leave
 * via process.exit, so nothing may wait on an exit hook), and the caller reads that file back as
 * the account's final state.
 *
 * State shape:
 *   buckets   [name, ...]
 *   lifecycle {bucket: rules[]}
 *   managed   {bucket: {enabled, domain}}   (absent bucket reads disabled)
 *   custom    {bucket: domains[]}
 *   bindings  worker bindings array, or null for a worker that is not deployed
 *   subdomain the account's workers.dev subdomain
 *   origin    {"METHOD /path": {status, text}} — what the worker's own public origin answers
 *   raw       {"METHOD /path": {status, text}} — serves that route a verbatim body instead of the
 *             account, which is how a test hands a script a non-envelope answer
 *
 * Anything the fake cannot serve — an unknown route, a fetch to any other host — exits 86, a code
 * no script under test uses, so a test can never mistake the harness's own failure for a verdict.
 */
import { writeFileSync } from "node:fs";

process.env.CLOUDFLARE_ACCOUNT_ID ??= "fake-cf-account";
process.env.CLOUDFLARE_API_TOKEN ??= "fake-cf-token";

const STATE_FILE = process.env.FAKE_CF_STATE;
const script = process.argv[2];
function harnessFailure(message) {
	console.error(`fake-cf: ${message}`);
	process.exit(86);
}
if (!STATE_FILE) harnessFailure("FAKE_CF_STATE names no state file");
if (!["provision", "doctor"].includes(script))
	harnessFailure(`no script named ${JSON.stringify(script)}`);

const state = await Bun.file(STATE_FILE).json();
const persist = () => writeFileSync(STATE_FILE, JSON.stringify(state));

const PREFIX = `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}`;
const envelope = (result) =>
	new Response(
		JSON.stringify({ success: true, errors: [], messages: [], result }),
	);
const refusal = (status, code, message) =>
	new Response(
		JSON.stringify({
			success: false,
			errors: [{ code, message }],
			result: null,
		}),
		{ status },
	);

globalThis.fetch = async (url, init = {}) => {
	const originHit = String(url).match(/^https:\/\/[^/]+\.workers\.dev(\/.*)?$/);
	if (originHit) {
		const answer =
			state.origin?.[`${init.method ?? "GET"} ${originHit[1] || "/"}`];
		if (!answer) harnessFailure(`the origin has no answer for ${url}`);
		return new Response(answer.text, { status: answer.status ?? 200 });
	}
	if (!String(url).startsWith(PREFIX))
		harnessFailure(`a request left the account: ${url}`);
	const path = String(url).slice(PREFIX.length);
	const method = init.method ?? "GET";
	const route = `${method} ${path}`;

	const override = state.raw?.[route];
	if (override)
		return new Response(override.text, { status: override.status ?? 200 });

	if (route === "GET /r2/buckets")
		return envelope({ buckets: state.buckets.map((name) => ({ name })) });
	if (route === "POST /r2/buckets") {
		state.buckets.push(JSON.parse(init.body).name);
		persist();
		return envelope({});
	}
	const bucketRoute = route.match(/^(GET|PUT) \/r2\/buckets\/([^/]+)\/(.+)$/);
	if (bucketRoute) {
		const [, verb, bucket, rest] = bucketRoute;
		if (verb === "GET" && rest === "lifecycle")
			return envelope({ rules: state.lifecycle[bucket] ?? [] });
		if (verb === "PUT" && rest === "lifecycle") {
			state.lifecycle[bucket] = JSON.parse(init.body).rules;
			persist();
			return envelope({});
		}
		if (verb === "GET" && rest === "domains/managed")
			return envelope(
				state.managed[bucket] ?? {
					enabled: false,
					domain: `pub-${bucket}.r2.dev`,
				},
			);
		if (verb === "GET" && rest === "domains/custom")
			return envelope({ domains: state.custom[bucket] ?? [] });
	}
	if (route.match(/^GET \/workers\/scripts\/[^/]+\/bindings$/))
		return state.bindings === null
			? refusal(404, 10007, "workers.api.error.script_not_found")
			: envelope(state.bindings);
	if (route === "GET /workers/subdomain")
		return envelope({ subdomain: state.subdomain });

	return harnessFailure(`no route for ${route}`);
};

await import(`../scripts/${script}.js`);
