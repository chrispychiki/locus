/**
 * doctor — assert the live deployment matches its declared intent, read-only. Reports each fact as OK or DRIFT and exits non-zero on any drift. It mutates nothing: provision converges drift, doctor names it.
 *
 * Desired state comes from the same sources provision reconciles to — the bucket name and the worker's expected bindings from wrangler.toml, the retention horizon from deploy.config.toml, the privacy invariant from public-access.js — so doctor and provision cannot disagree about what "correct" is.
 */

import deployment from "../deploy.config.toml";
import wrangler from "../wrangler.toml";
import { account, cf, errMsg } from "./cf.js";
import { RULE_ID, ruleDays } from "./lifecycle.js";
import { publicExposures } from "./public-access.js";
import { NOT_FOUND } from "../src/worker.js";

const bucket = wrangler.r2_buckets[0].bucket_name;
const worker = wrangler.name;

let drift = 0;
const ok = (k, v) => console.log(`  OK    ${k.padEnd(16)}${v}`);
const bad = (k, v) => {
	console.log(`  DRIFT ${k.padEnd(16)}${v}`);
	drift++;
};
function die(label, r) {
	const m = errMsg(r);
	console.error(`doctor: ${label} failed (${r.status})${m ? `: ${m}` : ""}`);
	process.exit(2);
}

console.log(`current timestamp: ${new Date().toISOString().slice(0, 19)}Z`);
console.log(
	`doctor ${bucket} / worker ${worker} (account ${account.slice(0, 6)}…)`,
);

// Every reading is a fact the API stated, never a field that failed to appear: a response whose
// shape is not what the endpoint returns leaves the setting unread, and an unread setting is not
// a healthy one. Dying here is what keeps doctor from certifying a bucket private without ever
// having asked.
function must(label, response, read) {
	if (!response.ok) die(label, response);
	const value = read(response.body.result);
	if (value === undefined)
		die(`${label} — the response carried no ${label}`, response);
	return value;
}

const list = await cf("/r2/buckets");
const buckets = must("bucket list", list, (r) =>
	Array.isArray(r?.buckets) ? r.buckets : undefined,
);
const bucketExists = buckets.some((b) => b.name === bucket);
if (bucketExists) ok("bucket", "exists");
else bad("bucket", "MISSING");

if (bucketExists) {
	const lc = await cf(`/r2/buckets/${bucket}/lifecycle`);
	const rules = must("lifecycle", lc, (r) =>
		Array.isArray(r?.rules) ? r.rules : undefined,
	);
	const days = ruleDays(rules);
	if (days === deployment.retention_days)
		ok("lifecycle", `${RULE_ID} ${days}d`);
	else if (days == null)
		bad(
			"lifecycle",
			`${RULE_ID} absent (declared ${deployment.retention_days}d)`,
		);
	else bad("lifecycle", `${days}d, declared ${deployment.retention_days}d`);

	// Public access is an invariant (public-access.js) and the one drift provision will not
	// converge, so each exposure is reported with the command that closes it.
	const exposed = await publicExposures(bucket, must);
	for (const { domain, remedy } of exposed) {
		bad(
			"public access",
			`PUBLIC via ${domain} — every recording in the bucket is exposed; close it with: ${remedy}`,
		);
	}
	if (exposed.length === 0) ok("public access", "disabled");
}

const wb = await cf(`/workers/scripts/${worker}/bindings`);
if (!wb.ok) {
	bad("worker", `not deployed or unreadable (${wb.status})`);
} else {
	const live = must("worker bindings", wb, (r) =>
		Array.isArray(r) ? r : undefined,
	);
	const expected = [
		...(wrangler.r2_buckets || []).map((b) => ({
			name: b.binding,
			type: "r2_bucket",
			target: b.bucket_name,
			key: "bucket_name",
		})),
		...(wrangler.analytics_engine_datasets || []).map((d) => ({
			name: d.binding,
			type: "analytics_engine",
			target: d.dataset,
			key: "dataset",
		})),
	];
	for (const e of expected) {
		const found = live.find((b) => b.name === e.name && b.type === e.type);
		if (found && found[e.key] === e.target)
			ok(`binding ${e.name}`, `${e.type} → ${e.target}`);
		else if (found)
			bad(
				`binding ${e.name}`,
				`${e.type} → ${found[e.key]}, declared ${e.target}`,
			);
		else bad(`binding ${e.name}`, `missing (${e.type} → ${e.target})`);
	}
}

// Everything above reads the control plane, and the control plane can be green while the public
// origin serves nothing — a disabled workers.dev subdomain fails no binding check, yet the origin
// is the one surface visitors' tags pull the bundle from and upload to. So doctor resolves the
// account's subdomain and asks the origin itself. The worker owns no root route, so its own
// not-found answer there is the proof that this worker is what answers at the URL.
const sd = await cf("/workers/subdomain");
const subdomain = must("workers.dev subdomain", sd, (r) => r?.subdomain);
const origin = `https://${worker}.${subdomain}.workers.dev`;
try {
	const root = await fetch(origin, { redirect: "manual" });
	const body = await root.text();
	if (root.status === 404 && body === NOT_FOUND)
		ok("origin", `${origin} answers`);
	else
		bad(
			"origin",
			`${origin} answered ${root.status} ${JSON.stringify(body.slice(0, 80))} — not this worker's own answer`,
		);
} catch (error) {
	bad("origin", `${origin} unreachable (${error?.cause?.code ?? error})`);
}

console.log();
if (drift) {
	console.log(
		`${drift} drifted setting${drift > 1 ? "s" : ""} — a missing bucket or lifecycle reconciles with \`bun run provision\`, a binding gap needs a redeploy, public access closes with the command named above, and a dead origin usually means the account's workers.dev subdomain is disabled.`,
	);
	process.exit(1);
}
console.log("live deployment matches the declaration.");
