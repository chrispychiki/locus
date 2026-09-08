/**
 * Reconcile the live R2 deployment to deploy.config.toml — idempotent and loud. It runs ahead of `wrangler deploy`, so the bucket exists before the worker binds it, and stands alone for a config change: edit deploy.config.toml, `bun run provision`, no worker redeploy. Every run reports each setting as already-current or names what it changed.
 *
 * The bucket name is read from wrangler.toml, never restated; the retention horizon comes from deploy.config.toml. The bucket's privacy is neither: it is an invariant, asserted here rather than declared anywhere.
 */

import deployment from "../deploy.config.toml";
import wrangler from "../wrangler.toml";
import { account, cf, errMsg } from "./cf.js";
import { expiryRule, RULE_ID, ruleDays } from "./lifecycle.js";
import { publicExposures } from "./public-access.js";

const bucket = wrangler.r2_buckets[0].bucket_name;

function die(label, r) {
	const msg = errMsg(r);
	console.error(
		`provision: ${label} failed (${r.status})${msg ? `: ${msg}` : ""}`,
	);
	process.exit(1);
}

const line = (k, v) => console.log(`  ${k.padEnd(16)}${v}`);
console.log(`current timestamp: ${new Date().toISOString().slice(0, 19)}Z`);
console.log(`provision ${bucket} (account ${account.slice(0, 6)}…)`);

// A setting is reconciled only against a reading the API actually gave. A field that failed to
// appear is not a value: read as one it means "absent", and provision would converge a bucket it
// never saw and assert an access policy it never checked.
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
if (buckets.some((b) => b.name === bucket)) {
	line("bucket", "exists");
} else {
	const made = await cf("/r2/buckets", {
		method: "POST",
		body: JSON.stringify({ name: bucket }),
	});
	if (!made.ok) die(`create bucket ${bucket}`, made);
	line("bucket", "created");
}

const want = deployment.retention_days;
const cur = await cf(`/r2/buckets/${bucket}/lifecycle`);
const rules = must("lifecycle", cur, (r) =>
	Array.isArray(r?.rules) ? r.rules : undefined,
);
const curDays = ruleDays(rules);
if (curDays === want) {
	line("lifecycle", `${RULE_ID} ${want}d (current)`);
} else {
	const others = rules.filter((r) => r.id !== RULE_ID);
	const put = await cf(`/r2/buckets/${bucket}/lifecycle`, {
		method: "PUT",
		body: JSON.stringify({ rules: [...others, expiryRule(want)] }),
	});
	if (!put.ok) die("set lifecycle", put);
	const after = await cf(`/r2/buckets/${bucket}/lifecycle`);
	const live = must("lifecycle", after, (r) =>
		Array.isArray(r?.rules) ? r.rules : undefined,
	);
	const setDays = ruleDays(live);
	if (setDays !== want)
		die(
			`confirm lifecycle (wanted ${want}d, live ${setDays == null ? "absent" : `${setDays}d`})`,
			after,
		);
	line(
		"lifecycle",
		`${RULE_ID} ${want}d (was ${curDays == null ? "absent" : `${curDays}d`})`,
	);
}

// Public access asserted off (public-access.js). An invariant, not a setting: provision names
// the exposure and refuses rather than converging it, because detaching a domain someone attached
// on purpose is a decision, not a reconciliation.
const exposed = await publicExposures(bucket, must);
if (exposed.length > 0) {
	console.error(
		`provision: the bucket is publicly readable — it holds visitor recordings and must be private. Close every exposure, then re-run:`,
	);
	for (const { domain, remedy } of exposed)
		console.error(`  ${domain}\n    ${remedy}`);
	process.exit(1);
}
line("public access", "disabled");
