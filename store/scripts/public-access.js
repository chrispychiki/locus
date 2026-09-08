/**
 * Every way the bucket is exposed to the internet, and the command that closes each one.
 *
 * Privacy is an invariant rather than a declared setting: the bucket holds visitor recordings, so there is no configuration under which it may be readable. R2 exposes a bucket exactly two ways — the managed r2.dev domain, and any custom domain attached to it — so both are asked; reading only the first would certify as private a bucket the whole internet can GET. Provision and doctor both read the invariant from here, so they cannot disagree about what "exposed" means. Each exposure carries the command that closes it: configuration is changed by command, never in the dashboard, which the next deploy reverts.
 *
 * The caller supplies `must` — the reader that turns an unanswered API response into that script's own fatal exit — because an exposure inferred from a field that never arrived is no check at all.
 */
import { cf } from "./cf.js";

export async function publicExposures(bucket, must) {
	const managed = await cf(`/r2/buckets/${bucket}/domains/managed`);
	const enabled = must("public access", managed, (r) =>
		typeof r?.enabled === "boolean" ? r.enabled : undefined,
	);
	const custom = await cf(`/r2/buckets/${bucket}/domains/custom`);
	const attached = must("custom domains", custom, (r) =>
		Array.isArray(r?.domains) ? r.domains : undefined,
	);

	return [
		...(enabled
			? [
					{
						domain: managed.body.result.domain,
						remedy: `bunx wrangler r2 bucket dev-url disable ${bucket}`,
					},
				]
			: []),
		...attached
			.filter((d) => d.enabled !== false)
			.map((d) => ({
				domain: d.domain,
				remedy: `bunx wrangler r2 bucket domain remove ${bucket} --domain ${d.domain}`,
			})),
	];
}
