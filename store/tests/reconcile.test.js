/**
 * The reconcile/drift layer whole: provision converges a live account to the declarations and
 * doctor asserts the same desired state, both read off deploy.config.toml + wrangler.toml — the same
 * files the scripts themselves import, so what these tests expect and what the scripts want can
 * never be two copies. Each script runs for real in a subprocess (both execute at import and leave
 * via process.exit) against the fake account in fake-cf.js, and the verdict is the script's own
 * exit code, output, and the account state it leaves behind.
 */
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import deployment from "../deploy.config.toml";
import { RULE_ID } from "../scripts/lifecycle.js";
import { NOT_FOUND } from "../src/worker.js";
import wrangler from "../wrangler.toml";

const BUCKET = wrangler.r2_buckets[0].bucket_name;
const DAYS = deployment.retention_days;

const ruleAt = (days, id = RULE_ID) => ({
	id,
	enabled: true,
	conditions: {},
	deleteObjectsTransition: { condition: { type: "Age", maxAge: days * 86400 } },
});

// The worker's bindings exactly as wrangler.toml declares them — the shape doctor calls healthy.
const declaredBindings = () => [
	...wrangler.r2_buckets.map((b) => ({
		name: b.binding,
		type: "r2_bucket",
		bucket_name: b.bucket_name,
	})),
	...wrangler.analytics_engine_datasets.map((d) => ({
		name: d.binding,
		type: "analytics_engine",
		dataset: d.dataset,
	})),
];

const converged = (over = {}) => ({
	buckets: [BUCKET],
	lifecycle: { [BUCKET]: [ruleAt(DAYS)] },
	managed: { [BUCKET]: { enabled: false, domain: `pub-${BUCKET}.r2.dev` } },
	custom: { [BUCKET]: [] },
	bindings: declaredBindings(),
	subdomain: "fake-sub",
	origin: { "GET /": { status: 404, text: NOT_FOUND } },
	...over,
});

const ORIGIN = `https://${wrangler.name}.fake-sub.workers.dev`;

function run(script, state) {
	const file = join(mkdtempSync(join(tmpdir(), "fake-cf-")), "state.json");
	writeFileSync(file, JSON.stringify(state));
	const proc = Bun.spawnSync(
		[process.execPath, join(import.meta.dir, "fake-cf.js"), script],
		{ env: { ...process.env, FAKE_CF_STATE: file } },
	);
	expect(proc.exitCode).not.toBe(86);
	return {
		code: proc.exitCode,
		out: proc.stdout.toString(),
		err: proc.stderr.toString(),
		state: JSON.parse(readFileSync(file, "utf8")),
	};
}

describe("doctor's verdicts and its exit contract", () => {
	test("a deployment matching the declarations reads clean and exits 0", () => {
		// The disabled custom domain is attached but serves nothing — not an exposure.
		const r = run(
			"doctor",
			converged({
				custom: { [BUCKET]: [{ domain: "old.example.com", enabled: false }] },
			}),
		);
		expect(r.code).toBe(0);
		expect(r.out).toContain("live deployment matches the declaration.");
		expect(r.out).toContain(`OK    lifecycle       ${RULE_ID} ${DAYS}d`);
		expect(r.out).toContain("OK    public access   disabled");
		expect(r.out).toContain(`OK    origin          ${ORIGIN} answers`);
	});

	test("an origin that answers as something other than the worker reads DRIFT, URL named", () => {
		// The control plane can be green while the origin serves nothing: a disabled workers.dev
		// subdomain fails no binding check. The worker owns no root route, so anything but its own
		// not-found answer there means the tag's URL is not reaching this worker.
		const r = run(
			"doctor",
			converged({
				origin: { "GET /": { status: 530, text: "error code: 1042" } },
			}),
		);
		expect(r.code).toBe(1);
		expect(r.out).toContain(`DRIFT origin`);
		expect(r.out).toContain(`${ORIGIN} answered 530`);
		expect(r.out).toContain("workers.dev subdomain is disabled");
	});

	test("a lifecycle rule with the right id but the wrong horizon reads DRIFT", () => {
		const r = run(
			"doctor",
			converged({ lifecycle: { [BUCKET]: [ruleAt(DAYS + 7)] } }),
		);
		expect(r.code).toBe(1);
		expect(r.out).toContain(
			`DRIFT lifecycle       ${DAYS + 7}d, declared ${DAYS}d`,
		);
	});

	test("an absent rule reads DRIFT naming the declared horizon", () => {
		const r = run("doctor", converged({ lifecycle: { [BUCKET]: [] } }));
		expect(r.code).toBe(1);
		expect(r.out).toContain(`${RULE_ID} absent (declared ${DAYS}d)`);
	});

	test("a foreign rule at the declared horizon is not the deployment's rule", () => {
		// The rule is found by its id, never by its horizon: someone else's rule that happens to
		// expire at the declared age must not read as ours being present.
		const r = run(
			"doctor",
			converged({ lifecycle: { [BUCKET]: [ruleAt(DAYS, "someone-elses")] } }),
		);
		expect(r.code).toBe(1);
		expect(r.out).toContain(`${RULE_ID} absent`);
	});

	test("an exposed bucket reads DRIFT with the command that closes each exposure", () => {
		const r = run(
			"doctor",
			converged({
				managed: {
					[BUCKET]: { enabled: true, domain: `pub-${BUCKET}.r2.dev` },
				},
				custom: { [BUCKET]: [{ domain: "chunks.example.com", enabled: true }] },
			}),
		);
		expect(r.code).toBe(1);
		expect(r.out).toContain(`PUBLIC via pub-${BUCKET}.r2.dev`);
		expect(r.out).toContain(
			`bunx wrangler r2 bucket dev-url disable ${BUCKET}`,
		);
		expect(r.out).toContain("PUBLIC via chunks.example.com");
		expect(r.out).toContain(
			`bunx wrangler r2 bucket domain remove ${BUCKET} --domain chunks.example.com`,
		);
	});

	test("a drifted worker binding reads DRIFT, and an undeployed worker says so", () => {
		const bent = declaredBindings().map((b) =>
			b.type === "r2_bucket"
				? { ...b, bucket_name: "someone-elses-bucket" }
				: b,
		);
		const drifted = run("doctor", converged({ bindings: bent }));
		expect(drifted.code).toBe(1);
		expect(drifted.out).toContain(`someone-elses-bucket, declared ${BUCKET}`);

		const undeployed = run("doctor", converged({ bindings: null }));
		expect(undeployed.code).toBe(1);
		expect(undeployed.out).toContain("not deployed or unreadable");
	});

	test("a binding the live worker does not carry at all reads DRIFT, named", () => {
		// A binding can go missing rather than drift: a deploy from a wrangler.toml with the
		// dataset commented out leaves a worker that ingests and writes no datapoint. Absent is a
		// verdict of its own, and a worker carrying a subset of the declared bindings must never
		// read as matching the declaration.
		for (const declared of declaredBindings()) {
			const r = run(
				"doctor",
				converged({
					bindings: declaredBindings().filter((b) => b.name !== declared.name),
				}),
			);
			expect(r.code).toBe(1);
			expect(r.out).toContain(`DRIFT binding ${declared.name}`);
			expect(r.out).toContain("missing");
		}
	});

	test("a non-envelope answer is exit 2, never a safe negative", () => {
		// Read as an answer, an HTML 502 would say the bucket is missing, the rule absent, public
		// access disabled — every invariant at its safe case. Doctor must refuse the read outright:
		// exit 2 (unreadable), never 1 (drift) and never 0.
		const r = run(
			"doctor",
			converged({
				raw: {
					"GET /r2/buckets": {
						status: 502,
						text: "<!doctype html><title>502 Bad Gateway</title>",
					},
				},
			}),
		);
		expect(r.code).toBe(2);
		expect(r.err).toContain("not a Cloudflare API envelope");
		expect(r.out).not.toContain("MISSING");
	});

	test("a successful envelope whose result lacks the asked-for field is exit 2 too", () => {
		const r = run(
			"doctor",
			converged({
				raw: {
					"GET /r2/buckets": {
						status: 200,
						text: '{"success":true,"errors":[],"result":{}}',
					},
				},
			}),
		);
		expect(r.code).toBe(2);
		expect(r.err).toContain("carried no bucket list");
	});
});

describe("provision converges to the state doctor asserts", () => {
	test("an empty account converges to a deployment doctor calls clean", () => {
		const empty = converged({ buckets: [], lifecycle: {} });
		const before = run("doctor", empty);
		expect(before.code).toBe(1);
		expect(before.out).toContain("DRIFT bucket          MISSING");

		const p = run("provision", empty);
		expect(p.code).toBe(0);
		expect(p.out).toContain("bucket          created");
		expect(p.out).toContain(`${RULE_ID} ${DAYS}d (was absent)`);

		const after = run("doctor", p.state);
		expect(after.code).toBe(0);
		expect(after.out).toContain("live deployment matches the declaration.");
	});

	test("a converged account is left untouched, and says each setting is current", () => {
		const state = converged();
		const p = run("provision", state);
		expect(p.code).toBe(0);
		expect(p.out).toContain("bucket          exists");
		expect(p.out).toContain(`${RULE_ID} ${DAYS}d (current)`);
		expect(p.state).toEqual(state);
	});

	test("a drifted horizon reconciles to the declaration, foreign rules intact", () => {
		const foreign = ruleAt(365, "someone-elses");
		const p = run(
			"provision",
			converged({ lifecycle: { [BUCKET]: [foreign, ruleAt(DAYS + 70)] } }),
		);
		expect(p.code).toBe(0);
		expect(p.out).toContain(`${RULE_ID} ${DAYS}d (was ${DAYS + 70}d)`);
		expect(p.state.lifecycle[BUCKET]).toContainEqual(foreign);
		expect(p.state.lifecycle[BUCKET]).toContainEqual(ruleAt(DAYS));

		expect(run("doctor", p.state).code).toBe(0);
	});

	test("an exposed bucket is refused, never converged", () => {
		// Detaching a domain someone attached on purpose is a decision, not a reconciliation:
		// provision names every exposure with its closing command and exits — the exposure stands.
		const exposed = converged({
			managed: { [BUCKET]: { enabled: true, domain: `pub-${BUCKET}.r2.dev` } },
		});
		const p = run("provision", exposed);
		expect(p.code).toBe(1);
		expect(p.err).toContain("publicly readable");
		expect(p.err).toContain(
			`bunx wrangler r2 bucket dev-url disable ${BUCKET}`,
		);
		expect(p.state.managed[BUCKET].enabled).toBe(true);
	});

	test("a non-envelope answer stops provision before it converges anything", () => {
		const empty = converged({
			buckets: [],
			lifecycle: {},
			raw: {
				"GET /r2/buckets": {
					status: 502,
					text: "<!doctype html><title>502 Bad Gateway</title>",
				},
			},
		});
		const p = run("provision", empty);
		expect(p.code).toBe(1);
		expect(p.err).toContain("not a Cloudflare API envelope");
		expect(p.state.buckets).toEqual([]);
	});
});
