# locus-store — the R2 chunk store

The Locus recorder's recommended deployment: one Cloudflare Worker in front of one R2 bucket. Chunk uploads and recorder telemetry go to the worker; the recorder bundle is served straight from the worker's static-asset binding, which never runs worker code; `locus` reads the bucket directly with its own credentials. The bucket is never publicly readable.

At low-to-moderate volume the whole thing runs inside Cloudflare's free tier. Which limit binds first depends on the traffic's shape — the projection arithmetic is the quantify skill's — and a static-asset serve is a free, unlimited request class (https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/), so bundle serving never meters.

## Deploy

The account needs an R2 subscription first: free to start, but a checkout (https://developers.cloudflare.com/r2/get-started/). Analytics Engine is optional — see **Setup gotchas**. Token scopes are the three the `.env` block names (https://developers.cloudflare.com/fundamentals/api/reference/permissions/).

```sh
# Credentials live in a gitignored .env here; wrangler auto-loads it. Never commit it.
cat > .env <<'EOF'
CLOUDFLARE_API_TOKEN=...   # scopes: Workers Scripts:Edit + Workers R2 Storage:Edit + Account Analytics:Read
CLOUDFLARE_ACCOUNT_ID=...
EOF

bun run deploy   # install, build the bundle, reconcile the bucket, ship the worker
bun run doctor   # read-only: does the live deployment still match the declaration?
```

(`deploy` installs its own deps and the recorder's. The suite does not: `bun install && bun test`.)

`deploy` is idempotent. It builds the recorder bundle into `public/`, runs `bun run provision` — which ensures the bucket exists, sets the chunk-expiry lifecycle, and asserts the bucket is private — and then ships the worker. `provision` stands alone for a bucket-policy change; a capture-config change lives in the bundle, so it needs the full `deploy`. It prints the worker URL, `https://locus-api.<account-subdomain>.workers.dev`, which is both the bundle origin and the upload endpoint. Every deploy reconciles the live bucket back to the declaration and prints what it changed: the repo is the source of truth, and a setting changed in the Cloudflare dashboard silently reverts on the next deploy.

`doctor` asserts the bucket, the lifecycle horizon, the bucket's privacy, the worker's R2 and Analytics Engine bindings, and the worker answering at its own public origin — the URL the tags pull from. It names each fact OK or DRIFT, hands a drifted setting the command that fixes it, and exits non-zero on drift.

## Retention

`deploy.config.toml` declares `retention_days` — the one knob; TOML so the store's scripts and Locus both read it natively. Change it there and run `bun run provision`. R2 expires chunks past the horizon itself: there is no sweeper to run, and the deletions are free (https://developers.cloudflare.com/r2/pricing/).

The bucket is not the only copy. The same number governs the local db and everything derived beside it — `evidence/README.md` states what a load and a doctor drop by it — so lowering the horizon expires recordings in both places, and there is no copy that outlives it.

It is two policies at once. Cost: steady-state storage ≈ daily chunk volume × the horizon, at $0.015/GB-month (https://developers.cloudflare.com/r2/pricing/). Privacy: it is how long recordings of your visitors are kept at all.

## The tag

One tag per site; `recorder/README.md` carries it. Its origin is the worker URL the deploy printed, and its `?id=` is the snippet id (`src/keys.js` gates the shape). The bundle wires both sinks off that origin itself, `{origin}/chunks/…` and `{origin}/telemetry/…`, so the key layout never reaches the page and stays changeable store-side without touching a pasted tag.

Capture configuration — masking, drain cadence, a rollout gate — is `recorder/src/global.js`, the facade this deploy builds into the bundle it serves. Edit it and redeploy; the bundle is served `Cache-Control: no-cache` (`public/_headers`), so every page load revalidates it against the edge and a change binds on the next page load instead of waiting out anyone's cache — an unchanged bundle costs a 304, never a re-download. The served bytes still converge within a few minutes of a deploy: a stale read right after one is the asset deploy propagating across the edge, not an HTTP cache — and not a failed deploy. Masking is record-time only: nothing downstream recovers what wasn't captured, or un-captures what was. A bundle that wires no telemetry sink emits no observability datapoints. Visitor identity is the facade's to mint — the store keys chunks by whatever it supplies (`src/keys.js` gates the shape). To send uploads somewhere other than this worker, replace the chunk sink: import the recorder package into your own bundle — `recorder/README.md` carries that surface.

With the tag on the page, `window.LocusRecorder` means the bundle loaded and ran. It carries `flush()` only while recording — a local hostname, a detected bot, or a rollout gate leaves the marker bare — and awaiting `flush()` delivers everything captured so far, so a capture check need not wait on the drain timer.

## Reading it back

`locus ls` inventories the bucket and diffs it against what is already loaded; `locus load <prefix>` loads the chunks under a key prefix into a local `events.db`, hydrating them, materializing slices, and distilling as it goes. Both self-serve the bucket name, endpoint, and S3 credentials from `store/.env` and `wrangler.toml`, so nothing is exported by hand.

Whether it is recording, whether capture is complete, what it costs, whether it is leaking PII, what it does to visitors' devices — those are judgments across both planes, the objects in R2 and the Analytics Engine datapoints. The quantify skill owns them.

## Key layout and access

`src/keys.js` declares the key an upload path becomes, and the shape gate the path must pass — a chunk body's only gate is the size cap, and why its bytes are deliberately never one is that file's own preamble; `src/telemetry.js` declares the gate for telemetry bodies, and `src/telemetry-schema.json` declares the layout of every observability datapoint — the worker writes rows by it, and `locus ae` reads it back. `locus ls` reads those keys back positionally, so moving the layout is a change on both sides of the store.

The ingest endpoint is open-write, defended by shape not auth — there is no upload secret to lean on. What bounds abuse: every body is capped (`src/keys.js`), junk that passes the shape gate never reaches an analysis — hydration rejects bodies without the recorder's stamps at decode and dedups duplicates away — and everything stored expires at the retention horizon, so a flood's storage cost is bounded in time. What a flood does to the bill is the same plan arithmetic as legitimate traffic — which meter binds first, and where the free tier stops serving instead of charging, is the quantify skill's projection.

**The snippet id is an organizational prefix, not a security boundary.** R2 API tokens scope to a whole bucket — there is no prefix-level grant (https://developers.cloudflare.com/r2/api/tokens/) — so any credential that reads the bucket reads every snippet in it. One bucket is one trust domain: sites that must not share read access go in separate buckets, not just separate snippet ids.

A bucket reaches the internet only through the managed `r2.dev` domain or a custom domain attached to it (https://developers.cloudflare.com/r2/buckets/public-buckets/); `scripts/public-access.js` names both and the command that closes each. Provision refuses to converge an exposed bucket, and doctor reports it as drift. Every object is encrypted at rest, AES-256, with nothing to enable (https://developers.cloudflare.com/r2/reference/data-security/).

## Setup gotchas

**Analytics Engine is optional.** It is free (it shows as Paid, $0 — no Workers Paid plan, no card) and decoupled from capture: the worker writes its observability datapoints when the `CAPTURE` binding is present and no-ops them when it is not. The shipped `wrangler.toml` includes the binding, which needs AE enabled on the account to deploy as shipped. On a fresh account the entitlement is sometimes slow to provision even after it reads as enabled, and the deploy then fails on the binding — comment out `[[analytics_engine_datasets]]` in `wrangler.toml` and ingest, bundle serving, and read-back all still work; the only loss is the rates plane. Uncomment it and redeploy to turn it on later.

**A brand-new account has no `workers.dev` subdomain** until it claims one, which the first deploy does; until then there is no origin to serve the bundle from. That account subdomain is the globally-unique half of the worker URL — the worker name is only account-local.

**The R2 panel can misreport itself.** Right after R2 is enabled it may read "subscribed but disabled" while the API creates buckets fine. The API is the truth.
