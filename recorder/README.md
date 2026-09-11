# Locus recorder

rrweb capture in the visitor's browser → stamped events → IndexedDB buffer → periodic drain to a deployment-injected sink, as self-describing gzipped chunks. The events are rrweb's; replay is rrweb-player.

## Install

One tag per site, carrying only a snippet id:

```html
<script async src="https://YOUR-STORE/locus-recorder.min.js?id=YOUR-SNIPPET-ID"></script>
```

The store — a Cloudflare Worker fronting an R2 bucket, `store/` in this repo — serves the bundle and receives its uploads: its deploy builds this bundle and prints the worker URL the tag's origin is (`store/README.md`).

It must be its own classic `<script src>` tag: the bundle reads its id and origin off `document.currentScript` and wires the chunk sink and the telemetry channel back to that origin. Module-wrapped, inlined, or eval'd, it finds neither and stops with an error.

On a site with a Content-Security-Policy, allowlist the store origin in `connect-src` as well as `script-src`: the bundle loads under the latter, but every upload and ping needs `connect-src` (or its `default-src` fallback) to allow the store origin — and each blocked request also files a violation report with the site's own `report-uri` collector.

## Configure

`src/global.js` is the one configuration point — the facade the tag loads. The deployment's config is written into it as code: masking and the drain cadence as `start()` options, and a partial rollout as a gate you write ahead of the `start()` call. The untouched facade mints the visitor identity, wires the sink, and records verbatim except credentials. Its docstring carries the how, `start()`'s jsdoc in `src/index.js` is the option surface, and `src/sink.js` carries the sink contract if you are replacing the store.

```sh
bun install
bun run build   # → dist/locus-recorder.min.js — the bundle the store's deploy builds and serves
bun test --parallel   # the suite; the flag fans test files out across cores
```

`bun test` includes the cross-package contract suites, which shell into the evidence package — so they additionally need `uv`, a synced workspace, and Playwright's Chromium (`analysis/README.md` § Install). A failure there naming a missing command or package is workspace setup, not a broken recorder.

## What this adds over rrweb

- **Slices.** Snapshots are periodic, and each one anchors a slice every event is stamped with: a slice is stored self-covering, the store needs no index to serve it, and a lost chunk costs its slice rather than the rest of the session. Stamps come from each page's own recorder, so a visitor's concurrent tabs interleave without corrupting each other's slices, and a pointer-move batch that flushes after a snapshot is re-homed onto the slice its motion happened in.
- **Context.** rrweb carries no identity, cross-chunk ordering, device, or SPA routes; the recorder adds them. Identity is a first-party cookie validated against the store's key charset, and every start-up ping states how the id was come by — found in the jar, written and read back, or refused by the browser — so a visitor the browser won't persist is never mistaken for a stream of first-timers.
- **Arrival and departure.** rrweb has no visibility event and its load events carry no data. The recorder emits its own: a page load with url, title, and referrer on the first snapshot and on every SPA route change, and every visible/hidden transition. There is no unload event — it does not fire in most mobile teardowns — so departure is read downstream from hidden-then-silence, and the hidden marker rides a fire-and-forget shot, so a page that bounces before rrweb has a DOM to capture still testifies where it came from.
- **Route-aware masking.** rrweb fixes its options when recording starts. Rules resolve per URL, and a route whose resolved rules differ restarts capture on a fresh slice, so a checkout never records under the catalog's posture; a route with the same rules gets its page-load marker in place, no snapshot spent.
- **Delivery.** Events commit to IndexedDB and are retried until they land — from a later page if need be, so bounces, offline stretches, and backgrounded tabs cost latency, not recordings. Each slice's visitor id and device facts are stored beside its records at capture, so a backlog shipped days later by another tab under another cookie still states who was recorded. Where IndexedDB is missing or breaks, capture continues in a memory buffer that ships what it can before the page ends. Concurrent tabs share the one buffer safely — keys are writer-minted and collision-free, a sibling tab stalled mid-transaction cannot starve this one's writes, and a page being frozen or hidden away closes its buffer before the browser stops its clock, so a suspended tab never holds the store against the visitor's other tabs — the backlog is a bounded ring that evicts oldest-first at its ceiling, and a batch the store keeps refusing is dropped rather than left to wedge the queue. The store's own refusal is told from a middlebox answering in its place by the stamp the store puts on every response, so a captive portal's 404 retries like an outage and only the store's no counts against a batch.
- **Named loss.** An evicted, poisoned, or malformed record is named, best-effort — slice and counter range — inside the chunks that do land (a bounded queue, newest kept), and faults ride the telemetry channel, so a device that delivered nothing still says why and quiet traffic is distinguishable from broken capture. A page-load snapshot too large to store is the one loss nothing downstream can rescue, so the recorder stops.
- **Restraint.** Local environments record nothing; bots are screened by BotD, each exclusion attested with the detectors that fired, and a verdict resting on one shape heuristic that real browsers trip records instead, that detector stamped on the start-up ping; a prerendered page records nothing until a human activates it; and the recorder accounts for its own footprint — main-thread time, heap high-water, on-device backlog, bytes spent versus delivered — and past a few internal errors stops rather than harm the page.

## Telemetry

A second channel alongside the chunks, to the same store origin: fire-and-forget pings of operational facts — counts, timings, arrivals, fault reasons with their error text — never the recording itself. Where a ping names a page it names it by address: origin, path, and fragment, with the query string dropped — no masking rule reaches this channel, and a query string is where emails, tokens, and click ids ride (`src/address.js`). It rides independently of the uploads, so it distinguishes a slice that was born and never arrived from one that never started. The transport is assessed per environment — a probe shot and its observed fate pick between sendBeacon and keepalive fetch (`src/sink.js`) — and the verdict governs every fire-and-forget shot, the chunk channel's last-moment sends included. `store/src/worker.js` receives it, `store/src/telemetry.js` gates every datapoint; `store/src/telemetry-schema.json` declares each one's layout. Drop the `telemetry` option in `src/global.js` and the channel does not exist.

## Privacy

Credential content is always masked: passwords, one-time codes, and payment fields, recognized per element from what it is, what it has been, and what its own markup calls it. `src/masking.js` is the one home of the recognizer — its arms, its token sets, and its named residual limit. The mask sits below the rules — a rule may add masking, never take credential masking away. It reads `input` — hidden fields included, where a server can render a credential — and `textarea`, so a value picked from a `<select>` (an expiry dropdown) is a rule's to mask — `maskAllInputs`, or `select: true` in `maskInputOptions`; `select: false` beside `maskAllInputs` keeps selects out, since rrweb's select masking also drops the selection replay shows. Everything else records verbatim.

To mask more, pass `rrwebRules`: an ordered list of `{ pattern, options }`, where `options` is any subset of rrweb's record options — the `recordOptions` type in the installed `node_modules/rrweb/dist/rrweb.d.ts`, the full list at exactly the pinned version (`maskAllInputs`, `maskTextSelector`, `blockClass`/`blockSelector`, ...) and the rules matching a URL layer in order, each merging over the last. `src/masking.js` declares what a pattern is and what it matches — and exports its all-inputs routing gate for a rule whose `maskInputFn` judges per element, since rrweb asks a fn only about the types its gate lists; `start()`'s jsdoc carries the option's route-change semantics.

Two blind spots the input and text knobs cannot reach, both answerable only by blocking the element or accepting that whatever the visitor's screen showed is what any analysis sees. Content drawn into the page as media — a webcam frame written into an `img` as a data URI, say — matches no input or text rule. And text masking reaches text nodes, so anything that renders from somewhere else does not: an attribute, an inlined stylesheet.

Masking happens at record time and is permanent both ways: masked content is gone for good, and content already recorded cannot be masked after the fact.

Visitor identity is one first-party cookie, `locusVisitorId` (`src/visitor.js` declares its scope and lifetime). Being a cookie, the browser attaches it to every request to your own origin — so a cache layer that treats any cookie as personalization (Varnish's builtin VCL passes every cookied request to origin) should strip or ignore it in its cache logic.
