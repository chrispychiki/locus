# Locus evidence

The mechanical layer: the store's chunks made trustworthy, queryable, and replayable. Load and hydration land raw rrweb in one local db, distillation derives compact text projections and flat columns onto the same rows, and the replay component and screenshot renderer turn any slice back into what the visitor's screen showed. Everything here is faithful derivation — meaning is never assigned; that is the analysis layer's.

## Storage

One anchor, the deployment root — and one split at it. The deployment-wide declarations live in its `config/` directory (package-owned ones stay with their packages, e.g. store/wrangler.toml); everything the deployment accretes lives under its `data/` directory: the single `events.db`, the operator's own `context/{snippet}.md` beside it, and the derived families as its plain siblings. Every output's path is derived from what it is — occasion and subject for an analysis, the slice set for a replay payload — never chosen by the agent; a slice is always named by its recorder slice id (`<padded-open-ms>-<disambiguator>`, stamped at capture), and the db's integer key is a private rowid that never surfaces.

```
config/                                              the deployment-wide declarations; ship with defaults
    spend.toml                                       paid-analysis walls (analysis/README.md); ships with a default cap
    definitions.toml                                 the operator's definition of a session and of engaged — the numbers the db's sessions table derives by
    cards/{name}.toml                                the model cards — what the deployment can analyze with, one marked `default = true` (analysis/README.md)
data/                                                everything the deployment accretes, created as first use needs it
    events.db
    context/{snippet}.md                             the operator's business frame
    spend.jsonl                                      the paid-analysis ledger (analysis/README.md)
    operator.md  notes.jsonl                         the deployment's memory of its operator (AGENTS.md)
    browse/                                          the deployment's browser: profile, log, screenshots
    outputs/{utc-stamp}_{verb}.log                   everything one run said. One log per run, so
                                                     concurrent reads never cross, and a verb's oldest
                                                     logs drop on a disk budget
    pages/                                           composed pages and the replay material they embed
        locus-replay.js                              the player component (browse open keeps it current)
        {visitor}_{slice}[_plusN_{digest}].js        a slice set's replay payload
        *.html                                       the default pages browse open composes at derived
                                                     names, and the agent's own pages under its own names
    analyses/…                                       analysis directories — one per run (analysis/README.md)
    cache/                                           tool caches (ruff, the workspace-root pytest run)
```

`locus browse` is the deployment's browser: headed windows the agent drives and the operator watches — a replay page opened at a citation, a slice played back for a doubted finding, a page the agent composed, the live site. Each instance is its own window, addressed `w1`, `w2`, …, every command names the window it acts on, and `show` is the only command that brings a window to the front. It is a daemon over the same Playwright Chromium the renderer uses, run as its own process, so windows, logins, and refs from `read` survive between commands and across the daemon's restarts. `open` on a page a window already holds refuses and names the window. `locus browse help` lists the commands.

## Replay

Replay is a component a page embeds, not a page the CLI emits. `locus browse open`, given a slice set or an analysis directory, materializes the scripts a page needs into `data/pages/` — the self-contained player component (the vendored rrweb-player plus the seek contract — no CDN, nothing to serve, works from a local file) and each set's events payload, which registers itself on `window.LOCUS_REPLAYS` — then composes the default page over them, one mount per time-disjoint lane of a visitor's slices on one shared clock, opens it in the deployment's browser, and prints every path it wrote. A payload is one visitor's time-disjoint slice timeline — one player plays it through, later snapshots replaying as ordinary checkouts — and a set one player could not honestly play (interleaved visitors, time-overlapping slices from concurrent tabs) is refused, never rendered plausible and wrong.

Any sibling HTML composes its own replay from the same material: load the component and a payload, then `LocusReplay.mount(element, LOCUS_REPLAYS[0])` — a side-by-side comparison, a summary page with an embedded moment. The composition contract is the header of `data/pages/locus-replay.js`; the default page is refreshed on every open, so an authored page gets its own name beside it. Payloads travel as scripts rather than JSON because a page opened from file:// cannot fetch; script tags are the one loader that works there.

Every mount plays on the recording's absolute epoch-ms clock — the clock answer citations use — and the page's URL fragment addresses it: `<page>.html#t=<epoch-ms>` opens paused at the cited moment, `#t=<start>-<end>` marks the range on the timeline and pauses playback at its end, `&m=<mount name>` narrows the seek to one mount (each is named in its topbar), and a hash change re-seeks in place, no reload; a fragment carrying `t=` or `m=` in any other shape is refused. A moment shows every event stamped at or before it and nothing of the route to it — a player already moved is replaced by a fresh one opened at the moment — and a mount shows only moments its recording holds a page at: a seek outside the set, into a gap between slices, or into a slice's lead (opened, page not yet captured) is refused with an error naming what the mount can show — and a seek across several mounts is refused whole if any of them cannot. `window.locus = { player, seek }` is the page's drive surface for a scripting browser; a refusal there is a thrown error, and `locus browse` relays every error a page throws during a command as that command's failure.

## rrweb constants are generated, never typed

Every numeric↔name mapping in the rrweb schema — event types, incremental sources, mouse-interaction subtypes, node types — comes from `distill/extract_rrweb_constants.js`, which reads the pinned `@rrweb/types` and emits the constants each language imports. A bare rrweb numeric literal anywhere in the repo is a defect even when it happens to be correct; `tests/test_canonical_constants.py` sweeps for the shapes that mistake takes and re-runs the generator to prove the checked-in files are what it emits. Regenerate after any rrweb upgrade. Locus's own event types are declared in the generator too, and nowhere else.

The same rule holds for what the projection knows about HTML: which elements a browser lays out on their own line or does not render, what inline markup means to the eye, which attributes are ARIA states and which native attributes map onto them. `distill/extract_html_semantics.js` reads those from the WHATWG rendering section, the WAI-ARIA specification, and HTML-AAM and emits `distill/html_semantics.js`, stamped with each source's publication date; `distill/ua_style.js` applies the user-agent stylesheet from it. A hand-typed list of block elements or state attributes anywhere is the same defect. Regenerate deliberately, when a spec revision matters.

## Layout

| path | what |
|---|---|
| `src/locus/evidence/` | the tools (the package docstring carries the data-flow spine) |
| `distill/` | the bun-side distillation machinery — internal: `locus load` derives with it, `locus doctor` repairs through it |
| `vendor/` | pinned rrweb replay assets + provenance (`vendor/README.md`) |
| `tests/` | the hermetic suite — `uv run pytest`, and `bun test distill/` for the JS side |
| `tests_live/` | the live capture→store→replay chain against the deployed worker + R2 — `uv run pytest tests_live`, on a configured deployment |

## Conventions

- `events.db` is the keystone: `raw_json` is the source of truth, distilled columns are materialized onto the same rows, slices are accounted in their own table including discards and reasons, and sessions — the operator's definition applied to the recording — in theirs, every event stamped with the session that held it. Canonical order is `(timestamp, counter, id)` — counter is the recorder's emission-order stamp.
- The input is the recorder's chunk stream: slice- and counter-stamped events under the visitor envelope, the custom Locus events in band. A chunk body or event without the recorder's stamps is rejected at decode, the loss named by its chunk.
- Fail loud. Dropped or discarded data is counted and queryable, never silent: `locus status` reconciles, and every recorded event is analyzable, discarded for a named reason, or its slice never materialized.
- The db is not a permanent copy. The deployment's retention horizon (`store/deploy.config.toml`, the same number the bucket expires by) governs it: `locus load` passes over any object holding a recording past it, and both `load` and `locus doctor` drop the slices past it — their events, their chunk manifest and error rows, the sessions re-derived from what remains, and the replay material under `data/pages/`, which regenerates on the next open. Freed pages are zeroed as they go; the file reuses the space rather than shrinking. Analyses are findings, not recordings, and are kept — a citation into a recording the horizon took no longer opens.
- Recordings are verbatim (credentials masked at capture; nothing else) and nothing downstream scrubs: whatever the visitor's screen showed and typed is what screenshots render and what any model sees. Record-time masking is the deployment's decision (`recorder/README.md` § Privacy).
- Dependency cooldown, 48h, one declaration per package manager: `[tool.uv]` [`exclude-newer`](https://docs.astral.sh/uv/reference/settings/#exclude-newer) in the workspace root's `pyproject.toml` and [`minimumReleaseAge`](https://bun.com/docs/runtime/bunfig#install-minimumreleaseage) in `distill/bunfig.toml`. `vendor/` is checked by hand against publish dates. `./update-deps.sh` at the deployment root updates both managers under the cooldown.
