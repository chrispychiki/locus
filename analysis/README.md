# Locus analysis

The semantic layer: the evidence read into grounded, cited findings. An analysis composes an evidence window over a named slice set, drives a vision model over the event stream and screenshots, and lands a timestamp-cited answer whose citations open the replay. The `locus` CLI lives here.

`locus --help` lists the commands and `locus <command> --help` documents each one's flags — the CLI is the only authoritative account of what exists. A command hands back one line, the address of the log it says everything into, then the log's own body when the whole of it is small — a log ends with its run's result, so the tail of a long log is the answer, and a log that stops short of one is a run that died; the exceptions are `browse`, whose replies are a page being looked at, and an analysis run, which hands back its directory and says everything into the `analysis.log` inside, echoed the same way when small.

## Install

`./install.sh` from the deployment root: it syncs the workspace, installs the distillation's bun dependencies, links `locus` onto PATH pointing at the workspace env, and installs the Chromium that `browse` and the screenshot renderer drive (the Python dependency does not fetch the binary). Idempotent — re-run it after a dependency change and the linked CLI picks it up. `bun` must be on PATH: the distillation pass, the rescue gate, and the store integrity check are bun scripts the CLI shells into.

The bucket is read through the `locus` commands, which self-serve their credentials from `store/.env` — nothing exported, nothing configured. For the rare job none of them covers, generic S3 tooling (aws, rclone) can reach the same bucket: `install.sh` writes an `~/.aws` `[profile locus]` — opt-in per invocation via `--profile locus`, never a default — whose `credential_process` line shells out to `.venv/bin/locus-r2-credentials` (not a `locus` verb; that config line and install.sh are its only callers) for the same keys. `locus-r2-credentials --help` carries the profile text; `uninstall.sh` removes the profile.

## The analysis and its record

An analysis directory in `analyses/` beside the db is one `locus analyze --run`: named by the UTC moment it ran (first, so the family sorts chronologically), then what it analyzed. Two analyses over the same subject are different artifacts by design. An analysis is written once: created fresh, failing loud rather than overwriting — its contents are the evidence for a finding, so a rerun makes a new one.

```
analyses/{utc-stamp}_{visitor}-{disambiguator}[_plusN]/  one analysis: occasion, then subject — and the
                                                     address the run hands back
    analysis.log                                     everything the run said as it went
    window.json                                      the machine manifest — question, clock anchor, slice table
    N_answer_input.txt  N_answer_response.txt  N_answer_meta.json
    screenshots/{label}_screenshot_{ms}.png
```

The answer is the record itself: the final `N_answer_response.txt` — thoughts where the backend exposes them, the meta section addressed to whoever tunes the system when the model wrote one, and the cited answer, exactly as the wire returned them — with `window.json` carrying everything a reader wants beside it: the question, the absolute clock the cited offsets anchor to, and the slice table.

Every model call leaves three files in the analysis dir: `N_<label>_input.txt` (the entire interpolated input as sent, roles delimited, images as markdown links resolving from where the file sits), `N_<label>_response.txt` (thoughts inline, delimited), `N_<label>_meta.json` (config, usage, wall-clock elapsed, error). Persistence lives at the wire in the conversation layer and is never optional — a conversation states its record home at construction, so every backend and every caller inherits the record, failed calls included.

## Model backends

A session analysis call carries dozens of rendered screenshots, so a backend must be **vision-capable on a transport where the model's context is the actual limit**: nothing between the caller and the model may bound how much visual evidence fits in a call, and nothing may drop images from a request that otherwise succeeds.
Two transports have that property, and `model/factory.py` is the only place a model name becomes one — by the card's own `conversation` declaration (each card one file in `config/cards/` under the deployment root: `gemini`, or `openai-compatible` at the card's `base_url`), looked up in the registry there. A new transport is a `Conversation` subclass (`model/protocol.py`) and one registry line.

- **Gemini-direct** — the stateful [Interactions API](https://ai.google.dev/gemini-api/docs/interactions): each turn sends what it adds and chains the previous interaction, the server holding the conversation exactly as the model produced it, thought signatures included. Images upload through the [Files API](https://ai.google.dev/gemini-api/docs/files) and ride as URIs, so request size never bounds visual context; it holds up to the model's context itself. Spends money on every call, inside the deployment's spend walls (Counting and pricing, below).
- **local** — an OpenAI-compatible server sharing a filesystem: images travel as file paths, nothing is re-encoded. A local call spends nothing, which is why it doubles as the engine's end-to-end harness: with the bundled mlx server up (The local server, below), `uv run pytest` drives the engine for real — render → activity-screenshots → pull → cited answer, every model call real — for free.

## Counting and pricing

- **Nothing stores a token count.** A stored count goes stale at the next projection or distillation change; counts are computed fresh at decision time.
- **Counting is exact by construction.** The model-payload is composed through the real composer into a tally conversation, so the counted text is byte-for-byte what the wire would carry. Text goes through the pinned tokenizer, exact-verified against Gemini's own free `countTokens` on that path (a deployment declaring no local card pins no tokenizer and counts with `countTokens` alone); screenshots are priced at the flat per-image cost of the Gemini tier they ride, or locally by the ViT closed form over the geometry actually sent (constants read from the pinned processor config).
- **Never budget against bytes or characters.** Noise collapse makes byte size anti-correlated with event-stream size: a slice many times larger on disk can project to a small fraction of the tokens.
- **An analysis is priced against turn 1** — the only turn that exists before anything runs. What the model will think, write, or ask to see cannot be counted before it has been asked, so it is not counted: an analysis fits when its turn-1 model-payload comes in under `headroom` × the model's context, and the rest of the context is left for the later turns (`headroom` declared on each card beside `context_tokens`, because the right ceiling differs by backend).
  Every turn after the first is measured exactly, at the wire, against the context genuinely left — when the model pulls screenshots mid-call, the engine serves what fits and tells it plainly what did not.
  The local backend prices each request before send and refuses loud when it would not fit the card's context, sparing the doomed request its tokenization and upload.
- **An oversized model-payload is refused with its price.** The bare `locus analyze` invocation is the price report — exact tokens, context percentage, per-slice standalone prices — and a run whose model-payload out-prices the budget is refused with that report attached.
  When one slice alone out-prices the budget, the report enumerates the slice's route-boundary pieces, each priced and addressable as `<slice>#<k>`, so the agent picks a coherent piece by structure, never by milliseconds.
- **Screenshot spend has two owners.** The activity-screenshot interval — a flag on `locus analyze` — thins the pushed screenshots on either backend.
  What each screenshot costs on the wire is the model card's (its file in `config/cards/`): a local card declares a per-image token ceiling per kind (`max_image_tokens`) its backend realizes by downscaling at the wire and pricing what it sent, while a Gemini card declares a flat-priced resolution tier per kind — either way one for activity-screenshots, one for the screenshots a pull returns any moment as, pushed or not.
- **Selection is the agent's; composition is the engine's.** Which slices deserve the model's read, and how to whittle an over-priced analysis, is judgment informed by the price report.
  Composing the chosen set into one evidence window — one shared clock, every slice labeled, the activity-screenshots, the event stream — is invariant-bound mechanics, so `analyze` owns it.
- **A window is any slice set on one shared clock.** Each named slice is labeled S1..Sn in time order — the label a window-local abbreviation of the slice id — and slices may overlap in time — a second tab, concurrent visitors — their offsets genuinely overlapping on the one clock rather than being serialized apart.
  Any subset is a legal window: a seam between named slices gets a mechanical gap statement in the evidence, like the window's edges.
  A single-slice window cites bare offsets (`[MM:SS.mmm]`); a multi-slice window's evidence and citations carry the slice label (`[S2 04:31.220]`). The contract states both cases in one text for every window, and the model tells them apart by the SUMMARY's Slices line, printed only when the window has several.
  `window.json` carries the slice table (label → visitor, slice, bounds), so resolving a citation to a replay is a table lookup: the label names the recorder slice, window_start_ts + offset is the absolute moment a replay page's `#t=` fragment takes. Wall-clock is never an address — the agent keeps addressing by slice ids.
- **Billed spend is walled and ledgered at the conversation layer.** A card that declares a `pricing` block (in its file in `config/cards/`, USD per million tokens; thinking bills at the output rate) has every generation call governed: before the request goes to the wire, the deployment's declared caps — `config/spend.toml`, USD caps over UTC calendar periods, the shipped file's own comment naming the keys — are checked against the ledger, and a reached cap refuses loud, naming the cap, the recorded spend, and the declaration to edit; after the call, the spend lands in `spend.jsonl` beside the db, computed from the call's persisted usage × the declared prices.
  The declaration ships with the repo carrying a default weekly cap, so a fresh clone is walled from its first paid call; the file is the only truth — what it declares is the walls, and an operator who wants no walls deletes the caps.
  The check is check-then-spend: one in-flight call can carry spend past a cap, and the next paid call refuses.
  Every Gemini card declares prices; the shipped local cards declare none and spend nothing.
  The spend block of `locus status` reports each period's spend against its cap, and on a priced model the price report states the turn-1 input dollars — budgeting a multi-analysis investigation is the agent's own arithmetic over those two surfaces.
- **Declared prices are the record; the registry is the check.** Enforcement reads only the card's declared prices.
  They are verified best-effort against LiteLLM's public price registry wherever a price is stated (the price report, the first billed call of a process, `locus doctor`) — fetched at most once per UTC day per machine, loud on a mismatch with both numbers, and honestly `unverified` when the model is unlisted or the registry unreachable.

## Grounding

The oracle (`oracle.py` over `ground/`) is a parked proof of concept: nothing invokes it, and it is not expected to be used as shipped, standalone or otherwise. It scores a finished analysis — the answer decomposed into typed claims, each judged against the screenshots and events in the span it cites — running offline against what the analysis left behind.

## The local server (mlx)

A thin layer over the upstream [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) OpenAI-compatible server, at `src/locus/analysis/mlx/`.
The local card (`config/cards/qwen3.6-35b-a3b-6bit.toml`) owns the endpoint it dials (`base_url`) and which repo and context the server loads; this layer boots that declaration.
Images travel as plain absolute paths on the shared filesystem, so request size never bounds visual context, and nothing leaves the machine.

Apple Silicon only (MLX), installed as the `mlx` extra: `uv sync --extra mlx` at the deployment root. Models load from `mlx-community` on first run.

### Run

```sh
sudo sysctl iogpu.wired_limit_mb=<mb>   # the operator's dial — set from what the machine tolerates
scripts/start-server.sh qwen3.6-35b-a3b-6bit
```

The script detaches the server, logs it to `data/mlx/serve.log` under the deployment root, and returns when it answers — or fails with the log's tail when it doesn't. `uv run locus-mlx-serve <slug>` is the same server in the foreground. Either way it refuses to boot with the sysctl unset: it is the wired ceiling MLX is held to.

Liveness and footprint come from `GET /health`, which carries the live memory snapshot — never from `ps`: a resident model's memory is wired and compressed out of the process's resident set, so RSS reads a fraction of the true size.

`locus analyze` runs against this server whenever its model is a local card — the deployment's default when the roster card marked `default = true` is local (the change-deployment skill carries the choice), or a local card an invocation names with `--model`.

To probe the running server directly — check it answers, try a prompt, watch a schema come back:

```sh
uv run locus-mlx-chat --prompt "what is in /abs/path/image.png?"
uv run locus-mlx-chat --prompt "count the vowels in banana" \
    --json-schema '{"type":"object","properties":{"vowels":{"type":"integer"}}}'
uv run locus-mlx-chat --prompt "say hello" --verbose   # the whole body: usage, timings
```

One prompt, one reply, exit. An image path anywhere in the prompt is sent as an image. It sends no sampling values, so what comes back is raw upstream behavior, not what the cards ask for.

Stop with `scripts/stop-server.sh`, which verifies process death: a free port does not mean the model is unloaded — in-flight generation can hold tens of GiB wired for minutes after the listener closes.

**Out of memory kills the server.** A request whose forward pass needs more than the dial does not error; Metal fails the command buffer and the process dies — the analysis's stream closes with no result, its log stops at its last phase line, and `data/mlx/serve.log` ends in `[METAL] Command buffer execution failed: Insufficient Memory` from the generation thread.
Nothing on the server or in `locus analyze` predicts it: memory is not projectable from the request (§ Throughput, caching, concurrency).
Recover by rebooting (`scripts/stop-server.sh`, then `scripts/start-server.sh`), and then cut what was sent — fewer slices, a wider `--screenshot-interval`, a route-boundary piece — or lower the headroom in the card's file so the context the card admits is one this dial serves. A window that still dies after the cuts an operator's question can bear is a model that does not fit this machine.

**Thermals are the operator's, like the dial.** A heavy window is minutes of saturated GPU, and a long unattended run is many back to back. This layer ships no fan or thermal management. If you run unattended, put your own GPU-temp-driven fan control in front of these runs (restored to auto on exit); throttling shows up in the per-request telemetry.

### Throughput, caching, concurrency

Every response carries a `timings` sibling to `usage` (prompt/generation token rates, wall-clock, peak model memory); budget wall-clock for heavy windows from your own deployment's timings. Expect prefill to dominate and to slow as context grows.

Prompt caching (upstream APC) is off in `src/locus/analysis/mlx/models.toml`, so every turn prefills its whole conversation: a pull turn or a citation-repair turn re-pays the window it extends. The residency is the price it declines — this model's cache layout is hybrid, so upstream reuses a prefix only as whole-prompt KV snapshots held resident between requests, roughly three copies of the prompt's KV where an uncached server holds one, which for a large enough window is the difference between fitting under the dial and dying in Metal.
The vision feature cache is on: a window's screenshots pay their encode once, and immediate re-analysis of the same window skips it.

Concurrency: clients may fire freely, and the server admits one request at a time to the three routes that generate (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) — first come, first served, the line as long as it is. One resident request is the whole memory policy: the machine holds the weights, one cache, and one prefill's working set, never a batch. Whether one request fits under the dial is not checked anywhere, because it cannot be projected: a request's peak is what upstream's forward pass materializes over that request's exact shape — text and image tokens cost differently, the encoder's transient tracks pixels, a prefill chunk's attention scratch tracks the whole context, the allocator rounds on top — a fixed number for a given shape and this pin, but not linear in any count, not in any config, and not recoverable from other requests' peaks. The only gates are on context: pricing before the send, and upstream's own check at admission. A request that does not fit dies in Metal (below), and the cut is made after the fact. It is also the whole throughput policy on this backend: a prefill saturates the GPU and starves any decode sharing it, and a batched decode over rows padded to the longest reads more cache per step than it saves in weights, so a second resident request costs more than it returns. A waiting `locus analyze` holds nothing while it waits — its browser is closed between rounds of captures — and its log says which phase it is in (waiting for admission, admitted and prefilling, generating) with the server's request id; `/health` carries the admission state (`admission`: the request being served, how many wait) for anyone in line.

### What upstream provides

Generation is entirely upstream [mlx-vlm](https://github.com/Blaizzy/mlx-vlm): continuous batching (`server/generation.py` ResponseGenerator: "Continuous batching for concurrent requests via a single GPU thread"), thinking with a per-request `thinking_budget` (`server/schemas.py` request field, `utils.py` ThinkingBudgetCriteria), constrained JSON via llguidance (`structured.py` LLGuidanceLogitsProcessor; `response_format` json_schema, grammar deferred until `</think>` by ThinkingAwareLogitsProcessor so thinking stays unconstrained), automatic prefix caching that is exact and supports hybrid-attention `RotatingKVCache` layouts via an exact-prefix snapshot (`apc.py`: "APC itself is *exact*", `_cache_entry_supports_exact_apc`), and speculative decoding (`speculative/`). Thinking returns as `message.reasoning` (`server/schemas.py` ChatMessage.reasoning) and streams as `delta.reasoning` (`server/openai.py`).

### What this layer adds

Only what upstream lacks and a resident model demands:

- **Memory limits with fail-loud boot** — the `iogpu.wired_limit_mb` sysctl is honored as both the wired (`mx.set_wired_limit`) and allocator (`mx.set_memory_limit`) limit. MLX documents the wired limit as "the total size in bytes of memory that will be kept resident" and names this sysctl as the system wired limit it must stay under (`mlx.core.set_wired_limit`, [ml-explore/mlx](https://github.com/ml-explore/mlx)); the wired portion is unreclaimable by the OS.
- **Liveness guard** — a machine-level pidfile boot guard (one fixed path, whichever clone boots) makes two resident model instances, the state a machine does not recover from, structurally impossible.
- **Admission** — one request at a time, as above.
- **Memory telemetry** — every generating request is bracketed by the full-signal memory sampler in `src/locus/analysis/mlx/telemetry.py`, streamed to `data/mlx/memory.jsonl` under the deployment root, keyed by a request id the response returns as its `x-locus-request` header — so a caller can read exactly its own request's rows and serve-log summary line.
- **A lock around upstream's CPU preprocessing** — upstream tokenizes and loads images on each request's own thread against one shared HF fast tokenizer, whose Rust core raises "Already borrowed" on concurrent use ([tokenizers#537](https://github.com/huggingface/tokenizers/issues/537)).
  One process-wide lock closes the race; GPU batching is untouched.
- **A decode-speed patch for an upstream throttle** — long-context decode is bound by KV-cache reads, and mlx's decode-attention kernel leaves roughly half the memory bandwidth unused on this model's GQA shape (each query head of a group re-reads the group's K/V); eligible decode calls route to a shared-tile Metal kernel with identical math (`gqa_decode.py`).
  The patch is keyed to the installed upstream's call signature.
- **Media rendered where the conversation put it** — upstream's chat-completions handler strips each message's images out before templating, and the template layer, left with no marker to place them by, hangs every image on the newest user message; within a message, the prompt builder then joins the text into one string and re-attaches the images ahead of it. A conversation's evidence therefore jumps forward to whatever turn is being written now, while a message's own text/image interleaving collapses to an image block before its text.
  `media_placement.py` renders each image in the message, and at the position, that carried it; the model's own chat template renders parts in the order it is handed. The server records each request's rendered prompt at `data/mlx/rendered_prompt.txt` (latest request wins), so the placement is checkable against the wire.
- **Pixel-bounded vision encode** — upstream encodes a request's entire image set as one graph, so the encode's memory transient grows with total pixels and a large enough image set kills the server with an uncatchable Metal OOM even when it fits the model's context.
  `vision_chunking.py` encodes in contiguous whole-image chunks bounded by `LOCUS_MLX_ENCODE_CHUNK_MPX`, output-identical on the pinned upstream — attention is image-local, so nothing crosses chunk boundaries — and keyed to upstream's source.
- **Refusal on upstream drift** — every patch above stands in for exact upstream source, so at boot each verifies what it replaces (a source hash, a call signature) and any mismatch refuses the boot, naming the drifted function and the re-validation owed (`patching.py`). The pinned versions arrive via the lockfile, so a mismatch is only ever the result of a deliberate dependency upgrade — which is exactly a re-validation event.

The model cards (`config/cards/` under the deployment root) state the model and every sampling value on each request, and whatever a caller omits falls to upstream's own schema defaults, exactly as against bare upstream. The card also owns which repo a slug serves, the context it holds, and the endpoint the client dials: the server derives its boot repo, KV budget (`MAX_KV_SIZE`), and bind host/port from the card at boot, so what it serves, guards, and where it listens are exactly what analysis requests and prices.
`models.toml` carries only the slug's serving env.

Slugs are the `models.toml` keys, identical to the card slugs. To serve something else, add a card and a serving-env entry.

## Tests

`uv run pytest` from `analysis/` — the composition, pricing, engine, grounding, and CLI suites, hermetic. The engine's end-to-end test runs **free** against the local model when the mlx server is up, and skips with the command to start it otherwise. `tests/mlx/` drives the server layer itself: unit suites offline, and an integration suite against the live server through three levels of proof — reception (prompt tokens reflect vision tokens), accounting (stable across cache hits), semantics (counts, colors, and UI-scale text that exist only in pixels) — plus streaming, speed, and a vision-cache correctness gate: an answer computed from cached image features must equal the same request answered cold.
