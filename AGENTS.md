# Locus

Locus records real visitor sessions on the operator's site, stores them as self-describing chunks in the operator's own object store, and gives you the tools to load, replay, and analyze them into grounded, cited findings. The operator speaks intent; you drive the tools and supply the judgment. The tools are composable; compose them in whatever order the work needs.

## Vocabulary

The terms this system speaks precisely, defined once. The operator won't necessarily know them — define when needed.

- **operator** — the human whose Locus this is: their site, their data, their money. Speaks intent.
- **agent** — you, the operator's interface to all of Locus: hears intent, drives the tools, supplies the judgment; runs in whatever harness the operator brought — Locus ships none.
- **model** — the vision model `locus analyze` puts recorded evidence and a question to. "The model" never means the agent's own.
- **visitor** — the persistent identity the recorder mints: a first-party cookie, so one person is one visitor per browser, not an authenticated user.
- **session** — a visitor's episode on the site. Colloquial use is free; any quantification takes its boundary from `config/definitions.toml`, where the operator's meanings live, applied to the recordings as the db's `sessions` table.
- **chunk** — the recorder's upload unit: a gzipped, self-describing batch of events; arrives arbitrarily late by design.
- **store** — the operator's own object store, where chunks land; the agent reads it directly with the operator's credentials.
- **deployment** — one clone, one store, one db.
- **snippet** — a site's id within the deployment: public in the tag, an organizational prefix in the bucket, not a security boundary.
- **slice** — the unit a recording is stored and replayed in, stamped self-covering by the recorder so it replays standalone. Machinery, not behavior: how slices compose into a session is derived from the evidence under the operator's definition, never stamped by the recorder.
- **analysis** — one run of `locus analyze`: one question put to the model over recorded evidence, one cited answer, recorded in its own directory.
- **window** — the evidence one analysis reads: the named slices, placed on one shared clock.
- **model-payload** — everything one analysis sends the model.
- **user activity** — the visitor doing something at the device, as the recording testifies it. One definition, in `evidence/src/locus/evidence/user_activity.py`: rrweb's own user-interaction rule plus what it omits; every count of user activity and every activity-screenshot reads it.
- **activity-screenshot** — a screenshot included in the model-payload, taken at a fixed interval wherever there was user activity.
- **pull** — the model asking, mid-analysis, to see additional moments.

## Driving it

The `locus` CLI is your one interface across the whole deployment — reading the store, replaying and analyzing sessions, and querying its observability telemetry. `setup` installs it on your PATH. Ask `locus --help` for the commands and `locus <command> --help` for flags; never guess a verb or option, because the CLI is the only authoritative list of what exists.

Depth lives beside the code, in the package READMEs: `recorder/` is capture in the visitor's browser, `store/` the ingest worker and its deploy, `evidence/` the local db and replay, `analysis/` the model layer and the CLI's home. Read the one for the layer you're working in.

Use the skills, and err toward using them. Each is a few thousand tokens of pure signal — what the data cannot say, what a call costs, where an answer that looks right is wrong — none of it in the code or the schema. Reading one costs almost nothing; skipping one costs the answer. When a skill fits what you're about to do, read it, even mid-task, even when you already know how to do the thing.

Lint and format from inside a module, where the pinned tools live: `uv run ruff check` / `uv run ruff format` in Python, `bun run lint` / `bunx biome format` in JS. Both resolve their config (`ruff.toml`, `biome.jsonc`) from the repo root.

## Time

You don't know what time it is, and you don't know how much time has passed since your last turn. It feels like you do. You don't. You were never trained to sense time, and the feeling is just based on how much work you did, and is nearly always wildly off. Run `date -u` before you write any timestamp or reason about how recent or how long ago anything was.

Your shell runs in the machine's local timezone, and Locus is UTC end to end, so anything it prints about time — `date`, file mtimes, `git log`, `stat` — is local unless you force it: `date -u`, or `TZ=UTC` in front of the command. Say it's UTC when you write a time down.

## The operator

`data/notes.jsonl` is a log of what you observe about the operator. Create it the first time you have something to write, then add an entry as things happen: the time, from running `date -u` (don't guess it), and a plain description of what happened — what the operator said or did, what you delivered and how they took it, including no reaction at all. Keep it descriptive. No conclusions. Never edit or trim old entries; only append.

`data/operator.md` describes the operator and how to work with them. Read it before any work and let it shape how you ask, present, and anticipate. Don't quote it to the operator, don't withhold or soften a finding because of it, and don't let it override what the operator is telling you right now — if they contradict it, that's a note. It doesn't exist until the first consolidation creates it. You must invoke the skill before writing to this file. Any contradictions you just observed are jotted first, and absorbed *after* you call `/consolidate-notes`.

## Invariants

- **Configuration is declared and applied by deploying — never the dashboard.** The repo is the source of truth; every deploy reconciles the live deployment back to the declaration, so a setting clicked into the Cloudflare dashboard, or changed with a one-off CLI command, silently reverts on the next deploy. Change a setting where it is declared, redeploy, then confirm the live state matches rather than assuming the command worked — `locus doctor` surfaces drift. `store/README.md` has the deploy commands.
- **Masking is decided at record time and is irreversible both ways.** Nothing downstream recovers what wasn't captured, or un-captures what was; reconfiguring masking changes only future recordings.
- **One declared horizon governs every copy of a recording.** The deployment declares how long recordings are kept once; the bucket expires its chunks by it, and `locus load` and `locus doctor` drop the local db's copy — and the replay material derived beside it — by the same rule, never fetching what it has already dropped. Analyses are findings, not recordings: they are kept, and a citation into a recording past the horizon no longer opens.
- **Paid models spend the operator's money, within the walls in `config/spend.toml`.** The analysis runs on Gemini (paid) or locally on mlx (free); the cap the operator set is their standing ok for paid calls under it.
- **One bucket is one trust domain.** Any credential that reads the bucket reads every snippet in it; sites that shouldn't share read access go in separate buckets.
- **The store is open-write by shape, not by auth.** The worker validates that an upload looks like what the recorder emits; it is not gated by a secret. Don't design as if there is upload auth to lean on.
- **The worker is only the write path.** Every read is `locus` on the bucket directly, and the bucket is never publicly readable.
- **Time is UTC end to end.** Every minted timestamp is epoch-milliseconds, and every derived day — the R2 key's date partition, an AE query window — is the UTC day, so reason in UTC.
