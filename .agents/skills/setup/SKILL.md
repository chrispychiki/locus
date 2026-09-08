---
name: setup
description: The operator wants Locus set up for a site. The recorded urls (`locus ae`, or the events db) show which sites are already recording, if any.
---

Ask the operator up front for whatever needs their hands, so their slow acts — minting the token, creating a key — overlap your setup. Credentials go into the gitignored `store/.env` by the operator's own hands, never through you, so no secret lands in your context or the transcript; everything credentialed reads them from there. Knobs, health checks, replay, and analysis are ordinary operation, available in any later conversation (the change-deployment, quantify, and analyze-recordings skills).

## Set up the CLI — yours alone, no operator

With `store/.env` in place, read `analysis/README.md` § Install, then run `./install.sh` from the deployment root — the whole setup, idempotent, stopping loud on anything missing. `locus` then reads the bucket directly, credentials self-served from `store/.env`.

## 1 · Deploy the store

Read `store/README.md` in full before this step — one Cloudflare Worker fronting one R2 bucket, free tier at typical volumes; the account prerequisites, the `.env` block the operator fills, and the setup gotchas you walk them through are all there.

Then deploy and confirm — that README carries the exact commands, run them rather than restating them: the deploy provisions the bucket, ships the worker, and prints the worker URL; doctor confirms the live state matches the declaration. Mention how long recordings are kept — say the number `retention_days` in `store/deploy.config.toml` declares, that it is every copy and not only the bucket's, and that findings you write up outlive the recordings they cite, so a citation into one past the horizon no longer opens. Changing the number is the change-deployment skill's.

## 2 · The context question

While the operator is still here, ask the one question only they can answer — "what is this site's top goal: what should a visitor end up doing, and what does a wasted visit look like?" Mint the snippet id yourself — short random lowercase alphanumeric, the shape the store validates; it is public in the tag, and its one rule is one id per site — and bank the answer at **`data/context/<snippet-id>.md` under the deployment root**. `locus analyze` resolves that file from the slice set's snippet automatically; a missing file fails loud naming this step.

The file carries the business frame and nothing else: what the site is for, who it serves, what a good and a bad visit mean. Never pad it with site description — pages, flows, prices, offers, and content are already in the evidence every analysis reads — and never infer the goals yourself; they come from the operator's mouth. Draft the file from their answer, read it back, and follow up only where the frame is genuinely ambiguous. What converting means on this site lives here and nowhere else — every later conversion count is judged against this frame.

## 3 · Analysis backend, key, and spend wall

Settle the backend choice now — the key needs the operator's own hands. Which model runs is declared: the card marked `default = true` in the `config/cards/` roster is what every analysis runs on unless one is named per invocation, and the repo ships with the mark on the Gemini card. **Gemini is the recommended choice.** It costs money per call — say so — and needs a key: have them create one (https://ai.google.dev/gemini-api/docs/api-key) and add it to `store/.env` beside the Cloudflare token, so it self-serves; without the key, every analysis on that card fails loud naming it. **The local mlx backend** is free per call but Apple-Silicon-only, memory-bound, and real setup (model download, a running server) — read `analysis/README.md` § The local server (mlx) before committing to it. Check the machine you're on — chip and unified memory — and say whether it can serve a usable model; if it can't, Gemini is the only option. Going local is moving the `default = true` mark to the local card. If they're unsure, leave the declaration on Gemini and grab the key now.

With Gemini chosen, settle the spend wall while they're here: the repo ships `config/spend.toml` with a default cap already standing — read it, tell them the number and its period, and ask whether it suits. Edit the declaration to their answer; declining the question means the shipped default stands. Local calls are free and never touch the wall. The change-deployment skill carries later changes to all of this.

## 4 · The tag

Read `recorder/README.md` in full before this step, then hand over its one-line tag with the worker URL and the snippet id filled in.

Placement scopes coverage: a shared template (a framework layout, a server-side header include) emits it on every page — the common case; without one, only the pages carrying the tag record. A single-page app is one document, so the tag covers every route inherently. Recording skips local/dev hostnames and detected bots by default.

Tell the operator what the tag records: credential fields always masked, everything else verbatim (`recorder/README.md` § Privacy). More masking is a capture-config change (the change-deployment skill) — masking applies at record time, so rules the site wants belong in force before the tag goes live.

However it is placed, the tag must stay its own `<script src>` tag — a template, a tag manager, or an injected loader are all fine, but inlining the bundle's contents or importing it through a bundler breaks the id-and-origin read and the recorder stops with a console error naming exactly that (`recorder/README.md`). And never SRI-pin it: the bundle updates in place on every store deploy, so an `integrity` hash silently stops all recording at the next deploy.

## Done

Recordings arrive when real visitors do. "Is it recording" is the R2 object browser or `locus ls <snippet-id>/` once traffic exists (the quantify skill); every knob is the change-deployment skill; analysis is analyze-recordings.
