# Locus

Locus is product analytics reimagined for the agentic era.

Traditional analytics is tedious to set up and interpret:
- Instrumentation is manual work: defining tracking plans, tagging events, creating funnels, etc.
- If you forgot to predefine part of a flow, there's nothing you can do about historical data. And when your site changes, the events you tagged quietly stop meaning what they used to.
- All the telemetry you set up ends up in an opaque data warehouse, viewed through arcane dashboards/reports you can't deeply customize.

The tedium exists because deriving semantics from syntax takes judgment, and that judgment used to be all human: a series of clicks and page loads had to be prelabeled to mean anything, e.g. a checkout drop-off. Now the model reads that off the raw stream at analysis time, for any event sequence, and nothing has to be predefined at instrumentation time. Retroactive semantics.

The interpretation half can't be fully reduced, unless you intend to completely outsource product decisions to your agents, which I wouldn't recommend. The rote work can and should be offloaded — trawling through replays, wrestling with schemas and dashboards — which means your work is figuring out what questions to ask.

Now you can close the product loop from one surface: your agent builds → sees how users engage → discusses with you → builds.

## What agent-native unlocks

Locus is an open source toolkit that makes your session replay corpus agent-legible. It is not a harness itself; bring your own. I've only tested with Claude Code, but Codex, Grok/Cursor, and other harnesses should theoretically work out of the box.

This probably only works on Macs at the moment; I'll check that it works elsewhere soon.

You do everything through your agent: setup, management, and every question you ask. Locus provides the primitives and your agent composes them, picking which sessions are worth looking at for each question.

One prompt sets up a background sentry (use your harness's built-in scheduling if no /loop):
- `/loop every morning, email me a rundown of yesterday's sessions that hit a dead end. overall counts and suspected root causes`
- `/loop every hour, score new sessions for frustration, in a structured log for easy aggregations later`
- `/loop every monday, slack me a tldr of user activity past 7 days`

To get findings, hop into the sentry's session and ask, or have it keep a log a fresh session can read. To be notified instead, give it a connector for the channel you want and ask.

Or just ask anything, whenever it occurs to you:
- `where do mobile users drop off in checkout?`
- `why did churn spike last week? dig through the sessions and give me your strongest hypotheses, with reasoning, plus remediations i could try`
- `is retention any better for people who watched the demo on their first visit?`
- `build me a dashboard for how site search is doing`

The pipeline underneath is standard. What changed is where the processed data lands: not in a dashboard someone else designed, but in your agent, which defines the retention cohort at question time instead of inheriting the one a preset chart baked in, and builds the dashboard to match.

## Getting started

```sh
git clone https://github.com/chrispychiki/locus && cd locus
claude "What is Locus and how does it work? How do I get started?"
```

## How it works

Your agent runs on your machine and uses the `locus` CLI to manage every stage of the pipeline:

```
  recorder ──chunks──▶ store ──load──▶ evidence ──▶ analysis ──▶ cited findings ──▶  historical trends/behavior reports
 (browser)           (R2 bucket)      (your machine) (gemini/mlx)                     (custom dashboards)
```

On the Gemini backend, the heaviest thing Locus runs is Chromium, so 8GB of RAM is enough. Disk is a function of what you load, on the order of a few MB per visitor.

### Capture

`recorder/` contains a hardened script built on top of [rrweb](https://github.com/rrweb-io/rrweb) for recording user sessions. Install it with a one-line script tag on your website (run /setup).

Recordings are stored in an [R2 bucket](https://developers.cloudflare.com/r2/), and metrics are emitted to [Analytics Engine](https://developers.cloudflare.com/analytics/analytics-engine/). Without loading, you know what arrived and the recorder's telemetry, not what anyone saw or did. A chunk's R2 key is `site/date/visitor/slice/chunk`: a date range is a cheap prefix listing, a whole site is a scan.

Visitors are pseudonymous ids, a first-party cookie the recorder mints; `identify` attaches your user id from your backend when they log in. Locus doesn't collect your backend's events, so before your agent can join the two you have to give it a way into wherever they live (see `recorder/`).

`store/` contains the [Cloudflare Worker](https://developers.cloudflare.com/workers/) that accepts chunks and vends the recorder script to the tag. Our stack is Cloudflare (CF) because it is cheaper than AWS, has free data egress, and serves the script as a static asset, which doesn't count as a worker invocation.

### Evidence

`rrweb` recordings are meant for session replay, which means we incidentally get a comprehensive stream of user actions and DOM events that LLMs can extract meaning from — the fundamental reason retroactive semantics works. The raw stream is far too noisy to dump into context, so [`evidence/`](evidence/) loads recordings into a local db and distills them into what a model can read, with enough flattened into columns that your agent can query for the relevant visitors and sessions.

Everything after capture runs on your machine, so your agent has to load recordings before it can ask anything of them, and it loads only what a question needs.

### Analysis

Your high-level directive gets decomposed by your agent into a set of analyses and their questions, based on complexity and scope. [`analysis/`](analysis/) — also home of the `locus` CLI — hands each one to the model as events, screenshots around user activity, recording facts, your website context, and the question; the model can ask to see more moments (its call, not a heuristic's). Every analysis is priced before it spends, and your spend caps are enforced on every billed call.

I recommend Gemini for these low level analyses — a vibes-based prior, but I believe it has the best visual intelligence per dollar. That might change in a month, so the backend is swappable. Local models, unless you have a server rack at home, aren't a primary workhorse: ~30B models aren't ready for this yet (again, vibes), but they're a free way to see it all work end to end, and might suffice if your site and questions are simple enough.

The output is a folder with the question, timestamp-cited answers, and the evidence. As they accumulate, this becomes a corpus about user behavior on your site over time that future agent sessions can peruse — synthesize high level findings from it, or dig all the way into the replay to see the evidence for a specific claim yourself.

## Caveats

You fully own and control all data and infra; there are no Locus servers to potentially mishandle your data. The trust boundaries are around CF and Gemini. This does mean you're responsible for handling your customer data appropriately (e.g. GDPR). Unlike most session replay tools, Locus records form input verbatim by default, credentials excepted: the recordings are in your bucket and the liability call is yours. Masking is decided at record time and shapes only future recordings.

Everything after capture lives in your clone — the db, the analyses, what your agent has learned about you — so a teammate's clone on the same bucket shares the recordings and none of the rest.

Costs are per use since you're directly billing with CF/Gemini. Inference costs will likely dominate total spend.

Your agent needs to be a strong model; I wouldn't use anything weaker than Claude Opus 5.5 or GPT-6.1 Sol. Every website and every operator is different, and no configurable layer covers every case, so beyond orchestrating analyses and reading your intent it will sometimes have to change the business logic itself. That takes the highest discernment available.

And no matter how smart your agent is, you cannot abdicate responsibility. Some information only lives in your head, and it's on you to bake it into the website context and business logic. Set up evals. Tune your prompts. Review every tweak your agent makes. Don't treat models as fungible, even point releases: recheck prompts and context when anything changes, and spot check findings. Product decisions you make on what your agent delivers are yours.

A deeper open question, no matter how good the models get, is whether session replay has enough signal to be useful. User behavior is overdetermined and a poor proxy for intent: someone lingers on your checkout page — bad CTA, overpriced product, or afk? Aggregates don't get you out either. You can rigorously claim there *is* an issue with a flow, but not, in principle, *why* — not the why in the visitor's head. The other why, what the page did to them, is on the recording: the form that reset, the tap that did nothing, the error that appeared. Session replay settles that one.

So it remains unclear whether agents armed with replay evidence can guide product decisions. What they can do is scale up the hypotheses you A/B test and iterate faster on clear usability issues, and maybe that's all this needs to be. How to grow your business is yours to answer; Locus is one tool.

## Contributing

Open an issue for anything — a bug, an idea, or just how it went, good or bad; I want to hear it. PRs welcome.

## Security

Vulnerabilities go through GitHub's private reporting, not the issue tracker — see [SECURITY.md](SECURITY.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
