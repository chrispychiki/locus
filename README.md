# Locus

Locus is product analytics reimagined for the agentic era.

Traditional analytics is tedious to set up and interpret:
- Instrumentation is manual work: defining tracking plans, tagging events, creating funnels, etc.
- If you forgot to predefine part of a flow, there's nothing you can do about historical data. And when your site changes, the events you tagged quietly stop meaning what they used to.
- All the telemetry you set up ends up in an opaque data warehouse, viewed through arcane dashboards/reports you can't deeply customize.

Deriving semantics from syntax requires judgment, and the traditional tedium exists because that judgment used to be all human. A series of clicks and page loads are bare facts that need to be prelabeled so we know what the sequence actually means, e.g. a checkout drop-off.

Now that we have increasingly intelligent models, you don't need to predefine anything at instrumentation time. The model understands the click/load sequence is a checkout drop-off at analysis time, and can do the same for any arbitrary event sequence from the raw stream. Retroactive semantics.

The interpretation half can't be fully reduced, unless you intend to completely outsource product decisions to your agents, which I wouldn't recommend. The rote work can and should be offloaded, which means your work is figuring out what questions to ask.

For instance, synthesizing large amounts of information should be offloaded. No more trawling through endless session replays yourself — have your agent do it for you. No more dealing with esoteric data schemas and wrestling with dashboard customizations — have your agent build a custom dashboard or report for you with exactly what you need.

Now you can close the product loop from one surface: your agent builds → sees how users engage → discusses with you → builds.

## What agent-native unlocks

Locus is an open source toolkit that makes your session replay corpus agent-legible. It is not a harness itself; bring your own. I've only tested with Claude Code, but Codex, Grok/Cursor, and other harnesses should theoretically work out of the box.

This probably only works on Macs at the moment; I'll check that it works elsewhere soon.

You do everything through your agent: setup, management, and every question you ask. Locus provides the primitives and your agent composes them, picking which sessions are worth looking at and adjusting knobs like the screenshot interval to fit each question.

One prompt sets up a background sentry (use your harness's built-in scheduling if no /loop):
- `/loop every morning, email me a rundown of yesterday's sessions that hit a dead end. overall counts and suspected root causes`
- `/loop every hour, score new sessions for frustration, in a structured log for easy aggregations later`
- `/loop every night, classify the day's sessions by intent and outcome into a separate sql table`
- `/loop every monday, slack me a tldr of user activity past 7 days`

To get findings, hop into the sentry's session and ask, or have it keep a log so a brand-new session can read it and catch you up. To be notified instead, set up a connector for whatever channel you want (e.g. a Resend API key for email) and ask.

Or just ask anything, whenever it occurs to you:
- `where do mobile users drop off in checkout?`
- `why did churn spike last week? dig through the sessions and give me your strongest hypotheses, with reasoning, plus remediations i could try`
- `is retention any better for people who watched the demo on their first visit?`
- `build me a dashboard for how site search is doing`

The data pipeline underneath is standard stuff. The processed data used to land in a dashboard designed by someone else. Now it lands in your agent, which defines e.g. the retention cohort at question time instead of inheriting the one a preset chart baked in, and builds you a dashboard on the fly tailored to your needs.

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

Recordings are stored in an [R2 bucket](https://developers.cloudflare.com/r2/), and metrics are emitted to [Analytics Engine](https://developers.cloudflare.com/analytics/analytics-engine/).

The R2 key format is site / date / visitor / slice / chunk, for efficient querying. Your agent will use this metadata to load the superset of data needed for evidence gathering, based on what you want to find out, e.g. "how did users respond to the feature I launched the other day".

`store/` contains the [Cloudflare Worker](https://developers.cloudflare.com/workers/). Once deployed, it accepts recorder payloads (aka chunks) and vends the recorder script to the tag.

Our stack is Cloudflare (CF) because it is cheaper than AWS, has free data egress, and serves the recorder script as a static asset, which doesn't count as a worker invocation.

### Evidence

`rrweb` recordings are meant for session replay, which means we incidentally get a comprehensive stream of user actions and DOM events that LLMs can extract meaning from (the fundamental reason retroactive semantics works).

This raw event stream is very noisy for analysis purposes; a naive dump into context would be extremely token inefficient. Thus [`evidence/`](evidence/) loads from R2 and hydrates/distills into a local SQLite db. The events are compressed to their salient aspects, and a subset of attributes are flattened into columns so your agent can query efficiently for relevant users and sessions.

After selecting the relevant subset, your agent hands a consolidated model-payload of events + screenshots + recording facts + website context + question to Gemini/MLX for granular analysis. The screenshots are automatically generated on the fly for each model-payload, at a fixed interval around user activity (you can customize this heuristic for your own site).

### Analysis

The question portion of the model-payload is generated by your agent. Your high-level directive gets decomposed to a set of analyses and their questions, based on complexity and scope. [`analysis/`](analysis/) — also home of the `locus` CLI — prices every analysis before it spends; spend caps are declared in `config/spend.toml` and enforced on every billed call.

At the analysis level, the model starts with a sparse set of screenshots and can request additional moments it wants to see. On Gemini that follow-up turn is cheap, because the evidence already sent is cached.

I recommend using Gemini for these low level analyses. This is admittedly my vibes-based prior, but I believe Gemini has the best visual intelligence per dollar. Of course, this might change in a month, so the backend is designed to be swappable. Switching models within a provider is trivial; integrating a whole new provider should still be straightforward. The MLX backend is an example of how you might integrate a new provider: a client module speaking the provider's API, plus a card declaring the model.

Regarding MLX, unless you have a server rack at home, you probably shouldn't use local models as your primary workhorse. Personal computer class models (~30B) are not ready for this use case yet (again, vibes-based prior). They are useful as a sanity check to see how it all works end to end without spending any money. And perhaps if your website and the questions are simple enough it might perform sufficiently well.

The final output is a folder with the question, timestamp-cited answers, and the corresponding evidence. As questions and answers accumulate, this becomes a valuable corpus of data about user behavior on your site over time that future agent sessions can peruse. You can synthesize high level findings or dig all the way into individual session replays to see the evidence for yourself for a specific claim.

Don't treat models as fungible. Even new point releases in the same model family. Review your prompts and website context regularly, and spot check findings.

## Two-way doors

Back at Amazon we sorted decisions into one-way doors and two-way doors: the former get the slow, careful process, the latter you just walk through. A refinement I'd propose now that agents make the changes: a door is only two-way if an agent can be expected to walk through it with minimal supervision. Where a mistake would be subtle or the scope isn't tractable, you won't know to walk back, so it's one-way in practice.

As of Sep 2026, the one-way doors are the recorder internals and the CF stack; I wouldn't muck with them beyond the documented knobs and levers. They directly affect your customer experience and a lot can go wrong in nonobvious ways. Also the analysis engine is coupled to the recorder in terms of assuming how events are stamped and sequenced. Everything else is a two-way door.

## Caveats

You fully own and control all data and infra; there are no Locus servers to potentially mishandle your data. The trust boundaries are around CF and Gemini. This does mean you're responsible for handling your customer data appropriately (e.g. GDPR).

Costs are per use since you're directly billing with CF/Gemini. Inference costs will likely dominate total spend.

Your agent needs to be a strong model. It's responsible for orchestrating analyses, querying dbs to find relevant users/sessions, understanding your intent, and a thousand other complex things. I wouldn't use anything weaker than Claude Fable 5.1 or GPT-6 Astra.

The other core reason you need a strong agent is that every website is different, and every operator is different. You have a unique set of concerns and perspectives, and a host of gotchas specific to your site. I've tried to design the seams and configurable points as carefully as I could, and staying at the configurable layer will likely suffice for most, but no one can think through all use cases a priori. Which means your agent might have to get its hands dirty and make decisions on how to change the business logic. This requires the highest level of discernment available.

And no matter how smart your agent is, you cannot abdicate all responsibility. Some information only lives in your head and it's your responsibility to bake that into the website context and business logic. Be rigorous. Set up evals. Tune your prompts. Review every tweak your agent makes. Ask your agent for help, but any product decisions you make based on the information your agent delivers are on you.

A deeper open question I have that remains no matter how good the models get is whether session replay even has enough signal to be useful. User behavior is overdetermined, and a poor proxy for intent. If someone lingers on your checkout page, was your CTA bad? Maybe your product is simply overpriced. Maybe the user was afk. Who can say. And collecting aggregate data doesn't get you all the way out, I don't think. You can more rigorously claim that there *is* an issue with some specific flow, but you can't, in principle, know for certain *why* there's an issue.

Making your session replay corpus agent-legible unlocks many exciting possibilities for user analytics, but it remains unclear if agents armed with session replay evidence can effectively guide product decisions. It does at least enable scaling up the number of hypotheses you can A/B test. Arguably, faster iteration on clear usability issues is worth the token spend, and maybe that's all this needs to be. How to grow your business is something only you can answer ultimately, and Locus is but one tool in your arsenal.

## License

Apache 2.0 — see [LICENSE](LICENSE).
