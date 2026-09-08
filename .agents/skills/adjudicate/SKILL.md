---
name: adjudicate
description: A delivered claim is in doubt — the operator disputes it, or you've caught the evidence misrepresenting a session yourself.
---

Take the operator's word on their own business — what the site is for, what their terms mean. On what actually happened in a session, check before you agree, however sure they sound and however small the resulting change would be.

A disputed claim may be one cited claim from one analysis, or a conclusion you built from many supports — model claims, query results, your own reasoning. Break it into its supports and check each one the way it was made: re-run the queries, redo the reasoning, and for a model's cited claim, compare the cited moments against what the claim says.

That comparison is a lookup. `window.json` in the analysis directory maps each citation to its slice and absolute moment; use it rather than re-deriving the clock. `locus browse open` on the analysis directory lands on any cited moment in a window, and the persisted outputs and screenshots show what the model said and saw. If an activity-screenshot for the moment in question is missing, capture it yourself with `locus browse screenshot <window>`. An analysis outlives the recording it cites, so an old enough finding opens onto nothing: the retention horizon took the recording, and the analysis's own transcripts and screenshots are then the whole of what can be checked — say that plainly rather than treating the gap as a fault, and judge the claim on what the model was actually shown.

Some possible verdicts:

- **The claim is right.** Sometimes the site really did do that.
- **The evidence misrepresented the session.** The event stream doesn't match what the replay shows, or a relevant screenshot was missing.
- **The evidence was faithful and the model misread it.** The model simply isn't capable enough, or a bad prompt steered it wrong.
- **The analysis was constructed badly.** The question was vague, or the window didn't include the relevant slice(s).
- **Your synthesis was wrong.** The supports hold and the conclusion built on them doesn't.
- **The operator's inputs steered it.** The site context or a definition is stale or wrong, and the model read faithfully under the bad frame.
- **The recording can't answer it.** Masking withheld the content, capture never got the moment, or the claim is an inference the evidence can neither prove nor refute.

Walk the operator through your verdict and how you arrived at it. Present replays if relevant. Discuss what the ground truth is and whether it can even be determined before proposing any changes.

## The bar for a change

A verdict either cashes out to a change or it doesn't. You should at least jot a note down, but anything more needs strong positive justification. One ostensible instance plus a plausible story is a hypothesis, not proof. What is the failure's blast radius? Is it reproducible? Are there confounding variables? Can you confirm the causal link between the suspected mechanism and the wrong claim?

Don't jump to conclusions. What assumptions are you implicitly making here? Are they all warranted? Comb through the data and evidence and carefully consider each logical step in your proof. Explicitly look for disproofs in the corpus.

Once you've established a compelling case for making a change, consider what sort of change is needed to address the true root cause. Do we need to change how the evidence is composed? Do we need to change the context the model receives? Both? Something else? Take your time. A subtly wrong "fix" can be nearly impossible to debug down the road.

Read the relevant code/docs/prose in `evidence/` or `analysis/` carefully to understand the nuances of the current system and how to surgically change it. Any change you make must be rigorously tested against actual, not synthetic, data. Change one thing at a time and prove it.

Beyond correctness, deeply consider the performance and architecture ramifications of your changes. Jumping to the naive unconsidered patch that only solves the immediate problem is a tax every future session pays. Latency, especially with common operations, subtly (or not so subtly) degrades the operator's experience. Cruft and hacks will make the subsequent changes exponentially more intractable. Instrument and time the performance where applicable, and approach architecture level changes with a craftsman's eye. Remember, you need to understand the system deeply before touching any aspect of it.

## Changing what the evidence can show

The shipped composition defaults are general and may work poorly for the operator's site:

- a cap too small for the site's real text
- a projection blindness eliding important site semantics, or a delta granularity that misses what actually mutates here
- an activity-screenshot heuristic that misses what matters on this site

Or the issue might be a replay-fidelity limit the site keeps hitting — the replay can differ from what the visitor saw. `render.py`'s header names the known limits; the levers that exist are record-time capture config, so they help future recordings only.

A distillation change moves the derivation vintage, and the next `locus load` or `locus doctor` re-derives the db through the new path, which is when the fix takes effect.

## Changing the model-facing prose

The model-facing prose is the engine's standing instructions in `analysis/src/locus/analysis/prompts.py`, the operator's site context at `data/context/<snippet>.md`, and the recording-facts and session-context blocks composed in `analysis/`, so a change to any of them moves every future answer on this deployment.

- Don't patch the incident in as an exception. Exceptions accumulate, and the pile overfits even within one corpus. Go up a level of abstraction, find the general reading principle the failures share, state it once, and delete the patches it makes redundant.
- Write for the reader the prompt has: a fresh model that has only ever seen the current prompt. A ghost rebuttal — a line arguing with a version of the prompt that no longer exists, "not X" where only deleted text ever said X — is noise to that reader, and can plant the very X it argues against.
- A line earns its place by changing how the model reads the evidence, for the better. Your reasons for a rule and your knowledge of how the system is built are not instructions; they go in only where they change that reading.
- Chesterton's fence: assume every existing line is load-bearing; behavior was measured with it in place. Know why a line is there before weakening or removing it.
- Spend emphasis sparingly; a prompt where everything is critical has no priorities.
- Brevity is the soul of wit: no hedge stacks, no editorializing, no self-narration, no contrastive parallelism, no ungrounded speculation, no redundant appositives, no intra-doc cross references, no throat clearing, etc.
