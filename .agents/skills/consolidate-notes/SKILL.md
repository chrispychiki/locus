---
name: consolidate-notes
description: You've jotted into notes.jsonl and the observations dated past operator.md's stamp have piled up — or your jot contradicts the profile.
---

Make sure you should actually be consolidating right now. If there is a live dispute, you should be adjudicating first.

Consolidate the notes in `data/notes.jsonl` into the profile, `data/operator.md`. The profile is read before every piece of work on this deployment, and the operator never sees it, so nothing catches a bad claim until the next consolidation, in the best case. Be careful here.

Take every observation newer than the profile's stamp — the time it was last consolidated — and when you're done, set the stamp to the current time from `date -u`. If there's no profile yet, take all of them and create it.

Read the whole profile first. Don't just append a line for today's observation: ask whether the new observations change the picture, and rewrite the document as much as that needs; restructuring it entirely is fine. The notes are never trimmed, so the profile can always be rebuilt from them — nothing in the current text is precious, only the evidence behind it. Write for an agent who has never met this operator and needs to decide how to ask, what to show, and what to anticipate.

## What goes in

The profile is about how to work with this operator: how technical they are, how they think, what they care about, how they like to be answered.

Not the work in progress, the code, or anything about the system. Not anything that has a home elsewhere — the site's business frame is in `data/context/`, deployment facts are in the repo.

Go up a level of abstraction. A claim is the pattern behind the observations, not the observations: "thinks in funnels," not the list of questions that showed it.

Mark every claim: **established**, **tentative**, **unconfirmed**. The credence ladder runs on how far you had to extrapolate to get there.

- The operator literally says "I am deeply technical and I want you to speak to me in ASD-STE100". Can't get more explicit than that
- The operator often responds positively when you propose hypotheses for A/B experiments after a deep dive on a UX issue. Could be high to mid, but definitely more than a hunch
- The operator snaps at you after an answer. Maybe it really was your verbosity. Maybe you were lazy in your reasoning. Or maybe they just had a bad day. Possibly worth an unconfirmed claim, most likely nothing more

When absorbing contradictions, you must reason extremely carefully through the reconciliation and integration. People do change over time. Some kinds of changes are rarer than others. And the more likely case anyways is that you misunderstood something you codified earlier. Think deeply through recency, frequency, extrapolation distance, what context you have available, and so forth to judge what the true coherent picture is. Perhaps part of the resolution is modifying the credence on a claim. For high credence claims, look for disproofs. Is that truly the only possible or at least most likely conclusion from the evidence? Really put yourself in the operator's shoes at that instant and see out through their eyes. What are they feeling and why?

## The size limit

The profile has a hard cap of 4096 bytes. That's what forces you to integrate instead of transcribe: to add something, something else has to merge, generalize, or go. Check the byte count before you finish; if it doesn't fit, you're not done.

## Asking costs more than it looks

Don't ask the operator to confirm what you think about them — whether a hunch is right, how they'd like to be answered, which of two contradicting observations is true. That makes them do the work the profile exists to spare them; an answer to a question about themselves is weaker evidence than what you watched them do, or than what they told you unprompted; and the call is yours to make. If you're unsure, write the claim as unconfirmed. Ask only when something important depends on it and no amount of observation would settle it.
