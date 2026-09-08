"""Locus analysis — stored sessions to grounded findings.

The spine, in the order the data flows:

    window           the resolved evidence window: slice labels, resolution walls, citation validation
    session_context  session-context blocks from the flat columns
    select_screenshots    mechanical screenshot-timestamp selection per slice
    analyze          event-stream composition: events + screenshots -> the window's evidence blocks
    prompts          every word the model reads — the editable policy surface
    budget           composition-faithful model-payload pricing, tokens counted live
    engine           the analysis loop: price -> render -> converse -> answer artifacts
    model/           provider-agnostic conversation layer (model calls, persisted payloads)
    spend            the operator's walls and ledger over every billed call
    ground/, oracle  a parked proof of concept: offline grounding measurement
                     over a finished analysis, invoked by nothing

Beside the spine: cli (the `locus` verbs over both packages), ae (the `locus ae`
result envelope), doctor (mechanical integrity), mlx/ (the bundled local server,
an optional extra).
"""
