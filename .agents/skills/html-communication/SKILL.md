---
name: html-communication
description: Sometimes communicating through a custom html page, with figures, trends, or inline session replay, is more efficient and effective than prose. If you're about to deliver a long winding info heavy answer, try delivering a visual page instead.
---

A page holds what a message can't: figures and tables, a trend, a dashboard, a session replaying beside the claim it supports, two sessions side by side, a still of the moment with the claim beside it, anything the operator wants to keep and come back to. When the operator wants to see something, the page is how they see it.

## What the page carries

A page is read later, without you there to answer questions. So it carries what you would have said if asked: every number with its denominator, every "session" or "bounce" with the definition it was counted under (`config/definitions.toml`), every sampled figure as "about", and the UTC moment the page was true as of.

The page speaks the operator's words: sites by domain (`locus status` names each snippet's), visitors by when and what they did. A visitor or slice id is a citation — small, beside the thing it locates — never a heading. Prose on the page points ("watch the third tap"); it never narrates what plays beside it. A chart earns its place only when a shape says what a sentence can't — a minimal hand-drawn mark on a quiet surface, so the replays and numbers stay loudest.

## Building it

A page opens from a local file, where nothing can fetch and no CDN loads. Everything is inline: the numbers as literals, styles and scripts in the file, charts drawn by hand as inline SVG. No frameworks, no build step.

So you build the page in code: a script, run with `uv run python` at the deployment root, that queries the db, does the arithmetic, and writes the HTML to `data/pages/` under a name you choose. The numbers come from the db and the moments from the replays, verified by your own eyes on the player; a model read is for what those can't settle, so start one only when the page needs it — a run you start, you wait for, and a run you've stopped needing, you stop. (`locus browse open` composes its default replay page there at a derived name and refreshes it on every open; yours gets its own name.)

A replay on the page needs two scripts beside it: the self-contained player component, and the slice set's events as a payload. Both come from `locus.evidence.replay` — `materialize` writes them into `data/pages/` and returns the paths; the payload refuses a set no single player could honestly play. Read that module's header and the component's (`data/pages/locus-replay.js`), which is the composition contract: load the component, load the payload, mount. What follows is what trips a first page:

- A payload is one visitor's time-disjoint slices on one player. Two visitors, or two tabs open at once, are separate payloads on separate mounts.
- The URL fragment (`#t=…`) seeks every mount on the page on the shared absolute clock. That's what you want for one visitor's lanes; for two unrelated sessions side by side, name the mounts (`mount(el, replay, {name})`) and address one with `&m=<name>`, or drive each through the handle `mount()` returns.
- A payload is the whole event stream, so a page with many players is heavy. Embed the moments that matter; a table of sessions doesn't need a player per row.
- Where a live player is more than the point needs, a still does: open a replay page at the moment and `locus browse screenshot` that window, or a screenshot the analysis already took (`screenshots/` in its directory), embedded by relative path with a link to the replay page at `#t=`.

## Look before you show

Source is text; the page is what it renders to, and nothing about that is in the file — a collapsed mount, a chart that reads wrong, a replay opened at the wrong instant, the eye landing on the wrong number. So the page is built by looking: `locus browse open <page path>` opens it in a window and prints that window's id; screenshot the window, fix the code, open it back into the same window, until what is on the screen is what you meant.

Look at the whole page first, then at each thing on it: every replay at its cited moment (`#t=` on the page's URL), the range you marked, a chart at the size it will actually be read at. `locus browse read` and `eval` on the window settle what a screenshot can't — whether a mount exists, what a player's current time is. Scroll; a page longer than a screen has a below-the-fold you haven't seen.

A window opens behind everything else, and the operator has seen nothing until `locus browse show <window>` — the delivery: it lands the page where their eye should land (the top, or a ref you name) and raises it. The operator is at the screen, and a page not shown was never delivered. A page is a file, so a later change is rewriting it and opening it back into the same window.
