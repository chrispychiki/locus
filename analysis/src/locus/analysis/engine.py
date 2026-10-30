"""The analysis engine — the control flow of one analysis.

Every word the model reads besides the evidence and the question is
prompts.py's. How densely the activity-screenshots sample the session defaults
to select_screenshots.SCREENSHOT_INTERVAL_MS, overridable per run on `locus analyze`.

The question is never the engine's. The caller composes it entire —
`run_analysis(question=...)` — and the engine supplies only the evidence service
around it: composition, the pushed activity-screenshots, on-demand pulls,
persistence, and the citation contract that makes the answer verifiable. The
contract and the pull tool are one text and one shape for every window
(prompts); the window's slice count decides only what the engine accepts —
a bare offset in a one-slice window, a labeled one in a window of several.

It is control flow over modules whose logic stays in their own homes:
  - render.RenderSession — the painting browser, a service opened by a capture
    and released the moment a round of captures is done, so an analysis waiting
    on the model holds no browser; refuses a non-self-covering slice (the
    replayability reject rule) and reports per-screenshot render faults.
    Every screenshot is addressed by its slice's label — slices may overlap in
    time, so a timestamp alone routes nowhere.
  - analyze.compose_window — assembles the evidence into the conversation: the
    per-visitor recording-facts and session-context blocks, the event stream
    interleaved with the screenshots, every slice on the one shared window
    clock. The engine renders the activity-screenshots and hands them in, so
    render moves inside the control flow while the composition stays in its one
    home.
  - budget.price_payload — the same composition counted, never sent. Every
    model-payload is priced before anything runs: the default invocation
    returns the price and stops, and a run refuses loud when the priced payload
    does not fit the card's context — nothing is cut or split on the caller's
    behalf.
  - model — the conversation transport (tool calls, payload record). What an image
    costs and how much context is left are the backend's to answer, exactly, on
    the conversation as it stands; the engine serves the screenshots that fit.

Runtime: compose the window with the pushed activity-screenshots; the model
calls request_screenshots at the moments they need detail on — pushed or not —
which the renderer produces fresh on demand and the engine feeds back inside the
tool's reply, at the card's detail tier, with the question after them; the next
turn is the timestamp-cited answer.
Before anything ships, the answer is validated mechanically — every citation
against the citation format, the window's labels, and its bounds (window.citation_violations),
and an empty answer as the same kind of defect — and a defective answer goes
back once with the defect named; what that one repair turn returns is the
deliverable, residual citation defects and all — a completed analysis is never
destroyed over them: an answer whose repair turn would not fit the context ships
as written, and only a run that never returns an answer at all errors. Whether
a valid citation supports its sentence is judgment, and none lives here: the
operator's dispute path owns that, offline.
Screenshots live and die inside this call; the return value is a manifest of
paths and counts — no image bytes cross back to the caller. One invocation is
one window, one conversation, one analysis directory: window.json (the machine
manifest — question, clock anchor, slice table), the full wire transcripts (the
final answer-labeled response is the answer, thoughts and the contract's meta
section in place as the wire returned them), and screenshots/. The directory is the
evidence. Nothing is assembled: the record is the deliverable, and the manifest
names the final response's path.
"""

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from locus.evidence.render import RenderSession
from locus.evidence.speak import beating

from .analyze import compose_window
from .budget import price_payload
from .model.cards import declared_pricing
from .model.factory import make_conversation
from .model.protocol import latest_answer_response
from .prompts import (
    EMPTY_ANSWER_REPAIR,
    REPAIR_NEEDS_NO_SCREENSHOTS,
    SCREENSHOTS_ALREADY_SERVED,
    question_task,
    repair_task,
    reply_block,
    request_screenshots_tool,
    split_meta,
    system_prompt,
)
from .select_screenshots import SCREENSHOT_INTERVAL_MS, screenshot_moments
from .spend import check_walls
from .window import citation_violations, format_offset

# One round of pulls is served for the posed task; after that the model is
# refused and re-asked, and a model that keeps asking is failed rather than
# looped forever. Invalid requests are bounded the same way: each is failed
# back with its defects and retried fresh — only a served round spends the
# one round.
MAX_REFUSAL_ROUNDS = 2
MAX_INVALID_ROUNDS = 2


class PayloadOverflow(ValueError):
    """An analysis whose priced turn-1 model-payload does not fit the model's context. Carries the full price report so the caller can whittle from it."""

    def __init__(self, price: dict):
        self.price = price
        turn1 = price["turn1"]
        super().__init__(
            f"the composed model-payload is {turn1['total_tokens']} tokens "
            f"({turn1['context_pct']}% of {price['model']}'s context) — over "
            f"the fit budget of {price['budget_tokens']}. Narrow the slice "
            f"set, widen --screenshot-interval, or name one of an oversized "
            f"slice's route-boundary pieces; the price report carries the "
            f"per-slice and per-piece numbers"
        )


def _contract(labels: list[str], question: str) -> tuple[str, str, str]:
    """(system prompt, turn-1 task, answer-turn closing). The texts are the same for every window; `labels` is the price instrument's per-set callback signature (budget.price_payload), so every measured set composes through this one path."""
    return (
        system_prompt().strip(),
        question_task(question, reply_block(pull=True)),
        reply_block(pull=False),
    )


def price_analysis(
    conn: sqlite3.Connection,
    window,
    *,
    question: str,
    model: str,
    site_contexts: dict[str, str],
    screenshot_interval_ms: int = SCREENSHOT_INTERVAL_MS,
) -> dict:
    """The model-payload priced, exactly, spending nothing — the default `locus analyze` invocation, over a resolved window (window.resolve_window). The counted payload is byte-for-byte what a run would send: the same contract, composition, activity-screenshots, question, and tool declaration."""

    def prompts(labels: list[str]) -> tuple[str, str, list[dict]]:
        system, task, _ = _contract(labels, question)
        return system, task, [request_screenshots_tool()]

    return price_payload(
        conn,
        window,
        model=model,
        prompts=prompts,
        site_contexts=site_contexts,
        screenshot_interval_ms=screenshot_interval_ms,
    )


def _parse_requests(
    response, *, by_label, labels, multi, start_ms
) -> tuple[list, list]:
    """The response's screenshot requests, validated mechanically: (requested (label, moment-offset) pairs, defects). The arguments are the model's own writing, not schema-guaranteed on every backend — a call can even carry argument text that is no JSON object at all (the backend marks those arguments None) — so every request is held to what the window can actually serve: parseable arguments, an integral timestamp, a label from the window's own roster, a moment inside that slice's recording. Each failure is a defect rendered in window-offset vocabulary, ready to fail the call in the tool's own channel rather than crash an analysis minutes in or serve a moment the model never meant."""

    def moment_of(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    roster = ", ".join(by_label)
    candidates, defects = [], []
    for call in response.tool_calls:
        if call.arguments is None:
            defects.append("a request's arguments were not a JSON object")
            continue
        entries = call.arguments.get("screenshots")
        for entry in entries if isinstance(entries, list) else [None]:
            label = entry.get("label") if isinstance(entry, dict) else None
            moment = (
                moment_of(entry.get("timestamp")) if isinstance(entry, dict) else None
            )
            if moment is None:
                defects.append("a timestamp was not an integer")
            elif multi and label is None:
                defects.append(f"a moment has no slice label; the slices are {roster}")
            elif multi and label not in by_label:
                defects.append(
                    f"label {label!r} is not one of the recording's ({roster})"
                )
            elif not multi and label is not None:
                defects.append(
                    f"a moment carries a slice label ({label!r}), but this recording has one slice, so moments are bare times"
                )
            else:
                candidates.append((label if multi else labels[0], moment))

    requested = []
    for label, off in candidates:
        s = by_label[label]
        if off < 0:
            defects.append(f"a timestamp was negative ({off})")
        elif s.start_ts <= start_ms + off <= s.end_ts:
            requested.append((label, off))
        else:
            stamp = format_offset(start_ms + off, start_ms, label if multi else None)
            where = label if multi else "the recording"
            span = (
                f"{format_offset(s.start_ts, start_ms)} to "
                f"{format_offset(s.end_ts, start_ms)}"
            )
            defects.append(f"{stamp} is outside {where}, which runs {span}")
    return requested, defects


def _serve_round(
    conv,
    renderer,
    response,
    requested,
    *,
    by_label,
    multi,
    start_ms,
    resume_task,
    on_screenshot=None,
) -> tuple[list, int]:
    """The one round served for the posed task: render every requested screenshot as a pull, at the conversation's pulled size, and answer the tool calls with every one the remaining context holds — the screenshots ride inside the reply to the call, each under its moment, followed by the task to resume, so the model reads them as the answer to its own call. `requested` is _parse_requests' validated pairs — every moment already inside its slice's recording. Returns (the served screenshots as (label, ts, screenshot) triples, refused count).

    A pull rendered but refused for lack of context is deleted: the analysis directory is the record, and a screenshot on disk that the model was never shown would read as evidence to anything that reads the record, the oracle's judge included.

    `on_screenshot`, when given, is called with the running count as each screenshot lands, so a caller's pulse moves through the rendering."""
    # A pull is a purchase of detail: a moment already pushed at the
    # cheaper tier returns at the detail tier, so re-requests of pushed
    # moments are served. Only a duplicate within this round is
    # dropped — the same moment twice at the same tier buys nothing.
    # Screenshots go back in time order, the order the event stream reads in.
    # A moment in a slice's lead — inside its recording but ahead of its
    # covering snapshot, where no DOM was yet captured — serves the snapshot's
    # own instant, the first the screenshot plane holds, and the serving line
    # stamps the moment actually shown.
    served, wanted, duplicate = set(), [], 0
    for label, off in sorted(requested, key=lambda r: (r[1], r[0])):
        s = by_label[label]
        ts = max(s.screenshot_start_ts, start_ms + off)
        if (label, ts) in served:
            duplicate += 1
        else:
            wanted.append((label, ts))
            served.add((label, ts))

    def pulled_size(width: int, height: int) -> tuple:
        return conv.screenshot_size(width, height, pushed=False)

    rendered = []
    try:
        for label, ts in wanted:
            rendered.append(
                (
                    label,
                    ts,
                    renderer.capture(label, ts, kind="pulled", size=pulled_size),
                )
            )
            if on_screenshot is not None:
                on_screenshot(len(rendered))
    finally:
        renderer.release()

    def note(n: int) -> str:
        refused = len(rendered) - n
        if not n:
            return "No screenshots could be sent — no room left in the context. Answer from what you have."
        text = f"Rendered {n} screenshot{'' if n == 1 else 's'}, each below under its moment."
        if duplicate:
            text += " A moment requested more than once was rendered once."
        if refused:
            text += f" {refused} more could not be sent — no room left in the context; answer from what you have."
        return text

    def round_parts(n: int) -> list[tuple]:
        """The round as it is appended when n of the rendered screenshots are served: every call answered, the last one's reply carrying the note, the screenshots under their moments, and the task to resume — calls made together are served together."""
        *together, last = response.tool_calls
        reply = [("text", note(n))]
        for label, ts, screenshot in rendered[:n]:
            stamp = format_offset(ts, start_ms, label if multi else None)
            reply += [
                ("text", f"\nscreenshot at {stamp}:\n"),
                ("image", screenshot.path),
            ]
        reply.append(("text", f"\n{resume_task}"))
        pointer = (
            "Served together with this turn's other request_screenshots calls — "
            f"the reply to call {last.id} holds the screenshots."
        )
        return [("tool", call.id, pointer) for call in together] + [
            ("tool", last.id, reply)
        ]

    # The context is the only cap, and here it is exactly knowable: the backend
    # counts the next request with the whole round in place — the tool answers,
    # the labels, the screenshots and the task — so the round serves the most
    # screenshots whose turn fits.
    served_n = len(rendered)
    if conv.context_remaining(round_parts(served_n)) < 0:
        fits, over = 0, served_n
        while over - fits > 1:
            mid = (fits + over) // 2
            if conv.context_remaining(round_parts(mid)) >= 0:
                fits = mid
            else:
                over = mid
        served_n = fits

    screenshots = rendered[:served_n]
    for _, _, screenshot in rendered[served_n:]:
        Path(screenshot.path).unlink()
    for _, call_id, content in round_parts(served_n):
        conv.add_tool_result(call_id, content)
    refused = len(rendered) - served_n
    return screenshots, refused


def _pose(
    conv,
    renderer,
    *,
    what,
    open_task,
    resume_task,
    screenshots_tool,
    by_label,
    labels,
    multi,
    start_ms,
    out_dir,
):
    """Pose the task and come back with what the model wrote: an invalid request — malformed, an unknown label, a moment outside its slice's recording — is failed in the tool's own channel with each defect named and retried fresh, and only a served round spends the task's one round; after it, further requests are refused, and a model that never writes is failed rather than looped forever. Returns (response, the served screenshots, refused).

    `what` names the work for its pulse, whose detail is the round-trips made, the screenshots served, and where the current call stands as the backend reports it (waiting at the server, prefilling, generating)."""
    done = {"calls": 0, "screenshots": 0}

    def send(task=None):
        if task is not None:
            conv.add_user_text(task)
        done["calls"] += 1
        return conv.get_response(label="answer", tools=[screenshots_tool])

    def served(n: int) -> None:
        done["screenshots"] = n

    def detail() -> str:
        line = (
            f"{done['calls']} round-trip(s), {done['screenshots']} screenshot(s) served"
        )
        standing = conv.progress_line()
        return f"{line}; {standing}" if standing else line

    with beating(what, detail):
        response = send(open_task)
        screenshots, refused = [], 0
        invalid_rounds = 0
        while response.tool_calls:
            requested, defects = _parse_requests(
                response,
                by_label=by_label,
                labels=labels,
                multi=multi,
                start_ms=start_ms,
            )
            if not defects:
                break
            # The whole call fails as a unit — nothing served, the one round
            # unspent — so the retry is a fresh request, not a spent budget.
            invalid_rounds += 1
            if invalid_rounds > MAX_INVALID_ROUNDS:
                raise RuntimeError(
                    f"model kept requesting screenshots at invalid moments "
                    f"after {MAX_INVALID_ROUNDS} corrections — see the "
                    f"transcript in {out_dir}"
                )
            note = "No screenshots — the request was invalid:\n"
            note += "\n".join(f"- {defect}" for defect in defects)
            note += (
                "\nCall request_screenshots again with corrected moments, "
                "or write your answer."
            )
            for call in response.tool_calls:
                conv.add_tool_result(call.id, note)
            response = send()
        if response.tool_calls:
            screenshots, refused = _serve_round(
                conv,
                renderer,
                response,
                requested,
                by_label=by_label,
                multi=multi,
                start_ms=start_ms,
                resume_task=resume_task,
                on_screenshot=served,
            )
            # The tool stays declared for the rest of the conversation: the
            # declaration renders ahead of every message (Qwen's chat template
            # puts it in a system block), so withdrawing it would rewrite the
            # head of a conversation whose every request otherwise strictly
            # extends the last — the shape Gemini's implicit caching prices
            # and any server-side prefix cache reuses on. The answer is forced
            # by the task the reply carried; a call made anyway is failed in
            # the tool's own channel.
            response = send()
            refusals = 0
            while response.tool_calls:
                refusals += 1
                if refusals > MAX_REFUSAL_ROUNDS:
                    raise RuntimeError(
                        f"model requested screenshots again after the served round "
                        f"and {MAX_REFUSAL_ROUNDS} refusals without writing an "
                        f"answer — see the transcript in {out_dir}"
                    )
                for call in response.tool_calls:
                    conv.add_tool_result(call.id, SCREENSHOTS_ALREADY_SERVED)
                response = send()
    return response, screenshots, refused


def home_analysis(out_dir: str | Path) -> Path:
    """The analysis's directory, made empty and refusing to overwrite: its contents are the evidence for a finding, so a rerun makes a new one. Made before anything is decided, so the caller has the address to speak into while the wall and the price run."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(
            f"output dir already exists: {out_dir} — outputs are never overwritten"
        )
    out_dir.mkdir(parents=True)
    return out_dir


def run_analysis(
    conn: sqlite3.Connection,
    window,
    *,
    question: str,
    model: str,
    site_contexts: dict[str, str],
    out_dir: str | Path,
    screenshot_interval_ms: int = SCREENSHOT_INTERVAL_MS,
) -> dict:
    """One analysis, into a directory home_analysis already made: one conversation over the caller's resolved window (window.resolve_window) and entire question, one cited answer with full provenance.

    Everything that decides whether there is an analysis at all runs first, before anything is composed or a screenshot rendered.
    The model-payload is priced and refused loud (PayloadOverflow, price attached) when it does not fit the card's context — nothing is ever cut or split for the caller.
    An analysis on a priced model also stands under the deployment's declared spend caps (config/spend.toml): the wall is enforced at the conversation layer before every billed call, and asked once up front here so a capped deployment refuses in milliseconds instead of after minutes of rendering.

    The directory holds window.json (the question, the window's absolute clock anchor, the slice table), the wire transcripts — the final answer-labeled response is the answer, thoughts and the contract's meta section in place as the wire returned them — and screenshots/. The model reads one shared window clock; a pulled moment is a window offset plus — when the window is multi-slice — the label of the slice it belongs to, resolved through the slice table. A request outside its slice's recording is failed back to the model in the tool's own channel; a moment inside the recording but ahead of the covering snapshot — where no DOM was yet captured — serves the snapshot's own instant, stamped with the moment actually shown."""
    if declared_pricing(model):
        check_walls()
    price = price_analysis(
        conn,
        window,
        question=question,
        model=model,
        site_contexts=site_contexts,
        screenshot_interval_ms=screenshot_interval_ms,
    )
    if not price["turn1"]["fits"]:
        raise PayloadOverflow(price)

    out_dir = Path(out_dir)
    table = list(window.slices)
    labels = window.labels
    by_label = {s.label: s for s in table}
    label_of = {s.slice_id: s.label for s in table}
    multi = len(table) > 1
    system, turn1_task, closing = _contract(labels, question)
    screenshots_tool = request_screenshots_tool()

    pushed = screenshot_moments(
        conn, window, screenshot_interval_ms=screenshot_interval_ms
    )
    # The conversation records its transcript in the analysis dir, beside the
    # output. It is made before rendering because it sets every screenshot's size.
    conv = make_conversation(model, system, record_dir=out_dir)

    def activity_size(width: int, height: int) -> tuple:
        return conv.screenshot_size(width, height, pushed=True)

    with RenderSession(
        conn, {s.label: s.slice_id for s in table}, out_dir / "screenshots"
    ) as renderer:
        moments = [
            (label_of[sid], ts)
            for sid, timestamps in pushed.items()
            for ts in timestamps
        ]
        pushed_screenshots: dict = {}
        with beating(
            "rendering activity-screenshots",
            lambda: f"{len(pushed_screenshots)}/{len(moments)} rendered",
        ):
            try:
                for label, ts in moments:
                    pushed_screenshots[(label, ts)] = renderer.capture(
                        label, ts, kind="activity", size=activity_size
                    )
            finally:
                renderer.release()
        pushed_paths = {key: Path(s.path) for key, s in pushed_screenshots.items()}

        composed = compose_window(
            conn,
            conv,
            window,
            site_contexts=site_contexts,
            screenshots=pushed_paths,
        )
        start_ms = composed.window_start_ts
        # The record lands before the first call, so a run that dies mid-conversation
        # still leaves what its transcripts resolve through; what the run found joins
        # it at the end.
        manifest = {
            "question": question,
            "model": model,
            "window_start_ts": start_ms,
            "window_start": window.window_start,
            "window_end": window.window_end,
            "slices": [asdict(s) for s in table],
            "n_events": composed.n_events,
            "n_activity_screenshots": composed.n_screenshots,
            # Per-screenshot provenance: which moments rode as activity-screenshots
            # and which as served pulls, so a reader of the record (the oracle's
            # judge, holding to parity) knows what kind each screenshot file was,
            # not just how many.
            "activity_screenshots": {
                label_of[sid]: timestamps for sid, timestamps in pushed.items()
            },
        }
        (out_dir / "window.json").write_text(json.dumps(manifest, indent=2))

        response, pulls, refused = _pose(
            conv,
            renderer,
            what=f"asking {model}",
            open_task=turn1_task,
            resume_task=question_task(question, closing),
            screenshots_tool=screenshots_tool,
            by_label=by_label,
            labels=labels,
            multi=multi,
            start_ms=start_ms,
            out_dir=out_dir,
        )

        # Validation is mechanical — format, label, bounds, and an empty
        # answer is the same kind of defect — and the repair turn hands back
        # exactly the defect.
        # One repair turn, and what it returns is the deliverable: a completed
        # analysis is never destroyed over residual citation defects — they
        # stand in the record as what they are. Only a run left with no
        # answer at all fails, below. Whether a valid
        # citation supports its sentence is judgment, and it lives offline
        # (the operator's dispute path).
        answer = split_meta(response.text)[0].strip()
        if not answer:
            task = EMPTY_ANSWER_REPAIR
            repairing = f"{model} wrote no answer; asking once more"
        else:
            violations = citation_violations(answer, table, start_ms)
            task = repair_task(violations, labels) if violations else None
            repairing = f"{model} repairing {len(violations)} invalid citation(s)"
        # A repair turn the context has no room for is not sent, and the
        # answer ships as written.
        refused_repair = bool(task) and conv.context_remaining([("text", task)]) < 0
        if refused_repair and not answer:
            raise RuntimeError(
                f"engine produced an empty answer, and no room is left in the "
                f"context to ask again — see the transcript in {out_dir}"
            )
        if task and not refused_repair:
            with beating(repairing, conv.progress_line):
                conv.add_user_text(task)
                response = conv.get_response(label="answer", tools=[screenshots_tool])
                refusals = 0
                while response.tool_calls:
                    refusals += 1
                    if refusals > MAX_REFUSAL_ROUNDS:
                        raise RuntimeError(
                            f"model requested screenshots on a repair turn "
                            f"{MAX_REFUSAL_ROUNDS} times over without writing "
                            f"an answer — see the transcript in {out_dir}"
                        )
                    for call in response.tool_calls:
                        conv.add_tool_result(call.id, REPAIR_NEEDS_NO_SCREENSHOTS)
                    response = conv.get_response(
                        label="answer", tools=[screenshots_tool]
                    )

        pulled = [screenshot for _, _, screenshot in pulls]
        pulled_moments = [[label, ts] for label, ts, _ in pulls]

    # The deliverable must contain an answer: a reply that is only the
    # contract's meta section answered nothing, and the repair turn was its
    # one chance — with nothing to ship, the run errors.
    if not split_meta(response.text)[0].strip():
        raise RuntimeError(
            f"engine produced an empty answer, and its repair turn returned "
            f"none either — see the transcript in {out_dir}"
        )

    manifest = {
        **manifest,
        "pulled": pulled_moments,
        "n_pulled_screenshots": len(pulled),
        # What the readiness wait could not deliver, by screenshot file, failed
        # resources named by URL — so a reader judging a screenshot can tell a
        # replay artifact from what the visitor really saw. A screenshot that
        # rendered clean has no entry.
        "render_faults": {
            Path(f.path).name: f.faults
            for f in [*pushed_screenshots.values(), *pulled]
            if f.faults
        },
        "refused_screenshots": refused,
        # A repair was owed but the context had no room for it, so the
        # answer's citation defects stand unrepaired.
        "refused_repair": refused_repair,
    }
    (out_dir / "window.json").write_text(json.dumps(manifest, indent=2))

    # The answer is the record itself: the final answer-labeled response
    # transcript, already persisted wire-exact by the conversation layer;
    # window.json carries the question, clock anchor, and slice table beside it.
    final = latest_answer_response(out_dir)
    return {**manifest, "dir": str(out_dir), "response_path": str(final)}
