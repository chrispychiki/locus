"""The grounded-reader contract — the standing instructions that hold for every analysis.

Every sentence in the RECORDING_FORMAT block is a claim about what the other modules actually produce: the page projection and its diff (evidence's distill/project.js and diff.js), the event stream (analyze.py), the screenshot moments (select_screenshots.py), and the replay cursor the renderer paints (render.py, whose harness colors it while the mouse is down). Change one of those and this block is what has to follow.

The question itself is never here. It is the caller's entire, sent last on whichever turn produces the answer (question_task), after the composed evidence — so everything before it is a byte-stable prefix a cache can reuse across different questions over the same evidence. What this module owns is everything the model reads besides the evidence and the question: the contract that makes any answer usable — how the evidence reads, that every factual statement cites the moment it rests on — the reply-format block that rides every fresh writing turn (the citation demand restated, the feedback channel addressed to whoever tunes the system rather than to the asker), the screenshot-request block beside it while the task's one request stands, the repair task that hands back invalid citations, and the request_screenshots tool schema. split_meta is the channel's single splitting rule, keyed on the exact header the contract demands: the oracle excludes the channel before extraction, and the engine refuses a reply with no answer above it.

The standing texts are one text for every window. The citation format has two cases — a bare offset in a window of one slice, the slice's label first in a window of several — and both are stated every time, the model telling them apart by the SUMMARY's Slices line, which a window prints exactly when it has more than one slice. The repair ask is addressed to one window's citations and names that window's form.
"""

from .window import citation_forms

META_HEADER = "## Ambiguities and Feedback"

CITE_RULE = (
    "cite every claim inline with the time it happened; the reader will open "
    "the replay at that time to check it. A moment is [MM:SS.mmm], a span is "
    "[MM:SS.mmm, MM:SS.mmm]; when the SUMMARY lists slices the label comes "
    "first, [S1 MM:SS.mmm] and [S1 MM:SS.mmm, MM:SS.mmm]; a bracket holds one "
    "moment or one span, nothing else. Brackets only — never bold, backticks, "
    "or parentheses"
)

_CITATIONS = """<CITATIONS>
    Cite every claim about the session inline with the time it happened in the recording. The reader will open the replay at that time to check the claim.
    - A claim about one moment cites [MM:SS.mmm]; a claim about a stretch of time cites the span, [MM:SS.mmm, MM:SS.mmm]. A bracket holds one moment or one span, nothing else.
    - Times count from the start of the recording, on one clock across all its slices.
    - When the SUMMARY lists slices (S1, S2, …), the slice's label leads: [S1 MM:SS.mmm], [S1 MM:SS.mmm, MM:SS.mmm]. With no Slices line, the bare forms.
    - Brackets only. Never bold, backticks, parentheses, or any other form.
</CITATIONS>"""


def system_prompt() -> str:
    """The contract: what the evidence is and is not, the citation format, and how the evidence reads."""
    return f"""
<TASK>
    You are answering a question about a recording of a visitor on a website. The recording shows what the visitor did, not why. Anything you say about what they wanted or felt is an inference from their actions, so word it as one. If the visitor did little or nothing, say so. If the recording cannot settle something the question asks, say so rather than guess.
</TASK>

{_CITATIONS}

<RECORDING_FORMAT>
    - A FullSnapshot's body is the page as markdown — the document, not the screen: most of it sits below the viewport, and content a stylesheet hides reads the same as content in view; what was on screen is the screenshots' testimony. Controls are bracketed with their label, value, and options.
    - label="…" is the page's name for a control that shows no text; those words were not on screen. {{…}} is state the page declares — expanded, selected, disabled, open.
    - A value cut at a ceiling ends in … with (first N of M chars); its line breaks are written \\n.
    - A Mutation's body is a diff of that markdown. Content already shown is not reprinted: [N chars: 'head' … 'tail'] stands for a removed line or a value shown before, a count line for a run of removed or repeated lines, … inside a changed line for what it shares with its previous form. Events that changed nothing in the page text are counted, not shown; the screen may still have changed.
    - A mutation is the document changing, not the visitor seeing it: pages add content out of view, ahead of need. Where the events and a screenshot disagree about what was on screen, the screenshot wins; note the disagreement in the "{META_HEADER}" section.
    - An Input line is a displayed field's value changing — typed, pasted, or set by the page. Consecutive Inputs on one field print only the change, @offset 'before'→'after', with the whole value on the first and last of the run.
    - The recording is split into slices: a new one begins at each page load, and periodically during a long stay on one page.
    - A second TouchStart before a TouchEnd is a second finger.
    - MouseMove and Scroll are sampled several times a second; a gap shorter than a second between them is not a pause.
    - Screenshots fall at a fixed interval wherever the visitor was active, plus each slice's first captured moment and its last. A slice's first events can precede its first capture, so they have no screenshot.
    - A mouse pointer is drawn on the screenshots, red while the button is down, and sits top-left when its position for that frame is unknown. A touch recording draws no pointer; a ring marks where a finger is down.
    - More than one visitor: each recording follows its own SUMMARY block naming the visitor. Keep them separate.
</RECORDING_FORMAT>
"""


_FEEDBACK = f"""After the answer, add one last section headed exactly "{META_HEADER}". It is read by the engineers who tune this system, not by the asker. It holds two things: what the question needed that the recording cannot settle even after looking, with what would settle it; and anything in these instructions or the recording that got in your way. Everything for the asker goes above it. Omit the section if there is nothing to say."""

_PULL = """The screenshots above are a sample; you can ask to see any moment, including one already shown. If seeing a moment would settle something you would otherwise guess at or file as an ambiguity, call request_screenshots now with every moment you want, before writing — only one request is served. You will be shown them and asked again. Otherwise, write your answer."""


def reply_block(*, pull: bool) -> str:
    """The blocks that close every turn that writes an answer fresh. REPLY_FORMAT restates the citation demand at the moment of writing — the system prompt states it once at the top of a long context, and a model answering a sparse window drops it from there — and asks for the feedback section. SCREENSHOT_REQUESTS rides beside it while the task's one request is unspent. The citation format is pinned because citations are addresses (claim spans, replay seeks), and a model left to itself drifts into bold, backticks, or bare timestamps."""
    blocks = [
        f"\n<REPLY_FORMAT>\n    As you write, {CITE_RULE}.\n    {_FEEDBACK}\n</REPLY_FORMAT>\n"
    ]
    if pull:
        blocks.append(f"\n<SCREENSHOT_REQUESTS>\n    {_PULL}\n</SCREENSHOT_REQUESTS>\n")
    return "".join(blocks)


SCREENSHOTS_ALREADY_SERVED = (
    "No screenshots: your one request was already served. "
    "Write your answer from what you have."
)

REPAIR_NEEDS_NO_SCREENSHOTS = (
    "No screenshots: a repair needs none. Write the complete answer from "
    "what you already have."
)


EMPTY_ANSWER_REPAIR = """
<REPAIR>
    Your reply has no answer. Write the complete answer now, everything for the asker, every claim cited.
</REPAIR>
"""


def repair_task(violations: list[str], labels: list[str]) -> str:
    """The repair turn's task: each invalid citation with its defect, and nothing else — a concrete list of addresses to fix, never an invitation to reconsider the answer. The violations come from window.citation_violations, already rendered in the window's own offset vocabulary. The closing line re-demands this window's exact bracket form: a repair ask whose defect lines lead with bracketed spans reads to the model as "brackets were the problem", and the rewrite abandons the citation format unless the form is demanded again at the moment of writing."""
    form = citation_forms(labels)["cite"]
    listed = "\n".join(f"    - {violation}" for violation in violations)
    return f"""
<REPAIR>
    These citations in your answer are invalid:
{listed}
    Correct them and write the complete answer again — the same answer, every citation valid and written exactly as {form}, nothing else changed.
</REPAIR>
"""


_OFFSET_PARAM = {
    "type": "integer",
    "description": (
        "milliseconds from the recording's start; a stamp's MM:SS.mmm converts as MM*60000 + SS*1000 + mmm"
    ),
}


def request_screenshots_tool() -> dict:
    """The pull tool, one shape for every window: a moment is an offset plus, when the window has more than one slice, the label of the slice it is in — two slices may cover the same offset. The engine holds each request to the window it serves (engine._parse_requests): a label where the window has one slice, or none where it has several, is a defect."""
    return {
        "name": "request_screenshots",
        "description": "Show the screenshots at these moments of the recording.",
        "parameters": {
            "type": "object",
            "properties": {
                "screenshots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": _OFFSET_PARAM,
                            "label": {
                                "type": "string",
                                "description": (
                                    "the slice's label, as the events are stamped; required when the SUMMARY lists slices, omitted otherwise"
                                ),
                            },
                        },
                        "required": ["timestamp"],
                    },
                }
            },
            "required": ["screenshots"],
        },
    }


def split_meta(text: str) -> tuple[str, str | None]:
    """A reply split at the contract's meta header: (answer, channel or None). The channel is the model's words to the tuner, not claims about the session — the oracle excludes it before extraction, and the engine fails a reply whose answer half is empty. The split keys on the exact header the contract demands; text under a drifted header stays in the answer and is judged with it, the honest outcome for prose the channel did not sanction."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == META_HEADER:
            return (
                "\n".join(lines[:i]).rstrip(),
                "\n".join(lines[i + 1 :]).strip() or None,
            )
    return text.rstrip(), None


def question_task(question: str, closing: str = "") -> str:
    """The caller's entire question, sent last on the turn that answers it. closing lands after all of it — the last thing the model reads before it acts — for a caller whose model can still do something other than write (the engine, whose model can pull more screenshots)."""
    return f"<QUESTION>\n{question}\n</QUESTION>\n{closing}"
