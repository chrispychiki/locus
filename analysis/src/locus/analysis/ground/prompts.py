"""Grounding prompt sets.

Two prompt sets score an answer for groundedness, chosen per run, independent of backend. Set A scores each claim on a 0-100 scale. Set B's judge is Set A's criteria with the 0-100 scale removed — each claim is grounded or it is not (binary_judge=True, mapped to 100/0 downstream) — and its extraction is its own leaner variant (an inference cap, a one-line system prompt), not a transform of A's. A 0-100 scale carries a lenient middle where a judge can acknowledge a defect and still pay a passing score; a binary verdict forces the commitment the scale lets it dodge.

The {session_context} slot is filled mechanically from the db, and {label_instructions} is the extraction prompt's citation-format slot: interpolated per window from its real slice labels (label_instructions below), so an answer over a multi-slice window is decomposed under the exact labeled format it was written in, and a single-slice answer's prompt never mentions labels. The answer's shape is the analysis's own business — however it is written, its factual statements about the session are claims, either cited or not, and extraction owes no section layout anything.
"""

from dataclasses import dataclass

from ..window import citation_forms


def label_instructions(labels: list[str]) -> str:
    """The extraction prompt's `label` field instructions, concrete for this window. Empty for a single-slice window — its citations carry no label and the schema's `label` field stays null."""
    if len(labels) <= 1:
        return ""
    forms = citation_forms(list(labels))
    return (
        f"- `label`: this answer's citations open with a slice label — "
        f"{forms['cite']} — and the window's slices are labeled "
        f"{labels[0]}–{labels[-1]}. Populate this field with the cited label "
        f'(e.g. "{labels[1]}"). If a time reference carries no label '
        f"and none can be resolved from context, leave this field null.\n"
    )


@dataclass(frozen=True)
class PromptSet:
    name: str
    extraction_system: str
    extraction_prompt: str
    validation_system: str
    validation_opening: str
    validation_closing: str
    binary_judge: bool


_A_EXTRACTION_PROMPT = """<InputAnswer>
{answer_text}
</InputAnswer>

<OutputRequirements>
Generate a JSON object whose `claims` field is a list of Claim objects based on the <InputAnswer>. Each object in the list represents a unique, substantive claim identified from the answer. Be exhaustive: every distinct factual statement about the session is a claim — a single sentence that bundles several facts (an action, a target, a time) becomes several claims, not zero. Filter out only true noise: headers, meta-commentary, introductions, filler text, and statements about the answer itself rather than the session.

ASSERTING "NO CLAIMS" IS ALMOST NEVER CORRECT. The schema lets you assert the answer has no claims. That is reserved for the rare answer that contains no statement about any visitor action or system event — an error stub, a placeholder, a refusal. ANY answer that mentions even one thing the visitor did or that happened to them — a page load, a click, a scroll, a text selection, a navigation, an exit — contains claims and MUST be extracted. Never assert "no claims" because the prose reads as narrative, interpretive, or "not atomic enough", or because separating claims feels lossy: those are never valid reasons. Do not deliberate about extractability — when an answer describes a session, extract it.

DEDUPLICATION: When the same claim appears multiple times (within a section or across sections), extract it ONLY ONCE:
- Identify when different wordings describe the identical claim (e.g., "closed popup" vs "dismissed newsletter popup", or "visitor was engaged" vs "visitor showed high engagement")
- For duplicate claims, prefer the version with timestamps over one without
- If all duplicates have timestamps, prefer the one with more detail or clarity

For each Claim object, ensure the following fields are populated accurately:
- `claim_id`: Assign a sequential integer starting from 1 based on the order claims appear in the answer.
- `claim_text`: The exact text of the unique, substantive claim.
- `claim_type`: Classify the claim using one of these types:
    - OBSERVATION: Direct statement about a visitor action or system event. Capture negative claims about actions/events that did NOT occur in addition to positive ones.
    - INFERENCE: Conclusion, interpretation, or recommendation derived from other claims.
- `evidence_ref`: Look for timestamp citations that provide temporal context for this claim (e.g., `[T1]`, `first 3s`, `(Fig. 2)`, `"as stated above"`, `[00:01.234]`, etc.). Populate this field with that exact text. Look for relevant citations from anywhere in the answer. Only if there is absolutely no way to reasonably deduce when the claim occured according to the answer, leave this field null.
- `supporting_claim_ids`: If you set `claim_type` to INFERENCE, this field must contain a list of `claim_id`s of claims that provide direct logical support. Leave null for OBSERVATION claims.
- `start_timestamp_components` / `end_timestamp_components`: If `evidence_ref` contains a time reference, attempt to parse it and populate these fields (containing minute/second/millisecond objects). If time information cannot be resolved, leave these fields null.
  Parsing examples (not exhaustive): "[00:01:02.345]" -> parse directly. "here", "then" -> resolve contextually. "first 3s" -> start={{minutes:0, seconds:0, milliseconds:0}}, end={{minutes:0, seconds:3, milliseconds:0}}. "around 1:30" -> {{minutes:1, seconds:30, milliseconds:0}}. "(Fig. 2)" -> null components. "during loading" -> null components if unresolvable.
{label_instructions}</OutputRequirements>"""

_A_VALIDATION_OPENING = """<BatchValidationRequest>
<Objective>
Your task is to determine if claims from a session analysis are well grounded.

The original evidence for the analysis consisted of an event stream and screenshots from the visitor's session.
You have been provided with a portion of that original evidence based on proximity to claim timestamps.
You do NOT have access to the full event stream or all screenshots.
</Objective>

{session_context}

<EventStream>
{event_stream}
</EventStream>"""

_A_VALIDATION_CLOSING = """<EvaluationCriteria>
Rate each claim's groundedness from 0-100 based on how well the evidence supports it. Avoid using external knowledge unrelated to standard UX patterns and visitor behaviors.

OBSERVATION claims are direct statements about visitor actions or system events.
- 100: Clear evidence around the cited timestamp(s) directly supports the claim.
- 0: No timestamp is cited, no relevant evidence exists around the cited timestamp(s), or evidence contradicts the claim.

Observations can be negative claims about lack of behavior. Treat these the same as positive claims: use relevant evidence to determine if the behavior truly did not occur.

You ONLY have evidence from time spans around cited timestamps. If an observation has no timestamp, contradicting evidence may exist elsewhere in the session that you do not have access to.
A narrow observation, e.g. "The visitor navigated to XYZ product page after searching for it", if supported by evidence coincidentally brought in via other claims, is more acceptable than a broader claim like "The visitor never went to XYZ throughout the session".

INFERENCE claims are conclusions, interpretations, or recommendations derived from other claims.
- 100: The inference clearly follows from its supporting claims.
- 0: The inference is not justified by its supporting claims.
</EvaluationCriteria>

<Task>
For each claim, evaluate using the criteria above. Respond with: claim ID, brief reasoning, then groundedness score (0-100). You MUST preserve the exact ID numbers of each claim in your response.

<Claims>
{claims_section}
</Claims>
</Task>
</BatchValidationRequest>"""

_B_EXTRACTION_PROMPT = """<InputAnswer>
{answer_text}
</InputAnswer>

<OutputRequirements>
Generate a JSON object whose `claims` field is a list of Claim objects based on the <InputAnswer>. Each object in the list represents a unique, substantive claim about the session. Filter out noise like headers, meta-commentary, introductions, and filler text. Do NOT extract statements about the answer itself.

DEDUPLICATION: The same event may be described more than once, within a section or across sections. Extract each event ONCE — prefer the version with more detail or timestamps.
- Two claims describing the same conclusion in different words are duplicates (e.g., "deep engagement" vs "high engagement through navigation"). Extract only one.

INFERENCE LIMIT: Extract at most 3 INFERENCE claims. Each must express a genuinely distinct conclusion — not a restatement or logical entailment of another inference. If one inference logically implies another (e.g., "persistent" implies "patient" implies "engaged"), keep only the most specific one.

For each Claim object, ensure the following fields are populated accurately:
- `claim_id`: Sequential integer starting from 1 based on order in the answer.
- `claim_text`: The exact text of the unique, substantive claim from the answer.
- `claim_type`: Classify as:
    - OBSERVATION: Direct statement about a visitor action or system event, including negative claims about actions that did NOT occur.
    - INFERENCE: Conclusion, interpretation, or recommendation derived from other claims.
- `evidence_ref`: Look for timestamp citations that provide temporal context (e.g., `[T1]`, `first 3s`, `[00:01.234]`). Use only temporal references — not section headers. Leave null if no timestamp exists.
- `supporting_claim_ids`: For INFERENCE claims only, list `claim_id`s that provide direct logical support. Leave null for OBSERVATIONs.
- `start_timestamp_components` / `end_timestamp_components`: If `evidence_ref` contains a time reference, parse into minute/second/millisecond objects. Leave null if unresolvable.
  Parsing examples: "[00:01:02.345]" -> parse directly. "first 3s" -> start={{minutes:0, seconds:0, milliseconds:0}}, end={{minutes:0, seconds:3, milliseconds:0}}. "around 1:30" -> {{minutes:1, seconds:30, milliseconds:0}}.
{label_instructions}</OutputRequirements>"""

_B_VALIDATION_CLOSING = """<EvaluationCriteria>
Decide for each claim whether it is grounded in the evidence: grounded means the evidence supports the claim, in full. There is no partial credit — a claim with any unsupported or contradicted part is not grounded. Avoid using external knowledge unrelated to standard UX patterns and visitor behaviors.

OBSERVATION claims are direct statements about visitor actions or system events.
- grounded: clear evidence around the cited timestamp(s) directly supports the claim, all of it.
- not grounded: no timestamp is cited, no relevant evidence exists around the cited timestamp(s), or evidence contradicts any part of the claim.

Observations can be negative claims about lack of behavior. Treat these the same as positive claims: use relevant evidence to determine if the behavior truly did not occur.

You ONLY have evidence from time spans around cited timestamps. If an observation has no timestamp, contradicting evidence may exist elsewhere in the session that you do not have access to.
A narrow observation, e.g. "The visitor navigated to XYZ product page after searching for it", if supported by evidence coincidentally brought in via other claims, is more acceptable than a broader claim like "The visitor never went to XYZ throughout the session".

INFERENCE claims are conclusions, interpretations, or recommendations derived from other claims.
- grounded: the inference clearly follows from its supporting claims.
- not grounded: the inference is not justified by its supporting claims.
</EvaluationCriteria>

<Task>
For each claim, evaluate using the criteria above. Respond with: claim ID, brief reasoning, then the grounded verdict (true or false). You MUST preserve the exact ID numbers of each claim in your response.

<Claims>
{claims_section}
</Claims>
</Task>
</BatchValidationRequest>"""

SET_A = PromptSet(
    name="a",
    extraction_system=(
        "You are an expert claim extractor. You decompose an answer about a UX "
        "session into every distinct, citable factual claim it makes about the "
        "session, exhaustively. An answer that narrates what a visitor did is full "
        "of claims by definition; your job is to surface them all, never to judge "
        "whether the answer is worth extracting. Interpretive or narrative phrasing "
        "never disqualifies a statement from yielding a claim, and you never decline "
        "an answer that describes a session."
    ),
    extraction_prompt=_A_EXTRACTION_PROMPT,
    validation_system="You are an expert UX evaluator validating claims against evidence.",
    validation_opening=_A_VALIDATION_OPENING,
    validation_closing=_A_VALIDATION_CLOSING,
    binary_judge=False,
)

SET_B = PromptSet(
    name="b",
    extraction_system="You are an expert claim extractor.",
    extraction_prompt=_B_EXTRACTION_PROMPT,
    validation_system="You are an expert UX evaluator validating claims against evidence.",
    validation_opening=_A_VALIDATION_OPENING,
    validation_closing=_B_VALIDATION_CLOSING,
    binary_judge=True,
)
