"""The claim model — typed claims with relative time spans.

A claim is OBSERVATION (a direct statement about a visitor action or system event) or INFERENCE (a conclusion derived from other claims; must reference them). Timestamp components are MM:SS.mmm relative to the analysis window's first event, and `label` is the slice label a multi-slice window's citations open with ([S2 04:31.220]); window.resolve_citation maps the pair back to the slice and absolute epoch time — the one lookup that ties a claim to replay (a page's citation seek) and to its evidence.

Extraction's output schema is a union: an object holding a NON-EMPTY claims list, or an explicit NoClaims assertion with a stated reason. Either single constraint alone fails, in opposite directions: a bare list lets `[]` be the zero-cost grammar escape (a model works through every claim in its deliberation, treats that as having delivered, and closes the constrained list empty), while min_length=1 alone makes a genuinely claim-free answer fabricate a claim against the model's own correct reasoning. The union closes both: emptiness stays expressible but is never cheap — it must be affirmatively asserted, with a reason that persists in the payloads.

Both branches are objects, so under constrained decoding they open with the same token and diverge only at a named key — `claims` vs `no_claims` — a choice with semantic weight the model's deliberation actually bears on. Branches that diverge at the opening token instead put the whole decision on one low-semantic character, and the grammar then force-completes whichever branch that character entered: one stray `{` where a list would have started is a full NoClaims, "asserted" by a model that had just finished enumerating claims.
"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    OBSERVATION = "OBSERVATION"
    INFERENCE = "INFERENCE"


class TimestampComponents(BaseModel):
    # Bounds ride the wire schema, so a model can't emit seconds=75 or a negative
    # component — the constraint holds at generation, not as a runtime crash.
    minutes: int = Field(ge=0)
    seconds: int = Field(ge=0, le=59)
    milliseconds: int = Field(ge=0, le=999)

    def to_ms(self) -> int:
        return (self.minutes * 60 + self.seconds) * 1000 + self.milliseconds


class Claim(BaseModel):
    # Field order here is wire grammar, not style: the local backend compiles the
    # schema through llguidance, which admits an object's declared properties only
    # in declaration order (model/protocol.py, the backend schema contract). So the
    # order must match the order the extraction prompt walks the fields — label
    # last, where the prompt's {label_instructions} slot sits — or the model's
    # taught emission order is masked at generation and the field is lost.
    claim_id: int
    claim_text: str
    claim_type: ClaimType
    evidence_ref: str | None = None
    supporting_claim_ids: list[int] | None = None
    start_timestamp_components: TimestampComponents | None = None
    end_timestamp_components: TimestampComponents | None = None
    label: str | None = None


class NoClaims(BaseModel):
    # A required constant so the branch must be asserted, never fallen into — a
    # string, not the bool it reads as, per the backend schema contract beside
    # json_schema_of (model/protocol.py): Gemini accepts only string constants.
    no_claims: Literal["true"]
    reason: str


class ClaimsFound(BaseModel):
    claims: Annotated[list[Claim], Field(min_length=1)]


ClaimExtraction = ClaimsFound | NoClaims


class ClaimVerdict(BaseModel):
    claim_id: int
    reasoning: str
    groundedness_pct: int = Field(ge=0, le=100)


class BatchVerdicts(BaseModel):
    evaluations: list[ClaimVerdict]


def batch_verdicts_schema(n_claims: int, binary: bool) -> type:
    """A verdict-batch schema whose list length is exactly the claim count.

    An unconstrained list lets the grammar close early: a raw double-quote inside one verdict's reasoning terminates the JSON string, and the model takes the legal exit, silently dropping the remaining claims. The judge knows the batch size, so under-delivery is made grammatically impossible the same way extraction's empty list was.
    """
    verdict = BinaryVerdict if binary else ClaimVerdict
    base = BinaryBatchVerdicts if binary else BatchVerdicts

    class _Sized(base):
        evaluations: Annotated[
            list[verdict], Field(min_length=n_claims, max_length=n_claims)
        ]

    _Sized.__name__ = base.__name__
    return _Sized


class BinaryVerdict(BaseModel):
    claim_id: int
    reasoning: str
    grounded: bool

    def to_pct(self) -> ClaimVerdict:
        return ClaimVerdict(
            claim_id=self.claim_id,
            reasoning=self.reasoning,
            groundedness_pct=100 if self.grounded else 0,
        )


class BinaryBatchVerdicts(BaseModel):
    evaluations: list[BinaryVerdict]


def structural_failures(claims: list[Claim], labels: list[str] = ()) -> dict[int, str]:
    """Claims rejected before any judge sees them, with reasons. `labels` is the window's slice-label roster: a claim citing an unknown label, or citing a timestamp without naming its slice's label in a multi-slice window, is mechanically unverifiable — its citation resolves to no evidence — and fails here without spending a token."""
    known = {claim.claim_id for claim in claims}
    failures: dict[int, str] = {}
    for claim in claims:
        if claim.claim_type is ClaimType.INFERENCE:
            if not claim.supporting_claim_ids:
                failures[claim.claim_id] = (
                    "inference lacks required supporting_claim_ids"
                )
                continue
            if unknown := set(claim.supporting_claim_ids) - known:
                failures[claim.claim_id] = (
                    f"inference references unknown claim ids {sorted(unknown)}"
                )
                continue
        cited = (
            claim.start_timestamp_components is not None
            or claim.end_timestamp_components is not None
        )
        if claim.label is not None and claim.label not in labels:
            failures[claim.claim_id] = (
                f"claim cites unknown label {claim.label!r} (window labels: {list(labels)})"
            )
        elif cited and claim.label is None and len(labels) > 1:
            failures[claim.claim_id] = (
                "claim cites a timestamp without naming its slice's label — "
                "in a multi-slice window the citation resolves to no evidence"
            )
    return failures


def merge_spans(spans: list[tuple[int, int]], pad_ms: int) -> list[tuple[int, int]]:
    """Pad each span by pad_ms and merge overlaps into disjoint spans."""
    if not spans:
        return []
    padded = sorted((start - pad_ms, end + pad_ms) for start, end in spans)
    merged = [padded[0]]
    for start, end in padded[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
