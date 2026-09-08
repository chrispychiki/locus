"""Claim extraction — the answer decomposed into typed, citable claims.

Zero claims is a legitimate result, but only as an explicit assertion: the schema is a union of a non-empty claims list and a NoClaims{reason} object (see claims.py), so an extraction can never silently come back empty. A NoClaims return yields [] with the reason preserved in the persisted payloads.
"""

from ..model.protocol import Conversation
from .claims import Claim, ClaimExtraction, NoClaims
from .prompts import SET_A, PromptSet, label_instructions


def extract_claims(
    answer_text: str,
    conversation: Conversation,
    prompt_set: PromptSet = SET_A,
    labels: list[str] = (),
) -> tuple[list[Claim], dict]:
    """`labels` is the analyzed window's slice-label roster: it interpolates the prompt's citation format (label_instructions), so a multi-slice answer's labeled citations are parsed into Claim.label and a single-slice answer's prompt never mentions labels."""
    conversation.add_user_text(
        prompt_set.extraction_prompt.format(
            answer_text=answer_text, label_instructions=label_instructions(list(labels))
        )
    )
    response = conversation.get_response(
        response_schema=ClaimExtraction, label="claims"
    )
    if isinstance(response.parsed, NoClaims):
        return [], dict(response.usage_metadata)
    claims: list[Claim] = response.parsed.claims

    ids = [claim.claim_id for claim in claims]
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"extraction produced duplicate claim ids {duplicates}")

    return claims, dict(response.usage_metadata)
