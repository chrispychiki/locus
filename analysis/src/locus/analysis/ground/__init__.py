"""Grounding — the measurement instrument, never a stage of analysis.

A finished answer is decomposed into typed claims, and each claim is judged against the screenshots and events in the span it cites. Both halves run on the model, so grounding stays in the semantic layer. It runs offline against an analysis's persisted artifacts (oracle.py), never inline while an answer is being written: the thing that measures an answer must be unreachable from the thing that writes it. The spans a claim cites are the same moments a replay page opens, so a doubted claim is settled by looking.
"""

from .claims import Claim, ClaimType, ClaimVerdict
from .evaluate import evaluate_answer, groundedness, judged_claims

__all__ = [
    "Claim",
    "ClaimType",
    "ClaimVerdict",
    "evaluate_answer",
    "groundedness",
    "judged_claims",
]
