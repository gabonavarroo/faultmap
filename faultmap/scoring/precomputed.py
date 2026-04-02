from __future__ import annotations

from ..models import ScoringResult
from .base import BaseScorer


class PrecomputedScorer(BaseScorer):
    """Mode 1: User-provided scores. Pure passthrough."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    async def score(self, prompts, responses, **kwargs) -> ScoringResult:
        return ScoringResult(scores=list(self._scores), mode="precomputed")
