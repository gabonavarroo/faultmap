from __future__ import annotations
from abc import ABC, abstractmethod

from ..models import ScoringResult


class BaseScorer(ABC):
    """Abstract base for all scoring modes."""

    @abstractmethod
    async def score(
        self,
        prompts: list[str],
        responses: list[str],
        **kwargs,
    ) -> ScoringResult:
        """
        Compute scores for prompt-response pairs.
        Returns ScoringResult with scores in [0, 1] where higher = better.
        """
        ...
