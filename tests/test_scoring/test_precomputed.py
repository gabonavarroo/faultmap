import pytest

from faultmap.scoring.precomputed import PrecomputedScorer


class TestPrecomputedScorer:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        scores = [0.1, 0.5, 0.9]
        scorer = PrecomputedScorer(scores)
        result = await scorer.score(["a", "b", "c"], ["x", "y", "z"])
        assert result.scores == [0.1, 0.5, 0.9]
        assert result.mode == "precomputed"

    @pytest.mark.asyncio
    async def test_does_not_mutate(self):
        original = [0.5, 0.6]
        scorer = PrecomputedScorer(original)
        result = await scorer.score(["a", "b"], ["x", "y"])
        result.scores  # should be a copy
        original[0] = 999
        assert result.scores[0] == 0.5
