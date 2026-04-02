from unittest.mock import AsyncMock

import pytest

from faultmap.labeling import (
    _parse_label_response,
    label_cluster,
    label_clusters,
)


class TestParseLabelResponse:
    def test_well_formed(self):
        response = "Name: Legal Questions\nDescription: Questions about legal compliance."
        label = _parse_label_response(response)
        assert label.name == "Legal Questions"
        assert label.description == "Questions about legal compliance."

    def test_malformed_fallback(self):
        response = "These are billing inquiries\nabout payment disputes"
        label = _parse_label_response(response)
        assert label.name == "These are billing inquiries"
        assert "payment disputes" in label.description

    def test_extra_whitespace(self):
        response = "  Name:   Billing Issues  \n  Description:   About billing.  "
        label = _parse_label_response(response)
        assert label.name == "Billing Issues"
        assert label.description == "About billing."


class TestLabelCluster:
    @pytest.mark.asyncio
    async def test_calls_llm_and_parses(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = (
            "Name: Setup Questions\nDescription: Questions about initial setup."
        )

        label = await label_cluster(
            mock_client,
            ["How do I set up X?", "Setup guide for Y"],
            context="failure slice",
        )

        assert label.name == "Setup Questions"
        mock_client.complete.assert_called_once()


class TestLabelClusters:
    @pytest.mark.asyncio
    async def test_labels_multiple_concurrently(self):
        mock_client = AsyncMock()
        mock_client.complete.side_effect = [
            "Name: Group A\nDescription: First group.",
            "Name: Group B\nDescription: Second group.",
        ]

        labels = await label_clusters(
            mock_client,
            [["text1", "text2"], ["text3", "text4"]],
            context="failure slice",
        )

        assert len(labels) == 2
        assert labels[0].name == "Group A"
        assert labels[1].name == "Group B"
