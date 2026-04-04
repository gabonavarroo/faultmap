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

    def test_four_field_response(self):
        response = (
            "Name: Legal Questions\n"
            "Description: Questions about legal compliance.\n"
            "Root Cause: Model lacks legal domain training data.\n"
            "Suggested Fix: Add legal context to the system prompt."
        )
        label = _parse_label_response(response)
        assert label.name == "Legal Questions"
        assert label.description == "Questions about legal compliance."
        assert label.root_cause == "Model lacks legal domain training data."
        assert label.suggested_remediation == "Add legal context to the system prompt."

    def test_two_field_response_defaults_empty(self):
        """Old 2-field LLM responses still parse; new fields default to empty string."""
        response = "Name: Billing Issues\nDescription: About billing disputes."
        label = _parse_label_response(response)
        assert label.root_cause == ""
        assert label.suggested_remediation == ""


class TestLabelCluster:
    @pytest.mark.asyncio
    async def test_calls_llm_and_parses(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = (
            "Name: Setup Questions\nDescription: Questions about initial setup.\n"
            "Root Cause: Missing setup documentation context.\n"
            "Suggested Fix: Add setup guide context to the system prompt."
        )

        label = await label_cluster(
            mock_client,
            ["How do I set up X?", "Setup guide for Y"],
            context="failure slice",
        )

        assert label.name == "Setup Questions"
        assert label.root_cause == "Missing setup documentation context."
        assert label.suggested_remediation == "Add setup guide context to the system prompt."
        mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_failure_slice_context_no_insights(self):
        """Coverage gap and comparison contexts use 2-field prompt; insights stay empty."""
        mock_client = AsyncMock()
        mock_client.complete.return_value = (
            "Name: Auth Topics\nDescription: Authentication-related questions."
        )

        label = await label_cluster(
            mock_client,
            ["How do I log in?", "Reset my password"],
            context="coverage gap",
        )

        assert label.name == "Auth Topics"
        assert label.root_cause == ""
        assert label.suggested_remediation == ""
        # Verify the call used 150 max_tokens (not 400)
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 150

    @pytest.mark.asyncio
    async def test_failure_slice_context_uses_high_token_limit(self):
        """Failure slice context uses 400 max_tokens for the enhanced prompt."""
        mock_client = AsyncMock()
        mock_client.complete.return_value = (
            "Name: Setup Questions\nDescription: Questions about initial setup.\n"
            "Root Cause: Missing context.\nSuggested Fix: Add context."
        )

        await label_cluster(
            mock_client,
            ["How do I set up X?"],
            context="failure slice",
        )

        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 400


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
