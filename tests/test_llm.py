import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from faultmap.llm import AsyncLLMClient
from faultmap.exceptions import LLMError


@pytest.fixture
def client():
    return AsyncLLMClient(model="gpt-4o-mini", max_concurrent_requests=5, max_retries=3)


class TestAsyncLLMClient:
    @pytest.mark.asyncio
    async def test_complete_success(self, client):
        """Mock litellm.acompletion, verify we get the response content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await client.complete([{"role": "user", "content": "Hi"}])
            assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_complete_retries_on_failure(self, client):
        """Fail twice, succeed on third attempt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"

        mock_acompletion = AsyncMock(
            side_effect=[Exception("fail 1"), Exception("fail 2"), mock_response]
        )

        with patch("litellm.acompletion", mock_acompletion):
            result = await client.complete([{"role": "user", "content": "Hi"}])
            assert result == "Success"
            assert mock_acompletion.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_raises_after_max_retries(self, client):
        """All retries fail → LLMError."""
        mock_acompletion = AsyncMock(side_effect=Exception("always fails"))

        with patch("litellm.acompletion", mock_acompletion):
            with pytest.raises(LLMError, match="failed after 3 retries"):
                await client.complete([{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_complete_batch_preserves_order(self, client):
        """Batch results come back in same order as input."""
        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            idx = call_count
            await asyncio.sleep(0.01 * (3 - idx))  # reverse delay
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = f"response-{idx}"
            return mock_resp

        messages_list = [
            [{"role": "user", "content": f"prompt-{i}"}] for i in range(3)
        ]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            results = await client.complete_batch(
                messages_list, show_progress=False
            )
            assert results == ["response-1", "response-2", "response-3"]

    @pytest.mark.asyncio
    async def test_complete_batch_empty(self, client):
        result = await client.complete_batch([], show_progress=False)
        assert result == []
