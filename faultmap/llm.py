from __future__ import annotations

import asyncio
import logging
from typing import Any

from .exceptions import LLMError

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """
    Async wrapper around litellm.acompletion with semaphore-based rate limiting.

    Used by:
    - scoring/entropy.py (sampling N responses per prompt)
    - labeling.py (naming clusters)
    """

    def __init__(
        self,
        model: str,
        max_concurrent_requests: int = 50,
        max_retries: int = 10,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """
        Single completion call with retry + semaphore.

        Algorithm:
        1. Acquire semaphore
        2. For attempt in range(max_retries):
            a. Call litellm.acompletion(model, messages, temperature, max_tokens, timeout)
            b. On success: return response.choices[0].message.content.strip()
            c. On failure: exponential backoff = 2^attempt seconds, log warning
        3. Raise LLMError with last exception

        Edge cases:
        - LLM returns None content → raise LLMError
        - Rate limit (429) → handled by backoff
        - Timeout → handled by backoff
        """
        import litellm
        litellm.telemetry = False

        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self.timeout,
                        **kwargs,
                    )
                    content = response.choices[0].message.content
                    if content is None:
                        raise LLMError("LLM returned None content")
                    return content.strip()
                except Exception as e:
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e}. Retrying in {wait}s."
                    )
                    await asyncio.sleep(wait)

        raise LLMError(
            f"LLM call failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    async def complete_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        desc: str = "LLM calls",
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """
        Batch completion with tqdm progress bar.

        Algorithm:
        1. Create async tasks for each message set
        2. Gather with tqdm_asyncio.gather for progress
        3. Return results in original order

        Edge cases:
        - Empty list → return []
        - Single failure in batch → entire gather raises (caller handles)
        """
        if not messages_list:
            return []

        from tqdm.asyncio import tqdm_asyncio

        tasks = [
            self.complete(msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
            for msgs in messages_list
        ]

        if show_progress:
            results = await tqdm_asyncio.gather(*tasks, desc=desc)
        else:
            results = await asyncio.gather(*tasks)

        return list(results)
