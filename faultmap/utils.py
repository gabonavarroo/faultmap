from __future__ import annotations

import asyncio
from typing import TypeVar, Awaitable

import numpy as np

T = TypeVar("T")


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity between rows of a and b.
    Args: a (n, d), b (m, d)
    Returns: (n, m) matrix in [-1, 1].
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def cosine_similarity_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Element-wise cosine similarity between paired rows.
    Args: a (n, d), b (n, d)
    Returns: (n,) array in [-1, 1].
    """
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1) + 1e-10
    norm_b = np.linalg.norm(b, axis=1) + 1e-10
    return dot / (norm_a * norm_b)


def run_sync(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine synchronously.
    Handles Jupyter's already-running event loop via nest_asyncio.

    Logic:
    1. Try asyncio.get_running_loop()
    2. If loop exists: apply nest_asyncio, use loop.run_until_complete()
    3. If no loop: apply nest_asyncio globally, use asyncio.run()
    """
    import nest_asyncio
    try:
        loop = asyncio.get_running_loop()
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    except RuntimeError:
        nest_asyncio.apply()
        return asyncio.run(coro)


def validate_inputs(
    prompts: list[str],
    responses: list[str],
    scores: list[float] | None,
    references: list[str] | None,
) -> None:
    """
    Validate user inputs. Raises ConfigurationError.

    Checks:
    - prompts and responses are non-empty and equal length
    - scores (if given): same length, all numeric, all in [0, 1]
    - references (if given): same length
    """
    from .exceptions import ConfigurationError

    if not prompts:
        raise ConfigurationError("prompts must be a non-empty list")
    if len(prompts) != len(responses):
        raise ConfigurationError(
            f"prompts ({len(prompts)}) and responses ({len(responses)}) "
            f"must have equal length"
        )
    if scores is not None:
        if len(scores) != len(prompts):
            raise ConfigurationError(
                f"scores ({len(scores)}) must have same length as prompts ({len(prompts)})"
            )
        for i, s in enumerate(scores):
            if not isinstance(s, (int, float)):
                raise ConfigurationError(
                    f"scores[{i}] must be numeric, got {type(s).__name__}"
                )
            if not (0.0 <= s <= 1.0):
                raise ConfigurationError(
                    f"scores[{i}]={s} is out of range [0, 1]. "
                    "Normalize scores to [0, 1] before passing to faultmap."
                )
    if references is not None:
        if len(references) != len(prompts):
            raise ConfigurationError(
                f"references ({len(references)}) must have same length "
                f"as prompts ({len(prompts)})"
            )


def batch_items(items: list, batch_size: int) -> list[list]:
    """Split a list into batches of at most batch_size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
