from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

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


def validate_comparison_inputs(
    prompts: list[str],
    responses_a: list[str],
    responses_b: list[str],
    scores_a: list[float] | None,
    scores_b: list[float] | None,
    references: list[str] | None,
) -> None:
    """Validate user inputs for :meth:`SliceAnalyzer.compare_models`.

    Raises :class:`~faultmap.exceptions.ConfigurationError` on any violation.

    Checks performed:

    - ``prompts`` is non-empty.
    - ``len(prompts) == len(responses_a) == len(responses_b)``.
    - ``scores_a`` and ``scores_b`` must both be provided or both be ``None``.
      Providing only one raises :class:`~faultmap.exceptions.ConfigurationError`
      (asymmetric scores are ambiguous).
    - When both scores are provided: same length as ``prompts``; each element
      is numeric and in ``[0, 1]``.
    - When ``references`` is provided: same length as ``prompts``.

    Args:
        prompts: Shared prompts for both models.
        responses_a: Model A responses (one per prompt).
        responses_b: Model B responses (one per prompt).
        scores_a: Optional precomputed scores for Model A, each in ``[0, 1]``.
        scores_b: Optional precomputed scores for Model B, each in ``[0, 1]``.
        references: Optional reference answers (one per prompt) for Mode 2.

    Raises:
        ConfigurationError: On any input violation described above.
    """
    from .exceptions import ConfigurationError

    if not prompts:
        raise ConfigurationError("prompts must be a non-empty list")

    n = len(prompts)

    if len(responses_a) != n:
        raise ConfigurationError(
            f"responses_a ({len(responses_a)}) must have the same length "
            f"as prompts ({n})"
        )
    if len(responses_b) != n:
        raise ConfigurationError(
            f"responses_b ({len(responses_b)}) must have the same length "
            f"as prompts ({n})"
        )

    # Asymmetric scores check — both or neither
    scores_a_given = scores_a is not None
    scores_b_given = scores_b is not None
    if scores_a_given != scores_b_given:
        missing = "scores_b" if scores_a_given else "scores_a"
        present = "scores_a" if scores_a_given else "scores_b"
        raise ConfigurationError(
            f"scores_a and scores_b must both be provided or both be None. "
            f"Received {present} but {missing} is missing."
        )

    if scores_a is not None and scores_b is not None:
        if len(scores_a) != n:
            raise ConfigurationError(
                f"scores_a ({len(scores_a)}) must have the same length "
                f"as prompts ({n})"
            )
        if len(scores_b) != n:
            raise ConfigurationError(
                f"scores_b ({len(scores_b)}) must have the same length "
                f"as prompts ({n})"
            )
        for i, s in enumerate(scores_a):
            if not isinstance(s, (int, float)):
                raise ConfigurationError(
                    f"scores_a[{i}] must be numeric, got {type(s).__name__}"
                )
            if not (0.0 <= float(s) <= 1.0):
                raise ConfigurationError(
                    f"scores_a[{i}]={s} is out of range [0, 1]. "
                    "Normalize scores to [0, 1] before passing to faultmap."
                )
        for i, s in enumerate(scores_b):
            if not isinstance(s, (int, float)):
                raise ConfigurationError(
                    f"scores_b[{i}] must be numeric, got {type(s).__name__}"
                )
            if not (0.0 <= float(s) <= 1.0):
                raise ConfigurationError(
                    f"scores_b[{i}]={s} is out of range [0, 1]. "
                    "Normalize scores to [0, 1] before passing to faultmap."
                )

    if references is not None:
        if len(references) != n:
            raise ConfigurationError(
                f"references ({len(references)}) must have the same length "
                f"as prompts ({n})"
            )
