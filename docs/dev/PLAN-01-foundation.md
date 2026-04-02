# Phase 1 — Foundation (Day 1)

**Goal**: Package scaffolding, data contracts, shared utilities. Everything else builds on this.

**Files to create**:
- `pyproject.toml`
- `faultmap/__init__.py`
- `faultmap/exceptions.py`
- `faultmap/models.py`
- `faultmap/utils.py`
- `tests/conftest.py` (skeleton)
- `tests/test_utils.py`

**Milestone**: `pip install -e .` works, all models importable, utils tests pass.

---

## 1. `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "faultmap"
version = "0.1.0"
description = "Automatically discover where and why your LLM is failing"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [{ name = "Gabriel Navarro" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "litellm>=1.30",
    "tqdm>=4.60",
    "nest-asyncio>=1.5",
]

[project.optional-dependencies]
local = ["sentence-transformers>=2.2"]
rich = ["rich>=13.0"]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.4",
]
all = ["faultmap[local,rich,dev]"]

[tool.ruff]
target-version = "py310"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## 2. `faultmap/exceptions.py`

```python
"""Custom exceptions for faultmap."""


class FaultmapError(Exception):
    """Base exception for all faultmap errors."""


class EmbeddingError(FaultmapError):
    """Raised when embedding computation fails."""


class ScoringError(FaultmapError):
    """Raised when scoring computation fails."""


class LLMError(FaultmapError):
    """Raised when LLM calls fail after retries."""


class ClusteringError(FaultmapError):
    """Raised when clustering produces degenerate results."""


class ConfigurationError(FaultmapError):
    """Raised for invalid parameter combinations."""
```

---

## 3. `faultmap/models.py` — Complete Implementation

This is the contract everything else builds against. All dataclasses are `frozen=True`.

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoringResult:
    """Output of any scoring mode."""
    scores: list[float]           # Per-prompt score in [0, 1]. Higher = better.
    mode: str                     # "precomputed" | "reference" | "entropy"
    metadata: dict = field(default_factory=dict)
    # metadata varies by mode:
    #   precomputed: {}
    #   reference:   {"embedding_model": str}
    #   entropy:     {"n_samples": int, "temperature": float,
    #                 "semantic_entropy": list[float],
    #                 "self_consistency": list[float],
    #                 "normalized_entropy": list[float]}


@dataclass(frozen=True)
class FailureSlice:
    """A single discovered failure cluster."""
    name: str                          # LLM-generated name ("Legal compliance questions")
    description: str                   # LLM-generated 1-sentence explanation
    size: int                          # Number of prompts in this slice
    failure_rate: float                # Failure rate within this slice
    baseline_rate: float               # Overall failure rate for comparison
    effect_size: float                 # failure_rate / baseline_rate (risk ratio)
    p_value: float                     # Raw p-value from statistical test
    adjusted_p_value: float            # BH-corrected p-value
    test_used: str                     # "chi2" | "fisher"
    sample_indices: list[int]          # ALL indices in the cluster (into original prompts list)
    examples: list[dict]               # Top-5 dicts: {"prompt": str, "response": str, "score": float}
    representative_prompts: list[str]  # Top-5 prompts closest to cluster centroid
    cluster_id: int                    # Internal cluster label


@dataclass(frozen=True)
class AnalysisReport:
    """Complete output of SliceAnalyzer.analyze()."""
    slices: list[FailureSlice]    # Sorted by adjusted_p_value ascending
    total_prompts: int
    total_failures: int
    baseline_failure_rate: float
    significance_level: float
    failure_threshold: float
    scoring_mode: str             # "precomputed" | "reference" | "entropy"
    num_clusters_tested: int
    num_significant: int          # len(slices)
    clustering_method: str        # "hdbscan" | "agglomerative"
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-paragraph plain-text summary."""
        if not self.slices:
            return (
                f"No statistically significant failure slices found among "
                f"{self.total_prompts} prompts (baseline failure rate: "
                f"{self.baseline_failure_rate:.1%}, alpha={self.significance_level})."
            )
        worst = self.slices[0]
        return (
            f"Found {self.num_significant} significant failure slice(s) among "
            f"{self.total_prompts} prompts. Worst slice: \"{worst.name}\" "
            f"({worst.size} prompts, {worst.failure_rate:.1%} failure rate vs "
            f"{self.baseline_failure_rate:.1%} baseline, adj. p={worst.adjusted_p_value:.4f})."
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        from .report import format_analysis_report
        return format_analysis_report(self)


@dataclass(frozen=True)
class CoverageGap:
    """A single discovered gap in test coverage."""
    name: str                     # LLM-generated name
    description: str              # LLM-generated description
    size: int                     # Number of production prompts in this gap
    mean_distance: float          # Average NN distance to nearest test prompt
    representative_prompts: list[str]  # Top-k production prompts
    prompt_indices: list[int]     # Indices into production_prompts list
    cluster_id: int


@dataclass(frozen=True)
class CoverageReport:
    """Complete output of SliceAnalyzer.audit_coverage()."""
    gaps: list[CoverageGap]       # Sorted by mean_distance descending
    num_test_prompts: int
    num_production_prompts: int
    num_gaps: int                 # len(gaps)
    overall_coverage_score: float # 1 - (fraction of prod prompts in gaps)
    distance_threshold: float     # Threshold used for "uncovered"
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.gaps:
            return (
                f"Test suite appears to cover production traffic well. "
                f"No significant gaps found among {self.num_production_prompts} "
                f"production prompts."
            )
        worst = self.gaps[0]
        return (
            f"Found {self.num_gaps} coverage gap(s). "
            f"Overall coverage score: {self.overall_coverage_score:.1%}. "
            f"Largest gap: \"{worst.name}\" ({worst.size} uncovered production prompts, "
            f"mean distance={worst.mean_distance:.3f})."
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        from .report import format_coverage_report
        return format_coverage_report(self)
```

---

## 4. `faultmap/utils.py`

```python
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
```

---

## 5. `faultmap/__init__.py`

```python
"""faultmap: Automatically discover where and why your LLM is failing."""

from .analyzer import SliceAnalyzer
from .models import (
    AnalysisReport,
    CoverageGap,
    CoverageReport,
    FailureSlice,
    ScoringResult,
)
from .exceptions import (
    FaultmapError,
    EmbeddingError,
    ScoringError,
    LLMError,
    ClusteringError,
    ConfigurationError,
)

__version__ = "0.1.0"

__all__ = [
    "SliceAnalyzer",
    "AnalysisReport",
    "FailureSlice",
    "CoverageReport",
    "CoverageGap",
    "ScoringResult",
    "FaultmapError",
    "EmbeddingError",
    "ScoringError",
    "LLMError",
    "ClusteringError",
    "ConfigurationError",
    "__version__",
]
```

**Note**: `__init__.py` imports `SliceAnalyzer` from `analyzer.py`, which doesn't exist yet on Day 1. Two options:
- Option A: Create a stub `analyzer.py` with just `class SliceAnalyzer: pass`
- Option B: Use lazy imports. **Recommended: Option A** — simple, lets `pip install -e .` succeed.

---

## 6. Day 1 Tests — `tests/test_utils.py`

```python
import numpy as np
import pytest
from faultmap.utils import (
    cosine_similarity_matrix,
    cosine_similarity_pairs,
    validate_inputs,
    batch_items,
)
from faultmap.exceptions import ConfigurationError


class TestCosineSimilarityMatrix:
    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        result = cosine_similarity_matrix(a, a)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 1.0)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result = cosine_similarity_matrix(a, b)
        assert np.isclose(result[0, 0], 0.0, atol=1e-6)

    def test_antiparallel_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        result = cosine_similarity_matrix(a, b)
        assert np.isclose(result[0, 0], -1.0)

    def test_batch_shape(self):
        a = np.random.randn(5, 10)
        b = np.random.randn(3, 10)
        result = cosine_similarity_matrix(a, b)
        assert result.shape == (5, 3)


class TestCosineSimilarityPairs:
    def test_identical(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = cosine_similarity_pairs(a, a)
        assert np.allclose(result, [1.0, 1.0], atol=1e-6)

    def test_orthogonal(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result = cosine_similarity_pairs(a, b)
        assert np.isclose(result[0], 0.0, atol=1e-6)


class TestValidateInputs:
    def test_empty_prompts(self):
        with pytest.raises(ConfigurationError, match="non-empty"):
            validate_inputs([], [], None, None)

    def test_length_mismatch(self):
        with pytest.raises(ConfigurationError, match="equal length"):
            validate_inputs(["a"], ["b", "c"], None, None)

    def test_score_out_of_range(self):
        with pytest.raises(ConfigurationError, match="out of range"):
            validate_inputs(["a"], ["b"], [1.5], None)

    def test_score_not_numeric(self):
        with pytest.raises(ConfigurationError, match="numeric"):
            validate_inputs(["a"], ["b"], ["bad"], None)

    def test_valid_inputs(self):
        validate_inputs(["a", "b"], ["c", "d"], [0.5, 0.8], None)

    def test_valid_with_references(self):
        validate_inputs(["a"], ["b"], None, ["c"])


class TestBatchItems:
    def test_even_split(self):
        result = batch_items([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = batch_items([1, 2, 3], 2)
        assert result == [[1, 2], [3]]

    def test_empty(self):
        assert batch_items([], 5) == []
```

---

## Verification

After completing Day 1:
```bash
pip install -e ".[dev]"
pytest tests/test_utils.py -v
python -c "from faultmap import SliceAnalyzer, AnalysisReport; print('OK')"
```
