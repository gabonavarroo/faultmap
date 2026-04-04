"""faultmap: Automatically discover where and why your LLM is failing."""

from .analyzer import SliceAnalyzer
from .exceptions import (
    ClusteringError,
    ConfigurationError,
    EmbeddingError,
    FaultmapError,
    LLMError,
    ScoringError,
)
from .models import (
    AnalysisReport,
    ComparisonReport,
    CoverageGap,
    CoverageReport,
    FailureSlice,
    ScoringResult,
    SliceComparison,
)

__version__ = "0.4.1"

__all__ = [
    "SliceAnalyzer",
    "AnalysisReport",
    "FailureSlice",
    "CoverageReport",
    "CoverageGap",
    "ScoringResult",
    "ComparisonReport",
    "SliceComparison",
    "FaultmapError",
    "EmbeddingError",
    "ScoringError",
    "LLMError",
    "ClusteringError",
    "ConfigurationError",
    "__version__",
]
