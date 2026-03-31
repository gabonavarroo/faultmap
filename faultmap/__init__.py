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
