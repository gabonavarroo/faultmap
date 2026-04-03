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
    CoverageGap,
    CoverageReport,
    FailureSlice,
    ScoringResult,
)

__version__ = "0.3.0"

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
