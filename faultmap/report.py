"""Report formatting — stub for Phase 1."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AnalysisReport, CoverageReport


def format_analysis_report(report: "AnalysisReport") -> str:
    return report.summary()


def format_coverage_report(report: "CoverageReport") -> str:
    return report.summary()
