from __future__ import annotations

from .models import AnalysisReport, CoverageReport


def format_analysis_report(report: AnalysisReport) -> str:
    """Format AnalysisReport. Try rich, fall back to plain text."""
    try:
        return _format_analysis_rich(report)
    except ImportError:
        return _format_analysis_plain(report)


def format_coverage_report(report: CoverageReport) -> str:
    """Format CoverageReport. Try rich, fall back to plain text."""
    try:
        return _format_coverage_rich(report)
    except ImportError:
        return _format_coverage_plain(report)


def _format_analysis_plain(report: AnalysisReport) -> str:
    sep = "=" * 55
    thin = "-" * 55
    lines = [
        sep,
        "FAULTMAP ANALYSIS REPORT",
        sep,
        f"Total prompts:     {report.total_prompts}",
        f"Total failures:    {report.total_failures} ({report.baseline_failure_rate:.1%})",
        f"Scoring mode:      {report.scoring_mode}",
        f"Clustering:        {report.clustering_method}",
        f"Embedding model:   {report.embedding_model}",
        f"Significance:      alpha={report.significance_level}",
        f"Clusters tested:   {report.num_clusters_tested}",
        f"Significant:       {report.num_significant}",
    ]

    if not report.slices:
        lines.append(thin)
        lines.append("No statistically significant failure slices found.")
    else:
        for i, s in enumerate(report.slices, 1):
            lines.append(thin)
            lines.append(f'Slice {i}: "{s.name}"')
            lines.append(f"  Description:    {s.description}")
            lines.append(f"  Size:           {s.size} prompts")
            lines.append(f"  Failure rate:   {s.failure_rate:.1%} (vs {s.baseline_rate:.1%} baseline)")
            lines.append(f"  Effect size:    {s.effect_size:.1f}x")
            lines.append(f"  Adj. p-value:   {s.adjusted_p_value:.6f} ({s.test_used})")
            lines.append(f"  Examples:")
            for prompt in s.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_analysis_rich(report: AnalysisReport) -> str:
    from rich.console import Console
    from rich.table import Table
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=120)

    console.print("[bold]FAULTMAP ANALYSIS REPORT[/bold]", style="cyan")
    console.print(
        f"Prompts: {report.total_prompts} | "
        f"Failures: {report.total_failures} ({report.baseline_failure_rate:.1%}) | "
        f"Mode: {report.scoring_mode} | "
        f"Clustering: {report.clustering_method}"
    )
    console.print(
        f"Clusters tested: {report.num_clusters_tested} | "
        f"Significant: {report.num_significant} | "
        f"Alpha: {report.significance_level}"
    )

    if report.slices:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Name", width=25)
        table.add_column("Size", width=6, justify="right")
        table.add_column("Fail Rate", width=10, justify="right")
        table.add_column("Baseline", width=10, justify="right")
        table.add_column("Effect", width=8, justify="right")
        table.add_column("Adj. p", width=10, justify="right")
        table.add_column("Test", width=6)

        for i, s in enumerate(report.slices, 1):
            table.add_row(
                str(i), s.name, str(s.size),
                f"{s.failure_rate:.1%}", f"{s.baseline_rate:.1%}",
                f"{s.effect_size:.1f}x", f"{s.adjusted_p_value:.4f}",
                s.test_used,
            )
        console.print(table)

        for i, s in enumerate(report.slices, 1):
            console.print(f"\n[bold]Slice {i}: {s.name}[/bold]")
            console.print(f"  {s.description}")
            console.print("  Examples:")
            for p in s.representative_prompts[:3]:
                truncated = p[:100] + ("..." if len(p) > 100 else "")
                console.print(f"    - {truncated}", style="dim")
    else:
        console.print("[green]No statistically significant failure slices found.[/green]")

    return console.file.getvalue()


def _format_coverage_plain(report: CoverageReport) -> str:
    sep = "=" * 55
    thin = "-" * 55
    lines = [
        sep,
        "FAULTMAP COVERAGE REPORT",
        sep,
        f"Test prompts:        {report.num_test_prompts}",
        f"Production prompts:  {report.num_production_prompts}",
        f"Coverage score:      {report.overall_coverage_score:.1%}",
        f"Distance threshold:  {report.distance_threshold:.4f}",
        f"Embedding model:     {report.embedding_model}",
        f"Gaps found:          {report.num_gaps}",
    ]

    if not report.gaps:
        lines.append(thin)
        lines.append("No significant coverage gaps found.")
    else:
        for i, g in enumerate(report.gaps, 1):
            lines.append(thin)
            lines.append(f'Gap {i}: "{g.name}"')
            lines.append(f"  Description:     {g.description}")
            lines.append(f"  Size:            {g.size} prompts")
            lines.append(f"  Mean distance:   {g.mean_distance:.4f}")
            lines.append(f"  Examples:")
            for prompt in g.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_coverage_rich(report: CoverageReport) -> str:
    from rich.console import Console
    from rich.table import Table
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=120)
    console.print("[bold]FAULTMAP COVERAGE REPORT[/bold]", style="cyan")
    console.print(
        f"Test: {report.num_test_prompts} | "
        f"Production: {report.num_production_prompts} | "
        f"Coverage: {report.overall_coverage_score:.1%}"
    )

    if report.gaps:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Name", width=25)
        table.add_column("Size", width=6, justify="right")
        table.add_column("Mean Dist", width=10, justify="right")

        for i, g in enumerate(report.gaps, 1):
            table.add_row(str(i), g.name, str(g.size), f"{g.mean_distance:.4f}")
        console.print(table)
    else:
        console.print("[green]No significant coverage gaps found.[/green]")

    return console.file.getvalue()
