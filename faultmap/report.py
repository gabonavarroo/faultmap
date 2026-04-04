from __future__ import annotations

from .models import AnalysisReport, ComparisonReport, CoverageReport


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
            lines.append(
                f"  Failure rate:   {s.failure_rate:.1%} "
                f"(vs {s.baseline_rate:.1%} baseline)"
            )
            lines.append(f"  Effect size:    {s.effect_size:.1f}x")
            lines.append(f"  Adj. p-value:   {s.adjusted_p_value:.6f} ({s.test_used})")
            lines.append("  Examples:")
            for prompt in s.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_analysis_rich(report: AnalysisReport) -> str:
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

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
    unclustered = int(report.metadata.get("num_unclustered_uncovered", 0))
    total_uncovered = report.metadata.get("num_uncovered_total")
    if total_uncovered is not None:
        lines.append(f"Uncovered prompts:   {int(total_uncovered)}")
    if unclustered:
        lines.append(f"Unclustered:         {unclustered}")

    if not report.gaps:
        lines.append(thin)
        if unclustered:
            lines.append(
                "No reportable coverage gap clusters found, "
                "but some prompts remain uncovered."
            )
        else:
            lines.append("No significant coverage gaps found.")
    else:
        for i, g in enumerate(report.gaps, 1):
            lines.append(thin)
            lines.append(f'Gap {i}: "{g.name}"')
            lines.append(f"  Description:     {g.description}")
            lines.append(f"  Size:            {g.size} prompts")
            lines.append(f"  Mean distance:   {g.mean_distance:.4f}")
            lines.append("  Examples:")
            for prompt in g.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_coverage_rich(report: CoverageReport) -> str:
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

    console = Console(file=StringIO(), force_terminal=True, width=120)
    console.print("[bold]FAULTMAP COVERAGE REPORT[/bold]", style="cyan")
    console.print(
        f"Test: {report.num_test_prompts} | "
        f"Production: {report.num_production_prompts} | "
        f"Coverage: {report.overall_coverage_score:.1%}"
    )
    total_uncovered = report.metadata.get("num_uncovered_total")
    unclustered = int(report.metadata.get("num_unclustered_uncovered", 0))
    if total_uncovered is not None:
        console.print(
            f"Uncovered: {int(total_uncovered)} | "
            f"Named gaps: {report.num_gaps} | "
            f"Unclustered: {unclustered}"
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
        if unclustered:
            console.print(
                "[yellow]No reportable gap clusters found, "
                "but some prompts remain uncovered.[/yellow]"
            )
        else:
            console.print("[green]No significant coverage gaps found.[/green]")

    return console.file.getvalue()


# ---------------------------------------------------------------------------
# Comparison report formatting
# ---------------------------------------------------------------------------


def format_comparison_report(report: ComparisonReport) -> str:
    """Format ComparisonReport. Try rich, fall back to plain text."""
    try:
        return _format_comparison_rich(report)
    except ImportError:
        return _format_comparison_plain(report)


def _format_comparison_plain(report: ComparisonReport) -> str:
    """Full plain-text formatter for ComparisonReport."""
    sep = "=" * 55
    thin = "-" * 55

    def _winner_label(winner: str) -> str:
        if winner == "a":
            return f"{report.model_a_name} (Model A)"
        if winner == "b":
            return f"{report.model_b_name} (Model B)"
        return "tie"

    lines = [
        sep,
        "FAULTMAP MODEL COMPARISON REPORT",
        sep,
        f"Model A:           {report.model_a_name}",
        f"Model B:           {report.model_b_name}",
        f"Total prompts:     {report.total_prompts}",
        f"Scoring mode:      {report.scoring_mode}",
        f"Clustering:        {report.clustering_method}",
        f"Embedding model:   {report.embedding_model}",
        f"Significance:      alpha={report.significance_level}",
        thin,
        "GLOBAL COMPARISON",
        thin,
        f"Failure rate (A):  {report.failure_rate_a:.1%}",
        f"Failure rate (B):  {report.failure_rate_b:.1%}",
        f"Winner:            {_winner_label(report.global_winner)}",
        (
            f"Advantage rate:    {report.global_advantage_rate:.2f} "
            f"({report.global_advantage_rate:.0%} of disagreements favor A)"
        ),
        f"McNemar p-value:   {report.global_p_value:.4f} ({report.global_test_used})",
        thin,
        f"Clusters tested:   {report.num_clusters_tested}",
        f"Significant:       {report.num_significant}",
    ]

    if not report.slices:
        lines.append(thin)
        lines.append("No statistically significant per-slice differences found.")
    else:
        for i, s in enumerate(report.slices, 1):
            lines.append(thin)
            if s.winner == "a":
                winner_tag = f"** {report.model_a_name} wins **"
            elif s.winner == "b":
                winner_tag = f"** {report.model_b_name} wins **"
            else:
                winner_tag = "** tie **"
            lines.append(f'Slice {i}: "{s.name}" {winner_tag}')
            lines.append(f"  Description:    {s.description}")
            lines.append(f"  Size:           {s.size} prompts")
            lines.append(
                f"  Failure rate:   A={s.failure_rate_a:.1%} vs B={s.failure_rate_b:.1%}"
            )
            lines.append(
                f"  Discordant:     A wins {s.discordant_a_wins}, B wins {s.discordant_b_wins}"
            )
            lines.append(f"  Advantage rate: {s.advantage_rate:.2f}")
            lines.append(f"  Adj. p-value:   {s.adjusted_p_value:.4f} ({s.test_used})")
            prompts_to_show = s.representative_prompts[:5]
            if prompts_to_show:
                lines.append("  Examples:")
                for prompt in prompts_to_show:
                    truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                    lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_comparison_rich(report: ComparisonReport) -> str:
    """Rich-formatted ComparisonReport with summary table and per-slice details."""
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

    console = Console(file=StringIO(), force_terminal=True, width=120)

    def _winner_name(winner: str) -> str:
        if winner == "a":
            return report.model_a_name
        if winner == "b":
            return report.model_b_name
        return "tie"

    console.print("[bold]FAULTMAP MODEL COMPARISON REPORT[/bold]", style="cyan")
    console.print(
        f"Model A: {report.model_a_name} | "
        f"Model B: {report.model_b_name} | "
        f"Prompts: {report.total_prompts} | "
        f"Mode: {report.scoring_mode} | "
        f"Clustering: {report.clustering_method}"
    )
    console.print(
        f"Global winner: {_winner_name(report.global_winner)} | "
        f"Adv rate: {report.global_advantage_rate:.2f} | "
        f"Global p: {report.global_p_value:.4f} | "
        f"Fail A: {report.failure_rate_a:.1%} | "
        f"Fail B: {report.failure_rate_b:.1%}"
    )
    console.print(
        f"Clusters tested: {report.num_clusters_tested} | "
        f"Significant: {report.num_significant} | "
        f"Alpha: {report.significance_level}"
    )

    if report.slices:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Name", width=22)
        table.add_column("Size", width=6, justify="right")
        table.add_column("Fail A", width=8, justify="right")
        table.add_column("Fail B", width=8, justify="right")
        table.add_column("Winner", width=12)
        table.add_column("Adv Rate", width=9, justify="right")
        table.add_column("Adj. p", width=8, justify="right")

        for i, s in enumerate(report.slices, 1):
            table.add_row(
                str(i),
                s.name[:20],
                str(s.size),
                f"{s.failure_rate_a:.1%}",
                f"{s.failure_rate_b:.1%}",
                _winner_name(s.winner)[:11],
                f"{s.advantage_rate:.2f}",
                f"{s.adjusted_p_value:.4f}",
            )
        console.print(table)

        for i, s in enumerate(report.slices, 1):
            console.print(f"\n[bold]Slice {i}: {s.name}[/bold]")
            console.print(f"  {s.description}")
            console.print(
                f"  Failure rate: A={s.failure_rate_a:.1%} vs B={s.failure_rate_b:.1%} | "
                f"Discordant: A wins {s.discordant_a_wins}, B wins {s.discordant_b_wins}"
            )
            if s.representative_prompts:
                console.print("  Examples:")
                for p in s.representative_prompts[:3]:
                    truncated = p[:100] + ("..." if len(p) > 100 else "")
                    console.print(f"    - {truncated}", style="dim")
    else:
        console.print(
            "[green]No statistically significant per-slice differences found.[/green]"
        )

    return console.file.getvalue()
