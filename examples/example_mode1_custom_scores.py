"""
Example: Mode 1 — Bring Your Own Scores

Use pre-computed scores from any evaluation framework (DeepEval, Ragas,
human reviewers, LLM-as-judge, etc.). faultmap discovers which input slices
have statistically elevated failure rates.

Requirements:
    pip install faultmap[rich]
    export OPENAI_API_KEY=...
"""

from faultmap import SliceAnalyzer

# ---------------------------------------------------------------------------
# Simulated evaluation data with a known failure pattern:
#   - Legal/compliance questions → scores around 0.2 (model struggles)
#   - Password/account reset questions → scores around 0.85
#   - Pricing questions → scores around 0.90
# ---------------------------------------------------------------------------

prompts = (
    [f"What are the legal requirements for {topic}?"
     for topic in ["GDPR", "HIPAA", "SOC2", "PCI-DSS", "CCPA"] * 6]
    + [f"How do I reset {item}?"
       for item in ["password", "2FA", "email", "phone", "settings"] * 6]
    + [f"What is the price of the {product} plan?"
       for product in ["Basic", "Pro", "Enterprise", "Team", "Starter"] * 6]
)

responses = [f"Here is information about: {p}" for p in prompts]

# Scores from your eval framework (higher = better, in [0, 1])
# Legal questions fail, others pass
scores = [0.2] * 30 + [0.85] * 30 + [0.90] * 30

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

analyzer = SliceAnalyzer(
    # model="nvidia_nim/meta/llama-3.3-70b-instruct",
    # embedding_model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
    # # Optional for asymmetric embedding APIs. Symmetric models can omit this.
    # embedding_usage_kwargs={"query": {"input_type": "query"}},
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    min_slice_size=10,
    failure_threshold=0.5,  # score < 0.5 = failure
    significance_level=0.05,
    clustering_method="hdbscan",
)

print("Analyzing failure slices...")
report = analyzer.analyze(prompts, responses, scores=scores)

# ---------------------------------------------------------------------------
# Inspect results
# ---------------------------------------------------------------------------

print(report)  # formatted output (rich if installed, else plain text)

print("\n--- Per-slice summary ---")
for s in report.slices:
    print(
        f"\nSlice: {s.name!r}\n"
        f"  Description: {s.description}\n"
        f"  Size: {s.size} prompts\n"
        f"  Failure rate: {s.failure_rate:.0%} "
        f"(vs {s.baseline_rate:.0%} baseline)\n"
        f"  Effect size: {s.effect_size:.1f}x worse than baseline\n"
        f"  Adjusted p-value: {s.adjusted_p_value:.4f}\n"
        f"  Test used: {s.test_used}\n"
        f"  Representative prompts:"
    )
    for p in s.representative_prompts[:3]:
        print(f"    - {p}")

print(f"\nBaseline failure rate: {report.baseline_failure_rate:.1%}")
print(f"Total failures: {report.total_failures}/{report.total_prompts}")

# ---------------------------------------------------------------------------
# Next step: Compare against another model
# ---------------------------------------------------------------------------
# If you have scores from a second model on the same prompts, you can follow
# up with a paired comparison to find which model wins on each failure slice:
#
# scores_b = [...]      # scores from Model B on the same prompts
# responses_b = [...]   # responses from Model B
#
# comparison = analyzer.compare_models(
#     prompts, responses, responses_b,
#     scores_a=scores, scores_b=scores_b,
#     model_a_name="Current Model", model_b_name="Candidate Model",
# )
# print(comparison)
#
# comparison.global_winner         # "a", "b", or "tie"
# comparison.global_advantage_rate # fraction of disagreements favoring A
# for s in comparison.slices:
#     print(s.name, s.winner, s.advantage_rate)
#
# See examples/example_model_comparison.py for a full walkthrough.
