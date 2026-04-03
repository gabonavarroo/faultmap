"""
Example: Model Comparison

Compare two models on the same prompt set to understand where each one wins
and where each one struggles. faultmap uses McNemar's test on discordant pairs
to identify per-slice winners with statistical rigor.

This example simulates GPT-4o vs GPT-4o-mini on 90 prompts across 3 topics:
  - Legal/compliance:  GPT-4o excels (~0.85), GPT-4o-mini struggles (~0.25)
  - Billing disputes:  GPT-4o-mini excels (~0.90), GPT-4o struggles (~0.30)
  - Technical setup:   Both perform similarly (~0.80)

Requirements:
    pip install faultmap[rich]
    export OPENAI_API_KEY=...
"""

from faultmap import SliceAnalyzer

# ---------------------------------------------------------------------------
# Simulated evaluation data — 90 prompts, 3 semantic groups
# ---------------------------------------------------------------------------

legal_topics = ["GDPR", "HIPAA", "SOC2", "PCI-DSS", "CCPA"]
billing_topics = [
    "double charge",
    "failed payment",
    "refund request",
    "invoice error",
    "subscription cancel",
]
tech_topics = [
    "SSO setup",
    "webhook configuration",
    "API key rotation",
    "SAML integration",
    "OAuth flow",
]

prompts = (
    [f"What are the compliance requirements for {t}?" for t in legal_topics * 6]
    + [f"My {t} needs to be resolved urgently." for t in billing_topics * 6]
    + [f"How do I complete the {t}?" for t in tech_topics * 6]
)

# Responses from Model A (GPT-4o): strong on legal, weak on billing
responses_a = [f"[GPT-4o] Answer to: {p}" for p in prompts]

# Responses from Model B (GPT-4o-mini): weak on legal, strong on billing
responses_b = [f"[GPT-4o-mini] Answer to: {p}" for p in prompts]

# Pre-computed quality scores (higher = better, in [0, 1])
# Legal:   A passes (0.85), B fails (0.25)
# Billing: A fails (0.30),  B passes (0.90)
# Tech:    both pass (~0.80)
scores_a = [0.85] * 30 + [0.30] * 30 + [0.80] * 30
scores_b = [0.25] * 30 + [0.90] * 30 + [0.80] * 30

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

analyzer = SliceAnalyzer(
    # model="nvidia_nim/meta/llama-3.3-70b-instruct",
    # embedding_model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
    # embedding_usage_kwargs={"query": {"input_type": "query"}},
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    min_slice_size=10,
    failure_threshold=0.5,
    significance_level=0.05,
    clustering_method="hdbscan",
)

print("Comparing models...")
comparison = analyzer.compare_models(
    prompts=prompts,
    responses_a=responses_a,
    responses_b=responses_b,
    scores_a=scores_a,
    scores_b=scores_b,
    model_a_name="GPT-4o",
    model_b_name="GPT-4o-mini",
)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

print(comparison)  # formatted output (rich if installed, else plain text)

# ---------------------------------------------------------------------------
# Programmatic inspection
# ---------------------------------------------------------------------------

print("\n--- Global result ---")
print(f"Global winner:      {comparison.global_winner}")
print(f"Advantage rate:     {comparison.global_advantage_rate:.0%} of disagreements favor A")
print(f"Global p-value:     {comparison.global_p_value:.4f} ({comparison.global_test_used})")
print(f"Failure rate A:     {comparison.failure_rate_a:.1%}")
print(f"Failure rate B:     {comparison.failure_rate_b:.1%}")
print(f"Significant slices: {comparison.num_significant}")

print("\n--- Per-slice breakdown ---")
for s in comparison.slices:
    winner_label = (
        f"{comparison.model_a_name} wins"
        if s.winner == "a"
        else f"{comparison.model_b_name} wins"
        if s.winner == "b"
        else "tie"
    )
    print(
        f"\nSlice: {s.name!r}  [{winner_label}]\n"
        f"  Description:      {s.description}\n"
        f"  Size:             {s.size} prompts\n"
        f"  Failure rate A:   {s.failure_rate_a:.0%}\n"
        f"  Failure rate B:   {s.failure_rate_b:.0%}\n"
        f"  Discordant:       A wins {s.discordant_a_wins}, B wins {s.discordant_b_wins}\n"
        f"  Advantage rate:   {s.advantage_rate:.0%} of disagreements favor A\n"
        f"  Adj. p-value:     {s.adjusted_p_value:.4f} ({s.test_used})\n"
        f"  Representative prompts:"
    )
    for p in s.representative_prompts[:3]:
        print(f"    - {p}")

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

print("\n--- Interpretation ---")
a_slices = [s for s in comparison.slices if s.winner == "a"]
b_slices = [s for s in comparison.slices if s.winner == "b"]

if a_slices:
    print(f"Use {comparison.model_a_name} for: {', '.join(s.name for s in a_slices)}")
if b_slices:
    print(f"Use {comparison.model_b_name} for: {', '.join(s.name for s in b_slices)}")
if not a_slices and not b_slices:
    print("No statistically significant per-slice differences — models are equivalent.")

# Export to JSON
comparison_json = comparison.to_dict()
print(f"\nJSON keys: {list(comparison_json.keys())}")
# json.dumps(comparison_json)  # fully serializable
