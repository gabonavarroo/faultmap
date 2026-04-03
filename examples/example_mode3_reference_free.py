"""
Example: Mode 3 — Reference-Free (Fully Autonomous)

No ground truth needed. faultmap estimates response reliability by:
  1. Sampling n_samples additional responses per prompt at high temperature
  2. Measuring semantic entropy: how spread out are the responses in embedding space?
     - High entropy = model gives very different answers each time = uncertain
     - Low entropy = model is consistent = likely reliable
  3. Measuring self-consistency: what fraction of samples are similar to the original?
  4. Score = 0.5 * (1 - normalized_entropy) + 0.5 * self_consistency

This mode is ideal for:
  - Discovering unknown unknowns when you have no ground truth
  - Identifying topics where the model hallucinates or is uncertain
  - Exploratory analysis of a new model or domain

NOTE: Mode 3 makes (n_samples + 1) LLM API calls per prompt — costs scale with
dataset size. Use a small n_samples (4-8) for cost efficiency, or run on a sample
of your data first.

Requirements:
    pip install faultmap[rich]
    export OPENAI_API_KEY=...
"""

from faultmap import SliceAnalyzer

# ---------------------------------------------------------------------------
# Mixed prompts: some factual (should be consistent), some speculative
# (should be inconsistent), and some genuinely uncertain.
# ---------------------------------------------------------------------------

prompts = [
    # Factual — model should be consistent
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "What year did the Berlin Wall fall?",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    "What planet is closest to the sun?",
    "What language is spoken in Brazil?",
    "How many sides does a hexagon have?",

    # Speculative / opinion — model should vary
    "What will AI be capable of in 2035?",
    "Should companies adopt a 4-day workweek?",
    "What is the best programming language to learn in 2030?",
    "Will quantum computers replace classical computers?",
    "What is the most important invention of the 21st century?",
    "Should social media platforms be regulated by governments?",
    "Is remote work better than office work for productivity?",
    "What will the stock market do next year?",
    "What career should a recent computer science graduate pursue?",
    "Which country will lead AI development in 2040?",

    # Ambiguous / knowledge boundary — mixed consistency expected
    "What causes long COVID?",
    "Is dark matter made of undiscovered particles?",
    "Can consciousness be fully explained by neuroscience?",
    "What is the best diet for longevity?",
    "Does intermittent fasting improve cognitive function?",
]

responses = [
    # Factual
    "Paris is the capital of France.",
    "2 + 2 equals 4.",
    "William Shakespeare wrote Romeo and Juliet.",
    "Au is the chemical symbol for gold.",
    "The Berlin Wall fell in 1989.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "Mercury is the closest planet to the sun.",
    "Portuguese is spoken in Brazil.",
    "A hexagon has 6 sides.",

    # Speculative
    "AI will be able to perform most cognitive tasks better than humans.",
    "A 4-day workweek could improve employee wellbeing and productivity.",
    "Python and Rust are likely to remain important languages.",
    "Quantum computers will excel at specific tasks but not replace classical ones entirely.",
    "The internet remains the most transformative invention of the 21st century.",
    "Some regulation is necessary to prevent misinformation and protect users.",
    "It depends on the individual and the nature of the work.",
    "Predictions are inherently uncertain given market complexity.",
    "AI and machine learning fields offer strong growth opportunities.",
    "The US and China are currently leading, but others may catch up.",

    # Ambiguous
    "Long COVID appears to involve immune dysregulation and viral persistence.",
    "Current evidence points to weakly interacting massive particles (WIMPs).",
    "Neuroscience can explain many aspects of consciousness but not all.",
    "A Mediterranean-style diet shows strong evidence for longevity.",
    "Some studies suggest benefits, but evidence is mixed.",
]

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

analyzer = SliceAnalyzer(
    # model="nvidia_nim/meta/llama-3.3-70b-instruct",
    # embedding_model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
    # Optional for asymmetric embedding APIs. Symmetric models can omit this.
    # embedding_usage_kwargs={"query": {"input_type": "query"}},
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    n_samples=6,           # sample 6 additional responses per prompt
    temperature=1.0,       # high temperature for diverse sampling
    consistency_threshold=0.8,  # cosine similarity threshold for self-consistency
    min_slice_size=5,
    failure_threshold=0.5,
    significance_level=0.05,
)

print("Running autonomous scoring (this makes LLM API calls)...")
print(f"Expected API calls: {len(prompts)} prompts × {analyzer.n_samples} samples "
      f"= {len(prompts) * analyzer.n_samples} calls for scoring\n")

report = analyzer.analyze(prompts, responses)
# report = analyzer.analyze(prompts[:3], responses[:3])  # smaller dev run
# No scores or references — Mode 3 is triggered automatically

print(report)

# Inspect the scoring metadata (entropy + consistency per prompt)
scoring_meta = report.metadata.get("scoring_metadata", {})
if "semantic_entropy" in scoring_meta:
    entropy = scoring_meta["semantic_entropy"]
    consistency = scoring_meta["self_consistency"]
    scores = scoring_meta.get("scores", [])
    print("\n--- Scoring metadata (first 10 prompts) ---")
    for i, (p, e, c, s) in enumerate(
        zip(prompts[:10], entropy[:10], consistency[:10], scores[:10])
    ):
        print(f"  [{i:2d}] entropy={e:.2f}  consistency={c:.2f}  score={s:.2f}  "
              f"| {p[:60]}")

print(f"\nScoring mode: {report.scoring_mode}")
print(f"Detected slices: {report.num_significant}")
