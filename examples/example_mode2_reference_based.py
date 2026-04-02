"""
Example: Mode 2 — Reference-Based Scoring

Provide ground-truth answers. faultmap scores each response by computing
cosine similarity between the response embedding and the reference embedding.
Higher similarity = better response.

This mode is well-suited for:
  - RAG evaluation (compare retrieved+generated answer to gold answer)
  - Q&A benchmarks where reference answers exist
  - Translation quality (embed source and translation)

Requirements:
    pip install faultmap[rich]
    export OPENAI_API_KEY=...
"""

from faultmap import SliceAnalyzer

# ---------------------------------------------------------------------------
# Simulated Q&A evaluation data.
# The "technical" questions have poor responses (off-topic or vague),
# while factual and general questions are answered well.
# ---------------------------------------------------------------------------

prompts = [
    # Factual — well answered
    "What is the capital of France?",
    "What is the boiling point of water at sea level?",
    "Who wrote Pride and Prejudice?",
    "What year did World War II end?",
    "What is the speed of light in a vacuum?",
    "What is the chemical formula for table salt?",
    "How many continents are there on Earth?",
    "What is the largest planet in the solar system?",
    "What language is spoken in Brazil?",
    "What is the currency of Japan?",

    # Technical — poorly answered (vague or incorrect)
    "How does gradient descent work in neural networks?",
    "What is the difference between TCP and UDP?",
    "Explain what a transformer architecture is.",
    "What is the bias-variance tradeoff?",
    "How does HTTPS encryption work?",
    "What is a race condition in concurrent programming?",
    "Explain the CAP theorem.",
    "What does O(n log n) complexity mean?",
    "How does backpropagation calculate gradients?",
    "What is a deadlock in operating systems?",

    # General knowledge — well answered
    "What is photosynthesis?",
    "How do vaccines work?",
    "What causes rainbows?",
    "What is inflation in economics?",
    "How do airplanes generate lift?",
] * 2  # repeat to have 50 prompts total

responses = [
    # Factual — good responses
    "Paris is the capital of France.",
    "Water boils at 100 degrees Celsius (212°F) at sea level.",
    "Pride and Prejudice was written by Jane Austen.",
    "World War II ended in 1945.",
    "The speed of light is approximately 299,792,458 meters per second.",
    "The chemical formula for table salt is NaCl.",
    "There are 7 continents on Earth.",
    "Jupiter is the largest planet in our solar system.",
    "Portuguese is spoken in Brazil.",
    "The currency of Japan is the Japanese Yen (JPY).",

    # Technical — poor responses (vague)
    "It's a method used in machine learning.",
    "They are both network protocols.",
    "It's a type of neural network architecture.",
    "It's something about model accuracy.",
    "It's a security thing for websites.",
    "It happens in programs sometimes.",
    "It's related to distributed systems.",
    "It describes how fast an algorithm is.",
    "It's how neural networks learn.",
    "It's when programs get stuck.",

    # General — good responses
    "Photosynthesis is the process plants use to convert sunlight into chemical energy.",
    "Vaccines train the immune system to recognize and fight specific pathogens.",
    "Rainbows form when sunlight refracts and reflects inside water droplets.",
    "Inflation is the general increase in prices and fall in purchasing power over time.",
    "Airplanes generate lift through the Bernoulli effect and wing angle of attack.",
] * 2  # repeat to match prompts

references = [
    # Factual references
    "The capital of France is Paris.",
    "Water boils at 100°C or 212°F at standard sea-level pressure (1 atm).",
    "Jane Austen wrote Pride and Prejudice, published in 1813.",
    "World War II ended in 1945 with Germany surrendering in May and Japan in September.",
    "The speed of light in a vacuum is exactly 299,792,458 m/s.",
    "NaCl (sodium chloride) is the chemical formula for common table salt.",
    (
        "Earth has 7 continents: Africa, Antarctica, Asia, Australia, Europe, "
        "North America, South America."
    ),
    (
        "Jupiter is the largest planet, with a mass more than twice that of all "
        "other planets combined."
    ),
    "Brazil's official language is Portuguese, a legacy of Portuguese colonization.",
    "Japan's official currency is the Yen (¥), issued by the Bank of Japan.",

    # Technical references (detailed correct answers)
    (
        "Gradient descent minimizes a loss function by iteratively updating "
        "parameters in the direction of the negative gradient, scaled by the "
        "learning rate."
    ),
    (
        "TCP provides reliable, ordered, connection-based delivery with error "
        "checking; UDP is connectionless and faster but offers no delivery "
        "guarantees."
    ),
    (
        "The transformer architecture uses self-attention mechanisms to weigh "
        "the importance of different tokens when encoding sequences, enabling "
        "parallelization unlike RNNs."
    ),
    (
        "The bias-variance tradeoff describes the tension between underfitting "
        "(high bias) and overfitting (high variance) — reducing one typically "
        "increases the other."
    ),
    (
        "HTTPS encrypts HTTP traffic using TLS: the server presents a "
        "certificate, a shared symmetric key is negotiated via asymmetric "
        "crypto, and data is encrypted with that key."
    ),
    (
        "A race condition occurs when program behavior depends on the relative "
        "timing of uncontrollable events like thread scheduling, often causing "
        "incorrect results."
    ),
    (
        "The CAP theorem states that a distributed system can only guarantee "
        "two of three properties simultaneously: Consistency, Availability, "
        "and Partition tolerance."
    ),
    (
        "O(n log n) means the algorithm's runtime grows proportionally to n "
        "times log(n) — typical of efficient sorting algorithms like "
        "mergesort and heapsort."
    ),
    (
        "Backpropagation uses the chain rule of calculus to compute partial "
        "derivatives of the loss with respect to each weight, propagating "
        "gradients backward through layers."
    ),
    (
        "A deadlock occurs when two or more processes each hold a resource the "
        "other needs and none can proceed, creating a circular wait."
    ),

    # General references
    (
        "Photosynthesis converts light energy into chemical energy stored as "
        "glucose: 6CO2 + 6H2O + light → C6H12O6 + 6O2."
    ),
    (
        "Vaccines introduce antigens (weakened/inactivated pathogens or mRNA "
        "instructions) that trigger the immune system to produce antibodies "
        "without causing disease."
    ),
    (
        "Rainbows occur when white sunlight enters a water droplet, refracts, "
        "reflects off the back, and refracts again on exit — splitting into "
        "spectral colors."
    ),
    (
        "Inflation measures the rate at which the general price level of goods "
        "and services rises, eroding purchasing power; tracked by indices "
        "like CPI."
    ),
    (
        "Aircraft wings are shaped (airfoil) so air moves faster over the top, "
        "reducing pressure above the wing, while the wing's angle of attack "
        "deflects air downward."
    ),
] * 2

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

analyzer = SliceAnalyzer(
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    min_slice_size=5,
    failure_threshold=0.5,  # cosine similarity < 0.5 = failure
    significance_level=0.05,
)

print("Analyzing with reference-based scoring...")
report = analyzer.analyze(prompts, responses, references=references)

print(report)

print("\n--- Detected failure slices ---")
if report.slices:
    for s in report.slices:
        print(f"\n{s.name}: {s.failure_rate:.0%} failure rate "
              f"(baseline {s.baseline_rate:.0%}, {s.effect_size:.1f}x worse)")
        print(f"  Cluster: {s.size} prompts, adj. p={s.adjusted_p_value:.4f}")
else:
    print("No statistically significant failure slices found.")

print(f"\nScoring mode: {report.scoring_mode}")
print(f"Total prompts: {report.total_prompts}, failures: {report.total_failures}")
