# Architecture Reference

Quick reference for algorithms, data flows, and module contracts. Full code in PLAN-*.md files.

---

## analyze() Pipeline

```
inputs: prompts (list[str]), responses (list[str])
        + one of: scores (list[float]) | references (list[str]) | neither

1. VALIDATE        validate_inputs() → ConfigurationError if invalid
2. MODE DETECT     scores → Mode 1 | references → Mode 2 | neither → Mode 3
3. SCORE           scorer.score() → ScoringResult (scores in [0, 1])
4. BINARIZE        failure[i] = score[i] < failure_threshold
5. EARLY RETURN    if sum(failures) == 0 → return empty AnalysisReport
6. EMBED           embedder.embed(prompts) → (N, D) ndarray
7. CLUSTER         cluster_embeddings() → labels (N,), -1 = noise
8. TEST            for each cluster: test_cluster_failure_rate() → ClusterTestResult
9. BH CORRECT      benjamini_hochberg() → adjusted p-values
10. FILTER         keep clusters where adjusted_p < significance_level
11. NAME           label_clusters() via LLM (concurrent asyncio.gather)
12. ASSEMBLE       FailureSlice per significant cluster
13. RETURN         AnalysisReport(slices sorted by adjusted_p asc)
```

## audit_coverage() Pipeline

```
inputs: test_prompts, production_prompts

1. VALIDATE        lengths > 0
2. EMBED           embed both sets → (T, D) and (P, D)
3. NN DISTANCE     NearestNeighbors(k=1) fit on test, query prod → distances (P,)
4. THRESHOLD       auto: mean + 1.5*std if not provided
5. UNCOVERED       prod points where distance > threshold
6. CLUSTER GAPS    cluster_embeddings() on uncovered subset → gap_labels
7. REPRESENTATIVES top-k closest to centroid per gap
8. NAME            label_clusters() for each gap
9. RETURN          CoverageReport(gaps sorted by mean_distance desc)
```

---

## Module Contracts

### `llm.py — AsyncLLMClient`

```python
async def complete(messages, temperature=0.0, max_tokens=512) -> str
    # semaphore → litellm.acompletion → retry (3x, exponential backoff)
    # raises LLMError after max_retries

async def complete_batch(messages_list, temperature=0.0, desc, show_progress) -> list[str]
    # tqdm_asyncio.gather(complete(m) for m in messages_list)
    # preserves order
```

### `embeddings.py — Embedder`

```python
class Embedder(ABC):
    def embed(self, texts: list[str]) -> np.ndarray  # (n, d), float32
    def dimension(self) -> int

# LocalEmbedder: lazy load sentence-transformers model
# APIEmbedder: batched litellm.embedding() calls (batch_size=100)
# get_embedder(model_name) → auto-detect local vs API
```

Auto-detection rules for `get_embedder`:
- Starts with `all-MiniLM`, `all-mpnet`, `paraphrase-`, `multi-qa-`, `msmarco-` → `LocalEmbedder`
- Contains `/` (HuggingFace org format) → `LocalEmbedder`
- Otherwise → `APIEmbedder`

### `scoring/entropy.py — EntropyScorer`

```
per prompt i:
  samples_i = [sample_1 ... sample_n]          ← n LLM calls at temperature=1.0
  embs_i = embed(samples_i + [original_i])      ← (n+1, D)

  # semantic entropy
  sim_matrix = pairwise cosine sim of samples_i embeddings
  clusters = greedy_cluster(sim_matrix, threshold=consistency_threshold)
  p_k = |cluster_k| / n  for each cluster k
  H = -sum(p_k * log(p_k))
  H_norm = H / log(n), clipped to [0, 1]

  # self-consistency
  sims_to_original = cosine(samples_embs, original_emb)
  consistency = mean(sims_to_original >= consistency_threshold)

  score_i = 0.5 * (1 - H_norm) + 0.5 * consistency
```

### `slicing/statistics.py`

```
2x2 contingency table:
         Fail    Pass
Cluster    a      b    | a+b
Other      c      d    | c+d
         -----  -----
         a+c    b+d    | N

Expected cells: E_11 = (a+b)(a+c)/N
if E_11 < 5 or (N - E_11) < 5: Fisher exact (one-sided)
else: chi-squared with Yates correction

chi2 = N * (|ad-bc| - N/2)^2 / ((a+b)(c+d)(a+c)(b+d))
p_two = erfc(sqrt(chi2 / 2))  ← exact for df=1
p = p_two / 2 if a/（a+b) > c/(c+d) else 1.0

Fisher one-sided P(X >= a) via hypergeometric + lgamma:
  P(X=k) = C(K,k)*C(N-K,n-k)/C(N,n) in log space
  P(X>=a) = sum P(X=k) for k in [a, min(n,K)]
```

Benjamini-Hochberg:
```
sorted by p asc → rank 1..m
adj[i] = p[i] * m / rank[i]
enforce monotonicity: backward cummin
clip to [0, 1]
```

### `coverage/detector.py`

```python
def detect_coverage_gaps(
    test_embeddings,    # (T, D) float32
    prod_embeddings,    # (P, D) float32
    prod_prompts,       # list[str]
    distance_threshold=None,   # auto if None
    min_gap_size=5,
    clustering_method="hdbscan",
) -> tuple[np.ndarray, np.ndarray, float]:
    # returns: (gap_labels (P,), nn_distances (P,), threshold)
    # gap_labels: -1 = covered, 0..K = gap cluster id
```

---

## Data Models (frozen dataclasses)

```python
ScoringResult(scores, mode, metadata)

FailureSlice(
    name, description,
    size, failure_rate, baseline_rate, effect_size,   # effect_size = failure_rate / baseline_rate
    p_value, adjusted_p_value, test_used,
    sample_indices,           # ALL indices in cluster (into original prompts list)
    examples,                 # top-5 list[dict{prompt, response, score}]
    representative_prompts,   # top-5 closest to centroid
    cluster_id,
)

AnalysisReport(
    slices,                   # sorted by adjusted_p_value asc
    total_prompts, total_failures, baseline_failure_rate,
    significance_level, failure_threshold, scoring_mode,
    num_clusters_tested, num_significant, clustering_method, embedding_model,
    metadata,
)
→ __str__() calls format_analysis_report()
→ summary() returns 1-paragraph plain text
→ to_dict() returns JSON-serializable dict

CoverageGap(name, description, size, mean_distance, representative_prompts, prompt_indices, cluster_id)

CoverageReport(
    gaps,                     # sorted by mean_distance desc
    num_test_prompts, num_production_prompts, num_gaps,
    overall_coverage_score,   # 1 - (fraction of prod in gaps)
    distance_threshold, embedding_model, metadata,
)
→ same __str__, summary(), to_dict() pattern
```

---

## Exception Hierarchy

```
FaultmapError
├── EmbeddingError      ← model not found, missing sentence-transformers
├── ScoringError        ← scoring pipeline failures
├── LLMError            ← after max retries exhausted
├── ClusteringError     ← all points noise, too few points
└── ConfigurationError  ← invalid params, length mismatches
```

---

## Report Formatting

`report.py` tries `rich` first, falls back to plain text:

```
Plain text uses box-drawing chars:
╔══════════════════════════════════╗
║  faultmap Analysis Report        ║
╠══════════════════════════════════╣
║  Total prompts: 90               ║
║  Failure slices found: 1         ║
╚══════════════════════════════════╝

Slice #1: Legal compliance questions
─────────────────────────────────────
Failure rate:  72.0% (vs 33.3% baseline)
Effect size:   2.16x
p-value:       0.0003 (adjusted)
Samples:       30
Test used:     chi2
```

---

## Testing Fixtures (conftest.py)

```python
MockEmbedder       # deterministic hash-based embeddings, no model download, DIM=64
mock_llm_client    # AsyncMock returning "Name: Test Cluster\nDescription: ..."
clustered_data     # 3 clusters × 30 prompts, cluster 0 fails (score=0.2), others pass (0.8)
small_clustered_data  # 2 clusters × 15 prompts
coverage_data      # test in region A, prod in A (covered) + B (gap)
```

`make_clustered_data(n_clusters, n_per_cluster, dim, failure_clusters, failure_score, pass_score, seed)`
`make_coverage_data(n_test, n_prod_covered, n_prod_gap, dim, seed)`
