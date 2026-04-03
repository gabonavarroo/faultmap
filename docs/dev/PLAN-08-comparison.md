# Model Comparison Feature — Implementation Plan (faultmap v0.4.0)

## Context

faultmap currently answers "where does my LLM fail?" (`analyze()`) and "what is my test suite missing?" (`audit_coverage()`). The natural third question is **"which model is better, and where?"** — comparing two models (or configurations) on the same prompt set to understand per-slice performance differences with statistical rigor.

This plan adds `compare_models()` as a third public method on `SliceAnalyzer`, following every established pattern: frozen dataclasses, async internals with sync wrapper, McNemar's test (the correct paired-binary test), BH FDR correction, LLM-based cluster naming, and rich + plain text reporting.

---

## A. Public API

### Method signature on `SliceAnalyzer`

```python
def compare_models(
    self,
    prompts: list[str],
    responses_a: list[str],
    responses_b: list[str],
    *,
    scores_a: list[float] | None = None,
    scores_b: list[float] | None = None,
    references: list[str] | None = None,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> ComparisonReport:
```

**Mode detection** (mirrors `analyze()` exactly):
- `scores_a` + `scores_b` provided → **Mode 1** (precomputed). Must both be present or neither; asymmetric raises `ConfigurationError`.
- `references` provided (no scores) → **Mode 2**. Score each model's responses against shared references via `ReferenceScorer`.
- Neither → **Mode 3**. Run `EntropyScorer` independently on each model's responses.
- Both scores and references → Mode 1 wins with `UserWarning` (existing pattern).

**Key design decision**: Prompts are shared. Embeddings are computed once. Clustering is done once. Both models are evaluated on the *same* slices. This is critical — comparing models on different slices would be meaningless.

### Usage examples

```python
# Mode 1: precomputed scores
comparison = analyzer.compare_models(
    prompts, gpt4_responses, gpt4mini_responses,
    scores_a=gpt4_scores, scores_b=gpt4mini_scores,
    model_a_name="GPT-4o", model_b_name="GPT-4o-mini",
)
print(comparison)

# Mode 2: reference-based
comparison = analyzer.compare_models(
    prompts, gpt4_responses, gpt4mini_responses,
    references=ground_truth,
    model_a_name="GPT-4o", model_b_name="GPT-4o-mini",
)

# Mode 3: reference-free (entropy scoring for each model)
comparison = analyzer.compare_models(
    prompts, gpt4_responses, gpt4mini_responses,
    model_a_name="GPT-4o", model_b_name="GPT-4o-mini",
)
```

---

## B. Statistical Design

### Why McNemar's test

The data is **paired binary**: each prompt produces (A pass/fail, B pass/fail). McNemar's test is the standard test for this structure. It uses only **discordant pairs** (where the models disagree):

```
                  Model B Pass    Model B Fail
Model A Pass:        a               b          (b = "A wins")
Model A Fail:        c               d          (c = "B wins")
```

Only `b` and `c` carry information about which model is better.

### Test statistic

**Chi-squared approximation** (when `b + c >= 25`):
```
chi2 = (|b - c| - 1)^2 / (b + c)     # continuity correction
p = erfc(sqrt(chi2 / 2))              # two-sided, exact for df=1
```

This uses the **exact same** `erfc(sqrt(chi2/2))` pattern already in `slicing/statistics.py:_chi2_yates`. No new math.

**Exact binomial fallback** (when `b + c < 25`):
Under H0, `b ~ Binomial(b+c, 0.5)`. Two-sided p-value via `math.lgamma` (same pattern as Fisher exact). This handles small-sample slices correctly.

### Per-slice testing

For each cluster, extract the subset of prompts and run McNemar's on that subset. This produces per-slice p-values.

### Multiple comparison correction

Apply **Benjamini-Hochberg FDR** across all per-slice p-values. The algorithm is identical to the existing `benjamini_hochberg()` in `slicing/statistics.py` (20 lines). A parallel function `benjamini_hochberg_comparison()` operates on `ComparisonTestResult` objects rather than `ClusterTestResult` — keeping types separate avoids modifying the existing tested interface.

### Effect size

**Advantage rate** = `b / (b + c)` — the proportion of disagreements where model A wins.
- `> 0.5` → A is better
- `< 0.5` → B is better  
- `= 0.5` → tied

This is intuitive and directly actionable ("78% of the time the models disagree, GPT-4o gives the better answer").

### Winner determination

- `adjusted_p_value < significance_level` AND `advantage_rate > 0.5` → **A wins**
- `adjusted_p_value < significance_level` AND `advantage_rate < 0.5` → **B wins**
- Otherwise → **tie** (no statistically significant difference)

### Edge cases

| Case | Handling |
|------|----------|
| No discordant pairs (b=0, c=0) | p=1.0, test_used="none", winner="tie" |
| All discordant one direction | Highly significant, exact binomial used if small |
| Very small slice | `min_slice_size` filters before testing (existing behavior) |
| Both models fail everything | All concordant-fail, no discordant pairs → tie |
| Both models pass everything | All concordant-pass, no discordant pairs → tie |
| No failures in either model | Early return with empty slices (report still has global stats) |

### No new dependencies

Everything implementable with `math.erfc`, `math.lgamma`, `math.exp`, `math.sqrt` — all stdlib. Follows the existing "no scipy, no statsmodels" constraint.

---

## C. Data Models

All frozen dataclasses following established patterns.

### `SliceComparison` (analogous to `FailureSlice`)

```python
@dataclass(frozen=True)
class SliceComparison:
    name: str                      # LLM-generated cluster name
    description: str               # LLM-generated description
    size: int                      # Prompts in this slice
    failure_rate_a: float          # Model A failure rate in slice
    failure_rate_b: float          # Model B failure rate in slice
    failure_rate_diff: float       # failure_rate_a - failure_rate_b
    concordant_pass: int           # Both pass (cell a)
    concordant_fail: int           # Both fail (cell d)
    discordant_a_wins: int         # A pass, B fail (cell b)
    discordant_b_wins: int         # A fail, B pass (cell c)
    advantage_rate: float          # b / (b+c), 0.5 if no discordant
    p_value: float                 # Raw McNemar p-value
    adjusted_p_value: float        # BH-corrected p-value
    test_used: str                 # "mcnemar_chi2" | "mcnemar_exact" | "none"
    winner: str                    # "a" | "b" | "tie"
    sample_indices: list[int]      # All prompt indices in cluster
    examples: list[dict]           # Top-5 discordant examples
    representative_prompts: list[str]
    cluster_id: int
```

**`examples` format** — discordant pairs where models disagree:
```python
{"prompt": str, "response_a": str, "response_b": str, "score_a": float, "score_b": float}
```

### `ComparisonReport` (analogous to `AnalysisReport`)

```python
@dataclass(frozen=True)
class ComparisonReport:
    slices: list[SliceComparison]  # Sorted by adjusted_p_value ascending
    total_prompts: int
    model_a_name: str
    model_b_name: str
    failure_rate_a: float          # Global failure rate for A
    failure_rate_b: float          # Global failure rate for B
    global_p_value: float          # Global McNemar p-value
    global_test_used: str          # "mcnemar_chi2" | "mcnemar_exact" | "none"
    global_winner: str             # "a" | "b" | "tie"
    global_advantage_rate: float   # Global b/(b+c)
    significance_level: float
    failure_threshold: float
    scoring_mode: str
    num_clusters_tested: int
    num_significant: int
    clustering_method: str
    embedding_model: str
    metadata: dict = field(default_factory=dict)
```

With methods: `summary()`, `to_dict()`, `__str__()` — identical pattern to `AnalysisReport` and `CoverageReport`.

### `ComparisonTestResult` (internal, mutable for BH correction)

```python
@dataclass
class ComparisonTestResult:
    cluster_id: int
    size: int
    b_count: int                   # discordant A-wins
    c_count: int                   # discordant B-wins
    advantage_rate: float
    p_value: float
    test_used: str
    winner: str
    adjusted_p_value: float = 1.0  # set after BH
```

---

## D. Internal Architecture

### New subpackage: `faultmap/comparison/`

Following the `coverage/` pattern:

```
faultmap/comparison/
├── __init__.py          # exports test_mcnemar, benjamini_hochberg_comparison, ComparisonTestResult
└── statistics.py        # ~130 lines
```

**Contents of `statistics.py`:**
- `ComparisonTestResult` dataclass
- `test_mcnemar(b_count, c_count, cluster_id, size)` → `ComparisonTestResult`
- `_mcnemar_chi2(b, c)` → two-sided p-value (uses `erfc(sqrt(chi2/2))`)
- `_exact_binomial_two_sided(b, c)` → exact p-value (uses `lgamma`)
- `benjamini_hochberg_comparison(results, alpha)` → sorted list with adjusted p-values
- `test_mcnemar.__test__ = False` (prevent pytest collection, same pattern as `test_cluster_failure_rate`)

### Reuse map

| Existing module | What's reused | How |
|----------------|---------------|-----|
| `scoring/` (all 3 scorers) | Score each model's responses | Called twice: once for A, once for B |
| `slicing/clustering.py` | `cluster_embeddings()`, `get_representative_prompts()` | Cluster prompts once (shared) |
| `labeling.py` | `label_clusters()` | Name significant comparison slices |
| `utils.py` | `run_sync()` | Sync/async bridge for `compare_models()` |
| `report.py` | Pattern (new formatter added) | `format_comparison_report()` |
| `embeddings.py` | `Embedder` | Embed prompts once |

### New validation function in `utils.py`

```python
def validate_comparison_inputs(
    prompts: list[str],
    responses_a: list[str],
    responses_b: list[str],
    scores_a: list[float] | None,
    scores_b: list[float] | None,
    references: list[str] | None,
) -> None:
```

Checks:
- `prompts` non-empty
- `len(prompts) == len(responses_a) == len(responses_b)`
- If `scores_a` provided, `scores_b` must also be provided (and vice versa)
- Both scores: same length as prompts, numeric, all in [0, 1]
- `references` (if given): same length as prompts

---

## E. Pipeline (orchestration in `analyzer.py`)

### `_compare_models_async()` — 14-step pipeline

```
 1. VALIDATE       validate_comparison_inputs()
 2. MODE DETECT    scores_a+b → Mode 1 | references → Mode 2 | neither → Mode 3
 3. SCORE A        scorer.score(prompts, responses_a) → ScoringResult
 4. SCORE B        scorer.score(prompts, responses_b) → ScoringResult
 5. BINARIZE       failures_a = scores_a < threshold; failures_b = scores_b < threshold
 6. GLOBAL TEST    test_mcnemar(global b, global c) → global p-value, winner
 7. EMBED          embedder.embed_queries(prompts) → (N, D)  [once, shared]
 8. CLUSTER        cluster_embeddings() → labels (N,)
 9. PER-SLICE TEST for each cluster: test_mcnemar(slice b, slice c)
10. BH CORRECT     benjamini_hochberg_comparison() → adjusted p-values
11. FILTER         keep adjusted_p < significance_level
12. NAME           label_clusters() for significant slices
13. ASSEMBLE       SliceComparison objects
14. RETURN         ComparisonReport
```

**Step 6 (global test)**: Always performed regardless of whether any clusters are significant. Users get a headline answer even if no individual slice reaches significance.

**Step 7 note**: Embeddings are of **prompts** (not responses), same as `analyze()`. Both models see the same prompts, so one embedding pass suffices.

**Mode 2 scoring detail**: Create two `ReferenceScorer` instances sharing the same embedder and references. Each scores against its own responses. The reference embeddings are computed once internally by the first scorer call (litellm/sentence-transformers caches).

**Mode 3 scoring detail**: Create two `EntropyScorer` instances sharing the same LLM client. Each independently samples and scores. This is 2x the API calls — documented clearly.

---

## F. Reporting

### `format_comparison_report()` in `report.py`

Follows the exact existing pattern: try rich, fall back to plain text.

### Plain text output

```
=======================================================
FAULTMAP MODEL COMPARISON REPORT
=======================================================
Model A:           GPT-4o
Model B:           GPT-4o-mini
Total prompts:     500
Scoring mode:      precomputed
Clustering:        hdbscan
Embedding model:   text-embedding-3-small
Significance:      alpha=0.05
-------------------------------------------------------
GLOBAL COMPARISON
-------------------------------------------------------
Failure rate (A):  12.0%
Failure rate (B):  24.0%
Winner:            GPT-4o (Model A)
Advantage rate:    0.78 (78% of disagreements favor A)
McNemar p-value:   0.0001 (mcnemar_chi2)
-------------------------------------------------------
Clusters tested:   8
Significant:       3
-------------------------------------------------------
Slice 1: "Legal Compliance" ** GPT-4o wins **
  Description:    Questions about regulatory requirements
  Size:           45 prompts
  Failure rate:   A=4.4% vs B=42.2%
  Discordant:     A wins 18, B wins 1
  Advantage rate: 0.95
  Adj. p-value:   0.0002 (mcnemar_exact)
  Examples:
    - How do I comply with GDPR requirements?
-------------------------------------------------------
Slice 2: "Billing Disputes" ** GPT-4o-mini wins **
  Description:    Handling payment and billing issues
  Size:           38 prompts
  Failure rate:   A=31.6% vs B=5.3%
  Discordant:     A wins 2, B wins 11
  Advantage rate: 0.15
  Adj. p-value:   0.0120 (mcnemar_exact)
  Examples:
    - My credit card was charged twice
=======================================================
```

### Rich table

```
# | Name               | Size | Fail A | Fail B | Winner    | Adv Rate | Adj. p
1 | Legal Compliance   |   45 |  4.4%  | 42.2%  | GPT-4o    |   0.95   | 0.0002
2 | Billing Disputes   |   38 | 31.6%  |  5.3%  | GPT-4o-mi |   0.15   | 0.0120
```

Plus per-slice detail blocks.

---

## G. Files to Create/Modify

### New files

| File | Lines (est.) | Contents |
|------|-------------|----------|
| `faultmap/comparison/__init__.py` | ~10 | Exports: `test_mcnemar`, `benjamini_hochberg_comparison`, `ComparisonTestResult` |
| `faultmap/comparison/statistics.py` | ~130 | `ComparisonTestResult`, `test_mcnemar()`, `_mcnemar_chi2()`, `_exact_binomial_two_sided()`, `benjamini_hochberg_comparison()` |
| `tests/test_comparison/__init__.py` | ~1 | Empty |
| `tests/test_comparison/test_statistics.py` | ~120 | Unit tests for McNemar + BH |
| `examples/example_model_comparison.py` | ~100 | Standalone comparison example: two models with known per-slice strengths |
| `docs/dev/PLAN-08-comparison.md` | — | This plan file, saved to follow the existing PLAN-01..07 convention |

### Modified files

| File | Changes |
|------|---------|
| `faultmap/models.py` | Add `SliceComparison` + `ComparisonReport` frozen dataclasses (~100 lines) |
| `faultmap/analyzer.py` | Add `compare_models()` + `_compare_models_async()` (~180 lines) |
| `faultmap/report.py` | Add `format_comparison_report()`, `_format_comparison_rich()`, `_format_comparison_plain()` (~130 lines) |
| `faultmap/utils.py` | Add `validate_comparison_inputs()` (~35 lines) |
| `faultmap/__init__.py` | Export `ComparisonReport`, `SliceComparison` |
| `tests/conftest.py` | Add `make_comparison_data()` fixture + `comparison_data` fixture (~50 lines) |
| `tests/test_analyzer.py` | Add `TestCompareModels` class (~160 lines) |
| `tests/test_report.py` | Add `TestFormatComparisonReportRich` + `TestFormatComparisonPlain` + helpers (~90 lines) |
| `README.md` | New "Model Comparison" section, API reference, statistical details |
| `CLAUDE.md` | Update file table, test count, public API contract, architecture |
| `docs/dev/ARCHITECTURE.md` | Add compare_models pipeline, module contracts, data models |
| `CHANGELOG.md` | v0.4.0 entry |
| `examples/example_mode1_custom_scores.py` | Add a "Next step: compare models" section at the bottom showing how to follow up single-model analysis with comparison |
| `notebooks/tutorial.ipynb` | New "Model Comparison" section |

---

## G.1. Examples Detail

### New: `examples/example_model_comparison.py` (standalone)

A complete standalone example demonstrating the comparison workflow:

- Simulates two models (e.g., GPT-4o vs GPT-4o-mini) on 90 prompts across 3 semantic groups
- **Legal questions**: GPT-4o excels (scores ~0.85), GPT-4o-mini struggles (scores ~0.25)
- **Billing questions**: GPT-4o-mini excels (scores ~0.90), GPT-4o struggles (scores ~0.30)
- **Technical questions**: Both perform similarly (scores ~0.80)
- Shows: creating analyzer, calling `compare_models()`, printing the report, iterating slices, inspecting `global_winner`, `advantage_rate`, per-slice `winner`
- Demonstrates practical interpretation: "Use GPT-4o for legal queries, GPT-4o-mini for billing, either for technical"

### Modified: `examples/example_mode1_custom_scores.py` (append)

Add a commented section at the bottom (after the existing analysis) showing the natural follow-up:

```python
# ---------------------------------------------------------------------------
# Next step: Compare against another model
# ---------------------------------------------------------------------------
# If you have scores from a second model on the same prompts:
#
# scores_b = [...]  # scores from model B on the same prompts
# responses_b = [...]  # responses from model B
#
# comparison = analyzer.compare_models(
#     prompts, responses, responses_b,
#     scores_a=scores, scores_b=scores_b,
#     model_a_name="Current Model", model_b_name="Candidate Model",
# )
# print(comparison)
```

This is lightweight — just a commented hint showing users the natural progression from single-model analysis to comparison.

---

## H. Testing Plan

### `tests/test_comparison/test_statistics.py` — statistical unit tests

1. `test_mcnemar_chi2_a_wins` — b=30, c=10: significant, A wins, uses chi2
2. `test_mcnemar_chi2_no_difference` — b=20, c=20: not significant, tie
3. `test_mcnemar_exact_a_wins` — b=8, c=1: significant, exact binomial, A wins
4. `test_mcnemar_exact_no_difference` — b=5, c=5: not significant, tie
5. `test_mcnemar_no_discordant_pairs` — b=0, c=0: p=1.0, test="none", winner="tie"
6. `test_mcnemar_all_discordant_one_direction` — b=15, c=0: highly significant
7. `test_mcnemar_b_wins` — b=2, c=12: B wins
8. `test_mcnemar_threshold_exact_vs_chi2` — b+c=24 uses exact, b+c=25 uses chi2
9. `test_bh_correction_comparison` — known p-values, verify adjusted values
10. `test_bh_comparison_empty` — empty list returns empty
11. `test_bh_comparison_monotonicity` — adjusted p-values non-decreasing

### `tests/test_analyzer.py` — `TestCompareModels` integration class

Uses `_make_clustered_embeddings()` for well-separated prompt clusters + `_make_cluster_routing_embedder()` for deterministic clustering.

1. `test_compare_mode1_a_wins_globally` — precomputed scores, A clearly better globally
2. `test_compare_mode1_per_slice_winners` — cluster 0: A wins, cluster 1: B wins, cluster 2: tie
3. `test_compare_mode1_tie` — equal scores, no significant slices
4. `test_compare_mode2_detected` — references trigger Mode 2 (patch ReferenceScorer)
5. `test_compare_mode3_detected` — neither scores nor references → Mode 3 (patch EntropyScorer)
6. `test_compare_scores_and_references_warns` — both provided, Mode 1 wins with UserWarning
7. `test_compare_scores_a_without_b_raises` — asymmetric scores → ConfigurationError
8. `test_compare_mismatched_lengths_raises` — validation errors
9. `test_compare_report_to_dict_serializable` — JSON serializable
10. `test_compare_report_str` — string formatting works
11. `test_compare_report_summary` — summary method
12. `test_compare_no_discordant_pairs` — both models identical on all prompts

### `tests/test_report.py` — comparison report formatting

1. `test_comparison_no_significant_slices` — empty slices message
2. `test_comparison_with_slices_contains_key_values` — winner, advantage rate
3. `test_comparison_global_shown` — global p-value and winner displayed
4. `test_comparison_plain_returns_nonempty` — basic sanity
5. `test_comparison_rich_returns_nonempty` — basic sanity
6. `test_comparison_import_error_fallback` — rich missing falls back to plain

### `tests/conftest.py` — `make_comparison_data()` fixture

```python
def make_comparison_data(
    n_clusters: int = 3,
    n_per_cluster: int = 30,
    dim: int = 64,
    a_better_clusters: list[int] | None = None,  # default [0]
    b_better_clusters: list[int] | None = None,  # default [1]
    # remaining clusters: tied
    a_fail_score: float = 0.2,
    a_pass_score: float = 0.8,
    b_fail_score: float = 0.2,
    b_pass_score: float = 0.8,
    seed: int = 42,
) -> dict:
```

Returns: `prompts`, `responses_a`, `responses_b`, `scores_a`, `scores_b`, `embeddings`, `labels`, `a_better_clusters`, `b_better_clusters`.

Default: 3 clusters x 30 = 90 prompts. Cluster 0: A passes, B fails. Cluster 1: A fails, B passes. Cluster 2: both pass.

---

## I. Implementation Order

### Phase 1: Data models + statistics (no integration)

1. **`faultmap/models.py`** — Add `SliceComparison` and `ComparisonReport` dataclasses
2. **`faultmap/comparison/__init__.py`** — Create with exports
3. **`faultmap/comparison/statistics.py`** — McNemar's test + exact binomial + BH correction
4. **`tests/test_comparison/__init__.py`** + **`tests/test_comparison/test_statistics.py`** — Statistical unit tests

**Verify**: `pytest tests/test_comparison/ -v` passes

### Phase 2: Validation + fixtures

5. **`faultmap/utils.py`** — Add `validate_comparison_inputs()`
6. **`tests/conftest.py`** — Add `make_comparison_data()` + `comparison_data` fixture

### Phase 3: Orchestration

7. **`faultmap/analyzer.py`** — Add `compare_models()` + `_compare_models_async()`
8. **`tests/test_analyzer.py`** — Add `TestCompareModels` class

**Verify**: `pytest tests/test_analyzer.py::TestCompareModels -v` passes

### Phase 4: Reporting

9. **`faultmap/report.py`** — Add `format_comparison_report()` with rich + plain text
10. **`tests/test_report.py`** — Add comparison report tests

**Verify**: `pytest tests/test_report.py -v` passes

### Phase 5: Exports + docs

11. **`faultmap/__init__.py`** — Export new types
12. **`docs/dev/PLAN-08-comparison.md`** — Save this plan to follow PLAN-01..07 convention
13. **README.md** — New sections
14. **CLAUDE.md** — Update tables, API contract, architecture
15. **`docs/dev/ARCHITECTURE.md`** — Pipeline + module contracts
16. **CHANGELOG.md** — v0.4.0 entry
17. **`examples/example_model_comparison.py`** — Standalone comparison example
18. **`examples/example_mode1_custom_scores.py`** — Append commented "next step: compare" section
19. **`notebooks/tutorial.ipynb`** — New comparison section

**Verify**: `pytest tests/ -v --cov=faultmap --cov-report=term-missing` — full suite passes, no regressions

---

## J. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Mode 3 comparison is 2x API cost | Document clearly in docstring and README |
| BH type duplication (20 lines) | Acceptable tradeoff vs modifying existing tested interface |
| McNemar on tiny slices (few discordant pairs) | Exact binomial fallback handles small samples; `min_slice_size` filters tiny clusters |
| Both models identical everywhere | Returns `ComparisonReport` with empty slices, global winner="tie", clearly reported |
| `__test__ = False` needed on `test_mcnemar` | Same pattern already used for `test_cluster_failure_rate` in statistics.py |

---

## K. Backward Compatibility

- **No existing API changes**. `analyze()` and `audit_coverage()` signatures untouched.
- **No existing dataclass changes**. `AnalysisReport`, `FailureSlice`, `CoverageReport`, `CoverageGap` unchanged.
- **New exports only**. `ComparisonReport` and `SliceComparison` added to `__all__`.
- **No dependency changes**. All math uses stdlib (`math.erfc`, `math.lgamma`).
- **No test modifications**. Existing 115+ tests untouched; new tests are additive.
