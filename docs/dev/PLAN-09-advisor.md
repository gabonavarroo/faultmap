# Actionable Advisor (Insight Layer) Implementation Plan

## Context

faultmap currently discovers **where** an LLM fails (via clustering + hypothesis testing) and names each failure slice using an LLM. We want to upgrade the labeling step so the LLM also deduces **why** the model fails on that slice (`root_cause`) and **how to fix it** (`suggested_remediation`). This transforms faultmap from a diagnostic tool into an actionable advisor — without breaking the mathematical rigor (no auto-optimization).

The feature is scoped to **failure slices** only (from `analyze()`). Coverage gaps and model comparison slices are not affected since root-cause analysis doesn't apply to them.

---

## Files to Modify

| File | Change |
|------|--------|
| `faultmap/labeling.py` | Add `root_cause`/`suggested_remediation` to `ClusterLabel`, update LLM prompt + parser |
| `faultmap/models.py` | Add `root_cause`/`suggested_remediation` fields to `FailureSlice` |
| `faultmap/analyzer.py` | Map new label fields into `FailureSlice` construction (~line 378) |
| `faultmap/report.py` | Show insights in `_format_analysis_plain` and `_format_analysis_rich` |
| `tests/conftest.py` | Update mock LLM response to include new fields |
| `tests/test_labeling.py` | Update mock responses + add assertions for new fields |
| `tests/test_analyzer.py` | Update mock response + assert new fields on FailureSlice |
| `tests/test_report.py` | Update `_make_failure_slice()`, add assertions for insight text |
| `README.md` | Update "Reading Results" code example to show new fields |

---

## Step-by-step

### Step 1: `faultmap/labeling.py` — ClusterLabel + LLM prompt + parser

**ClusterLabel** (line 9-12): Add two fields with defaults for backward compat:
```python
@dataclass(frozen=True)
class ClusterLabel:
    name: str
    description: str
    root_cause: str = ""
    suggested_remediation: str = ""
```

**`label_cluster()`** (line 19-65): Make the prompt context-aware.
- When `context == "failure slice"`: use an enhanced system prompt that asks for all 4 fields (Name, Description, Root Cause, Suggested Fix). Increase `max_tokens` from 150 → 400.
- For all other contexts ("coverage gap", "model comparison slice"): keep the existing 2-field prompt and `max_tokens=150`.

Enhanced system prompt for failure slices:
```
You are an expert AI debugger analyzing a cluster of similar text inputs where an LLM is systematically failing. Given the example inputs below, provide:
1. A concise name (2-5 words) that captures the common theme
2. A one-sentence description of what these texts have in common
3. A root cause analysis: why is the LLM likely failing on this type of input?
4. A suggested fix: a concrete 1-2 sentence addition or modification to the system prompt that would address this failure pattern

Respond in exactly this format:
Name: <your name>
Description: <your description>
Root Cause: <your root cause analysis>
Suggested Fix: <your suggested remediation>
```

**`_parse_label_response()`** (line 68-89): Add parsing for `root cause:` and `suggested fix:` prefixes:
```python
root_cause = ""
suggested_remediation = ""
for line in lines:
    stripped = line.strip()
    if stripped.lower().startswith("root cause:"):
        root_cause = stripped[len("root cause:"):].strip()
    elif stripped.lower().startswith("suggested fix:"):
        suggested_remediation = stripped[len("suggested fix:"):].strip()
```
Return `ClusterLabel(name=name, description=description, root_cause=root_cause, suggested_remediation=suggested_remediation)`.

### Step 2: `faultmap/models.py` — FailureSlice

Add two new fields at the end of `FailureSlice` (after `cluster_id`, line 67) with defaults:
```python
root_cause: str = ""               # LLM-generated root cause analysis
suggested_remediation: str = ""    # LLM-generated suggested system prompt fix
```

Update the docstring to document them. Since these have defaults, they won't break any existing construction that uses keyword arguments (which all callers do).

### Step 3: `faultmap/analyzer.py` — Wire fields through

At ~line 378, where `FailureSlice(...)` is constructed, add:
```python
root_cause=label.root_cause,
suggested_remediation=label.suggested_remediation,
```

No changes needed for `audit_coverage` or `compare_models` — they use `CoverageGap` and `SliceComparison` which don't get these fields.

### Step 4: `faultmap/report.py` — Display insights

**`_format_analysis_plain`** (line 42-57): After the `Adj. p-value` line and before `Examples:`, inject:
```python
if s.root_cause:
    lines.append(f"  Root Cause:     {s.root_cause}")
if s.suggested_remediation:
    lines.append(f"  Suggested Fix:  {s.suggested_remediation}")
```

**`_format_analysis_rich`** (line 104-110): After the description line and before `Examples:`, inject:
```python
if s.root_cause:
    console.print(f"  Root Cause: {s.root_cause}", style="yellow")
if s.suggested_remediation:
    console.print(f"  Suggested Fix: {s.suggested_remediation}", style="green")
```

### Step 5: Tests

**`tests/conftest.py`** (~line 50): Update mock response:
```python
"Name: Test Cluster\nDescription: A test cluster of similar prompts.\n"
"Root Cause: The model lacks domain knowledge.\n"
"Suggested Fix: Add domain context to the system prompt."
```

**`tests/test_analyzer.py`** (~line 23): Update mock response:
```python
"Name: Test Slice\nDescription: Test description\n"
"Root Cause: Test root cause\nSuggested Fix: Test fix"
```
Add assertions on `worst.root_cause` and `worst.suggested_remediation` in `test_significant_slice_found_when_one_cluster_fails` (~line 409).

**`tests/test_labeling.py`**: Update all mock LLM responses to 4-field format. Add test for parsing new fields. Add test that old 2-field responses still parse (root_cause/suggested_remediation default to "").

**`tests/test_report.py`**: Update `_make_failure_slice()` to include `root_cause="Model lacks legal domain training data"` and `suggested_remediation="Add legal compliance context to system prompt"`. Add assertions that these strings appear in both plain and rich output. Also update `test_prompt_truncated_at_120_chars` which constructs a bare `FailureSlice`.

### Step 6: README.md

Update the "Reading Results > Analysis Report" section (~line 173) to add:
```python
    print(s.root_cause)          # "Model lacks context for regulatory terminology..."
    print(s.suggested_remediation)  # "Add: 'You are a compliance expert...'"
```

---

## Verification

1. `source venv/bin/activate && pytest tests/ -v --cov=faultmap --cov-report=term-missing` — all 190+ tests pass, coverage ≥ 90%
2. `ruff check .` — no linting errors
3. Spot-check `to_dict()` works automatically (uses `dataclasses.asdict`, will pick up new fields)
4. Verify backward compat: old 2-field LLM responses still parse correctly (root_cause/suggested_remediation = "")
