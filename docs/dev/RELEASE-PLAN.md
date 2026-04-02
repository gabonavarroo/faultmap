# faultmap: Production Release Plan

## Context

faultmap is a fully implemented Python library (130 tests, all passing) that needs to go from a working dev branch to a published PyPI package. The code is complete across all 7 phases of the original plan. What remains is: pre-publication hardening, PyPI publication workflow, tutorial notebook, README polish, and Docker support.

**Current state:** All changes on `implementations` branch (unstaged). Main branch has only the initial commit.

---

## Phase 1: Pre-Publication Code Audit & Fixes

### 1.1 — `pyproject.toml` fixes
- Add author email: `gabriel.navarrocr@gmail.com`
- Fix `[project.optional-dependencies]` → `all` should NOT include `dev` (users doing `pip install faultmap[all]` shouldn't get pytest/ruff). Change to: `all = ["faultmap[local,rich]"]`
- Add `build` and `twine` to `dev` deps
- Add sdist exclude for dev planning files:
  ```toml
  [tool.hatch.build.targets.sdist]
  exclude = ["PLAN*.md", "PROGRESS.md", "ARCHITECTURE.md", "merry-*.md", "CLAUDE.md", ".claude/", "docs/dev/"]
  ```
- Add `Changelog` to `[project.urls]`

### 1.2 — Add `faultmap/py.typed` marker
- Empty file for PEP 561 type checking support

### 1.3 — Move dev planning files out of root
- Create `docs/dev/` directory
- Move: `PLAN.md`, `PLAN-01` through `PLAN-07`, `PROGRESS.md`, `ARCHITECTURE.md`, `merry-*.md` → `docs/dev/`
- Keep `CLAUDE.md` in root (needed by Claude Code) but exclude from sdist

### 1.4 — Add `.claude/` to `.gitignore`

**Files modified:** `pyproject.toml`, `.gitignore`
**Files created:** `faultmap/py.typed`
**Files moved:** 11 planning files → `docs/dev/`

---

## Phase 2: CI Hardening & Test Verification

### 2.1 — Run full test suite, verify 130 tests pass

```bash
source venv/bin/activate
pytest tests/ -v --cov=faultmap --cov-report=term-missing
```

### 2.2 — Enhance `.github/workflows/ci.yml`
- Add coverage threshold: `--cov-fail-under=90`
- Add build verification step: `python -m build && python -m twine check dist/*`
- Add `implementations` branch to push trigger (for now)

### 2.3 — Build and verify the package locally
```bash
python -m build
python -m twine check dist/*
pip install dist/faultmap-0.1.0-py3-none-any.whl  # test clean install
```

**Regarding .env:** Not needed. All tests use mocks. Real usage requires env vars per litellm conventions (`OPENAI_API_KEY`, etc.) — this is standard and documented in README/examples.

**Files modified:** `.github/workflows/ci.yml`

---

## Phase 3: PyPI Publication Workflow

### 3.1 — Create PyPI & TestPyPI accounts
Manual steps:
1. Register at https://pypi.org/account/register/
2. Register at https://test.pypi.org/account/register/
3. Enable 2FA on both
4. On both sites: go to "Publishing" → "Add a new pending publisher"
   - Owner: `gabonavarroo`
   - Repository: `faultmap`
   - Workflow: `publish.yml`
   - Environment: `pypi`

### 3.2 — Create GitHub environment
- Go to repo Settings → Environments → New: `pypi`
- Add deployment protection rule: "Required reviewers" (add yourself)

### 3.3 — Create `.github/workflows/publish.yml`
Triggered on tag push (`v*`). Jobs:
1. **test** — run pytest on Python 3.12
2. **build** — `python -m build`, upload artifact
3. **publish-testpypi** — publish to TestPyPI via `pypa/gh-action-pypi-publish` with OIDC trusted publishing
4. **publish-pypi** — publish to PyPI (requires `pypi` environment approval)

### 3.4 — Create `RELEASING.md`
Document the release process:
1. Bump version in `pyproject.toml` + `faultmap/__init__.py`
2. Update `CHANGELOG.md`
3. Commit, tag `v0.1.0`, push tag
4. CI publishes to TestPyPI → verify → approve PyPI publish

**Files created:** `.github/workflows/publish.yml`, `RELEASING.md`

---

## Phase 4: Tutorial Jupyter Notebook

### 4.1 — Create `notebooks/tutorial.ipynb`

**Structure (both mock + real paths):**

| Cell | Content |
|------|---------|
| 1 | Title + "Open in Colab" badge + description |
| 2 | `!pip install faultmap[rich] -q` |
| 3 | **The Problem** — why aggregate metrics hide failures |
| 4 | **How faultmap Works** — pipeline diagram |
| 5 | **Setup: Mock vs Real** — define `MockEmbedder` + mock LLM for tutorial; show real key setup (Colab secrets + env vars) as togglable alternative |
| 6 | **Mode 1: Precomputed Scores** — synthetic data (90 prompts, 3 topics, 1 failing). Run with mock. Show commented real-key version |
| 7 | **Inspecting Results** — iterate slices, access fields, export JSON |
| 8 | **Mode 2: Reference-Based** — prompts + references. Mock + real paths |
| 9 | **Mode 3: Entropy/Autonomous** — mock LLM sampling + real path |
| 10 | **Coverage Auditing** — test vs prod prompts with known gap |
| 11 | **Interpreting & Exporting** — JSON export, prioritizing by effect size |
| 12 | **Real Usage Quick Start** — complete code for real API usage |
| 13 | **Next Steps** — links to README, examples, GitHub |

**Mock approach:** Use `unittest.mock.patch` to replace `get_embedder` and `AsyncLLMClient` at module level within faultmap. The regular `SliceAnalyzer` API works identically — users see the exact same code they'd use in production. Each section has a toggle: "Using mock (default)" vs "Using real API keys".

**Colab secrets integration:**
```python
# For Colab users:
from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

**Files created:** `notebooks/tutorial.ipynb`

---

## Phase 5: README Improvements

### 5.1 — Update badges
```markdown
[![PyPI version](https://img.shields.io/pypi/v/faultmap.svg)](https://pypi.org/project/faultmap/)
[![CI](https://github.com/gabonavarroo/faultmap/actions/workflows/ci.yml/badge.svg)](https://github.com/gabonavarroo/faultmap/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabonavarroo/faultmap/blob/main/notebooks/tutorial.ipynb)
```

### 5.2 — Add "Tutorial" section after Quick Start
Link to Colab notebook with "Open in Colab" button.

### 5.3 — Add "Contributing" section
Brief: clone, install dev, run tests.

### 5.4 — Remove "Release Readiness" section
Internal-facing content, not for end users.

### 5.5 — Add link to CHANGELOG.md

### 5.6 — Create `CHANGELOG.md`
Initial release entry for v0.1.0 with all features listed.

**Files modified:** `README.md`
**Files created:** `CHANGELOG.md`

---

## Phase 6: Docker Support

### 6.1 — Create `Dockerfile`
- Base: `python:3.12-slim`
- Installs `faultmap[local,rich]` (user-facing, no dev deps)
- Copies examples
- Default CMD: print version

### 6.2 — Create `Dockerfile.dev`
- Same base, installs `faultmap[all]` (including dev)
- Copies tests + examples
- Default CMD: `pytest -q`

### 6.3 — Create `docker-compose.yml`
Two services:
- `faultmap` — runs examples (passes API keys from host env)
- `dev` — runs tests (volume-mounted for live dev)

### 6.4 — Create `.dockerignore`
Exclude: `.git`, `venv/`, `__pycache__`, `.env`, `dist/`, `*.egg-info`, `.claude/`, dev planning files

### 6.5 — Create `.devcontainer/devcontainer.json`
VS Code dev container config with Python, ruff, jupyter extensions.

**Files created:** `Dockerfile`, `Dockerfile.dev`, `docker-compose.yml`, `.dockerignore`, `.devcontainer/devcontainer.json`

---

## Phase 7: Final Assembly & Release

### 7.1 — Version sync check
Verify `pyproject.toml` version == `faultmap/__init__.py.__version__` == `0.1.0`

### 7.2 — Full local build + install verification
```bash
python -m build
pip install dist/faultmap-0.1.0-py3-none-any.whl --force-reinstall
python -c "from faultmap import SliceAnalyzer; print('OK')"
```

### 7.3 — Git workflow
1. Stage and commit all changes on `implementations`
2. Merge `implementations` → `main`
3. Tag `main` with `v0.1.0`
4. Push (triggers CI + publish workflow)

---

## Execution Order

```
Phase 1 (Code Audit)  ─────────────────────────┐
Phase 2 (CI + Tests)  ← depends on Phase 1     │
Phase 3 (Publish Workflow) ← Phase 1           │ All sequential
Phase 4 (Notebook)     ← Phase 1               │
Phase 5 (README)       ← Phase 4 (Colab link)  │
Phase 6 (Docker)       ← Phase 1               │
Phase 7 (Final)        ← ALL complete          ┘
```

Phases 3, 4, and 6 can run in parallel after Phase 1. Phase 5 needs Phase 4 done first.

---

## Verification Plan

1. **Unit tests:** `pytest tests/ -v --cov=faultmap --cov-report=term-missing` → 130 pass, >90% coverage
2. **Lint:** `ruff check .` → clean
3. **Build:** `python -m build && twine check dist/*` → passes
4. **Clean install:** `pip install dist/*.whl` in fresh venv → imports work
5. **Notebook:** Open in Colab, run all cells with mock path → no errors
6. **Docker:** `docker compose run dev` → tests pass; `docker compose run faultmap` → prints version
7. **TestPyPI:** `pip install -i https://test.pypi.org/simple/ faultmap` → installs correctly

---

## Key Files Summary

| Action | File |
|--------|------|
| Modify | `pyproject.toml`, `.github/workflows/ci.yml`, `README.md`, `.gitignore` |
| Create | `faultmap/py.typed`, `.github/workflows/publish.yml`, `notebooks/tutorial.ipynb`, `CHANGELOG.md`, `RELEASING.md`, `Dockerfile`, `Dockerfile.dev`, `docker-compose.yml`, `.dockerignore`, `.devcontainer/devcontainer.json` |
| Move | `PLAN*.md`, `PROGRESS.md`, `ARCHITECTURE.md`, `merry-*.md` → `docs/dev/` |
