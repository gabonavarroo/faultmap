# Releasing faultmap

## Prerequisites

- PyPI account at https://pypi.org with 2FA enabled
- TestPyPI account at https://test.pypi.org with 2FA enabled
- OIDC trusted publisher configured on both (owner: `gabonavarroo`, repo: `faultmap`, workflow: `publish.yml`)
- GitHub environment `pypi` created with required reviewer protection
- GitHub environment `testpypi` created (no protection required)

## Release Steps

### 1. Bump version

Edit both files to the new version (e.g. `0.2.0`):

```bash
# pyproject.toml
version = "0.2.0"

# faultmap/__init__.py
__version__ = "0.2.0"
```

### 2. Update CHANGELOG.md

Add a new section at the top:

```markdown
## [0.2.0] — YYYY-MM-DD

### Added
- ...

### Fixed
- ...
```

### 3. Commit and tag

```bash
git add pyproject.toml faultmap/__init__.py CHANGELOG.md
git commit -m "Release v0.2.0"
git tag v0.2.0
git push origin main --tags
```

### 4. CI publishes automatically

The `publish.yml` workflow triggers on the tag push:

1. **test** — runs pytest with coverage gate
2. **build** — builds sdist + wheel, runs `twine check`
3. **publish-testpypi** — publishes to TestPyPI automatically
4. **publish-pypi** — waits for manual approval in the `pypi` environment

### 5. Verify TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ faultmap==0.2.0
python -c "from faultmap import SliceAnalyzer; print('OK')"
```

### 6. Approve PyPI publish

Go to the GitHub Actions run → `publish-pypi` job → click **Review deployments** → Approve.

### 7. Verify PyPI

```bash
pip install faultmap==0.2.0
python -c "from faultmap import SliceAnalyzer; print('OK')"
```
