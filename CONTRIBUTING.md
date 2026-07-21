# Contributing

Thanks for your interest in improving phased-array-modeling.

## Development setup

```bash
git clone https://github.com/jman4162/Phased-Array-Antenna-Model.git
cd Phased-Array-Antenna-Model
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

The suite includes doctests for every module (`tests/test_docstrings.py`),
so docstring examples must run correctly.

## Linting

CI runs these checks; run them locally before opening a PR:

```bash
isort --check-only phased_array/
flake8 phased_array/ --select=E9,F63,F7,F82
```

## Pull requests

1. Fork the repository and create a feature branch from `main`.
2. Add tests for new functionality; keep the suite green.
3. Update CHANGELOG.md under an "Unreleased" heading, and docs under
   `docs/` if the public API changes.
4. Open a PR against `main` with a description of what changed and why.

## Reporting bugs and requesting features

Use the [issue tracker](https://github.com/jman4162/Phased-Array-Antenna-Model/issues).
For bugs, include a minimal reproducible example and your Python/NumPy
versions.

## Release checklist (maintainers)

1. Bump the version in `pyproject.toml` and `phased_array/__init__.py`
   (both must match), and `version`/`date-released` in `CITATION.cff`.
2. Move CHANGELOG "Unreleased" entries under the new version heading with
   the release date.
3. Commit, then tag: `git tag vX.Y.Z && git push origin main vX.Y.Z`.
4. The `release.yml` workflow builds, publishes to PyPI via trusted
   publishing, and creates the GitHub release.
