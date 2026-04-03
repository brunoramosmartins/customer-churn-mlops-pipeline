# Contributing

Thanks for improving this portfolio project. Short guidelines so PRs stay easy to review.

## Environment

```bash
python -m venv .venv
pip install -e ".[dev,portfolio]"
pytest -q
```

CI runs on **Python 3.10 and 3.11** (see `.github/workflows/ci.yml`).

## Branches and commits

- Prefer branches like `feat/phase-N-short-name`, `fix/...`, or `docs/...` as in [docs/ML_PROJECT_ROADMAP.md](docs/ML_PROJECT_ROADMAP.md).
- Use [Conventional Commits](https://www.conventionalcommits.org/) when possible (`feat:`, `fix:`, `docs:`, `test:`, `ci:`).

## Pull requests

- Use the [PR template](.github/pull_request_template.md): summary, how to test, checklist (no test-set tuning, no secrets/large data).
- Link issues with `Closes #N` when applicable.

## Machine learning hygiene

- **Do not** tune thresholds or select models using the **test** split; test is one-shot after validation (see Phase 8 docs).
- Keep **train / validation / test** disjoint; fit preprocessors on **train** only.
- Regenerate `reports/evaluation_summary.*` and `configs/champion.yaml` with `churn-evaluate` after meaningful training changes; prefer **repo-relative paths** in committed configs (no machine-specific absolute paths in YAML).

## Python style

- Prefer type hints on public functions and clarity over cleverness.
- Add or extend **docstrings** when behavior is non-obvious or part of the public API.
