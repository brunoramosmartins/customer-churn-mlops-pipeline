# Customer churn MLOps pipeline

[![CI](https://github.com/brunoramosmartins/customer-churn-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoramosmartins/customer-churn-mlops-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end machine learning project for **Telco customer churn** (binary classification), structured for a portfolio-ready, production-minded workflow: data validation, feature engineering, training, evaluation, experiment tracking, packaging, API serving, and monitoring.

## Problem statement (Phase 1)

- **Human-readable:** [docs/PROBLEM.md](docs/PROBLEM.md) — business goal, ML task, metrics, cost of errors, threshold policy.
- **Machine-readable:** [configs/metrics.yaml](configs/metrics.yaml) — same contract for code.
- **Python API:** `churn_ml.metrics` (`target_column()`, `positive_class_label()`, `primary_metrics()`, …).

## Roadmap and workflow

**[docs/ML_PROJECT_ROADMAP.md](docs/ML_PROJECT_ROADMAP.md)** — phases, GitHub conventions, architecture, issues, scripts.

## Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows — on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

No dataset is required for Phase 1 or for the current tests.

## Data (manual step — Phase 2+)

The Telco CSV is **not** committed. When you start **Phase 2**, download the dataset (e.g. [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)), place the file under `data/raw/`, and record the source URL + `sha256` in `data/raw/README.md` as the roadmap describes.

## Repository layout (summary)

| Path | Purpose |
|------|---------|
| `configs/` | `metrics.yaml` and future hyperparameters |
| `data/raw/` | Raw CSV (gitignored; see `data/raw/README.md`) |
| `data/processed/` | Train / validation / test artifacts |
| `docs/` | Roadmap, **PROBLEM.md** |
| `models/` | Serialized pipelines (gitignored) |
| `notebooks/` | EDA (not production logic) |
| `reports/` | Figures, drift, evaluation summaries |
| `scripts/` | GitHub CLI automation |
| `src/churn_ml/` | Package (`metrics` for Phase 1) |
| `tests/` | Pytest |

## GitHub automation (Bash + `gh`)

On **macOS**, **Linux**, or **Git Bash / WSL** on Windows:

```bash
gh auth login
chmod +x scripts/*.sh
./scripts/create_labels.sh
./scripts/create_milestones.sh
./scripts/create_issues.sh
```

## Status

| Phase | Status |
|-------|--------|
| 0 — Bootstrap | `pyproject.toml`, package layout, CI workflow |
| 1 — Problem & metrics | **Done** — `docs/PROBLEM.md`, `configs/metrics.yaml`, `churn_ml.metrics` |
| 2+ | Pending (needs raw data download for ingestion) |

## License

See [LICENSE](LICENSE).
