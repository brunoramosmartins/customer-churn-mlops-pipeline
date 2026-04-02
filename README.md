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

CI uses **`tests/fixtures/telco_sample.csv`** — no full dataset needed for `pytest`.

## Data — full Telco file (manual)

The production-sized CSV is **gitignored**. After you download it (e.g. [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)), save it as `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, fill in **source URL + `sha256` + row count** in [`data/raw/README.md`](data/raw/README.md), then validate:

```bash
pip install -e .
python -m churn_ml.data.validate
# or: churn-validate
```

## EDA (Phase 3)

Lean, reproducible summaries (no auto-profiling stack): class balance, missing values, categorical cardinality, numeric correlations with churn, figures, and modeling notes. **Stratify on `Churn` in Phase 4** is stated in the generated Markdown.

- **Leakage checklist:** [docs/EDA_LEAKAGE_CHECKLIST.md](docs/EDA_LEAKAGE_CHECKLIST.md)
- **CLI** (default input = `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` if present):

```bash
python -m churn_ml.eda.run -i tests/fixtures/telco_sample.csv -o reports
python -m churn_ml.eda.run
# or: churn-eda …
```

Outputs: `reports/eda_summary.md`, `reports/eda_summary.json`, `reports/figures/`.

- **Notebook:** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) · [notebooks/README.md](notebooks/README.md)

## Repository layout (summary)

| Path | Purpose |
|------|---------|
| `configs/` | `metrics.yaml` and future hyperparameters |
| `data/raw/` | Raw CSV (gitignored; see `data/raw/README.md`) |
| `data/processed/` | Train / validation / test artifacts |
| `docs/` | Roadmap, **PROBLEM.md**, **EDA_LEAKAGE_CHECKLIST.md** |
| `models/` | Serialized pipelines (gitignored) |
| `notebooks/` | EDA (not production logic) |
| `reports/` | Figures, drift, evaluation summaries |
| `scripts/` | GitHub CLI automation |
| `src/churn_ml/` | `metrics`, `data` (validation), `eda` (reports) |
| `tests/` | Pytest + `fixtures/telco_sample.csv` for CI |

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
| 2 — Ingestion & validation | **Done** in code — Pandera schema, CLI, fixture; **you** still download full CSV + fill `sha256` in `data/raw/README.md` |
| 3 — EDA | **Done** — `churn_ml.eda`, `churn-eda` / `python -m churn_ml.eda.run`, `docs/EDA_LEAKAGE_CHECKLIST.md`, `notebooks/01_eda.ipynb` |
| 4+ | Pending |

## License

See [LICENSE](LICENSE).
