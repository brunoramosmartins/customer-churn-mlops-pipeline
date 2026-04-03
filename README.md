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

- **Notebooks:** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) (EDA) · [notebooks/02_pipeline_walkthrough.ipynb](notebooks/02_pipeline_walkthrough.ipynb) (sequential walkthrough: same `churn_ml` steps as the CLIs + design notes) · [notebooks/README.md](notebooks/README.md)

## Train / val / test split (Phase 4)

Stratified split on the churn column (`configs/metrics.yaml`), fixed seed, ratios in **`configs/split.yaml`**. Writes **`train.parquet`**, **`validation.parquet`**, **`test.parquet`**, and **`split_manifest.json`** under `data/processed/` (or `-o`). `TotalCharges` and `SeniorCitizen` are normalized before split.

```bash
# Default: reads data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv, writes data/processed/
python -m churn_ml.data.split
# Explicit paths:
python -m churn_ml.data.split -i path/to/raw.csv -o data/processed
# or: churn-split …
```

Omit `--skip-validation` when using real raw data (Pandera check). The tiny CI fixture has only four rows and is **not** enough for stratified 70/15/15; use the full Kaggle CSV (or any sample with enough rows per class in each split).

## Feature engineering (Phase 5)

Single sklearn **`Pipeline`** with a **`ColumnTransformer`**: numeric columns → median imputation + `StandardScaler`; categoricals → most-frequent imputation + **one-hot** (`handle_unknown='ignore'` for val/test). **Fit on `train.parquet` only**; artifacts are written under **`models/`** (gitignored).

- **Config:** [`configs/features.yaml`](configs/features.yaml) — ID column, numeric vs categorical lists, encoding note.
- **Outputs:** `models/feature_pipeline.joblib`, `models/feature_manifest.json` (`n_features_out`, feature names, per-column train cardinalities).

```bash
# Requires data/processed/train.parquet (Phase 4)
python -m churn_ml.features.run
python -m churn_ml.features.run -t data/processed/train.parquet -o models -c configs/features.yaml
# or: churn-features …
```

Phases 6–7 append a classifier to the same preprocess `Pipeline` (or you can load `feature_pipeline.joblib` and chain a model separately).

## Baseline model and MLflow (Phase 6)

End-to-end **`Pipeline`** (preprocess from Phase 5 + **logistic regression**), **fit on train only**, metrics on **validation** (`val_roc_auc`, `val_average_precision`, `val_f1`, `val_recall_churn`). **`class_weight: balanced`** is configurable in YAML when classes are imbalanced.

- **Config:** [`configs/train_baseline.yaml`](configs/train_baseline.yaml) — LR hyperparameters, MLflow experiment / run name.
- **Artifacts:** `models/baseline.joblib` (full fitted pipeline); MLflow stores runs under **`mlruns/`** (gitignored) unless you point elsewhere.

**Tracking URI:** set `MLFLOW_TRACKING_URI` for a remote store or shared folder; if unset, the default is a **local file store** `./mlruns` (relative to the process working directory).

```bash
# Requires data/processed/train.parquet and validation.parquet (Phase 4)
# Optional: custom store — Windows CMD: set MLFLOW_TRACKING_URI=file:./mlruns
# PowerShell: $env:MLFLOW_TRACKING_URI="file:./mlruns"  ·  Unix: export MLFLOW_TRACKING_URI=file:./mlruns
python -m churn_ml.models.run_baseline
python -m churn_ml.models.run_baseline --tracking-uri file:./mlruns
# or: churn-train-baseline …

mlflow ui --backend-store-uri file:./mlruns
# Then open http://127.0.0.1:5000
```

## LightGBM tuning (Phase 7)

**LightGBM only** (no RF/XGB sprawl): same preprocess as Phases 5–6, **`RandomizedSearchCV`** with **stratified K-fold on the train split only** (`roc_auc` by default). After refit on full train, reports the **same validation metrics** as the baseline for apples-to-apples comparison in MLflow. **`test.parquet` is not used** for tuning (reserved for Phase 8).

- **Search space:** [`configs/tune_lightgbm.yaml`](configs/tune_lightgbm.yaml) — `n_splits`, `n_iter`, `param_distributions`, fixed `lightgbm` kwargs (`is_unbalance`, etc.).
- **Outputs:** `models/lightgbm_tuned.joblib` (default) and auto-generated [`configs/lightgbm_best.yaml`](configs/lightgbm_best.yaml) (best hyperparameters + validation snapshot).
- **MLflow:** parent run + **nested runs** per search trial (`cv_mean_roc_auc`, `cv_std_roc_auc`); parent logs best CV score, `val_*` metrics, and model artifacts. Use the same `experiment_name` as the baseline to compare runs in the UI.

```bash
python -m churn_ml.models.run_lightgbm
python -m churn_ml.models.run_lightgbm --tune-config configs/tune_lightgbm.yaml -o models/lightgbm_tuned.joblib
# or: churn-train-lightgbm …
```

## Repository layout (summary)

| Path | Purpose |
|------|---------|
| `configs/` | `metrics.yaml`, `split.yaml`, `features.yaml`, `train_baseline.yaml`, `tune_lightgbm.yaml`, `lightgbm_best.yaml` (generated), … |
| `data/raw/` | Raw CSV (gitignored; see `data/raw/README.md`) |
| `data/processed/` | Train / validation / test artifacts |
| `docs/` | Roadmap, **PROBLEM.md**, **EDA_LEAKAGE_CHECKLIST.md** |
| `models/` | Serialized pipelines (gitignored) |
| `notebooks/` | EDA + end-to-end walkthrough (`_walkthrough_outputs/` gitignored) |
| `reports/` | Figures, drift, evaluation summaries |
| `scripts/` | GitHub CLI automation |
| `src/churn_ml/` | `metrics`, `data`, `eda`, `features`, `models` (training) |
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
| 4 — Split | **Done** — `churn_ml.data.split`, `churn-split` / `python -m churn_ml.data.split`, `configs/split.yaml`, Parquet + manifest under `data/processed/` |
| 5 — Features | **Done** — `churn_ml.features`, `churn-features` / `python -m churn_ml.features.run`, `configs/features.yaml`, `feature_pipeline.joblib` + manifest under `models/` |
| 6 — Baseline + MLflow | **Done** — `churn_ml.models`, `churn-train-baseline` / `python -m churn_ml.models.run_baseline`, `configs/train_baseline.yaml`, `models/baseline.joblib`, `mlruns/` (local or remote via `MLFLOW_TRACKING_URI`) |
| 7 — LightGBM | **Done** — `churn-train-lightgbm` / `python -m churn_ml.models.run_lightgbm`, `configs/tune_lightgbm.yaml`, `configs/lightgbm_best.yaml`, `models/lightgbm_tuned.joblib`, MLflow nested trials |
| 8+ | Pending |

## License

See [LICENSE](LICENSE).
