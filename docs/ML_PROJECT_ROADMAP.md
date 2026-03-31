# Customer Churn ML / MLOps — Portfolio Roadmap

End-to-end roadmap for the Telco Customer Churn binary classification project. This document is the single source of truth for phases, GitHub workflow, repository layout, releases, and automation scripts.

---

## GitHub Semantic Guide (This Repository)

This section explains how **labels**, **milestones**, **issues**, **tags**, and **releases** relate in an ML lifecycle repo, with concrete examples for this project.

### Labels

**What they are:** Metadata on issues and pull requests for filtering, automation, and reporting.

**When to create:** Once per repository (or when you add a new category). Run `scripts/create_labels.sh` after `gh auth login`.

**Examples:**

- `type:feature` — new training code, API route, or pipeline stage.
- `phase:data` — work tied to ingestion, validation, or EDA.
- `priority:high` — blocks the next phase or a demo deadline.

**Commands:**

```bash
gh label create "type:feature" --color "0E8A16" --description "New capability"
gh issue list --label "phase:modeling"
```

### Milestones

**What they are:** Time-bounded or logical buckets that group issues (e.g., “Phase 3 — EDA & data quality”).

**When to create:** At project start, aligned with roadmap phases. Run `scripts/create_milestones.sh`.

**Relation to issues:** Each issue should reference one milestone in the GitHub UI (or via `gh issue create --milestone "..."`).

**Commands:**

```bash
gh api repos/OWNER/REPO/milestones --jq '.[].title'
gh issue list --milestone "Phase 3 — EDA & data quality"
```

### Issues

**What they are:** Units of work with a full body (Context, Tasks, Definition of done, References). They map to ML stages: validation, feature engineering, experiments, evaluation, deployment, monitoring.

**When to create:** When starting a phase or when discovering scoped work. Run `scripts/create_issues.sh` to bootstrap; afterward create issues manually for ad-hoc bugs.

**Commands:**

```bash
gh issue create --title "feat: add drift monitoring job" --label "type:feature,area:monitoring"
gh issue view 12
```

### Tags (Git tags)

**What they are:** Immutable pointers to commits, used for versioning (Semantic Versioning policy below).

**When to create:**

- **Internal phase tags:** e.g. `v0.3-eda-complete` when EDA is reproducible and merged to `main`.
- **Portfolio release:** `v1.0.0` when API + docs + reproducible train path exist.

**Commands:**

```bash
git tag -a v0.5-baseline-model -m "Baseline logistic regression + metrics"
git push origin v0.5-baseline-model
```

### Releases

**What they are:** GitHub Release objects (often with changelog and assets) built from a tag.

**When to create:** Only when there is **external value**: runnable pipeline, packaged model, or API image someone can consume without reading every issue.

**When NOT to create:** Purely internal milestones (e.g., “labels created”, “draft notebook only”) — use milestones and tags only, or no tag at all.

**Commands:**

```bash
gh release create v1.0.0 --title "v1.0.0 Portfolio" --notes-file CHANGELOG.md
```

### How they fit together

1. **Milestone** closes when all its **issues** are done.
2. **PRs** close **issues** via `Closes #N` in the description.
3. **Tag** marks a commit that passed review and meets the phase exit criteria.
4. **Release** (optional per phase) publishes that tag with notes for reviewers or hiring managers.

---

## Architecture Overview (ASCII)

Training and inference share feature logic but not the same entrypoints. Storage is layered: raw (immutable inputs), processed (parquet/csv), artifacts (models, plots, MLflow).

```
                         ┌─────────────────────────────────────────┐
                         │              DEVELOPER / CI              │
                         └─────────────────────────────────────────┘
                                            │
    ┌───────────────────────────────────────┼───────────────────────────────────────┐
    │ TRAINING PATH                         │                         INFERENCE PATH   │
    v                                       v                                        v
┌─────────────┐                    ┌────────────────┐                      ┌──────────────────┐
│ data/raw/   │── ingest + hash ──▶│ validation     │                      │  Load artifact   │
│ (Telco CSV) │                    │ (schema/GE)    │                      │  (joblib/pickle) │
└──────┬──────┘                    └───────┬────────┘                      └────────┬─────────┘
       │                                    │                                        │
       │ prepare                            │ features                               │ same transforms
       v                                    v                                        v
┌─────────────┐                    ┌────────────────┐                      ┌──────────────────┐
│data/processed│◀── splits ──────│ feature eng.   │──────────────────────│ sklearn Pipeline │
│ train/val/test                  │ (code + cfg)   │                      │  (predict only)  │
└──────┬──────┘                    └───────┬────────┘                      └────────┬─────────┘
       │                                    │                                        │
       │                                    ├──▶ MLflow (params/metrics/artifacts)   │
       │                                    │                                        │
       │ train/tune                         v                                        v
       v                           ┌────────────────┐                      ┌──────────────────┐
┌─────────────┐                   │ models/        │                      │ FastAPI /predict │
│ experiments │──────────────────▶│ mlruns/        │                      │ (optional Docker) │
│ notebooks/  │                   │ reports/       │                      └────────┬─────────┘
└─────────────┘                   └────────────────┘                               │
                                                                                    v
                                                                           ┌──────────────────┤
                                                                           │ monitoring/      │
                                                                           │ Evidently reports│
                                                                           └──────────────────┘
```

**Storage layers**

| Layer | Path (suggested) | Purpose |
|-------|-------------------|---------|
| Raw | `data/raw/` | Immutable downloaded dataset; checksums |
| Processed | `data/processed/` | Split, encoded, scaled tables |
| Artifacts | `models/`, `reports/`, MLflow `mlruns/` | Serialized model, curves, drift HTML |

---

## Repository Structure (Target Tree)

```
customer-churn-mlops-pipeline/
├── .github/
│   ├── ISSUE_TEMPLATE/          # Task + bug templates
│   └── pull_request_template.md
├── configs/                     # YAML/JSON: seeds, paths, model hyperparams
│   └── .gitkeep
├── data/
│   ├── raw/                     # Telco CSV (gitignored; README documents source)
│   │   └── .gitkeep
│   └── processed/               # Train/val/test parquet
│       └── .gitkeep
├── docs/
│   └── ML_PROJECT_ROADMAP.md    # This file
├── models/                      # Serialized pipelines (gitignored or DVC)
│   └── .gitkeep
├── notebooks/                   # EDA and ad-hoc analysis (not production)
│   └── .gitkeep
├── reports/                     # Static evaluation outputs, drift reports
│   └── .gitkeep
├── scripts/                     # gh automation + utility CLIs
│   ├── create_labels.sh
│   ├── create_milestones.sh
│   └── create_issues.sh
├── src/
│   ├── churn_ml/                # Importable package
│   │   ├── __init__.py
│   │   ├── data/                # load, validate, split
│   │   ├── features/            # transforms, feature builders
│   │   ├── models/              # train, predict wrappers
│   │   ├── evaluation/        # metrics, plots, threshold search
│   │   └── serving/             # FastAPI app (optional submodule pattern)
│   └── pipelines/               # train_batch.py, evaluate.py, export_artifacts.py
├── tests/
│   └── .gitkeep
├── .env.example                 # No secrets; MLFLOW_TRACKING_URI, etc.
├── .gitignore
├── pyproject.toml or requirements.txt   # To be added in Phase 0
├── README.md
└── LICENSE
```

**Principles:** Config and code separate; training scripts thin orchestrators; feature transforms live in one place reused at inference; notebooks do not hold production logic.

---

## Phased Roadmap

### Phase 0 — Repository foundation and reproducibility

**Objective:** Establish a version-controlled baseline so every later phase (data, features, modeling) runs in a documented environment with fixed seeds and clear dependencies. This mirrors the “platform” slice of the ML lifecycle before any model exists.

**Tasks**

- [ ] Add Python packaging (`pyproject.toml` or `requirements.txt` + optional `requirements-dev.txt`).
- [ ] Pin versions for `pandas`, `scikit-learn`, `mlflow`, `fastapi`, `pydantic`, `evidently` (when introduced).
- [ ] Add `.gitignore` for `data/raw/*.csv`, `mlruns/`, `__pycache__/`, `.env`, virtualenvs.
- [ ] Add `.env.example` with non-secret keys only.
- [ ] Configure `pre-commit` (optional): `ruff` or `black`, `isort`.
- [ ] Create empty package `src/churn_ml/` and `tests/` with one smoke test.
- [ ] Document `PYTHONPATH` or editable install for local runs.

**Deliverables checklist**

- [ ] `pip install -e .` or equivalent works from clean venv.
- [ ] README section “Environment setup” is accurate.
- [ ] CI placeholder (optional) runs tests on push.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `chore/phase-0-bootstrap` |
| Merge strategy | Squash merge to `main` |
| PR title convention | `chore: phase 0 bootstrap reproducibility` |
| Milestone | Phase 0 — Bootstrap |
| Tag | None (internal scaffolding only) |
| Release | **No** — no consumer-facing artifact yet |

---

### Phase 1 — Problem definition and metric contract

**Objective:** Translate the business goal (retention / churn reduction) into a supervised binary classification task with explicit success metrics and error-cost assumptions, so modeling and threshold choices stay aligned with stakeholders.

**Tasks**

- [ ] Write problem statement: positive class = churn, horizon, population.
- [ ] Define primary metrics: ROC-AUC, PR-AUC, F1, recall@churn.
- [ ] Document cost asymmetry: false negative (missed churn) vs false positive (unnecessary retention spend).
- [ ] Add `configs/metrics.yaml` or markdown in `docs/` with decision rules (e.g., minimum recall).
- [ ] Align with Telco column semantics (`Churn` as target).

**Deliverables checklist**

- [ ] Single source of truth for metrics in repo (`docs/PROBLEM.md` or config).
- [ ] Team/reviewer can answer “what does success mean?” without opening notebooks.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `docs/phase-1-problem-metrics` |
| Merge strategy | Squash merge |
| PR title | `docs: define problem statement and metric contract` |
| Milestone | Phase 1 — Problem & metrics |
| Tag | `v0.1-problem-metrics` (optional lightweight) |
| Release | **No** — documentation-only |

---

### Phase 2 — Data acquisition and validation

**Objective:** Implement reproducible ingestion into `data/raw/` and automated checks for schema, row counts, and obvious drift from expected distributions. This is the “Data” gate before EDA and modeling trust.

**Tasks**

- [ ] Script download or documented manual fetch (Kaggle / IBM Telco sample) with version note.
- [ ] Store raw file with checksum (`sha256`) in `data/raw/README.md` or manifest.
- [ ] Implement validation: `pydantic` / `pandera` / Great Expectations (choose one for portfolio clarity).
- [ ] Fail fast on missing columns or wrong dtypes.
- [ ] CLI: `python -m churn_ml.data.validate` or `scripts/validate_raw.py`.

**Deliverables checklist**

- [ ] Validation runnable in CI on a small fixture CSV in `tests/fixtures/`.
- [ ] Raw data path documented; large files not committed.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-2-ingest-validate` |
| Merge strategy | Squash merge |
| PR title | `feat: raw ingestion and data validation` |
| Milestone | Phase 2 — Data ingestion & validation |
| Tag | `v0.2-data-validation` |
| Release | **Optional** — only if you publish a standalone “validation CLI” others would run |

---

### Phase 3 — Exploratory data analysis (EDA)

**Objective:** Quantify class imbalance, missingness, categorical cardinality, numeric skew, and candidate leakage before building pipelines. Outputs feed feature engineering and modeling choices.

**Tasks**

- [ ] Notebook or script: target distribution, baseline churn rate.
- [ ] Per-feature summaries; correlation with target; TotalCharges parsing edge cases.
- [ ] Document “no obvious leakage” checklist (e.g., future information).
- [ ] Export static report (`reports/eda_profile.html` via ydata-profiling or similar).
- [ ] Stratification plan for split (stratify on `Churn`).

**Deliverables checklist**

- [ ] EDA artifact committed or generated reproducibly from raw data.
- [ ] Written conclusions: imbalance handling, which features need encoding.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-3-eda` |
| Merge strategy | Squash merge |
| PR title | `feat: EDA report and data understanding` |
| Milestone | Phase 3 — EDA |
| Tag | `v0.3-eda-complete` |
| Release | **No** — report is internal unless you attach HTML as release asset for portfolio |

---

### Phase 4 — Data preparation and splitting

**Objective:** Build a deterministic preprocessing path: train/validation/test splits, missing value policy, and reproducible random seeds. Separation of concerns: no modeling yet, only tabular hygiene.

**Tasks**

- [ ] Implement stratified split ratios (e.g., 70/15/15) in code, not notebook-only.
- [ ] Persist split indices or hashed row IDs to avoid leakage across reruns.
- [ ] Handle `TotalCharges` blank strings; `SeniorCitizen` dtype consistency.
- [ ] Save processed parquet/csv under `data/processed/` with naming convention.
- [ ] Unit tests on small synthetic data.

**Deliverables checklist**

- [ ] Three datasets on disk with identical schema pre-transform.
- [ ] Documented seed and split logic in `configs/` or docstring.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-4-preprocess-split` |
| Merge strategy | Squash merge |
| PR title | `feat: stratified splits and preprocessing baseline` |
| Milestone | Phase 4 — Data preparation |
| Tag | `v0.4-data-prep` |
| Release | **No** |

---

### Phase 5 — Feature engineering

**Objective:** Encode categoricals, scale if needed, and add domain-motivated derived features (e.g., tenure groups, charges per tenure). Package transforms in a `sklearn` `Pipeline` for training and inference parity.

**Tasks**

- [ ] One-hot or target encoding decision with cardinality justification.
- [ ] Optional: bin `tenure`, interaction terms (keep interpretability).
- [ ] Fit encoders on train only; apply to val/test via pipeline.
- [ ] Feature importance placeholder (for later comparison).
- [ ] Remove redundant columns after encoding.

**Deliverables checklist**

- [ ] `sklearn` pipeline object or clear `ColumnTransformer` in `src/churn_ml/features/`.
- [ ] Processed feature matrix shape documented.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-5-feature-engineering` |
| Merge strategy | Squash merge |
| PR title | `feat: feature engineering and sklearn preprocessing pipeline` |
| Milestone | Phase 5 — Feature engineering |
| Tag | `v0.5-features` |
| Release | **No** |

---

### Phase 6 — Baseline model

**Objective:** Establish a simple, interpretable model (logistic regression) to set ROC-AUC/F1 floor and validate the full path from raw → features → metrics.

**Tasks**

- [ ] Train logistic regression with class weights or baseline threshold 0.5.
- [ ] Log metrics to stdout and optionally MLflow stub.
- [ ] Save first serialized pipeline to `models/baseline.joblib`.
- [ ] Document assumptions (linearity, calibration).

**Deliverables checklist**

- [ ] Reproducible training command in README.
- [ ] Metrics table in `reports/baseline_metrics.json` or MLflow.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-6-baseline` |
| Merge strategy | Squash merge |
| PR title | `feat: logistic regression baseline` |
| Milestone | Phase 6 — Baseline modeling |
| Tag | `v0.6-baseline-model` |
| Release | **Optional** — if model artifact is meaningful standalone |

---

### Phase 7 — Advanced models and hyperparameter tuning

**Objective:** Improve ranking and calibration with tree ensembles (Random Forest, XGBoost or LightGBM), using stratified cross-validation and controlled tuning budgets to avoid overfitting the test set.

**Tasks**

- [ ] Implement RF and one boosting library with consistent evaluation harness.
- [ ] `RandomizedSearchCV` or `Optuna` with stratified CV on train fold only.
- [ ] Track best params in config files under `configs/models/`.
- [ ] Compare to baseline on **validation**; touch **test** only for final report.

**Deliverables checklist**

- [ ] Leaderboard (validation) documented.
- [ ] No test-set-driven tuning loops.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-7-ensemble-tuning` |
| Merge strategy | Squash merge |
| PR title | `feat: ensemble models and hyperparameter search` |
| Milestone | Phase 7 — Advanced modeling |
| Tag | `v0.7-ensembles` |
| Release | **No** |

---

### Phase 8 — Model evaluation and threshold optimization

**Objective:** Rigorously characterize model behavior with ROC, PR curves, confusion matrices, and business-aligned threshold selection (not default 0.5). Final test evaluation once.

**Tasks**

- [ ] Plot ROC and PR; save to `reports/figures/`.
- [ ] Confusion matrix at default and optimized threshold.
- [ ] Implement threshold search maximizing F-beta or constrained recall.
- [ ] Calibration curve (optional, `sklearn.calibration`).
- [ ] Single frozen “champion” definition for packaging.

**Deliverables checklist**

- [ ] `reports/evaluation_summary.md` or JSON with all metrics + chosen threshold.
- [ ] Clear statement which split each metric came from.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-8-evaluation` |
| Merge strategy | Squash merge |
| PR title | `feat: evaluation suite and threshold optimization` |
| Milestone | Phase 8 — Evaluation |
| Tag | `v0.8-evaluation` |
| Release | **Optional** — attach figure bundle for portfolio |

---

### Phase 9 — Experiment tracking (MLflow)

**Objective:** Centralize parameters, metrics, and artifacts for every training run to support comparison, reproducibility, and future registry integration.

**Tasks**

- [ ] Local MLflow tracking URI (`mlruns/` gitignored) or remote server note.
- [ ] Log params, metrics, model, and plots per run.
- [ ] Standardize run naming: `churn_{model}_{date}`.
- [ ] Document how to start UI: `mlflow ui`.

**Deliverables checklist**

- [ ] At least three comparable runs visible in MLflow.
- [ ] README “Experiment tracking” subsection.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-9-mlflow` |
| Merge strategy | Squash merge |
| PR title | `feat: MLflow experiment tracking` |
| Milestone | Phase 9 — Experiment tracking |
| Tag | `v0.9-mlflow` |
| Release | **No** |

---

### Phase 10 — Model packaging and inference pipeline

**Objective:** Serialize the champion `sklearn` pipeline (preprocessing + estimator) with `joblib` or MLflow model format, and provide a single `predict` entrypoint used by the API later.

**Tasks**

- [ ] Export `predict_proba` contract: feature order, dtypes.
- [ ] Version artifact filename or MLflow model version.
- [ ] Smoke script: load model, run on sample row from val set.
- [ ] Input validation with `pydantic` for batch inference CLI.

**Deliverables checklist**

- [ ] `python -m churn_ml.serving.predict_batch --input ...` or similar works.
- [ ] Hash or version string recorded in metadata JSON.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-10-packaging` |
| Merge strategy | Squash merge |
| PR title | `feat: packaged inference pipeline` |
| Milestone | Phase 10 — Packaging |
| Tag | `v0.10-packaging` |
| Release | **Yes (pre-1.0)** — tag `v0.10.0-rc1` if API not ready but batch inference is demoable |

---

### Phase 11 — Deployment (FastAPI + optional Docker)

**Objective:** Expose churn scoring over HTTP for synchronous inference, with health checks and schema-validated payloads—closest to production serving in a portfolio.

**Tasks**

- [ ] FastAPI app: `POST /predict`, `GET /health`.
- [ ] Load model at startup; document memory implications.
- [ ] Optional: Dockerfile multi-stage; `docker compose` for local demo.
- [ ] OpenAPI schema export for reviewers.

**Deliverables checklist**

- [ ] `uvicorn` one-liner in README.
- [ ] Example `curl` with JSON body.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-11-fastapi` |
| Merge strategy | Squash merge |
| PR title | `feat: FastAPI churn prediction service` |
| Milestone | Phase 11 — Deployment |
| Tag | `v0.11-api` |
| Release | **Yes** — `v0.11.0` with run instructions (major portfolio milestone) |

---

### Phase 12 — Monitoring (drift and performance)

**Objective:** Define how data drift and silent performance decay would be detected in a real system, using tools such as Evidently for tabular reports and scheduled batch checks.

**Tasks**

- [ ] Reference dataset: training or validation snapshot as “expected.”
- [ ] Current batch: recent production-like or holdout slice.
- [ ] Evidently (or custom) report: column drift, target drift if labels delayed.
- [ ] Define retraining triggers (e.g., PSI threshold, weekly job).
- [ ] Store HTML/JSON under `reports/monitoring/`.

**Deliverables checklist**

- [ ] One generated drift report committed as example or reproducible script.
- [ ] Short runbook: what to do when drift fires.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `feat/phase-12-monitoring` |
| Merge strategy | Squash merge |
| PR title | `feat: data drift monitoring with Evidently` |
| Milestone | Phase 12 — Monitoring |
| Tag | `v0.12-monitoring` |
| Release | **Optional** — if report template is useful as release asset |

---

### Phase 13 — Documentation and portfolio release

**Objective:** Make the repository self-explanatory for hiring managers: architecture, how to train, how to serve, results summary, and limitations.

**Tasks**

- [ ] README: problem, metrics, structure, commands, results table.
- [ ] Architecture diagram (Mermaid in README or PNG export).
- [ ] `CHANGELOG.md` for `v1.0.0`.
- [ ] License and citation for Telco dataset.

**Deliverables checklist**

- [ ] New contributor can run train + API from README alone.
- [ ] Clear “known limitations” and “future work” pointers to Phase 14.

**GitHub**

| Item | Value |
|------|--------|
| Branch | `docs/phase-13-readme-release` |
| Merge strategy | Squash merge |
| PR title | `docs: portfolio README and v1.0.0 release notes` |
| Milestone | Phase 13 — Documentation |
| Tag | `v1.0.0` |
| Release | **Yes** — `v1.0.0` full portfolio release |

---

### Phase 14 — Future improvements (backlog)

**Objective:** Capture production-grade extensions without blocking v1.0: feature store, CI/CD, registry, A/B testing—standard MLOps evolution path.

**Tasks**

- [ ] Feast or lightweight feature registry spike.
- [ ] GitHub Actions: lint, test, train on schedule (CPU budget).
- [ ] MLflow Model Registry promotion workflow.
- [ ] Shadow or A/B deployment narrative (even if simulated).

**Deliverables checklist**

- [ ] Labeled backlog issues with `phase:future`.

**GitHub**

| Item | Value |
|------|--------|
| Branch | Per issue (`feat/...`) |
| Merge strategy | Squash merge |
| Milestone | Phase 14 — Future / backlog |
| Tag | Per deliverable (e.g. `v1.1.0-ci`) |
| Release | **When** a net-new capability ships (e.g. CI greenfield on forks) |

---

## Issue Specifications (Full Bodies)

Each issue below is created by `scripts/create_issues.sh` and follows the required structure.

---

### Issue: chore: Phase 0 — Python env, packaging, and test skeleton

## Context: why this issue exists in the ML lifecycle

Without a reproducible environment and importable package layout, later stages (validation, training, serving) become notebook-only and fail CI and collaboration expectations.

## Tasks: specific, actionable checklist

- [ ] Add `pyproject.toml` (or requirements files) with pinned core deps.
- [ ] Create `src/churn_ml/` package and `tests/test_smoke.py`.
- [ ] Add `.gitignore` and `.env.example`.
- [ ] Document editable install in README.

## Definition of done: verifiable completion criteria

- [ ] Fresh venv: install works; `pytest` passes at least one test.
- [ ] No secrets in repo; `.env` gitignored.

## References: relevant links, docs, datasets, or code

- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- PEP 621 — `pyproject.toml` metadata

---

### Issue: docs: problem statement, metrics, and error-cost framing

## Context: why this issue exists in the ML lifecycle

Churn projects often optimize the wrong metric; anchoring ROC-AUC, recall on churn, and business cost early prevents threshold and model selection drift.

## Tasks: specific, actionable checklist

- [ ] Add `docs/PROBLEM.md` with positive class definition and metrics.
- [ ] Document false negative vs false positive business interpretation.
- [ ] Link metrics to evaluation code constants.

## Definition of done: verifiable completion criteria

- [ ] Reviewer can read one file and understand success criteria.
- [ ] PR references Telco target column semantics.

## References: relevant links, docs, datasets, or code

- IBM / Kaggle Telco Customer Churn dataset documentation
- scikit-learn metrics user guide

---

### Issue: feat: raw dataset ingestion and checksum manifest

## Context: why this issue exists in the ML lifecycle

Raw data is the system of record; ingestion must be traceable and repeatable for audits and reproducibility.

## Tasks: specific, actionable checklist

- [ ] Implement download script or documented curl/Kaggle flow.
- [ ] Write `data/raw/README.md` with source URL and `sha256`.
- [ ] Ensure large files are gitignored.

## Definition of done: verifiable completion criteria

- [ ] Same script reproduces identical checksum on same file version.
- [ ] README lists minimum file name and expected rows.

## References: relevant links, docs, datasets, or code

- https://www.kaggle.com/datasets/blastchar/telco-customer-churn (or current canonical source)

---

### Issue: feat: data validation layer for Telco schema

## Context: why this issue exists in the ML lifecycle

Validation gates prevent silent schema changes from breaking training and serving and mimic production data contracts.

## Tasks: specific, actionable checklist

- [ ] Define expected columns and dtypes in code.
- [ ] CLI exits non-zero on validation failure.
- [ ] Add fixture CSV for CI under `tests/fixtures/`.

## Definition of done: verifiable completion criteria

- [ ] CI runs validation on fixture without network.
- [ ] Breaking change to schema is caught by tests.

## References: relevant links, docs, datasets, or code

- Pandera / Great Expectations / pydantic documentation

---

### Issue: feat: EDA notebook or script and profiling export

## Context: why this issue exists in the ML lifecycle

EDA reduces leakage risk and informs encoding, imbalance handling, and feature design before irreversible pipeline choices.

## Tasks: specific, actionable checklist

- [ ] Summarize class balance and missing values.
- [ ] Export HTML or PDF report to `reports/`.
- [ ] List recommended preprocessing steps.

## Definition of done: verifiable completion criteria

- [ ] Report regenerates from raw data via documented command.
- [ ] Written conclusions section exists (not only plots).

## References: relevant links, docs, datasets, or code

- ydata-profiling / Sweetviz (optional)

---

### Issue: feat: stratified train/validation/test split pipeline

## Context: why this issue exists in the ML lifecycle

Proper splitting is the foundation of honest metrics; stratification preserves churn rate across splits.

## Tasks: specific, actionable checklist

- [ ] Implement ratios via config; fixed `random_state`.
- [ ] Write parquet outputs to `data/processed/`.
- [ ] Unit test: no row overlap between splits.

## Definition of done: verifiable completion criteria

- [ ] Three disjoint files with expected row counts logged.
- [ ] Churn rate printed per split.

## References: relevant links, docs, datasets, or code

- scikit-learn `train_test_split` stratify parameter

---

### Issue: feat: sklearn preprocessing and feature engineering pipeline

## Context: why this issue exists in the ML lifecycle

Feature parity between training and inference is mandatory; `sklearn` `Pipeline` is the standard portable abstraction.

## Tasks: specific, actionable checklist

- [ ] `ColumnTransformer` for numeric vs categorical.
- [ ] Fit on train only; transform val/test in training scripts.
- [ ] Optional derived features with tests.

## Definition of done: verifiable completion criteria

- [ ] Single pipeline object serializable with `joblib`.
- [ ] Documented feature count after transform.

## References: relevant links, docs, datasets, or code

- https://scikit-learn.org/stable/modules/compose.html

---

### Issue: feat: baseline logistic regression training job

## Context: why this issue exists in the ML lifecycle

A baseline quantifies lift from complex models and validates the data pipeline end-to-end.

## Tasks: specific, actionable checklist

- [ ] Train with `class_weight` if imbalanced.
- [ ] Save metrics JSON and model artifact.
- [ ] Log run metadata (git sha, seed).

## Definition of done: verifiable completion criteria

- [ ] One command trains baseline and writes `models/baseline.joblib`.
- [ ] Metrics include ROC-AUC and F1 on validation.

## References: relevant links, docs, datasets, or code

- scikit-learn `LogisticRegression`

---

### Issue: feat: Random Forest and boosting with CV tuning

## Context: why this issue exists in the ML lifecycle

Tree ensembles typically improve ranking on tabular churn data; CV tuning estimates generalization without touching the test set.

## Tasks: specific, actionable checklist

- [ ] Implement RF + XGBoost or LightGBM with same evaluation harness.
- [ ] Stratified K-fold on training portion only.
- [ ] Persist best hyperparameters to `configs/`.

## Definition of done: verifiable completion criteria

- [ ] Leaderboard compares models on validation.
- [ ] Test set used once in Phase 8 final report only.

## References: relevant links, docs, datasets, or code

- XGBoost / LightGBM sklearn API docs

---

### Issue: feat: ROC/PR curves, confusion matrix, threshold search

## Context: why this issue exists in the ML lifecycle

Classification metrics depend on threshold; business-aligned thresholding is as important as model choice.

## Tasks: specific, actionable checklist

- [ ] Generate plots saved under `reports/figures/`.
- [ ] Implement threshold grid or F-beta optimization.
- [ ] Document chosen threshold and rationale.

## Definition of done: verifiable completion criteria

- [ ] All plots reproducible from evaluation script.
- [ ] Summary file lists val vs test metrics clearly.

## References: relevant links, docs, datasets, or code

- scikit-learn `precision_recall_curve`, `roc_curve`

---

### Issue: feat: MLflow tracking for training runs

## Context: why this issue exists in the ML lifecycle

Experiment tracking is the operational memory of modeling work and enables registry and governance later.

## Tasks: specific, actionable checklist

- [ ] Wrap training in MLflow start_run; log params, metrics, artifact.
- [ ] Standardize experiment name `churn-telco`.
- [ ] Document local UI usage.

## Definition of done: verifiable completion criteria

- [ ] Multiple runs comparable in MLflow UI.
- [ ] `mlruns/` gitignored.

## References: relevant links, docs, datasets, or code

- https://www.mlflow.org/docs/latest/tracking.html

---

### Issue: feat: packaged inference pipeline and batch predict CLI

## Context: why this issue exists in the ML lifecycle

Production systems consume a versioned artifact and a narrow API; packaging separates experimentation from deployment.

## Tasks: specific, actionable checklist

- [ ] Load champion model from disk or MLflow.
- [ ] Validate input schema with `pydantic`.
- [ ] Batch inference CLI with CSV/Parquet output.

## Definition of done: verifiable completion criteria

- [ ] CLI runs on sample file and produces predictions + probabilities.
- [ ] README documents command.

## References: relevant links, docs, datasets, or code

- joblib persistence best practices

---

### Issue: feat: FastAPI service for real-time churn scoring

## Context: why this issue exists in the ML lifecycle

HTTP APIs are the default synchronous serving pattern for many ML products; this demonstrates deployment readiness.

## Tasks: specific, actionable checklist

- [ ] Implement `/predict` and `/health`.
- [ ] Return probabilities and optional SHAP later (optional).
- [ ] Add example request/response in README.

## Definition of done: verifiable completion criteria

- [ ] `uvicorn` starts without error; `curl` example works locally.
- [ ] Errors return structured JSON on bad input.

## References: relevant links, docs, datasets, or code

- https://fastapi.tiangolo.com/

---

### Issue: feat: Docker image for API (optional compose)

## Context: why this issue exists in the ML lifecycle

Container images encode environment assumptions and simplify handoff to platform teams.

## Tasks: specific, actionable checklist

- [ ] Dockerfile: install deps, copy package, expose port.
- [ ] Optional `docker-compose.yml` with volume for model.
- [ ] Document build and run.

## Definition of done: verifiable completion criteria

- [ ] `docker build` succeeds on clean machine.
- [ ] Container serves `/health` OK.

## References: relevant links, docs, datasets, or code

- https://docs.docker.com/engine/reference/builder/

---

### Issue: feat: Evidently drift report and monitoring runbook

## Context: why this issue exists in the ML lifecycle

Live systems face distribution shift; monitoring is required for safe retraining and risk management.

## Tasks: specific, actionable checklist

- [ ] Build reference vs current comparison for key features.
- [ ] Save HTML report to `reports/monitoring/`.
- [ ] Add `docs/MONITORING.md` with retraining triggers.

## Definition of done: verifiable completion criteria

- [ ] Script regenerates report; one example checked in or ignored per policy.
- [ ] Runbook states who acts on alerts in a real org (template).

## References: relevant links, docs, datasets, or code

- https://docs.evidentlyai.com/

---

### Issue: docs: README architecture, results table, and v1.0 polish

## Context: why this issue exists in the ML lifecycle

Documentation is the primary deliverable for portfolio reviewers and future maintainers.

## Tasks: specific, actionable checklist

- [ ] README: setup, train, evaluate, serve, monitor commands.
- [ ] Embed architecture (Mermaid) or link to diagram.
- [ ] Final results summary with caveats.

## Definition of done: verifiable completion criteria

- [ ] Independent reader reproduces core flow in under 30 minutes (given data access).
- [ ] `CHANGELOG.md` updated for `v1.0.0`.

## References: relevant links, docs, datasets, or code

- This roadmap file `docs/ML_PROJECT_ROADMAP.md`

---

## Release Strategy (Summary)

| Version pattern | Example | Meaning |
|-----------------|---------|---------|
| Internal phase | `v0.5-features` | Git tag on `main` after phase merge; optional |
| RC | `v0.10.0-rc1` | Pre-release packaging without full API |
| Portfolio | `v1.0.0` | Train + evaluate + serve + docs complete |

**Rule:** Create a **GitHub Release** when a stranger can obtain value (run API, run batch predict, or follow README to full pipeline). Internal phases use milestones + optional lightweight tags only.

---

## Optional: Agent / AI Assistant Section

Use an AI coding agent for boilerplate (FastAPI stubs, tests) but not for metrics you have not verified on real data.

**System prompt structure (suggested)**

- Role: MLOps-aware Python engineer for this repo only.
- Workflow: read `docs/ML_PROJECT_ROADMAP.md` → implement issue scope → run tests → update README if behavior changes.
- Analytical principles: no test-set tuning; match train/inference transforms; log seeds.
- Stopping criteria: issue Definition of done satisfied; CI green.
- Communication: PR description with commands run and metrics summary.

**Do NOT ask the agent to**

- Re-derive the full roadmap (already in this doc).
- Invent dataset statistics without running code.
- Add redundant ML frameworks without issue approval.

---

## Optional: Agent Governance

- **Observability:** training logs include git SHA, data checksum, MLflow run id.
- **Key metrics:** validation ROC-AUC, calibration error, inference latency p95.
- **Tool usage:** expected — run linters/tests locally; problematic — bulk refactors across unrelated modules.
- **Logs:** treat MLflow missing run as incident for training jobs; API logs should not print PII.

---

## Optional: MCP & External Integrations Appendix

| Suggested MCP / tool | Purpose |
|---------------------|---------|
| GitHub MCP | Create issues/PRs from IDE |
| Browser MCP | Manual UI check of MLflow or FastAPI Swagger |
| DVC / cloud storage | Optional large artifact versioning outside git |

Installation: follow vendor docs; keep secrets out of repo.

---

## CLI Automation Scripts

The repository includes Bash scripts using **GitHub CLI** (`gh`). Run from repo root on macOS, Linux, or **Git Bash / WSL** on Windows.

**Prerequisites**

```bash
gh auth login
gh repo set-default OWNER/REPO   # if needed
```

**Order of execution**

```bash
chmod +x scripts/*.sh
./scripts/create_labels.sh
./scripts/create_milestones.sh
./scripts/create_issues.sh
```

Script source files live at:

- `scripts/create_labels.sh`
- `scripts/create_milestones.sh`
- `scripts/create_issues.sh`

---

## GitHub Workflow Standards Appendix

### Branch naming convention

| Pattern | Example |
|---------|---------|
| feature | `feat/phase-7-xgb-tuning` |
| fix | `fix/validation-tenure-dtype` |
| docs | `docs/monitoring-runbook` |
| chore | `chore/ci-test-workflow` |

### Conventional Commits (examples)

| Type | Example message |
|------|-----------------|
| feat | `feat: add stratified split CLI` |
| fix | `fix: parse TotalCharges empty string` |
| docs | `docs: add MLflow section to README` |
| chore | `chore: pin scikit-learn 1.4.x` |
| test | `test: assert no split overlap` |
| ci | `ci: add pytest github action` |

### PR template

See `.github/pull_request_template.md`.

### Issue templates

See `.github/ISSUE_TEMPLATE/task.md` and `.github/ISSUE_TEMPLATE/bug.md`.

### Labels table

| Label | Color (hex) | Description |
|-------|-------------|-------------|
| `type:feature` | `0E8A16` | New capability |
| `type:bug` | `D73A4A` | Incorrect behavior |
| `type:task` | `1D76DB` | General work item |
| `type:docs` | `0075CA` | Documentation |
| `type:chore` | `FEF2C0` | Maintenance |
| `phase:bootstrap` | `5319E7` | Phase 0 |
| `phase:data` | `0052CC` | Ingestion, validation, EDA |
| `phase:features` | `0B9586` | Feature engineering |
| `phase:modeling` | `B60205` | Training and tuning |
| `phase:evaluation` | `F9D0C4` | Metrics and thresholds |
| `phase:deployment` | `C5DEF5` | API, Docker |
| `phase:monitoring` | `FBCA04` | Drift, performance |
| `phase:docs` | `D4C5F9` | README, roadmap |
| `priority:high` | `B60205` | Blocks progress |
| `priority:medium` | `FBCA04` | Normal |
| `priority:low` | `0E8A16` | Nice to have |
| `area:mlflow` | `1D76DB` | Experiment tracking |
| `area:api` | `5319E7` | FastAPI serving |
| `area:ci` | `000000` | Automation |

---

*End of roadmap document.*
