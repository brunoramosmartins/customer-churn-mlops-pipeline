#!/usr/bin/env bash
# create_issues.sh — Create roadmap issues with full bodies (Context, Tasks, DoD, References).
#
# Usage (from repository root):
#   ./scripts/create_labels.sh && ./scripts/create_milestones.sh
#   chmod +x scripts/create_issues.sh
#   ./scripts/create_issues.sh
#
# Prerequisites: gh auth login; labels and milestones must exist (run sibling scripts first).
#
# Idempotency: skips if any issue (open or closed) has the same title.

set -euo pipefail

issue_exists() {
  local title="$1"
  gh issue list --state all --json title --jq '.[].title' | grep -Fxq "$title"
}

create_issue() {
  local title="$1"
  local milestone="$2"
  shift 2
  # remaining args are label flags: pass as "$@"
  # Read body from stdin first so heredoc is always consumed (even when skipping).
  local body
  body=$(cat)

  if issue_exists "$title"; then
    echo "Issue exists, skip: $title"
    return 0
  fi

  gh issue create \
    --title "$title" \
    --body "$body" \
    --milestone "$milestone" \
    "$@"
  echo "Created issue: $title"
}

echo "Creating issues (this may take a minute)..."

create_issue \
  "chore: Phase 0 — Python env, packaging, and test skeleton" \
  "Phase 0 — Bootstrap" \
  --label "type:chore" --label "phase:bootstrap" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "docs: problem statement, metrics, and error-cost framing" \
  "Phase 1 — Problem & metrics" \
  --label "type:docs" --label "phase:docs" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "feat: raw dataset ingestion and checksum manifest" \
  "Phase 2 — Data ingestion & validation" \
  --label "type:feature" --label "phase:data" --label "priority:high" <<'EOF'
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

- https://www.kaggle.com/datasets/blastchar/telco-customer-churn (verify current canonical URL)
EOF

create_issue \
  "feat: data validation layer for Telco schema" \
  "Phase 2 — Data ingestion & validation" \
  --label "type:feature" --label "phase:data" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "feat: EDA notebook or script and profiling export" \
  "Phase 3 — EDA" \
  --label "type:feature" --label "phase:data" --label "priority:medium" <<'EOF'
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
EOF

create_issue \
  "feat: stratified train/validation/test split pipeline" \
  "Phase 4 — Data preparation" \
  --label "type:feature" --label "phase:data" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "feat: sklearn preprocessing and feature engineering pipeline" \
  "Phase 5 — Feature engineering" \
  --label "type:feature" --label "phase:features" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "feat: baseline logistic regression training job" \
  "Phase 6 — Baseline modeling" \
  --label "type:feature" --label "phase:modeling" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "feat: Random Forest and boosting with CV tuning" \
  "Phase 7 — Advanced modeling" \
  --label "type:feature" --label "phase:modeling" --label "priority:medium" <<'EOF'
## Context: why this issue exists in the ML lifecycle

Tree ensembles typically improve ranking on tabular churn data; CV tuning estimates generalization without touching the test set.

## Tasks: specific, actionable checklist

- [ ] Implement RF + XGBoost or LightGBM with same evaluation harness.
- [ ] Stratified K-fold on training portion only.
- [ ] Persist best hyperparameters to `configs/`.

## Definition of done: verifiable completion criteria

- [ ] Leaderboard compares models on validation.
- [ ] Test set used once in final evaluation milestone only.

## References: relevant links, docs, datasets, or code

- XGBoost / LightGBM sklearn API docs
EOF

create_issue \
  "feat: ROC/PR curves, confusion matrix, threshold search" \
  "Phase 8 — Evaluation" \
  --label "type:feature" --label "phase:evaluation" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "feat: MLflow tracking for training runs" \
  "Phase 9 — Experiment tracking" \
  --label "type:feature" --label "area:mlflow" --label "phase:modeling" --label "priority:medium" <<'EOF'
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
EOF

create_issue \
  "feat: packaged inference pipeline and batch predict CLI" \
  "Phase 10 — Packaging" \
  --label "type:feature" --label "phase:deployment" --label "priority:high" <<'EOF'
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
EOF

create_issue \
  "feat: FastAPI service for real-time churn scoring" \
  "Phase 11 — Deployment" \
  --label "type:feature" --label "area:api" --label "phase:deployment" --label "priority:high" <<'EOF'
## Context: why this issue exists in the ML lifecycle

HTTP APIs are the default synchronous serving pattern for many ML products; this demonstrates deployment readiness.

## Tasks: specific, actionable checklist

- [ ] Implement `/predict` and `/health`.
- [ ] Return probabilities (and optional explanations later).
- [ ] Add example request/response in README.

## Definition of done: verifiable completion criteria

- [ ] `uvicorn` starts without error; `curl` example works locally.
- [ ] Errors return structured JSON on bad input.

## References: relevant links, docs, datasets, or code

- https://fastapi.tiangolo.com/
EOF

create_issue \
  "feat: Docker image for API (optional compose)" \
  "Phase 11 — Deployment" \
  --label "type:feature" --label "area:api" --label "phase:deployment" --label "priority:low" <<'EOF'
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
EOF

create_issue \
  "feat: Evidently drift report and monitoring runbook" \
  "Phase 12 — Monitoring" \
  --label "type:feature" --label "phase:monitoring" --label "priority:medium" <<'EOF'
## Context: why this issue exists in the ML lifecycle

Live systems face distribution shift; monitoring is required for safe retraining and risk management.

## Tasks: specific, actionable checklist

- [ ] Build reference vs current comparison for key features.
- [ ] Save HTML report to `reports/monitoring/`.
- [ ] Add `docs/MONITORING.md` with retraining triggers.

## Definition of done: verifiable completion criteria

- [ ] Script regenerates report; policy for committing example artifact documented.
- [ ] Runbook states who acts on alerts in a real org (template).

## References: relevant links, docs, datasets, or code

- https://docs.evidentlyai.com/
EOF

create_issue \
  "docs: README architecture, results table, and v1.0 polish" \
  "Phase 13 — Documentation" \
  --label "type:docs" --label "phase:docs" --label "priority:high" <<'EOF'
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

- `docs/ML_PROJECT_ROADMAP.md`
EOF

create_issue \
  "chore: backlog — CI workflow, Feast spike, model registry" \
  "Phase 14 — Future / backlog" \
  --label "type:chore" --label "phase:docs" --label "priority:low" --label "area:ci" <<'EOF'
## Context: why this issue exists in the ML lifecycle

Post-v1 improvements harden the system toward real MLOps: automation, feature store, and governed promotion — without blocking the portfolio MVP.

## Tasks: specific, actionable checklist

- [ ] GitHub Actions: lint + pytest on push.
- [ ] Optional Feast or feature registry spike issue breakdown.
- [ ] MLflow Model Registry promotion narrative (even if simulated).

## Definition of done: verifiable completion criteria

- [ ] At least one CI workflow is green on `main`.
- [ ] Child issues created per initiative OR documented deferral.

## References: relevant links, docs, datasets, or code

- https://docs.github.com/en/actions
- Feast / MLflow Model Registry documentation
EOF

echo "Done. List issues: gh issue list"
