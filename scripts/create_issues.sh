#!/usr/bin/env bash
# create_issues.sh — Issues matching docs/ML_PROJECT_ROADMAP.md (Phases 0–12).
#
# Run after: ./scripts/create_labels.sh && ./scripts/create_milestones.sh
# Idempotency: skips if any issue has the same title (open or closed).
# Stdin (heredoc) is always read first so skips do not leak stdin to the next call.

set -euo pipefail

issue_exists() {
  local title="$1"
  gh issue list --state all --json title --jq '.[].title' | grep -Fxq "$title"
}

create_issue() {
  local title="$1"
  local milestone="$2"
  shift 2
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

echo "Creating issues..."

create_issue \
  "chore: Phase 0 — Python env, packaging, and test skeleton" \
  "Phase 0 — Bootstrap" \
  --label "type:chore" --label "phase:bootstrap" --label "priority:high" <<'EOF'
## Context
Reproducible venv + package layout so later work is not notebook-only.

## Tasks
- [ ] `pyproject.toml` or requirements; `src/churn_ml/`, `tests/test_smoke.py`
- [ ] `.gitignore`, `.env.example`, README install

## Definition of done
- [ ] `pytest` passes in fresh venv; no secrets committed

## References
- https://packaging.python.org/en/latest/tutorials/packaging-projects/
EOF

create_issue \
  "docs: problem statement, metrics, and error-cost framing" \
  "Phase 1 — Problem & metrics" \
  --label "type:docs" --label "phase:docs" --label "priority:high" <<'EOF'
## Context
Align churn metrics and FN/FP costs before modeling.

## Tasks
- [ ] `docs/PROBLEM.md` (or config); Telco `Churn` semantics

## Definition of done
- [ ] One file explains success criteria

## References
- Dataset docs; sklearn metrics user guide
EOF

create_issue \
  "feat: raw ingestion, checksum, and Pandera validation" \
  "Phase 2 — Data ingestion & validation" \
  --label "type:feature" --label "phase:data" --label "priority:high" <<'EOF'
## Context
Traceable raw data + Pandera as the single validation standard.

## Tasks
- [ ] Download/docs + `sha256` in `data/raw/README.md`
- [ ] Pandera schema + CLI exit non-zero on failure; `tests/fixtures/` for CI

## Definition of done
- [ ] Validate fixture offline; raw CSV gitignored

## References
- https://pandera.readthedocs.io/
- Telco dataset source (e.g. Kaggle)
EOF

create_issue \
  "feat: lean EDA — target, missingness, leakage notes" \
  "Phase 3 — EDA" \
  --label "type:feature" --label "phase:data" --label "priority:medium" <<'EOF'
## Context
Fast pre-pipeline checks without mandatory auto-profiling tools.

## Tasks
- [ ] Target balance; missing values (`TotalCharges`); leakage checklist; stratify note

## Definition of done
- [ ] Written conclusions; optional plots if time

## References
- Telco field definitions
EOF

create_issue \
  "feat: stratified train/validation/test pipeline" \
  "Phase 4 — Data preparation" \
  --label "type:feature" --label "phase:data" --label "priority:high" <<'EOF'
## Context
Disjoint splits and stable churn rate across sets.

## Tasks
- [ ] Config ratios + seed; parquet under `data/processed/`; test no overlap

## Definition of done
- [ ] Three files + churn rate per split logged

## References
- sklearn `train_test_split(..., stratify=)`
EOF

create_issue \
  "feat: sklearn feature pipeline (MVP + optional extras)" \
  "Phase 5 — Feature engineering" \
  --label "type:feature" --label "phase:features" --label "priority:high" <<'EOF'
## Context
Train/inference parity via one serialized `Pipeline`.

## Tasks
- [ ] MVP: `ColumnTransformer`, encoding, fit on train only
- [ ] Optional: tenure bins / ratios if they help validation

## Definition of done
- [ ] `joblib` round-trip; feature count documented

## References
- https://scikit-learn.org/stable/modules/compose.html
EOF

create_issue \
  "feat: logistic regression baseline with MLflow logging" \
  "Phase 6 — Baseline & MLflow" \
  --label "type:feature" --label "phase:modeling" --label "area:mlflow" --label "priority:high" <<'EOF'
## Context
Baseline + experiment tracking from the first real model run.

## Tasks
- [ ] Train baseline; MLflow params/metrics/artifact; `models/baseline.joblib`
- [ ] README: `MLFLOW_TRACKING_URI`, `mlflow ui`

## Definition of done
- [ ] At least one MLflow run; `mlruns/` gitignored

## References
- https://www.mlflow.org/docs/latest/tracking.html
EOF

create_issue \
  "feat: LightGBM with stratified CV and MLflow" \
  "Phase 7 — LightGBM & tuning" \
  --label "type:feature" --label "phase:modeling" --label "area:mlflow" --label "priority:medium" <<'EOF'
## Context
One gradient-boosted model for portfolio clarity; no test-set tuning.

## Tasks
- [ ] LightGBM + search; log runs; best params in `configs/`; leaderboard vs baseline on validation

## Definition of done
- [ ] Validation leaderboard; test reserved for Phase 8

## References
- LightGBM sklearn API
EOF

create_issue \
  "feat: evaluation — ROC/PR, confusion, threshold, champion" \
  "Phase 8 — Evaluation" \
  --label "type:feature" --label "phase:evaluation" --label "priority:high" <<'EOF'
## Context
Threshold drives business metrics; freeze champion for packaging.

## Tasks
- [ ] Plots in `reports/figures/`; threshold search; summary with split labels
- [ ] Calibration not required for MVP

## Definition of done
- [ ] Reproducible script; champion + threshold documented

## References
- sklearn `roc_curve`, `precision_recall_curve`
EOF

create_issue \
  "feat: package champion and batch predict CLI" \
  "Phase 9 — Packaging" \
  --label "type:feature" --label "phase:deployment" --label "priority:high" <<'EOF'
## Context
Versioned artifact + batch path before HTTP.

## Tasks
- [ ] Versioned joblib (or MLflow model); metadata JSON; pydantic batch CLI

## Definition of done
- [ ] One command on sample file; README command block

## References
- joblib persistence
EOF

create_issue \
  "feat: FastAPI, optional Docker, one drift demo" \
  "Phase 10 — Serving & monitoring" \
  --label "type:feature" --label "area:api" --label "phase:deployment" --label "priority:high" <<'EOF'
## Context
Serving + minimal monitoring story without a full ops stack.

## Tasks
- [ ] `/predict`, `/health`, README `curl`
- [ ] Docker optional—document as add-on, do not block on it
- [ ] One drift report (e.g. Evidently) + short `docs/` note on drift and retraining

## Definition of done
- [ ] API runs locally; drift script or artifact reproducible

## References
- https://fastapi.tiangolo.com/
- https://docs.evidentlyai.com/
EOF

create_issue \
  "docs: README, results, CHANGELOG for v1.0.0" \
  "Phase 11 — Documentation & release" \
  --label "type:docs" --label "phase:docs" --label "priority:high" <<'EOF'
## Context
Primary reviewer-facing deliverable.

## Tasks
- [ ] Train/eval/serve commands; results + limits; diagram (Mermaid or link)
- [ ] `CHANGELOG.md` for `v1.0.0`

## Definition of done
- [ ] Reader reproduces core path from README (with data access)

## References
- `docs/ML_PROJECT_ROADMAP.md`
EOF

create_issue \
  "chore: backlog — CI, Feast/registry, post-v1" \
  "Phase 12 — Future / backlog" \
  --label "type:chore" --label "area:ci" --label "priority:low" <<'EOF'
## Context
Post-MVP hardening; does not block v1.0.0.

## Tasks
- [ ] GitHub Actions; Feast/registry spike; MLflow registry narrative (as follow-ups)

## Definition of done
- [ ] Issues filed or explicit deferral

## References
- GitHub Actions; Feast; MLflow registry docs
EOF

echo "Done. List issues: gh issue list"
