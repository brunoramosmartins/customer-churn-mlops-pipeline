#!/usr/bin/env bash
# create_milestones.sh — Create GitHub milestones aligned with ML lifecycle phases.
#
# Usage (from repository root):
#   chmod +x scripts/create_milestones.sh
#   ./scripts/create_milestones.sh
#
# Prerequisites: gh auth login; repository context set (clone or gh repo set-default).
#
# Idempotency: skips if a milestone with the same title already exists (open or closed).

set -euo pipefail

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

milestone_exists() {
  local title="$1"
  gh api --paginate "repos/${REPO}/milestones" --jq '.[].title' | grep -Fxq "$title"
}

create_milestone() {
  local title="$1"
  local description="$2"

  if milestone_exists "$title"; then
    echo "Milestone exists, skip: $title"
    return 0
  fi

  gh api "repos/${REPO}/milestones" \
    -f title="$title" \
    -f description="$description" \
    -f state="open"
  echo "Created milestone: $title"
}

echo "Creating milestones for ${REPO}..."

create_milestone "Phase 0 — Bootstrap" \
  "Python packaging, .gitignore, test skeleton, reproducible environment. ML lifecycle: platform foundation before data work."

create_milestone "Phase 1 — Problem & metrics" \
  "Problem statement, success metrics, error-cost framing for churn classification."

create_milestone "Phase 2 — Data ingestion & validation" \
  "Raw Telco data layout, checksums, schema validation, CI-friendly checks."

create_milestone "Phase 3 — EDA" \
  "Class imbalance, missingness, distributions, leakage review, EDA artifacts."

create_milestone "Phase 4 — Data preparation" \
  "Stratified train/val/test, preprocessing, persisted processed tables."

create_milestone "Phase 5 — Feature engineering" \
  "Encoding, scaling, derived features, sklearn Pipeline for train/inference parity."

create_milestone "Phase 6 — Baseline modeling" \
  "Logistic regression baseline, first end-to-end metrics and saved artifact."

create_milestone "Phase 7 — Advanced modeling" \
  "Tree ensembles, hyperparameter search, validation leaderboard."

create_milestone "Phase 8 — Evaluation" \
  "ROC/PR, confusion matrix, threshold optimization, champion selection."

create_milestone "Phase 9 — Experiment tracking" \
  "MLflow runs: params, metrics, artifacts; documented UI workflow."

create_milestone "Phase 10 — Packaging" \
  "Serialized inference pipeline, batch predict CLI, version metadata."

create_milestone "Phase 11 — Deployment" \
  "FastAPI service, optional Docker; synchronous churn scoring API."

create_milestone "Phase 12 — Monitoring" \
  "Data drift reports (e.g. Evidently), retraining trigger runbook."

create_milestone "Phase 13 — Documentation" \
  "Portfolio README, architecture, results, v1.0.0 release readiness."

create_milestone "Phase 14 — Future / backlog" \
  "Feature store, CI/CD hardening, registry, A/B testing — post v1.0."

echo "Done. Verify with: gh api repos/${REPO}/milestones --jq '.[].title'"
