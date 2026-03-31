#!/usr/bin/env bash
# create_milestones.sh — Milestones aligned with consolidated roadmap (Phases 0–12).
#
# Usage: ./scripts/create_milestones.sh
# Prerequisites: gh auth login; gh repo set-default if needed.
# Idempotency: skips if milestone title already exists.

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
  "Python packaging, tests, .gitignore, reproducible venv."

create_milestone "Phase 1 — Problem & metrics" \
  "Problem statement, metrics, FN/FP cost framing."

create_milestone "Phase 2 — Data ingestion & validation" \
  "Raw data, checksums, Pandera schema validation, CI fixture."

create_milestone "Phase 3 — EDA" \
  "Target balance, missingness, leakage notes—lean EDA."

create_milestone "Phase 4 — Data preparation" \
  "Stratified train/val/test, persisted processed tables."

create_milestone "Phase 5 — Feature engineering" \
  "sklearn Pipeline / ColumnTransformer; MVP + optional extras."

create_milestone "Phase 6 — Baseline & MLflow" \
  "Logistic regression baseline; MLflow from first model run."

create_milestone "Phase 7 — LightGBM & tuning" \
  "Single boosting model, stratified CV, MLflow; no test tuning."

create_milestone "Phase 8 — Evaluation" \
  "ROC/PR, confusion matrix, threshold, champion selection."

create_milestone "Phase 9 — Packaging" \
  "Versioned artifact, batch predict CLI, pydantic inputs."

create_milestone "Phase 10 — Serving & monitoring" \
  "FastAPI; Docker optional; one drift demo + short concept doc."

create_milestone "Phase 11 — Documentation & release" \
  "README, results, CHANGELOG, v1.0.0 portfolio release."

create_milestone "Phase 12 — Future / backlog" \
  "CI, Feast/registry, registry narrative—post v1.0."

echo "Done. Verify with: gh api repos/${REPO}/milestones --jq '.[].title'"
