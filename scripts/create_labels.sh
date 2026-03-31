#!/usr/bin/env bash
# create_labels.sh — Create GitHub labels for the churn MLOps project.
#
# Usage (from repository root):
#   chmod +x scripts/create_labels.sh
#   ./scripts/create_labels.sh
#
# Prerequisites:
#   - GitHub CLI: https://cli.github.com/
#   - gh auth login
#   - gh repo set-default <owner>/<repo>  (if not inside a cloned gh repo context)
#
# Idempotency: skips creation if a label with the same name already exists.

set -euo pipefail

create_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  if gh label list --json name -q '.[].name' 2>/dev/null | grep -Fxq "$name"; then
    echo "Label exists, skip: $name"
    return 0
  fi

  gh label create "$name" --color "$color" --description "$description"
  echo "Created label: $name"
}

echo "Creating labels (type)..."
create_label "type:feature" "0E8A16" "New capability or ML component"
create_label "type:bug" "D73A4A" "Incorrect or broken behavior"
create_label "type:task" "1D76DB" "General scoped work item"
create_label "type:docs" "0075CA" "Documentation only"
create_label "type:chore" "FEF2C0" "Maintenance, tooling, housekeeping"

echo "Creating labels (phase)..."
create_label "phase:bootstrap" "5319E7" "Phase 0 — repo and environment"
create_label "phase:data" "0052CC" "Ingestion, validation, EDA, splits"
create_label "phase:features" "0B9586" "Feature engineering and pipelines"
create_label "phase:modeling" "B60205" "Training, tuning, baselines"
create_label "phase:evaluation" "F9D0C4" "Metrics, curves, thresholding"
create_label "phase:deployment" "C5DEF5" "API, containers, serving"
create_label "phase:monitoring" "FBCA04" "Drift and performance monitoring"
create_label "phase:docs" "D4C5F9" "README, roadmap, runbooks"

echo "Creating labels (priority)..."
create_label "priority:high" "B60205" "Blocks next phase or deadline"
create_label "priority:medium" "FBCA04" "Normal priority"
create_label "priority:low" "0E8A16" "Nice to have"

echo "Creating labels (area)..."
create_label "area:mlflow" "1D76DB" "Experiment tracking and registry"
create_label "area:api" "5319E7" "FastAPI and HTTP serving"
create_label "area:ci" "000000" "CI/CD and automation"

echo "Done. Verify with: gh label list"
