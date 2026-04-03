# Customer Churn ML / MLOps — Portfolio Roadmap

End-to-end roadmap for the Telco Customer Churn binary classification project. This document is the single source of truth for phases, GitHub workflow, repository layout, releases, and automation scripts.

**Scope note (portfolio MVP):** Phases are consolidated (**0–12**), validation standard is **Pandera**, **MLflow** starts at the **baseline** phase, **LightGBM** is the sole advanced model, **Docker** is optional, and monitoring is **one drift demo + a short concept note**—see phased roadmap below.

---

## GitHub Semantic Guide (Essentials)

**Labels** — Filter work (`type:*`, `phase:*`, `priority:*`). Create once with `scripts/create_labels.sh` (after `gh auth login`).

**Milestones** — Group issues by roadmap phase. Create with `scripts/create_milestones.sh`.

**Issues** — Scoped work with Context / Tasks / Definition of done / References. Bootstrap via `scripts/create_issues.sh`.

**Tags** — Mark meaningful commits (`v0.x-...` during build-out, `v1.0.0` for portfolio).

**Releases** — Only when outsiders get clear value (runnable API, batch predict, full README path). Internal steps: milestone + optional tag only.

**Flow:** milestones → issues → PRs (`Closes #N`) → optional tag → release if warranted.

Command examples: [Appendix: GitHub CLI command reference](#appendix-github-cli-command-reference).

---

## Architecture Overview (ASCII)

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
│ (Telco CSV) │                    │ (Pandera)      │                      │  (joblib)        │
└──────┬──────┘                    └───────┬────────┘                      └────────┬─────────┘
       │                                    │                                        │
       v                                    v                                        v
┌─────────────┐                    ┌────────────────┐                      ┌──────────────────┐
│data/processed│◀── splits ──────│ feature eng.   │──────────────────────│ sklearn Pipeline │
└──────┬──────┘                    └───────┬────────┘                      └────────┬─────────┘
       │                                    ├──▶ MLflow (from first model runs)       │
       v                                    v                                        v
┌─────────────┐                   ┌────────────────┐                      ┌──────────────────┐
│ notebooks/  │──────────────────▶│ models/        │                      │ FastAPI /predict │
│ (EDA only)  │                   │ mlruns/        │                      │ Docker: optional │
└─────────────┘                   │ reports/       │                      └────────┬─────────┘
                                  └────────────────┘                               │
                                                                                    v
                                                                           ┌──────────────────┐
                                                                           │ 1× drift report  │
                                                                           │ + short concept  │
                                                                           │ note in docs/    │
                                                                           └──────────────────┘
```

---

## Repository Structure (Target Tree)

Add **`churn_ml/api/`** (or `serving/`), extra **CLI modules**, and **`reports/` drift outputs** when you reach Phases 9–10—not before.

```
customer-churn-mlops-pipeline/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── configs/                     # seeds, paths, LightGBM search space
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── ML_PROJECT_ROADMAP.md
├── models/
├── notebooks/                   # EDA only
├── reports/                     # figures, evaluation summary, one drift example
├── scripts/
│   ├── create_labels.sh
│   ├── create_milestones.sh
│   └── create_issues.sh
├── src/
│   └── churn_ml/
│       ├── __init__.py
│       ├── data/                # load, validate (Pandera), split
│       ├── features/            # sklearn Pipeline / ColumnTransformer
│       ├── models/              # baseline + LightGBM; MLflow logging
│       └── evaluation/          # ROC/PR, confusion, threshold
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml or requirements.txt
├── README.md
└── LICENSE
```

**Principles:** One transform path for train and predict; configs separate from code; grow folders when the phase needs them.

---

## Phased Roadmap

Phases **0–12** (was 0–14): MLflow starts with the **first** training run; **packaging** is one phase; **serving and monitoring** are one phase; **Docker** is explicitly optional and non-blocking.

### Phase 0 — Repository foundation and reproducibility

**Objective:** Version-controlled environment and package layout before data or modeling work.

**Tasks**

- [ ] Add `pyproject.toml` or `requirements.txt` (+ optional `requirements-dev.txt`).
- [ ] Pin `pandas`, `scikit-learn`, `mlflow` (used from Phase 6 onward); add `fastapi` / `pydantic` when you reach Phase 10.
- [ ] `.gitignore`: `data/raw/*.csv`, `mlruns/`, `.env`, venvs, caches.
- [ ] `.env.example` (no secrets). Optional: `pre-commit` + `ruff` or `black`.
- [ ] `src/churn_ml/`, `tests/` with one smoke test; document editable install.

**Deliverables:** Clean venv install; `pytest` passes; README “Environment setup.”

**GitHub:** Branch `chore/phase-0-bootstrap` · Milestone **Phase 0 — Bootstrap** · Tag none · Release **No**

---

### Phase 1 — Problem definition and metric contract

**Objective:** Binary churn classification with explicit metrics and error-cost notes.

**Tasks**

- [ ] `docs/PROBLEM.md` (or `configs/metrics.yaml`): positive class, ROC-AUC, PR-AUC, F1, recall on churn; FN vs FP cost.
- [ ] Telco `Churn` column semantics documented.

**Deliverables:** One file answers “what does success mean?”

**GitHub:** `docs/phase-1-problem-metrics` · Milestone **Phase 1 — Problem & metrics** · Tag `v0.1-problem-metrics` optional · Release **No**

---

### Phase 2 — Data acquisition and validation (Pandera)

**Objective:** Reproducible raw data + **Pandera** schema validation (single standard—no GE/pydantic decision fork).

**Tasks**

- [ ] Download or documented fetch; `sha256` in `data/raw/README.md`.
- [ ] Pandera `DataFrameSchema` (or equivalent) for expected columns/dtypes; CLI exits non-zero on failure.
- [ ] CI fixture under `tests/fixtures/`.

**Deliverables:** Validation runs offline on fixture; raw CSV not in git.

**GitHub:** `feat/phase-2-ingest-validate` · Milestone **Phase 2 — Data ingestion & validation** · Tag `v0.2-data-validation` · Release **Optional**

---

### Phase 3 — EDA (lean)

**Objective:** Fast sanity check—not a second project.

**Tasks**

- [ ] **Target distribution** (class balance) and **missing values** (incl. `TotalCharges` blanks).
- [ ] Short **leakage / future-information** checklist in `docs/` or notebook markdown cell.
- [ ] Note **stratify on `Churn`** for the split phase.
- [ ] Optional later: extra plots only if time allows (no auto-profiling dependency required for MVP).

**Deliverables:** Short written conclusions (imbalance, encoding hints); notebook or script in `notebooks/`.

**GitHub:** `feat/phase-3-eda` · Milestone **Phase 3 — EDA** · Tag `v0.3-eda` · Release **No**

---

### Phase 4 — Data preparation and splitting

**Objective:** Stratified train/val/test, fixed seed, persisted tables under `data/processed/`.

**Tasks**

- [x] Ratios in config (e.g. 70/15/15); handle `TotalCharges` / dtypes; tests for no overlap.

**Deliverables:** Three disjoint Parquet files + `split_manifest.json`; churn rate per split in manifest (implemented in `churn_ml.data.split`, `configs/split.yaml`).

**GitHub:** `feat/phase-4-preprocess-split` · Milestone **Phase 4 — Data preparation** · Tag `v0.4-data-prep` · Release **No**

---

### Phase 5 — Feature engineering

**Objective:** One `sklearn` `Pipeline` / `ColumnTransformer` for train and inference.

**Tasks — MVP (must ship)**

- [x] Numeric vs categorical handling; one-hot (or justified alternative) with cardinality note.
- [x] Fit on train only; serialize pipeline.

**Tasks — Advanced (optional)**

- [ ] Binned `tenure`, simple ratios (e.g. charges per tenure), interactions—only if they help validation metrics.

**Deliverables:** Serializable pipeline; feature count documented (`churn_ml.features`, `models/feature_pipeline.joblib`, `feature_manifest.json`).

**GitHub:** `feat/phase-5-feature-engineering` · Milestone **Phase 5 — Feature engineering** · Tag `v0.5-features` · Release **No**

---

### Phase 6 — Baseline model and MLflow (from run one)

**Objective:** Logistic regression floor **plus** MLflow logging from this phase onward (params, metrics, artifact).

**Tasks**

- [x] Train baseline (`class_weight` if needed); log run to MLflow; save `models/baseline.joblib`.
- [x] Document `MLFLOW_TRACKING_URI` and `mlflow ui` in README.

**Deliverables:** Comparable runs in MLflow; `mlruns/` gitignored (`churn_ml.models.run_baseline`, `configs/train_baseline.yaml`).

**GitHub:** `feat/phase-6-baseline-mlflow` · Milestone **Phase 6 — Baseline & MLflow** · Tag `v0.6-baseline-mlflow` · Release **Optional**

---

### Phase 7 — LightGBM and tuning

**Objective:** One strong tree model **only** (LightGBM; avoids RF + XGB + LightGBM sprawl). Same evaluation harness as baseline; stratified CV on train; **no** test-driven tuning.

**Tasks**

- [x] LightGBM with `RandomizedSearchCV` or small `Optuna` study; persist best params under `configs/`.
- [x] Log each run to MLflow; compare to baseline on **validation**.

**Deliverables:** Validation leaderboard (MLflow nested trials + `val_*` on holdout val); test set touched once in Phase 8 (`churn_ml.models.run_lightgbm`, `configs/lightgbm_best.yaml`).

**GitHub:** `feat/phase-7-lightgbm` · Milestone **Phase 7 — LightGBM & tuning** · Tag `v0.7-lightgbm` · Release **No**

---

### Phase 8 — Evaluation and threshold optimization

**Objective:** ROC, PR, confusion matrix, threshold search (F-beta or recall constraint); freeze champion for packaging.

**Tasks**

- [x] Plots under `reports/figures/`; summary JSON/MD with **which split** each metric uses.
- [x] No calibration plots required for v1 portfolio path (add later if needed).

**Deliverables:** Champion + threshold documented (`churn_ml.evaluation.run`, `configs/champion.yaml`, `reports/evaluation_summary.{json,md}`, `phase8_*` figures).

**GitHub:** `feat/phase-8-evaluation` · Milestone **Phase 8 — Evaluation** · Tag `v0.8-evaluation` · Release **Optional**

---

### Phase 9 — Packaging and batch inference

**Objective:** `joblib` (or MLflow model) artifact + batch CLI with `pydantic` row validation—**without** requiring a separate “MLflow phase.”

**Tasks**

- [ ] Versioned artifact path; metadata JSON (git sha, data hash optional).
- [ ] CLI: load champion, read CSV/Parquet, write predictions (module path e.g. `churn_ml.batch_predict`—add folder when you implement).

**Deliverables:** One command runs batch predict on a sample file.

**GitHub:** `feat/phase-9-packaging` · Milestone **Phase 9 — Packaging** · Tag `v0.9-packaging` · Release **Yes (optional)** — `v0.9.0-rc1` if batch-only demo matters

---

### Phase 10 — Serving and monitoring (portfolio MVP)

**Objective:** HTTP API **and** a **single** drift demonstration—not a full production monitoring stack.

**Tasks**

- [ ] FastAPI: `POST /predict`, `GET /health`; load model at startup; `curl` example in README.
- [ ] **Docker:** optional—`Dockerfile` / compose only if you want; do **not** block Phase 10 on containers.
- [ ] **Monitoring:** one Evidently (or similar) **drift report** (reference vs holdout/current slice) + **short conceptual note** in `docs/` (what drift means, when you would retrain)—no full on-call runbook required for MVP.

**Deliverables:** API runs locally; one regenerable drift artifact or script; optional Docker docs marked optional.

**GitHub:** `feat/phase-10-serving-monitoring` · Milestone **Phase 10 — Serving & monitoring** · Tag `v0.10-serving` · Release **Yes** — `v0.10.0` when API + README path works (Docker not required)

---

### Phase 11 — Documentation and portfolio release

**Objective:** README-first experience for reviewers.

**Tasks**

- [ ] README: problem, metrics, train/eval/serve commands, results table, limitations.
- [ ] Architecture (Mermaid or link to this doc’s diagram); `CHANGELOG.md` for `v1.0.0`; dataset citation.

**Deliverables:** New reader reproduces core flow from README (given data access).

**GitHub:** `docs/phase-11-readme-release` · Milestone **Phase 11 — Documentation & release** · Tag **`v1.0.0`** · Release **Yes**

---

### Phase 12 — Future improvements (backlog)

**Objective:** Post–v1.0: Feast, CI, registry, A/B narrative.

**Tasks:** Track as labeled issues; no blocking scope for portfolio MVP.

**GitHub:** Per issue · Milestone **Phase 12 — Future / backlog** · Tag per shipped item · Release **When** net-new capability warrants it

---

## Issue specifications (for `create_issues.sh`)

Same section order in every issue: **Context** → **Tasks** → **Definition of done** → **References**. Wording below is shortened for GitHub usability; full phase detail lives in [Phased Roadmap](#phased-roadmap).

---

### Issue: chore: Phase 0 — Python env, packaging, and test skeleton

**Context:** Reproducible venv + package layout so later work is not notebook-only.

**Tasks:** `pyproject.toml` or requirements; `src/churn_ml/`, `tests/test_smoke.py`; `.gitignore`, `.env.example`; README install.

**Definition of done:** `pytest` passes in fresh venv; no secrets committed.

**References:** Python packaging tutorial, PEP 621.

---

### Issue: docs: problem statement, metrics, and error-cost framing

**Context:** Align churn metrics and FN/FP costs before modeling.

**Tasks:** `docs/PROBLEM.md` (or config); Telco `Churn` semantics; tie metrics to evaluation code later.

**Definition of done:** One file explains success criteria.

**References:** Dataset docs, sklearn metrics guide.

---

### Issue: feat: raw ingestion, checksum, and Pandera validation

**Context:** Traceable raw data + single validation standard (Pandera) for CI and serving contracts.

**Tasks:** Download/docs + `sha256` in `data/raw/README.md`; Pandera schema + CLI non-zero on failure; `tests/fixtures/` for CI.

**Definition of done:** Validate fixture offline; raw CSV gitignored.

**References:** Pandera docs, Kaggle Telco dataset.

---

### Issue: feat: lean EDA — target, missingness, leakage notes

**Context:** Fast pre-pipeline checks without a profiling-tool dependency.

**Tasks:** Target balance; missing values (`TotalCharges`); short leakage checklist; stratify note for splits.

**Definition of done:** Written conclusions in notebook or `docs/`; optional extra plots only if time.

**References:** Telco field definitions.

---

### Issue: feat: stratified train/validation/test pipeline

**Context:** Honest metrics require disjoint splits and stable churn rate.

**Tasks:** Config ratios + seed; parquet to `data/processed/`; test no overlap.

**Definition of done:** Three files + churn rate per split logged.

**References:** `train_test_split(..., stratify=)`.

---

### Issue: feat: sklearn feature pipeline (MVP + optional extras)

**Context:** Train/inference parity via one serialized `Pipeline`.

**Tasks — MVP:** `ColumnTransformer`, encoding, fit-on-train. **Optional:** tenure bins / ratios if they help validation.

**Definition of done:** `joblib` round-trip; feature count documented.

**References:** sklearn `Pipeline` docs.

---

### Issue: feat: logistic regression baseline with MLflow logging

**Context:** Baseline + experiment tracking from the first real model run.

**Tasks:** Train baseline; MLflow params/metrics/artifact; `models/baseline.joblib`; README `mlflow ui`.

**Definition of done:** At least one MLflow run visible; `mlruns/` gitignored.

**References:** MLflow tracking docs, `LogisticRegression`.

---

### Issue: feat: LightGBM with stratified CV and MLflow

**Context:** One gradient-boosted model (portfolio clarity); no test-set tuning.

**Tasks:** LightGBM + search; log runs; save best params to `configs/`; leaderboard vs baseline on validation.

**Definition of done:** Validation leaderboard; test reserved for Phase 8.

**References:** LightGBM sklearn API.

---

### Issue: feat: evaluation — ROC/PR, confusion, threshold, champion

**Context:** Threshold drives business metrics; freeze champion for packaging.

**Tasks:** Plots in `reports/figures/`; threshold search; summary with split labels; no calibration required for MVP.

**Definition of done:** Reproducible script; champion + threshold documented.

**References:** `roc_curve`, `precision_recall_curve`.

---

### Issue: feat: package champion and batch predict CLI

**Context:** Versioned artifact + batch path before HTTP.

**Tasks:** Versioned `joblib` (or MLflow model); metadata JSON; pydantic-validated batch CLI.

**Definition of done:** One command runs on sample file; README command block.

**References:** joblib persistence.

---

### Issue: feat: FastAPI, optional Docker, one drift demo

**Context:** Portfolio serving + minimal monitoring story without a full ops stack.

**Tasks:** `/predict`, `/health`, README `curl`; **Docker optional** (document as add-on); **one** drift report (e.g. Evidently) + short `docs/` concept note (drift + when to retrain)—not a full on-call runbook.

**Definition of done:** API runs locally; drift script or artifact reproducible.

**References:** FastAPI docs, Evidently docs, Docker reference (if used).

---

### Issue: docs: README, results, CHANGELOG for v1.0.0

**Context:** Primary reviewer-facing deliverable.

**Tasks:** Commands for train/eval/serve; results + limits; Mermaid or link to roadmap diagram; `CHANGELOG.md` for `v1.0.0`.

**Definition of done:** Reader reproduces core path from README (with data access).

**References:** This roadmap.

---

### Issue: chore: backlog — CI, Feast/registry, post-v1

**Context:** Post-MVP hardening; does not block v1.0.0.

**Tasks:** Break out or defer: GitHub Actions, Feast/registry spike, MLflow registry narrative.

**Definition of done:** Issues labeled or explicit deferral note.

**References:** GitHub Actions, Feast, MLflow registry docs.

---

## Release strategy (summary)

| Version pattern | Example | Meaning |
|-----------------|---------|---------|
| Internal phase | `v0.5-features` | Optional tag after a phase merges to `main` |
| RC | `v0.9.0-rc1` | Batch inference or pre-API demo |
| Portfolio | `v1.0.0` | Train + evaluate + serve + docs (Docker not required) |

**Rule:** Open a **GitHub Release** when a stranger gets clear value (API, batch predict, or full README path). Internal work: milestones + optional tags only.

---

## CLI automation scripts (`scripts/`)

Bash + **GitHub CLI** (`gh`). Run from repo root on macOS, Linux, or **Git Bash / WSL** on Windows.

**Prerequisites**

```bash
gh auth login
gh repo set-default OWNER/REPO   # if the repo is not implicit from cwd
```

**Run order (once per empty repo, or after you delete labels/milestones/issues you want recreated)**

```bash
chmod +x scripts/*.sh
./scripts/create_labels.sh
./scripts/create_milestones.sh
./scripts/create_issues.sh
```

**Expected outputs**

| Script | Success looks like | stdout clues |
|--------|-------------------|--------------|
| `create_labels.sh` | One line per label: `Created label: …` or `Label exists, skip: …` | Ends with `Done. Verify with: gh label list` |
| `create_milestones.sh` | One line per milestone created or skipped | Ends with `Done. Verify with: gh api repos/.../milestones` |
| `create_issues.sh` | One line per issue: `Created issue: …` or `Issue exists, skip: …` | Ends with `Done. List issues: gh issue list` |

**Idempotency**

- **Labels:** skips if the name already exists (`gh label list`).
- **Milestones:** skips if the title already exists (any state), via paginated API list.
- **Issues:** skips if **any** issue (open or closed) has the **exact same title**—so you do not get duplicates when re-running; to recreate, rename or close/delete old issues first.

**Failure modes:** `gh` not authenticated → login error; wrong repo context → `gh repo set-default`; missing labels before issues → run `create_labels.sh` first.

---

## Appendix: GitHub CLI command reference

```bash
gh label create "type:feature" --color "0E8A16" --description "New capability"
gh label list
gh api repos/OWNER/REPO/milestones --jq '.[].title'
gh issue list --label "phase:modeling"
gh issue create --title "feat: example" --body "..." --label "type:feature"
git tag -a v0.6-baseline-mlflow -m "Baseline + MLflow"
git push origin v0.6-baseline-mlflow
gh release create v1.0.0 --title "v1.0.0" --notes-file CHANGELOG.md
```

Replace `OWNER/REPO` with `gh repo view --json nameWithOwner -q .nameWithOwner`.

---

## GitHub Workflow Standards Appendix

### Branch naming convention

| Pattern | Example |
|---------|---------|
| feature | `feat/phase-7-lightgbm` |
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
