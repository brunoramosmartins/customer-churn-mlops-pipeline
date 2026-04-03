# Changelog

All notable portfolio-facing changes are documented here. The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.1] — 2026-03-30

### Changed

- **Portable artifacts:** `churn-evaluate` and batch metadata now record **repo-relative POSIX paths** in JSON/Markdown; `configs/champion.yaml` no longer stores machine-specific `model_path_resolved` (inference uses `model_path` under repo root).
- **`churn_ml.__version__`** reads from package metadata (`importlib.metadata`) with a safe fallback.

### Added

- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [`.github/dependabot.yml`](.github/dependabot.yml) (pip + GitHub Actions).
- [`churn_ml.fsutil.path_relative_to_repo`](src/churn_ml/fsutil.py) and tests.

### Fixed

- CI workflow: explicit read-only `permissions` and PR **concurrency** (`cancel-in-progress`).

## [1.0.0] — 2026-03-30

### Documentation (Phase 11 — portfolio release)

- README-first flow: quick start to reproduce train → evaluate → optional batch/API/drift; holdout **results table** aligned with committed `reports/evaluation_summary.*`; **limitations**; **dataset citation**.
- **Architecture:** Mermaid diagram in the README, plus the ASCII overview in [docs/ML_PROJECT_ROADMAP.md](docs/ML_PROJECT_ROADMAP.md#architecture-overview-ascii).
- This file (`CHANGELOG.md`) added for the **`v1.0.0`** release tag.

### Notes

- **Runtime behavior** is unchanged from Phase 10; `v1.0.0` marks documentation and portfolio-readiness, not a new model contract.
- Re-run `churn-evaluate` after retraining to refresh `reports/evaluation_summary.*` and [configs/champion.yaml](configs/champion.yaml).

When you publish on GitHub, create releases **`v1.0.0`** and **`v1.0.1`** (or a single **`v1.0.1`** that supersedes 1.0.0) and paste the matching sections into the release notes.
