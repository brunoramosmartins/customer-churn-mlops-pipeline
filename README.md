# Customer churn MLOps pipeline

End-to-end machine learning project for **Telco customer churn** (binary classification), structured for a portfolio-ready, production-minded workflow: data validation, feature engineering, training, evaluation, experiment tracking, packaging, API serving, and monitoring.

## Roadmap and workflow

The full phased roadmap, GitHub conventions (labels, milestones, tags, releases), ASCII architecture, issue specifications, and automation instructions are in:

**[docs/ML_PROJECT_ROADMAP.md](docs/ML_PROJECT_ROADMAP.md)**

## Repository layout (summary)

| Path | Purpose |
|------|---------|
| `configs/` | Hyperparameters, paths, seeds |
| `data/raw/` | Immutable downloaded data (CSV gitignored; see `data/raw/README.md`) |
| `data/processed/` | Train / validation / test artifacts |
| `docs/` | Roadmap and problem docs |
| `models/` | Serialized pipelines (gitignored) |
| `notebooks/` | EDA and experiments (not production logic) |
| `reports/` | Figures, drift reports, evaluation summaries |
| `scripts/` | GitHub CLI automation (`create_labels`, `create_milestones`, `create_issues`) |
| `src/churn_ml/` | Importable Python package |
| `tests/` | Unit and integration tests |

## GitHub automation (Bash + `gh`)

On **macOS**, **Linux**, or **Git Bash / WSL** on Windows:

```bash
gh auth login
chmod +x scripts/*.sh
./scripts/create_labels.sh
./scripts/create_milestones.sh
./scripts/create_issues.sh
```

Run these once per repository (or after deleting labels/milestones/issues you want to recreate). Issues are skipped if a title already exists.

## Status

Scaffolding and roadmap are in place; training code, API, and CI are implemented phase by phase per the roadmap.

## License

See [LICENSE](LICENSE).
