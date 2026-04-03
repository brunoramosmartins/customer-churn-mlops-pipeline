# Notebooks

Optional exploration. The **reproducible** EDA path is the CLI:

```bash
pip install -e .
python -m churn_ml.eda.run --input path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
# or: churn-eda -i path/to/file.csv
```

Outputs: `reports/eda_summary.md`, `reports/eda_summary.json`, `reports/figures/`.

- `01_eda.ipynb` — same EDA pipeline as the CLI, step by step.
- `02_pipeline_walkthrough.ipynb` — **end-to-end narrative** (validate → split → features → baseline + MLflow) using `churn_ml` APIs, with design notes (e.g. imputation trade-offs). Writes to `notebooks/_walkthrough_outputs/` (gitignored).
