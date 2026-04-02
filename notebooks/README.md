# Notebooks

Optional exploration. The **reproducible** EDA path is the CLI:

```bash
pip install -e .
python -m churn_ml.eda.run --input path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
# or: churn-eda -i path/to/file.csv
```

Outputs: `reports/eda_summary.md`, `reports/eda_summary.json`, `reports/figures/`.

Open `01_eda.ipynb` if you prefer stepping through the same flow in Jupyter.
