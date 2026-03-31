# Raw data — Telco churn

The full CSV is **gitignored**. Automated tests use `tests/fixtures/telco_sample.csv`.

Place the IBM / Kaggle file here as **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**.  
Dataset page: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Provenance

| Field | Value |
|-------|-------|
| Source URL | |
| SHA256 | |
| Data rows (excl. header) | |

## SHA256

Run from this directory (`data/raw/`):

```bash
sha256sum WA_Fn-UseC_-Telco-Customer-Churn.csv
```

```bat
certutil -hashfile WA_Fn-UseC_-Telco-Customer-Churn.csv SHA256
```

## Validation

```bash
pip install -e .
python -m churn_ml.data.validate
```

Optional path: `python -m churn_ml.data.validate /path/to/file.csv`. Same via `churn-validate`.

Exit codes: `0` valid, `1` schema error, `2` I/O or missing file.

Schema: `src/churn_ml/data/schema.py` (Pandera). `TotalCharges` is coerced to numeric after load (empty → NaN) before checks.
