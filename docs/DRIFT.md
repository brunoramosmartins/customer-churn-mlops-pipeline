# Data drift (conceptual note — Phase 10)

**What it means.** In churn (and most supervised ML), the model was trained on a **reference** distribution of inputs (for example, training data or a baseline month). **Drift** is when the **live or recent** data distribution differs enough that the model’s behavior may degrade—even if the code and weights never changed.

**Why it matters.** Tree and linear models assume future rows “look like” past rows in the feature space. If marketing, pricing, or the customer base shifts, calibration and optimal thresholds can slip; rare categories may appear that preprocessing never saw.

**What we do here.** The repo includes a **lightweight univariate check** (`churn_ml.monitoring`): Kolmogorov–Smirnov on numeric columns and chi-square–style tests on categoricals, aligned with `configs/features.yaml`. It produces `reports/drift_report.html` and `reports/drift_summary.json`. This is a **portfolio demonstration**, not a full production monitoring stack (no Evidently server, no automated alerts).

**When to retrain (rule of thumb).** Combine drift signals with **business metrics** and **model performance** on labeled slices: if many features show low p-values *and* precision/recall on recent labeled churn moves the wrong way, schedule a retrain or recalibrate. A few significant tests alone can be noise—especially with large *n*—so avoid blind automation on p-values alone.

**Regenerate the report** (after Phase 4 splits exist):

```bash
pip install -e ".[portfolio]"
python -m churn_ml.monitoring.run_drift --config configs/drift_report.yaml
```
