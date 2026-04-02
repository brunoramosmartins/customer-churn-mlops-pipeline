# Problem definition — Telco customer churn

## Business objective

Reduce **customer churn** by identifying subscribers at risk of leaving, so retention actions (offers, support, pricing) can be targeted. Success is measured by model quality on held-out data and, in production, by retention lift—not only by accuracy in a spreadsheet.

## Machine learning formulation

| Item | Choice |
|------|--------|
| Task | **Binary classification** |
| Unit of prediction | One row = one customer (one snapshot in the Telco feature table) |
| Target column | `Churn` |
| Positive class (`y=1`) | **`Yes`** — customer churned within the observation window used in the dataset |
| Negative class (`y=0`) | **`No`** |

The public Telco dataset encodes churn as categorical `Yes` / `No`. Training code will map this to integers `{0, 1}` with **`1 = churn (positive)`** for metrics such as recall and PR-AUC.

## Success metrics (primary)

Report these on **validation** during development; use **test** once for a final, honest estimate after the champion model and threshold are fixed.

| Metric | Role |
|--------|------|
| **ROC-AUC** | Ranking quality across thresholds; robust for imbalanced problems when you care about separability. |
| **Average precision (PR-AUC)** | Sensitive to imbalance; useful when the positive class (churn) is rare. |
| **F1-score** | Harmonic mean of precision and recall at a chosen threshold; single scalar for balance. |
| **Recall on churn (sensitivity)** | Share of actual churners correctly flagged. Often **business-critical** if missing churn is expensive. |

Optional secondary metrics: precision at churn, specificity, confusion matrix counts at the operating threshold.

## Cost of errors (conceptual)

| Prediction | Reality | Typical interpretation |
|------------|---------|-------------------------|
| **False negative** | Churner predicted as stay | Missed intervention; direct revenue / CLV loss. Often **more costly** than a false positive. |
| **False positive** | Non-churner predicted as churn | Unnecessary retention cost, contact fatigue, or discount given when not needed. |

**Threshold policy:** Default `0.5` on predicted churn probability is usually **not** optimal. After Phase 8, choose a threshold on **validation** (e.g. maximize F-beta with emphasis on recall, or enforce a **minimum recall on churn**). The numeric floor in `configs/metrics.yaml` is a **starting placeholder**—revise after EDA (Phase 3).

## Non-goals (for this portfolio scope)

- Causal inference of *why* customers churn (explainability add-ons may come later).
- Multi-horizon or survival models unless explicitly added as future work.
- Training on the test split or tuning on test data.

## References

- IBM / Kaggle **Telco Customer Churn** dataset (column names and semantics follow the published CSV).
- Project metrics contract (machine-readable): [`configs/metrics.yaml`](../configs/metrics.yaml).
- Code entrypoint for the same contract: `churn_ml.metrics` in `src/churn_ml/metrics.py`.
- After EDA, review **[EDA_LEAKAGE_CHECKLIST.md](EDA_LEAKAGE_CHECKLIST.md)** before modeling.
