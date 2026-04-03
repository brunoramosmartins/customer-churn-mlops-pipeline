# Phase 8 — Evaluation summary

## Champion

- **Model:** `models/lightgbm_tuned.joblib` (source: `champion_model_path`)
- **Threshold (selected on validation):** `0.470000`
- **Recall floor used:** `0.5`; **F-beta (maximized under floor):** beta = `1.25`

## Which metric belongs to which split

| Metric | Split | Notes |
|--------|-------|-------|
| ROC-AUC, Average precision (ranking) | validation / test | Threshold-free; test reported once. |
| Recall / precision / F1 / F-beta at threshold | validation | Used to *choose* threshold under recall floor. |
| Same at *frozen* threshold | test | One-shot; same cutoff as validation. |
| Confusion matrix figure | test | At frozen threshold. |

## Key numbers

### Validation (ranking)

- ROC-AUC: **0.8332**
- Average precision: **0.6374**

### Validation (at chosen threshold)

- Recall (churn): **0.8043**
- Precision (churn): **0.5067**
- F1: **0.6217**

### Test — one shot (ranking)

- ROC-AUC: **0.8509**
- Average precision: **0.6581**

### Test — frozen threshold from validation

- Recall (churn): **0.8429**
- Precision (churn): **0.5187**
- F1: **0.6422**

## Figures

- `roc_validation` → `reports/figures/phase8_roc_validation.png`
- `pr_validation` → `reports/figures/phase8_pr_validation.png`
- `roc_test` → `reports/figures/phase8_roc_test.png`
- `pr_test` → `reports/figures/phase8_pr_test.png`
- `threshold_sweep_validation` → `reports/figures/phase8_threshold_sweep_validation.png`
- `confusion_matrix_test` → `reports/figures/phase8_confusion_matrix_test.png`

## Notes

- Calibration plots are out of scope for portfolio v1 (roadmap).
- Do not re-tune threshold on test; Phase 9+ should load configs/champion.yaml.
