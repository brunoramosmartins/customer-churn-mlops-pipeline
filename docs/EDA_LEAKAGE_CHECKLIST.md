# EDA — leakage and target sanity (Telco churn)

Use this once per dataset version before trusting train/val/test metrics.

## Target and timing

- [ ] **`Churn`** is the label; confirm values match contract (`Yes` / `No` in raw data).
- [ ] Dataset is a **single-period snapshot** (no explicit time column): there is no time-based split in the public file—acknowledge that **concept drift** is out of scope for this CSV-only project.

## Identifier and high-cardinality columns

- [ ] **`customerID`** is a unique key, not a causal feature—**exclude from modeling** (or use only as row id).
- [ ] No column is a **deterministic function** of the target (e.g. a “churn_reason” field filled only after churn).

## Features that are plausible but collinear

- [ ] **`TotalCharges`** vs **`tenure`** × **`MonthlyCharges`**: expect correlation; not necessarily leakage, but watch **multicollinearity** in linear models.

## Post-hoc and operational fields

- [ ] Confirm no columns encode **outcome after** the churn decision (none expected in the standard Telco schema).

## Splits (Phase 4)

- [ ] Use **stratified** splits on **`Churn`** to preserve class balance across train/val/test.
