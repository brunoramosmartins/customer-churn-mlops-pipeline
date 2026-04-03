"""Phase 10 — univariate drift helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_ml.features.pipeline import load_features_config
from churn_ml.monitoring.drift import run_drift_analysis, write_drift_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_FEATURES = REPO_ROOT / "configs" / "features.yaml"


def test_run_drift_detects_numeric_shift():
    feat = load_features_config(REPO_FEATURES)
    n = 500
    ref = pd.DataFrame(
        {
            "SeniorCitizen": [0] * n,
            "tenure": [12] * n,
            "MonthlyCharges": pd.Series(range(n), dtype=float) * 0.1 + 20,
            "TotalCharges": [100.0] * n,
            "gender": ["Female"] * n,
            "Partner": ["Yes"] * n,
            "Dependents": ["No"] * n,
            "PhoneService": ["Yes"] * n,
            "MultipleLines": ["No"] * n,
            "InternetService": ["DSL"] * n,
            "OnlineSecurity": ["No"] * n,
            "OnlineBackup": ["No"] * n,
            "DeviceProtection": ["No"] * n,
            "TechSupport": ["No"] * n,
            "StreamingTV": ["No"] * n,
            "StreamingMovies": ["No"] * n,
            "Contract": ["Month-to-month"] * n,
            "PaperlessBilling": ["Yes"] * n,
            "PaymentMethod": ["Electronic check"] * n,
        }
    )
    cur = ref.copy()
    cur["MonthlyCharges"] = cur["MonthlyCharges"] + 80.0
    rows = run_drift_analysis(ref, cur, feat, alpha=0.05)
    mc = next(r for r in rows if r.column == "MonthlyCharges")
    assert mc.p_value is not None and mc.p_value < 0.05
    assert mc.drift_flag is True


def test_write_drift_artifacts(tmp_path: Path):
    feat = load_features_config(REPO_FEATURES)
    ref = pd.DataFrame(
        {
            "SeniorCitizen": [0, 0],
            "tenure": [1, 2],
            "MonthlyCharges": [20.0, 21.0],
            "TotalCharges": [20.0, 42.0],
            "gender": ["Female", "Female"],
            "Partner": ["Yes", "Yes"],
            "Dependents": ["No", "No"],
            "PhoneService": ["No", "No"],
            "MultipleLines": ["No phone service", "No phone service"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Electronic check", "Electronic check"],
        }
    )
    rows = run_drift_analysis(ref, ref.copy(), feat)
    html_p = tmp_path / "d.html"
    json_p = tmp_path / "d.json"
    write_drift_artifacts(
        rows,
        html_path=html_p,
        json_path=json_p,
        reference_path="ref.parquet",
        current_path="cur.parquet",
        alpha=0.05,
    )
    assert html_p.is_file() and json_p.is_file()
    assert "Drift" in html_p.read_text(encoding="utf-8")
