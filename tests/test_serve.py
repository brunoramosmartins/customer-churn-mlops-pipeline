"""Phase 10 — FastAPI /health and /predict."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from churn_ml.batch_predict.row_model import build_inference_row_model
from churn_ml.data.split import load_split_config, run_split
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.features.pipeline import load_features_config, select_feature_matrix
from churn_ml.models.baseline import build_baseline_pipeline, load_train_baseline_config, y_positive_binary
from churn_ml.serve.app import create_test_app
from churn_ml.serve.state import load_champion_state

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_FEATURES = REPO_ROOT / "configs" / "features.yaml"
REPO_TRAIN_BASELINE = REPO_ROOT / "configs" / "train_baseline.yaml"
REPO_SPLIT = REPO_ROOT / "configs" / "split.yaml"


def _expanded_telco(n_copies: int = 20):
    base = load_raw_csv(FIXTURE)
    parts = []
    for i in range(n_copies):
        b = base.copy()
        b["customerID"] = b["customerID"].astype(str) + f"_sv{i}"
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def _serve_bundle(tmp_path: Path):
    df = validate_raw_dataframe(_expanded_telco(25))
    cfg = load_split_config(REPO_SPLIT)
    train_p = run_split(df, tmp_path, cfg, input_path="fixture")["train"]
    train_df = pd.read_parquet(train_p)
    feat_cfg = load_features_config(REPO_FEATURES)
    train_cfg = load_train_baseline_config(REPO_TRAIN_BASELINE)
    pipe = build_baseline_pipeline(feat_cfg, train_cfg)
    pipe.fit(select_feature_matrix(train_df, feat_cfg), y_positive_binary(train_df))
    joblib_path = tmp_path / "champ.joblib"
    joblib.dump(pipe, joblib_path)
    champ_yaml = tmp_path / "champion.yaml"
    champ_yaml.write_text(
        yaml.dump({"model_path": "champ.joblib", "threshold": 0.45}, default_flow_style=False),
        encoding="utf-8",
    )
    return tmp_path, champ_yaml


def test_health_and_predict(tmp_path: Path):
    root, champ = _serve_bundle(tmp_path)
    state = load_champion_state(root, manifest_path=champ, features_path=REPO_FEATURES)
    app = create_test_app(state)
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "threshold" in body

    feat = load_features_config(REPO_FEATURES)
    Row = build_inference_row_model(feat)
    row = Row(
        customerID="x1",
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=55.0,
        TotalCharges=None,
        Churn=None,
    )
    pr = client.post("/predict", json=row.model_dump())
    assert pr.status_code == 200
    out = pr.json()
    assert "churn_probability" in out
    assert "predicted_churn" in out
    assert 0.0 <= out["churn_probability"] <= 1.0


def test_predict_rejects_extra_field(tmp_path: Path):
    root, champ = _serve_bundle(tmp_path)
    state = load_champion_state(root, manifest_path=champ, features_path=REPO_FEATURES)
    app = create_test_app(state)
    client = TestClient(app)
    payload = {
        "customerID": "x",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.0,
        "bogus": 1,
    }
    pr = client.post("/predict", json=payload)
    assert pr.status_code == 422
