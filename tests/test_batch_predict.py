"""Phase 9 — batch predict with Pydantic validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from churn_ml.batch_predict.predict import batch_predict, load_batch_predict_config, load_champion_manifest
from churn_ml.batch_predict.row_model import build_inference_row_model
from churn_ml.batch_predict.run import main as batch_main
from churn_ml.data.split import load_split_config, run_split
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.features.pipeline import load_features_config, select_feature_matrix
from churn_ml.models.baseline import (
    build_baseline_pipeline,
    load_train_baseline_config,
    y_positive_binary,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_FEATURES = REPO_ROOT / "configs" / "features.yaml"
REPO_TRAIN_BASELINE = REPO_ROOT / "configs" / "train_baseline.yaml"
REPO_SPLIT = REPO_ROOT / "configs" / "split.yaml"


def expanded_telco(n_copies: int = 20):
    base = load_raw_csv(FIXTURE)
    parts = []
    for i in range(n_copies):
        b = base.copy()
        b["customerID"] = b["customerID"].astype(str) + f"_bp{i}"
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def champion_bundle(tmp_path: Path):
    df = validate_raw_dataframe(expanded_telco(25))
    cfg = load_split_config(REPO_SPLIT)
    train_p = run_split(df, tmp_path, cfg, input_path="fixture")["train"]
    train_df = pd.read_parquet(train_p)
    feat_cfg = load_features_config(REPO_FEATURES)
    train_cfg = load_train_baseline_config(REPO_TRAIN_BASELINE)
    pipe = build_baseline_pipeline(feat_cfg, train_cfg)
    pipe.fit(select_feature_matrix(train_df, feat_cfg), y_positive_binary(train_df))
    joblib_path = tmp_path / "champ.joblib"
    import joblib

    joblib.dump(pipe, joblib_path)
    champ_yaml = tmp_path / "champion.yaml"
    champ_yaml.write_text(
        yaml.dump(
            {
                "model_path": "champ.joblib",
                "threshold": 0.45,
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    batch_yaml = tmp_path / "batch_predict.yaml"
    batch_yaml.write_text(
        yaml.dump(
            {
                "champion_manifest": "champion.yaml",
                "features_config": str(REPO_FEATURES.resolve()),
                "artifact_version": "test",
                "default_output": "out.parquet",
                "metadata_output": "meta.json",
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "in.csv"
    df.to_csv(csv_path, index=False)
    return batch_yaml, csv_path, joblib_path


def test_total_charges_whitespace_string_becomes_none():
    feat = load_features_config(REPO_FEATURES)
    M = build_inference_row_model(feat)
    base = {
        "customerID": "x",
        "SeniorCitizen": 0,
        "tenure": 1,
        "MonthlyCharges": 29.0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
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
        "Churn": "No",
    }
    r = M.model_validate({**base, "TotalCharges": " "})
    assert r.TotalCharges is None


def test_build_inference_row_model():
    feat = load_features_config(REPO_FEATURES)
    M = build_inference_row_model(feat)
    r = M.model_validate(
        {
            "customerID": "x",
            "SeniorCitizen": 0,
            "tenure": 1,
            "MonthlyCharges": 29.0,
            "TotalCharges": None,
            "gender": "Female",
            "Partner": "Yes",
            "Dependents": "No",
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
            "Churn": "No",
        }
    )
    assert r.customerID == "x"


def test_build_inference_row_model_rejects_extra_column():
    feat = load_features_config(REPO_FEATURES)
    M = build_inference_row_model(feat)
    row = {
        "customerID": "x",
        "SeniorCitizen": 0,
        "tenure": 1,
        "MonthlyCharges": 29.0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
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
        "bogus": 1,
    }
    with pytest.raises(ValidationError):
        M.model_validate(row)


def test_batch_predict_end_to_end(tmp_path: Path):
    batch_yaml, csv_path, _ = champion_bundle(tmp_path)
    cfg = load_batch_predict_config(batch_yaml)
    out = tmp_path / "pred.parquet"
    meta = batch_predict(
        tmp_path,
        input_path=csv_path,
        output_path=out,
        batch_cfg=cfg,
        champion_manifest_path=tmp_path / "champion.yaml",
        features_config_path=REPO_FEATURES,
        write_metadata=True,
    )
    assert out.is_file()
    pred = pd.read_parquet(out)
    assert "churn_probability" in pred.columns
    assert "predicted_churn" in pred.columns
    assert "actual_churn" in pred.columns
    assert (tmp_path / "meta.json").is_file()
    m = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert m["n_rows"] == len(pred)
    assert m["input_sha256"]


def test_batch_predict_cli(tmp_path: Path):
    batch_yaml, csv_path, _ = champion_bundle(tmp_path)
    out = tmp_path / "cli_out.csv"
    code = batch_main(
        [
            "--root",
            str(tmp_path),
            "--batch-config",
            str(batch_yaml),
            "-i",
            str(csv_path),
            "-o",
            str(out),
            "--no-metadata",
        ]
    )
    assert code == 0
    assert out.is_file()


def test_batch_cli_subprocess(tmp_path: Path):
    batch_yaml, csv_path, _ = champion_bundle(tmp_path)
    out = tmp_path / "sub_out.parquet"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_ml.batch_predict.run",
            "--root",
            str(tmp_path),
            "--batch-config",
            str(batch_yaml),
            "-i",
            str(csv_path),
            "-o",
            str(out),
            "--no-metadata",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()


def test_load_champion_manifest_requires_threshold(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump({"model_path": "x.joblib"}), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_champion_manifest(p)
