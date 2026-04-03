"""Phase 6 — logistic baseline + MLflow logging."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from churn_ml.data.split import load_split_config, run_split
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.features.pipeline import load_features_config, select_feature_matrix
from churn_ml.models.baseline import (
    build_baseline_pipeline,
    compute_val_metrics,
    load_train_baseline_config,
    train_baseline,
)
from churn_ml.metrics import positive_class_label, target_column
from churn_ml.models.run_baseline import main as baseline_main

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SPLIT = REPO_ROOT / "configs" / "split.yaml"
REPO_FEATURES = REPO_ROOT / "configs" / "features.yaml"
REPO_TRAIN = REPO_ROOT / "configs" / "train_baseline.yaml"


def expanded_telco(n_copies: int = 50):
    base = load_raw_csv(FIXTURE)
    parts = []
    for i in range(n_copies):
        b = base.copy()
        b["customerID"] = b["customerID"].astype(str) + f"_b{i}"
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def train_val_parquets(tmp_path: Path):
    df = validate_raw_dataframe(expanded_telco(55))
    cfg = load_split_config(REPO_SPLIT)
    paths = run_split(df, tmp_path, cfg, input_path="fixture")
    return paths["train"], paths["validation"]


def test_load_train_baseline_config():
    cfg = load_train_baseline_config(REPO_TRAIN)
    assert cfg["mlflow"]["experiment_name"]
    assert cfg["logistic_regression"]["solver"] == "lbfgs"


def test_build_baseline_pipeline_fit_predict(tmp_path: Path):
    train_p, val_p = train_val_parquets(tmp_path)
    train_df = pd.read_parquet(train_p)
    val_df = pd.read_parquet(val_p)
    feat_cfg = load_features_config(REPO_FEATURES)
    train_cfg = load_train_baseline_config(REPO_TRAIN)
    pipe = build_baseline_pipeline(feat_cfg, train_cfg)
    X_tr = select_feature_matrix(train_df, feat_cfg)
    y_tr = (train_df[target_column()] == positive_class_label()).astype(int)
    pipe.fit(X_tr, y_tr)
    X_va = select_feature_matrix(val_df, feat_cfg)
    y_va = (val_df[target_column()] == positive_class_label()).astype(int)
    m = compute_val_metrics(pipe, X_va, y_va)
    assert 0.0 <= m["val_roc_auc"] <= 1.0
    assert pipe.predict(X_va).shape[0] == len(val_df)


def test_train_baseline_mlflow_and_joblib(tmp_path: Path, monkeypatch):
    mlruns = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlruns.as_uri())
    train_p, val_p = train_val_parquets(tmp_path)
    train_df = pd.read_parquet(train_p)
    val_df = pd.read_parquet(val_p)
    feat_cfg = load_features_config(REPO_FEATURES)
    train_cfg = load_train_baseline_config(REPO_TRAIN)
    out = tmp_path / "baseline.joblib"
    pipe, metrics = train_baseline(
        train_df,
        val_df,
        feat_cfg,
        train_cfg,
        output_path=out,
    )
    assert out.is_file()
    loaded = joblib.load(out)
    assert loaded.predict(select_feature_matrix(val_df, feat_cfg)).shape[0] == len(val_df)
    assert "val_roc_auc" in metrics
    assert mlruns.is_dir()


def test_baseline_cli(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns2").as_uri())
    train_p, val_p = train_val_parquets(tmp_path)
    out = tmp_path / "models" / "baseline.joblib"
    code = baseline_main(
        [
            "-t",
            str(train_p),
            "-v",
            str(val_p),
            "--features-config",
            str(REPO_FEATURES),
            "--train-config",
            str(REPO_TRAIN),
            "-o",
            str(out),
        ]
    )
    assert code == 0
    assert out.is_file()


def test_baseline_cli_subprocess(tmp_path: Path):
    train_p, val_p = train_val_parquets(tmp_path)
    out = tmp_path / "bl.joblib"
    env = {**os.environ, "MLFLOW_TRACKING_URI": (tmp_path / "mlruns3").as_uri()}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_ml.models.run_baseline",
            "-t",
            str(train_p),
            "-v",
            str(val_p),
            "--features-config",
            str(REPO_FEATURES),
            "--train-config",
            str(REPO_TRAIN),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
