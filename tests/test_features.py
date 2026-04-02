"""Phase 5 — sklearn preprocessing pipeline fit on train only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from churn_ml.data.split import load_split_config, run_split
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.features.pipeline import (
    build_feature_pipeline,
    build_manifest,
    fit_feature_pipeline,
    load_features_config,
    select_feature_matrix,
)
from churn_ml.features.run import main as features_main
from churn_ml.metrics import target_column

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"
REPO_SPLIT = Path(__file__).resolve().parents[1] / "configs" / "split.yaml"
REPO_FEATURES = Path(__file__).resolve().parents[1] / "configs" / "features.yaml"


def expanded_telco(n_copies: int = 35):
    base = load_raw_csv(FIXTURE)
    parts = []
    for i in range(n_copies):
        b = base.copy()
        b["customerID"] = b["customerID"].astype(str) + f"_b{i}"
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def train_parquet_dir(tmp_path: Path) -> Path:
    df = validate_raw_dataframe(expanded_telco(40))
    cfg = load_split_config(REPO_SPLIT)
    paths = run_split(df, tmp_path, cfg, input_path="fixture")
    return paths["train"]


def test_load_features_config():
    cfg = load_features_config(REPO_FEATURES)
    assert "gender" in cfg["categorical_features"]
    assert "tenure" in cfg["numeric_features"]


def test_select_feature_matrix_drops_id_and_target_implicitly():
    cfg = load_features_config(REPO_FEATURES)
    df = validate_raw_dataframe(expanded_telco(5))
    X = select_feature_matrix(df, cfg)
    assert "customerID" not in X.columns
    assert target_column() not in X.columns
    assert list(X.columns) == cfg["numeric_features"] + cfg["categorical_features"]


def test_select_feature_matrix_missing_column():
    cfg = load_features_config(REPO_FEATURES)
    df = pd.DataFrame({"tenure": [1]})
    with pytest.raises(ValueError, match="missing feature"):
        select_feature_matrix(df, cfg)


def test_fit_transform_shape_and_manifest(tmp_path: Path):
    train_path = train_parquet_dir(tmp_path)
    df = pd.read_parquet(train_path)
    cfg = load_features_config(REPO_FEATURES)
    pipe = fit_feature_pipeline(df, cfg)
    X = select_feature_matrix(df, cfg)
    out = pipe.transform(X)
    man = build_manifest(pipe, df, cfg, train_path=str(train_path))
    assert out.shape[0] == len(df)
    assert out.shape[1] == man["n_features_out"]
    assert man["n_features_out"] == len(man["feature_names_out"])
    assert man["max_categorical_cardinality_train"] >= 2


def test_unknown_category_at_transform(tmp_path: Path):
    train_path = train_parquet_dir(tmp_path)
    df = pd.read_parquet(train_path)
    cfg = load_features_config(REPO_FEATURES)
    pipe = fit_feature_pipeline(df, cfg)
    X = select_feature_matrix(df, cfg).copy()
    X.iloc[0, X.columns.get_loc("gender")] = "AlienGender"
    out = pipe.transform(X)
    assert out.shape[0] == len(X)


def test_joblib_roundtrip(tmp_path: Path):
    train_path = train_parquet_dir(tmp_path)
    df = pd.read_parquet(train_path)
    cfg = load_features_config(REPO_FEATURES)
    pipe = fit_feature_pipeline(df, cfg)
    path = tmp_path / "fp.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    X = select_feature_matrix(df, cfg)
    assert loaded.transform(X).shape == pipe.transform(X).shape


def test_features_cli(tmp_path: Path):
    train_path = train_parquet_dir(tmp_path)
    out = tmp_path / "models"
    code = features_main(["-t", str(train_path), "-o", str(out), "-c", str(REPO_FEATURES)])
    assert code == 0
    assert (out / "feature_pipeline.joblib").is_file()
    assert (out / "feature_manifest.json").is_file()
    man = json.loads((out / "feature_manifest.json").read_text(encoding="utf-8"))
    assert man["n_features_out"] > 0


def test_features_cli_subprocess(tmp_path: Path):
    train_path = train_parquet_dir(tmp_path)
    out = tmp_path / "models2"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_ml.features.run",
            "-t",
            str(train_path),
            "-o",
            str(out),
            "-c",
            str(REPO_FEATURES),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "feature_manifest.json").is_file()
