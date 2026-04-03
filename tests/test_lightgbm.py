"""Phase 7 — LightGBM RandomizedSearchCV + MLflow (small search for CI)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import joblib
import yaml

from churn_ml.data.split import load_split_config, run_split
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.features.pipeline import load_features_config, select_feature_matrix
from churn_ml.models.lightgbm_tune import load_tune_lightgbm_config, train_lightgbm_tuned
from churn_ml.models.run_lightgbm import main as lightgbm_main

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SPLIT = REPO_ROOT / "configs" / "split.yaml"
REPO_FEATURES = REPO_ROOT / "configs" / "features.yaml"


def expanded_telco(n_copies: int = 55):
    import pandas as pd

    base = load_raw_csv(FIXTURE)
    parts = []
    for i in range(n_copies):
        b = base.copy()
        b["customerID"] = b["customerID"].astype(str) + f"_b{i}"
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def train_val_parquets(tmp_path: Path):
    df = validate_raw_dataframe(expanded_telco(60))
    cfg = load_split_config(REPO_SPLIT)
    paths = run_split(df, tmp_path, cfg, input_path="fixture")
    return paths["train"], paths["validation"]


MINIMAL_TUNE_YAML = """
random_state: 0
cv:
  n_splits: 2
  shuffle: true
  n_iter: 3
scoring: roc_auc
lightgbm:
  objective: binary
  verbosity: -1
  n_jobs: 1
  is_unbalance: true
param_distributions:
  classifier__learning_rate: [0.05, 0.1]
  classifier__num_leaves: [31, 50]
  classifier__n_estimators: [40, 80]
  classifier__max_depth: [4, 6]
mlflow:
  experiment_name: churn-ci-lgbm
  run_name: ci_lightgbm_search
outputs:
  model_path: model.joblib
  best_config_path: best.yaml
"""


def write_minimal_tune(tmp_path: Path) -> Path:
    p = tmp_path / "tune_min.yaml"
    p.write_text(MINIMAL_TUNE_YAML.strip(), encoding="utf-8")
    return p


def test_load_tune_lightgbm_config(tmp_path: Path):
    p = write_minimal_tune(tmp_path)
    cfg = load_tune_lightgbm_config(p)
    assert cfg["cv"]["n_iter"] == 3
    assert "classifier__learning_rate" in cfg["param_distributions"]


def test_train_lightgbm_tuned_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").resolve().as_uri())
    train_p, val_p = train_val_parquets(tmp_path)
    tune_p = write_minimal_tune(tmp_path)
    tune_cfg = load_tune_lightgbm_config(tune_p)
    import pandas as pd

    train_df = pd.read_parquet(train_p)
    val_df = pd.read_parquet(val_p)
    feat_cfg = load_features_config(REPO_FEATURES)
    model_p = tmp_path / "lgbm.joblib"
    best_p = tmp_path / "lgbm_best.yaml"
    pipe, metrics, search = train_lightgbm_tuned(
        train_df,
        val_df,
        feat_cfg,
        tune_cfg,
        tune_cfg_path=tune_p,
        output_path=model_p,
        best_config_path=best_p,
    )
    assert model_p.is_file()
    assert best_p.is_file()
    assert "val_roc_auc" in metrics
    loaded = joblib.load(model_p)
    X_va = select_feature_matrix(val_df, feat_cfg)
    assert loaded.predict(X_va).shape[0] == len(val_df)
    best_doc = yaml.safe_load(best_p.read_text(encoding="utf-8"))
    assert "lightgbm" in best_doc
    assert search.best_score_ <= 1.0


def test_lightgbm_cli(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns2").resolve().as_uri())
    train_p, val_p = train_val_parquets(tmp_path)
    tune_p = write_minimal_tune(tmp_path)
    out = tmp_path / "cli_lgbm.joblib"
    best = tmp_path / "cli_best.yaml"
    code = lightgbm_main(
        [
            "-t",
            str(train_p),
            "-v",
            str(val_p),
            "--features-config",
            str(REPO_FEATURES),
            "--tune-config",
            str(tune_p),
            "-o",
            str(out),
            "--best-config",
            str(best),
        ]
    )
    assert code == 0
    assert out.is_file()
    assert best.is_file()


def test_lightgbm_cli_subprocess(tmp_path: Path):
    train_p, val_p = train_val_parquets(tmp_path)
    tune_p = write_minimal_tune(tmp_path)
    out = tmp_path / "sub_lgbm.joblib"
    best = tmp_path / "sub_best.yaml"
    env = {**os.environ, "MLFLOW_TRACKING_URI": (tmp_path / "mlruns3").resolve().as_uri()}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_ml.models.run_lightgbm",
            "-t",
            str(train_p),
            "-v",
            str(val_p),
            "--features-config",
            str(REPO_FEATURES),
            "--tune-config",
            str(tune_p),
            "-o",
            str(out),
            "--best-config",
            str(best),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
