"""Logistic baseline: preprocess (Phase 5) + classifier; validation metrics for MLflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from churn_ml.features.pipeline import build_feature_pipeline, select_feature_matrix
from churn_ml.metrics import positive_class_label, target_column


def load_train_baseline_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("train_baseline config must be a YAML mapping")
    for key in ("random_state", "logistic_regression", "mlflow"):
        if key not in cfg:
            raise ValueError(f"train_baseline config missing required key: {key}")
    if not isinstance(cfg["mlflow"], dict):
        raise ValueError("mlflow section must be a mapping")
    if "experiment_name" not in cfg["mlflow"]:
        raise ValueError("mlflow.experiment_name is required")
    return cfg


def y_positive_binary(df: pd.DataFrame) -> pd.Series:
    col = target_column()
    pos = positive_class_label()
    if col not in df.columns:
        raise ValueError(f"target column {col!r} not in dataframe")
    return (df[col] == pos).astype(int)


def build_baseline_pipeline(feat_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> Pipeline:
    """Unfitted Pipeline: preprocess (ColumnTransformer) + LogisticRegression."""
    prep = build_feature_pipeline(feat_cfg)
    ct = prep.named_steps["preprocess"]
    lr_block = dict(train_cfg["logistic_regression"])
    lr_block["random_state"] = int(train_cfg["random_state"])
    cw = train_cfg.get("class_weight")
    if cw is not None:
        lr_block["class_weight"] = cw
    clf = LogisticRegression(**lr_block)
    return Pipeline([("preprocess", ct), ("classifier", clf)])


def compute_val_metrics(
    pipe: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "val_roc_auc": float(roc_auc_score(y_val, proba)),
        "val_average_precision": float(average_precision_score(y_val, proba)),
        "val_f1": float(f1_score(y_val, pred)),
        "val_recall_churn": float(recall_score(y_val, pred)),
    }


def default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feat_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    *,
    output_path: Path,
    tracking_uri: str | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit on train, score on val, log to MLflow, save ``output_path`` (full fitted pipeline)."""
    uri = tracking_uri if tracking_uri is not None else default_tracking_uri()
    mlflow.set_tracking_uri(uri)

    X_tr = select_feature_matrix(train_df, feat_cfg)
    y_tr = y_positive_binary(train_df)
    X_va = select_feature_matrix(val_df, feat_cfg)
    y_va = y_positive_binary(val_df)

    pipe = build_baseline_pipeline(feat_cfg, train_cfg)
    pipe.fit(X_tr, y_tr)
    metrics = compute_val_metrics(pipe, X_va, y_va)

    mf = train_cfg["mlflow"]
    experiment_name = str(mf["experiment_name"])
    run_name = str(mf.get("run_name") or "baseline_logistic")
    mlflow.set_experiment(experiment_name)

    lr_params = dict(train_cfg["logistic_regression"])
    lr_params["random_state"] = train_cfg["random_state"]
    if train_cfg.get("class_weight") is not None:
        lr_params["class_weight"] = train_cfg["class_weight"]

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", "logistic_regression")
        mlflow.set_tag("phase", "6_baseline")
        for k, v in lr_params.items():
            mlflow.log_param(f"lr_{k}", v)
        mlflow.log_param("random_state", train_cfg["random_state"])
        mlflow.log_param(
            "class_weight",
            train_cfg.get("class_weight") if train_cfg.get("class_weight") is not None else "sklearn_default",
        )
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_val", len(val_df))
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(pipe, artifact_path="sklearn-model")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out)
        mlflow.log_artifact(str(out), artifact_path="baseline-joblib")

    return pipe, metrics
