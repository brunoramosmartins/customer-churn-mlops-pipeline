"""Sklearn preprocessing pipeline: numeric impute+scale, categorical impute+one-hot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_ml.metrics import target_column


def load_features_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("features config must be a YAML mapping")
    for key in ("id_column", "numeric_features", "categorical_features"):
        if key not in cfg:
            raise ValueError(f"features config missing required key: {key}")
    cfg["id_column"] = str(cfg["id_column"])
    cfg["numeric_features"] = [str(c) for c in cfg["numeric_features"]]
    cfg["categorical_features"] = [str(c) for c in cfg["categorical_features"]]
    enc = cfg.get("encoding", "one_hot")
    if enc != "one_hot":
        raise ValueError(f"unsupported encoding: {enc!r} (only 'one_hot' is implemented)")
    return cfg


def select_feature_matrix(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return only modeling columns (no ID, no target)."""
    num = cfg["numeric_features"]
    cat = cfg["categorical_features"]
    want = num + cat
    missing = set(want) - set(df.columns)
    if missing:
        raise ValueError(f"train data missing feature columns: {sorted(missing)}")
    return df[want].copy()


def build_feature_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """Unfitted Pipeline with a single step ``preprocess`` (ColumnTransformer)."""
    num_cols = cfg["numeric_features"]
    cat_cols = cfg["categorical_features"]
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", ct)])


def fit_feature_pipeline(df: pd.DataFrame, cfg: dict[str, Any]) -> Pipeline:
    """Fit preprocessing on training rows only."""
    X = select_feature_matrix(df, cfg)
    pipe = build_feature_pipeline(cfg)
    pipe.fit(X)
    return pipe


def build_manifest(
    fitted: Pipeline,
    train_df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    train_path: str | None = None,
) -> dict[str, Any]:
    ct = fitted.named_steps["preprocess"]
    names = list(ct.get_feature_names_out())
    X = select_feature_matrix(train_df, cfg)
    cat = cfg["categorical_features"]
    levels = {c: int(X[c].nunique(dropna=False)) for c in cat}
    max_card = max(levels.values()) if levels else 0
    return {
        "train_path": train_path,
        "n_train_rows": len(train_df),
        "id_column": cfg["id_column"],
        "target_column": target_column(),
        "numeric_features": cfg["numeric_features"],
        "categorical_features": cfg["categorical_features"],
        "categorical_nunique_train": levels,
        "max_categorical_cardinality_train": max_card,
        "cardinality_note": (
            "Telco Kaggle features are low-cardinality categoricals; one-hot is appropriate. "
            "Unknown categories at inference map to zero via handle_unknown='ignore'."
        ),
        "encoding": "one_hot",
        "n_features_out": len(names),
        "feature_names_out": names,
    }


def save_artifacts(
    fitted: Pipeline,
    manifest: dict[str, Any],
    *,
    pipeline_path: Path,
    manifest_path: Path,
) -> None:
    pipeline_path = Path(pipeline_path)
    manifest_path = Path(manifest_path)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, pipeline_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_feature_pipeline(path: Path) -> Pipeline:
    return joblib.load(path)
