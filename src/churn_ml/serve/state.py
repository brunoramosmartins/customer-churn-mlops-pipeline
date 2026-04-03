"""Load champion pipeline + threshold + Pydantic row model for serving."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from churn_ml.batch_predict.predict import load_champion_manifest, resolve_model_path
from churn_ml.batch_predict.row_model import build_inference_row_model
from churn_ml.features.pipeline import load_features_config


def serve_project_root() -> Path:
    env = os.environ.get("CHURN_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def champion_manifest_path(root: Path) -> Path:
    p = os.environ.get("CHURN_CHAMPION_MANIFEST")
    return Path(p).expanduser().resolve() if p else (root / "configs" / "champion.yaml")


def features_config_path(root: Path) -> Path:
    p = os.environ.get("CHURN_FEATURES_CONFIG")
    return Path(p).expanduser().resolve() if p else (root / "configs" / "features.yaml")


@dataclass
class ChampionState:
    root: Path
    manifest_path: Path
    features_path: Path
    model_path: Path
    pipeline: Any
    threshold: float
    feat_cfg: dict[str, Any]
    row_model: type


def load_champion_state(
    root: Path | None = None,
    *,
    manifest_path: Path | None = None,
    features_path: Path | None = None,
) -> ChampionState:
    root = root or serve_project_root()
    root = root.resolve()
    mp = manifest_path or champion_manifest_path(root)
    fp = features_path or features_config_path(root)
    manifest = load_champion_manifest(mp)
    model_path = resolve_model_path(root, manifest)
    pipeline = joblib.load(model_path)
    feat_cfg = load_features_config(fp)
    row_model = build_inference_row_model(feat_cfg)
    threshold = float(manifest["threshold"])
    return ChampionState(
        root=root,
        manifest_path=mp,
        features_path=fp,
        model_path=model_path,
        pipeline=pipeline,
        threshold=threshold,
        feat_cfg=feat_cfg,
        row_model=row_model,
    )
