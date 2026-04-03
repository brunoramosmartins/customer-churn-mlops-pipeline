"""CLI: LightGBM tuning on train (CV), validation metrics, MLflow, configs/lightgbm_best.yaml (Phase 7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from churn_ml.features.pipeline import load_features_config
from churn_ml.models.baseline import default_tracking_uri
from churn_ml.models.lightgbm_tune import load_tune_lightgbm_config, train_lightgbm_tuned


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tune LightGBM on train (stratified CV); log MLflow; save tuned pipeline + best config (Phase 7).",
    )
    parser.add_argument(
        "--train",
        "-t",
        type=Path,
        default=None,
        help="Training Parquet (default: data/processed/train.parquet).",
    )
    parser.add_argument(
        "--validation",
        "-v",
        type=Path,
        default=None,
        help="Validation Parquet (default: data/processed/validation.parquet).",
    )
    parser.add_argument(
        "--features-config",
        type=Path,
        default=None,
        help="features.yaml (default: configs/features.yaml).",
    )
    parser.add_argument(
        "--tune-config",
        type=Path,
        default=None,
        help="tune_lightgbm.yaml (default: configs/tune_lightgbm.yaml).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Fitted pipeline joblib (default: from tune config outputs.model_path).",
    )
    parser.add_argument(
        "--best-config",
        type=Path,
        default=None,
        help="Write best hyperparameters YAML (default: from tune config outputs.best_config_path).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (else MLFLOW_TRACKING_URI env or file:./mlruns).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    train_p = (args.train or (root / "data" / "processed" / "train.parquet")).expanduser().resolve()
    val_p = (args.validation or (root / "data" / "processed" / "validation.parquet")).expanduser().resolve()
    feat_c = (args.features_config or (root / "configs" / "features.yaml")).expanduser().resolve()
    tune_c = (args.tune_config or (root / "configs" / "tune_lightgbm.yaml")).expanduser().resolve()

    if not train_p.is_file():
        print(f"error: train parquet not found: {train_p}", file=sys.stderr)
        return 2
    if not val_p.is_file():
        print(f"error: validation parquet not found: {val_p}", file=sys.stderr)
        return 2
    if not feat_c.is_file():
        print(f"error: features config not found: {feat_c}", file=sys.stderr)
        return 2
    if not tune_c.is_file():
        print(f"error: tune config not found: {tune_c}", file=sys.stderr)
        return 2

    try:
        tune_cfg = load_tune_lightgbm_config(tune_c)
        out_p = args.output
        if out_p is None:
            out_p = (root / tune_cfg["outputs"]["model_path"]).expanduser().resolve()
        else:
            out_p = out_p.expanduser().resolve()
        best_c = args.best_config
        if best_c is None:
            best_c = (root / tune_cfg["outputs"]["best_config_path"]).expanduser().resolve()
        else:
            best_c = best_c.expanduser().resolve()

        train_df = pd.read_parquet(train_p)
        val_df = pd.read_parquet(val_p)
        feat_cfg = load_features_config(feat_c)
        tracking = args.tracking_uri or default_tracking_uri()
        _, metrics, _ = train_lightgbm_tuned(
            train_df,
            val_df,
            feat_cfg,
            tune_cfg,
            tune_cfg_path=tune_c,
            output_path=out_p,
            best_config_path=best_c,
            tracking_uri=tracking,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved tuned pipeline: {out_p}")
    print(f"Best hyperparameters: {best_c}")
    print(f"MLflow tracking URI: {tracking}")
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.4f}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
