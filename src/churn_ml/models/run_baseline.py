"""CLI: train logistic baseline, log MLflow, save models/baseline.joblib (Phase 6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from churn_ml.features.pipeline import load_features_config
from churn_ml.models.baseline import default_tracking_uri, load_train_baseline_config, train_baseline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train logistic regression baseline with MLflow logging (Phase 6).",
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
        "--train-config",
        type=Path,
        default=None,
        help="train_baseline.yaml (default: configs/train_baseline.yaml).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Fitted pipeline joblib (default: models/baseline.joblib).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Override MLflow tracking URI (else MLFLOW_TRACKING_URI env or file:./mlruns).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    train_p = (args.train or (root / "data" / "processed" / "train.parquet")).expanduser().resolve()
    val_p = (args.validation or (root / "data" / "processed" / "validation.parquet")).expanduser().resolve()
    feat_c = (args.features_config or (root / "configs" / "features.yaml")).expanduser().resolve()
    trn_c = (args.train_config or (root / "configs" / "train_baseline.yaml")).expanduser().resolve()
    out_p = (args.output or (root / "models" / "baseline.joblib")).expanduser().resolve()

    if not train_p.is_file():
        print(
            f"error: train parquet not found: {train_p}\n"
            "  Run: python -m churn_ml.data.split",
            file=sys.stderr,
        )
        return 2
    if not val_p.is_file():
        print(
            f"error: validation parquet not found: {val_p}\n"
            "  Run: python -m churn_ml.data.split",
            file=sys.stderr,
        )
        return 2
    if not feat_c.is_file():
        print(f"error: features config not found: {feat_c}", file=sys.stderr)
        return 2
    if not trn_c.is_file():
        print(f"error: train config not found: {trn_c}", file=sys.stderr)
        return 2

    tracking = args.tracking_uri or default_tracking_uri()

    try:
        train_df = pd.read_parquet(train_p)
        val_df = pd.read_parquet(val_p)
        feat_cfg = load_features_config(feat_c)
        train_cfg = load_train_baseline_config(trn_c)
        _, metrics = train_baseline(
            train_df,
            val_df,
            feat_cfg,
            train_cfg,
            output_path=out_p,
            tracking_uri=tracking,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved pipeline: {out_p}")
    print(f"MLflow tracking URI: {tracking}")
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.4f}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
