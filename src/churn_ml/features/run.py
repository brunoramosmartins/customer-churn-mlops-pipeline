"""CLI: fit preprocessing Pipeline on train.parquet, save joblib + manifest (Phase 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from churn_ml.features.pipeline import (
    build_manifest,
    fit_feature_pipeline,
    load_features_config,
    save_artifacts,
)
from churn_ml.metrics import target_column


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit sklearn preprocessing pipeline on train split only (Phase 5).",
    )
    parser.add_argument(
        "--train",
        "-t",
        type=Path,
        default=None,
        help="Training Parquet from Phase 4 (default: data/processed/train.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory for feature_pipeline.joblib and feature_manifest.json (default: models/).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="features YAML (default: configs/features.yaml).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    train_path = (args.train or (root / "data" / "processed" / "train.parquet")).expanduser().resolve()
    out_dir = (args.output_dir or (root / "models")).expanduser().resolve()
    cfg_path = (args.config or (root / "configs" / "features.yaml")).expanduser().resolve()

    if not train_path.is_file():
        print(
            f"error: train parquet not found: {train_path}\n"
            "  Run Phase 4 split first: python -m churn_ml.data.split",
            file=sys.stderr,
        )
        return 2
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 2

    tgt = target_column()
    df = pd.read_parquet(train_path)
    if tgt not in df.columns:
        print(f"error: target column {tgt!r} missing from training data", file=sys.stderr)
        return 1

    try:
        cfg = load_features_config(cfg_path)
        fitted = fit_feature_pipeline(df, cfg)
        manifest = build_manifest(
            fitted,
            df,
            cfg,
            train_path=str(train_path),
        )
        pipe_path = out_dir / "feature_pipeline.joblib"
        man_path = out_dir / "feature_manifest.json"
        save_artifacts(fitted, manifest, pipeline_path=pipe_path, manifest_path=man_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Feature pipeline (fit on train only):")
    print(f"  pipeline: {pipe_path}")
    print(f"  manifest: {man_path}")
    print(f"  n_features_out: {manifest['n_features_out']}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
