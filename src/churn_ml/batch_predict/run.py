"""CLI: batch predictions from champion joblib + Pydantic-validated rows (Phase 9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn_ml.batch_predict.predict import (
    _repo_root,
    batch_predict,
    load_batch_predict_config,
)


def _resolve_under_root(root: Path, p: str | Path) -> Path:
    pp = Path(p).expanduser()
    return pp.resolve() if pp.is_absolute() else (root / pp).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch churn inference: Pydantic row validation, champion pipeline, frozen threshold (Phase 9).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input CSV or Parquet (Telco columns; Churn optional).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV or Parquet (default from configs/batch_predict.yaml).",
    )
    parser.add_argument(
        "--batch-config",
        type=Path,
        default=None,
        help="batch_predict.yaml (default: configs/batch_predict.yaml).",
    )
    parser.add_argument(
        "--champion-manifest",
        type=Path,
        default=None,
        help="champion.yaml (default: from batch config).",
    )
    parser.add_argument(
        "--features-config",
        type=Path,
        default=None,
        help="features.yaml (default: from batch config).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root for relative paths (default: auto-detect).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override probability threshold from champion manifest.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing batch_predict_metadata.json.",
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve() if args.root else _repo_root()
    batch_c = (args.batch_config or (root / "configs" / "batch_predict.yaml")).expanduser().resolve()
    if not batch_c.is_file():
        print(f"error: batch config not found: {batch_c}", file=sys.stderr)
        return 2

    cfg = load_batch_predict_config(batch_c)
    champ_arg = args.champion_manifest
    champ = _resolve_under_root(root, champ_arg) if champ_arg else _resolve_under_root(root, cfg["champion_manifest"])
    feat_arg = args.features_config
    feat = _resolve_under_root(root, feat_arg) if feat_arg else _resolve_under_root(root, cfg["features_config"])
    out_p = args.output
    if out_p is None:
        out_p = _resolve_under_root(root, cfg["default_output"])
    else:
        out_p = Path(out_p).expanduser().resolve()

    if not champ.is_file():
        print(f"error: champion manifest not found: {champ}", file=sys.stderr)
        return 2
    if not feat.is_file():
        print(f"error: features config not found: {feat}", file=sys.stderr)
        return 2

    try:
        meta = batch_predict(
            root,
            input_path=args.input.expanduser().resolve(),
            output_path=out_p,
            batch_cfg=cfg,
            champion_manifest_path=champ,
            features_config_path=feat,
            threshold_override=args.threshold,
            write_metadata=not args.no_metadata,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {meta['n_rows']} rows -> {out_p}")
    if "metadata_written" in meta:
        print(f"Metadata: {meta['metadata_written']}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
