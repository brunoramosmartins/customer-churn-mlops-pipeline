"""Stratified train/val/test split and Parquet export (Phase 4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from churn_ml.data.schema import EXPECTED_RAW_FILENAME
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.metrics import positive_class_label, target_column


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_split_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("split config must be a YAML mapping")
    tr = float(cfg["train_ratio"])
    vr = float(cfg["val_ratio"])
    te = float(cfg["test_ratio"])
    s = tr + vr + te
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {s}")
    cfg["train_ratio"] = tr
    cfg["val_ratio"] = vr
    cfg["test_ratio"] = te
    cfg["random_state"] = int(cfg["random_state"])
    return cfg


def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Align dtypes before split/save: TotalCharges numeric (from loader), SeniorCitizen int."""
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = (
            pd.to_numeric(out["SeniorCitizen"], errors="coerce").fillna(0).astype("int64")
        )
    return out


def stratified_train_val_test(
    df: pd.DataFrame,
    *,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Disjoint train / val / test with stratification on `target_col`."""
    y = df[target_col]
    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )
    val_in_tv = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_in_tv,
        random_state=random_state,
        stratify=train_val[target_col],
        shuffle=True,
    )
    return train, val, test


def _churn_rate(df: pd.DataFrame, col: str) -> float:
    if len(df) == 0:
        return 0.0
    return float((df[col] == positive_class_label()).mean())


def build_split_manifest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    input_path: str | None = None,
) -> dict[str, Any]:
    col = target_column()
    n = len(train) + len(val) + len(test)
    return {
        "random_state": cfg["random_state"],
        "train_ratio": cfg["train_ratio"],
        "val_ratio": cfg["val_ratio"],
        "test_ratio": cfg["test_ratio"],
        "input_csv": input_path,
        "target_column": col,
        "positive_class": positive_class_label(),
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "churn_rate_train": round(_churn_rate(train, col), 6),
        "churn_rate_val": round(_churn_rate(val, col), 6),
        "churn_rate_test": round(_churn_rate(test, col), 6),
        "output_files": ["train.parquet", "validation.parquet", "test.parquet", "split_manifest.json"],
    }


def run_split(
    df: pd.DataFrame,
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    input_path: str | None = None,
) -> dict[str, Path]:
    """Split `df`, write Parquet + manifest; returns paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    col = target_column()
    train, val, test = stratified_train_val_test(
        df,
        target_col=col,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        random_state=cfg["random_state"],
    )
    paths = {
        "train": output_dir / "train.parquet",
        "validation": output_dir / "validation.parquet",
        "test": output_dir / "test.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["validation"], index=False)
    test.to_parquet(paths["test"], index=False)
    manifest = build_split_manifest(train, val, test, cfg, input_path=input_path)
    man_path = output_dir / "split_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = man_path
    return paths


def run_split_from_raw(
    input_csv: Path,
    output_dir: Path,
    config_path: Path,
    *,
    skip_validation: bool = False,
) -> dict[str, Path]:
    cfg = load_split_config(config_path)
    df = load_raw_csv(Path(input_csv))
    df = prepare_for_modeling(df)
    if not skip_validation:
        df = validate_raw_dataframe(df)
    return run_split(df, output_dir, cfg, input_path=str(Path(input_csv).resolve()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split → Parquet under data/processed (Phase 4).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help=f"Raw CSV (default: data/raw/{EXPECTED_RAW_FILENAME} if present).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory for Parquet files (default: data/processed).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="YAML split config (default: configs/split.yaml).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Pandera validation (not recommended for real raw files).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    inp = args.input
    if inp is None:
        default_in = root / "data" / "raw" / EXPECTED_RAW_FILENAME
        if not default_in.is_file():
            print(
                f"error: no --input and default missing: {default_in}",
                file=sys.stderr,
            )
            return 2
        inp = default_in
    else:
        inp = inp.expanduser().resolve()
        if not inp.is_file():
            print(f"error: file not found: {inp}", file=sys.stderr)
            return 2

    out_dir = (args.output_dir or (root / "data" / "processed")).expanduser().resolve()
    cfg_path = (args.config or (root / "configs" / "split.yaml")).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 2

    try:
        paths = run_split_from_raw(inp, out_dir, cfg_path, skip_validation=args.skip_validation)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Split written:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    man = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    print(
        f"Churn rates — train: {man['churn_rate_train']:.4f}, "
        f"val: {man['churn_rate_val']:.4f}, test: {man['churn_rate_test']:.4f}"
    )
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
