"""Validate raw Telco CSV against the Pandera schema (Phase 2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors

from churn_ml.data.schema import EXPECTED_RAW_FILENAME, telco_raw_schema


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalize_telco_raw_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes after `read_csv` so validation matches the real IBM/Kaggle file.

    `TotalCharges` often loads as `object` because some rows use empty strings for
    new customers; `pd.to_numeric(..., errors='coerce')` turns those into NaN and
    the rest into float64 (what the Pandera schema expects).
    """
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_telco_raw_dtypes(df)


def validate_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return validated copy; raises `pandera.errors.SchemaError` on failure."""
    df = normalize_telco_raw_dtypes(df)
    return telco_raw_schema().validate(df)


def validate_raw_csv(path: Path) -> pd.DataFrame:
    df = load_raw_csv(path)
    return validate_raw_dataframe(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Telco Customer Churn raw CSV (Pandera). Exits 0 if valid, 1 on schema failure, 2 on I/O or missing file.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=f"Path to raw CSV. If omitted, uses data/raw/{EXPECTED_RAW_FILENAME} when that file exists.",
    )
    args = parser.parse_args(argv)

    if args.csv_path:
        path = Path(args.csv_path).expanduser().resolve()
    else:
        default = _repo_root() / "data" / "raw" / EXPECTED_RAW_FILENAME
        if not default.is_file():
            print(
                f"error: no path given and default file missing: {default}\n"
                f"       download the dataset (see data/raw/README.md) or pass a path explicitly.",
                file=sys.stderr,
            )
            return 2
        path = default

    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    try:
        validate_raw_csv(path)
    except pa_errors.SchemaError as exc:
        print(f"validation failed: {path}", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error reading {path}: {exc}", file=sys.stderr)
        return 2

    print(f"ok: {path} matches Telco raw schema ({path.stat().st_size} bytes)")
    return 0


def cli() -> None:
    """Console entrypoint for setuptools (`churn-validate`)."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
