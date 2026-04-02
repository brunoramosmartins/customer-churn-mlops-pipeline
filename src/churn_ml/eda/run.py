"""CLI: generate EDA reports under `reports/`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn_ml.data.schema import EXPECTED_RAW_FILENAME
from churn_ml.eda.summary import run_eda_pipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_input_path(arg: Path | None) -> Path:
    if arg is not None:
        p = arg.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    default = _repo_root() / "data" / "raw" / EXPECTED_RAW_FILENAME
    if default.is_file():
        return default
    raise FileNotFoundError(
        f"No --input given and default raw file missing: {default}\n"
        f"  Pass --input path/to.csv or place {EXPECTED_RAW_FILENAME} under data/raw/."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lean EDA for Telco churn: JSON + Markdown + figures under reports/.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help=f"Raw CSV (default: data/raw/{EXPECTED_RAW_FILENAME} if present).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: reports/).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Pandera schema check (not recommended for production raw files).",
    )
    args = parser.parse_args(argv)

    out = (args.output or (_repo_root() / "reports")).expanduser().resolve()

    try:
        inp = resolve_input_path(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        paths = run_eda_pipeline(inp, out, skip_validation=args.skip_validation)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("EDA written:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
