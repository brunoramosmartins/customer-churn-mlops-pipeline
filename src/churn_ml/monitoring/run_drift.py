"""CLI: compare reference vs current table and write HTML + JSON drift summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from churn_ml.monitoring.drift import drift_from_paths, write_drift_artifacts


def load_drift_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("drift config must be a YAML mapping")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Univariate drift report (KS + chi-square).")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML with features_config, reference, current, output paths (optional).",
    )
    parser.add_argument("--reference", type=Path, default=None, help="Reference Parquet/CSV.")
    parser.add_argument("--current", type=Path, default=None, help="Current Parquet/CSV.")
    parser.add_argument("--features-config", type=Path, default=None, help="features.yaml path.")
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="p-value threshold for drift flag.")
    args = parser.parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_drift_config(args.config)

    def pick(key: str, cli_val: Any, default: Any = None) -> Any:
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    root = Path(__file__).resolve().parents[3]
    ref = pick("reference", args.reference, root / "data" / "processed" / "train.parquet")
    cur = pick("current", args.current, root / "data" / "processed" / "test.parquet")
    feat = pick("features_config", args.features_config, root / "configs" / "features.yaml")
    out_html = pick("output_html", args.output_html, root / "reports" / "drift_report.html")
    out_json = pick("output_json", args.output_json, root / "reports" / "drift_summary.json")
    alpha = float(pick("alpha", args.alpha, 0.05))

    ref_p = Path(ref).expanduser().resolve()
    cur_p = Path(cur).expanduser().resolve()
    feat_p = Path(feat).expanduser().resolve()

    rows = drift_from_paths(ref_p, cur_p, feat_p, alpha=alpha)
    write_drift_artifacts(
        rows,
        html_path=Path(out_html).expanduser().resolve(),
        json_path=Path(out_json).expanduser().resolve(),
        reference_path=str(ref_p),
        current_path=str(cur_p),
        alpha=alpha,
    )
    print(f"Wrote {out_html}")
    print(f"Wrote {out_json}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
