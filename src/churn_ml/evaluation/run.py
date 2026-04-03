"""CLI: Phase 8 evaluation — threshold on validation; test metrics; figures + champion manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn_ml.evaluation.evaluate import load_eval_config, run_evaluation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8: choose threshold on validation; one-shot test metrics; plots and champion.yaml.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="eval.yaml (default: configs/eval.yaml).",
    )
    parser.add_argument(
        "--features-config",
        type=Path,
        default=None,
        help="features.yaml (default: configs/features.yaml).",
    )
    parser.add_argument(
        "--validation",
        "-v",
        type=Path,
        default=None,
        help="Validation Parquet (default: data/processed/validation.parquet).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Test Parquet (default: data/processed/test.parquet).",
    )
    parser.add_argument(
        "--champion",
        type=Path,
        default=None,
        help="Override champion joblib path (default: eval.yaml primary/fallback under repo root).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root for resolving relative paths in eval.yaml (default: repository root).",
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve() if args.root else _repo_root()
    eval_c = (args.eval_config or (root / "configs" / "eval.yaml")).expanduser().resolve()
    feat_c = (args.features_config or (root / "configs" / "features.yaml")).expanduser().resolve()
    val_p = (args.validation or (root / "data" / "processed" / "validation.parquet")).expanduser().resolve()
    test_p = (args.test or (root / "data" / "processed" / "test.parquet")).expanduser().resolve()
    champ = args.champion.expanduser().resolve() if args.champion else None

    for path, label in [
        (eval_c, "eval config"),
        (feat_c, "features config"),
        (val_p, "validation parquet"),
        (test_p, "test parquet"),
    ]:
        if not path.is_file():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2

    try:
        cfg = load_eval_config(eval_c)
        summary = run_evaluation(
            root,
            cfg,
            features_config_path=feat_c,
            validation_parquet=val_p,
            test_parquet=test_p,
            champion_path=champ,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = summary["figures"]
    print("Phase 8 evaluation written:")
    print("  model:", summary["champion"]["model_path_resolved"])
    print("  threshold:", summary["champion"]["threshold"])
    for k in sorted(out):
        print(f"  figure {k}: {out[k]}")
    print("  summary json:", (root / cfg["outputs"]["summary_json"]).resolve())
    print("  summary md:  ", (root / cfg["outputs"]["summary_md"]).resolve())
    print("  champion:    ", (root / cfg["outputs"]["champion_manifest"]).resolve())
    print(
        "  test ROC-AUC:",
        summary["metrics_by_split"]["test"]["ranking_threshold_free"]["roc_auc"],
    )
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
