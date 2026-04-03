"""Orchestrate Phase 8: load champion, pick threshold on val, score test, write reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn_ml.evaluation.plots import (
    save_confusion_matrix_figure,
    save_pr_figure,
    save_roc_figure,
    save_threshold_sweep_figure,
)
from churn_ml.evaluation.threshold import select_threshold
from churn_ml.features.pipeline import load_features_config, select_feature_matrix
from churn_ml.metrics import (
    negative_class_label,
    positive_class_label,
    suggested_minimum_recall_churn,
    target_column,
)
from churn_ml.models.baseline import y_positive_binary


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_eval_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("eval config must be a YAML mapping")
    for key in ("champion_model_path", "fallback_model_path", "threshold_search", "outputs"):
        if key not in cfg:
            raise ValueError(f"eval config missing required key: {key}")
    return cfg


def _resolve_output_path(root: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (root / pp).resolve()


def resolve_champion_path(root: Path, cfg: dict[str, Any]) -> tuple[Path, str]:
    """Return (resolved_path, label) where label describes which path was used."""
    primary = (root / cfg["champion_model_path"]).resolve()
    if primary.is_file():
        return primary, "champion_model_path"
    fb = (root / cfg["fallback_model_path"]).resolve()
    if fb.is_file():
        return fb, "fallback_model_path"
    raise FileNotFoundError(
        f"Champion model not found at {primary} or fallback {fb}. Train Phase 6/7 first."
    )


def _ranking_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
    }


def _threshold_metrics_with_beta(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    fbeta_beta: float,
) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "recall_churn": float(recall_score(y, pred, zero_division=0)),
        "precision_churn": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fbeta": float(fbeta_score(y, pred, beta=fbeta_beta, zero_division=0)),
    }


def run_evaluation(
    root: Path,
    eval_cfg: dict[str, Any],
    *,
    features_config_path: Path,
    validation_parquet: Path,
    test_parquet: Path,
    champion_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    feat_cfg = load_features_config(Path(features_config_path).resolve())

    val_df = pd.read_parquet(Path(validation_parquet).resolve())
    test_df = pd.read_parquet(Path(test_parquet).resolve())

    if target_column() not in val_df.columns or target_column() not in test_df.columns:
        raise ValueError(f"expected target column {target_column()!r} in val and test")

    X_val = select_feature_matrix(val_df, feat_cfg)
    y_val = y_positive_binary(val_df).to_numpy()
    X_test = select_feature_matrix(test_df, feat_cfg)
    y_test = y_positive_binary(test_df).to_numpy()

    path, which = (
        (Path(champion_path).resolve(), "override")
        if champion_path is not None
        else resolve_champion_path(root, eval_cfg)
    )
    pipe: Pipeline = joblib.load(path)
    proba_val = pipe.predict_proba(X_val)[:, 1]
    proba_test = pipe.predict_proba(X_test)[:, 1]

    ts = eval_cfg["threshold_search"]
    min_rec_cfg = ts.get("min_recall_churn")
    min_recall = (
        float(min_rec_cfg) if min_rec_cfg is not None else suggested_minimum_recall_churn()
    )
    fbeta_beta = float(ts.get("fbeta_beta", 1.25))
    n_grid = int(ts.get("n_threshold_grid", 99))

    threshold, th_meta = select_threshold(
        y_val,
        proba_val,
        min_recall_churn=min_recall,
        fbeta_beta=fbeta_beta,
        n_grid=n_grid,
    )

    out = eval_cfg["outputs"]
    fig_dir = _resolve_output_path(root, out["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = "phase8"
    save_roc_figure(y_val, proba_val, fig_dir / f"{stem}_roc_validation.png", title_suffix="validation")
    save_pr_figure(y_val, proba_val, fig_dir / f"{stem}_pr_validation.png", title_suffix="validation")
    save_roc_figure(y_test, proba_test, fig_dir / f"{stem}_roc_test.png", title_suffix="test (one-shot)")
    save_pr_figure(y_test, proba_test, fig_dir / f"{stem}_pr_test.png", title_suffix="test (one-shot)")
    save_threshold_sweep_figure(
        y_val,
        proba_val,
        fig_dir / f"{stem}_threshold_sweep_validation.png",
        fbeta_beta=fbeta_beta,
        n_grid=n_grid,
        chosen_threshold=threshold,
    )

    pred_test = (proba_test >= threshold).astype(int)
    neg_lab = negative_class_label()
    pos_lab = positive_class_label()
    save_confusion_matrix_figure(
        y_test,
        pred_test,
        fig_dir / f"{stem}_confusion_matrix_test.png",
        title=f"Test — confusion @ threshold={threshold:.4f} (frozen from validation)",
        labels=(f"{neg_lab} (0)", f"{pos_lab} (1)"),
    )

    val_rank = _ranking_metrics(y_val, proba_val)
    test_rank = _ranking_metrics(y_test, proba_test)
    val_at_t = _threshold_metrics_with_beta(y_val, proba_val, threshold, fbeta_beta)
    test_at_t = _threshold_metrics_with_beta(y_test, proba_test, threshold, fbeta_beta)

    summary: dict[str, Any] = {
        "phase": 8,
        "champion": {
            "model_path_resolved": str(path),
            "model_path_source": which,
            "threshold": threshold,
            "threshold_selected_on": "validation",
            "threshold_search": {
                "min_recall_churn_floor": min_recall,
                "fbeta_beta": fbeta_beta,
                **{k: v for k, v in th_meta.items() if k != "warning"},
            },
        },
        "metrics_by_split": {
            "validation": {
                "description": "Ranking curves and threshold sweep; operating metrics at chosen threshold.",
                "ranking_threshold_free": val_rank,
                "at_chosen_threshold": val_at_t,
            },
            "test": {
                "description": (
                    "One-shot holdout. Ranking metrics (ROC-AUC, AP) need no threshold. "
                    "Confusion-based metrics use the threshold frozen from validation only."
                ),
                "ranking_threshold_free": test_rank,
                "at_frozen_threshold_from_validation": test_at_t,
            },
        },
        "figures": {
            "roc_validation": str(fig_dir / f"{stem}_roc_validation.png"),
            "pr_validation": str(fig_dir / f"{stem}_pr_validation.png"),
            "roc_test": str(fig_dir / f"{stem}_roc_test.png"),
            "pr_test": str(fig_dir / f"{stem}_pr_test.png"),
            "threshold_sweep_validation": str(fig_dir / f"{stem}_threshold_sweep_validation.png"),
            "confusion_matrix_test": str(fig_dir / f"{stem}_confusion_matrix_test.png"),
        },
        "notes": [
            "Calibration plots are out of scope for portfolio v1 (roadmap).",
            "Do not re-tune threshold on test; Phase 9+ should load configs/champion.yaml.",
        ],
    }
    if th_meta.get("warning"):
        summary["notes"].insert(0, th_meta["warning"])

    json_path = _resolve_output_path(root, out["summary_json"])
    md_path = _resolve_output_path(root, out["summary_md"])
    champ_path = _resolve_output_path(root, out["champion_manifest"])

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    champ_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Phase 8 — Evaluation summary",
        "",
        "## Champion",
        "",
        f"- **Model:** `{path}` (source: `{which}`)",
        f"- **Threshold (selected on validation):** `{threshold:.6f}`",
        f"- **Recall floor used:** `{min_recall}`; **F-beta (maximized under floor):** beta = `{fbeta_beta}`",
        "",
        "## Which metric belongs to which split",
        "",
        "| Metric | Split | Notes |",
        "|--------|-------|-------|",
        "| ROC-AUC, Average precision (ranking) | validation / test | Threshold-free; test reported once. |",
        "| Recall / precision / F1 / F-beta at threshold | validation | Used to *choose* threshold under recall floor. |",
        "| Same at *frozen* threshold | test | One-shot; same cutoff as validation. |",
        "| Confusion matrix figure | test | At frozen threshold. |",
        "",
        "## Key numbers",
        "",
        "### Validation (ranking)",
        "",
        f"- ROC-AUC: **{val_rank['roc_auc']:.4f}**",
        f"- Average precision: **{val_rank['average_precision']:.4f}**",
        "",
        "### Validation (at chosen threshold)",
        "",
        f"- Recall (churn): **{val_at_t['recall_churn']:.4f}**",
        f"- Precision (churn): **{val_at_t['precision_churn']:.4f}**",
        f"- F1: **{val_at_t['f1']:.4f}**",
        "",
        "### Test — one shot (ranking)",
        "",
        f"- ROC-AUC: **{test_rank['roc_auc']:.4f}**",
        f"- Average precision: **{test_rank['average_precision']:.4f}**",
        "",
        "### Test — frozen threshold from validation",
        "",
        f"- Recall (churn): **{test_at_t['recall_churn']:.4f}**",
        f"- Precision (churn): **{test_at_t['precision_churn']:.4f}**",
        f"- F1: **{test_at_t['f1']:.4f}**",
        "",
        "## Figures",
        "",
    ]
    for k, p in summary["figures"].items():
        md_lines.append(f"- `{k}` → `{p}`")
    md_lines.extend(["", "## Notes", ""])
    for n in summary["notes"]:
        md_lines.append(f"- {n}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    try:
        model_rel = str(path.relative_to(root))
    except ValueError:
        model_rel = str(path)
    champ_doc = {
        "_phase": 8,
        "model_path": model_rel,
        "model_path_resolved": str(path),
        "threshold": float(threshold),
        "threshold_selected_on": "validation",
        "min_recall_churn_floor": min_recall,
        "fbeta_beta": fbeta_beta,
        "constraint_satisfied": th_meta["constraint_satisfied"],
        "test_ranking_metrics": test_rank,
        "test_at_frozen_threshold": test_at_t,
    }
    with champ_path.open("w", encoding="utf-8") as f:
        f.write("# Frozen champion for packaging / batch inference (Phase 9+).\n")
        yaml.safe_dump(champ_doc, f, sort_keys=False, default_flow_style=False)

    return summary


def run_evaluation_from_cli(
    root: Path | None = None,
    *,
    eval_config: Path | None = None,
    features_config: Path | None = None,
    validation: Path | None = None,
    test: Path | None = None,
    champion: Path | None = None,
) -> dict[str, Any]:
    root = root or _repo_root()
    eval_c = eval_config or (root / "configs" / "eval.yaml")
    feat_c = features_config or (root / "configs" / "features.yaml")
    val_p = validation or (root / "data" / "processed" / "validation.parquet")
    test_p = test or (root / "data" / "processed" / "test.parquet")
    cfg = load_eval_config(eval_c)
    return run_evaluation(
        root,
        cfg,
        features_config_path=feat_c,
        validation_parquet=val_p,
        test_parquet=test_p,
        champion_path=champion,
    )
