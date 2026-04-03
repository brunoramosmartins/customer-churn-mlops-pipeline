"""Save ROC, PR, confusion matrix, and threshold-sweep figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def save_roc_figure(
    y_true: np.ndarray,
    proba: np.ndarray,
    out_path: Path,
    *,
    title_suffix: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax, name="model")
    auc = roc_auc_score(y_true, proba)
    ax.set_title(f"ROC — {title_suffix} (AUC = {auc:.4f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_pr_figure(
    y_true: np.ndarray,
    proba: np.ndarray,
    out_path: Path,
    *,
    title_suffix: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax, name="model")
    ap = average_precision_score(y_true, proba)
    ax.set_title(f"Precision–Recall — {title_suffix} (AP = {ap:.4f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    title: str,
    labels: tuple[str, str],
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_threshold_sweep_figure(
    y_true: np.ndarray,
    proba: np.ndarray,
    out_path: Path,
    *,
    fbeta_beta: float,
    n_grid: int,
    chosen_threshold: float,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.01, 0.99, int(n_grid))
    recalls, precs, f1s, fbetas = [], [], [], []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        recalls.append(recall_score(y_true, pred, zero_division=0))
        precs.append(precision_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))
        fbetas.append(fbeta_score(y_true, pred, beta=fbeta_beta, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, recalls, label="Recall (churn)")
    ax.plot(thresholds, precs, label="Precision")
    ax.plot(thresholds, f1s, label="F1")
    ax.plot(thresholds, fbetas, label=f"F-beta (beta={fbeta_beta})")
    ax.axvline(chosen_threshold, color="black", linestyle="--", label="Chosen threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Validation — metrics vs threshold (selection split)")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
