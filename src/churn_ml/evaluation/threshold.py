"""Scan probability thresholds on validation under a recall floor; maximize F-beta."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import fbeta_score, recall_score


def select_threshold(
    y_true: np.ndarray,
    proba_positive: np.ndarray,
    *,
    min_recall_churn: float,
    fbeta_beta: float,
    n_grid: int = 99,
) -> tuple[float, dict[str, Any]]:
    """
    Choose threshold maximizing F-beta among grid points with recall_churn >= floor.
    If none qualify, pick threshold with highest recall (constraint relaxed) and set warning.
    """
    y_true = np.asarray(y_true).astype(int)
    proba_positive = np.asarray(proba_positive, dtype=float)
    thresholds = np.linspace(0.01, 0.99, int(n_grid))
    best_t: float | None = None
    best_fbeta = -1.0
    best_recall = -1.0
    feasible: list[tuple[float, float, float]] = []

    for t in thresholds:
        pred = (proba_positive >= t).astype(int)
        rec = float(recall_score(y_true, pred, zero_division=0))
        fbeta = float(fbeta_score(y_true, pred, beta=fbeta_beta, zero_division=0))
        if rec >= min_recall_churn:
            feasible.append((t, rec, fbeta))
            if fbeta > best_fbeta or (fbeta == best_fbeta and rec > best_recall):
                best_fbeta = fbeta
                best_recall = rec
                best_t = float(t)

    warning: str | None = None
    if best_t is None:
        warning = (
            f"No threshold in grid achieved recall_churn >= {min_recall_churn}; "
            "selected threshold with maximum recall on validation instead."
        )
        for t in thresholds:
            pred = (proba_positive >= t).astype(int)
            rec = float(recall_score(y_true, pred, zero_division=0))
            fbeta = float(fbeta_score(y_true, pred, beta=fbeta_beta, zero_division=0))
            if rec > best_recall or (rec == best_recall and fbeta > best_fbeta):
                best_recall = rec
                best_fbeta = fbeta
                best_t = float(t)

    meta = {
        "min_recall_churn_floor": min_recall_churn,
        "fbeta_beta": fbeta_beta,
        "n_feasible_thresholds": len(feasible),
        "constraint_satisfied": len(feasible) > 0,
        "warning": warning,
    }
    assert best_t is not None
    return best_t, meta
