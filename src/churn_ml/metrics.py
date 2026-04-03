"""Problem and metrics contract (Phase 1). Evaluation modules should use this, not hard-coded strings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_metrics_contract() -> dict[str, Any]:
    path = _repo_root() / "configs" / "metrics.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("metrics.yaml must define a mapping at the top level")
    return data


def target_column() -> str:
    return str(load_metrics_contract()["target_column"])


def positive_class_label() -> str:
    return str(load_metrics_contract()["positive_class_label"])


def negative_class_label() -> str:
    return str(load_metrics_contract()["negative_class_label"])


def primary_metrics() -> tuple[str, ...]:
    raw = load_metrics_contract()["primary_metrics"]
    return tuple(str(m) for m in raw)


def suggested_minimum_recall_churn() -> float:
    """Minimum recall on churn (positive class) used as a floor when scanning thresholds (Phase 8)."""
    pol = load_metrics_contract().get("threshold_policy") or {}
    return float(pol.get("suggested_minimum_recall_churn", 0.5))
