"""Phase 1 — metrics contract matches Telco churn semantics."""

from pathlib import Path

from churn_ml.metrics import (
    load_metrics_contract,
    negative_class_label,
    positive_class_label,
    primary_metrics,
    target_column,
)


def test_metrics_yaml_file_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "metrics.yaml").is_file()


def test_telco_churn_semantics():
    assert target_column() == "Churn"
    assert positive_class_label() == "Yes"
    assert negative_class_label() == "No"
    assert "roc_auc" in primary_metrics()
    assert "recall_churn" in primary_metrics()


def test_contract_loads():
    c = load_metrics_contract()
    assert c["task"] == "binary_classification"
    assert "threshold_policy" in c
