"""Phase 4 — stratified split, no leakage across sets, manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from churn_ml.data.split import (
    load_split_config,
    prepare_for_modeling,
    run_split,
    stratified_train_val_test,
)
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.metrics import positive_class_label, target_column

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"
REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "split.yaml"


def expanded_telco(n_copies: int = 30) -> "object":
    """Repeat fixture rows with unique customerID for stable stratified splits."""
    import pandas as pd

    base = load_raw_csv(FIXTURE)
    parts = []
    for i in range(n_copies):
        b = base.copy()
        b["customerID"] = b["customerID"].astype(str) + f"_b{i}"
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def test_load_split_config():
    cfg = load_split_config(REPO_CONFIG)
    assert cfg["train_ratio"] + cfg["val_ratio"] + cfg["test_ratio"] == pytest.approx(1.0)
    assert cfg["random_state"] == 42


def test_load_split_config_ratios_must_sum_one(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump({"random_state": 1, "train_ratio": 0.5, "val_ratio": 0.5, "test_ratio": 0.1}))
    with pytest.raises(ValueError, match="sum to 1"):
        load_split_config(p)


def test_stratified_split_no_overlap_and_stratify():
    df = validate_raw_dataframe(prepare_for_modeling(expanded_telco(40)))
    col = target_column()
    cfg = load_split_config(REPO_CONFIG)
    train, val, test = stratified_train_val_test(
        df,
        target_col=col,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        random_state=cfg["random_state"],
    )
    ids_train = set(train["customerID"])
    ids_val = set(val["customerID"])
    ids_test = set(test["customerID"])
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)

    global_rate = (df[col] == positive_class_label()).mean()
    for part in (train, val, test):
        r = (part[col] == positive_class_label()).mean()
        assert abs(r - global_rate) < 0.12


def test_prepare_senior_citizen_int():
    import pandas as pd

    df = pd.DataFrame({"SeniorCitizen": ["0", "1", None], "x": [1, 2, 3]})
    out = prepare_for_modeling(df)
    assert out["SeniorCitizen"].dtype == "int64"
    assert list(out["SeniorCitizen"]) == [0, 1, 0]


def test_run_split_writes_parquet_and_manifest(tmp_path: Path):
    df = validate_raw_dataframe(prepare_for_modeling(expanded_telco(35)))
    cfg = load_split_config(REPO_CONFIG)
    paths = run_split(df, tmp_path, cfg, input_path="/tmp/mock.csv")
    assert paths["train"].is_file()
    assert paths["validation"].is_file()
    assert paths["test"].is_file()
    assert paths["manifest"].is_file()
    man = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert man["n_train"] + man["n_val"] + man["n_test"] == man["n_total"]


def test_split_cli_subprocess(tmp_path: Path):
    """CLI end-to-end on a CSV large enough for stratified 70/15/15."""
    big = tmp_path / "telco_expanded.csv"
    expanded_telco(35).to_csv(big, index=False)
    out = tmp_path / "processed"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_ml.data.split",
            "-i",
            str(big),
            "-o",
            str(out),
            "-c",
            str(REPO_CONFIG),
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "train.parquet").is_file()
    assert (out / "split_manifest.json").is_file()
