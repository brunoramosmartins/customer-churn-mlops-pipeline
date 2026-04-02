"""Phase 3 — EDA artifacts from fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from churn_ml.data.validate import load_raw_csv
from churn_ml.eda.summary import build_eda_summary, run_eda_pipeline, write_eda_artifacts
from churn_ml.metrics import positive_class_label

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"


def test_build_eda_summary_structure():
    df = load_raw_csv(FIXTURE)
    s = build_eda_summary(df)
    assert s["n_rows"] == 4
    assert positive_class_label() in s["churn_value_counts"]


def test_write_eda_artifacts(tmp_path: Path):
    df = load_raw_csv(FIXTURE)
    paths = write_eda_artifacts(df, tmp_path)
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert paths["churn_figure"].is_file()
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["n_rows"] == 4
    assert "categorical_cardinality" in data


def test_run_eda_pipeline_validates_and_writes(tmp_path: Path):
    paths = run_eda_pipeline(FIXTURE, tmp_path, skip_validation=False)
    assert paths["markdown"].read_text(encoding="utf-8").startswith("# EDA summary")


def test_eda_cli_module(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_ml.eda.run",
            "-i",
            str(FIXTURE),
            "-o",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "eda_summary.json").is_file()
