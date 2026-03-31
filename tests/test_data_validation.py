"""Phase 2 — Pandera validation on CI fixture and failure cases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from churn_ml.data.schema import telco_raw_schema
from churn_ml.data.validate import validate_raw_csv, validate_raw_dataframe

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"


def test_fixture_csv_validates():
    out = validate_raw_csv(FIXTURE)
    assert len(out) == 4
    assert out["Churn"].isin(["Yes", "No"]).all()


def test_validate_rejects_extra_column():
    df = pd.read_csv(FIXTURE)
    df["extra"] = 0
    with pytest.raises(SchemaError):
        validate_raw_dataframe(df)


def test_validate_rejects_bad_churn():
    df = pd.read_csv(FIXTURE)
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(SchemaError):
        validate_raw_dataframe(df)


def test_cli_module_ok_on_fixture():
    proc = subprocess.run(
        [sys.executable, "-m", "churn_ml.data.validate", str(FIXTURE)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok:" in proc.stdout


def test_schema_strict_columns():
    schema = telco_raw_schema()
    assert schema.strict is True
    assert "customerID" in schema.columns
    assert "Churn" in schema.columns


def test_totalcharges_object_strings_coerced(tmp_path: Path) -> None:
    """Real Kaggle/IBM CSV often has TotalCharges as object; loader must coerce."""
    df = pd.read_csv(FIXTURE)
    df["TotalCharges"] = ["29.85", "", "108.15", "151.65"]
    path = tmp_path / "telco_object_total.csv"
    df.to_csv(path, index=False)
    out = validate_raw_csv(path)
    assert out["TotalCharges"].dtype == "float64"


def test_validate_raw_dataframe_coerces_totalcharges_object() -> None:
    df = pd.read_csv(FIXTURE)
    df["TotalCharges"] = df["TotalCharges"].astype(str).replace("nan", "")
    validate_raw_dataframe(df)
