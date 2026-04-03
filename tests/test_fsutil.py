"""churn_ml.fsutil — portable repo-relative paths."""

from __future__ import annotations

from pathlib import Path

import re

import churn_ml
from churn_ml.fsutil import path_relative_to_repo


def test_churn_ml_version_matches_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", churn_ml.__version__)


def test_path_relative_to_repo_under_root(tmp_path: Path):
    root = tmp_path
    f = tmp_path / "models" / "m.joblib"
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")
    assert path_relative_to_repo(root, f) == "models/m.joblib"


def test_path_relative_to_repo_outside_root(tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    f = other / "x.txt"
    f.write_text("y", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    rel = path_relative_to_repo(root, f)
    assert rel == str(f.resolve())
