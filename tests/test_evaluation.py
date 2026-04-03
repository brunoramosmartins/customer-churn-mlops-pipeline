"""Phase 8 — threshold on validation; test metrics; figures and champion manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import yaml

from churn_ml.data.split import load_split_config, run_split
from churn_ml.data.validate import load_raw_csv, validate_raw_dataframe
from churn_ml.evaluation.evaluate import load_eval_config, run_evaluation
from churn_ml.evaluation.threshold import select_threshold
from churn_ml.evaluation.run import main as eval_main
from churn_ml.features.pipeline import load_features_config, select_feature_matrix
from churn_ml.metrics import suggested_minimum_recall_churn
from churn_ml.models.baseline import (
    build_baseline_pipeline,
    load_train_baseline_config,
    y_positive_binary,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telco_sample.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SPLIT = REPO_ROOT / "configs" / "split.yaml"
REPO_FEATURES = REPO_ROOT / "configs" / "features.yaml"
REPO_TRAIN_BASELINE = REPO_ROOT / "configs" / "train_baseline.yaml"


def expanded_telco(n_copies: int = 50):
    import pandas as pd

    base = load_raw_csv(FIXTURE)
    parts = []
    for i in range(n_copies):
        b = base.copy()
        b["customerID"] = b["customerID"].astype(str) + f"_b{i}"
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def split_three(tmp_path: Path):
    df = validate_raw_dataframe(expanded_telco(55))
    cfg = load_split_config(REPO_SPLIT)
    paths = run_split(df, tmp_path, cfg, input_path="fixture")
    return paths["train"], paths["validation"], paths["test"]


def fit_small_champion(tmp_path: Path, train_p: Path) -> Path:
    import pandas as pd

    train_df = pd.read_parquet(train_p)
    feat_cfg = load_features_config(REPO_FEATURES)
    train_cfg = load_train_baseline_config(REPO_TRAIN_BASELINE)
    pipe = build_baseline_pipeline(feat_cfg, train_cfg)
    X = select_feature_matrix(train_df, feat_cfg)
    y = y_positive_binary(train_df)
    pipe.fit(X, y)
    out = tmp_path / "champion.joblib"
    joblib.dump(pipe, out)
    return out


def write_eval_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "eval.yaml"
    p.write_text(
        """
champion_model_path: champion.joblib
fallback_model_path: champion.joblib
threshold_search:
  min_recall_churn: 0.25
  fbeta_beta: 1.0
  n_threshold_grid: 25
outputs:
  figure_dir: figs
  summary_json: summary.json
  summary_md: summary.md
  champion_manifest: champion_out.yaml
""".strip(),
        encoding="utf-8",
    )
    return p


def test_suggested_minimum_recall_churn():
    assert 0.0 <= suggested_minimum_recall_churn() <= 1.0


def test_select_threshold():
    import numpy as np

    y = np.array([0, 0, 1, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.85, 0.8, 0.3, 0.75])
    t, meta = select_threshold(y, p, min_recall_churn=0.5, fbeta_beta=1.0, n_grid=50)
    assert 0.01 <= t <= 0.99
    assert "constraint_satisfied" in meta


def test_run_evaluation_end_to_end(tmp_path: Path):
    train_p, val_p, test_p = split_three(tmp_path)
    champ = fit_small_champion(tmp_path, train_p)
    eval_y = write_eval_yaml(tmp_path)
    cfg = load_eval_config(eval_y)
    summary = run_evaluation(
        tmp_path,
        cfg,
        features_config_path=REPO_FEATURES,
        validation_parquet=val_p,
        test_parquet=test_p,
        champion_path=champ,
    )
    fig_dir = tmp_path / "figs"
    assert (fig_dir / "phase8_roc_validation.png").is_file()
    assert (fig_dir / "phase8_confusion_matrix_test.png").is_file()
    sj = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert sj["metrics_by_split"]["test"]["ranking_threshold_free"]["roc_auc"] > 0
    assert "threshold" in sj["champion"]
    champ_out = yaml.safe_load((tmp_path / "champion_out.yaml").read_text(encoding="utf-8"))
    assert champ_out["threshold"] == summary["champion"]["threshold"]
    assert "model_path" in champ_out
    assert "model_path_resolved" not in champ_out


def test_eval_cli(tmp_path: Path):
    train_p, val_p, test_p = split_three(tmp_path)
    champ = fit_small_champion(tmp_path, train_p)
    eval_y = write_eval_yaml(tmp_path)
    code = eval_main(
        [
            "--root",
            str(tmp_path),
            "--eval-config",
            str(eval_y),
            "--features-config",
            str(REPO_FEATURES),
            "-v",
            str(val_p),
            "--test",
            str(test_p),
            "--champion",
            str(champ),
        ]
    )
    assert code == 0


def test_eval_cli_subprocess(tmp_path: Path):
    train_p, val_p, test_p = split_three(tmp_path)
    champ = fit_small_champion(tmp_path, train_p)
    eval_y = write_eval_yaml(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_ml.evaluation.run",
            "--root",
            str(tmp_path),
            "--eval-config",
            str(eval_y),
            "--features-config",
            str(REPO_FEATURES),
            "-v",
            str(val_p),
            "--test",
            str(test_p),
            "--champion",
            str(champ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

