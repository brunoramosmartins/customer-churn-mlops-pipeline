"""Load champion pipeline, validate rows, write predictions + metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import TypeAdapter

from churn_ml.batch_predict.row_model import build_inference_row_model
from churn_ml.features.pipeline import load_features_config, select_feature_matrix
from churn_ml.fsutil import path_relative_to_repo
from churn_ml.metrics import negative_class_label, positive_class_label, target_column


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_batch_predict_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("batch_predict config must be a YAML mapping")
    for key in ("champion_manifest", "features_config", "artifact_version", "default_output", "metadata_output"):
        if key not in cfg:
            raise ValueError(f"batch_predict config missing required key: {key}")
    return cfg


def load_champion_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("champion manifest must be a YAML mapping")
    if "model_path" not in data and "model_path_resolved" not in data:
        raise ValueError("champion manifest missing model_path")
    if "threshold" not in data:
        raise ValueError("champion manifest missing threshold")
    return data


def resolve_model_path(root: Path, manifest: dict[str, Any]) -> Path:
    resolved = manifest.get("model_path_resolved")
    if resolved:
        p = Path(str(resolved))
        if p.is_file():
            return p
    mp = Path(str(manifest["model_path"]))
    if mp.is_absolute() and mp.is_file():
        return mp
    cand = (root / mp).resolve()
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"Champion model file not found (tried manifest and {cand})")


def _git_sha(root: Path) -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return proc.stdout.strip(), None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def batch_predict(
    root: Path,
    *,
    input_path: Path,
    output_path: Path,
    batch_cfg: dict[str, Any],
    champion_manifest_path: Path,
    features_config_path: Path,
    threshold_override: float | None = None,
    write_metadata: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")

    manifest = load_champion_manifest(champion_manifest_path)
    feat_cfg = load_features_config(features_config_path)
    model_path = resolve_model_path(root, manifest)
    threshold = float(threshold_override if threshold_override is not None else manifest["threshold"])

    pipe = joblib.load(model_path)
    RowModel = build_inference_row_model(feat_cfg)

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df_in = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df_in = pd.read_csv(input_path)
    else:
        raise ValueError("input must be .csv or .parquet")

    records = df_in.to_dict(orient="records")
    for r in records:
        c = r.get("Churn")
        if c is not None and pd.isna(c):
            r["Churn"] = None
    adapter = TypeAdapter(list[RowModel])
    validated = adapter.validate_python(records)
    df = pd.DataFrame([m.model_dump() for m in validated])

    X = select_feature_matrix(df, feat_cfg)
    proba = pipe.predict_proba(X)[:, 1]
    pos = positive_class_label()
    neg = negative_class_label()
    pred = np.where(proba >= threshold, pos, neg)

    id_col = feat_cfg["id_column"]
    out = pd.DataFrame(
        {
            id_col: df[id_col],
            "churn_probability": proba,
            "predicted_churn": pred,
        }
    )
    tgt = target_column()
    if tgt in df.columns and df[tgt].notna().any():
        out["actual_churn"] = df[tgt]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        out.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        out.to_csv(output_path, index=False)
    else:
        raise ValueError("output must be .csv or .parquet")

    git_sha, git_err = _git_sha(root)
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_version": str(batch_cfg["artifact_version"]),
        "champion_manifest": _path_for_metadata(root, champion_manifest_path),
        "model_path_resolved": path_relative_to_repo(root, model_path),
        "threshold": threshold,
        "threshold_source": "override" if threshold_override is not None else "champion_manifest",
        "input_path": str(input_path),
        "input_sha256": _file_sha256(input_path),
        "n_rows": len(out),
        "output_path": str(output_path),
        "git_sha": git_sha,
        "git_sha_error": git_err,
    }

    if write_metadata:
        meta_path = _resolve_out(root, batch_cfg["metadata_output"])
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta["metadata_written"] = str(meta_path)

    return meta


def _resolve_out(root: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (root / pp).resolve()


def _path_for_metadata(root: Path, p: Path) -> str:
    return path_relative_to_repo(root, p)
