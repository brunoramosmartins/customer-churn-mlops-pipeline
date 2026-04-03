"""Univariate drift: Kolmogorov–Smirnov (numeric) and chi-square (categorical)."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from churn_ml.features.pipeline import load_features_config


@dataclass
class DriftRow:
    column: str
    kind: str
    statistic: float | None
    p_value: float | None
    drift_flag: bool
    note: str | None = None


def _numeric_drift(ref: pd.Series, cur: pd.Series) -> DriftRow:
    col = ref.name or "column"
    a = ref.dropna().astype(float)
    b = cur.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return DriftRow(col, "numeric", None, None, False, "too few non-null values")
    stat, p = ks_2samp(a, b)
    return DriftRow(col, "numeric", float(stat), float(p), bool(p < 0.05), None)


def _categorical_drift(ref: pd.Series, cur: pd.Series) -> DriftRow:
    col = ref.name or "column"
    r = ref.fillna("__NA__").astype(str)
    c = cur.fillna("__NA__").astype(str)
    cats = sorted(set(r.unique()) | set(c.unique()))
    if len(cats) < 2:
        return DriftRow(col, "categorical", None, None, False, "single category after NA fill")
    r_counts = r.value_counts().reindex(cats, fill_value=0)
    c_counts = c.value_counts().reindex(cats, fill_value=0)
    table = np.array([r_counts.values, c_counts.values], dtype=float)
    if table.sum(axis=1).min() == 0:
        return DriftRow(col, "categorical", None, None, False, "empty split for contingency")
    chi2, p, dof, _ = chi2_contingency(table)
    if dof == 0:
        return DriftRow(col, "categorical", None, None, False, "zero degrees of freedom")
    return DriftRow(col, "categorical", float(chi2), float(p), bool(p < 0.05), None)


def run_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feat_cfg: dict[str, Any],
    *,
    alpha: float = 0.05,
) -> list[DriftRow]:
    """Compare distributions column-wise on modeling features only."""
    rows: list[DriftRow] = []
    for col in feat_cfg["numeric_features"]:
        if col not in reference.columns or col not in current.columns:
            rows.append(DriftRow(col, "numeric", None, None, False, "column missing in one frame"))
            continue
        r = _numeric_drift(reference[col], current[col])
        rows.append(
            DriftRow(
                r.column,
                r.kind,
                r.statistic,
                r.p_value,
                r.p_value is not None and r.p_value < alpha,
                r.note,
            )
        )
    for col in feat_cfg["categorical_features"]:
        if col not in reference.columns or col not in current.columns:
            rows.append(DriftRow(col, "categorical", None, None, False, "column missing in one frame"))
            continue
        r = _categorical_drift(reference[col], current[col])
        rows.append(
            DriftRow(
                r.column,
                r.kind,
                r.statistic,
                r.p_value,
                r.p_value is not None and r.p_value < alpha,
                r.note,
            )
        )
    return rows


def write_drift_artifacts(
    rows: list[DriftRow],
    *,
    html_path: Path,
    json_path: Path,
    reference_path: str,
    current_path: str,
    alpha: float,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": reference_path,
        "current": current_path,
        "alpha": alpha,
        "n_columns_flagged": sum(1 for r in rows if r.drift_flag),
        "columns": [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    esc = html.escape
    trs = []
    for r in rows:
        flag = "yes" if r.drift_flag else "no"
        trs.append(
            "<tr>"
            f"<td>{esc(r.column)}</td><td>{esc(r.kind)}</td>"
            f"<td>{esc('' if r.statistic is None else f'{r.statistic:.6g}')}</td>"
            f"<td>{esc('' if r.p_value is None else f'{r.p_value:.6g}')}</td>"
            f"<td><b>{esc(flag)}</b></td>"
            f"<td>{esc(r.note or '')}</td>"
            "</tr>"
        )
    body = "\n".join(trs)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><title>Drift report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; max-width: 56rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Data drift (univariate)</h1>
<p>Reference: <code>{esc(reference_path)}</code><br/>
Current: <code>{esc(current_path)}</code><br/>
Flag if p-value &lt; {alpha} (exploratory; not a substitute for production monitoring).</p>
<table>
<thead><tr><th>Column</th><th>Kind</th><th>Statistic</th><th>p-value</th><th>Drift?</th><th>Note</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
<p>See <code>docs/DRIFT.md</code> for interpretation.</p>
</body>
</html>
"""
    html_path.write_text(doc, encoding="utf-8")


def load_table(path: Path) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suf in (".csv",):
        return pd.read_csv(p)
    raise ValueError(f"unsupported table format: {p}")


def drift_from_paths(
    reference_path: Path,
    current_path: Path,
    features_config_path: Path,
    *,
    alpha: float = 0.05,
) -> list[DriftRow]:
    feat_cfg = load_features_config(features_config_path)
    ref = load_table(reference_path)
    cur = load_table(current_path)
    return run_drift_analysis(ref, cur, feat_cfg, alpha=alpha)
