"""Dynamic Pydantic row model from features.yaml (strict: no extra columns)."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field, create_model


def _blank_to_none_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)


def _int_non_neg(v: Any) -> int:
    return int(v)


def _senior_01(v: Any) -> int:
    if isinstance(v, str):
        v = v.strip()
        if v in ("0", "1"):
            return int(v)
    x = int(float(v))
    if x not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1")
    return x


def build_inference_row_model(feat_cfg: dict) -> type:
    """One row = all modeling columns + optional ``Churn`` (ignored for features). ``extra='forbid'``."""
    id_col = feat_cfg["id_column"]
    TotalChargesT = Annotated[float | None, BeforeValidator(_blank_to_none_float)]

    fields: dict[str, Any] = {
        "__config__": ConfigDict(str_strip_whitespace=True, extra="forbid"),
        id_col: (str, Field(..., min_length=1)),
    }
    for c in feat_cfg["numeric_features"]:
        if c == "SeniorCitizen":
            fields[c] = (Annotated[int, BeforeValidator(_senior_01)], Field(...))
        elif c == "tenure":
            fields[c] = (Annotated[int, BeforeValidator(_int_non_neg)], Field(..., ge=0))
        elif c == "MonthlyCharges":
            fields[c] = (float, Field(..., ge=0))
        elif c == "TotalCharges":
            fields[c] = (TotalChargesT, Field(default=None))
        else:
            fields[c] = (float, Field(...))
    for c in feat_cfg["categorical_features"]:
        fields[c] = (str, Field(..., min_length=1))
    fields["Churn"] = (str | None, Field(default=None))

    return create_model("TelcoInferenceRow", **fields)
