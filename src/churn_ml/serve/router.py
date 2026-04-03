"""HTTP routes: /health and /predict."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import ValidationError

from churn_ml.features.pipeline import select_feature_matrix
from churn_ml.metrics import negative_class_label, positive_class_label

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    st = getattr(request.app.state, "churn", None)
    if st is None:
        return {"status": "degraded", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(st.model_path),
        "threshold": st.threshold,
    }


@router.post("/predict")
def predict(request: Request, row: dict[str, Any] = Body(...)) -> dict[str, Any]:
    st = getattr(request.app.state, "churn", None)
    if st is None:
        return {"error": "model not loaded"}

    try:
        validated = st.row_model.model_validate(row)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    df = pd.DataFrame([validated.model_dump()])
    X = select_feature_matrix(df, st.feat_cfg)
    proba = float(st.pipeline.predict_proba(X)[0, 1])
    pos = positive_class_label()
    neg = negative_class_label()
    pred = pos if proba >= st.threshold else neg
    return {
        "churn_probability": proba,
        "predicted_churn": pred,
    }
