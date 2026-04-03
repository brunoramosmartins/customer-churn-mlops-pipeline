"""Phase 9 — batch inference with Pydantic row validation."""

from churn_ml.batch_predict.predict import batch_predict, load_batch_predict_config, load_champion_manifest
from churn_ml.batch_predict.row_model import build_inference_row_model

__all__ = [
    "batch_predict",
    "build_inference_row_model",
    "load_batch_predict_config",
    "load_champion_manifest",
]
