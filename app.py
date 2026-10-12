from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_pipeline.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "fraud_detection_metadata.json"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError("Model artifacts not found under project_root/models.")

loaded_artifact = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

model = loaded_artifact
if isinstance(loaded_artifact, dict):
    candidate_keys = ["model", "pipeline", "best_model", "best_estimator",
                      "estimator", "classifier", "fraud_detection_pipeline"]
    model = next(
        (loaded_artifact[key] for key in candidate_keys
         if key in loaded_artifact and hasattr(loaded_artifact[key], "predict_proba")),
        None
    )
    if model is None:
        candidates = [value for value in loaded_artifact.values()
                      if hasattr(value, "predict_proba")]
        if len(candidates) == 1:
            model = candidates[0]
        else:
            raise TypeError("Could not identify a unique predict_proba estimator in saved artifact.")

if not hasattr(model, "predict_proba"):
    raise TypeError("Loaded model does not provide predict_proba().")

feature_names = metadata.get("features") or metadata.get("feature_names")
if not feature_names and isinstance(loaded_artifact, dict):
    feature_names = loaded_artifact.get("features") or loaded_artifact.get("feature_names")
if not feature_names:
    feature_names = [
        "type", "amount", "oldbalanceOrg", "oldbalanceDest", "hour_of_day",
        "night_flag", "amount_to_origin_balance_ratio", "origin_balance_was_zero",
        "origin_txn_count_prior_steps"
    ]

threshold = float(metadata.get("threshold", 0.5))

app = FastAPI(title="Fraud Detection API", version="1.0.0")

class TransactionFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)
    night_flag: int = Field(ge=0, le=1)
    amount_to_origin_balance_ratio: float = Field(ge=0)
    origin_balance_was_zero: int = Field(ge=0, le=1)
    origin_txn_count_prior_steps: int = Field(ge=0)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "features": feature_names,
        "threshold": threshold,
    }

@app.post("/predict")
def predict(transaction: TransactionFeatures):
    row = transaction.model_dump()
    missing = [name for name in feature_names if name not in row]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required model features: {missing}")
    X = pd.DataFrame([{name: row[name] for name in feature_names}], columns=feature_names)
    try:
        probabilities = model.predict_proba(X)
        fraud_index = list(model.classes_).index(1)
        fraud_probability = float(probabilities[0][fraud_index])
        prediction = int(fraud_probability >= threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")
    return {
        "is_fraud": prediction,
        "fraud_probability": fraud_probability,
        "threshold": threshold,
        "decision": "review_as_potential_fraud" if prediction else "not_flagged_by_model",
        "disclaimer": "Model output is a risk signal, not proof of fraud.",
    }
