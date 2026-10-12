from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

features = [
    "type", "amount", "oldbalanceOrg", "oldbalanceDest", "hour_of_day",
    "night_flag", "amount_to_origin_balance_ratio", "origin_balance_was_zero",
    "origin_txn_count_prior_steps"
]
X = pd.DataFrame([
    ["TRANSFER", 100.0, 200.0, 50.0, 10, 0, 0.5, 0, 1],
    ["CASH_OUT", 250.0, 300.0, 0.0, 23, 1, 0.8, 0, 2],
], columns=features)
y = [0, 1]

model = DummyClassifier(strategy="prior")
model.fit(X, y)
joblib.dump(model, MODELS / "fraud_detection_pipeline.pkl")
metadata = {
    "features": features,
    "feature_names": features,
    "threshold": 0.5,
    "artifact_note": "SYNTHETIC CI TEST MODEL ONLY — never use for real predictions."
}
(MODELS / "fraud_detection_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
print("Created synthetic CI-only model artifacts.")
