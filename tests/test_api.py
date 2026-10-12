from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_TRANSACTION = {
    "type": "TRANSFER",
    "amount": 1500.0,
    "oldbalanceOrg": 2000.0,
    "oldbalanceDest": 500.0,
    "hour_of_day": 14,
    "night_flag": 0,
    "amount_to_origin_balance_ratio": 0.75,
    "origin_balance_was_zero": 0,
    "origin_txn_count_prior_steps": 3,
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["feature_count"] == 9
    assert response.json()["threshold"] is not None

def test_valid_prediction_request():
    response = client.post("/predict", json=VALID_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] in (0, 1)
    assert 0 <= body["fraud_probability"] <= 1
    assert 0 <= body["threshold"] <= 1

def test_negative_amount_is_rejected():
    bad_transaction = {**VALID_TRANSACTION, "amount": -1}
    response = client.post("/predict", json=bad_transaction)
    assert response.status_code == 422

def test_unexpected_field_is_rejected():
    bad_transaction = {**VALID_TRANSACTION, "unexpected_field": "not allowed"}
    response = client.post("/predict", json=bad_transaction)
    assert response.status_code == 422

def test_invalid_hour_is_rejected():
    bad_transaction = {**VALID_TRANSACTION, "hour_of_day": 24}
    response = client.post("/predict", json=bad_transaction)
    assert response.status_code == 422
