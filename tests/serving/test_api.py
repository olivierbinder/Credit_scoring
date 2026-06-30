import pandas as pd
from fastapi.testclient import TestClient

from credit_scoring.serving import api


class DummyInput:
    def model_dump(self):
        return {
            "EXT_SOURCE_1": 0.1,
            "EXT_SOURCE_2": 0.2,
            "EXT_SOURCE_3": 0.3,
            "AMT_ANNUITY": 1000.0,
            "AMT_GOODS_PRICE": 10000.0,
            "DAYS_BIRTH": -10000,
            "DAYS_EMPLOYED": -100,
            "PAYMENT_RATE": 0.1,
            "OWN_CAR_AGE": 2.0,
            "CODE_GENDER": "M",
            "NAME_EDUCATION_TYPE": "Higher education",
            "INSTAL_DPD_MEAN": 0.0,
            "INSTAL_AMT_PAYMENT_SUM": 100.0,
            "POS_CNT_INSTALMENT_FUTURE_MEAN": 1.0,
            "POS_SK_DPD_DEF_MEAN": 0.0,
            "PREV_CNT_PAYMENT_MEAN": 2.0,
            "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN": -10.0,
            "ACTIVE_DAYS_CREDIT_MAX": -20.0,
            "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.0,
            "CC_CNT_DRAWINGS_CURRENT_VAR": 0.0,
        }


def _fake_prediction_payload():
    return {
        "probability": 0.73,
        "prediction": "Likely to default",
    }


def test_root_healthcheck():
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lookup_returns_404_when_client_missing(monkeypatch):
    monkeypatch.setattr(api, "lookup", lambda sk_id: None)
    client = TestClient(api.app)

    response = client.get("/lookup/999")

    assert response.status_code == 404
    body = response.json()
    assert "request_id" in body
    assert body["detail"] == "SK_ID_CURR 999 not found"


def test_predict_returns_422_on_missing_required_field():
    client = TestClient(api.app)
    payload = DummyInput().model_dump()
    payload.pop("EXT_SOURCE_2")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "request_id" in body
    assert isinstance(body["detail"], list)


def test_predict_returns_422_on_wrong_type():
    client = TestClient(api.app)
    payload = DummyInput().model_dump()
    payload["AMT_ANNUITY"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "request_id" in body
    assert isinstance(body["detail"], list)


def test_predict_returns_422_on_out_of_range_value():
    client = TestClient(api.app)
    payload = DummyInput().model_dump()
    payload["EXT_SOURCE_2"] = 1.5

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "request_id" in body
    assert isinstance(body["detail"], list)


def test_predict_success_logs_and_returns_prediction(monkeypatch, tmp_path):
    log_file = tmp_path / "predictions.jsonl"
    api.FILE_PRED = log_file

    monkeypatch.setattr(api, "predict", lambda input_data: _fake_prediction_payload())

    client = TestClient(api.app)
    response = client.post("/predict", json=DummyInput().model_dump())

    assert response.status_code == 200
    assert response.json() == _fake_prediction_payload()
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    assert '"success": true' in content[0]
    assert '"event": "prediction"' in content[0]


def test_reference_endpoint_returns_parquet(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_reference_df",
        lambda: pd.DataFrame({"SK_ID_CURR": [1], "VALUE": [10]}),
    )
    client = TestClient(api.app)

    response = client.get("/reference")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.content


def test_api_request_log_is_written(monkeypatch, tmp_path):
    log_file = tmp_path / "api_calls.jsonl"
    api.FILE_API = log_file
    monkeypatch.setattr(api, "lookup", lambda sk_id: {"SK_ID_CURR": sk_id})

    client = TestClient(api.app)
    response = client.get("/lookup/1")

    assert response.status_code == 200
    assert log_file.exists()
    assert '"event": "api_call"' in log_file.read_text(encoding="utf-8")
