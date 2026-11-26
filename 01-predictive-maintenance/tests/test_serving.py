"""Integration tests for the serving layer.

These require a trained model (run ``uv run python -m src.models.engine.train``
first); they are skipped automatically when the registry is absent so ``pytest``
stays green on a fresh checkout.
"""
import pytest

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("models/registry.json").exists(),
    reason="model not trained yet - run src.data.prepare + src.models.engine.train first",
)


def test_predictor_on_raw_sensor_window():
    from src.data.synthetic import generate_fleet
    from src.serving.predictor import Predictor

    raw = generate_fleet(20, seed=5)
    unit = raw["unit"].iloc[0]
    window = raw[raw["unit"] == unit].tail(40)
    rows = window[["cycle"] + [c for c in raw.columns if c.startswith("s_")]].to_dict("records")

    p = Predictor()
    input_arr = p.window_from_rows(rows)
    assert input_arr.shape == (1, p.window, len(p.sensor_columns))

    rul, risk = p.predict_from_rows(rows)
    assert 0 <= rul
    assert risk in {"high", "medium", "low"}


def test_predictor_pads_short_histories():
    from src.data.synthetic import generate_fleet
    from src.serving.predictor import Predictor

    raw = generate_fleet(5, seed=6)
    unit = raw["unit"].iloc[0]
    rows = raw[raw["unit"] == unit].head(10)
    rows = rows[["cycle"] + [c for c in raw.columns if c.startswith("s_")]].to_dict("records")

    p = Predictor()
    rul, risk = p.predict_from_rows(rows)
    assert 0 <= rul
    assert risk in {"high", "medium", "low"}


def test_predict_endpoint_accepts_raw_readings():
    from fastapi.testclient import TestClient
    from src.data.synthetic import generate_fleet
    from src.serving.app import app

    raw = generate_fleet(20, seed=5)
    unit = raw["unit"].iloc[0]
    window = raw[raw["unit"] == unit].tail(40)
    rows = window[["cycle"] + [c for c in raw.columns if c.startswith("s_")]].to_dict("records")

    client = TestClient(app)
    resp = client.post("/predict", json={"unit": int(unit), "rows": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert body["unit"] == int(unit)
    assert body["risk"] in {"high", "medium", "low"}


def test_predict_endpoint_rejects_missing_sensors():
    from fastapi.testclient import TestClient
    from src.serving.app import app

    client = TestClient(app)
    resp = client.post("/predict", json={"unit": 1, "rows": [{"cycle": 1}]})
    assert resp.status_code == 400
