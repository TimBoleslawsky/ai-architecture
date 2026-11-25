"""Integration tests for the ONNX serving layer.

Skipped automatically until a model is trained (``uv run python -m src.models.train``).
"""
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("models/registry.json").exists(),
    reason="model not trained yet - run src.models.train first",
)


def test_predictor_on_held_out_image(tmp_path):
    from src.data.synthetic import generate_dataset
    from src.serving.app import Predictor

    generate_dataset(tmp_path, per_class=3, seed=11)
    img = (tmp_path / "scratches" / "img_0000.png").read_bytes()

    p = Predictor()
    probs = p.predict(img)
    assert set(probs) == set(p.classes)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert max(probs, key=probs.get) == "scratches"


def test_predict_image_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from src.data.synthetic import generate_dataset
    from src.serving.app import app

    generate_dataset(tmp_path, per_class=3, seed=11)
    img = (tmp_path / "crazing" / "img_0001.png").read_bytes()

    client = TestClient(app)
    resp = client.post(
        "/predict/image", files={"file": ("c.png", img, "image/png")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["class"] in {"crazing", "inclusion", "patches",
                             "pitted_surface", "rolled-in_scale", "scratches"}
    assert 0 <= body["confidence"] <= 1
    assert len(body["probabilities"]) == 6


def test_predict_endpoint_rejects_garbage():
    from fastapi.testclient import TestClient
    from src.serving.app import app

    client = TestClient(app)
    resp = client.post(
        "/predict/image", files={"file": ("x.png", b"not an image", "image/png")}
    )
    assert resp.status_code == 422
