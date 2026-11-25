"""Defect-classification product API (ONNX Runtime) + web UI.

"AI as a product" lifecycle in one slice:
- a versioned, self-describing model artifact (registry.json + .onnx)
- an inference API that only depends on ONNX Runtime (no PyTorch at serving time)
- a web UI to consume the model
- optional prediction logging to Postgres (drift/product telemetry)

Run with:  uv run uvicorn src.serving.app:app --reload
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

MODELS_DIR = Path("models")
REGISTRY = MODELS_DIR / "registry.json"


class Predictor:
    def __init__(self, registry_path: Path = REGISTRY) -> None:
        if not registry_path.exists():
            raise FileNotFoundError(
                f"no model registry at {registry_path} - run src.models.train first"
            )
        registry = json.loads(registry_path.read_text())
        self.registry = registry
        self.classes = registry["classes"]
        self.input_name = registry["input_name"]
        self.output_name = registry["output_name"]
        pre = registry["preprocess"]
        self.size = int(pre["size"])
        self.mean = np.asarray(pre["mean"], dtype=np.float32)
        self.std = np.asarray(pre["std"], dtype=np.float32)
        self.session = ort.InferenceSession(
            registry["artifact"], providers=["CPUExecutionProvider"]
        )

    def predict(self, image_bytes: bytes) -> dict[str, float]:
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
            (self.size, self.size)
        )
        arr = np.asarray(img, dtype=np.float32)[None, None, :, :] / 255.0
        arr = (arr - self.mean.reshape(1, 1, 1, 1)) / self.std.reshape(1, 1, 1, 1)

        logits = self.session.run([self.output_name], {self.input_name: arr})[0][0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        return {cls: round(float(p), 4) for cls, p in zip(self.classes, probs)}


predictor = Predictor()

app = FastAPI(
    title="Defect Inspection API",
    description="Steel-surface defect classification (AI as a product).",
    version="1.0.0",
)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def log_prediction(result: dict) -> None:
    if not DATABASE_URL:
        return
    try:
        import psycopg2

        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (predicted_class, confidence, probabilities) "
                    "VALUES (%s, %s, %s)",
                    (result["class"], result["confidence"], json.dumps(result["probabilities"])),
                )
            conn.commit()
    except Exception:  # noqa: BLE001 - logging must never break the product
        pass


@app.get("/", response_class=FileResponse)
def index() -> str:
    return str(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": predictor.registry["name"]}


@app.get("/model")
def model_info() -> dict:
    return {
        "name": predictor.registry["name"],
        "version": predictor.registry["version"],
        "classes": predictor.classes,
        "metrics": predictor.registry["metrics"],
        "preprocess": predictor.registry["preprocess"],
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        probabilities = predictor.predict(content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"could not decode image: {exc}"
        ) from exc

    top_class = max(probabilities, key=probabilities.get)
    result = {
        "class": top_class,
        "confidence": probabilities[top_class],
        "probabilities": probabilities,
    }
    log_prediction(result)
    return result
