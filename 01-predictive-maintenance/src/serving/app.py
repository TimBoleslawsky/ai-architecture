"""Online scoring API (FastAPI).

POST /predict accepts the last N cycles of sensor readings for one unit and
returns the predicted remaining useful life plus a risk band. This is the
on-demand path (e.g. a technician asks "how is unit 42 doing?"), complementing
the scheduled batch scorer.

Run with:  uv run uvicorn src.serving.app:app --reload
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.predictor import Predictor

app = FastAPI(
    title="Predictive Maintenance API",
    description="RUL prediction and risk assessment for industrial equipment.",
    version="1.0.0",
)

predictor = Predictor(
    high_under=float(os.environ.get("RISK_HIGH_UNDER", 30)),
    medium_under=float(os.environ.get("RISK_MEDIUM_UNDER", 60)),
)


class SensorRow(BaseModel):
    cycle: int = Field(..., description="cycle number (1-based)")
    # sensor columns are validated dynamically at predict time
    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    unit: int = Field(..., description="unit identifier")
    rows: list[SensorRow] = Field(..., min_length=1, description="recent sensor readings")


class PredictResponse(BaseModel):
    unit: int
    predicted_rul: float
    risk: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": predictor.registry["name"]}


@app.get("/model")
def model_info() -> dict:
    return {
        "name": predictor.registry["name"],
        "version": predictor.registry["version"],
        "metrics": predictor.registry["metrics"],
        "window": predictor.window,
        "sensor_columns": predictor.sensor_columns,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    rows = [r.model_dump() for r in req.rows]
    try:
        rul, risk = predictor.predict_from_rows(rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(unit=req.unit, predicted_rul=round(rul, 1), risk=risk)
