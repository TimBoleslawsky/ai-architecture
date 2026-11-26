"""Shared inference for the online API and the batch scorer.

Both entry points load the *same* registry entry, the *same* ONNX artifact and
apply the *same* preprocessing (take the last ``window`` cycles, normalize with
the train-set per-sensor statistics stored in the registry) - so serving can
never drift from training. Only ONNX Runtime is needed at serving time; the
training stack (PyTorch/Lightning) is a dev dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

MODELS_DIR = Path("models")
REGISTRY = MODELS_DIR / "registry.json"


class Predictor:
    def __init__(
        self,
        registry_path: Path = REGISTRY,
        high_under: float = 30,
        medium_under: float = 60,
    ) -> None:
        if not registry_path.exists():
            raise FileNotFoundError(
                f"no model registry at {registry_path} - run src.data.prepare and "
                "src.models.engine.train first"
            )
        registry = json.loads(registry_path.read_text())
        self.registry = registry
        self.high_under = high_under
        self.medium_under = medium_under

        self.window = int(registry["window"])
        self.sensor_columns = registry["sensor_columns"]
        self.sensor_mean = np.asarray(registry["sensor_mean"], dtype=np.float32)
        self.sensor_std = np.asarray(registry["sensor_std"], dtype=np.float32)
        self.input_name = registry["input_name"]
        self.output_name = registry["output_name"]
        self.session = ort.InferenceSession(
            registry["artifact"], providers=["CPUExecutionProvider"]
        )

    def window_from_rows(self, rows: list[dict]) -> np.ndarray:
        """Build the (1, window, n_sensors) input from raw per-cycle readings."""
        df = pd.DataFrame(rows)
        for col in self.sensor_columns:
            if col not in df.columns:
                raise ValueError(f"missing required sensor column: {col}")
        if "cycle" in df.columns:
            df = df.sort_values("cycle")
        tail = df[self.sensor_columns].tail(self.window).to_numpy(dtype=np.float64)
        if len(tail) < self.window:
            pad = np.repeat(tail[:1], self.window - len(tail), axis=0)
            tail = np.vstack([pad, tail])
        arr = (tail - self.sensor_mean) / self.sensor_std
        return arr.astype(np.float32)[None, :, :]  # (1, window, n_sensors)

    def predict_window(self, window: np.ndarray) -> tuple[float, str]:
        rul = float(self.session.run([self.output_name], {self.input_name: window})[0][0, 0])
        return rul, self.risk_label(rul)

    def predict_from_rows(self, rows: list[dict]) -> tuple[float, str]:
        return self.predict_window(self.window_from_rows(rows))

    def risk_label(self, rul: float) -> str:
        if rul < self.high_under:
            return "high"
        if rul < self.medium_under:
            return "medium"
        return "low"
