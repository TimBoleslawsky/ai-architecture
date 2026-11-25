"""Shared inference helpers for the online API and the batch scorer.

Both entry points load the *same* registry entry, the *same* artifact and the
*same* feature-engineering code, so there is a single definition of truth for
how a raw sensor window becomes a RUL prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.engineer import add_rolling_features

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
                f"no model registry at {registry_path} - run src.data.prepare and src.models.train first"
            )
        registry = json.loads(registry_path.read_text())
        self.registry = registry
        self.high_under = high_under
        self.medium_under = medium_under
        self.feature_columns = registry["feature_columns"]
        self.sensor_columns = registry["sensor_columns"]
        self.model = joblib.load(registry["artifact"])

    def features_from_rows(self, rows: list[dict]) -> pd.DataFrame:
        """Build the model feature row(s) from raw per-cycle sensor readings."""
        df = pd.DataFrame(rows)
        for col in self.sensor_columns:
            if col not in df.columns:
                raise ValueError(f"missing required sensor column: {col}")
        if "unit" not in df.columns:
            df["unit"] = 1
        engineered = add_rolling_features(df, self.sensor_columns)
        return engineered[self.feature_columns]

    def predict(self, feature_row: pd.Series) -> tuple[float, str]:
        rul = float(self.model.predict(feature_row.to_frame().T)[0])
        return rul, self.risk_label(rul)

    def risk_label(self, rul: float) -> str:
        if rul < self.high_under:
            return "high"
        if rul < self.medium_under:
            return "medium"
        return "low"
