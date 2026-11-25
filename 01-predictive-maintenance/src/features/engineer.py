"""Causal rolling-window feature engineering shared by training and serving.

All features are *causal*: each row only uses sensor readings up to and including
its own cycle, so the same code can be used offline (training) and online (the
REST API), which is what keeps train/serve skew at zero.
"""
from __future__ import annotations

import pandas as pd

WINDOW = 15


def infer_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return sensor columns, i.e. columns that are named ``s_<n>``."""
    return [c for c in df.columns if c.startswith("s_")]


def add_rolling_features(
    df: pd.DataFrame, sensor_cols: list[str], window: int = WINDOW
) -> pd.DataFrame:
    """Add rolling mean/std/min/max per sensor per unit.

    ``df`` must contain ``unit`` and ``cycle`` columns and be sorted per unit by
    cycle (we sort defensively here). Returns a new DataFrame with the rolling
    features appended.
    """
    if not sensor_cols:
        raise ValueError("no sensor columns found (expected columns named s_<n>)")

    df = df.copy()
    df = df.sort_values(["unit", "cycle"])
    grouped = df.groupby("unit", sort=False)[sensor_cols]

    extra: dict[str, pd.Series] = {}
    for stat in ("mean", "std", "min", "max"):
        rolled = grouped.rolling(window, min_periods=1)
        res = getattr(rolled, stat)()
        for col in sensor_cols:
            extra[f"{col}_{stat}_w{window}"] = res[col].values

    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)


def build_feature_set(df: pd.DataFrame, sensor_cols: list[str], window: int = WINDOW) -> pd.DataFrame:
    """Return the model-ready feature matrix (with rolling features added)."""
    return add_rolling_features(df, sensor_cols, window)
