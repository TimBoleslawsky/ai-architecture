"""Tests for data synthesis and feature engineering."""
import numpy as np
import pandas as pd

from src.data.synthetic import generate_fleet
from src.features.engineer import add_rolling_features, infer_sensor_columns


def test_synthetic_fleet_shape():
    df = generate_fleet(num_units=10, seed=7)
    assert list(df.columns)[:3] == ["unit", "cycle", "s_1"]
    assert {"unit", "cycle", "rul"}.issubset(df.columns)
    assert df["unit"].nunique() == 10
    assert (df.groupby("unit")["cycle"].diff().dropna() == 1).all()


def test_synthetic_is_reproducible():
    a = generate_fleet(20, seed=3)
    b = generate_fleet(20, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_rul_is_monotonic_decreasing():
    df = generate_fleet(5, seed=1).sort_values(["unit", "cycle"])
    per_unit = df.groupby("unit")["rul"]
    for _, rul in per_unit:
        assert (rul.diff().fillna(0) <= 0).all()


def test_rolling_features_added_per_sensor():
    df = generate_fleet(5, seed=1)
    sensors = infer_sensor_columns(df)
    out = add_rolling_features(df, sensors, window=10)
    new_cols = {c for c in out.columns if c not in df.columns}
    assert len(new_cols) == 4 * len(sensors)


def test_rolling_features_are_causal():
    df = generate_fleet(3, seed=2).sort_values(["unit", "cycle"])
    sensors = infer_sensor_columns(df)
    out = add_rolling_features(df, sensors, window=5)
    col = f"{sensors[0]}_mean_w5"
    # For cycle 1, the rolling mean must equal the raw value (min_periods=1).
    first = out.groupby("unit").head(1)
    np.testing.assert_allclose(
        first[col].values, first[sensors[0]].values, rtol=1e-6
    )
