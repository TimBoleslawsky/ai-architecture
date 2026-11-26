"""Tests for data preparation (windows + unit split)."""
from src.data.prepare import build_windows, split_by_unit
from src.data.synthetic import generate_fleet


def test_split_by_unit_has_no_overlap():
    df = generate_fleet(20, seed=2)
    train, test = split_by_unit(df, test_fraction=0.2, seed=42)
    assert set(train["unit"]).isdisjoint(set(test["unit"]))


def test_build_windows_shape_and_targets():
    df = generate_fleet(5, seed=1)
    sensors = [c for c in df.columns if c.startswith("s_")]
    X, y, units = build_windows(df, sensors, window=30, step=10)
    assert X.shape == (len(X), 30, len(sensors))
    assert X.shape[0] == len(y) == len(units)
    assert y.min() >= 0
    assert y.max() <= df["rul"].max()


def test_build_windows_is_ordered_and_causal():
    import numpy as np

    from src.data.prepare import build_windows, split_by_unit
    from src.data.synthetic import generate_fleet

    df = generate_fleet(3, seed=3).sort_values(["unit", "cycle"])
    sensors = [c for c in df.columns if c.startswith("s_")]
    X, y, units = build_windows(df, sensors, window=8, step=4)
    first_unit = units[0]
    # First window starts at cycle 1 for that unit.
    expected = df[(df.unit == first_unit) & (df.cycle == 1)][sensors[0]].iloc[0]
    np.testing.assert_allclose(X[0, 0, 0], expected, rtol=1e-5)
