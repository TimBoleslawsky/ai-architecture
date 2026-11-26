"""End-to-end data preparation: raw -> sequence windows + tabular features.

The bucket-1 example is efficacy-first, so we prepare **two** representations:

1. **Sequence windows** (``windows_*.npz``) for the deep TCN model: sliding
   windows of the last ``W`` cycles per unit, normalized with train-set
   per-sensor statistics (the same statistics are persisted to the model
   registry and reused at serving time -> zero train/serve skew).
2. **Rolling-window tabular features** (``features_*.csv``) for the classical
   baseline (RandomForest) used in the comparison.

Both representations share the same *by-unit* train/test split so no unit
appears on both sides (genuine out-of-sample fleet).

Run with:  uv run python -m src.data.prepare
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.synthetic import generate_fleet
from src.features.engineer import add_rolling_features, infer_sensor_columns

RAW_CMAPSS = Path("data/raw/cmapss/FD001_train.csv")
RAW_SYNTHETIC = Path("data/raw/synthetic_fd001.csv")
PROCESSED = Path("data/processed")
TRAIN_OUT = PROCESSED / "features_train.csv"
TEST_OUT = PROCESSED / "features_test.csv"
WINDOWS_TRAIN = PROCESSED / "windows_train.npz"
WINDOWS_TEST = PROCESSED / "windows_test.npz"
META_OUT = PROCESSED / "meta.json"

TEST_FRACTION = 0.2
SEED = 42
WINDOW = 30
WINDOW_STEP = 1


def load_raw() -> tuple[pd.DataFrame, str]:
    if RAW_CMAPSS.exists():
        return pd.read_csv(RAW_CMAPSS), "cmapss"
    if not RAW_SYNTHETIC.exists():
        print("no raw data found, generating synthetic fleet data...")
        RAW_SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
        generate_fleet(300, SEED).to_csv(RAW_SYNTHETIC, index=False)
    return pd.read_csv(RAW_SYNTHETIC), "synthetic"


def split_by_unit(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = np.array(sorted(df["unit"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_test = max(1, int(round(len(units) * test_fraction)))
    test_units = set(units[:n_test])
    mask = df["unit"].isin(test_units)
    return df[~mask], df[mask]


def build_windows(
    df: pd.DataFrame, sensor_cols: list[str], window: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over each unit's cycle-sorted sensors -> (X, y, units).

    X shape: (n_windows, window, n_sensors); y is the RUL at the end of the window.
    """
    x_list, y_list, unit_list = [], [], []
    df = df.sort_values(["unit", "cycle"])
    for unit, group in df.groupby("unit"):
        values = group[sensor_cols].to_numpy(dtype=np.float64)
        rul = group["rul"].to_numpy(dtype=np.float64)
        for start in range(0, len(values) - window + 1, step):
            end = start + window
            x_list.append(values[start:end])
            y_list.append(rul[end - 1])
            unit_list.append(unit)
    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(unit_list, dtype=np.int64),
    )


def main() -> None:
    df, source = load_raw()
    sensor_cols = infer_sensor_columns(df)
    if not sensor_cols:
        raise ValueError("raw data has no s_<n> sensor columns")

    train, test = split_by_unit(df, TEST_FRACTION, SEED)

    # --- Sequence windows for the deep model (normalized) ---
    X_train, y_train, units_train = build_windows(train, sensor_cols, WINDOW, WINDOW_STEP)
    X_test, y_test, units_test = build_windows(test, sensor_cols, WINDOW, WINDOW_STEP)

    sensor_mean = X_train.reshape(-1, len(sensor_cols)).mean(axis=0, dtype=np.float64)
    sensor_std = X_train.reshape(-1, len(sensor_cols)).std(axis=0, dtype=np.float64)
    sensor_std = np.where(sensor_std < 1e-8, 1.0, sensor_std)

    X_train_n = ((X_train - sensor_mean) / sensor_std).astype(np.float32)
    X_test_n = ((X_test - sensor_mean) / sensor_std).astype(np.float32)

    # --- Tabular rolling features for the classical baseline (raw values) ---
    eng_train = add_rolling_features(train, sensor_cols)
    eng_test = add_rolling_features(test, sensor_cols)
    rolling_cols = [c for c in eng_train.columns if c not in set(df.columns)]
    feature_cols = ["cycle"] + sensor_cols + rolling_cols

    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(WINDOWS_TRAIN, X=X_train_n, y=y_train, units=units_train)
    np.savez(WINDOWS_TEST, X=X_test_n, y=y_test, units=units_test)
    eng_train.to_csv(TRAIN_OUT, index=False)
    eng_test.to_csv(TEST_OUT, index=False)

    meta = {
        "source": source,
        "n_units": int(df["unit"].nunique()),
        "n_rows": int(len(df)),
        "sensor_columns": sensor_cols,
        "feature_columns": feature_cols,
        "window": WINDOW,
        "sensor_mean": [round(float(v), 6) for v in sensor_mean],
        "sensor_std": [round(float(v), 6) for v in sensor_std],
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(X_test)),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    META_OUT.write_text(json.dumps(meta, indent=2))

    print(f"source={source} units={meta['n_units']} rows={meta['n_rows']:,}")
    print(f"deep windows: train={len(X_train):,} test={len(X_test):,} (W={WINDOW}, {len(sensor_cols)} sensors)")
    print(f"baseline features: train={len(eng_train):,} test={len(eng_test):,} (n_features={len(feature_cols)})")
    print(f"wrote {WINDOWS_TRAIN}, {WINDOWS_TEST}, {TRAIN_OUT}, {TEST_OUT}, {META_OUT}")


if __name__ == "__main__":
    main()
