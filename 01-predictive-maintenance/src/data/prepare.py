"""End-to-end data preparation: raw -> engineered features -> train/test split.

Strategy:
 1. Use the real C-MAPSS data if present, otherwise generate synthetic data.
 2. Add causal rolling-window features (same code as serving).
 3. Split *by unit* (no unit appears in both sets) so the test set is a genuine
    out-of-sample fleet, not randomly interleaved rows.

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
META_OUT = PROCESSED / "meta.json"

TEST_FRACTION = 0.2
SEED = 42


def load_raw() -> tuple[pd.DataFrame, str]:
    if RAW_CMAPSS.exists():
        return pd.read_csv(RAW_CMAPSS), "cmapss"
    if not RAW_SYNTHETIC.exists():
        print("no raw data found, generating synthetic fleet data...")
        RAW_SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
        generate_fleet(120, SEED).to_csv(RAW_SYNTHETIC, index=False)
    return pd.read_csv(RAW_SYNTHETIC), "synthetic"


def split_by_unit(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = np.array(sorted(df["unit"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_test = max(1, int(round(len(units) * test_fraction)))
    test_units = set(units[:n_test])
    mask = df["unit"].isin(test_units)
    return df[~mask], df[mask]


def main() -> None:
    df, source = load_raw()
    sensor_cols = infer_sensor_columns(df)
    if not sensor_cols:
        raise ValueError("raw data has no s_<n> sensor columns")

    engineered = add_rolling_features(df, sensor_cols)
    rolling_cols = [c for c in engineered.columns if c not in set(df.columns)]
    feature_cols = ["cycle"] + sensor_cols + rolling_cols

    train, test = split_by_unit(engineered, TEST_FRACTION, SEED)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    meta = {
        "source": source,
        "n_units": int(df["unit"].nunique()),
        "n_rows": int(len(df)),
        "sensor_columns": sensor_cols,
        "feature_columns": feature_cols,
        "target": "rul",
        "test_units": sorted(test["unit"].unique().tolist()),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    META_OUT.write_text(json.dumps(meta, indent=2))

    print(f"source={source} units={meta['n_units']} rows={meta['n_rows']}")
    print(f"train rows={len(train):,} test rows={len(test):,} features={len(feature_cols)}")
    print(f"wrote {TRAIN_OUT}, {TEST_OUT}, {META_OUT}")


if __name__ == "__main__":
    main()
