"""Synthetic fleet sensor data generator.

Falls back to a fully self-contained, reproducible dataset when the real
C-MAPSS turbofan data is not available (no network / no download). The
generator models a simple, learnable degradation process:

  sensor_i(cycle) = baseline_i + slope_i * RUL(cycle) + noise

A subset of sensors carries the signal: their reading is linearly related to
the remaining useful life, while the rest are pure noise. The model therefore
has to *learn which sensors are informative* to estimate RUL.

Run with:  uv run python -m src.data.synthetic [num_units] [out_path]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

N_SENSORS = 14
DEFAULT_UNITS = 120
MAX_CYCLES_RANGE = (80, 250)


def generate_fleet(num_units: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows: list[dict] = []
    for unit in range(1, num_units + 1):
        max_cycles = int(rng.integers(*MAX_CYCLES_RANGE))
        # Per-unit degradation: most sensors are noise, a handful carry signal.
        slopes = np.zeros(N_SENSORS)
        signal = random.sample(range(N_SENSORS), k=max(6, N_SENSORS // 2))
        slopes[[i for i in range(N_SENSORS) if i in signal]] = rng.normal(
            0.10, 0.02, len(signal)
        )
        baselines = rng.normal(100.0, 2.0, N_SENSORS)
        noise = rng.normal(0.0, 0.4, N_SENSORS)

        for cycle in range(1, max_cycles + 1):
            rul = max_cycles - cycle
            values = baselines + slopes * rul + noise
            row: dict = {"unit": unit, "cycle": cycle}
            for i in range(N_SENSORS):
                row[f"s_{i + 1}"] = round(float(values[i]), 3)
            row["rul"] = rul
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=DEFAULT_UNITS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw/synthetic_fd001.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_fleet(args.units, args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows ({args.units} units) -> {args.out}")


if __name__ == "__main__":
    main()
