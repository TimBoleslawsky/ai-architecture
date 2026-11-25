"""Seed the Postgres ``sensor_readings`` table from the raw dataset.

Models the (simplified) ingestion step of a real fleet telemetry pipeline:
sensors land in a database, from which the batch scorer builds features and
scores every unit on a schedule.

Run with (point DATABASE_URL at the compose Postgres):
    uv run python -m src.seed
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db import init_schema, seed_readings

RAW_CMAPSS = Path("data/raw/cmapss/FD001_train.csv")
RAW_SYNTHETIC = Path("data/raw/synthetic_fd001.csv")


def main() -> None:
    if RAW_CMAPSS.exists():
        df = pd.read_csv(RAW_CMAPSS)
    elif RAW_SYNTHETIC.exists():
        df = pd.read_csv(RAW_SYNTHETIC)
    else:
        from src.data.synthetic import generate_fleet

        RAW_SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
        df = generate_fleet(120, 42).to_csv(RAW_SYNTHETIC, index=False)
        df = pd.read_csv(RAW_SYNTHETIC)

    init_schema()
    inserted = seed_readings(df)
    print(f"seeded {inserted} sensor readings into the database")


if __name__ == "__main__":
    main()
