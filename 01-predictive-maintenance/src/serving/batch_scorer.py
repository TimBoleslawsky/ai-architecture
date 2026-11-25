"""Scheduled batch scorer.

On every cycle it pulls the current sensor readings for *all* units from
Postgres, builds the same features used in training, predicts RUL and risk, and
upserts the fleet status into the ``predictions`` table. This is the offline
"run on the whole fleet" path (as opposed to the on-demand /predict endpoint).

Run with:  uv run python -m src.serving.batch_scorer
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd

from src.db import connect, init_schema
from src.serving.predictor import Predictor

SCORE_INTERVAL_S = float(os.environ.get("SCORE_INTERVAL_S", 30))


def score_fleet(predictor: Predictor) -> int:
    sensor_cols = predictor.sensor_columns
    col_sql = ", ".join(sensor_cols)
    upsert = """
        INSERT INTO predictions (unit, predicted_rul, risk)
        VALUES %s, %s, %s
        ON CONFLICT (unit) DO UPDATE
        SET predicted_rul = EXCLUDED.predicted_rul,
            risk          = EXCLUDED.risk,
            scored_at     = now()
    """
    scored = 0
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT unit FROM sensor_readings ORDER BY unit")
            units = [r[0] for r in cur.fetchall()]

            for unit in units:
                cur.execute(
                    f"SELECT cycle, {col_sql} FROM sensor_readings "
                    "WHERE unit = %s ORDER BY cycle",
                    (unit,),
                )
                cols = ["cycle"] + sensor_cols
                df = pd.DataFrame(cur.fetchall(), columns=cols)
                if df.empty:
                    continue
                features = predictor.features_from_rows(df.to_dict("records"))
                rul, risk = predictor.predict(features.iloc[-1])
                cur.execute(upsert, (unit, float(rul), risk))
                scored += 1
        conn.commit()
    return scored


def main() -> None:
    init_schema()
    predictor = Predictor()
    print(f"batch scorer online (interval={SCORE_INTERVAL_S}s)")
    while True:
        try:
            started = time.monotonic()
            n = score_fleet(predictor)
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] scored {n} units in "
                f"{time.monotonic() - started:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"scoring cycle failed: {exc}")
        time.sleep(SCORE_INTERVAL_S)


if __name__ == "__main__":
    main()
