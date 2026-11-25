"""Database helpers for Postgres-backed components.

Used by the seed script, the batch scorer and (optionally) the API to persist
incoming sensor readings and predictions.
"""
from __future__ import annotations

import os

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://pm:pm@localhost:5432/pm"
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id       SERIAL PRIMARY KEY,
    unit     INTEGER NOT NULL,
    cycle    INTEGER NOT NULL,
    rul      INTEGER,
    ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (unit, cycle)
);

CREATE TABLE IF NOT EXISTS predictions (
    id            SERIAL PRIMARY KEY,
    unit          INTEGER NOT NULL UNIQUE,
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    predicted_rul FLOAT NOT NULL,
    risk          TEXT NOT NULL
);
"""


def connect():
    return psycopg2.connect(DATABASE_URL)


def init_schema() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


def seed_readings(df, chunk_size: int = 2000) -> int:
    """Upsert a sensor dataframe (unit, cycle, s_1.., rul) into the DB."""
    sensor_cols = [c for c in df.columns if c.startswith("s_")]
    cols = ["unit", "cycle"] + sensor_cols + ["rul"]
    rows = [tuple(r) for r in df[cols].to_records(index=False)]
    upsert = f"""
        INSERT INTO sensor_readings (unit, cycle, {", ".join(sensor_cols)}, rul)
        VALUES %s
        ON CONFLICT (unit, cycle) DO NOTHING
    """
    inserted = 0
    with connect() as conn:
        with conn.cursor() as cur:
            for chunk in range(0, len(rows), chunk_size):
                execute_values(cur, upsert, rows[chunk : chunk + chunk_size])
                inserted += cur.rowcount
        conn.commit()
    return inserted
