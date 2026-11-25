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
