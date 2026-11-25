CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    predicted_class  TEXT NOT NULL,
    confidence       REAL NOT NULL,
    probabilities    JSONB NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
