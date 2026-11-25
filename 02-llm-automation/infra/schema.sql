CREATE TABLE IF NOT EXISTS invoices (
    id             SERIAL PRIMARY KEY,
    invoice_number TEXT,
    vendor         TEXT,
    currency       TEXT,
    total          NUMERIC(14,2),
    status         TEXT NOT NULL DEFAULT 'review',
    json_data      JSONB NOT NULL,
    extracted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_by    TEXT,
    reviewed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ledger (
    id          SERIAL PRIMARY KEY,
    invoice_id  INTEGER NOT NULL REFERENCES invoices(id),
    posted_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
