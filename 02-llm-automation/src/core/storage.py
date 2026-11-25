"""Postgres storage for invoices, the review queue and the ledger.

Three tables reflect the workflow: invoices are stored as they arrive
(``invoices``), approved ones are *posted* to a ledger (``ledger``), and the
review queue is just ``invoices WHERE status = 'review'``.
"""
from __future__ import annotations

import json
import os

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://invoice:invoice@localhost:5432/invoice"
)

SCHEMA_SQL = """
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
"""


def connect():
    return psycopg2.connect(DATABASE_URL)


def init_schema() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


def insert_invoice(invoice_number: str, vendor: str, currency: str, total: float,
                   status: str, json_data: dict) -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO invoices (invoice_number, vendor, currency, total, status, json_data) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (invoice_number, vendor, currency, total, status, json.dumps(json_data)),
            )
            invoice_id = cur.fetchone()[0]
        conn.commit()
    return invoice_id


def get_invoice(invoice_id: int) -> dict | None:
    with connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM invoices WHERE id = %s", (invoice_id,))
            row = cur.fetchone()
    if row is None:
        return None
    row["json_data"] = json.loads(row["json_data"])
    return dict(row)


def list_invoices(status: str | None = None) -> list[dict]:
    query = "SELECT id, invoice_number, vendor, currency, total, status, extracted_at FROM invoices"
    params: tuple = ()
    if status:
        query += " WHERE status = %s"
        params = (status,)
    query += " ORDER BY extracted_at DESC"
    with connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]


def set_status(invoice_id: int, status: str, reviewer: str = "human") -> bool:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE invoices SET status = %s, reviewed_by = %s, reviewed_at = now() "
                "WHERE id = %s RETURNING id",
                (status, reviewer, invoice_id),
            )
            updated = cur.fetchone() is not None
        conn.commit()
    return updated


def post_to_ledger(invoice_id: int) -> bool:
    """Record the *action* the automation performs on approval."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ledger (invoice_id) VALUES (%s) RETURNING id",
                (invoice_id,),
            )
            posted = cur.fetchone() is not None
        conn.commit()
    return posted
