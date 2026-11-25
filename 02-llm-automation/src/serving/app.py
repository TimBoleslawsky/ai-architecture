"""Invoice automation API.

Flow per document:

    upload -> text -> LLM extraction (JSON) -> Pydantic schema -> business
    validation -> auto-approve & post to ledger, OR send to human review.

The automation contrast (vs. Project 1) is that this pipeline *takes an
action* (posts to the ledger) instead of just predicting something.

Run with:  uv run uvicorn src.serving.app:app --reload
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError

from src.core.llm import OllamaError, extract_invoice_json
from src.core.schema import Invoice
from src.core.storage import (
    get_invoice,
    init_schema,
    insert_invoice,
    list_invoices,
    post_to_ledger,
    set_status,
)
from src.core.validation import decide_status
from src.ingest.ingest import extract_text

AMOUNT_THRESHOLD = float(os.environ.get("AMOUNT_THRESHOLD", 1000))

app = FastAPI(
    title="Invoice Automation API",
    description="Open-source LLM based invoice processing with human-in-the-loop.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    init_schema()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/invoices/process")
async def process_invoice(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        text = extract_text(file.filename or "invoice.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        raw = extract_invoice_json(text)
    except OllamaError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM extraction failed - is Ollama running? {exc}",
        ) from exc

    # Schema enforcement: retry once if the LLM output does not validate.
    invoice: Invoice
    for attempt in range(2):
        try:
            invoice = Invoice.model_validate(raw)
            break
        except ValidationError:
            if attempt == 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"LLM output did not match the invoice schema: {raw}",
                )
            raw = extract_invoice_json(text, timeout_s=180)

    status = decide_status(invoice, AMOUNT_THRESHOLD)
    invoice_id = insert_invoice(
        invoice_number=invoice.invoice_number,
        vendor=invoice.vendor,
        currency=invoice.currency,
        total=invoice.total,
        status=status,
        json_data=invoice.model_dump(mode="json"),
    )

    if status == "approved":
        post_to_ledger(invoice_id)

    return {
        "id": invoice_id,
        "status": status,
        "invoice_number": invoice.invoice_number,
        "vendor": invoice.vendor,
        "total": invoice.total,
        "currency": invoice.currency,
    }


@app.get("/invoices")
def list_review_queue(status: str | None = None) -> list[dict]:
    return list_invoices(status)


@app.get("/invoices/{invoice_id}")
def invoice_detail(invoice_id: int) -> dict:
    inv = get_invoice(invoice_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invoice not found")
    return inv


@app.post("/invoices/{invoice_id}/approve")
def approve_invoice(invoice_id: int) -> dict:
    inv = get_invoice(invoice_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invoice not found")
    if not set_status(invoice_id, "approved"):
        raise HTTPException(status_code=500, detail="could not update invoice")
    post_to_ledger(invoice_id)
    return {"id": invoice_id, "status": "approved"}


@app.post("/invoices/{invoice_id}/reject")
def reject_invoice(invoice_id: int) -> dict:
    inv = get_invoice(invoice_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invoice not found")
    if not set_status(invoice_id, "rejected"):
        raise HTTPException(status_code=500, detail="could not update invoice")
    return {"id": invoice_id, "status": "rejected"}
