"""Structured output schema for extracted invoice data.

The LLM is asked to return JSON that fits this model; Pydantic then *enforces*
the shape, so the rest of the system never has to defend against free-form
LLM output. Dates arrive as ISO strings and are validated here.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = ""
    quantity: float = 1
    unit_price: float = 0
    amount: float = 0


class Invoice(BaseModel):
    invoice_number: str = ""
    vendor: str = ""
    vendor_tax_id: str = ""
    issue_date: date | None = None
    due_date: date | None = None
    currency: str = "EUR"
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float = 0
    tax_rate: float = 0
    tax_amount: float = 0
    total: float = 0
    bank_iban: str = ""


class ExtractionResult(BaseModel):
    """Wrapper returned by the extraction step (JSON from the LLM)."""

    invoice: Invoice
    raw: dict
