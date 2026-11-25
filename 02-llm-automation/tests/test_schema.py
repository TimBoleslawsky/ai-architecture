"""Tests for the structured-output schema enforcement."""
from datetime import date

import pytest
from pydantic import ValidationError

from src.core.schema import Invoice


def test_valid_json_becomes_typed_model():
    payload = {
        "invoice_number": "INV-99",
        "vendor": "Lager GmbH",
        "issue_date": "2026-11-10",
        "due_date": "2026-12-10",
        "currency": "EUR",
        "line_items": [{"description": "x", "quantity": 2, "unit_price": 5, "amount": 10}],
        "subtotal": 10.0,
        "tax_rate": 19.0,
        "tax_amount": 1.9,
        "total": 11.9,
    }
    inv = Invoice.model_validate(payload)
    assert inv.issue_date == date(2026, 11, 10)
    assert inv.line_items[0].amount == 10


def test_amounts_must_be_numeric_not_strings():
    # Pydantic coerces "12.50" to 12.5 (handy for LLM output)...
    inv = Invoice.model_validate(
        {"invoice_number": "X", "vendor": "Y", "subtotal": "12.50", "total": "12.50"}
    )
    assert inv.subtotal == 12.5
    # ...but genuinely non-numeric strings must be rejected.
    payload = {"invoice_number": "X", "vendor": "Y", "subtotal": "twelve", "total": "twelve"}
    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


def test_bad_date_fails():
    payload = {"invoice_number": "X", "vendor": "Y", "issue_date": "not-a-date"}
    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


def test_defaults_applied_for_missing_fields():
    inv = Invoice.model_validate({"invoice_number": "", "vendor": "", "total": 0})
    assert inv.currency == "EUR"
    assert inv.line_items == []
