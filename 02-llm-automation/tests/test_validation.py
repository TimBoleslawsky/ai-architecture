"""Tests for business-rule validation (pure logic, no LLM, no DB)."""
from datetime import date

import pytest

from src.core.schema import Invoice, LineItem
from src.core.validation import decide_status, validate_invoice


def make_invoice(**overrides) -> Invoice:
    base = {
        "invoice_number": "INV-1",
        "vendor": "Acme GmbH",
        "issue_date": "2026-11-10",
        "currency": "EUR",
        "line_items": [
            {"description": "bolts", "quantity": 50, "unit_price": 1.20, "amount": 60.00},
            {"description": "plates", "quantity": 10, "unit_price": 24.50, "amount": 245.00},
        ],
        "subtotal": 305.00,
        "tax_rate": 19.0,
        "tax_amount": 57.95,
        "total": 362.95,
    }
    base.update(overrides)
    return Invoice.model_validate(base)


def test_valid_invoice_passes():
    inv = make_invoice()
    res = validate_invoice(inv, amount_threshold=1000)
    assert res.valid is True
    assert res.errors == []


def test_missing_vendor_fails():
    res = validate_invoice(make_invoice(vendor=""), amount_threshold=1000)
    assert res.valid is False
    assert "missing vendor" in res.errors


def test_missing_issue_date_fails():
    res = validate_invoice(make_invoice(issue_date=None), amount_threshold=1000)
    assert res.valid is False
    assert "missing issue date" in res.errors


def test_line_items_must_match_subtotal():
    inv = make_invoice(line_items=[
        {"description": "a", "quantity": 1, "unit_price": 10, "amount": 10.00}
    ], subtotal=500.00, total=595.00, tax_rate=19.0, tax_amount=95.00)
    res = validate_invoice(inv, amount_threshold=1000)
    assert res.valid is False
    assert any("line items" in e for e in res.errors)


def test_total_must_match_tax_calc():
    inv = make_invoice(subtotal=305.00, tax_rate=19.0, total=999.00)
    res = validate_invoice(inv, amount_threshold=1000)
    assert res.valid is False
    assert any("does not match" in e for e in res.errors)


def test_amount_over_threshold_goes_to_review_even_if_valid():
    inv = make_invoice(total=5000.00, subtotal=4201.68, tax_rate=19.0, tax_amount=798.32)
    assert validate_invoice(inv, amount_threshold=1000).valid is False
    assert decide_status(inv, amount_threshold=1000) == "review"


def test_decide_status_approved_when_valid_and_under_threshold():
    assert decide_status(make_invoice(), amount_threshold=1000) == "approved"


def test_negative_total_fails():
    res = validate_invoice(make_invoice(total=-5), amount_threshold=1000)
    assert res.valid is False
