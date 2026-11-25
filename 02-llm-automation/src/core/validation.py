"""Business-rule validation for extracted invoices.

The LLM does the *unstructured* reading; deterministic code does the
*verification*. Totals are recomputed from line items, required fields are
checked, and the invoice is flagged for human review when anything is off.

Validating in code (not in the prompt) is what makes the automation safe: no
hallucinated number can be posted to the ledger without crossing these checks.
"""
from __future__ import annotations

import math

from pydantic import BaseModel

from src.core.schema import Invoice

REL_TOLERANCE = 0.01  # 1% tolerance on monetary recomputation


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]


def validate_invoice(inv: Invoice, amount_threshold: float = 1000.0) -> ValidationResult:
    errors: list[str] = []

    if not inv.vendor:
        errors.append("missing vendor")
    if not inv.invoice_number:
        errors.append("missing invoice number")
    if not inv.issue_date:
        errors.append("missing issue date")
    if inv.total <= 0:
        errors.append("total must be positive")

    if inv.line_items:
        computed = sum(item.amount for item in inv.line_items)
        if not math.isclose(computed, inv.subtotal, rel_tol=REL_TOLERANCE, abs_tol=0.01):
            errors.append(
                f"line items sum to {computed:.2f} but subtotal is {inv.subtotal:.2f}"
            )

    if inv.subtotal and inv.tax_rate:
        expected_total = inv.subtotal * (1 + inv.tax_rate / 100)
        if not math.isclose(expected_total, inv.total, rel_tol=REL_TOLERANCE, abs_tol=0.01):
            errors.append(
                f"total {inv.total:.2f} does not match subtotal+tax {expected_total:.2f}"
            )

    if inv.total > amount_threshold:
        errors.append(f"amount exceeds review threshold of {amount_threshold:g}")

    return ValidationResult(valid=not errors, errors=errors)


def decide_status(inv: Invoice, amount_threshold: float = 1000.0) -> str:
    """An invoice is *approved* only if it validates AND is under the threshold.

    Anything else goes to the human review queue.
    """
    result = validate_invoice(inv, amount_threshold)
    return "approved" if result.valid else "review"
