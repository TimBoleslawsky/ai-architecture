# ADR-0004: Human-in-the-loop for approval

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Automation that touches money should not be fully autonomous by default. Some
invoices are ambiguous or large; some extractions will be wrong no matter how
good the guardrails are.

## Decision

Two-way split:

- **Auto-approved** only when the invoice passes all business rules **and** the
  total is at or below a configurable threshold. Approval posts the invoice to
  the ledger.
- **Everything else** (validation failure, over-threshold) lands in a review
  queue (`status = 'review'`) where a human explicitly approves or rejects.

## Consequences

**Positive**
- The common, safe case is fully automated; the risky case is visibly gated.
- Full audit trail: stored JSON, status transitions, reviewer, timestamps.

**Negative**
- Someone must watch the queue (acceptable and desirable for finance).

## Alternatives considered

- **Fully autonomous:** no gate on errors; rejected as unsafe.
- **All manual:** no automation; defeats the purpose.
