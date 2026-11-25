# ADR-0002: Serving strategy — online API + scheduled batch

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Predictions are consumed in two genuinely different ways:

1. **On-demand:** "How is unit 42 right now?" — low latency, ad-hoc, interactive.
2. **Fleet-wide:** "Which units need attention today?" — whole fleet, periodic,
   results must be queryable/auditable.

A single "batch-only" or "online-only" design would serve only one real
consumption pattern.

## Decision

Provide both:

- `serve` — FastAPI `POST /predict` (on-demand), stateless, loads the model once.
- `worker` — batch scorer that pulls all units from Postgres, scores them, and
  upserts fleet status into `predictions` on a 30 s loop.

Both use the same `Predictor` and therefore the same feature code and model.

## Consequences

**Positive**
- Covers interactive and scheduled analytics; results are in a queryable table.
- No duplication: one predictor class powers both.

**Negative**
- Two processes to operate (already handled by one Compose stack).
- Batch latency depends on fleet size; acceptable at this scale.

## Alternatives considered

- **Online-only:** no fleet-wide picture, no audit trail.
- **Batch-only:** poor interactive experience for ad-hoc queries.
