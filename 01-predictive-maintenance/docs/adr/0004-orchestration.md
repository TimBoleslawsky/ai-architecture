# ADR-0004: Orchestration without Airflow

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

The data pipeline (prepare → train → batch score) is small. Airflow/Prefect
bring DAG semantics, scheduling and retry logic — and a heavy footprint
(webserver, scheduler, workers, metadata DB) and daily maintenance.

## Decision

For the vertical slice, orchestrate with a plain in-process loop (the batch
worker `while True: score_fleet(); sleep(30)`) and sequential CLI steps for
prepare/train. Keep each step idempotent so re-running is safe.

## Consequences

**Positive**
- Two files instead of an orchestration platform; trivial to debug.
- No extra services in Compose beyond Postgres.

**Negative**
- No retries, backfills, backpressure or DAG visualization.
- Not suitable for long, interdependent pipelines.

## Alternatives considered

- **Airflow / Prefect in Compose:** correct for complex pipelines; out of scope here.
- **cron:** fine for fixed schedules but less flexible than the worker loop.

## Follow-up

When pipelines grow (multiple sources, retraining schedules, alerts), migrate
the batch worker into a Prefect/Airflow DAG *without changing the scoring code*
— the domain logic is already isolated in `src/serving/batch_scorer.py`.
