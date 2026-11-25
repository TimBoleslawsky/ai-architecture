# ADR-0003: Self-describing registry + prediction telemetry

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

A model is a product: it has a version, an input contract, and a runtime
behavior that must be observable. Without an explicit contract, serving code
drifts from training code and nobody can tell which model is live.

## Decision

- **Self-describing registry:** `models/registry.json` stores the artifact
  path, class names, input/output tensor names, the exact preprocessing
  parameters (size, mean, std) and the evaluation metrics. The serving predictor
  loads everything it needs from this one file.
- **Prediction telemetry:** every inference is best-effort logged to Postgres
  (`predicted_class`, `confidence`, full probability vector) when
  `DATABASE_URL` is set. Logging failures never affect the response.

## Consequences

**Positive**
- Version, contract and metrics travel with the artifact → train/serve parity
  is enforced by construction and auditable at a glance.
- Telemetry is the raw material for drift monitoring and retraining decisions.

**Negative**
- Two artifacts (`.onnx` + `.json`) must be shipped together (handled by the
  same `models/` directory and Docker build context).

## Alternatives considered

- **External registry (MLflow/S3):** adds a service; overkill for a slice.
- **No telemetry:** the model is a black box in production; rejected.
