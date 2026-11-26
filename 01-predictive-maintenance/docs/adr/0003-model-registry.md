# ADR-0003: Model registry without a registry server

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Teams often adopt MLflow (or a cloud model registry) as default. That is a
justified choice in a multi-team org, but it is a *service* with its own
storage, auth and operations burden. For this project's philosophy (low
maintenance cost), a registry server is overkill.

## Decision

Store a single versioned `models/registry.json` manifest alongside the artifact:

```json
{
  "name": "rul-tcn",
  "version": 1,
  "artifact": "models/rul_tcn.onnx",
  "input_name": "input",
  "output_name": "output",
  "window": 30,
  "sensor_columns": ["s_1", "..."],
  "sensor_mean": [99.1, "..."],
  "sensor_std": [3.2, "..."],
  "metrics": {"rmse_cycles": 18.2, "r2": 0.90},
  "trained_at": "..."
}
```

Serving loads the artifact *and* the full input contract (window length, sensor
order, per-sensor normalization statistics) from this one file, making the model
self-describing.

## Consequences

**Positive**
- Zero extra infrastructure; the manifest is human-readable and diffable in git.
- Schema travels with the model → train/serve consistency is enforced by construction.

**Negative**
- No multi-version history, staging promotion or team sharing out of the box.

## Alternatives considered

- **MLflow server in Compose:** adds a service + artifact store + UI to operate
  without adding value to a single-model slice.
- **S3/SageMaker registry:** requires a cloud account; out of scope.
