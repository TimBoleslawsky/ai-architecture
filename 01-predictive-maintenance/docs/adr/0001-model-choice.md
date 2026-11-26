# ADR-0001: Model choice — efficacy-first TCN over an efficiency-first tabular model

- **Status:** Accepted
- **Date:** 2026-11-26

## Context

RUL prediction from sensor time-series is a *sequence* problem. In the taxonomy
of this repo, predictive maintenance is a **bucket-1 (analytics)** workload:
the model runs offline in the cloud, so the cost driver is data quality and the
data→decision bridge — **not** computational efficiency.

That flips the usual model-choice logic. A hand-crafted tabular feature set
(rolling window statistics) plus a RandomForest is the *computationally cheap*
answer, but it is the wrong trade for bucket 1: it caps efficacy at the quality
of hand-engineered features and treats efficiency (which we don't need here) as
if it mattered.

## Decision

Use a **TCN** (temporal convolutional network) on raw per-unit sensor windows as
the primary model, trained in a small Lightning/OmegaConf engine adapted from
the `neural-compression` framework. Keep a **RandomForest on rolling features as
an explicit baseline** so the efficacy/efficiency trade-off is measured, not
asserted.

## Consequences

**Positive**
- Matches the bucket's requirement: we optimise efficacy because we can afford to
  (offline, cloud). The sequence model can learn temporal structure that
  hand-crafted statistics cannot express.
- The engine is reusable and faithful to a proven config/factory/runner pattern.
- ONNX export keeps serving dependency-light (see ADR-0002 for the serving side).

**Negative**
- Higher training cost and more hyperparameters than a tabular baseline.
- On the current synthetic data the TCN *underperforms* the RandomForest
  baseline — tuning the architecture/data is the expected next step; the
  comparison harness exists precisely to surface this.

## Trade-off note (philosophy)

Contrast with bucket 3 (AI as a product), where computational efficiency is a
hard constraint and we use a *small* CNN (see `03-quality-inspection`). Here we
are free to trade compute for efficacy. The baseline harness keeps that
decision honest and auditable.

## Alternatives considered

- **RandomForest/LightGBM on rolling features:** cheap, good, but caps efficacy;
  kept only as the baseline.
- **LSTM/GRU on raw windows:** equivalent class of approach; the TCN trains
  faster and is the architecture already proven in the `neural-compression`
  framework.
- **Transformer/attention:** higher ceiling, higher cost; not needed at this scale.
