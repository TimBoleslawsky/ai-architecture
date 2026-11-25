# ADR-0001: Model choice — RandomForest over deep learning

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

The predictive-maintenance problem is tabular (sensor time-series aggregated
into rolling features), has only hundreds of rows per unit, and must be
re-trained and served cheaply. Deep time-series or sequence models (LSTM,
Transformer) are the fashionable default but carry real costs.

## Decision

Use a scikit-learn `RandomForestRegressor` (300 trees, `min_samples_leaf=3`) on
causally engineered rolling features.

## Consequences

**Positive**
- ~seconds to train on CPU; no GPU; ~MB model; trivially portable and serde-stable.
- No special infra; the same joblib artifact serves in a plain FastAPI process.
- Strong baseline for tabular data; easy to inspect feature importances.

**Negative**
- No explicit handling of long-range temporal structure (rolling window only).
- Memory grows with tree count; not a problem at this fleet scale.

## Trade-off note (philosophy)

This project deliberately optimises *maintenance cost, training cost and
computational efficiency* over squeezing the last point of accuracy — for this
problem class a deep model would need a GPU, more data, and ongoing upkeep for
little measured gain. Revisit if the data grows or accuracy saturates.

## Alternatives considered

- **XGBoost/LightGBM**: equivalent performance, extra native dependency; not needed.
- **LSTM/Transformer on raw sequences**: higher ceiling, much higher operational cost.
