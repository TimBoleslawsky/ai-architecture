# ADR-0004: Data strategy — synthetic default, real dataset optional

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

The real NEU steel-surface-defect dataset is the canonical source but is hosted
on an unstable university page. Relying on it would make the showcase
non-reproducible and not runnable offline.

## Decision

- **Default:** a procedural generator (`src/data/synthetic.py`) produces 200x200
  grayscale images mimicking all six NEU classes, with a fixed seed.
- **Optional:** `download_neu.py` fetches the real dataset from a
  user-supplied `NEU_ZIP_URL` mirror and the pipeline picks it up transparently
  (the dataset loader accepts png/jpg/bmp class folders).

## Consequences

**Positive**
- Fully reproducible and offline; training works identically either way.
- The generator is a stand-in that keeps the *pipeline*, not the *pixels*, as
  the thing under study.

**Negative**
- Synthetic defects are easier than real production surfaces → accuracy on
  synthetic data overstates real-world performance. The registry/metrics make
  this visible, and the NEU swap is the validation step.

## Alternatives considered

- **Bundling the NEU dataset in git:** bloats the repo and has unclear licensing.
- **Hard-coded download URLs:** known to rot; rejected.
