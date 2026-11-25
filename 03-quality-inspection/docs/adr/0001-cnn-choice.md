# ADR-0001: Small CNN from scratch over a pretrained backbone

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Image-classification tasks are commonly solved by fine-tuning a large
pretrained backbone (ResNet/EfficientNet/ImageNet checkpoints). That yields
great accuracy but pulls in big models, large dependencies and slow,
GPU-bound training.

## Decision

Train a **small convolutional network from scratch** (~60k parameters, three
conv+BN blocks + one linear layer) on 96x96 grayscale images.

## Consequences

**Positive**
- Trains to ~99% validation accuracy on the (synthetic/NEU) defect task in well
  under a minute on CPU.
- Tiny artifact, tiny memory footprint, trivial to re-train frequently.
- No pretrained weights → no license or distribution concerns, fully auditable.

**Negative**
- Lower ceiling than a large backbone on *hard* visual tasks; not the right
  choice for every problem.
- Needs a dataset big enough to learn from scratch — we have it.

## Trade-off note (philosophy)

This project optimises for **training cost, computational efficiency and
maintenance cost**. A frozen/backbone approach would add a 20-100 MB dependency
and GPU expectations for an accuracy delta that the business case does not need.

## Alternatives considered

- **Fine-tuned ResNet18/EfficientNet:** higher raw accuracy, heavier training +
  serving, larger attack surface.
- **Large ViT:** out of proportion for a 6-class grayscale QC task.
