# ADR-0001: Open-source LLM over proprietary hosted API

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

The extraction step needs an LLM. The default in many teams is a proprietary
hosted API (OpenAI, Anthropic, ...). For this automation the extraction task is
narrow (invoice JSON), the document volume is modest, and the data is
financially sensitive.

## Decision

Use a **small open-source LLM** (default `qwen2.5:1.5b-instruct`) served
locally. No proprietary API anywhere in the stack.

## Consequences

**Positive**
- Zero per-call cost and no vendor lock-in; predictable total cost of ownership.
- Privacy: document contents never leave the machine.
- Small models run on CPU and are re-startable at any time.

**Negative**
- Lower raw extraction quality than frontier models → mitigated by schema
  enforcement, a retry loop and a human review queue (see ADR-0003/0004).
- The team owns the runtime (see ADR-0002).

## Trade-off note (philosophy)

This project optimises for **maintenance cost, training cost and computational
efficiency**, and for staying in control of sensitive data. A frontier API is
objectively more capable; it is not the right trade for this class of workload.

## Alternatives considered

- **Hosted GPT/Claude API:** best-in-class accuracy, per-token cost, data leaves
  the perimeter.
- **Large open-source model (e.g. 70B):** higher quality, but GPU requirements
  and power costs that dwarf the accuracy gain for this narrow task.
