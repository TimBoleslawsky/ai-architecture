# ADR-0003: Structured output — prompt-to-JSON + Pydantic enforcement

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

LLM output is probabilistic free text. If the automation consumed it directly,
any hallucinated number could reach the ledger. We need a hard boundary between
"what the model says" and "what the system trusts".

## Decision

Two-layer enforcement:

1. **Prompt contract** — the system prompt instructs the model to return only
   JSON matching an exact schema (with `format=json` and `temperature=0`).
2. **Schema enforcement** — the raw JSON is parsed with `Invoice.model_validate`
   (Pydantic v2). If validation fails, the call is retried once with a longer
   timeout; if it fails again the request is rejected with a 422.

## Consequences

**Positive**
- The rest of the system sees only typed, validated objects — never raw tokens.
- One retry absorbs most transient schema slips from small models.
- Dates, floats and lists are coerced/validated deterministically.

**Negative**
- Adding a field means updating prompt + schema (two places, intentionally).
- Rejected outputs still cost the retry.

## Alternatives considered

- **Function/tool calling:** model-specific; the plain JSON contract works
  across Ollama models.
- **Trusting raw output:** rejected — unsafe by definition.
