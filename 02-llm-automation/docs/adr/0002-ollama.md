# ADR-0002: Ollama for local model serving

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Once we chose a self-hosted open-source model (ADR-0001), we needed a runtime.
Options: Ollama, `llama.cpp` directly, or a Python framework (transformers/vLLM).

## Decision

Run the model in **Ollama** (`ollama/ollama` container) and talk to it over its
REST API (`/api/generate`, JSON mode). The model is pulled by tag
(`qwen2.5:1.5b-instruct`) and the runtime takes care of quantization and
prompt batching.

## Consequences

**Positive**
- One command to run; model store is a Docker volume; easy upgrades.
- Quantization built in → CPU-friendly inference for 1.5B models.
- JSON mode plus `temperature=0` help stabilise structured output.

**Negative**
- Another service in the stack (already handled by one Compose file).
- Less fine-grained control than raw `llama.cpp`.

## Alternatives considered

- **llama.cpp server:** more control, more integration work.
- **transformers/vLLM:** heavier footprint and (for vLLM) GPU expectations.
