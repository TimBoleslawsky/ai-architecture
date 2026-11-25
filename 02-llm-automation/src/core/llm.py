"""Small, open-source LLM integration via Ollama.

Philosophy (see ADR-0001/0002): we deliberately use a *small* quantized
open-source model served locally (Ollama) instead of a proprietary hosted API.
The model only has to do one job well: extract structured JSON from an invoice
text. A 1.5B-3B parameter model is cheap to run, private, and fully
self-hosted.
"""
from __future__ import annotations

import json
import os

import httpx

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")

EXTRACT_SYSTEM_PROMPT = """\
You extract structured data from supplier invoices.
Return ONLY valid JSON matching exactly this schema (no markdown, no prose):

{
  "invoice_number": string,
  "vendor": string,
  "vendor_tax_id": string,
  "issue_date": "YYYY-MM-DD" or null,
  "due_date": "YYYY-MM-DD" or null,
  "currency": ISO 4217 code (default EUR),
  "line_items": [{"description": string, "quantity": number, "unit_price": number, "amount": number}],
  "subtotal": number,
  "tax_rate": number (percent),
  "tax_amount": number,
  "total": number,
  "bank_iban": string
}

Rules:
- Amounts are numeric, not strings. No thousand separators.
- If a field is not present in the document, use the default (empty string, 0, or null).
- Derive subtotal, tax_amount and total from the document; do not invent them.
"""


class OllamaError(RuntimeError):
    """Raised when the Ollama runtime cannot complete a request."""


def extract_invoice_json(
    text: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 120.0,
) -> dict:
    """Call the local LLM and return the raw extracted JSON dict."""
    model = model or OLLAMA_MODEL
    base_url = base_url or OLLAMA_BASE_URL

    prompt = f"{EXTRACT_SYSTEM_PROMPT}\n\nINVOICE TEXT:\n```\n{text}\n```"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = httpx.post(
            f"{base_url}/api/generate", json=payload, timeout=timeout_s
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Ollama request failed ({base_url}, model={model}): {exc}"
        ) from exc

    body = resp.json()
    raw = body.get("response", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"LLM returned invalid JSON: {raw[:200]!r}") from exc
