"""Document ingestion: file upload -> plain text for the LLM.

In production the "text" often comes from a scanner/OCR or a PDF pipeline; here
we accept plain text files (and try a naive PDF text extraction) so the
automation can be exercised without external OCR services.
"""
from __future__ import annotations

import io

import pdfminer.high_level

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return pdfminer.high_level.extract_text(io.BytesIO(content))
    if lower.endswith((".txt", ".md")):
        return content.decode("utf-8", errors="replace")
    raise ValueError(
        f"unsupported file type {lower!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )
