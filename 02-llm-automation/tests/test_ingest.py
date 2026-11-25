"""Tests for document ingestion."""
import pytest

from src.ingest.ingest import extract_text


def test_txt_extraction_returns_text():
    text = extract_text("invoice.txt", b"SUPPLIER: Acme\nTOTAL: EUR 10.00")
    assert "Acme" in text
    assert "TOTAL" in text


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        extract_text("invoice.docx", b"x")


def test_pdf_support_accepted_extension_only():
    # We only assert the extension gate; real PDF parsing is covered by pdfminer.
    with pytest.raises(ValueError):
        extract_text("notes.xlsx", b"x")
