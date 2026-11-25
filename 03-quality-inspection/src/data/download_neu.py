"""Optional downloader for the real NEU surface-defect dataset.

The NEU dataset (NEU-CLS, ~1.8k 200x200 grayscale images, 6 classes) is hosted
on an unstable university page. Rather than hard-coding flaky URLs, this script
downloads a zip from the ``NEU_ZIP_URL`` environment variable (set it to any
mirror you trust) and extracts it to ``data/raw/neu``. The pipeline runs fine
without it: ``prepare`` falls back to the synthetic generator.

Run with:  $env:NEU_ZIP_URL="<zip url>"; uv run python -m src.data.download_neu
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import httpx

OUT = Path("data/raw/neu")
NEU_ZIP_URL = os.environ.get("NEU_ZIP_URL", "")


def main() -> int:
    if not NEU_ZIP_URL:
        print(
            "NEU_ZIP_URL not set - skipping. The pipeline will use the synthetic "
            "dataset instead (see docs/adr/0004-data-strategy.md)."
        )
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "neu.zip"
    print(f"downloading NEU dataset from {NEU_ZIP_URL} ...")
    with httpx.stream("GET", NEU_ZIP_URL, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)

    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(OUT)
    tmp.unlink()

    n_images = len(list(OUT.rglob("*.jpg"))) + len(list(OUT.rglob("*.bmp")))
    print(f"extracted NEU dataset -> {OUT} ({n_images} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
