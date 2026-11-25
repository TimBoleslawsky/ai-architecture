"""Optional downloader for the real NASA C-MAPSS turbofan dataset.

The dataset is free but hosted on unstable links, so this script tries a
configurable URL (``CMAPSS_URL`` env var) followed by a couple of community
mirrors of the FD001 train/test CSVs. If every attempt fails it exits non-zero
and the pipeline transparently falls back to the synthetic generator.

Run with:  uv run python -m src.data.download_cmapss
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/cmapss")
TRAIN_FILE = RAW_DIR / "FD001_train.csv"
TEST_FILE = RAW_DIR / "FD001_test.csv"

# (kind, url) candidates. Mirrors publish the CSV with the same column layout:
# unit, cycle, op_1..3, s_1..21, rul (train only).
CANDIDATES = [
    ("train", os.environ.get("CMAPSS_URL")),
    ("train", "https://raw.githubusercontent.com/ferhat00/CMAPSS/master/FD001_train.csv"),
    ("train", "https://raw.githubusercontent.com/mbhattasaraiya/c-mapss/master/FD001_train.csv"),
    ("test", "https://raw.githubusercontent.com/ferhat00/CMAPSS/master/FD001_test.csv"),
    ("test", "https://raw.githubusercontent.com/mbhattasaraiya/c-mapss/master/FD001_test.csv"),
]

COLUMNS = ["unit", "cycle", "op_1", "op_2", "op_3"] + [f"s_{i}" for i in range(1, 22)] + ["rul"]


def load_fd001_csv(url: str) -> pd.DataFrame:
    """Read a C-MAPSS FD001 CSV from a URL (headerless, space separated)."""
    return pd.read_csv(url, sep=r"\s+", header=None, names=COLUMNS, engine="python")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_FILE.exists() and TEST_FILE.exists():
        print("C-MAPSS data already present, skipping download.")
        return 0

    train_urls = [u for kind, u in CANDIDATES if kind == "train" and u]
    test_urls = [u for kind, u in CANDIDATES if kind == "test" and u]

    for url in train_urls:
        try:
            df = load_fd001_csv(url)
            df.to_csv(TRAIN_FILE, index=False)
            print(f"downloaded train set from {url}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"train download failed ({url}): {exc}")
    else:
        print("no train URL worked; pipeline will fall back to synthetic data")
        return 1

    for url in test_urls:
        try:
            df = load_fd001_csv(url)
            # C-MAPSS test file has no RUL column; emit it separately.
            df.to_csv(TEST_FILE, index=False)
            df[["unit", "rul"]].groupby("unit").min().to_csv(
                RAW_DIR / "FD001_RUL.csv"
            )
            print(f"downloaded test set from {url}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"test download failed ({url}): {exc}")

    print("test set download failed; pipeline will fall back to synthetic data")
    return 1


if __name__ == "__main__":
    sys.exit(main())
