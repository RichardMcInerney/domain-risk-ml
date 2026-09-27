"""
Dataset acquisition utilities for Zaxonite Domain Risk ML.

This module downloads approved source datasets used by the project while
preserving the original source files in data/raw/.

Raw source data must not be modified in place.
"""

from pathlib import Path

import os
import urllib.request
import ssl
import certifi

from dotenv import load_dotenv
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

URLHAUS_AUTH_KEY = os.getenv("URLHAUS_AUTH_KEY")


def require_urlhaus_auth_key() -> str:
    """
    Return the configured URLhaus Auth-Key.

    Raises an error if the local environment has not been configured.
    """

    if not URLHAUS_AUTH_KEY:
        raise RuntimeError(
            "URLHAUS_AUTH_KEY is not configured. "
            "Create a local .env file using .env.example as the template."
        )

    return URLHAUS_AUTH_KEY

TRANCO_LATEST_URL = "https://tranco-list.eu/top-1m.csv.zip"


def download_urlhaus_recent() -> Path:
    """
    Download the URLhaus recent malware-URL CSV dataset.

    The downloaded CSV file is preserved unchanged in data/raw/.
    """

    auth_key = require_urlhaus_auth_key()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    destination = RAW_DATA_DIR / f"urlhaus_full_{timestamp}.csv.zip"

    if destination.exists():
        print(f"URLhaus dataset already exists: {destination}")
        return destination

    url = (
        "https://urlhaus-api.abuse.ch/v2/files/exports/"
        f"{auth_key}/full.csv.zip"
    )

    print("Downloading URLhaus full database dump...")

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Zaxonite-Domain-Risk-ML/0.1"},
    )

    try:
        with urllib.request.urlopen(request, context=ssl_context) as response:
            destination.write_bytes(response.read())
    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    print(f"Saved raw dataset to: {destination}")
    return destination


def download_tranco() -> Path:
    """
    Download the latest standard Tranco top-one-million domain list.

    The downloaded ZIP file is preserved unchanged in data/raw/.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    destination = RAW_DATA_DIR / f"tranco_top_1m_{timestamp}.csv.zip"

    if destination.exists():
        print(f"Tranco dataset already exists: {destination}")
        return destination

    print("Downloading latest Tranco list...")
    urllib.request.urlretrieve(TRANCO_LATEST_URL, destination)

    print(f"Saved raw dataset to: {destination}")
    return destination


if __name__ == "__main__":
    download_urlhaus_recent()