"""
FRED API client for US Treasury yield data.

Provides:
  get_yield_curve()     — latest on-the-run Treasury yields (5 maturities)
  fetch_rate_on_date()  — yield for a specific series on a specific date
                          (used by the ingest CLI for historical backfill)

FRED series used
----------------
DGS3MO   3-month constant maturity
DGS2     2-year  constant maturity
DGS5     5-year  constant maturity
DGS7     7-year  constant maturity
DGS10    10-year constant maturity
DGS30    30-year constant maturity

All rates are returned as decimals (e.g. 0.045 for 4.5%).
FRED_API_KEY must be set in the environment (or a .env file).
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY: Optional[str] = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

TREASURY_SERIES: dict[str, str] = {
    "3M":  "DGS3MO",
    "2Y":  "DGS2",
    "5Y":  "DGS5",
    "7Y":  "DGS7",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

# In-process cache keyed by series_id
_cache: dict[str, tuple[Optional[float], float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _fetch_series(series_id: str, observation_date: Optional[date] = None) -> Optional[float]:
    """Fetch a single FRED observation.

    Args:
        series_id:        FRED series ID.
        observation_date: If given, fetch the value on or just before that date.
                          If None, fetch the most recent observation.

    Returns:
        Rate as a decimal, or None if unavailable / no API key.

    Raises:
        RuntimeError: On non-200 FRED API response.
    """
    if not FRED_API_KEY:
        return None

    params: dict = {
        "series_id": series_id,
        "api_key":   FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }

    if observation_date is not None:
        params["observation_end"] = observation_date.isoformat()

    response = requests.get(FRED_BASE_URL, params=params, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(
            f"FRED API error {response.status_code} for series {series_id}"
        )

    for obs in response.json().get("observations", []):
        if obs["value"] != ".":
            return float(obs["value"]) / 100.0

    return None


def fetch_latest_rate(series_id: str) -> Optional[float]:
    """Fetch the most recent rate for a FRED series, with 5-minute caching.

    Args:
        series_id: FRED series identifier (e.g. "DGS10").

    Returns:
        Rate as a decimal, or None if unavailable.
    """
    now = time.monotonic()
    cached_value, cached_at = _cache.get(series_id, (None, 0.0))
    if now - cached_at < _CACHE_TTL_SECONDS and cached_value is not None:
        return cached_value

    rate = _fetch_series(series_id)
    if rate is not None:
        _cache[series_id] = (rate, now)
    return rate


def fetch_rate_on_date(series_id: str, observation_date: date) -> Optional[float]:
    """Fetch the FRED rate on or just before a specific date.

    Used by the ingest CLI for historical backfill.

    Args:
        series_id:        FRED series ID (e.g. "DGS10").
        observation_date: The date to look up.

    Returns:
        Rate as a decimal, or None if unavailable.
    """
    return _fetch_series(series_id, observation_date=observation_date)


def get_yield_curve(as_of: Optional[date] = None) -> dict[str, float]:
    """Fetch the US Treasury yield curve from FRED.

    Args:
        as_of: If given, fetch yields as of that date. If None, fetch latest.

    Returns:
        Dict mapping maturity label → decimal yield.
        Maturities with missing data are omitted.
    """
    curve: dict[str, float] = {}
    for maturity, series_id in TREASURY_SERIES.items():
        if as_of is not None:
            rate = fetch_rate_on_date(series_id, as_of)
        else:
            rate = fetch_latest_rate(series_id)
        if rate is not None:
            curve[maturity] = rate
    return curve