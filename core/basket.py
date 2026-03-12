"""
Deliverable basket for CME 10-Year Treasury Note futures (TY).

Contract: TYM26 (June 2026)
Delivery date: ~2026-06-30
Eligibility: remaining maturity >= 6.5 years and <= 10 years on first day of
             delivery month (2026-06-01), so bonds maturing Dec 2032 – Jun 2036.

Data source: Treasury Direct public API (no key required).
Falls back to hardcoded snapshot if the API is unreachable.

Each bond is a plain dict with keys:
    cusip        str    e.g. "91282CDA2"
    coupon       float  annual rate as decimal, e.g. 0.04375
    maturity     date
    conv_factor  float  CME conversion factor (calculated via 6% base yield)
"""

from __future__ import annotations

import math
import requests
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Contract configuration
# ---------------------------------------------------------------------------

CONTRACT       = "TYM26"
DELIVERY_DATE  = date(2026, 6, 30)   # last business day of delivery month
DELIVERY_MONTH = date(2026, 6,  1)   # CME measures remaining maturity from here

MIN_YEARS = 6.5
MAX_YEARS = 10.0

# Eligibility window
MIN_MATURITY = date(2032, 12, 1)   # 6.5 years from 2026-06-01
MAX_MATURITY = date(2036, 6,  1)   # 10  years from 2026-06-01

# ---------------------------------------------------------------------------
# Conversion factor (CME formula — 6% semi-annual base yield)
# ---------------------------------------------------------------------------

def conversion_factor(coupon: float, maturity: date, delivery: date = DELIVERY_DATE) -> float:
    """Calculate the CME conversion factor for a Treasury note/bond.

    The CME formula prices the bond at a 6% semi-annual yield, then divides
    by par (100). Fractional coupon periods are handled by discounting back
    to the nearest whole coupon period first.

    Args:
        coupon:   Annual coupon rate as a decimal (e.g. 0.04375).
        maturity: Bond maturity date.
        delivery: Futures delivery date (default: DELIVERY_DATE).

    Returns:
        Conversion factor rounded to 4 decimal places.
    """
    BASE_YIELD = 0.06
    freq       = 2                        # semi-annual
    r          = BASE_YIELD / freq        # 0.03 per period
    c          = coupon / freq            # semi-annual coupon per $1 par

    # Total remaining periods (fractional)
    days_remaining = (maturity - delivery).days
    N_exact        = days_remaining / (365.25 / freq)

    n = int(N_exact)        # whole coupon periods
    f = N_exact - n         # fractional period (0 ≤ f < 1)

    if n == 0:
        # Bond matures within one coupon period
        cf = (1.0 + c) / (1.0 + r) ** f
    else:
        # PV of coupons + par at whole-period boundary
        pv_annuity = c * (1.0 - (1.0 + r) ** -n) / r
        pv_par     = (1.0 + r) ** -n
        price_at_boundary = pv_annuity + pv_par

        # Discount back fractional period, subtract accrued
        cf = (price_at_boundary + c) / (1.0 + r) ** f - c * f

    return round(cf, 4)


# ---------------------------------------------------------------------------
# Treasury Direct fetch
# ---------------------------------------------------------------------------

_TD_URL = "https://www.treasurydirect.gov/TA_WS/securities/search"

def _fetch_from_treasury_direct() -> list[dict]:
    """Fetch eligible bonds from the Treasury Direct public API.

    Queries for Notes and Bonds maturing in the TYM26 eligibility window.
    Returns a list of bond dicts (without conv_factor — added by caller).

    Raises:
        requests.RequestException: if the API is unreachable.
    """
    params = {
        "dateFieldName": "maturityDate",
        "startDate":     MIN_MATURITY.strftime("%Y-%m-%d"),
        "endDate":       MAX_MATURITY.strftime("%Y-%m-%d"),
        "type":          "Note",       # 10-year notes
        "pagesize":      "250",
        "format":        "json",
    }

    resp = requests.get(_TD_URL, params=params, timeout=10)
    resp.raise_for_status()
    securities = resp.json()

    bonds = []
    for s in securities:
        try:
            maturity = datetime.strptime(s["maturityDate"], "%Y-%m-%dT%H:%M:%S").date()
        except (KeyError, ValueError):
            continue

        # Skip if outside eligibility window (belt-and-suspenders)
        years_left = (maturity - DELIVERY_MONTH).days / 365.25
        if not (MIN_YEARS <= years_left <= MAX_YEARS):
            continue

        try:
            coupon = float(s["interestRate"]) / 100.0
        except (KeyError, ValueError, TypeError):
            continue

        cusip = s.get("cusip", "")

        bonds.append({
            "cusip":    cusip,
            "coupon":   coupon,
            "maturity": maturity,
        })

    return bonds


# ---------------------------------------------------------------------------
# Hardcoded fallback (snapshot — update periodically)
# ---------------------------------------------------------------------------
# 10-year notes issued roughly Dec 2022 – Jun 2026, maturing Dec 2032 – Jun 2036.
# Coupons reflect approximate auction rates at issuance.

_FALLBACK: list[dict] = [
    {"cusip": "91282CGT8", "coupon": 0.04000, "maturity": date(2033,  2, 15)},
    {"cusip": "91282CHH3", "coupon": 0.03875, "maturity": date(2033,  4, 15)},
    {"cusip": "91282CHW0", "coupon": 0.03625, "maturity": date(2033,  5, 15)},
    {"cusip": "91282CJE8", "coupon": 0.04250, "maturity": date(2034,  5, 15)},
    {"cusip": "91282CJX6", "coupon": 0.04500, "maturity": date(2034,  8, 15)},
    {"cusip": "91282CKG2", "coupon": 0.04375, "maturity": date(2034, 11, 15)},
    {"cusip": "91282CKY3", "coupon": 0.04625, "maturity": date(2035,  2, 15)},
    {"cusip": "91282CLL0", "coupon": 0.04500, "maturity": date(2035,  5, 15)},
    {"cusip": "91282CMA3", "coupon": 0.04250, "maturity": date(2035,  8, 15)},
    {"cusip": "91282CMT2", "coupon": 0.04375, "maturity": date(2035, 11, 15)},
    {"cusip": "91282CNG9", "coupon": 0.04500, "maturity": date(2036,  2, 15)},
    {"cusip": "91282CNZ7", "coupon": 0.04375, "maturity": date(2036,  5, 15)},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_basket(use_api: bool = True) -> list[dict]:
    """Return the TYM26 deliverable basket with conversion factors.

    Tries the Treasury Direct API first; falls back to the hardcoded snapshot
    if the API is unreachable or returns no results.

    Args:
        use_api: Set False to skip the API and use the fallback directly.

    Returns:
        List of bond dicts sorted by maturity, each with keys:
            cusip, coupon, maturity, conv_factor
    """
    bonds = []

    if use_api:
        try:
            bonds = _fetch_from_treasury_direct()
        except Exception:
            bonds = []

    if not bonds:
        bonds = list(_FALLBACK)

    # Attach conversion factors and sort
    for b in bonds:
        b["conv_factor"] = conversion_factor(b["coupon"], b["maturity"])

    bonds.sort(key=lambda b: b["maturity"])
    return bonds


def bond_label(bond: dict) -> str:
    """Human-readable label, e.g. '4.375% Nov-34'."""
    return f"{bond['coupon'] * 100:.3g}% {bond['maturity'].strftime('%b-%y')}"