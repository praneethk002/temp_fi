"""
Market data: yield curve → per-bond cash prices.

The delivery basket needs a cash price for each bond to compute implied repo.
Two sources are supported:

  1. FRED-derived  — interpolate the FRED Treasury curve to each bond's
                     remaining maturity, then price via core.pricing.price_bond().
                     Good enough for daily monitoring; not tick-accurate.

  2. Manual override — supply a dict of {cusip: price} from Bloomberg or
                       another source.  Any CUSIP in the override dict takes
                       precedence over the FRED-derived price.

Usage::

    prices = get_bond_prices(basket, repo_rate=0.053)          # FRED only
    prices = get_bond_prices(basket, overrides={"91282CKG2": 97.25})  # mixed
    prices = get_bond_prices(basket, use_fred=False,
                              overrides={"91282CKG2": 97.25})  # manual only
"""

from __future__ import annotations

import math
from datetime import date

from core.pricing import price_bond
from data.fred_client import get_yield_curve

# Sorted FRED maturities in years — used for linear interpolation
_FRED_MATURITIES_YRS: list[float] = [3/12, 2.0, 5.0, 7.0, 10.0, 30.0]
_FRED_KEYS: list[str] = ["3M", "2Y", "5Y", "7Y", "10Y", "30Y"]


def _interpolate_yield(
    years_to_maturity: float,
    curve: dict[str, float],
) -> float:
    """Linearly interpolate/extrapolate the FRED curve to a given maturity.

    Args:
        years_to_maturity: Remaining life in years.
        curve:             Dict from get_yield_curve() — may be missing some keys.

    Returns:
        Interpolated yield as a decimal.

    Raises:
        ValueError: If the curve has fewer than 2 data points.
    """
    # Build parallel lists of available maturities and rates
    mats, rates = [], []
    for m, k in zip(_FRED_MATURITIES_YRS, _FRED_KEYS):
        if k in curve:
            mats.append(m)
            rates.append(curve[k])

    if len(mats) < 2:
        raise ValueError(
            f"Need at least 2 yield curve points; got {len(mats)}. "
            "Check FRED_API_KEY or provide manual overrides."
        )

    # Clamp to available range (flat extrapolation)
    if years_to_maturity <= mats[0]:
        return rates[0]
    if years_to_maturity >= mats[-1]:
        return rates[-1]

    # Linear interpolation between bracketing points
    for i in range(len(mats) - 1):
        if mats[i] <= years_to_maturity <= mats[i + 1]:
            t = (years_to_maturity - mats[i]) / (mats[i + 1] - mats[i])
            return rates[i] + t * (rates[i + 1] - rates[i])

    return rates[-1]  # unreachable but satisfies type checker


def get_bond_prices(
    basket: list[dict],
    as_of: date | None = None,
    use_fred: bool = True,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return a cash price for every bond in the delivery basket.

    Prices are clean prices as a percentage of par (e.g. 97.25).

    FRED-derived prices use price_bond() at the interpolated spot yield.
    Manual overrides take precedence over FRED prices.

    Args:
        basket:    List of bond dicts from core.basket.get_basket().
        as_of:     Date for which prices are needed. Defaults to today.
                   Used as both the pricing date and the FRED lookup date.
        use_fred:  Fetch FRED yields and price all bonds analytically.
                   Set False to use only manual overrides.
        overrides: Dict mapping cusip → clean price (% of par).
                   Overrides FRED-derived price for the given CUSIPs.

    Returns:
        Dict mapping cusip → clean price (% of par).
        Only bonds with a price (either source) are included.

    Raises:
        ValueError: If use_fred=True and the FRED curve has < 2 points,
                    and no overrides are provided for the missing bonds.
    """
    if as_of is None:
        as_of = date.today()

    overrides = overrides or {}
    prices: dict[str, float] = {}

    if use_fred:
        curve = get_yield_curve(as_of=as_of)
        if curve:
            for bond in basket:
                cusip = bond["cusip"]
                if cusip in overrides:
                    continue  # will be filled by overrides below
                years_left = (bond["maturity"] - as_of).days / 365.25
                if years_left <= 0:
                    continue
                try:
                    yield_ = _interpolate_yield(years_left, curve)
                    prices[cusip] = price_bond(100.0, bond["coupon"], years_left, yield_)
                except (ValueError, ZeroDivisionError):
                    pass  # skip bonds that can't be priced

    # Apply overrides (higher priority than FRED)
    prices.update(overrides)

    return prices