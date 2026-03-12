"""
Bond pricing module.

All prices are expressed per unit of face value unless face_value is supplied.
Yields and coupon rates are expressed as decimals (e.g. 0.05 for 5%).
This implementation prices bonds on a coupon date (no accrued interest).
For full settlement pricing use price_bond_with_accrued().
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def price_bond(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    discount_rate: float,
    frequency: int = 2,
) -> float:
    """Price a fixed-coupon bond by discounting cash flows.

    Prices as of a coupon date (accrued interest = 0).

    Implementation uses a vectorised numpy dot product instead of a Python
    loop, which matters when this function is called in tight bootstrap or
    scenario-grid loops (10x–50x speedup for long maturities).

    Args:
        face_value: Principal amount repaid at maturity.
        coupon_rate: Annual coupon rate as a decimal (e.g. 0.05 for 5%).
        years_to_maturity: Remaining life of the bond in years.
        discount_rate: Annual yield (discount rate) as a decimal.
        frequency: Coupon payments per year (1 = annual, 2 = semi-annual).

    Returns:
        Clean / full price (same on a coupon date) in the same currency as
        face_value.
    """
    coupon_payment = face_value * coupon_rate / frequency
    n_periods = int(round(years_to_maturity * frequency))
    periodic_rate = discount_rate / frequency

    periods = np.arange(1, n_periods + 1, dtype=float)
    cfs = np.full(n_periods, coupon_payment)
    cfs[-1] += face_value                               # principal at maturity
    discount_factors = (1.0 + periodic_rate) ** periods

    return float(np.dot(cfs, 1.0 / discount_factors))


def accrued_interest(
    face_value: float,
    coupon_rate: float,
    frequency: int,
    days_since_last_coupon: int,
    days_in_coupon_period: int,
) -> float:
    """Calculate accrued interest using the actual/actual (ICMA) convention.

    US Treasuries use actual/actual; this function is general-purpose.

    Args:
        face_value: Principal amount.
        coupon_rate: Annual coupon rate as a decimal.
        frequency: Coupon payments per year.
        days_since_last_coupon: Calendar days elapsed since the last coupon.
        days_in_coupon_period: Total calendar days in the current coupon period.

    Returns:
        Accrued interest in the same currency as face_value.
    """
    coupon_payment = face_value * coupon_rate / frequency
    accrual_fraction = days_since_last_coupon / days_in_coupon_period
    return coupon_payment * accrual_fraction


def dirty_price(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    discount_rate: float,
    days_since_last_coupon: int,
    days_in_coupon_period: int,
    frequency: int = 2,
) -> float:
    """Full (dirty) price = clean price + accrued interest.

    Args:
        face_value: Principal amount.
        coupon_rate: Annual coupon rate as a decimal.
        years_to_maturity: Remaining life from settlement in years.
        discount_rate: Annual yield as a decimal.
        days_since_last_coupon: Calendar days since the last coupon payment.
        days_in_coupon_period: Total calendar days in the current coupon period.
        frequency: Coupon payments per year.

    Returns:
        Dirty (invoice) price in the same currency as face_value.
    """
    clean = price_bond(face_value, coupon_rate, years_to_maturity, discount_rate, frequency)
    ai = accrued_interest(face_value, coupon_rate, frequency, days_since_last_coupon, days_in_coupon_period)
    return clean + ai


def ytm(
    price: float,
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    frequency: int = 2,
    tol: float = 1e-10,
) -> float:
    """Solve for yield-to-maturity given a clean price (on a coupon date).

    Uses Brent's method on the interval [0.01 bp, 100%]. Works for
    discount, par, and premium bonds including near-zero-coupon issues.

    Args:
        price: Clean (flat) price in the same currency as face_value.
        face_value: Principal amount repaid at maturity.
        coupon_rate: Annual coupon rate as a decimal.
        years_to_maturity: Remaining life in years.
        frequency: Coupon payments per year.
        tol: Yield tolerance for the root-finder (default 1e-10).

    Returns:
        Annual yield-to-maturity as a decimal.

    Raises:
        ValueError: If no yield in [0.0001%, 100%] prices the bond at
                    the given price (e.g. price is negative or absurd).
    """
    def _pv(y):
        return price_bond(face_value, coupon_rate, years_to_maturity, y, frequency) - price

    try:
        return brentq(_pv, 1e-6, 1.0, xtol=tol)
    except ValueError:
        raise ValueError(
            f"Could not solve YTM for price={price:.4f}, coupon={coupon_rate:.4%}, "
            f"ytm={years_to_maturity:.2f}y. Check that the price is reasonable."
        )


def macaulay_duration(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    discount_rate: float,
    frequency: int = 2,
) -> float:
    """Macaulay duration: present-value-weighted average time to cash flow.

    Args:
        face_value: Principal amount.
        coupon_rate: Annual coupon rate as a decimal.
        years_to_maturity: Remaining life in years.
        discount_rate: Annual yield as a decimal.
        frequency: Coupon payments per year.

    Returns:
        Macaulay duration in years.
    """
    coupon_payment = face_value * coupon_rate / frequency
    n_periods      = int(round(years_to_maturity * frequency))
    periodic_rate  = discount_rate / frequency

    periods = np.arange(1, n_periods + 1, dtype=float)
    cfs = np.full(n_periods, coupon_payment)
    cfs[-1] += face_value

    discount_factors = (1.0 + periodic_rate) ** periods
    pv_cfs = cfs / discount_factors
    price  = pv_cfs.sum()

    # Times in years, not periods
    times_years = periods / frequency
    return float(np.dot(times_years, pv_cfs) / price)


def modified_duration(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    discount_rate: float,
    frequency: int = 2,
) -> float:
    """Modified duration: percentage price change per unit yield change.

    Modified duration = Macaulay duration / (1 + y/m).
    A bond with modified duration of 7 loses ~7% in price per 100bp
    rise in yield.

    Args:
        face_value: Principal amount.
        coupon_rate: Annual coupon rate as a decimal.
        years_to_maturity: Remaining life in years.
        discount_rate: Annual yield as a decimal.
        frequency: Coupon payments per year.

    Returns:
        Modified duration in years.
    """
    mac = macaulay_duration(face_value, coupon_rate, years_to_maturity, discount_rate, frequency)
    return mac / (1.0 + discount_rate / frequency)


def dv01(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    discount_rate: float,
    frequency: int = 2,
    bump_bps: float = 1.0,
) -> float:
    """Dollar value of a basis point (DV01) via central finite difference.

    DV01 = (P(y - bump) - P(y + bump)) / (2 * bump_in_decimal).

    Uses a 2-sided bump so the result is symmetric and second-order
    accurate — suitable for convexity estimation as well.

    Args:
        face_value: Principal amount.
        coupon_rate: Annual coupon rate as a decimal.
        years_to_maturity: Remaining life in years.
        discount_rate: Annual yield as a decimal.
        frequency: Coupon payments per year.
        bump_bps: Yield bump in basis points (default 1 bp).

    Returns:
        DV01 in the same currency as face_value (positive number; price
        falls as yield rises for normal bonds).
    """
    bump = bump_bps / 10_000
    p_up   = price_bond(face_value, coupon_rate, years_to_maturity, discount_rate + bump, frequency)
    p_down = price_bond(face_value, coupon_rate, years_to_maturity, discount_rate - bump, frequency)
    return (p_down - p_up) / 2.0


def convexity(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    discount_rate: float,
    frequency: int = 2,
    bump_bps: float = 1.0,
) -> float:
    """Convexity via central second finite difference (units: years²).

    Convexity = (P(y+h) + P(y-h) - 2*P(y)) / (P(y) * h²)

    A positive convexity means price gains exceed losses for equal
    up/down yield moves — the bond is "convex" (desirable property).

    Args:
        face_value: Principal amount.
        coupon_rate: Annual coupon rate as a decimal.
        years_to_maturity: Remaining life in years.
        discount_rate: Annual yield as a decimal.
        frequency: Coupon payments per year.
        bump_bps: Yield bump in basis points for the finite difference.

    Returns:
        Convexity in years squared.
    """
    bump = bump_bps / 10_000
    p    = price_bond(face_value, coupon_rate, years_to_maturity, discount_rate,        frequency)
    p_up = price_bond(face_value, coupon_rate, years_to_maturity, discount_rate + bump, frequency)
    p_dn = price_bond(face_value, coupon_rate, years_to_maturity, discount_rate - bump, frequency)
    return (p_up + p_dn - 2.0 * p) / (p * bump ** 2)