"""
Yield curve scenario engine.

All shift magnitudes are expressed in basis points (bps).
Curves are represented as ``dict[str, float]`` mapping maturity label
(e.g. "2Y", "10Y") to yield in decimal form (e.g. 0.045 for 4.5%).
"""

from __future__ import annotations


def parallel_shift(curve: dict[str, float], shift_bps: float) -> dict[str, float]:
    """Shift every maturity by the same number of basis points.

    Args:
        curve: Yield curve mapping maturity label to decimal yield.
        shift_bps: Shift in basis points (positive = higher rates).

    Returns:
        New yield curve with the shift applied.
    """
    shift = shift_bps / 10_000
    return {maturity: rate + shift for maturity, rate in curve.items()}


def bear_steepening(curve: dict[str, float], shift_bps: float) -> dict[str, float]:
    """Long-end rates rise more than short-end (steeper curve, higher rates).

    The shift is linearly interpolated from 0 bps at the short end to
    ``shift_bps`` at the long end.

    Args:
        curve: Yield curve mapping maturity label to decimal yield.
        shift_bps: Maximum shift in basis points applied at the long end.

    Returns:
        New yield curve with bear steepening applied.
    """
    shift = shift_bps / 10_000
    maturities = list(curve.keys())
    n = len(maturities)
    return {
        maturity: rate + shift * (i / (n - 1))
        for i, (maturity, rate) in enumerate(curve.items())
    }


def bear_flattening(curve: dict[str, float], shift_bps: float) -> dict[str, float]:
    """Short-end rates rise more than long-end (flatter curve, higher rates).

    The shift is linearly interpolated from ``shift_bps`` at the short end
    down to 0 bps at the long end.

    Args:
        curve: Yield curve mapping maturity label to decimal yield.
        shift_bps: Maximum shift in basis points applied at the short end.

    Returns:
        New yield curve with bear flattening applied.
    """
    shift = shift_bps / 10_000
    maturities = list(curve.keys())
    n = len(maturities)
    return {
        maturity: rate + shift * (1 - i / (n - 1))
        for i, (maturity, rate) in enumerate(curve.items())
    }


def bull_steepening(curve: dict[str, float], shift_bps: float) -> dict[str, float]:
    """Short-end rates fall more than long-end (steeper curve, lower rates).

    Typical drivers: Fed signals easing while long-end anchored by term premium.
    The short end drops by ``shift_bps``; the long end is unchanged. Shifts
    are linearly interpolated from −shift_bps at the front to 0 at the back.

    Args:
        curve: Yield curve mapping maturity label to decimal yield.
        shift_bps: Magnitude of the short-end drop in basis points (positive value).

    Returns:
        New yield curve with bull steepening applied.
    """
    shift = shift_bps / 10_000
    maturities = list(curve.keys())
    n = len(maturities)
    return {
        maturity: rate - shift * (1 - i / (n - 1))
        for i, (maturity, rate) in enumerate(curve.items())
    }


def bull_flattening(curve: dict[str, float], shift_bps: float) -> dict[str, float]:
    """Long-end rates fall more than short-end (flatter curve, lower rates).

    Typical drivers: flight to quality, disinflation, strong duration demand.
    The long end drops by ``shift_bps``; the short end is unchanged. Shifts
    are linearly interpolated from 0 at the front to −shift_bps at the back.

    Args:
        curve: Yield curve mapping maturity label to decimal yield.
        shift_bps: Magnitude of the long-end drop in basis points (positive value).

    Returns:
        New yield curve with bull flattening applied.
    """
    shift = shift_bps / 10_000
    maturities = list(curve.keys())
    n = len(maturities)
    return {
        maturity: rate - shift * (i / (n - 1))
        for i, (maturity, rate) in enumerate(curve.items())
    }


def custom_shift(
    curve: dict[str, float], shifts_bps: dict[str, float]
) -> dict[str, float]:
    """Apply per-maturity shifts.

    Maturities not present in ``shifts_bps`` are left unchanged.

    Args:
        curve: Yield curve mapping maturity label to decimal yield.
        shifts_bps: Mapping of maturity label to shift in basis points.

    Returns:
        New yield curve with the per-maturity shifts applied.
    """
    return {
        maturity: rate + shifts_bps.get(maturity, 0) / 10_000
        for maturity, rate in curve.items()
    }