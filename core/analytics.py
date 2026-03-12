"""
Fixed income analytics: carry, roll-down, Z-spread, and total return decomposition.

Discount factors use continuous compounding (DF = e^{-z·τ}) to interface with
SpotCurve. Bond market prices use periodic compounding per core.pricing convention.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq
from pydantic import BaseModel, Field, model_validator

from core.curves import SpotCurve
from core.pricing import modified_duration as _modified_duration, convexity as _convexity

# ── Z-spread solver constants ─────────────────────────────────────────────────
_ZS_MIN = -0.05     # −500 bps lower bound  (rare negative spread)
_ZS_MAX = +0.50     # +5000 bps upper bound (captures all HY / distressed)
_ZS_XTOL = 1e-10    # convergence: sub-0.001 bp accuracy

# ── Default holding periods ───────────────────────────────────────────────────
HOLD_3M = 3.0 / 12.0   # 3 months in years
HOLD_6M = 6.0 / 12.0
HOLD_1Y = 1.0


# ── Bond data model ───────────────────────────────────────────────────────────

class Bond(BaseModel):
    """Immutable bond specification.

    All rates are decimals (e.g. 0.045 for 4.5%).  The model is the single
    source of truth passed between analytics functions; it owns the
    cashflow generation logic so callers never re-implement it.

    Attributes
    ----------
    face_value: Principal (par) amount.
    coupon_rate: Annual coupon as a decimal.
    years_to_maturity: Remaining life from today/settlement.
    frequency: Coupon payments per year (2 = semi-annual, US default).
    ytm: Flat yield to maturity as a decimal.
    """

    face_value: float = Field(default=100.0, gt=0.0)
    coupon_rate: float = Field(..., ge=0.0, le=0.25)
    years_to_maturity: float = Field(..., gt=0.0)
    frequency: int = Field(default=2, ge=1, le=4)
    ytm: float = Field(..., ge=0.0, le=0.30)

    @model_validator(mode="after")
    def _check_maturity_periods(self) -> "Bond":
        n = int(round(self.years_to_maturity * self.frequency))
        if n < 1:
            raise ValueError(
                f"years_to_maturity={self.years_to_maturity} gives < 1 coupon period "
                f"at frequency={self.frequency}."
            )
        return self

    @property
    def coupon_cashflow(self) -> float:
        """Single periodic coupon payment."""
        return self.face_value * self.coupon_rate / self.frequency

    @property
    def n_periods(self) -> int:
        """Total coupon periods remaining."""
        return int(round(self.years_to_maturity * self.frequency))

    def cashflow_times(self) -> np.ndarray:
        """Cash flow payment times in years, shape (n_periods,).

        Times run from 1/frequency to years_to_maturity in equal steps.
        The final entry is the maturity date (last coupon + principal).
        """
        dt = 1.0 / self.frequency
        return np.arange(1, self.n_periods + 1, dtype=float) * dt

    def cashflows(self) -> np.ndarray:
        """Cash flow amounts, shape (n_periods,).

        All entries equal coupon_cashflow; the final entry adds face_value.
        """
        cfs = np.full(self.n_periods, self.coupon_cashflow)
        cfs[-1] += self.face_value
        return cfs


# ── Z-spread ──────────────────────────────────────────────────────────────────

def z_spread(
    bond: Bond,
    dirty_price: float,
    spot_curve: SpotCurve,
) -> float:
    """Constant spread over the Treasury spot curve that equates PV to dirty price.

    Solves for s in: P_dirty = Σ CF_t · e^{−(z(t) + s) · t}

    Args:
        bond: Bond specification.
        dirty_price: Observed invoice price in the same units as bond.face_value.
        spot_curve: Treasury spot curve for discounting.

    Returns:
        Z-spread as a decimal (e.g. 0.0025 = +25 bps).

    Raises:
        ValueError: If the root cannot be bracketed within [−500bps, +5000bps].
    """
    times = bond.cashflow_times()           # shape (n,)
    cfs = bond.cashflows()                  # shape (n,)
    spot_rates = spot_curve.rate(times)     # vectorised: shape (n,)

    def _price_residual(s: float) -> float:
        dfs = np.exp(-(spot_rates + s) * times)
        return float(np.dot(cfs, dfs)) - dirty_price

    lo_val = _price_residual(_ZS_MIN)
    hi_val = _price_residual(_ZS_MAX)

    if lo_val * hi_val > 0:
        raise ValueError(
            f"Z-spread root not bracketed: residuals at bounds are "
            f"{lo_val:.4f} (s={_ZS_MIN*1e4:.0f}bps) and "
            f"{hi_val:.4f} (s={_ZS_MAX*1e4:.0f}bps). "
            f"Check dirty_price={dirty_price:.4f} is realistic."
        )

    return float(brentq(_price_residual, _ZS_MIN, _ZS_MAX, xtol=_ZS_XTOL))


# ── Roll-down return ──────────────────────────────────────────────────────────

class RollDownResult(NamedTuple):
    """Return components from a carry + roll-down analysis."""

    price_now: float
    """Current price using periodic compounding at bond.ytm."""

    price_rolled: float
    """Price after the holding period at the spot curve rate for the shorter maturity."""

    coupon_accrual_pct: float
    """Coupon income over the holding period as % of initial price."""

    roll_down_pct: float
    """Price appreciation (or depreciation) from rolling, as % of initial price."""

    total_carry_roll_pct: float
    """Total carry + roll-down return as % of initial price (= coupon + roll)."""

    forward_breakeven_ytm: float
    """YTM the bond must reach at horizon for the position to break even.
    Derived as the forward rate f(holding_period, years_to_maturity)."""

    holding_period_yrs: float
    """Holding period used for the calculation, in years."""


def roll_down_return(
    bond: Bond,
    spot_curve: SpotCurve,
    holding_period_yrs: float = HOLD_3M,
) -> RollDownResult:
    """Carry + roll-down return assuming the spot curve is unchanged.

    As the bond rolls to a shorter maturity, its price adjusts to the spot rate
    at that maturity. On an upward-sloping curve this produces a positive roll.
    The forward breakeven yield is where carry + roll is exactly offset.

    Args:
        bond: Bond specification.
        spot_curve: Current spot curve, assumed static over the holding period.
        holding_period_yrs: Calendar time elapsed (default 3 months).

    Returns:
        RollDownResult with all return components.

    Raises:
        ValueError: If holding_period_yrs >= bond.years_to_maturity.
    """
    from core.pricing import price_bond as _price_bond

    if holding_period_yrs >= bond.years_to_maturity:
        raise ValueError(
            f"holding_period_yrs={holding_period_yrs} must be < "
            f"bond.years_to_maturity={bond.years_to_maturity}"
        )

    # ── Current price: periodic compounding at bond's own YTM ────────────
    # Consistent with market-quoted clean prices. For a par bond
    # (coupon = ytm), this equals the face value exactly.
    price_now = _price_bond(
        bond.face_value, bond.coupon_rate,
        bond.years_to_maturity, bond.ytm, bond.frequency,
    )

    # ── Rolled price: same bond priced at the SHORTER maturity's spot rate ─
    # Reading the yield at the rolled maturity from the static spot curve
    # isolates the pure roll-down effect (yield pickup from sliding down
    # the curve). Using periodic compounding ensures consistency with
    # price_now and gives roll_down = 0 for a par bond on a flat curve.
    rolled_maturity = bond.years_to_maturity - holding_period_yrs
    rolled_yield = spot_curve.rate(rolled_maturity)
    price_rolled = _price_bond(
        bond.face_value, bond.coupon_rate,
        rolled_maturity, rolled_yield, bond.frequency,
    )

    # ── Coupon accrual over holding period ────────────────────────────────
    coupon_accrual = bond.coupon_rate * bond.face_value * holding_period_yrs
    coupon_pct = coupon_accrual / price_now * 100.0

    # ── Roll-down ─────────────────────────────────────────────────────────
    roll_pct = (price_rolled - price_now) / price_now * 100.0

    # ── Forward breakeven: forward rate for the remaining period ──────────
    fwd_breakeven = spot_curve.forward_rate(holding_period_yrs, bond.years_to_maturity)

    return RollDownResult(
        price_now=round(price_now, 6),
        price_rolled=round(price_rolled, 6),
        coupon_accrual_pct=round(coupon_pct, 4),
        roll_down_pct=round(roll_pct, 4),
        total_carry_roll_pct=round(coupon_pct + roll_pct, 4),
        forward_breakeven_ytm=round(fwd_breakeven, 6),
        holding_period_yrs=holding_period_yrs,
    )


# ── Total return decomposition ────────────────────────────────────────────────

class TotalReturnDecomposition(NamedTuple):
    """Full P&L attribution for a bond position over a holding period."""

    carry_pct: float
    """Coupon accrual minus repo financing, as % of initial price."""

    roll_down_pct: float
    """Price change from rolling down the static curve, as % of initial price."""

    duration_pnl_pct: float
    """P&L from a parallel yield shift: −MD × Δy × 100 (in %, sign = direction)."""

    convexity_pnl_pct: float
    """Convexity correction: +0.5 × convexity × Δy² × 100 (in %)."""

    total_pct: float
    """Sum of all four components, as % of initial price."""


def total_return_decomposition(
    bond: Bond,
    spot_curve: SpotCurve,
    repo_rate: float,
    holding_period_yrs: float = HOLD_3M,
    yield_change: float = 0.0,
    mod_dur: float | None = None,
    cvx: float | None = None,
) -> TotalReturnDecomposition:
    """P&L attribution: carry, roll-down, duration, and convexity components.

    Args:
        bond: Bond specification.
        spot_curve: Current spot curve, assumed static for carry + roll.
        repo_rate: Repo financing rate as a decimal.
        holding_period_yrs: Horizon in years.
        yield_change: Parallel yield shift as a decimal (e.g. +0.005 = +50bps).
        mod_dur: Pre-computed modified duration; computed from YTM if None.
        cvx: Pre-computed convexity; computed from YTM if None.

    Returns:
        TotalReturnDecomposition NamedTuple.
    """
    rd = roll_down_return(bond, spot_curve, holding_period_yrs)
    price_now = rd.price_now

    # Financing cost over holding period
    financing_pct = repo_rate * holding_period_yrs * 100.0
    carry_pct = rd.coupon_accrual_pct - financing_pct

    # Duration and convexity — use supplied values or compute from YTM
    md = mod_dur if mod_dur is not None else _modified_duration(
        bond.face_value, bond.coupon_rate, bond.years_to_maturity, bond.ytm, bond.frequency
    )
    cvx = cvx if cvx is not None else _convexity(
        bond.face_value, bond.coupon_rate, bond.years_to_maturity, bond.ytm, bond.frequency
    )

    duration_pnl_pct = -md * yield_change * 100.0
    convexity_pnl_pct = 0.5 * cvx * yield_change**2 * 100.0
    total_pct = carry_pct + rd.roll_down_pct + duration_pnl_pct + convexity_pnl_pct

    return TotalReturnDecomposition(
        carry_pct=round(carry_pct, 4),
        roll_down_pct=round(rd.roll_down_pct, 4),
        duration_pnl_pct=round(duration_pnl_pct, 4),
        convexity_pnl_pct=round(convexity_pnl_pct, 4),
        total_pct=round(total_pct, 4),
    )


# ── Curve spread metrics ──────────────────────────────────────────────────────

def curve_spreads(curve: dict[str, float]) -> dict[str, float]:
    """Standard Treasury curve spread metrics: 2s10s, 5s30s, 2s5s10s butterfly.

    Args:
        curve: Maturity label → decimal yield. Must contain "2Y", "5Y", "10Y", "30Y".

    Returns:
        Dict with "2s10s_bps", "5s30s_bps", "2s5s10s_fly_bps" in basis points.

    Raises:
        KeyError: If any required maturity is missing.
    """
    required = {"2Y", "5Y", "10Y", "30Y"}
    missing = required - curve.keys()
    if missing:
        raise KeyError(f"curve is missing required maturities: {missing}")

    to_bps = 10_000.0
    return {
        "2s10s_bps":       (curve["10Y"] - curve["2Y"]) * to_bps,
        "5s30s_bps":       (curve["30Y"] - curve["5Y"]) * to_bps,
        "2s5s10s_fly_bps": (2 * curve["5Y"] - curve["2Y"] - curve["10Y"]) * to_bps,
    }