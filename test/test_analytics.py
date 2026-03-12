"""Tests for core.analytics — Bond model, z-spread, roll-down, spreads."""

import numpy as np
import pytest

from core.curves import TREASURY_MATURITIES_YRS, SpotCurve, fit_nelson_siegel
from core.analytics import (
    Bond,
    RollDownResult,
    TotalReturnDecomposition,
    curve_spreads,
    roll_down_return,
    total_return_decomposition,
    z_spread,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

NORMAL_YIELDS = np.array([0.042, 0.044, 0.045, 0.046, 0.048])
FLAT_YIELDS = np.full(5, 0.05)


@pytest.fixture
def normal_spot_curve():
    return SpotCurve(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)


@pytest.fixture
def flat_spot_curve():
    return SpotCurve(TREASURY_MATURITIES_YRS, FLAT_YIELDS)


@pytest.fixture
def ten_year_bond():
    """Par bond: coupon = par yield at 10Y on the normal curve."""
    return Bond(
        face_value=100.0,
        coupon_rate=0.046,       # 10Y rate on NORMAL_YIELDS
        years_to_maturity=10.0,
        ytm=0.046,
    )


@pytest.fixture
def five_year_bond():
    return Bond(
        face_value=100.0,
        coupon_rate=0.045,       # 5Y rate on NORMAL_YIELDS
        years_to_maturity=5.0,
        ytm=0.045,
    )


# ── Bond model ────────────────────────────────────────────────────────────────

class TestBond:
    def test_coupon_cashflow(self):
        bond = Bond(face_value=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.05)
        assert abs(bond.coupon_cashflow - 25.0) < 1e-10  # 1000 * 5% / 2

    def test_n_periods(self):
        bond = Bond(face_value=100.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.05)
        assert bond.n_periods == 20

    def test_cashflow_times_shape(self):
        bond = Bond(face_value=100.0, coupon_rate=0.05, years_to_maturity=5.0, ytm=0.05)
        times = bond.cashflow_times()
        assert times.shape == (10,)

    def test_cashflow_times_last_equals_maturity(self):
        bond = Bond(face_value=100.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.05)
        times = bond.cashflow_times()
        assert abs(times[-1] - 10.0) < 1e-10

    def test_cashflows_last_includes_principal(self):
        bond = Bond(face_value=100.0, coupon_rate=0.05, years_to_maturity=5.0, ytm=0.05)
        cfs = bond.cashflows()
        # Last payment = coupon + face value
        assert abs(cfs[-1] - (bond.coupon_cashflow + 100.0)) < 1e-10

    def test_cashflows_all_positive(self):
        bond = Bond(face_value=100.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.05)
        assert np.all(bond.cashflows() > 0)

    def test_invalid_maturity_raises(self):
        with pytest.raises(Exception):
            Bond(face_value=100.0, coupon_rate=0.05, years_to_maturity=0.0, ytm=0.05)


# ── Z-spread ──────────────────────────────────────────────────────────────────

class TestZSpread:
    def test_z_spread_zero_when_priced_on_treasury_curve(self, ten_year_bond, normal_spot_curve):
        """If dirty price equals the spot-curve DCF price, z-spread = 0."""
        times = ten_year_bond.cashflow_times()
        cfs = ten_year_bond.cashflows()
        spot_rates = normal_spot_curve.rate(times)
        fair_price = float(np.dot(cfs, np.exp(-spot_rates * times)))

        zs = z_spread(ten_year_bond, fair_price, normal_spot_curve)
        assert abs(zs) < 1e-7  # sub-0.001 bp

    def test_z_spread_positive_when_bond_cheap(self, ten_year_bond, normal_spot_curve):
        """If bond is priced below the Treasury-curve DCF, z-spread > 0."""
        times = ten_year_bond.cashflow_times()
        cfs = ten_year_bond.cashflows()
        spot_rates = normal_spot_curve.rate(times)
        fair_price = float(np.dot(cfs, np.exp(-spot_rates * times)))
        cheap_price = fair_price * 0.99  # 1% below fair value

        zs = z_spread(ten_year_bond, cheap_price, normal_spot_curve)
        assert zs > 0

    def test_z_spread_negative_when_bond_rich(self, ten_year_bond, normal_spot_curve):
        """If bond is priced above the Treasury-curve DCF, z-spread < 0."""
        times = ten_year_bond.cashflow_times()
        cfs = ten_year_bond.cashflows()
        spot_rates = normal_spot_curve.rate(times)
        fair_price = float(np.dot(cfs, np.exp(-spot_rates * times)))
        rich_price = fair_price * 1.01  # 1% above fair value

        zs = z_spread(ten_year_bond, rich_price, normal_spot_curve)
        assert zs < 0

    def test_z_spread_monotone_in_price(self, ten_year_bond, normal_spot_curve):
        """Higher dirty price → lower (more negative) z-spread."""
        times = ten_year_bond.cashflow_times()
        cfs = ten_year_bond.cashflows()
        spot_rates = normal_spot_curve.rate(times)
        fair_price = float(np.dot(cfs, np.exp(-spot_rates * times)))

        prices = [fair_price * m for m in [0.95, 0.97, 1.00, 1.02, 1.05]]
        spreads = [z_spread(ten_year_bond, p, normal_spot_curve) for p in prices]
        # Spreads should be strictly decreasing as price increases
        for i in range(len(spreads) - 1):
            assert spreads[i] > spreads[i + 1]


# ── Roll-down return ──────────────────────────────────────────────────────────

class TestRollDownReturn:
    def test_returns_namedtuple(self, ten_year_bond, normal_spot_curve):
        result = roll_down_return(ten_year_bond, normal_spot_curve)
        assert isinstance(result, RollDownResult)

    def test_price_now_positive(self, ten_year_bond, normal_spot_curve):
        result = roll_down_return(ten_year_bond, normal_spot_curve)
        assert result.price_now > 0

    def test_coupon_accrual_positive(self, ten_year_bond, normal_spot_curve):
        """Coupon accrual should always be positive for coupon-paying bonds."""
        result = roll_down_return(ten_year_bond, normal_spot_curve)
        assert result.coupon_accrual_pct > 0

    def test_roll_down_positive_on_normal_curve(self, ten_year_bond, normal_spot_curve):
        """On an upward-sloping curve, rolling down produces a positive price gain."""
        result = roll_down_return(ten_year_bond, normal_spot_curve)
        assert result.roll_down_pct > 0

    def test_roll_down_near_zero_for_par_bond_on_flat_curve(self, flat_spot_curve):
        """A par bond on a flat curve has zero roll-down.

        A par bond (coupon = yield) prices at exactly face value at any maturity
        on a flat curve. Rolling from 5Y to 4.75Y leaves the price unchanged.
        A discount bond (coupon < yield) DOES have non-zero roll-down even on a
        flat curve, because it's pulling to par.
        """
        par_bond = Bond(face_value=100.0, coupon_rate=0.05,  # coupon = flat rate
                        years_to_maturity=5.0, ytm=0.05)
        result = roll_down_return(par_bond, flat_spot_curve)
        assert abs(result.roll_down_pct) < 0.05  # within 5bps for a par bond

    def test_total_carry_roll_equals_sum(self, ten_year_bond, normal_spot_curve):
        """total_carry_roll_pct ≈ coupon_accrual_pct + roll_down_pct.

        Values are rounded to 4 decimal places in the NamedTuple, so the
        sum may differ from the pre-rounded total by up to 0.5e-4.
        """
        result = roll_down_return(ten_year_bond, normal_spot_curve)
        expected = result.coupon_accrual_pct + result.roll_down_pct
        assert abs(result.total_carry_roll_pct - expected) < 1e-3  # rounding tolerance

    def test_holding_period_exceeds_maturity_raises(self, normal_spot_curve):
        short_bond = Bond(face_value=100.0, coupon_rate=0.045,
                          years_to_maturity=0.5, ytm=0.045)
        with pytest.raises(ValueError, match="holding_period"):
            roll_down_return(short_bond, normal_spot_curve, holding_period_yrs=1.0)

    def test_forward_breakeven_higher_than_current_yield_normal_curve(
        self, ten_year_bond, normal_spot_curve
    ):
        """Forward breakeven yield on a normal curve should exceed current spot 10Y rate."""
        result = roll_down_return(ten_year_bond, normal_spot_curve)
        current_10y = normal_spot_curve.rate(10.0)
        assert result.forward_breakeven_ytm > current_10y


# ── Total return decomposition ────────────────────────────────────────────────

class TestTotalReturnDecomposition:
    def test_zero_yield_change_no_duration_pnl(self, ten_year_bond, normal_spot_curve):
        """With no yield change, duration P&L and convexity correction = 0."""
        result = total_return_decomposition(
            ten_year_bond, normal_spot_curve,
            repo_rate=0.04, holding_period_yrs=0.25, yield_change=0.0
        )
        assert isinstance(result, TotalReturnDecomposition)
        assert abs(result.duration_pnl_pct) < 1e-8
        assert abs(result.convexity_pnl_pct) < 1e-8

    def test_positive_yield_change_negative_duration_pnl(
        self, ten_year_bond, normal_spot_curve
    ):
        """Rising yields → negative duration P&L for a long position."""
        result = total_return_decomposition(
            ten_year_bond, normal_spot_curve,
            repo_rate=0.04, holding_period_yrs=0.25, yield_change=0.01  # +100bps
        )
        assert result.duration_pnl_pct < 0

    def test_convexity_always_positive(self, ten_year_bond, normal_spot_curve):
        """Convexity correction is positive regardless of direction of yield change."""
        for dy in [-0.01, +0.01]:
            result = total_return_decomposition(
                ten_year_bond, normal_spot_curve,
                repo_rate=0.04, holding_period_yrs=0.25, yield_change=dy
            )
            assert result.convexity_pnl_pct > 0

    def test_total_is_sum_of_components(self, ten_year_bond, normal_spot_curve):
        result = total_return_decomposition(
            ten_year_bond, normal_spot_curve,
            repo_rate=0.04, holding_period_yrs=0.25, yield_change=0.005
        )
        expected = (result.carry_pct + result.roll_down_pct
                    + result.duration_pnl_pct + result.convexity_pnl_pct)
        assert abs(result.total_pct - expected) < 1e-4


# ── Curve spreads ─────────────────────────────────────────────────────────────

class TestCurveSpreads:
    NORMAL_CURVE = {"2Y": 0.044, "5Y": 0.045, "10Y": 0.046, "30Y": 0.048}
    INVERTED_CURVE = {"2Y": 0.055, "5Y": 0.052, "10Y": 0.047, "30Y": 0.045}
    FLAT_CURVE = {"2Y": 0.05, "5Y": 0.05, "10Y": 0.05, "30Y": 0.05}

    def test_2s10s_positive_normal_curve(self):
        result = curve_spreads(self.NORMAL_CURVE)
        assert result["2s10s_bps"] > 0

    def test_2s10s_negative_inverted_curve(self):
        result = curve_spreads(self.INVERTED_CURVE)
        assert result["2s10s_bps"] < 0

    def test_all_spreads_zero_flat_curve(self):
        result = curve_spreads(self.FLAT_CURVE)
        assert abs(result["2s10s_bps"]) < 1e-8
        assert abs(result["5s30s_bps"]) < 1e-8
        assert abs(result["2s5s10s_fly_bps"]) < 1e-8

    def test_butterfly_formula(self):
        """2s5s10s = 2*5Y - 2Y - 10Y, in bps."""
        c = self.NORMAL_CURVE
        expected_fly = (2 * c["5Y"] - c["2Y"] - c["10Y"]) * 10_000
        result = curve_spreads(c)
        assert abs(result["2s5s10s_fly_bps"] - expected_fly) < 1e-8

    def test_missing_maturity_raises(self):
        bad_curve = {"2Y": 0.044, "10Y": 0.046}  # missing 5Y and 30Y
        with pytest.raises(KeyError):
            curve_spreads(bad_curve)

    def test_spread_units_are_bps(self):
        """Verify output is in bps (not decimals)."""
        c = {"2Y": 0.040, "5Y": 0.045, "10Y": 0.050, "30Y": 0.055}
        result = curve_spreads(c)
        # 2s10s = 10Y − 2Y = 5.0% − 4.0% = 100bps
        assert abs(result["2s10s_bps"] - 100.0) < 1e-6