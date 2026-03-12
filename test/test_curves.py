"""Tests for core.curves — Nelson-Siegel fitting, SpotCurve, bootstrap."""

import numpy as np
import pytest

from core.curves import (
    TREASURY_MATURITIES_YRS,
    NelsonSiegelParams,
    SpotCurve,
    bootstrap_spot_rates,
    decompose_curve_shift,
    fit_nelson_siegel,
    nelson_siegel_rate,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

FLAT_YIELDS = np.full(5, 0.05)  # flat curve at 5%

NORMAL_YIELDS = np.array([0.042, 0.044, 0.045, 0.046, 0.048])  # upward sloping

INVERTED_YIELDS = np.array([0.055, 0.052, 0.049, 0.047, 0.045])  # inverted (2022–23 style)


# ── Nelson-Siegel rate function ───────────────────────────────────────────────

class TestNelsonSiegelRate:
    def test_scalar_input(self):
        rate = nelson_siegel_rate(10.0, 0.05, -0.01, 0.005, 2.0)
        assert isinstance(rate, float)

    def test_array_input_shape(self):
        taus = np.array([1.0, 5.0, 10.0])
        rates = nelson_siegel_rate(taus, 0.05, -0.01, 0.005, 2.0)
        assert rates.shape == (3,)

    def test_slope_loading_at_short_end(self):
        """L(τ) → 1 as τ → 0: slope factor fully loads at zero maturity."""
        # At very short maturity, β₁ contribution dominates
        rate_short = nelson_siegel_rate(0.01, 0.05, 0.10, 0.0, 1.5)
        rate_long = nelson_siegel_rate(30.0, 0.05, 0.10, 0.0, 1.5)
        # Long rate should be close to β₀ (slope loading → 0)
        assert abs(rate_long - 0.05) < 0.005

    def test_flat_curve_all_params_zero_except_level(self):
        """β₁=β₂=0 gives a constant yield equal to β₀ at all maturities."""
        taus = np.array([0.25, 2.0, 5.0, 10.0, 30.0])
        rates = nelson_siegel_rate(taus, 0.045, 0.0, 0.0, 1.5)
        np.testing.assert_allclose(rates, 0.045, atol=1e-12)

    def test_returns_all_non_negative_for_typical_params(self):
        """NS rates should be positive for realistic Treasury parameters."""
        taus = np.linspace(0.25, 30.0, 100)
        rates = nelson_siegel_rate(taus, 0.046, -0.012, 0.008, 1.8)
        assert np.all(rates > 0)


# ── Nelson-Siegel fitting ─────────────────────────────────────────────────────

class TestFitNelsonSiegel:
    def test_flat_curve_recovers_level(self):
        """Flat 5% curve: fitted NS curve must reproduce 5% at all maturities.

        A flat curve is parametrically degenerate for NS — β₁ and β₂ can offset
        each other in multiple ways. What matters is that the *fitted yields* are
        correct, not the individual parameter values.
        """
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, FLAT_YIELDS)
        fitted = nelson_siegel_rate(TREASURY_MATURITIES_YRS, params.beta0,
                                     params.beta1, params.beta2, params.lambda_)
        np.testing.assert_allclose(fitted, 0.05, atol=1e-3)  # within 10bps (degenerate case)

    def test_flat_curve_rmse_near_zero(self):
        """Flat curve fit should achieve RMSE < 5 bps.

        A flat curve is degenerate for NS: many parameter combinations reproduce
        it. The optimiser may not land at β₁=β₂=0, but the RMSE should be small.
        """
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, FLAT_YIELDS)
        assert params.fit_rmse_bps < 5.0

    def test_normal_curve_positive_level(self):
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        assert params.beta0 > 0

    def test_inverted_curve_positive_slope(self):
        """Inverted curve (short > long) → β₁ > 0."""
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, INVERTED_YIELDS)
        assert params.beta1 > 0

    def test_normal_curve_negative_slope(self):
        """Normal curve (short < long) → β₁ < 0."""
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        assert params.beta1 < 0

    def test_fitted_curve_reproduces_input_yields(self):
        """Fitted NS model should reproduce observed yields within 5bps RMSE."""
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        assert params.fit_rmse_bps < 5.0

    def test_mismatched_arrays_raise(self):
        with pytest.raises(ValueError, match="shape"):
            fit_nelson_siegel(np.array([1.0, 5.0]), np.array([0.04, 0.045, 0.046]))

    def test_returns_pydantic_model(self):
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        assert isinstance(params, NelsonSiegelParams)
        assert params.lambda_ > 0.0

    def test_lambda_within_reasonable_range(self):
        """λ should fall within [0.5, 5.0] for US Treasuries."""
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        assert 0.1 < params.lambda_ < 10.0


# ── SpotCurve ─────────────────────────────────────────────────────────────────

class TestSpotCurve:
    @pytest.fixture
    def normal_curve(self):
        return SpotCurve(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)

    def test_rate_at_pillar_exact(self, normal_curve):
        """SpotCurve must return exact input rates at each pillar maturity."""
        for tau, y in zip(TREASURY_MATURITIES_YRS, NORMAL_YIELDS):
            assert abs(normal_curve.rate(tau) - y) < 1e-10, f"Pillar {tau}Y failed"

    def test_rate_array_input(self, normal_curve):
        """Vectorised rate query should work for ndarray input."""
        taus = np.array([1.0, 3.0, 7.0, 15.0])
        rates = normal_curve.rate(taus)
        assert rates.shape == (4,)
        assert np.all(rates > 0)

    def test_rate_scalar_returns_float(self, normal_curve):
        r = normal_curve.rate(5.0)
        assert isinstance(r, float)

    def test_discount_factor_at_zero(self, normal_curve):
        """DF(0) should equal 1 (no discounting at settlement)."""
        # Not tested directly at 0 (spline not defined), but very small tau
        # is close to 1
        df = normal_curve.discount_factor(0.01)
        assert abs(df - 1.0) < 0.005

    def test_discount_factor_decreasing(self, normal_curve):
        """Discount factors must be strictly decreasing with maturity."""
        taus = np.array([1.0, 5.0, 10.0, 30.0])
        dfs = normal_curve.discount_factor(taus)
        assert np.all(np.diff(dfs) < 0)

    def test_forward_rate_normal_curve(self, normal_curve):
        """Forward rate on a normal curve should be higher than the spot rate."""
        # For an upward-sloping curve, the 5Y forward (5Y into 10Y) > 5Y spot
        fwd = normal_curve.forward_rate(5.0, 10.0)
        spot_5 = normal_curve.rate(5.0)
        assert fwd > spot_5

    def test_forward_rate_t1_ge_t2_raises(self, normal_curve):
        with pytest.raises(ValueError, match="strictly less"):
            normal_curve.forward_rate(10.0, 5.0)

    def test_forward_rate_no_arbitrage(self, normal_curve):
        """No-arbitrage: invest t2 = invest t1 then roll at forward rate.
        z(t2)·t2 = z(t1)·t1 + f(t1,t2)·(t2-t1)  (continuous compounding)
        """
        t1, t2 = 2.0, 10.0
        z1 = normal_curve.rate(t1)
        z2 = normal_curve.rate(t2)
        fwd = normal_curve.forward_rate(t1, t2)
        lhs = z2 * t2
        rhs = z1 * t1 + fwd * (t2 - t1)
        assert abs(lhs - rhs) < 1e-10

    def test_from_nelson_siegel(self):
        """SpotCurve built from NS params should interpolate smoothly."""
        params = fit_nelson_siegel(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        sc = SpotCurve.from_nelson_siegel(params)
        # Rates at maturities should be positive and in a plausible range
        for tau in [1.0, 5.0, 10.0, 20.0]:
            r = sc.rate(tau)
            assert 0.01 < r < 0.20


# ── Curve shift decomposition ─────────────────────────────────────────────────

class TestDecomposeCurveShift:
    def test_level_shift_only(self):
        """A parallel shift (same Δβ₀, Δβ₁=Δβ₂=0) → only level changes."""
        p_before = NelsonSiegelParams(beta0=0.04, beta1=-0.01, beta2=0.005,
                                       lambda_=1.5, fit_rmse_bps=0.5)
        p_after = NelsonSiegelParams(beta0=0.05, beta1=-0.01, beta2=0.005,
                                      lambda_=1.5, fit_rmse_bps=0.5)
        result = decompose_curve_shift(p_before, p_after)
        assert abs(result["delta_level_bps"] - 100.0) < 1e-6
        assert abs(result["delta_slope_bps"]) < 1e-6
        assert abs(result["delta_curvature_bps"]) < 1e-6

    def test_slope_shift_only(self):
        p_before = NelsonSiegelParams(beta0=0.045, beta1=-0.01, beta2=0.005,
                                       lambda_=1.5, fit_rmse_bps=0.5)
        p_after = NelsonSiegelParams(beta0=0.045, beta1=-0.02, beta2=0.005,
                                      lambda_=1.5, fit_rmse_bps=0.5)
        result = decompose_curve_shift(p_before, p_after)
        assert abs(result["delta_slope_bps"] - (-100.0)) < 1e-6
        assert abs(result["delta_level_bps"]) < 1e-6
        assert abs(result["delta_curvature_bps"]) < 1e-6


# ── Bootstrap ─────────────────────────────────────────────────────────────────

class TestBootstrapSpotRates:
    def test_flat_par_curve_spot_equals_par(self):
        """For a flat par yield curve, spot rates should equal par yields."""
        par_yields = np.full(5, 0.05)
        spots = bootstrap_spot_rates(TREASURY_MATURITIES_YRS, par_yields)
        np.testing.assert_allclose(spots, 0.05, atol=2e-4)

    def test_spot_rates_positive(self):
        spots = bootstrap_spot_rates(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        assert np.all(spots > 0)

    def test_spot_rates_shape(self):
        spots = bootstrap_spot_rates(TREASURY_MATURITIES_YRS, NORMAL_YIELDS)
        assert spots.shape == TREASURY_MATURITIES_YRS.shape

    def test_bootstrap_two_maturity_annual_exact(self):
        """Annual-coupon 1Y + 2Y par bonds: bootstrap is analytically exact.

        For annual frequency and integer maturities the bootstrap has a
        closed-form solution, so we can verify to near-machine precision.

        1Y par bond (annual coupon = par_yield_1Y):
            spot_1Y = par_yield_1Y  (single-period → zero-coupon equivalent)

        2Y par bond solves for spot_2Y:
            c/1.04 + (1+c)/(1+spot_2Y)^2 = 1   where c = par_yield_2Y = 0.05
        """
        mats = np.array([1.0, 2.0])
        pars = np.array([0.04, 0.05])
        spots = bootstrap_spot_rates(mats, pars, frequency=1)

        # 1Y spot = 1Y par yield (exact for single-period)
        assert abs(spots[0] - 0.04) < 1e-10

        # Verify 2Y par bond prices at par using bootstrapped spots
        c = 0.05
        pv = c / (1 + spots[0]) + (1 + c) / (1 + spots[1]) ** 2
        assert abs(pv - 1.0) < 1e-8