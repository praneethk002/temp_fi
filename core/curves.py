"""
Yield curve modelling: Nelson-Siegel (1987) fitting, spot rate bootstrapping,
and cubic spline interpolation.

Nelson-Siegel model:
    r(τ) = β₀ + β₁·L(τ) + β₂·C(τ)
    L(τ) = (1 − e^{−τ/λ}) / (τ/λ)   # slope loading
    C(τ) = L(τ) − e^{−τ/λ}           # curvature loading

β₀ = long-run level, β₁ = slope, β₂ = curvature, λ = decay speed (years).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, brentq, OptimizeResult
from pydantic import BaseModel, Field

# ── Maturity grid matching FRED series DGS3MO/DGS2/DGS5/DGS10/DGS30 ───────
TREASURY_MATURITIES_YRS: np.ndarray = np.array([0.25, 2.0, 5.0, 10.0, 30.0])

# ── Nelson-Siegel parameter bounds (economically motivated) ─────────────────
_NS_BOUNDS = [
    (0.000, 0.200),   # β₀: positive nominal long-run level
    (-0.150, 0.150),  # β₁: slope (positive = inverted, negative = normal)
    (-0.150, 0.150),  # β₂: curvature
    (0.100, 10.000),  # λ:  decay speed in years
]


class NelsonSiegelParams(BaseModel):
    """Fitted parameters from a Nelson-Siegel yield curve calibration.

    All rate parameters are stored as decimals (e.g. 0.045 for 4.5%).
    ``fit_rmse_bps`` is the root-mean-squared fitting error in basis points;
    a value below 2bps indicates an excellent fit.
    """

    beta0: float = Field(..., description="Long-run level (τ→∞), decimal")
    beta1: float = Field(..., description="Slope factor, decimal")
    beta2: float = Field(..., description="Curvature / hump factor, decimal")
    lambda_: float = Field(..., gt=0.0, description="Exponential decay speed, years")
    fit_rmse_bps: float = Field(..., ge=0.0, description="Root-mean-square fitting error, bps")


# ── Core Nelson-Siegel functions ─────────────────────────────────────────────

def nelson_siegel_rate(
    tau: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    lambda_: float,
) -> float | np.ndarray:
    """Evaluate the Nelson-Siegel model at one or more maturities.

    Fully vectorised: ``tau`` can be a scalar or any-shape ndarray.

    Args:
        tau: Maturity in years (must be > 0).
        beta0: Long-run level parameter.
        beta1: Slope parameter.
        beta2: Curvature parameter.
        lambda_: Decay speed (years).

    Returns:
        Model yield(s) as decimal(s), same shape as ``tau``.
    """
    tau = np.asarray(tau, dtype=float)
    x = tau / lambda_                              # normalised maturity
    loading_slope = (1.0 - np.exp(-x)) / x        # monotone decay: 1→0
    loading_curve = loading_slope - np.exp(-x)     # hump: 0, peaks near λ, →0

    rates = beta0 + beta1 * loading_slope + beta2 * loading_curve
    return float(rates) if rates.ndim == 0 else rates


def fit_nelson_siegel(
    maturities_yrs: np.ndarray,
    yields_decimal: np.ndarray,
) -> NelsonSiegelParams:
    """Fit the Nelson-Siegel model via L-BFGS-B minimisation of SSE.

    Args:
        maturities_yrs: Pillar maturities in years, shape (N,), sorted ascending.
        yields_decimal: Observed yields as decimals, shape (N,).

    Returns:
        NelsonSiegelParams with fitted parameters and RMSE in bps.

    Raises:
        ValueError: If arrays have mismatched lengths.
        RuntimeError: If L-BFGS-B fails to converge.
    """
    maturities = np.asarray(maturities_yrs, dtype=float)
    yields = np.asarray(yields_decimal, dtype=float)

    if maturities.shape != yields.shape:
        raise ValueError(
            f"maturities shape {maturities.shape} != yields shape {yields.shape}"
        )

    x0 = np.array([
        yields[-1],             # β₀: long-end yield
        yields[0] - yields[-1], # β₁: short − long
        0.0,                    # β₂: no curvature prior
        1.5,                    # λ:  typical US calibration
    ])

    def _sse(params: np.ndarray) -> float:
        b0, b1, b2, lam = params
        model = nelson_siegel_rate(maturities, b0, b1, b2, lam)
        return float(np.sum((model - yields) ** 2))

    result: OptimizeResult = minimize(
        _sse, x0, method="L-BFGS-B", bounds=_NS_BOUNDS,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
    )
    if not result.success and result.fun > 1e-10:
        raise RuntimeError(
            f"Nelson-Siegel fit did not converge: {result.message}"
        )

    b0, b1, b2, lam = result.x
    residuals = nelson_siegel_rate(maturities, b0, b1, b2, lam) - yields
    rmse_bps = float(np.sqrt(np.mean(residuals**2))) * 10_000.0

    return NelsonSiegelParams(
        beta0=float(b0),
        beta1=float(b1),
        beta2=float(b2),
        lambda_=float(lam),
        fit_rmse_bps=rmse_bps,
    )


def decompose_curve_shift(
    params_before: NelsonSiegelParams,
    params_after: NelsonSiegelParams,
) -> dict[str, float]:
    """Decompose a curve move into changes in NS level, slope, and curvature.

    Args:
        params_before: NS parameters at the start of the period.
        params_after: NS parameters at the end of the period.

    Returns:
        Dict with "delta_level_bps", "delta_slope_bps", "delta_curvature_bps",
        and "delta_lambda" (dimensionless, in years).
    """
    return {
        "delta_level_bps":     (params_after.beta0  - params_before.beta0)  * 10_000,
        "delta_slope_bps":     (params_after.beta1  - params_before.beta1)  * 10_000,
        "delta_curvature_bps": (params_after.beta2  - params_before.beta2)  * 10_000,
        "delta_lambda":         params_after.lambda_ - params_before.lambda_,
    }


# ── Spot curve interpolation ─────────────────────────────────────────────────

class SpotCurve:
    """Cubic spline interpolation of a zero-coupon spot rate curve.

    Discount factors use continuous compounding: DF(τ) = e^{−z(τ)·τ}.
    Spline uses not-a-knot boundary conditions (scipy default).

    Args:
        maturities_yrs: Pillar maturities in years, sorted ascending.
        spot_rates_decimal: Spot rates at each pillar as decimals.
    """

    def __init__(
        self,
        maturities_yrs: np.ndarray,
        spot_rates_decimal: np.ndarray,
    ) -> None:
        self._maturities = np.asarray(maturities_yrs, dtype=float)
        self._rates = np.asarray(spot_rates_decimal, dtype=float)
        self._spline = CubicSpline(self._maturities, self._rates, bc_type="not-a-knot")

    def rate(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Interpolated spot rate at maturity ``tau``.

        Args:
            tau: Maturity in years (scalar or array).

        Returns:
            Spot rate as decimal, same shape as ``tau``.
        """
        scalar = np.isscalar(tau)
        result = self._spline(np.asarray(tau, dtype=float))
        return float(result) if scalar else result

    def discount_factor(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Continuous-compounding discount factor: DF(τ) = e^{−z(τ)·τ}.

        Args:
            tau: Maturity in years (scalar or array).

        Returns:
            Discount factor, same shape as ``tau``.
        """
        z = self.rate(tau)
        t = np.asarray(tau, dtype=float) if not np.isscalar(tau) else tau
        return np.exp(-z * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward rate for [t1, t2] via no-arbitrage: [z(t2)·t2 − z(t1)·t1] / (t2 − t1).

        Args:
            t1: Start of forward period in years.
            t2: End of forward period in years (must be > t1).

        Returns:
            Forward rate as decimal.

        Raises:
            ValueError: If t1 >= t2.
        """
        if t1 >= t2:
            raise ValueError(f"t1={t1} must be strictly less than t2={t2}")
        z1 = self.rate(t1)
        z2 = self.rate(t2)
        return (z2 * t2 - z1 * t1) / (t2 - t1)

    @classmethod
    def from_nelson_siegel(
        cls,
        params: NelsonSiegelParams,
        maturities_yrs: np.ndarray = TREASURY_MATURITIES_YRS,
    ) -> "SpotCurve":
        """Build a SpotCurve from Nelson-Siegel fitted parameters.

        Uses the NS model to generate spot rates at standard Treasury maturities,
        then wraps them in a CubicSpline for continuous interpolation.

        Args:
            params: Fitted :class:`NelsonSiegelParams`.
            maturities_yrs: Maturity grid for spline knots.

        Returns:
            :class:`SpotCurve` backed by NS model rates.
        """
        rates = nelson_siegel_rate(
            maturities_yrs, params.beta0, params.beta1, params.beta2, params.lambda_
        )
        return cls(maturities_yrs, rates)


# ── Spot rate bootstrapping ───────────────────────────────────────────────────

def bootstrap_spot_rates(
    maturities_yrs: np.ndarray,
    par_yields_decimal: np.ndarray,
    frequency: int = 2,
) -> np.ndarray:
    """Strip zero rates from par yields by sequential bootstrapping.

    For each maturity, solves for z_n such that the par bond prices at par
    given all previously bootstrapped spot rates. Uses Brent's method.

    Args:
        maturities_yrs: Sorted pillar maturities in years, shape (N,).
        par_yields_decimal: Par yields as decimals, shape (N,).
        frequency: Coupon payments per year (default 2, semi-annual).

    Returns:
        Bootstrapped spot rates as decimals, shape (N,).

    Raises:
        ValueError: If the root cannot be bracketed at any pillar.
    """
    maturities = np.asarray(maturities_yrs, dtype=float)
    par_yields = np.asarray(par_yields_decimal, dtype=float)

    n_pillars = len(maturities)
    spot_rates = np.zeros(n_pillars)

    for i, (mat, par_yield) in enumerate(zip(maturities, par_yields)):
        n_periods = int(mat * frequency)  # floor: number of full coupon periods

        if n_periods == 0:
            # Short-end discount instrument (e.g. 3M T-bill with semi-annual frequency).
            # A T-bill is zero-coupon: its par yield equals its spot rate directly.
            spot_rates[i] = par_yield
            continue

        coupon = par_yield / frequency
        period_times = np.arange(1, n_periods + 1, dtype=float) / frequency

        # Interpolation function for already-bootstrapped short-end spot rates.
        # CubicSpline requires ≥ 2 knots; use linear for 1 or 2 known points.
        n_known = i  # number of already-bootstrapped pillars
        if n_known == 0:
            def _interp_z(tau: float) -> float: return 0.0  # unused (n_periods=0 handled above)
        elif n_known == 1:
            _z0 = spot_rates[0]
            def _interp_z(tau: float) -> float: return float(_z0)  # flat extrapolation
        elif n_known == 2:
            _m2, _z2 = maturities[:2], spot_rates[:2]
            def _interp_z(tau: float) -> float: return float(np.interp(tau, _m2, _z2))
        else:
            _spline = CubicSpline(maturities[:n_known], spot_rates[:n_known], bc_type="not-a-knot")
            def _interp_z(tau: float) -> float: return float(_spline(tau))

        _interp_z_cap = _interp_z  # capture for closure

        def _pv_minus_par(z_n: float, _times=period_times, _c=coupon,
                          _f=frequency, _interp=_interp_z_cap, _i=i) -> float:
            pv = 0.0
            for t in _times[:-1]:
                z_t = _interp(t) if _i > 0 else z_n
                pv += _c / (1.0 + z_t / _f) ** (t * _f)
            pv += (1.0 + _c) / (1.0 + z_n / _f) ** (_times[-1] * _f)
            return pv - 1.0

        try:
            spot_rates[i] = brentq(_pv_minus_par, 0.0, 0.50, xtol=1e-12)
        except ValueError as exc:
            raise ValueError(
                f"Bootstrap failed at maturity {mat}Y (pillar {i}): "
                f"root not in [0%, 50%]. Par yield={par_yield*100:.3f}%."
            ) from exc

    return spot_rates