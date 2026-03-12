"""Tests for data/market_data.py — bond pricing from yield curve."""

from datetime import date
from unittest.mock import patch

import pytest

from core.basket import get_basket
from data.market_data import _interpolate_yield, get_bond_prices

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NORMAL_CURVE = {
    "3M": 0.042,
    "2Y": 0.044,
    "5Y": 0.045,
    "7Y": 0.0455,
    "10Y": 0.046,
    "30Y": 0.048,
}


# ---------------------------------------------------------------------------
# TestInterpolateYield
# ---------------------------------------------------------------------------

class TestInterpolateYield:
    """Test _interpolate_yield directly with a known six-point curve."""

    def test_exact_pillar_2y(self):
        result = _interpolate_yield(2.0, NORMAL_CURVE)
        assert result == 0.044

    def test_exact_pillar_10y(self):
        result = _interpolate_yield(10.0, NORMAL_CURVE)
        assert result == 0.046

    def test_midpoint_interpolation(self):
        # Midpoint between 5Y (0.045) and 7Y (0.0455) is 6.0Y → 0.04525
        # Linear interpolation: 0.045 + 0.5 * (0.0455 - 0.045) = 0.04525
        result = _interpolate_yield(6.0, NORMAL_CURVE)
        assert abs(result - 0.04525) < 1e-10

    def test_below_shortest_maturity_clamps_to_short_rate(self):
        # 0.1 years is below 3M (0.25 years) — should clamp to 3M rate
        result = _interpolate_yield(0.1, NORMAL_CURVE)
        assert result == NORMAL_CURVE["3M"]

    def test_above_longest_maturity_clamps_to_long_rate(self):
        # 35 years is above 30Y — should clamp to 30Y rate
        result = _interpolate_yield(35.0, NORMAL_CURVE)
        assert result == NORMAL_CURVE["30Y"]

    def test_fewer_than_2_points_raises(self):
        with pytest.raises(ValueError):
            _interpolate_yield(5.0, {"10Y": 0.046})


# ---------------------------------------------------------------------------
# TestGetBondPricesOverrideOnly
# ---------------------------------------------------------------------------

class TestGetBondPricesOverrideOnly:
    """Test get_bond_prices with use_fred=False — only manual overrides."""

    def setup_method(self):
        self.basket = get_basket(use_api=False)
        self.first_cusip = self.basket[0]["cusip"]

    def test_override_prices_returned(self):
        prices = get_bond_prices(
            self.basket,
            use_fred=False,
            overrides={self.first_cusip: 97.25},
        )
        assert prices[self.first_cusip] == 97.25

    def test_no_overrides_empty_result(self):
        prices = get_bond_prices(self.basket, use_fred=False)
        assert prices == {}

    def test_only_overridden_bonds_present(self):
        # Supply exactly 1 override for a 12-bond basket
        assert len(self.basket) == 12
        prices = get_bond_prices(
            self.basket,
            use_fred=False,
            overrides={self.first_cusip: 98.50},
        )
        assert len(prices) == 1

    def test_price_is_float(self):
        prices = get_bond_prices(
            self.basket,
            use_fred=False,
            overrides={self.first_cusip: 97.25},
        )
        assert isinstance(prices[self.first_cusip], float)


# ---------------------------------------------------------------------------
# TestGetBondPricesWithFred
# ---------------------------------------------------------------------------

class TestGetBondPricesWithFred:
    """Test get_bond_prices with a mocked yield curve (no real FRED call)."""

    def setup_method(self):
        self.basket = get_basket(use_api=False)
        self.first_cusip = self.basket[0]["cusip"]
        self.as_of = date(2026, 3, 12)

    def test_all_basket_bonds_priced(self):
        with patch("data.market_data.get_yield_curve", return_value=NORMAL_CURVE):
            prices = get_bond_prices(self.basket, as_of=self.as_of)
        assert set(b["cusip"] for b in self.basket) == set(prices.keys())

    def test_prices_are_positive(self):
        with patch("data.market_data.get_yield_curve", return_value=NORMAL_CURVE):
            prices = get_bond_prices(self.basket, as_of=self.as_of)
        assert all(p > 0 for p in prices.values())

    def test_prices_are_percentage_of_par(self):
        with patch("data.market_data.get_yield_curve", return_value=NORMAL_CURVE):
            prices = get_bond_prices(self.basket, as_of=self.as_of)
        assert all(50 < p < 150 for p in prices.values())

    def test_override_takes_precedence(self):
        with patch("data.market_data.get_yield_curve", return_value=NORMAL_CURVE):
            prices = get_bond_prices(
                self.basket,
                as_of=self.as_of,
                overrides={self.first_cusip: 99.99},
            )
        assert prices[self.first_cusip] == 99.99

    def test_as_of_passed_to_fred(self):
        with patch("data.market_data.get_yield_curve", return_value=NORMAL_CURVE) as mock_gyr:
            get_bond_prices(self.basket, as_of=self.as_of)
        mock_gyr.assert_called_once()
        assert mock_gyr.call_args.kwargs.get("as_of") == self.as_of