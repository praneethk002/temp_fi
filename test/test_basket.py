"""Tests for core/basket.py — conversion factor, basket contents, bond label."""

from datetime import date
from core.basket import (
    MAX_MATURITY,
    MIN_MATURITY,
    bond_label,
    conversion_factor,
    get_basket,
)


class TestConversionFactor:
    def test_6pct_coupon_cf_near_one(self):
        cf = conversion_factor(0.06, date(2034, 11, 15))
        assert abs(cf - 1.0) < 0.05

    def test_low_coupon_cf_below_one(self):
        cf = conversion_factor(0.03, date(2034, 11, 15))
        assert cf < 1.0

    def test_high_coupon_cf_above_one(self):
        cf = conversion_factor(0.08, date(2034, 11, 15))
        assert cf > 1.0

    def test_cf_increases_with_coupon(self):
        mat = date(2034, 11, 15)
        assert conversion_factor(0.03, mat) < conversion_factor(0.045, mat) < conversion_factor(0.07, mat)

    def test_cf_decreases_with_maturity_for_discount_bond(self):
        # Sub-6% coupon: longer maturity means more discounting at 6% base yield
        coupon = 0.04375
        assert conversion_factor(coupon, date(2033, 2, 15)) > conversion_factor(coupon, date(2036, 5, 15))

    def test_cf_rounded_to_4_decimal_places(self):
        cf = conversion_factor(0.04375, date(2034, 11, 15))
        assert cf == round(cf, 4)

    def test_cf_positive(self):
        assert conversion_factor(0.04, date(2034, 5, 15)) > 0


class TestGetBasket:
    def test_returns_nonempty_list(self):
        assert len(get_basket(use_api=False)) > 0

    def test_all_bonds_have_required_keys(self):
        for bond in get_basket(use_api=False):
            for key in ("cusip", "coupon", "maturity", "conv_factor"):
                assert key in bond

    def test_all_maturities_within_eligibility_window(self):
        for bond in get_basket(use_api=False):
            assert MIN_MATURITY <= bond["maturity"] <= MAX_MATURITY

    def test_sorted_by_maturity(self):
        basket = get_basket(use_api=False)
        maturities = [b["maturity"] for b in basket]
        assert maturities == sorted(maturities)

    def test_all_coupons_positive(self):
        for bond in get_basket(use_api=False):
            assert bond["coupon"] > 0

    def test_all_conv_factors_positive(self):
        for bond in get_basket(use_api=False):
            assert bond["conv_factor"] > 0

    def test_no_duplicate_cusips(self):
        basket = get_basket(use_api=False)
        cusips = [b["cusip"] for b in basket]
        assert len(cusips) == len(set(cusips))


class TestBondLabel:
    def test_label_contains_coupon(self):
        bond = {"cusip": "X", "coupon": 0.04375, "maturity": date(2034, 11, 15)}
        label = bond_label(bond)
        assert "4.38" in label or "4.375" in label

    def test_label_contains_year(self):
        bond = {"cusip": "X", "coupon": 0.045, "maturity": date(2034, 8, 15)}
        assert "34" in bond_label(bond)

    def test_label_contains_month(self):
        bond = {"cusip": "X", "coupon": 0.045, "maturity": date(2034, 8, 15)}
        assert "Aug" in bond_label(bond)

    def test_label_is_string(self):
        bond = {"cusip": "X", "coupon": 0.04, "maturity": date(2033, 2, 15)}
        assert isinstance(bond_label(bond), str)