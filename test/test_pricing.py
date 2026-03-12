"""
Tests for core/pricing.py.

Reference values use a vanilla semi-annual 5% / 10yr bond at par as the
anchor — a well-known textbook example with closed-form expectations.
"""

import pytest
from core.pricing import (
    accrued_interest,
    convexity,
    dirty_price,
    dv01,
    macaulay_duration,
    modified_duration,
    price_bond,
    ytm,
)

# ---------------------------------------------------------------------------
# Shared fixture: 5% semi-annual 10yr bond priced at par
# ---------------------------------------------------------------------------
FACE        = 100.0
COUPON      = 0.05   # 5% annual, paid semi-annually
YEARS       = 10.0
YTM_PAR     = 0.05   # par bond ⟹ price == face


class TestPriceBond:
    def test_par_bond(self):
        p = price_bond(FACE, COUPON, YEARS, YTM_PAR)
        assert abs(p - 100.0) < 1e-8

    def test_discount_bond(self):
        # Higher yield → price < par
        p = price_bond(FACE, COUPON, YEARS, 0.06)
        assert p < 100.0

    def test_premium_bond(self):
        # Lower yield → price > par
        p = price_bond(FACE, COUPON, YEARS, 0.04)
        assert p > 100.0

    def test_zero_coupon(self):
        # Zero-coupon bond: price = face / (1 + y/2)^(2*n)
        p = price_bond(FACE, 0.0, YEARS, 0.05)
        expected = FACE / (1.0 + 0.025) ** 20
        assert abs(p - expected) < 1e-8

    def test_annual_frequency(self):
        # Annual coupon bond at par when coupon == yield
        p = price_bond(FACE, COUPON, YEARS, YTM_PAR, frequency=1)
        assert abs(p - FACE) < 1e-8

    def test_pull_to_par(self):
        # Par bond stays at par regardless of remaining maturity
        assert abs(price_bond(FACE, COUPON, 1.0, YTM_PAR) - FACE) < 1e-8
        assert abs(price_bond(FACE, COUPON, 0.5, YTM_PAR) - FACE) < 1e-8

    def test_face_scaling(self):
        # Price should scale linearly with face value
        p1 = price_bond(100, COUPON, YEARS, YTM_PAR)
        p2 = price_bond(200, COUPON, YEARS, YTM_PAR)
        assert abs(p2 - 2 * p1) < 1e-8


class TestAccruedInterest:
    def test_at_coupon_date(self):
        # Zero days elapsed → zero accrued
        ai = accrued_interest(FACE, COUPON, 2, days_since_last_coupon=0, days_in_coupon_period=181)
        assert ai == 0.0

    def test_half_period(self):
        # Halfway through period → half the coupon
        coupon_pmt = FACE * COUPON / 2   # = 2.5
        ai = accrued_interest(FACE, COUPON, 2, days_since_last_coupon=90, days_in_coupon_period=180)
        assert abs(ai - coupon_pmt * 0.5) < 1e-8

    def test_full_period(self):
        # Full period → entire coupon (happens at ex-dividend settlement)
        coupon_pmt = FACE * COUPON / 2
        ai = accrued_interest(FACE, COUPON, 2, days_since_last_coupon=181, days_in_coupon_period=181)
        assert abs(ai - coupon_pmt) < 1e-8


class TestDirtyPrice:
    def test_on_coupon_date_equals_clean(self):
        clean = price_bond(FACE, COUPON, YEARS, YTM_PAR)
        dp    = dirty_price(FACE, COUPON, YEARS, YTM_PAR, 0, 181)
        assert abs(dp - clean) < 1e-8

    def test_dirty_exceeds_clean_between_coupons(self):
        clean = price_bond(FACE, COUPON, YEARS, YTM_PAR)
        dp    = dirty_price(FACE, COUPON, YEARS, YTM_PAR, 45, 181)
        assert dp > clean


class TestYTM:
    def test_par_bond_ytm(self):
        p = price_bond(FACE, COUPON, YEARS, YTM_PAR)
        y = ytm(p, FACE, COUPON, YEARS)
        assert abs(y - YTM_PAR) < 1e-8

    def test_roundtrip_discount(self):
        y_input = 0.065
        p = price_bond(FACE, COUPON, YEARS, y_input)
        y_solved = ytm(p, FACE, COUPON, YEARS)
        assert abs(y_solved - y_input) < 1e-8

    def test_roundtrip_premium(self):
        y_input = 0.035
        p = price_bond(FACE, COUPON, YEARS, y_input)
        y_solved = ytm(p, FACE, COUPON, YEARS)
        assert abs(y_solved - y_input) < 1e-8

    def test_impossible_price_raises(self):
        with pytest.raises(ValueError):
            ytm(-10.0, FACE, COUPON, YEARS)


class TestMacaulayDuration:
    def test_par_bond_mac_duration(self):
        # For a par bond: D_mac = (1 + y/m)/y * (1 - 1/(1+y/m)^(m*n))
        y, m, n = YTM_PAR, 2, YEARS
        expected = (1 + y/m) / y * (1 - 1 / (1 + y/m) ** (m * n))
        d = macaulay_duration(FACE, COUPON, YEARS, YTM_PAR)
        assert abs(d - expected) < 1e-6

    def test_zero_coupon_mac_duration_equals_maturity(self):
        # Zero-coupon bond: Macaulay duration == years_to_maturity
        d = macaulay_duration(FACE, 0.0, YEARS, 0.05)
        assert abs(d - YEARS) < 1e-6

    def test_longer_maturity_increases_duration(self):
        d5  = macaulay_duration(FACE, COUPON, 5.0,  YTM_PAR)
        d10 = macaulay_duration(FACE, COUPON, 10.0, YTM_PAR)
        d20 = macaulay_duration(FACE, COUPON, 20.0, YTM_PAR)
        assert d5 < d10 < d20


class TestModifiedDuration:
    def test_relationship_to_macaulay(self):
        mac = macaulay_duration(FACE, COUPON, YEARS, YTM_PAR)
        mod = modified_duration(FACE, COUPON, YEARS, YTM_PAR)
        assert abs(mod - mac / (1 + YTM_PAR / 2)) < 1e-10

    def test_par_bond_approx(self):
        # For a 10yr par bond, mod duration ≈ 7.7–8.0 years
        mod = modified_duration(FACE, COUPON, YEARS, YTM_PAR)
        assert 7.5 < mod < 8.5


class TestDV01:
    def test_par_bond_dv01(self):
        # DV01 ≈ modified_duration * price * 0.0001
        mod = modified_duration(FACE, COUPON, YEARS, YTM_PAR)
        expected_dv01 = mod * 100.0 * 0.0001
        d = dv01(FACE, COUPON, YEARS, YTM_PAR)
        assert abs(d - expected_dv01) / expected_dv01 < 0.001   # within 0.1%

    def test_positive_for_normal_bond(self):
        # Price falls when yield rises → DV01 must be positive
        assert dv01(FACE, COUPON, YEARS, YTM_PAR) > 0

    def test_longer_maturity_has_higher_dv01(self):
        d5  = dv01(FACE, COUPON, 5.0,  YTM_PAR)
        d10 = dv01(FACE, COUPON, 10.0, YTM_PAR)
        d30 = dv01(FACE, COUPON, 30.0, YTM_PAR)
        assert d5 < d10 < d30


class TestConvexity:
    def test_positive_convexity(self):
        # Plain-vanilla bonds always have positive convexity
        c = convexity(FACE, COUPON, YEARS, YTM_PAR)
        assert c > 0

    def test_longer_maturity_more_convex(self):
        c5  = convexity(FACE, COUPON, 5.0,  YTM_PAR)
        c10 = convexity(FACE, COUPON, 10.0, YTM_PAR)
        c30 = convexity(FACE, COUPON, 30.0, YTM_PAR)
        assert c5 < c10 < c30

    def test_par_bond_convexity_range(self):
        # For a 10yr 5% par bond, convexity is typically 65–85 yr²
        c = convexity(FACE, COUPON, YEARS, YTM_PAR)
        assert 60 < c < 100

    def test_lower_coupon_more_convex(self):
        # Lower coupon → cash flows more back-loaded → higher convexity
        c_high = convexity(FACE, 0.08, YEARS, 0.08)
        c_low  = convexity(FACE, 0.02, YEARS, 0.02)
        assert c_low > c_high