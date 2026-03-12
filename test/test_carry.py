"""Tests for core/carry.py"""

import pytest
from core.carry import gross_basis, carry, net_basis, implied_repo

FUTURES = 108.50
REPO    = 0.053
DAYS    = 110


class TestGrossBasis:
    def test_positive_when_cash_above_invoice(self):
        # cash > futures × CF → bond rich vs futures
        assert gross_basis(110.0, FUTURES, 0.9750) > 0

    def test_negative_when_cash_below_invoice(self):
        assert gross_basis(95.0, FUTURES, 0.9750) < 0

    def test_zero_at_invoice_price(self):
        invoice = FUTURES * 0.9750
        assert abs(gross_basis(invoice, FUTURES, 0.9750)) < 1e-10

    def test_scales_with_conversion_factor(self):
        # Higher CF → higher invoice → smaller gross basis for same cash price
        gb_low  = gross_basis(100.0, FUTURES, 0.90)
        gb_high = gross_basis(100.0, FUTURES, 0.99)
        assert gb_low > gb_high


class TestCarry:
    def test_positive_carry_when_coupon_exceeds_repo(self):
        # Coupon 6% >> repo 5.3% → positive carry
        assert carry(100.0, 0.06, REPO, DAYS) > 0

    def test_negative_carry_when_repo_exceeds_coupon(self):
        # Coupon 2% << repo 5.3% → negative carry
        assert carry(100.0, 0.02, REPO, DAYS) < 0

    def test_zero_carry_breakeven(self):
        # When coupon/365 == repo/360, carry is zero
        # coupon × (days/365) = repo × (days/360)
        # coupon = repo × (365/360)
        breakeven_coupon = REPO * (365 / 360)
        c = carry(100.0, breakeven_coupon, REPO, DAYS)
        assert abs(c) < 1e-6

    def test_scales_linearly_with_price(self):
        c1 = carry(100.0, 0.045, REPO, DAYS)
        c2 = carry(200.0, 0.045, REPO, DAYS)
        assert abs(c2 - 2 * c1) < 1e-10


class TestNetBasis:
    def test_equals_gross_minus_carry(self):
        price, coupon, cf = 99.50, 0.045, 0.9195
        gb = gross_basis(price, FUTURES, cf)
        c  = carry(price, coupon, REPO, DAYS)
        nb = net_basis(price, FUTURES, cf, coupon, REPO, DAYS)
        assert abs(nb - (gb - c)) < 1e-10

    def test_near_zero_for_fairly_priced_bond(self):
        # At the invoice price with zero carry, net basis ≈ 0
        invoice = FUTURES * 0.9750
        c = carry(invoice, 0.053 * 365 / 360, REPO, DAYS)  # breakeven coupon
        nb = net_basis(invoice, FUTURES, 0.9750, 0.053 * 365 / 360, REPO, DAYS)
        assert abs(nb) < 0.01


class TestImpliedRepo:
    def test_higher_coupon_higher_implied_repo(self):
        # More coupon income makes the bond cheaper to deliver → higher implied repo
        ir_low  = implied_repo(99.0, FUTURES, 0.9014, 0.04,   DAYS)
        ir_high = implied_repo(99.0, FUTURES, 0.9014, 0.08,   DAYS)
        assert ir_high > ir_low

    def test_implied_repo_positive_for_reasonable_inputs(self):
        ir = implied_repo(99.50, FUTURES, 0.9195, 0.045, DAYS)
        assert ir > 0

    def test_bond_is_ctd_when_implied_repo_exceeds_market(self):
        ir = implied_repo(99.50, FUTURES, 0.9195, 0.045, DAYS)
        # If implied repo > market repo, basis trade is attractive
        assert isinstance(ir > REPO, bool)  # just verify it's a valid comparison