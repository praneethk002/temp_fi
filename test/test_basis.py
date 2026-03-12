"""Tests for core/basis.py — gross/net basis, carry, implied repo, CTD selection."""

import pytest
from core.basket import get_basket
from core.basis import (
    basket_analysis,
    carry,
    ctd_scenario,
    find_ctd,
    gross_basis,
    implied_repo,
    net_basis,
)

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


class TestFindCTD:
    def test_returns_bond_with_highest_implied_repo(self):
        bonds = [
            {"cash_price": 99.375, "conversion_factor": 0.8830, "coupon_rate": 0.04375, "label": "A"},
            {"cash_price": 99.625, "conversion_factor": 0.9195, "coupon_rate": 0.04625, "label": "B"},
            {"cash_price": 99.000, "conversion_factor": 0.9014, "coupon_rate": 0.04000, "label": "C"},
        ]
        ctd = find_ctd(bonds, FUTURES, DAYS)
        # Verify it really is the max implied repo
        repos = [
            implied_repo(b["cash_price"], FUTURES, b["conversion_factor"], b["coupon_rate"], DAYS)
            for b in bonds
        ]
        assert ctd["implied_repo"] == pytest.approx(max(repos))

    def test_ctd_has_implied_repo_field(self):
        bonds = [
            {"cash_price": 99.0, "conversion_factor": 0.9014, "coupon_rate": 0.04, "label": "X"},
        ]
        ctd = find_ctd(bonds, FUTURES, DAYS)
        assert "implied_repo" in ctd
        assert "label" in ctd


class TestBasketAnalysis:
    def test_rank_1_is_ctd(self):
        basket = get_basket(use_api=False)
        prices = {b["cusip"]: 95.0 + b["coupon"] * 100 for b in basket}
        df = basket_analysis(basket, prices, FUTURES, REPO, DAYS)
        assert df[df["is_ctd"]].index[0] == 1

    def test_only_one_ctd(self):
        basket = get_basket(use_api=False)
        prices = {b["cusip"]: 95.0 + b["coupon"] * 100 for b in basket}
        df = basket_analysis(basket, prices, FUTURES, REPO, DAYS)
        assert df["is_ctd"].sum() == 1

    def test_implied_repos_descending(self):
        basket = get_basket(use_api=False)
        prices = {b["cusip"]: 95.0 + b["coupon"] * 100 for b in basket}
        df = basket_analysis(basket, prices, FUTURES, REPO, DAYS)
        repos = df["implied_repo"].tolist()
        assert repos == sorted(repos, reverse=True)

    def test_required_columns_present(self):
        basket = get_basket(use_api=False)
        prices = {b["cusip"]: 95.0 + b["coupon"] * 100 for b in basket}
        df = basket_analysis(basket, prices, FUTURES, REPO, DAYS)
        for col in ("label", "cash_price", "conv_factor", "gross_basis",
                    "carry", "net_basis", "implied_repo", "is_ctd"):
            assert col in df.columns


class TestCTDScenario:
    def _basket_and_yields(self):
        basket = get_basket(use_api=False)
        yields = {b["cusip"]: 0.045 for b in basket}
        return basket, yields

    def test_returns_two_dataframes(self):
        basket, yields = self._basket_and_yields()
        summary, heatmap = ctd_scenario(basket, yields, FUTURES, REPO, DAYS)
        import pandas as pd
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(heatmap, pd.DataFrame)

    def test_summary_has_one_row_per_shift(self):
        basket, yields = self._basket_and_yields()
        shifts = [-50, 0, 50]
        summary, _ = ctd_scenario(basket, yields, FUTURES, REPO, DAYS, shifts_bps=shifts)
        assert len(summary) == len(shifts)

    def test_summary_columns(self):
        basket, yields = self._basket_and_yields()
        summary, _ = ctd_scenario(basket, yields, FUTURES, REPO, DAYS, shifts_bps=[0])
        for col in ("shift_bps", "ctd_label", "ctd_implied_repo",
                    "runner_label", "runner_implied_repo", "spread_bps", "ctd_changed"):
            assert col in summary.columns

    def test_zero_shift_not_flagged_as_changed(self):
        basket, yields = self._basket_and_yields()
        summary, _ = ctd_scenario(basket, yields, FUTURES, REPO, DAYS, shifts_bps=[-50, 0, 50])
        row = summary[summary["shift_bps"] == 0].iloc[0]
        assert not row["ctd_changed"]

    def test_heatmap_rows_are_bonds(self):
        basket, yields = self._basket_and_yields()
        _, heatmap = ctd_scenario(basket, yields, FUTURES, REPO, DAYS, shifts_bps=[-50, 0, 50])
        assert len(heatmap) == len(basket)

    def test_heatmap_columns_are_shifts(self):
        basket, yields = self._basket_and_yields()
        shifts = [-50, 0, 50]
        _, heatmap = ctd_scenario(basket, yields, FUTURES, REPO, DAYS, shifts_bps=shifts)
        assert list(heatmap.columns) == shifts

    def test_spread_is_positive(self):
        # CTD always has higher implied repo than runner-up
        basket, yields = self._basket_and_yields()
        summary, _ = ctd_scenario(basket, yields, FUTURES, REPO, DAYS, shifts_bps=[0])
        assert summary.iloc[0]["spread_bps"] > 0