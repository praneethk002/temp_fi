from datetime import date

import pandas as pd
import pytest

from core.basket import get_basket
from core.scenario import shocked_basket, scenario_grid

FUTURES = 108.50
REPO    = 0.053
DAYS    = 110
AS_OF   = date(2026, 1, 15)  # fixed date so tests are deterministic


@pytest.fixture
def basket_and_yields():
    basket = get_basket(use_api=False)
    yields = {b["cusip"]: 0.045 for b in basket}
    return basket, yields


class TestShockedBasket:
    def test_returns_dataframe(self, basket_and_yields):
        basket, yields = basket_and_yields
        result = shocked_basket(basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF)
        assert isinstance(result, pd.DataFrame)

    def test_zero_shift_same_ctd_as_rank_basket(self, basket_and_yields):
        from core.ctd import rank_basket
        from core.pricing import price_bond

        basket, yields = basket_and_yields

        direct_prices = {
            b["cusip"]: price_bond(
                100.0,
                b["coupon"],
                (b["maturity"] - AS_OF).days / 365.25,
                yields[b["cusip"]],
            )
            for b in basket
            if b["cusip"] in yields
        }
        direct_df = rank_basket(basket, FUTURES, direct_prices, REPO, DAYS)
        direct_ctd = direct_df[direct_df["is_ctd"]]["label"].iloc[0]

        shocked_df = shocked_basket(basket, yields, FUTURES, REPO, DAYS,
                                    yield_shift_bps=0, as_of=AS_OF)
        shocked_ctd = shocked_df[shocked_df["is_ctd"]]["label"].iloc[0]

        assert shocked_ctd == direct_ctd

    def test_required_columns_present(self, basket_and_yields):
        basket, yields = basket_and_yields
        result = shocked_basket(basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF)
        required = (
            "cusip", "label", "cash_price", "conv_factor",
            "gross_basis", "carry", "net_basis", "implied_repo", "is_ctd",
        )
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_only_one_ctd(self, basket_and_yields):
        basket, yields = basket_and_yields
        result = shocked_basket(basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF)
        assert result["is_ctd"].sum() == 1

    def test_positive_shift_decreases_prices(self, basket_and_yields):
        basket, yields = basket_and_yields

        base_df = shocked_basket(basket, yields, FUTURES, REPO, DAYS,
                                 yield_shift_bps=0, as_of=AS_OF)
        up_df = shocked_basket(basket, yields, FUTURES, REPO, DAYS,
                               yield_shift_bps=100, as_of=AS_OF)

        base_prices = base_df.set_index("cusip")["cash_price"]
        up_prices   = up_df.set_index("cusip")["cash_price"]

        for cusip in base_prices.index:
            assert up_prices[cusip] < base_prices[cusip], (
                f"Expected price to fall for {cusip} after +100bps shift"
            )

    def test_negative_shift_increases_prices(self, basket_and_yields):
        basket, yields = basket_and_yields

        base_df = shocked_basket(basket, yields, FUTURES, REPO, DAYS,
                                 yield_shift_bps=0, as_of=AS_OF)
        down_df = shocked_basket(basket, yields, FUTURES, REPO, DAYS,
                                 yield_shift_bps=-100, as_of=AS_OF)

        base_prices = base_df.set_index("cusip")["cash_price"]
        down_prices = down_df.set_index("cusip")["cash_price"]

        for cusip in base_prices.index:
            assert down_prices[cusip] > base_prices[cusip], (
                f"Expected price to rise for {cusip} after -100bps shift"
            )


class TestScenarioGrid:
    def test_returns_two_dataframes(self, basket_and_yields):
        basket, yields = basket_and_yields
        summary_df, heatmap_df = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF
        )
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(heatmap_df, pd.DataFrame)

    def test_summary_one_row_per_shift(self, basket_and_yields):
        basket, yields = basket_and_yields
        shifts = [-50, -25, 0, 25, 50]
        summary_df, _ = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, shifts_bps=shifts, as_of=AS_OF
        )
        assert len(summary_df) == len(shifts)

    def test_summary_required_columns(self, basket_and_yields):
        basket, yields = basket_and_yields
        summary_df, _ = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF
        )
        required = (
            "shift_bps", "ctd_label", "ctd_implied_repo",
            "runner_label", "runner_implied_repo", "spread_bps", "ctd_changed",
        )
        for col in required:
            assert col in summary_df.columns, f"Missing column: {col}"

    def test_zero_shift_not_ctd_changed(self, basket_and_yields):
        basket, yields = basket_and_yields
        summary_df, _ = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF
        )
        zero_row = summary_df[summary_df["shift_bps"] == 0].iloc[0]
        assert zero_row["ctd_changed"] == False

    def test_spread_always_positive(self, basket_and_yields):
        basket, yields = basket_and_yields
        summary_df, _ = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF
        )
        assert (summary_df["spread_bps"] > 0).all(), (
            "CTD implied repo should always exceed runner-up implied repo"
        )

    def test_heatmap_rows_equal_basket_size(self, basket_and_yields):
        basket, yields = basket_and_yields
        _, heatmap_df = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF
        )
        assert len(heatmap_df) == len(basket)

    def test_heatmap_columns_are_sorted_shifts(self, basket_and_yields):
        basket, yields = basket_and_yields
        shifts = [-50, 25, 0, -25, 50]
        _, heatmap_df = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, shifts_bps=shifts, as_of=AS_OF
        )
        assert heatmap_df.columns.tolist() == sorted(shifts)

    def test_default_shifts_nine_scenarios(self, basket_and_yields):
        basket, yields = basket_and_yields
        summary_df, _ = scenario_grid(
            basket, yields, FUTURES, REPO, DAYS, as_of=AS_OF
        )
        assert len(summary_df) == 9