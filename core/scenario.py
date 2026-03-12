"""
Scenario analytics: yield-shock the delivery basket and re-rank by implied repo.

The primary function is shocked_basket(), which reprices every bond in the
TY delivery basket at a shifted yield and returns the full CTD ranking.

scenario_grid() runs shocked_basket() across a range of parallel yield shifts
and returns two DataFrames useful for dashboard display:
  - summary_df:  one row per shift, with CTD label, implied repo, runner-up, spread
  - heatmap_df:  pivot of implied repo (rows=bond, columns=shift_bps)

These are the building blocks for the scenario heatmap page of the monitor.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from core.ctd import rank_basket
from core.pricing import price_bond


def shocked_basket(
    basket: list[dict],
    base_yields: dict[str, float],
    futures_price: float,
    repo_rate: float,
    days_to_delivery: int,
    yield_shift_bps: int = 0,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Reprice and re-rank the basket after a parallel yield shift.

    Each bond is repriced at (base_yield + shift/10_000) using price_bond().
    Years-to-maturity is measured from as_of (defaults to today).

    Args:
        basket:           List of bond dicts from core.basket.get_basket().
        base_yields:      Dict mapping cusip → current yield (decimal).
        futures_price:    Quoted futures price (% of par).
        repo_rate:        Financing rate as a decimal.
        days_to_delivery: Calendar days to futures delivery.
        yield_shift_bps:  Parallel yield shift in basis points (default 0 = no shock).
        as_of:            Date from which years-to-maturity is measured.
                          Defaults to today.

    Returns:
        Ranked DataFrame from rank_basket() under the shifted yields.
        See core.ctd.rank_basket for column definitions.
    """
    if as_of is None:
        as_of = date.today()

    shift = yield_shift_bps / 10_000
    shocked_prices = {
        b["cusip"]: price_bond(
            100.0,
            b["coupon"],
            (b["maturity"] - as_of).days / 365.25,
            base_yields[b["cusip"]] + shift,
        )
        for b in basket
        if b["cusip"] in base_yields
    }

    return rank_basket(basket, futures_price, shocked_prices, repo_rate, days_to_delivery)


def scenario_grid(
    basket: list[dict],
    base_yields: dict[str, float],
    futures_price: float,
    repo_rate: float,
    days_to_delivery: int,
    shifts_bps: list[int] | None = None,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run shocked_basket() across a grid of parallel yield shifts.

    Args:
        basket:           List of bond dicts from core.basket.get_basket().
        base_yields:      Dict mapping cusip → current yield (decimal).
        futures_price:    Quoted futures price (% of par).
        repo_rate:        Financing rate as a decimal.
        days_to_delivery: Calendar days to futures delivery.
        shifts_bps:       Yield shifts to test in basis points.
                          Defaults to [-100, -75, -50, -25, 0, 25, 50, 75, 100].
        as_of:            Date from which years-to-maturity is measured.
                          Defaults to today.

    Returns:
        summary_df: One row per shift. Columns: shift_bps, ctd_label,
                    ctd_implied_repo, runner_label, runner_implied_repo,
                    spread_bps, ctd_changed.
        heatmap_df: Pivot of implied_repo × 100 (as %). Rows = bond label,
                    columns = shift_bps. Sorted columns ascending.
    """
    if shifts_bps is None:
        shifts_bps = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

    if as_of is None:
        as_of = date.today()

    # Determine base CTD (zero shift) to detect transitions
    base_df = shocked_basket(basket, base_yields, futures_price, repo_rate,
                              days_to_delivery, yield_shift_bps=0, as_of=as_of)
    base_ctd_label = base_df[base_df["is_ctd"]]["label"].iloc[0]

    summary_rows = []
    heatmap_data: dict[str, dict[int, float]] = {}

    for shift in shifts_bps:
        df = shocked_basket(basket, base_yields, futures_price, repo_rate,
                            days_to_delivery, yield_shift_bps=shift, as_of=as_of)

        ctd    = df[df["is_ctd"]].iloc[0]
        runner = df[df.index == 2].iloc[0]
        spread = (ctd["implied_repo"] - runner["implied_repo"]) * 10_000

        summary_rows.append({
            "shift_bps":           shift,
            "ctd_label":           ctd["label"],
            "ctd_implied_repo":    ctd["implied_repo"],
            "runner_label":        runner["label"],
            "runner_implied_repo": runner["implied_repo"],
            "spread_bps":          spread,
            "ctd_changed":         ctd["label"] != base_ctd_label,
        })

        for _, row in df.iterrows():
            label = row["label"]
            if label not in heatmap_data:
                heatmap_data[label] = {}
            heatmap_data[label][shift] = row["implied_repo"] * 100  # as %

    summary_df = pd.DataFrame(summary_rows)
    heatmap_df = pd.DataFrame(heatmap_data).T.rename_axis("bond")
    heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

    return summary_df, heatmap_df