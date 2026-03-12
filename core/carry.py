"""
Cash-futures carry analytics for Treasury bond futures.

Key concepts:
  - Gross basis  = cash price − futures price × conversion factor
  - Carry        = coupon income − repo financing cost over the holding period
  - Net basis    = gross basis − carry
  - Implied repo = the financing rate implied by the cash-futures relationship

Price conventions
-----------------
Cash and futures prices are expressed as a percentage of face value
(e.g. 98.50 means 98.50% of par).

Day count conventions
---------------------
Coupon accrual uses ACT/365; repo financing uses ACT/360 (US market convention).
"""

from __future__ import annotations


def gross_basis(cash_price, futures_price, conversion_factor) -> float:
    """Gross basis: cash price minus futures invoice price.
    Args:
        cash_price: Clean cash price as % of par.
        futures_price: Quoted futures price as % of par.
        conversion_factor: CME/CBOT conversion factor.
    Returns: Gross basis in price points.
    """
    return cash_price - futures_price * conversion_factor


def carry(cash_price, coupon_rate, repo_rate, days_to_delivery) -> float:
    """Carry: net income from holding the bond and financing via repo.

    Carry = coupon accrual (ACT/365) − repo financing cost (ACT/360).
    Args:
        cash_price: Clean cash price as % of par.
        coupon_rate: Annual coupon rate as a decimal.
        repo_rate: Overnight/term repo rate as a decimal.
        days_to_delivery: Calendar days to futures delivery.
    Returns: Carry in price points.
    """
    coupon_income = cash_price * coupon_rate * (days_to_delivery / 365)
    financing_cost = cash_price * repo_rate * (days_to_delivery / 360)
    return coupon_income - financing_cost


def net_basis(cash_price, futures_price, conversion_factor, coupon_rate, repo_rate, days_to_delivery) -> float:
    """Net basis: gross basis minus carry.

    For the CTD bond, net basis ≈ 0 in a fair-value world; any residual
    reflects the embedded delivery option.
    Returns: Net basis in price points.
    """
    gb = gross_basis(cash_price, futures_price, conversion_factor)
    c = carry(cash_price, coupon_rate, repo_rate, days_to_delivery)
    return gb - c


def implied_repo(cash_price, futures_price, conversion_factor, coupon_rate, days_to_delivery) -> float:
    """Implied repo rate from the cash-futures cost-of-carry relationship.

    Derived by inverting the carry formula:
        implied_repo = [(invoice_price + coupon_accrual − cash_price)
                        / cash_price] × (360 / days)

    where invoice_price = futures_price × conversion_factor.
    Returns: Implied repo rate as a decimal.
    """
    invoice_price = futures_price * conversion_factor
    coupon_accrual = cash_price * coupon_rate * (days_to_delivery / 365)
    numerator = invoice_price + coupon_accrual - cash_price
    return (numerator / cash_price) * (360 / days_to_delivery)