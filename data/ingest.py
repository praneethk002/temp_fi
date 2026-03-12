"""
Daily ingest CLI for the CTD basis monitor.

Fetches (or accepts) market data for a given contract and date, runs the
full CTD ranking, and writes a snapshot to the SQLite database.

Usage
-----
# Ingest today's data (requires FRED_API_KEY):
python -m data.ingest --contract TYM26

# Ingest a specific historical date:
python -m data.ingest --contract TYM26 --date 2026-01-15

# Dry run (print basket, do not write to DB):
python -m data.ingest --contract TYM26 --dry-run

# Use manual price overrides (JSON string):
python -m data.ingest --contract TYM26 --overrides '{"91282CKG2": 97.25}'

# Use a custom database path:
python -m data.ingest --contract TYM26 --db /path/to/my.db
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime

from core.basket import DELIVERY_DATE, get_basket
from core.ctd import rank_basket
from data.db import DEFAULT_DB_PATH, BasisDB
from data.market_data import get_bond_prices

# ---------------------------------------------------------------------------
# Contract configuration
# ---------------------------------------------------------------------------

# Futures prices are not on FRED; use a sensible default and allow override.
# In production, supply --futures-price from a CME data feed.
_DEFAULT_FUTURES_PRICE = 108.50

# Repo rate: overnight GCF or term repo.  Override with --repo-rate.
_DEFAULT_REPO_RATE = 0.053


def _days_to_delivery(snapshot_dt: date, delivery_dt: date = DELIVERY_DATE) -> int:
    return max(1, (delivery_dt - snapshot_dt).days)


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}' — expected YYYY-MM-DD")


def run_ingest(
    contract: str,
    snapshot_dt: date,
    futures_price: float,
    repo_rate: float,
    overrides: dict[str, float],
    db_path,
    dry_run: bool,
) -> int:
    """Core ingest logic — separated from CLI parsing for testability.

    Returns:
        Number of snapshot rows written (0 for dry run).
    """
    basket = get_basket(use_api=False)
    days   = _days_to_delivery(snapshot_dt)

    # Fetch / build bond prices
    bond_prices = get_bond_prices(
        basket,
        as_of=snapshot_dt,
        use_fred=True,
        overrides=overrides,
    )

    if not bond_prices:
        print(
            "ERROR: No bond prices available. Set FRED_API_KEY or supply "
            "--overrides with at least one cusip→price mapping.",
            file=sys.stderr,
        )
        return 0

    # Rank the basket
    ranked = rank_basket(basket, futures_price, bond_prices, repo_rate, days)

    # Print summary
    ctd = ranked[ranked["is_ctd"]].iloc[0]
    print(f"\n=== {contract}  {snapshot_dt}  (dry_run={dry_run}) ===")
    print(f"  Futures price : {futures_price:.4f}")
    print(f"  Repo rate     : {repo_rate*100:.3f}%")
    print(f"  Days to deliv : {days}")
    print(f"  CTD           : {ctd['label']}  implied_repo={ctd['implied_repo']*100:.4f}%")
    print()
    print(ranked[["label", "cash_price", "gross_basis", "net_basis",
                  "implied_repo", "is_ctd"]].to_string())
    print()

    if dry_run:
        print("Dry run — nothing written to DB.")
        return 0

    db = BasisDB(db_path)
    db.init_schema()
    n = db.write_snapshot(snapshot_dt, contract, ranked, repo_rate, days, futures_price)
    print(f"Wrote {n} rows to {db_path}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a daily CTD basis snapshot into SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--contract", required=True,
        help='Futures contract label, e.g. "TYM26"',
    )
    parser.add_argument(
        "--date", type=_parse_date, default=date.today(),
        metavar="YYYY-MM-DD",
        help="Snapshot date (default: today)",
    )
    parser.add_argument(
        "--futures-price", type=float, default=_DEFAULT_FUTURES_PRICE,
        metavar="PRICE",
        help=f"Quoted futures price as %% of par (default {_DEFAULT_FUTURES_PRICE})",
    )
    parser.add_argument(
        "--repo-rate", type=float, default=_DEFAULT_REPO_RATE,
        metavar="RATE",
        help=f"Repo rate as a decimal (default {_DEFAULT_REPO_RATE})",
    )
    parser.add_argument(
        "--overrides", type=json.loads, default={},
        metavar='JSON',
        help='JSON dict of cusip→price overrides, e.g. \'{"91282CKG2": 97.25}\'',
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_PATH,
        metavar="PATH",
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the ranked basket but do not write to the database",
    )

    args = parser.parse_args()

    run_ingest(
        contract=args.contract,
        snapshot_dt=args.date,
        futures_price=args.futures_price,
        repo_rate=args.repo_rate,
        overrides=args.overrides,
        db_path=args.db,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()