"""
SQLite persistence layer for the CTD basis monitor.

Schema
------
basis_snapshots
    Daily snapshot: one row per bond per contract per date.
    Captures the full basis analytics so history can be reconstructed
    without replaying prices.

ctd_log
    Append-only log of CTD transitions. A new row is written whenever
    the CTD identity changes from one snapshot to the next.

BasisDB
    Query class wrapping both tables. Uses window functions for rolling
    statistics and percentile rank — SQLite supports these from version
    3.25.0 (2018-09-15), well within any current install.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).parent.parent / "basis_monitor.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS basis_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_dt      TEXT    NOT NULL,          -- ISO date "YYYY-MM-DD"
    contract         TEXT    NOT NULL,          -- e.g. "TYM26"
    cusip            TEXT    NOT NULL,
    coupon           REAL    NOT NULL,
    maturity         TEXT    NOT NULL,          -- ISO date
    cash_price       REAL    NOT NULL,
    futures_price    REAL    NOT NULL,
    conv_factor      REAL    NOT NULL,
    gross_basis      REAL    NOT NULL,
    net_basis        REAL    NOT NULL,
    implied_repo     REAL    NOT NULL,
    is_ctd           INTEGER NOT NULL,          -- 1 or 0
    repo_rate        REAL    NOT NULL,
    days_to_delivery INTEGER NOT NULL,
    UNIQUE(snapshot_dt, contract, cusip)
);

CREATE TABLE IF NOT EXISTS ctd_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    change_dt               TEXT    NOT NULL,   -- ISO date of the new snapshot
    contract                TEXT    NOT NULL,
    prev_ctd_cusip          TEXT,
    new_ctd_cusip           TEXT    NOT NULL,
    implied_repo_spread_bps REAL                -- spread at the time of the switch
);

CREATE INDEX IF NOT EXISTS idx_snapshots_contract_dt
    ON basis_snapshots(contract, snapshot_dt);

CREATE INDEX IF NOT EXISTS idx_snapshots_cusip_contract
    ON basis_snapshots(cusip, contract, snapshot_dt);
"""


# ---------------------------------------------------------------------------
# BasisDB
# ---------------------------------------------------------------------------

class BasisDB:
    """Read/write interface to the basis_monitor SQLite database.

    Usage::

        db = BasisDB()                          # default path
        db = BasisDB("/path/to/custom.db")      # custom path
        db.init_schema()                        # idempotent — safe to call every run
        db.write_snapshot(snapshot_date, contract, rows, repo_rate, days_to_delivery)
        df = db.get_basis_history("91282CKG2", "TYM26", days=90)
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables and indexes if they do not already exist (idempotent)."""
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_snapshot(
        self,
        snapshot_dt: date,
        contract: str,
        ranked_df: pd.DataFrame,
        repo_rate: float,
        days_to_delivery: int,
        futures_price: float,
    ) -> int:
        """Persist a ranked basket DataFrame as daily snapshot rows.

        Detects CTD transitions and appends a row to ctd_log when the
        CTD identity changes compared to the most recent stored snapshot.

        Args:
            snapshot_dt:      Date of this snapshot.
            contract:         Futures contract label, e.g. "TYM26".
            ranked_df:        DataFrame from core.ctd.rank_basket().
                              Required columns: cusip, coupon, maturity,
                              cash_price, conv_factor, gross_basis, net_basis,
                              implied_repo, is_ctd.
            repo_rate:        Repo rate used for this snapshot.
            days_to_delivery: Days to delivery used for this snapshot.
            futures_price:    Futures price used for this snapshot.

        Returns:
            Number of rows inserted into basis_snapshots.
        """
        dt_str = snapshot_dt.isoformat()
        rows_inserted = 0

        with self._conn() as conn:
            for _, row in ranked_df.iterrows():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO basis_snapshots
                        (snapshot_dt, contract, cusip, coupon, maturity,
                         cash_price, futures_price, conv_factor,
                         gross_basis, net_basis, implied_repo,
                         is_ctd, repo_rate, days_to_delivery)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        dt_str,
                        contract,
                        row["cusip"],
                        row["coupon"],
                        row["maturity"].isoformat()
                            if hasattr(row["maturity"], "isoformat")
                            else str(row["maturity"]),
                        row["cash_price"],
                        futures_price,
                        row["conv_factor"],
                        row["gross_basis"],
                        row["net_basis"],
                        row["implied_repo"],
                        int(row["is_ctd"]),
                        repo_rate,
                        days_to_delivery,
                    ),
                )
                rows_inserted += 1

            # Detect CTD transition
            new_ctd_cusip = ranked_df[ranked_df["is_ctd"]]["cusip"].iloc[0]
            prev = conn.execute(
                """
                SELECT cusip, implied_repo
                FROM basis_snapshots
                WHERE contract = ?
                  AND snapshot_dt < ?
                  AND is_ctd = 1
                ORDER BY snapshot_dt DESC
                LIMIT 1
                """,
                (contract, dt_str),
            ).fetchone()

            if prev is None or prev["cusip"] != new_ctd_cusip:
                # Compute spread between CTD and runner-up for this snapshot
                sorted_repos = ranked_df.sort_values("implied_repo", ascending=False)
                spread_bps: float | None = None
                if len(sorted_repos) >= 2:
                    ctd_ir    = sorted_repos.iloc[0]["implied_repo"]
                    runner_ir = sorted_repos.iloc[1]["implied_repo"]
                    spread_bps = (ctd_ir - runner_ir) * 10_000

                conn.execute(
                    """
                    INSERT INTO ctd_log
                        (change_dt, contract, prev_ctd_cusip, new_ctd_cusip,
                         implied_repo_spread_bps)
                    VALUES (?,?,?,?,?)
                    """,
                    (
                        dt_str,
                        contract,
                        prev["cusip"] if prev else None,
                        new_ctd_cusip,
                        spread_bps,
                    ),
                )

        return rows_inserted

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_basis_history(
        self,
        cusip: str,
        contract: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """Time-series of net basis for a specific bond with rolling stats.

        Uses a 20-day rolling average and percentile rank window functions.

        Args:
            cusip:    CUSIP of the bond.
            contract: Futures contract label.
            days:     Number of calendar days of history to return.

        Returns:
            DataFrame with columns: snapshot_dt, net_basis, implied_repo,
            is_ctd, ma_20d, pct_rank.
        """
        sql = """
            WITH history AS (
                SELECT snapshot_dt, net_basis, implied_repo, is_ctd
                FROM basis_snapshots
                WHERE cusip = ? AND contract = ?
                ORDER BY snapshot_dt DESC
                LIMIT ?
            )
            SELECT
                snapshot_dt,
                net_basis,
                implied_repo,
                is_ctd,
                AVG(net_basis) OVER (
                    ORDER BY snapshot_dt
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS ma_20d,
                PERCENT_RANK() OVER (ORDER BY net_basis) AS pct_rank
            FROM history
            ORDER BY snapshot_dt
        """
        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=(cusip, contract, days))
        return df

    def get_current_basket(self, contract: str) -> pd.DataFrame:
        """Return the most recent snapshot for all bonds in a contract.

        Args:
            contract: Futures contract label, e.g. "TYM26".

        Returns:
            DataFrame with one row per bond from the latest snapshot date,
            sorted by implied_repo descending (CTD first).
        """
        sql = """
            SELECT s.*
            FROM basis_snapshots s
            INNER JOIN (
                SELECT MAX(snapshot_dt) AS latest_dt
                FROM basis_snapshots
                WHERE contract = ?
            ) m ON s.snapshot_dt = m.latest_dt
            WHERE s.contract = ?
            ORDER BY s.implied_repo DESC
        """
        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=(contract, contract))
        return df

    def get_basis_percentile(
        self,
        contract: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """Percentile rank of each bond's current net basis over history.

        Answers: "Where does today's net basis sit in its 90-day distribution?"

        Args:
            contract: Futures contract label.
            days:     Lookback window in calendar days.

        Returns:
            DataFrame with columns: cusip, snapshot_dt, net_basis, pct_rank.
            One row per bond from the latest snapshot.
        """
        sql = """
            WITH ranked AS (
                SELECT
                    cusip,
                    snapshot_dt,
                    net_basis,
                    PERCENT_RANK() OVER (
                        PARTITION BY cusip
                        ORDER BY net_basis
                    ) AS pct_rank
                FROM basis_snapshots
                WHERE contract = ?
                  AND snapshot_dt >= DATE(
                        (SELECT MAX(snapshot_dt) FROM basis_snapshots WHERE contract = ?),
                        ? || ' days'
                  )
            )
            SELECT r.*
            FROM ranked r
            INNER JOIN (
                SELECT MAX(snapshot_dt) AS latest_dt
                FROM basis_snapshots WHERE contract = ?
            ) m ON r.snapshot_dt = m.latest_dt
            ORDER BY pct_rank DESC
        """
        with self._conn() as conn:
            df = pd.read_sql_query(
                sql, conn, params=(contract, contract, f"-{days}", contract)
            )
        return df

    def get_ctd_transitions(self, contract: str) -> pd.DataFrame:
        """Return the full CTD transition log for a contract.

        Args:
            contract: Futures contract label.

        Returns:
            DataFrame with columns: change_dt, prev_ctd_cusip, new_ctd_cusip,
            implied_repo_spread_bps. Most recent first.
        """
        sql = """
            SELECT change_dt, prev_ctd_cusip, new_ctd_cusip, implied_repo_spread_bps
            FROM ctd_log
            WHERE contract = ?
            ORDER BY change_dt DESC
        """
        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=(contract,))
        return df

    def get_transition_proximity(self, contract: str) -> pd.DataFrame:
        """Implied repo spread between CTD and runner-up over time.

        A narrowing spread signals elevated CTD transition risk.

        Args:
            contract: Futures contract label.

        Returns:
            DataFrame with columns: snapshot_dt, spread_to_second_bps.
            Most recent first.
        """
        sql = """
            SELECT
                snapshot_dt,
                MAX(implied_repo) - NTH_VALUE(implied_repo, 2) OVER (
                    PARTITION BY snapshot_dt
                    ORDER BY implied_repo DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS spread_to_second_bps
            FROM basis_snapshots
            WHERE contract = ?
            GROUP BY snapshot_dt
            HAVING spread_to_second_bps IS NOT NULL
            ORDER BY snapshot_dt DESC
        """
        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=(contract,))
        # Convert from decimal to bps
        if "spread_to_second_bps" in df.columns:
            df["spread_to_second_bps"] = df["spread_to_second_bps"] * 10_000
        return df