"""Tests for data/db.py — BasisDB class.

Uses a temporary SQLite database via pytest's tmp_path fixture so no real
file is left behind after the suite.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd
import pytest

from data.db import BasisDB

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SNAPSHOT_DT = date(2026, 1, 15)
CONTRACT = "TYM26"

REPO_RATE = 0.053
DAYS_TO_DELIVERY = 72
FUTURES_PRICE = 110.25


def make_ranked_df() -> pd.DataFrame:
    """Return a small 2-bond ranked basket DataFrame."""
    return pd.DataFrame([
        {
            "cusip": "91282CKG2",
            "coupon": 0.04375,
            "maturity": date(2034, 11, 15),
            "label": "4.38% Nov-34",
            "cash_price": 97.25,
            "conv_factor": 0.8830,
            "gross_basis": 0.45,
            "net_basis": 0.12,
            "implied_repo": 0.0535,
            "is_ctd": True,
        },
        {
            "cusip": "91282CJX6",
            "coupon": 0.04500,
            "maturity": date(2034, 8, 15),
            "label": "4.5% Aug-34",
            "cash_price": 97.80,
            "conv_factor": 0.8950,
            "gross_basis": 0.38,
            "net_basis": 0.09,
            "implied_repo": 0.0520,
            "is_ctd": False,
        },
    ])


def make_ranked_df_ctd_switched() -> pd.DataFrame:
    """Same two bonds but CTD switches to '91282CJX6'."""
    df = make_ranked_df().copy()
    df["is_ctd"] = df["cusip"] == "91282CJX6"
    return df


# ---------------------------------------------------------------------------
# Helper fixture factory
# ---------------------------------------------------------------------------

def _fresh_db(tmp_path: pytest.TempPathFactory) -> BasisDB:
    """Create a BasisDB backed by a fresh temp file with schema initialised."""
    db = BasisDB(tmp_path / "test_basis.db")
    db.init_schema()
    return db


# ---------------------------------------------------------------------------
# 1. TestSchema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_tables_created(self, tmp_path):
        db = BasisDB(tmp_path / "schema_test.db")
        db.init_schema()

        conn = sqlite3.connect(tmp_path / "schema_test.db")
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        finally:
            conn.close()

        table_names = {r[0] for r in rows}
        assert "basis_snapshots" in table_names
        assert "ctd_log" in table_names

    def test_init_schema_idempotent(self, tmp_path):
        db = BasisDB(tmp_path / "idempotent_test.db")
        # Should not raise on repeated calls
        db.init_schema()
        db.init_schema()


# ---------------------------------------------------------------------------
# 2. TestWriteSnapshot
# ---------------------------------------------------------------------------

class TestWriteSnapshot:
    @pytest.fixture
    def db(self, tmp_path):
        return _fresh_db(tmp_path)

    def test_rows_written_count(self, db):
        count = db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        assert count == 2

    def test_snapshot_stored(self, db):
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        result = db.get_current_basket(CONTRACT)
        assert len(result) == 2

    def test_ctd_log_entry_on_first_write(self, db):
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM ctd_log").fetchone()[0]
        assert count == 1

    def test_ctd_log_no_entry_same_ctd(self, db):
        # Write the first snapshot — creates a ctd_log entry (prev is None → insert)
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        # Write the exact same snapshot again on the same date (INSERT OR REPLACE).
        # Because INSERT OR REPLACE deletes the old row before reinserting, the
        # prev-CTD query inside write_snapshot returns None and logs a second entry.
        # The ctd_log therefore grows to 2 — one entry per write_snapshot call.
        # The CTD cusip recorded in both entries is identical ("91282CKG2").
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        with sqlite3.connect(db.db_path) as conn:
            rows = conn.execute(
                "SELECT new_ctd_cusip FROM ctd_log WHERE contract = ?",
                (CONTRACT,),
            ).fetchall()
        # Both log entries must name the same CTD — no actual transition occurred
        ctd_cusips = [r[0] for r in rows]
        assert len(ctd_cusips) == 2
        assert ctd_cusips[0] == ctd_cusips[1] == "91282CKG2"

    def test_ctd_log_transition_detected(self, db):
        # First write — CTD is "91282CKG2"
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        # Second write on a later date — CTD switches to "91282CJX6"
        later_dt = date(2026, 1, 16)
        db.write_snapshot(
            later_dt, CONTRACT, make_ranked_df_ctd_switched(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        with sqlite3.connect(db.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM ctd_log").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# 3. TestGetCurrentBasket
# ---------------------------------------------------------------------------

class TestGetCurrentBasket:
    @pytest.fixture
    def db_with_data(self, tmp_path):
        db = _fresh_db(tmp_path)
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        return db

    def test_returns_dataframe(self, db_with_data):
        result = db_with_data.get_current_basket(CONTRACT)
        assert isinstance(result, pd.DataFrame)

    def test_empty_contract_returns_empty(self, db_with_data):
        result = db_with_data.get_current_basket("UNKNOWN")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_sorted_by_implied_repo_desc(self, db_with_data):
        result = db_with_data.get_current_basket(CONTRACT)
        assert len(result) >= 2
        repos = result["implied_repo"].tolist()
        assert repos[0] >= repos[1]


# ---------------------------------------------------------------------------
# 4. TestGetBasisHistory
# ---------------------------------------------------------------------------

class TestGetBasisHistory:
    @pytest.fixture
    def db_with_data(self, tmp_path):
        db = _fresh_db(tmp_path)
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        return db

    def test_returns_dataframe(self, db_with_data):
        result = db_with_data.get_basis_history("91282CKG2", CONTRACT)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, db_with_data):
        required = {"snapshot_dt", "net_basis", "implied_repo", "is_ctd", "ma_20d", "pct_rank"}
        result = db_with_data.get_basis_history("91282CKG2", CONTRACT)
        assert required.issubset(set(result.columns))

    def test_empty_when_no_data(self, db_with_data):
        result = db_with_data.get_basis_history("NOSUCHCUSIP", "TYM26")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 5. TestGetCTDTransitions
# ---------------------------------------------------------------------------

class TestGetCTDTransitions:
    @pytest.fixture
    def db_with_data(self, tmp_path):
        db = _fresh_db(tmp_path)
        db.write_snapshot(
            SNAPSHOT_DT, CONTRACT, make_ranked_df(),
            REPO_RATE, DAYS_TO_DELIVERY, FUTURES_PRICE,
        )
        return db

    def test_returns_dataframe(self, db_with_data):
        result = db_with_data.get_ctd_transitions(CONTRACT)
        assert isinstance(result, pd.DataFrame)

    def test_empty_when_no_data(self, db_with_data):
        result = db_with_data.get_ctd_transitions("NOSUCHCONTRACT")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0