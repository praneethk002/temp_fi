"""Tests for data/ingest.py — run_ingest()."""

from datetime import date
from unittest.mock import patch

import pytest

from core.basket import get_basket
from data.db import BasisDB
from data.ingest import run_ingest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTRACT = "TYM26"
SNAPSHOT_DT = date(2026, 1, 15)
FUTURES = 108.50
REPO = 0.053


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def mock_prices():
    basket = get_basket(use_api=False)
    fake = {b["cusip"]: 97.0 for b in basket}
    with patch("data.ingest.get_bond_prices", return_value=fake) as m:
        yield m


# ---------------------------------------------------------------------------
# TestRunIngest
# ---------------------------------------------------------------------------

class TestRunIngest:
    """Tests for run_ingest() covering dry-run, live-run, and edge cases."""

    def test_dry_run_returns_zero(self, db_path, mock_prices):
        result = run_ingest(
            contract=CONTRACT,
            snapshot_dt=SNAPSHOT_DT,
            futures_price=FUTURES,
            repo_rate=REPO,
            overrides={},
            db_path=db_path,
            dry_run=True,
        )
        assert result == 0

    def test_dry_run_does_not_write_db(self, db_path, mock_prices):
        run_ingest(
            contract=CONTRACT,
            snapshot_dt=SNAPSHOT_DT,
            futures_price=FUTURES,
            repo_rate=REPO,
            overrides={},
            db_path=db_path,
            dry_run=True,
        )
        # Either the db file was never created, or it exists but has no rows
        if db_path.exists():
            db = BasisDB(db_path)
            df = db.get_current_basket(CONTRACT)
            assert df.empty

    def test_live_run_writes_rows(self, db_path, mock_prices):
        basket = get_basket(use_api=False)
        result = run_ingest(
            contract=CONTRACT,
            snapshot_dt=SNAPSHOT_DT,
            futures_price=FUTURES,
            repo_rate=REPO,
            overrides={},
            db_path=db_path,
            dry_run=False,
        )
        assert result == len(basket)

    def test_live_run_creates_db_file(self, db_path, mock_prices):
        run_ingest(
            contract=CONTRACT,
            snapshot_dt=SNAPSHOT_DT,
            futures_price=FUTURES,
            repo_rate=REPO,
            overrides={},
            db_path=db_path,
            dry_run=False,
        )
        assert db_path.exists()

    def test_live_run_snapshot_readable(self, db_path, mock_prices):
        run_ingest(
            contract=CONTRACT,
            snapshot_dt=SNAPSHOT_DT,
            futures_price=FUTURES,
            repo_rate=REPO,
            overrides={},
            db_path=db_path,
            dry_run=False,
        )
        df = BasisDB(db_path).get_current_basket(CONTRACT)
        assert not df.empty

    def test_overrides_passed_through(self, db_path, mock_prices):
        overrides = {"91282CKG2": 98.00}
        run_ingest(
            contract=CONTRACT,
            snapshot_dt=SNAPSHOT_DT,
            futures_price=FUTURES,
            repo_rate=REPO,
            overrides=overrides,
            db_path=db_path,
            dry_run=False,
        )
        mock_prices.assert_called_once_with(
            get_basket(use_api=False),
            as_of=SNAPSHOT_DT,
            use_fred=True,
            overrides=overrides,
        )

    def test_no_prices_returns_zero(self, db_path):
        with patch("data.ingest.get_bond_prices", return_value={}):
            result = run_ingest(
                contract=CONTRACT,
                snapshot_dt=SNAPSHOT_DT,
                futures_price=FUTURES,
                repo_rate=REPO,
                overrides={},
                db_path=db_path,
                dry_run=False,
            )
        assert result == 0