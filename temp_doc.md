# Plan: ctd-basis-monitor — Focused Rebuild
 
## Context
 
The existing `fi_risk_dashboard` repo has technically correct fixed income analytics (pricing, Nelson-Siegel, basis, carry) but fails as a portfolio piece because **it has no defined purpose**. It covers every fixed income topic superficially. A quant at Capula — a cash-futures basis shop — will immediately ask: "what decision does this tool support?" The current answer is "all decisions, sort of." That is the most damning signal of AI-generated code.
 
The rebuild creates a **new repo** (`ctd-basis-monitor`) with a single, practitioner-defined purpose: monitor the US Treasury cash-futures basis on the TY (10-year) contract, track CTD transitions over time via SQL, and evaluate whether the current basis level is historically attractive to trade. This is literally what Capula's relative value desks watch every morning.
 
**What Zach said:** Python + SQL + clean code + interest in fixed income (especially cash-futures basis). This project answers all four.
 
---
 
## Project Concept
 
**One-sentence description:** A live monitoring tool for the US Treasury cash-futures basis on the TY contract that tracks net basis history in SQL, identifies CTD transitions, and evaluates current basis attractiveness.
 
**The question it answers:** "For the front TY contract, where is today's net basis relative to its 90-day range? Who is the CTD? How close is a transition? What happens to the CTD identity if yields shift?"
 
**The narrative for the interview:**
> "I built a daily monitor that computes net basis and implied repo for every bond in the TY delivery basket, stores snapshots in SQLite using window functions to track rolling statistics, and identifies when the CTD is close to switching. The scenario grid shows how CTD identity changes under parallel yield shifts — because that's what you want to know before a rate move."
 
---
 
## Architecture
 
```
ctd-basis-monitor/
├── core/                    # USER WRITES (pure analytics, no I/O)
│   ├── basket.py            # TYH25 delivery basket + CF table (real CME data)
│   ├── ctd.py               # CTD ranking, transition threshold calculation
│   ├── carry.py             # Net basis carry under repo term structure
│   └── scenario.py          # Yield shift → reprice basket → new CTD ranking
│
├── data/                    # SHARED (user writes schema+queries, Claude wires FRED)
│   ├── db.py                # SQLite schema + BasisDB query class
│   ├── ingest.py            # CLI: python -m data.ingest --contract TYH25 --date 2025-03-11
│   ├── fred_client.py       # Reuse/adapt from existing fi_risk_dashboard
│   └── market_data.py       # FRED yields → prices (default) + manual override
│
├── dashboard/               # CLAUDE BUILDS (UI only)
│   ├── app.py               # Streamlit entry point + dark theme config
│   └── pages/
│       ├── 01_basis_monitor.py    # PRIMARY: live basket + 90-day history
│       ├── 02_delivery_basket.py  # Full basket ranking table
│       ├── 03_ctd_history.py      # CTD transition timeline (Gantt-style)
│       └── 04_scenario_grid.py    # Yield shift heat map → CTD matrix
│
├── tests/                   # USER WRITES (analytics tests)
│   ├── test_basket.py
│   ├── test_ctd.py
│   └── test_carry.py
│
├── requirements.txt
├── pyproject.toml
└── README.md                # Short, practitioner-style (not textbook)
```
 
---
 
## Division of Labor
 
### User writes (core analytics — the intellectual core)
 
| File | Key content |
|------|-------------|
| `core/basket.py` | Real TYH25 deliverable bonds (look up from CME Group website): CUSIPs, coupons, maturity dates, conversion factors as static dataclasses. Function `get_basket(contract)`. |
| `core/ctd.py` | `rank_basket(basket, futures_price, bond_prices, repo_rate)` → ranked DataFrame. **Key depth:** `ctd_transition_threshold(bond_a, bond_b, cf_a, cf_b, ...) → futures_price_at_switch`. DV01 of the basis position (cash DV01 − futures DV01 / CF). |
| `core/carry.py` | `net_basis_carry(bond, futures_price, cf, repo_rate, days_to_delivery)`. Extend to accept a repo term structure (dict of term→rate) for carry-across-delivery-dates analysis. |
| `core/scenario.py` | `shocked_basket(basket, bond_prices, futures_price, yield_shift_bps, repo_rate)` → full re-ranked DataFrame after repricing every bond. Uses `price_bond` from the existing code (can copy over). |
| `data/db.py` | SQLAlchemy Core schema (two tables below), `BasisDB` class with query methods using window functions. |
| `tests/` | Unit tests for CTD ranking, transition threshold accuracy (verify against numerical), carry formula. |
 
### Claude builds (UI — frontend only)
 
| File | Content |
|------|---------|
| `dashboard/app.py` | Page config, dark theme CSS (adapted from existing), DB connection singleton |
| `dashboard/pages/01_basis_monitor.py` | Three-column: basket table + net basis 90d chart with rolling mean/stdev + metric cards |
| `dashboard/pages/02_delivery_basket.py` | Sortable basket table, CTD transition risk gauge |
| `dashboard/pages/03_ctd_history.py` | Gantt-style CTD timeline + implied repo spread chart (proximity to switch) |
| `dashboard/pages/04_scenario_grid.py` | Plotly heatmap: bonds × yield shifts, CTD switch row highlighted |
 
### Shared (user defines interface, Claude implements)
 
| File | Split |
|------|-------|
| `data/market_data.py` | User defines `BondPrices` interface; Claude implements FRED fetch + manual override flag |
| `data/ingest.py` | User designs workflow; Claude writes CLI boilerplate |
| `README.md` | User writes the narrative paragraph; Claude formats markdown |
 
---
 
## SQL Schema (user writes this in `data/db.py`)
 
```sql
-- Table 1: daily snapshot per bond per contract
CREATE TABLE basis_snapshots (
    id            INTEGER PRIMARY KEY,
    snapshot_dt   DATE    NOT NULL,
    contract      TEXT    NOT NULL,   -- "TYH25"
    cusip         TEXT    NOT NULL,
    coupon        REAL,
    maturity      TEXT,
    cash_price    REAL,
    accrued       REAL,
    futures_price REAL,
    conv_factor   REAL,
    gross_basis   REAL,               -- (cash_price + accrued) - futures_price * conv_factor
    net_basis     REAL,               -- gross_basis - carry
    implied_repo  REAL,
    is_ctd        INTEGER,            -- 1 or 0
    repo_rate     REAL,
    days_to_delivery INTEGER,
    UNIQUE(snapshot_dt, contract, cusip)
);
 
-- Table 2: log of CTD transitions
CREATE TABLE ctd_log (
    id                    INTEGER PRIMARY KEY,
    change_dt             DATE NOT NULL,
    contract              TEXT NOT NULL,
    prev_ctd_cusip        TEXT,
    new_ctd_cusip         TEXT,
    implied_repo_spread_bps REAL   -- how close was the switch?
);
```
 
**Key SQL queries to implement in `BasisDB`:**
```sql
-- 90-day history with rolling statistics (window functions)
SELECT snapshot_dt, net_basis,
    AVG(net_basis) OVER (ORDER BY snapshot_dt ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma_20d,
    PERCENT_RANK() OVER (ORDER BY net_basis) AS pct_rank
FROM basis_snapshots
WHERE cusip = ? AND contract = ?
ORDER BY snapshot_dt;
 
-- Implied repo spread (proximity to CTD switch)
SELECT snapshot_dt,
    MAX(implied_repo) - NTH_VALUE(implied_repo, 2) OVER (
        PARTITION BY snapshot_dt ORDER BY implied_repo DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS spread_to_second_bps
FROM basis_snapshots WHERE contract = ?;
```
 
---
 
## The "Genuinely Impressive" Technical Depth
 
### What will make Zach stop and look carefully
 
**CTD transition threshold** — the futures price F* at which bond B overtakes bond A as CTD:
```python
# Derived from: implied_repo_A(F*) == implied_repo_B(F*)
# Closed-form (approximate, ignoring coupon timing differences):
F_switch = (P_B / CF_B + carry_B - carry_A) * CF_A / P_A
# Plus: verify numerically against the full implied_repo calculation
```
This is not in any textbook. A junior candidate never thinks to compute it. Showing this — with a clean docstring explaining the derivation — signals genuine understanding of the delivery option.
 
**DV01 of the basis position:**
```
DV01_basis = DV01_cash_bond - DV01_futures_CTD / CF_CTD
```
The futures DV01 is the CTD's DV01 divided by its conversion factor. This is the hedge ratio calculation a basis trader actually uses.
 
**No-arbitrage framing in docstrings:**
Net basis = value of the futures short's delivery option (quality option + timing option). The CTD is the bond that makes this option worth the least (highest implied repo = cheapest to deliver). Frame every calculation around this principle.
 
### What will impress Prof Basak (academic rigor)
 
- Derive the implied repo formula in a comment block from the cost-of-carry no-arbitrage equation
- Note that net basis > 0 for CTD because the short has optionality (net basis = 0 only under certainty with a single deliverable bond)
- Reference: Burghardt, Belton, Lane, Papa — "The Treasury Bond Basis" (the standard text)
 
### The "cool factor" for Zach's boss
 
The CTD transition timeline (page 3) — showing historically when the CTD switched and how narrow the spread was before switching — is something almost no portfolio project shows. It demonstrates you understand this is not a static calculation but a dynamic phenomenon that changes risk on the short futures position.
 
---
 
## Data Flow
 
```
CME Group website → core/basket.py (static, hardcoded for TYH25)
FRED API → data/fred_client.py → yield curve → data/market_data.py → bond prices
Manual override → data/market_data.py → bond prices (e.g. from Bloomberg screen)
                                               ↓
                                        data/ingest.py (daily CLI)
                                               ↓
                                        data/db.py (SQLite write)
                                               ↓
                                        dashboard/ (read + display)
                                               ↓
                                        core/ (compute on-the-fly for scenarios)
```
 
---
 
## Implementation Phases (in order)
 
### Phase 1: User builds (analytics core)
1. `core/basket.py` — look up TYH25 basket from CME, hardcode as dataclasses
2. `core/ctd.py` — CTD ranking + transition threshold + DV01 of basis
3. `core/carry.py` — net basis carry (adapt from existing `basis.py`)
4. `core/scenario.py` — yield-shift → reprice → re-rank
5. `tests/` — unit tests for all of the above
 
### Phase 2: Shared (data layer)
6. `data/db.py` — schema + BasisDB query class
7. `data/fred_client.py` — adapt from existing
8. `data/market_data.py` — FRED + manual override
9. `data/ingest.py` — CLI ingestion script
 
### Phase 3: Claude builds (UI)
10. `dashboard/app.py` — entry point + CSS
11. `dashboard/pages/01_basis_monitor.py` — primary view
12. `dashboard/pages/02_delivery_basket.py` — basket table
13. `dashboard/pages/03_ctd_history.py` — CTD timeline
14. `dashboard/pages/04_scenario_grid.py` — scenario heatmap
 
### Phase 4: Polish
15. `README.md` — short, practitioner-style (3 paragraphs max)
16. `pyproject.toml` — clean packaging
17. Final code review for cleanliness (Zach said "clean code — very important")
 
---
 
## What to Intentionally NOT Include
 
- ❌ Z-spread (not relevant to basis trading)
- ❌ Nelson-Siegel parameter visualization (academic, not operational)
- ❌ 6-tab generic dashboard
- ❌ Textbook-style README explaining DCF pricing
- ❌ OVERVIEW.md / exhaustive documentation
- ❌ Generic bond parameters sidebar (arbitrary face value / coupon)
- ❌ MCP server (unless the use case for LLM integration is clearly defined)
 
The project should feel **narrowly focused** — a tool built for a specific purpose by someone who actually understands the problem.
 
---
 
## Verification
 
1. **Run analytics tests:** `pytest tests/ -v` — all CTD ranking, transition threshold, carry tests pass
2. **Seed the DB:** `python -m data.ingest --contract TYH25 --date 2025-03-11` — writes a snapshot row per bond
3. **Run dashboard:** `streamlit run dashboard/app.py` — page 01 shows basket table + history chart, page 04 shows scenario heatmap
4. **Spot-check the CTD:** verify the highest implied repo bond is flagged as CTD, matches manually computed value
5. **Transition threshold test:** verify `ctd_transition_threshold(bond_A, bond_B, ...)` equals the futures price at which `implied_repo_A == implied_repo_B` (numerical verification in `test_ctd.py`)
 
---
 
## Critical Files (existing code to reuse/adapt)
 
- `fi_risk_dashboard/core/basis.py` — implied repo and gross/net basis formulas (adapt into `core/carry.py`)
- `fi_risk_dashboard/core/pricing.py` — `price_bond()` (copy as-is, it's correct)
- `fi_risk_dashboard/mcp_server/fred_client.py` — adapt for `data/fred_client.py`
- `fi_risk_dashboard/dashboard/app.py` — dark theme CSS tokens to reuse in new dashboard
 