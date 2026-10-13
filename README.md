# Short-term return reversals around earnings announcements

A reproducible Streamlit research application studying short-term stock return
reversals around earnings announcements for **70 long-standing S&P 500 companies**,
each analysed on its own against **SPY** as the market benchmark. Pick any stock to
see its research pages, or open the **30-stock summary** (the original universe) or
the **40-stock summary** (the added stocks) to line a group up side by side. Defaults to **AAPL**.

**Research question**

> Does a stock's market-adjusted return shortly before earnings negatively predict
> its return around the announcement, and is that relationship stronger than during
> comparable non-announcement periods?

The motivation is that some pre-earnings price movement may reflect temporary
buying or selling pressure, and that investors absorbing that pressure may require
greater compensation for holding a position through a risky announcement. **This
application cannot test that mechanism.** It only describes a statistical
relationship.

Inspired by So & Wang (2014), *News-driven return reversals*. It is **not** a
replication of that multi-company study, **not** a production trading strategy,
**not** evidence that liquidity provision causes anything, and **not** a machine
learning project. All results are exploratory and in-sample.

---

## Installation

Requires Python 3.10 or newer.

```bash
cd Earnings_Research
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: pip install -r requirements.lock.txt
```

`requirements.txt` pins the direct dependencies to the exact versions used to build
and test this project. `requirements.lock.txt` is the full `pip freeze` of the
tested environment.

## Launching the application

```bash
streamlit run app.py
```

Then open <http://localhost:8501>. On first launch the app downloads prices and
earnings dates and writes them to `cache/`; afterwards it reads from that cache, so
changing a sidebar widget never triggers a new download. Use **Refresh data** in
the sidebar to force one (it refreshes the selected stock, or every stock in the
summary view you are on).

The first visit to a summary view downloads and analyses all its stocks, which
takes a few minutes on a cold cache and roughly 10–15 seconds from a warm one.

The same summary can be produced without the UI:

```bash
python scripts/run_universe.py --group first30 --window announcement --out results
python scripts/run_universe.py --group added40 --window announcement --out results
```

It prints the cross-stock table, pooled group means and per-stock PAR quintile
strategy results, and writes `universe_<window>.csv`, `overview_<window>.csv`,
`pooled_events.csv` and `par_quintile_strategy.csv`.

## Running the tests

```bash
python -m pytest
```

The Streamlit smoke tests in `tests/test_app_smoke.py` skip automatically when the
local cache is empty, so the suite never requires network access. Run the app once
to populate the cache if you want those tests to execute; the summary-view tests
additionally need every stock of that view cached.

---

## Methodology

### Return definitions

Let `P[i,d]` be the adjusted close of stock `i` on trading session `d`.

```
daily return         r[i,d]      = P[i,d] / P[i,d-1] - 1
stock window return  R_stock     = prod(1 + r[i,d]) for d = a..b, minus 1
bench window return  R_bench     = prod(1 + r[m,d]) over the same sessions, minus 1
market-adjusted      R_adj[a,b]  = R_stock - R_bench
```

The market adjustment is a **difference between two compounded returns**. Daily
stock-minus-benchmark differences are never compounded. The starting price for
window `[a,b]` is the close at session `a-1`; the ending price is the close at
session `b`.

### Windows

| Window | Sessions | Start price | End price | Daily returns included |
| --- | --- | --- | --- | --- |
| PAR | `[-4,-2]` | close at `t=-5` | close at `t=-2` | `t=-4, -3, -2` |
| Announcement return | `[-1,+1]` | close at `t=-2` | close at `t=+1` | `t=-1, 0, +1` |
| Subsequent return | `[+2,+5]` | close at `t=+1` | close at `t=+5` | `t=+2 … +5` |
| Later return | `[+6,+8]` | close at `t=+5` | close at `t=+8` | `t=+6, +7, +8` |

> **The announcement window includes the trading day before, the release date, and
> the trading day after. It is not a pure post-release return.**

Windows are fixed in this first version to avoid an unnecessary parameter search.
Returns are stored as decimals internally and formatted as percentages in the UI.

### Data and adjustments

- Daily prices from Yahoo Finance via `yfinance`, requested with `interval="1d"`,
  `auto_adjust=False` and `actions=True`. OHLC, adjusted close, volume, dividends
  and splits are all preserved.
- Returns use **adjusted close**, which carries Yahoo's split and dividend
  adjustments. Because `auto_adjust=False` keeps the raw `Close` separate, nothing
  is adjusted twice.
- If adjusted close is missing, the unadjusted close is **never** substituted. The
  session stays missing and affected events are excluded with a stated reason.
- Each cached artefact is a Parquet file with a JSON provenance sidecar recording
  tickers, requested period, actual coverage, UTC retrieval timestamp, package
  versions, adjustment settings, data source, cache key and a SHA-256 file hash.
- Downloads use bounded retries (3 attempts with backoff). On failure the real
  error is displayed and an existing cache may be used. Failed downloads are never
  replaced with synthetic data.

### Trading calendar and event alignment

- Sessions come from the explicit `XNYS` `exchange_calendars` calendar, never from
  intersecting the dates returned by a price download.
- Prices are reindexed onto the expected sessions with missing values preserved.
  There is no forward-filling anywhere in the pipeline.
- `t=0` is the actual release calendar date when that date is a trading session.
  All offsets count trading sessions, not calendar days.
- Before-open / during-session / after-close classification is preserved, using the
  exchange calendar's real open and close times so early-close days are handled
  correctly. In this paper-style analysis an after-close release **keeps its own
  calendar date** and is not shifted to the next session.
- Releases on non-trading dates are flagged and excluded, never silently moved.
- Events missing any required price in either series are excluded, with the reason
  shown for every exclusion.

### Earnings dates

The event is the **first public earnings release**, which may precede the
conference call. It is deliberately distinguished from the fiscal quarter-end, the
SEC filing date and the conference-call time.

**Completed-release rule.** A record is retained only if (a) its release timestamp
is in the past relative to the retrieval time, and (b) it carries at least one
realised figure (reported EPS or surprise %) or is a user-verified manual record.
Date-only rows are evaluated at the end of the release day. Future estimates and
past-dated rows with no realised figure are excluded and listed in the UI.

- A date-only record or a midnight placeholder timestamp is never treated as a
  verified release time; it is classified `unknown`.
- Exact duplicates are removed. Records describing the same release but disagreeing
  on the date are **flagged for review**, not arbitrarily resolved, and are kept out
  of the analysis until resolved.
- Dates are never invented, never assumed to be 90 days apart, and never generated
  from a four-per-year rule. Four announcements per year is a diagnostic
  expectation only.

### Verified earnings-date upload

The *Data & Coverage* tab accepts a CSV with these columns:

```
ticker,release_date,release_time,timezone,source_url,verified
```

- `release_date` must be `YYYY-MM-DD`.
- `release_time` is optional (`HH:MM` or `HH:MM:SS`).
- `timezone` defaults to `America/New_York` when omitted.
- `source_url` identifies a company release or other supporting source.
- `verified` is a boolean indicating you checked the source; `verified=true`
  requires a `source_url`.

An **empty** template (headers only, no invented sample dates) is downloadable from
the UI. Uploads are validated row by row and row-level errors are displayed; invalid
rows are never used. A verified manual record overrides a downloaded record
describing the same release, and the override is disclosed in the records table.

### Pseudo-event rules

- For each eligible actual event, candidate pseudo-dates sit **10 to 40 trading
  sessions before it, inclusive**.
- A candidate is eligible only if every required stock and benchmark price exists
  across sessions `[-5,+8]` and its return-analysis interval contains no known
  earnings release.
- **Boundary handling.** The contamination screen covers the calendar interval from
  the day after the session `-5` baseline close through the session `+8` date. That
  covers every date that can influence a return in sessions `[-4,+8]`, including
  non-trading dates between sessions, while not treating the baseline close date
  itself as contaminated.
- Screening uses **all** available confirmed releases, including those outside the
  selected event range.
- One candidate is drawn uniformly at random per actual event using the configured
  seed, repeated for the configured number of repetitions.
- Events with no eligible candidate are reported and dropped from **both** the
  matched actual sample and the pseudo sample, so the two are always matched.

### Statistics

- Pearson correlation between PAR and each outcome return.
- OLS: `outcome_return = intercept + slope × PAR`, reporting slope, intercept and
  R-squared.
- A **reversal** is `PAR × outcome_return < 0`. Observations where either value is
  exactly zero are reported separately and never counted as a reversal or a
  continuation.
- Negative-PAR and positive-PAR groups are summarised separately by count, mean,
  median and opposite-sign frequency, for all three outcome windows.
- The **negative-PAR minus positive-PAR mean difference** compares events for one
  company occurring at different times. It is **not** a simultaneously tradable
  long-short portfolio return.
- Small samples, constant variables and empty groups return an explicit
  "not available" with a reason instead of a misleading number.

### Actual versus pseudo comparison

For each repetition the same statistics are recomputed on the pseudo sample. The UI
shows the actual matched-sample statistic, the pseudo median, the 5th and 95th
percentiles and the number of valid repetitions.

The percentile range is labelled **"Variation across pseudo-date draws — not a
confidence interval."** Repetitions are never pooled into one large sample,
significance is never computed as if the draws were independent, and the share of
pseudo statistics exceeding the actual statistic is deliberately **not** reported as
a p-value. Reuse and overlap diagnostics are shown because pseudo windows are drawn
with replacement and reused across repetitions.

---

## The stock universes

### The original 30 (30-stock summary)

| Sector | Stocks |
| --- | --- |
| Information Technology | AAPL, MSFT, IBM, INTC, CSCO |
| Communication Services | GOOGL, VZ, DIS |
| Consumer Discretionary | AMZN, HD, MCD, NKE |
| Consumer Staples | PG, KO, PEP, WMT |
| Health Care | UNH, PFE, ABT, LLY |
| Financials | JPM, BAC, WFC, AXP |
| Energy | XOM, COP |
| Industrials | CAT, HON, BA |
| Utilities | NEE |

**Selection rule.** Large companies that were S&P 500 members throughout
2010–2025, spread across sectors, and for which the data source actually returned
complete history: daily prices from before 2010 and all 62 earnings releases in the
2010–2025 range (through each company's latest record, April–June 2025). Each was
checked before inclusion; every one of the 30 has 62 included events and no
exclusions. Candidates with a missing quarter (MRK 2012, COST 2014) or a release
dated on a non-trading day (CVX 2023-07-23, T 2016-10-22) were left out so every
stock enters on the same footing; results were not looked at when choosing. The
list lives in `earnings_reversal/universe.py`.

### The added 40 (40-stock summary)

| Sector | Stocks |
| --- | --- |
| Information Technology | ORCL, QCOM, TXN, ADBE, NVDA |
| Communication Services | CMCSA, OMC |
| Consumer Discretionary | LOW, SBUX, TGT, TJX, BKNG |
| Consumer Staples | MO, CL, KMB, GIS |
| Health Care | JNJ, BMY, AMGN, GILD, MDT |
| Financials | GS, MS, C, USB, TRV |
| Energy | SLB, EOG, OXY |
| Industrials | MMM, UNP, UPS, LMT, DE |
| Utilities | DUK, SO |
| Materials | SHW, FCX |
| Real Estate | SPG, PSA |

Same rule as the first 30. 101 further candidates were screened and 87 passed;
the 40 were picked from those for sector spread (adding Materials and Real Estate)
before any of their results were looked at. Every one has 62 included events and
no exclusions. Left out: MU (uneven quarterly count); F, ADM, VLO and ECL (a
release on 2012-10-30, when the exchange was closed for Hurricane Sandy); EA, BK,
MMC, EQR and AVB (downloads failed during the check); and DHR (2016-07-05,
+61% adjusted vs +4% raw), VTR (2015-08-18, −24%) and HPQ (2015-11-02, +13%),
whose adjusted close shows unadjusted or mis-adjusted spin-off days.

**Survivorship bias.** The list was chosen in hindsight from companies that are
still large S&P 500 members. Firms that shrank, merged or were removed are absent.

**Per-stock settings.** The benchmark, event range, seed and repetition count in
the sidebar apply to every stock. Event exclusions and verified-CSV uploads are kept
per stock: rows in an uploaded CSV only affect the ticker they name.

## Application layout

The sidebar **View** switch chooses between **Single stock** (the five tabs below,
for the stock picked in the **Stock** menu, which also accepts any other ticker) and
the **30-stock summary** or the **40-stock summary**. The **Stock** menu lists all 70
stocks, the original 30 first.

| Tab | Contents |
| --- | --- |
| **Data & Coverage** | Requested vs actual coverage, event counts, earnings counts by year (including zero years), source/verification table, provenance, cache freshness and refresh, verified-CSV upload, event exclusion controls, download and alignment errors |
| **Earnings Event Analysis** | KPI cards, PAR vs announcement scatterplot with zero lines and fitted line, negative/positive PAR summary for all three outcomes, event-level audit table |
| **Actual vs Pseudo** | Comparison table, histograms of pseudo mean outcomes after negative and positive PAR with the actual means marked, a worked example of one fake date, candidate screening, matched sample size, reuse diagnostics |
| **PAR Quintile Strategy** | Per-stock historical quintile rule (below), thresholds and positions chart, results table with raw stock and signed market-adjusted returns reported separately, cumulative fixed-notional P&L, Q1−Q5 group spread, comparison with the existing PAR sign strategy, and the same rule applied to every pseudo-event repetition |
| **Event Inspector** | Searchable, sortable event table; per-event dates, adjusted closes, daily returns and window calculations; normalised price chart with the release and all four windows marked; source records |
| **Methodology** | Research question, formulas, windows, data and calendar handling, pseudo-date rules, differences from the original study, limitations |

### PAR quintile strategy (per-stock historical adaptation)

This is **not** the original study's cross-sectional quintile sort. Each stock's
PAR is ranked against that stock's own history:

- For each event, the 20th and 80th percentiles of PAR are computed from the
  stock's **earlier** included events only (numpy linear interpolation), requiring
  at least 20 of them. The current event never enters its own thresholds; events
  with less history are skipped (with the full 2010–2025 sample, the first 20
  events of each stock, i.e. 2010–2014, only build history).
- PAR below the 20th percentile (Q1) → long; above the 80th (Q5) → short;
  otherwise no trade. Quintile membership sets the direction even when a threshold
  is positive or negative.
- The position is held over the announcement window `[-1,+1]`.
- **Raw stock strategy return** = position × stock return. **Signed
  market-adjusted return** = position × (stock − benchmark return). They are
  reported separately.
- Win rate counts strictly positive signed returns. Cumulative P&L uses a fixed
  notional per trade, summed and never compounded.
- **Q1 mean minus Q5 mean** is labelled a *group spread*. It is not the average
  return per strategy trade.
- The existing negative/positive PAR split is traded as a comparison strategy,
  both on the same eligible events and on all events.
- **Timing.** PAR ends at the close of `t=-2`, which is also the start price of the
  announcement window, so trades assume **same-close execution** (a market-on-close
  order sized from a near-close PAR). Every earlier event used in a threshold sits
  at a strictly earlier session; the app checks and reports this for every trade.
  The release date is assumed known by `t=-2`.
- **Pseudo comparison.** The rule is applied to each pseudo-event repetition; each
  pseudo date's thresholds come only from actual events before that date. Actual
  matched-sample results are compared with the pseudo median and 5th–95th
  percentiles. Trade counts and matched eligible event counts are reported, and
  repetitions are never pooled.
- All results are **before costs** (commissions, spreads, borrow, financing,
  impact).

Each summary page (30-stock and 40-stock) shows, for a chosen outcome window: coverage KPIs; a
table of each statistic across stocks (median, equal-weight mean, how many stocks
are below or above zero, how many fall beyond their own pseudo 5th–95th range on the
reversal side and on the opposite side); a chart placing each stock's actual
statistic against its pseudo range; mean outcome after negative versus positive PAR
per stock; a per-stock table; pooled negative/positive PAR group means over all
events; each stock's actual statistic minus its own pseudo median; and, for the
PAR quintile strategy, any strategy statistic against each stock's pseudo range
and pseudo median plus mean Q1 (long) versus Q5 (short) event returns;
coverage details; and a button that opens any stock's pages.

**Reading "beyond pseudo range" counts.** Returns in the announcement window are
much more volatile on earnings dates than on the fake dates (the page reports the
measured ratio of standard deviations), so actual statistics land outside the
fake-date range more often *on both sides*. Compare the reversal-side count against
the opposite-side count rather than against a chance baseline. The stocks share
earnings seasons and market shocks, so they are not independent, and no pooled
significance test is computed.

CSV export buttons accompany the tables, but no export is needed to understand the
results. When there is nothing to show, the app displays an empty state with
instructions and **never** fills the dashboard with realistic-looking fake results.

## Project structure

```
app.py                        Streamlit entry point and sidebar
earnings_reversal/
  config.py                   Windows, offsets, defaults, caveat strings
  cache.py                    Parquet + JSON provenance disk cache
  trading_calendar.py         XNYS sessions, release-time classification
  prices.py                   Price download, retries, session alignment
  earnings.py                 Earnings retrieval, manual CSV, reconciliation
  returns.py                  Window return mathematics
  events.py                   Event alignment and the audit table
  stats.py                    Correlation, OLS, reversal and group statistics
  pseudo.py                   Candidate screening and the Monte Carlo comparison
  pipeline.py                 Retrieval and computation orchestration
  universe.py                 The 30 + 40 stocks, names, sectors and groups
  summary.py                  Cross-stock tables, pooled events, universe runner
  strategy.py                 PAR quintile strategy and its pseudo comparison
ui/                           One module per tab, plus tab_summary.py
scripts/run_universe.py       Command-line version of the summaries (--group)
tests/                        pytest suite
cache/                        Local data cache (created on first run)
```

Data retrieval (`load_inputs`) is separated from computation (`run_analysis`), so
cached or manually supplied data runs through the identical analysis path.

---

## Coverage limitations and known issues

- **Yahoo caps earnings history at 100 rows per ticker.** `get_earnings_dates`
  accepts a `limit` argument, but requesting more than 100 returns 100. For AAPL
  those 100 rows reach back to 1999, so the 2010–2025 range is covered; for a
  company with more frequent records the cap could truncate history. The UI warns
  whenever the source returns fewer rows than requested.
- **Recent quarters may be missing.** At the time of writing, the latest earnings
  record available from the source was 2025-05-01 for AAPL and between April and
  June 2025 for the other stocks, so the later part of the default 2010–2025 range
  has no events. The app states the analysed coverage and
  never claims full 2010–2025 coverage unless it is established.
- **Release times are often unknown.** Older records arrive as midnight placeholder
  timestamps. These are classified `unknown` rather than being treated as verified
  release times, and the count is displayed.
- **Nothing is verified by default.** Downloaded records are marked unverified.
  Verification requires uploading a CSV you have checked against company releases.
- **Incomplete earnings coverage weakens the contamination screen.** Unknown
  releases cannot be ruled out, so pseudo windows are *screened*, not *proven clean*.
- With quarterly reporting the 10–40 session search window sits well inside the gap
  between releases, so the contamination screen typically rejects nothing. That is
  the expected outcome for a regular reporter, not evidence the screen is inactive;
  the per-event rejection counts are displayed so you can confirm it.
- Candidates 10–13 sessions before an event have a `[-5,+8]` interval that partly
  overlaps that event's own analysis window. This follows from the fixed 10–40
  session rule and is reported rather than silently filtered.
- **Adjusted close carries float32 rounding from the source.** Repeating a download
  can shift an adjusted close by around `3e-5` (the raw `Close` is stable), which
  moves a window return in roughly its seventh decimal place. This is source-side
  precision, not a calculation difference, but it means results are reproducible to
  about six decimals across re-downloads rather than bit-for-bit. Reruns from the
  same cache are exactly reproducible.
- The `use_container_width` argument was replaced with `width="stretch"`, which
  requires a recent Streamlit. Use the pinned version.

## Limitations

- Roughly four earnings events per year means the sample is small, and it stays
  small no matter how many pseudo repetitions are run.
- Results for 70 hand-picked large companies may not generalise to the index.
- This is a set of time-series adaptations, not cross-sectional portfolio sorting.
- The original study's previous-quarter quintile method is not used.
- Actual historical release dates may not have been known in advance. This is an
  ex-post event study, not a point-in-time trading backtest.
- SPY is a practical benchmark substitution; the benchmark choice and the return
  definitions are explicit adaptations, not reproductions of an academic
  market-return dataset.
- Missing or inaccurate release dates feed straight through to the results.
- Adjusted-close returns do not represent executable trade prices.
- No transaction costs, financing, borrow costs or implementation constraints are
  modelled.
- A negative relationship does not prove a liquidity-provider mechanism.
- Pseudo-date contamination checks and repeated draws are adaptations designed for
  this pilot.
- No machine learning and no out-of-sample performance claim is included.

## Extending to other companies

The **Stock** menu's "Other ticker…" option runs the single-stock pages for any
ticker. To change the summary universe, edit `UNIVERSE` in
`earnings_reversal/universe.py`. Pooled event tables ignore cross-sectional
dependence; they are descriptive only.

## Synthetic data

Synthetic fixtures appear only in `tests/`. They are never used as research data in
the application, and the app never substitutes generated data for a failed download.
