"""Tab E: Methodology and limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from earnings_reversal.config import (
    ALL_WINDOWS,
    ANNOUNCEMENT_WINDOW_CAVEAT,
    CONTAMINATION_BOUNDARY,
    PSEUDO_MAX_OFFSET,
    PSEUDO_MIN_OFFSET,
    TRADING_CALENDAR_NAME,
)
from earnings_reversal.earnings import COMPLETED_RULE
from earnings_reversal.pseudo import CONTAMINATION_RULE
from earnings_reversal.strategy import (
    COSTS_NOTE,
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    MIN_PRIOR_EVENTS,
    SAME_CLOSE_NOTE,
    SPREAD_NOTE,
)

LIMITATIONS = [
    "Roughly four earnings events per year means the sample is small, and it stays "
    "small no matter how many pseudo repetitions are run.",
    "Results for one company need not generalise to any other company, and results "
    "for 70 large companies chosen in hindsight (survivorship bias) need not "
    "generalise to the index.",
    "The 70 stocks are not independent: they report in the same earnings seasons and "
    "share market-wide shocks. No pooled significance test is computed.",
    "This is a time-series adaptation. It is not the cross-sectional portfolio "
    "sorting used in the original study.",
    "The original study's previous-quarter quintile method is not used here.",
    "Actual historical release dates may not have been known in advance. This is an "
    "ex-post event study, not a point-in-time trading backtest.",
    "SPY is a practical substitute for the market return. The benchmark choice and "
    "the return definitions are explicit adaptations, not reproductions of an "
    "academic market-return dataset.",
    "Missing or inaccurate release dates feed straight through to the results.",
    "Adjusted-close returns do not represent executable trade prices.",
    "No transaction costs, financing, borrow costs or implementation constraints are "
    "modelled.",
    "A negative relationship between PAR and the announcement return does not prove "
    "a liquidity-provider mechanism, or any other mechanism.",
    "The pseudo-date contamination check and the repeated draws are adaptations "
    "designed for this pilot, not procedures taken from the original paper.",
    "No machine learning is used and no out-of-sample performance is claimed.",
]


def render(result=None) -> None:
    st.subheader("Methodology")

    st.markdown("### Research question")
    st.markdown(
        "> Does a stock's market-adjusted return shortly before earnings negatively "
        "predict its return around the announcement, and is that relationship "
        "stronger than during comparable non-announcement periods?"
    )
    st.markdown(
        "The motivation is that some pre-earnings price movement may reflect "
        "temporary buying or selling pressure. Investors who absorb that pressure may "
        "require greater compensation for holding a position through a risky earnings "
        "announcement. **This application cannot test that mechanism.** It only "
        "describes a statistical relationship. A descriptive pilot, run one company "
        "at a time for 70 S&P 500 companies, "
        "inspired by So & Wang (2014), *News-driven return reversals* — not a "
        "replication of it, not a trading strategy, and not causal evidence."
    )

    st.markdown("### Return definitions")
    st.markdown(
        "Let $P_{i,d}$ be the adjusted close of stock $i$ on trading session $d$."
    )
    st.latex(r"r_{i,d} = \frac{P_{i,d}}{P_{i,d-1}} - 1")
    st.markdown("Benchmark daily returns are computed identically. For a window $[a,b]$:")
    st.latex(r"R^{stock}_{[a,b]} = \prod_{d=a}^{b} (1 + r_{i,d}) - 1")
    st.latex(r"R^{bench}_{[a,b]} = \prod_{d=a}^{b} (1 + r_{m,d}) - 1")
    st.latex(r"R^{adj}_{[a,b]} = R^{stock}_{[a,b]} - R^{bench}_{[a,b]}")
    st.markdown(
        "The market-adjusted return is a **difference between two compounded "
        "returns**. Daily stock-minus-benchmark differences are never compounded. "
        "The starting price for window $[a,b]$ is the close at session $a-1$ and the "
        "ending price is the close at session $b$."
    )

    st.markdown("### Windows")
    st.dataframe(
        pd.DataFrame([{
            "Window": window.label,
            "Sessions": f"[{window.start:+d}, {window.end:+d}]",
            "Start price (close at)": f"session {window.baseline_offset:+d}",
            "End price (close at)": f"session {window.end:+d}",
            "Daily returns included": ", ".join(
                f"t={offset:+d}" for offset in range(window.start, window.end + 1)
            ),
        } for window in ALL_WINDOWS]),
        width="stretch", hide_index=True,
    )
    st.markdown(
        "**PAR** is the pre-announcement market-adjusted return over sessions "
        "[-4,-2]. It requires the close at $t=-5$ and includes the daily returns at "
        "$t=-4$, $t=-3$ and $t=-2$."
    )
    st.info(ANNOUNCEMENT_WINDOW_CAVEAT)

    st.markdown("### Data and adjustments")
    st.markdown(
        f"- Daily prices come from {result.inputs.stock.provenance.data_source if result and result.inputs.stock else 'Yahoo Finance via yfinance'}, "
        "requested with `interval=\"1d\"`, `auto_adjust=False` and `actions=True`.\n"
        "- Returns use **adjusted close**, which already carries Yahoo's split and "
        "dividend adjustments. Because `auto_adjust=False` keeps the raw `Close` "
        "separate, no price is adjusted twice.\n"
        "- If adjusted close is missing, the unadjusted close is **never** substituted. "
        "The session stays missing and the affected event is excluded with a stated "
        "reason.\n"
        "- Raw downloads are cached locally as Parquet with a JSON provenance sidecar "
        "recording tickers, requested period, actual coverage, UTC retrieval time, "
        "package versions, adjustment settings, data source and a SHA-256 file hash."
    )

    st.markdown("### Trading calendar and event alignment")
    st.markdown(
        f"- Sessions come from the explicit `{TRADING_CALENDAR_NAME}` exchange "
        "calendar, never from intersecting the dates that happened to arrive in a "
        "price download.\n"
        "- Prices are reindexed onto the expected sessions and missing values are "
        "preserved. There is no forward-filling anywhere in the pipeline.\n"
        "- $t=0$ is the actual release calendar date when that date is a trading "
        "session. All offsets count trading sessions, not calendar days.\n"
        "- Before-open, during-session and after-close information is preserved. In "
        "this paper-style analysis an after-close release **keeps its own calendar "
        "date** and is not shifted to the next session.\n"
        "- A release falling on a non-trading date is flagged and excluded, never "
        "silently moved.\n"
        "- Events missing any required price in either series are excluded, and the "
        "reason is shown for every excluded event."
    )

    st.markdown("### Earnings dates")
    st.markdown(
        "The event is the **first public earnings release**, which may precede the "
        "conference call. It is deliberately distinguished from the fiscal "
        "quarter-end, the SEC filing date and the conference-call time."
    )
    st.markdown(f"**Completed-release rule.** {COMPLETED_RULE}")
    st.markdown(
        "- A date-only record or a midnight placeholder timestamp is never treated as "
        "a verified release time; it is classified as `unknown`.\n"
        "- Exact duplicates are removed. Records describing the same release but "
        "disagreeing on the date are **flagged for review**, not arbitrarily resolved.\n"
        "- A user-verified manual record may resolve such a conflict. The override is "
        "retained in the record's provenance and disclosed in the tables.\n"
        "- Dates are never invented, never assumed to be 90 days apart, and never "
        "generated from a four-per-year rule."
    )

    st.markdown("### Pseudo-event rules")
    st.markdown(
        f"- For each eligible actual event, candidate pseudo-dates are placed "
        f"{PSEUDO_MIN_OFFSET}\u2013{PSEUDO_MAX_OFFSET} trading sessions before it, "
        "inclusive.\n"
        "- A candidate is eligible only if every required stock and benchmark price "
        "is present across sessions [-5,+8] and its return-analysis interval contains "
        "no known earnings release.\n"
        f"- Boundary handling (`{CONTAMINATION_BOUNDARY}`): {CONTAMINATION_RULE}\n"
        "- Screening uses **all** available confirmed releases, including those "
        "outside the selected event range, not only the events chosen for the main "
        "analysis.\n"
        "- Where the earnings source is incomplete, unknown releases cannot be ruled "
        "out. Pseudo windows are therefore *screened*, not *proven clean*.\n"
        "- For each actual event one candidate is drawn uniformly at random using the "
        "configured seed, and the draw is repeated for the configured number of "
        "repetitions.\n"
        "- Repetitions are a Monte Carlo sensitivity check. They do not add "
        "independent observations, the repetitions are never pooled into one large "
        "sample, and the 5th-95th percentile range across draws is not a confidence "
        "interval. The share of repetitions beyond the actual statistic is not "
        "reported as a p-value.\n"
        "- Pseudo windows drawn in different repetitions reuse the same dates; reuse "
        "and overlap diagnostics are reported on the *Actual vs Pseudo* tab."
    )

    st.markdown("### Statistics")
    st.markdown(
        "- Pearson correlation between PAR and each outcome return.\n"
        "- OLS: `outcome_return = intercept + slope \u00d7 PAR`, reporting slope, "
        "intercept and R-squared.\n"
        "- A **reversal** is `PAR \u00d7 outcome_return < 0`. Cases where either value "
        "is exactly zero are reported separately, never counted as a reversal or a "
        "continuation.\n"
        "- Negative-PAR and positive-PAR groups are summarised separately by count, "
        "mean, median and opposite-sign frequency for all three outcome windows.\n"
        "- The **negative-PAR minus positive-PAR mean difference** compares events for "
        "one company at different points in time. It is not a simultaneously tradable "
        "long-short portfolio return.\n"
        "- Small samples, constant variables and empty groups return an explicit "
        "\u201cnot available\u201d with a reason instead of a misleading number.\n"
        "- Everything is exploratory and in-sample."
    )

    st.markdown("### PAR quintile strategy")
    st.markdown(
        f"- Thresholds: the {LOW_PERCENTILE}th and {HIGH_PERCENTILE}th percentiles "
        f"(numpy linear interpolation) of PAR over the stock's earlier included events "
        f"only, requiring at least {MIN_PRIOR_EVENTS}. The current event never enters its "
        "own thresholds. Events with less history are skipped.\n"
        f"- PAR below the {LOW_PERCENTILE}th percentile (Q1): long. Above the "
        f"{HIGH_PERCENTILE}th (Q5): short. Otherwise no trade. Ties with a threshold do "
        "not trade.\n"
        "- Held over the announcement window [-1,+1]. Raw stock strategy return = "
        "position \u00d7 stock return; signed market-adjusted return = position \u00d7 "
        "(stock return \u2212 benchmark return).\n"
        "- Win rate: share of executed trades with a strictly positive signed return. "
        "Cumulative P&L: fixed notional per trade, summed, not compounded.\n"
        f"- {SPREAD_NOTE}\n"
        f"- {SAME_CLOSE_NOTE}\n"
        "- Pseudo comparison: each pseudo date gets thresholds from actual events "
        "before it, and each repetition is evaluated as a separate alternative history.\n"
        f"- {COSTS_NOTE}"
    )

    st.markdown("### Differences from the original multi-company study")
    st.markdown(
        "- One company at a time (70 separate studies, lined up side by side in a 30-stock "
        "and a 40-stock summary), rather "
        "than a cross-section of many firms.\n"
        "- A time-series comparison of a company's own events, rather than "
        "cross-sectional portfolio sorts.\n"
        "- No previous-quarter cross-sectional quintile assignment. The *PAR Quintile "
        "Strategy* tab ranks each stock's PAR against that stock's own earlier events "
        "instead, a per-stock historical adaptation.\n"
        "- SPY as the market benchmark, rather than an academic market-return series.\n"
        "- Pseudo-event construction and contamination screening are this pilot's own "
        "adaptation.\n"
        "- No claim is made about reproducing any particular magnitude reported in "
        "that study."
    )

    st.markdown("### Limitations")
    for limitation in LIMITATIONS:
        st.markdown(f"- {limitation}")
