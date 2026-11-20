"""Compare the PAR reversal with IMP's linear models.

    streamlit run compare_app.py

The first tab is the 2024 scorecard. The second tab is the PAR methodology
as it was written for the original study.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.compare_year import quarter_table, sector_table, stock_table, year_events

ROOT = Path(__file__).resolve().parent
REVERSAL = ROOT / "output" / "reversal"
LIN = ROOT / "output" / "lin" / "metrics.csv"

SCORE_COLUMNS = [
    "model", "n_test", "oos_r2", "ic", "rank_ic", "hit_rate",
    "rmse", "rmse_train_mean", "rmse_zero", "top_minus_bottom",
]

LABELS = {
    "reversal_ols": "PAR reversal",
    "positioning_ols": "Positioning (QScore + ratings)",
    "imp_ols": "IMP OLS",
    "imp_ridge": "IMP ridge",
    "imp_lasso": "IMP lasso",
}

FOLD_LABEL = {
    "train": "2020–2022 (used to fit)",
    "val": "2023 (check)",
    "test": "2024 (held out)",
}

HIGHER = ("oos_r2", "ic", "rank_ic", "hit_rate", "top_minus_bottom")
METRIC_LABEL = {
    "oos_r2": "Beats a constant guess?",
    "ic": "Moves with the actual return?",
    "rank_ic": "Ranks events in the right order?",
    "hit_rate": "Gets the sign right?",
    "top_minus_bottom": "High-forecast names beat low-forecast names?",
    "rmse": "Typical forecast error",
}

WINDOWS = pd.DataFrame([
    {
        "Window": "PAR (the signal)",
        "Sessions": "[−4, −2]",
        "Start price": "close at session −5",
        "End price": "close at session −2",
        "What it is": "The stock's move into the earnings release, after subtracting the market.",
    },
    {
        "Window": "Announcement",
        "Sessions": "[−1, +1]",
        "Start price": "close at session −2",
        "End price": "close at session +1",
        "What it is": "The week of the release. Includes the day before the print.",
    },
    {
        "Window": "Subsequent",
        "Sessions": "[+2, +5]",
        "Start price": "close at session +1",
        "End price": "close at session +5",
        "What it is": "The days after the announcement window.",
    },
    {
        "Window": "Later",
        "Sessions": "[+6, +8]",
        "Start price": "close at session +5",
        "End price": "close at session +8",
        "What it is": "A short follow-up window. Used in the original study only.",
    },
])


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    summary = pd.read_csv(REVERSAL / "metrics.csv")
    folds = pd.read_csv(REVERSAL / "fold_metrics.csv")
    strategy = pd.read_csv(REVERSAL / "strategy.csv")
    corr = pd.read_csv(REVERSAL / "correlation.csv")
    if LIN.exists():
        lin = pd.read_csv(LIN)
        lin["model"] = "imp_" + lin["model"].astype(str)
        keep = [c for c in SCORE_COLUMNS if c in lin.columns]
        summary = pd.concat([summary, lin[keep]], ignore_index=True)
    # Positioning OLS and IMP OLS are the same regression.
    summary = summary.loc[summary["model"] != "positioning_ols"].copy()
    summary["label"] = summary["model"].map(LABELS).fillna(summary["model"])
    events = None
    events_path = REVERSAL / "events.parquet"
    if events_path.exists():
        events = pd.read_parquet(events_path)
    return summary, folds, strategy, corr, events


def _fmt(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric == "hit_rate":
        return f"{value:.1%}"
    if metric in ("top_minus_bottom", "rmse"):
        return f"{value:.2%}"
    return f"{value:+.4f}"


def _leader(summary: pd.DataFrame, metric: str, higher: bool) -> tuple[str, float]:
    part = summary.dropna(subset=[metric])
    if part.empty:
        return "—", float("nan")
    row = part.loc[part[metric].idxmax() if higher else part[metric].idxmin()]
    return str(row["label"]), float(row[metric])


def _verdict(summary: pd.DataFrame) -> str:
    reversal = summary.loc[summary["model"] == "reversal_ols"]
    others = summary.loc[summary["model"] != "reversal_ols"]
    if reversal.empty or others.empty:
        return "Only one side of the comparison is on disk."
    rev = reversal.iloc[0]
    ahead, behind = [], []
    for metric in (*HIGHER, "rmse"):
        higher = metric in HIGHER
        rivals = others.dropna(subset=[metric])
        if rivals.empty or pd.isna(rev[metric]):
            continue
        best = rivals[metric].max() if higher else rivals[metric].min()
        wins = rev[metric] > best if higher else rev[metric] < best
        (ahead if wins else behind).append(METRIC_LABEL[metric])
    return (
        "On the year that was held out (2024), the PAR reversal is a slightly "
        "better forecast than Brian's linear models. It is the only one that "
        "beats a constant guess, and it lines up a bit better with the actual "
        "return. Brian's models do a bit better at one thing: names they rank "
        "highest do outperform names they rank lowest by a larger gap. "
        "The differences are small. A hit rate near 52% is almost a coin flip. "
        "Neither method is a trading strategy."
    )


def _render_comparison(summary, folds, strategy, corr, events) -> None:
    st.subheader("Which forecast is ahead in 2024?")
    st.write(_verdict(summary))

    cards = st.columns(3)
    for column, metric in zip(cards, ("oos_r2", "ic", "top_minus_bottom")):
        label, value = _leader(summary, metric, True)
        column.metric(METRIC_LABEL[metric], _fmt(metric, value))
        column.caption(f"Ahead: {label}")

    st.markdown("### The 2024 scorecard")
    show = summary.copy()
    for metric in (*HIGHER, "rmse"):
        if metric in show.columns:
            show[metric] = show[metric].map(lambda v, m=metric: _fmt(m, v))
    show = show.rename(columns={"label": "Method", "n_test": "Events in 2024", **METRIC_LABEL})
    columns = [
        "Method", "Events in 2024",
        METRIC_LABEL["oos_r2"], METRIC_LABEL["ic"], METRIC_LABEL["rank_ic"],
        METRIC_LABEL["hit_rate"], METRIC_LABEL["rmse"], METRIC_LABEL["top_minus_bottom"],
    ]
    st.dataframe(show[[c for c in columns if c in show.columns]], width="stretch", hide_index=True)
    st.caption(
        "Beats a constant guess? is out-of-sample R²: positive means the model "
        "beats the average 2020–2022 return. Moves with the actual return? is "
        "the correlation (IC). Gets the sign right? is the hit rate. "
        "High-forecast names beat low-forecast names? is the average realized "
        "return in the top fifth of forecasts minus the bottom fifth. "
        "IMP lasso shrinks to the training average, so it does not rank events."
    )

    st.markdown("### How much better than a constant guess")
    chart = summary.dropna(subset=["oos_r2"]).set_index("label")[["oos_r2"]].rename(
        columns={"oos_r2": "Out-of-sample R²"}
    )
    st.bar_chart(chart)
    st.caption("2024 only. Above zero means the forecast beat the 2020–2022 average return.")

    st.markdown("### The same score, year by year")
    by_fold = folds.pivot(index="fold", columns="model", values="oos_r2")
    by_fold = by_fold.reindex(["train", "val", "test"]).rename(index=FOLD_LABEL, columns=LABELS)
    st.bar_chart(by_fold)
    st.caption(
        "2020–2022 is in-sample for both fits. 2023 and 2024 were not used to "
        "estimate the slopes."
    )

    if events is not None and {"par_mkt", "ret_ann_mkt", "fold"}.issubset(events.columns):
        st.markdown("### Does a high PAR come with a low announcement return?")
        plot = events.dropna(subset=["par_mkt", "ret_ann_mkt"]).copy()
        plot["Year"] = plot["fold"].map({
            "train": "2020–2022",
            "val": "2023",
            "test": "2024",
        })
        st.scatter_chart(
            plot,
            x="par_mkt",
            y="ret_ann_mkt",
            color="Year",
            x_label="PAR (market-adjusted)",
            y_label="Announcement return (market-adjusted)",
            height=420,
        )
        st.caption(
            "Each dot is one earnings announcement on Brian's panel. "
            "A reversal slopes down: the stock already ran up (or down) before "
            "the print, and the announcement week goes the other way."
        )

    test_corr = corr.loc[corr["fold"] == "test"].copy()
    test_corr["outcome"] = test_corr["outcome"].map({
        "ret_ann_mkt": "Announcement, market-adjusted",
        "ret_ann": "Announcement, stock only",
        "ret_post_mkt": "Days after the announcement, market-adjusted",
    })
    st.markdown("### The 2024 correlations")
    st.dataframe(
        test_corr.rename(columns={
            "outcome": "Window",
            "n": "Events",
            "pearson": "Pearson",
            "spearman": "Spearman",
        })[["Window", "Events", "Pearson", "Spearman"]],
        width="stretch",
        hide_index=True,
    )
    st.caption("Negative numbers are the reversal direction.")

    st.markdown("### If you traded the reversal in 2023 and 2024")
    rules = strategy.loc[
        strategy["fold"].isin(["val", "test"])
        & strategy["rule"].isin(["sign", "quintile_pooled_train"])
    ].copy()
    rules["rule"] = rules["rule"].map({
        "sign": "Long after a down PAR, short after an up PAR",
        "quintile_pooled_train": "Only the lowest and highest fifths of PAR",
    })
    rules["fold"] = rules["fold"].map({"val": "2023", "test": "2024"})
    for column in ("mean_signed_mkt", "mean_signed_raw", "hit_rate_mkt"):
        rules[column] = rules[column].map(
            lambda v, c=column: "—" if pd.isna(v) else (f"{v:.1%}" if "hit" in c else f"{v:.2%}")
        )
    st.dataframe(
        rules.rename(columns={
            "rule": "Rule",
            "fold": "Year",
            "events": "Events",
            "trades": "Trades",
            "longs": "Long",
            "shorts": "Short",
            "mean_signed_mkt": "Avg. market-adjusted return",
            "mean_signed_raw": "Avg. stock return",
            "hit_rate_mkt": "Share of winning trades",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Average return per trade, before costs. The per-stock quintile from the "
        "original app does not trade here: no name has 20 earlier events inside "
        "2020–2022."
    )


def _render_methodology() -> None:
    st.subheader("PAR methodology")
    st.caption("This is the original study's write-up, as it appeared in the research app.")

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
    st.dataframe(WINDOWS, width="stretch", hide_index=True)
    st.markdown(
        "**PAR** is the pre-announcement market-adjusted return over sessions "
        "[−4, −2]. It requires the close at $t=-5$ and includes the daily returns at "
        "$t=-4$, $t=-3$ and $t=-2$."
    )
    st.info(
        "The announcement window includes the trading day before, the release date, "
        "and the trading day after. It is not a pure post-release return."
    )

    st.markdown("### How the days line up")
    st.markdown(
        """
| Session | What happens |
| --- | --- |
| −5 | Starting close for PAR |
| −4, −3, −2 | PAR is earned. Session −2 is also the starting close for the announcement window. |
| −1 | First day of the announcement window |
| 0 | Release calendar date in the original study |
| +1 | Last day of the announcement window |
| +2 to +5 | Subsequent window |
| +6 to +8 | Later window |
"""
    )

    st.markdown("### Data and adjustments (original study)")
    st.markdown(
        "- Daily prices come from Yahoo Finance via yfinance, requested with "
        "`interval=\"1d\"`, `auto_adjust=False` and `actions=True`.\n"
        "- Returns use **adjusted close**, which already carries Yahoo's split and "
        "dividend adjustments. Because `auto_adjust=False` keeps the raw `Close` "
        "separate, no price is adjusted twice.\n"
        "- If adjusted close is missing, the unadjusted close is **never** substituted. "
        "The session stays missing and the affected event is excluded with a stated "
        "reason.\n"
        "- Events run from 1 January 2010 through 31 December 2025. Prices are "
        "padded before 2010 so the first windows have a starting close.\n"
        "- The benchmark is **SPY**.\n"
        "- The sample is 70 large S&P 500 names, studied one company at a time."
    )

    st.markdown("### Trading calendar and event alignment (original study)")
    st.markdown(
        "- Sessions come from the explicit `XNYS` exchange calendar, never from "
        "intersecting the dates that happened to arrive in a price download.\n"
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
    st.markdown(
        "**Completed-release rule.** Only dates that have already occurred are kept. "
        "A date-only record or a midnight placeholder timestamp is never treated as "
        "a verified release time; it is classified as `unknown`."
    )
    st.markdown(
        "- Exact duplicates are removed. Records describing the same release but "
        "disagreeing on the date are **flagged for review**, not arbitrarily resolved.\n"
        "- Dates are never invented, never assumed to be 90 days apart, and never "
        "generated from a four-per-year rule."
    )

    st.markdown("### Pseudo-event rules")
    st.markdown(
        "- For each eligible actual event, candidate pseudo-dates are placed "
        "10–40 trading sessions before it, inclusive.\n"
        "- A candidate is eligible only if every required stock and benchmark price "
        "is present across sessions [−5, +8] and its return-analysis interval contains "
        "no known earnings release.\n"
        "- Screening uses **all** available confirmed releases, including those "
        "outside the selected event range.\n"
        "- Where the earnings source is incomplete, unknown releases cannot be ruled "
        "out. Pseudo windows are therefore *screened*, not *proven clean*.\n"
        "- For each actual event one candidate is drawn uniformly at random, and the "
        "draw is repeated (500 times in the default run).\n"
        "- Repetitions are a Monte Carlo sensitivity check. They do not add "
        "independent observations, they are never pooled into one large sample, and "
        "the 5th–95th percentile range across draws is not a confidence interval."
    )

    st.markdown("### Statistics")
    st.markdown(
        "- Pearson correlation between PAR and each outcome return.\n"
        "- OLS: `outcome_return = intercept + slope × PAR`, reporting slope, "
        "intercept and R-squared.\n"
        "- A **reversal** is `PAR × outcome_return < 0`. Cases where either value "
        "is exactly zero are reported separately, never counted as a reversal or a "
        "continuation.\n"
        "- Negative-PAR and positive-PAR groups are summarised separately by count, "
        "mean, median and opposite-sign frequency.\n"
        "- The **negative-PAR minus positive-PAR mean difference** compares events for "
        "one company at different points in time. It is not a simultaneously tradable "
        "long-short portfolio return.\n"
        "- Everything in the original app is exploratory and in-sample. There is no "
        "2022 / 2023 / 2024 cut."
    )

    st.markdown("### PAR quintile strategy")
    st.markdown(
        "- Thresholds: the 20th and 80th percentiles of PAR over the stock's earlier "
        "included events only, requiring at least 20. The current event never enters "
        "its own thresholds. Events with less history are skipped.\n"
        "- PAR below the 20th percentile (Q1): long. Above the 80th (Q5): short. "
        "Otherwise no trade. Ties with a threshold do not trade.\n"
        "- Held over the announcement window [−1, +1]. Raw stock strategy return = "
        "position × stock return; signed market-adjusted return = position × "
        "(stock return − benchmark return).\n"
        "- Win rate: share of executed trades with a strictly positive signed return. "
        "Cumulative P&L: fixed notional per trade, summed, not compounded.\n"
        "- The Q1 mean minus the Q5 mean is a group spread, not the average strategy "
        "return.\n"
        "- Entry is assumed at the same close that finishes PAR (session −2). That "
        "is a same-close execution assumption.\n"
        "- All figures are before costs."
    )

    st.markdown("### Limitations")
    for line in (
        "Roughly four earnings events per year means the sample is small, and it stays "
        "small no matter how many pseudo repetitions are run.",
        "Results for one company need not generalise to any other company, and results "
        "for 70 large companies chosen in hindsight (survivorship bias) need not "
        "generalise to the index.",
        "The 70 stocks are not independent: they report in the same earnings seasons and "
        "share market-wide shocks.",
        "This is a time-series adaptation. It is not the cross-sectional portfolio "
        "sorting used in the original academic study.",
        "Actual historical release dates may not have been known in advance. This is an "
        "ex-post event study, not a point-in-time trading backtest.",
        "SPY is a practical substitute for the market return.",
        "Adjusted-close returns do not represent executable trade prices.",
        "No transaction costs, financing, borrow costs or implementation constraints are "
        "modelled.",
        "A negative relationship between PAR and the announcement return does not prove "
        "a liquidity-provider mechanism, or any other mechanism.",
        "No machine learning is used and no out-of-sample performance is claimed.",
    ):
        st.markdown(f"- {line}")


def _pct(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.2%}"


def _render_stocks() -> None:
    year = year_events()
    stocks = stock_table(year)
    sectors = sector_table(stocks)
    quarters = quarter_table(year)

    n_tickers = year["TickerId"].nunique()
    n_symbols = year["Symbol"].nunique()
    n_events = len(year)
    par_stocks = int((stocks["ahead"] == "PAR").sum())
    imp_stocks = int((stocks["ahead"] == "IMP").sum())
    ties = int((stocks["ahead"] == "tie").sum())

    st.subheader("2024 by stock and quarter")
    st.write(
        f"The held-out year has **{n_events} announcements** across **{n_tickers} "
        f"names** ({n_symbols} ticker symbols; ANET and ARE each appear twice in "
        f"Brian's file). Almost every name has four quarterly prints. "
        f"PAR's signed announcement return is higher than IMP OLS on "
        f"**{par_stocks} names**, IMP is higher on **{imp_stocks}**, and "
        f"{ties} are tied. "
        "A return here is the market-adjusted announcement return if you took "
        "that method's direction: PAR goes long after a down move and short "
        "after an up move; IMP goes with the sign of its OLS forecast. "
        "Before costs."
    )

    st.markdown("### Which industries PAR leads")
    sector_show = sectors.copy()
    for column in ("avg_par", "avg_imp", "par_minus_imp"):
        sector_show[column] = sector_show[column].map(_pct)
    st.dataframe(
        sector_show.rename(columns={
            "avg_par": "Avg PAR trade",
            "avg_imp": "Avg IMP trade",
            "par_minus_imp": "PAR minus IMP",
            "par_ahead_stocks": "Names PAR leads",
            "imp_ahead_stocks": "Names IMP leads",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Average of the per-stock 2024 averages, so a name with four prints "
        "counts the same as a name with three. GICS sectors from the S&P 500 list."
    )

    st.markdown("### Which stocks PAR leads")
    sector_choice = st.selectbox(
        "Sector",
        ["All"] + sorted(stocks["Sector"].unique()),
    )
    ahead_choice = st.selectbox("Who is ahead for the year", ["All", "PAR", "IMP", "tie"])
    shown = stocks.copy()
    if sector_choice != "All":
        shown = shown.loc[shown["Sector"] == sector_choice]
    if ahead_choice != "All":
        shown = shown.loc[shown["ahead"] == ahead_choice]
    stock_show = shown.copy()
    for column in ("avg_realized", "avg_par", "avg_imp", "par_minus_imp"):
        stock_show[column] = stock_show[column].map(_pct)
    st.dataframe(
        stock_show.rename(columns={
            "avg_realized": "Avg realized",
            "avg_par": "Avg PAR trade",
            "avg_imp": "Avg IMP trade",
            "par_minus_imp": "PAR minus IMP",
            "par_ahead_events": "Prints PAR won",
            "imp_ahead_events": "Prints IMP won",
        }),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Every 2024 print")
    q_choice = st.selectbox("Quarter", ["All"] + sorted(quarters["Quarter"].unique()))
    q_sector = st.selectbox(
        "Sector for the print table",
        ["All"] + sorted(quarters["Sector"].unique()),
        key="print_sector",
    )
    prints = quarters.copy()
    if q_choice != "All":
        prints = prints.loc[prints["Quarter"] == q_choice]
    if q_sector != "All":
        prints = prints.loc[prints["Sector"] == q_sector]
    print_show = prints.copy()
    for column in ("realized", "par", "par_return", "imp_forecast", "imp_return", "par_minus_imp"):
        print_show[column] = print_show[column].map(_pct)
    st.dataframe(
        print_show.rename(columns={
            "EventDate": "Event date",
            "realized": "Realized announcement",
            "par": "PAR",
            "par_return": "PAR trade",
            "imp_forecast": "IMP forecast",
            "imp_return": "IMP trade",
            "par_minus_imp": "PAR minus IMP",
            "ahead": "Ahead this print",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Realized announcement is the market-adjusted return over IMP's "
        "[−1, +1] window. PAR is the pre-announcement signal. "
        "PAR trade and IMP trade are that realized return after each method "
        "chooses long or short. They are not the raw stock return, and they "
        "are not a portfolio you could have held at the same time in every name."
    )


def _render_imp_methodology() -> None:
    st.subheader("IMP methodology")
    st.caption(
        "One page. Identifying Market Positioning Ahead of Earnings. "
        "The full column list is also in docs/imp_methodology.md."
    )

    st.markdown(
        "The question is predictive: using only information known **before** the "
        "announcement, can a model forecast the stock’s market-adjusted return in "
        "the announcement week? The sample is Brian’s three Parquet files — about "
        "200 U.S. names, prices 2020–2024. One row is one earnings print that "
        "already sits on a price day. Features are copied from two sessions before "
        "the return clock starts. Nothing from the release itself enters the model."
    )

    st.markdown("### How a print is placed on the calendar")
    st.markdown(
        "`MarketTimeId` says when the company reported. **1 = before the open:** "
        "the return clock (`AnchorDate`) is that session. **2 = after the close:** "
        "the clock moves to the **next** session. `DecisionDate` is two trading "
        "sessions before that anchor. Predictors must not use anything after that close."
    )
    st.markdown(
        """
| Date | Meaning |
| --- | --- |
| `EventDate` | Vendor calendar date of the release |
| `ReleaseDate` | That date on the trading calendar |
| `AnchorDate` | `t = 0` for the return windows (after-close is shifted) |
| `DecisionDate` | Feature as-of date, and the date used to cut train / 2023 / 2024 |
"""
    )
    st.markdown(
        "Train: decision dates through 31 Dec 2022. Validation: 2023 (penalties "
        "are chosen here). Test: 2024. Medians and scaling are fit on train only."
    )

    st.markdown("### What the model is trying to forecast")
    st.markdown(
        "A window `[a, b]` is the close at session `a−1` through the close at "
        "session `b`, counted from `AnchorDate`. The default target is "
        "`ret_ann_mkt`. The market subtracted is the equal-weight universe index "
        "(not SPY, unless someone changes the config)."
    )
    st.dataframe(
        pd.DataFrame([
            {"Variable": "ret_ann", "Meaning": "Stock return, announcement window [−1, +1]: close at anchor−2 → close at anchor+1."},
            {"Variable": "ret_ann_mkt", "Meaning": "ret_ann minus the same window on the benchmark. Default target."},
            {"Variable": "ret_post", "Meaning": "Stock return [+1, +5]: close at the anchor → close at anchor+5. Overlaps the announcement window on day +1."},
            {"Variable": "ret_post_mkt", "Meaning": "ret_post minus the same window on the benchmark."},
            {"Variable": "ret_next", "Meaning": "One-session stock return from the ReleaseDate close to the next session (not from the anchor)."},
            {"Variable": "ret_next_mkt", "Meaning": "ret_next minus the same one-session move on the benchmark."},
            {"Variable": "dir_ann", "Meaning": "Sign of ret_ann_mkt. Reserved for classification; not the current target."},
            {"Variable": "vol_ann", "Meaning": "Standard deviation of daily stock returns inside [−1, +1]."},
            {"Variable": "vol_post", "Meaning": "Standard deviation of daily stock returns inside [+1, +5]."},
        ]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Variables the model is allowed to see")
    st.markdown(
        "These eight numbers are **copied** from the decision-day row. IMP does "
        "not recompute them. `QScoreLong` equals trend plus speed on observed rows. "
        "Analyst counts come from the latest monthly snapshot on or before the "
        "decision date, and only if that snapshot is at most 31 days old. "
        "Missing counts are filled with the training-fold median."
    )
    st.dataframe(
        pd.DataFrame([
            {"Variable": "QScoreTrendLong", "Source": "Daily prices", "Meaning": "Vendor long-horizon trend score at the decision close."},
            {"Variable": "QScoreSpeedLong", "Source": "Daily prices", "Meaning": "Vendor long-horizon speed / momentum score at the decision close."},
            {"Variable": "QScoreLong", "Source": "Daily prices", "Meaning": "Trend + speed at the decision close."},
            {"Variable": "analystRatingsStrongBuy", "Source": "Monthly analyst file", "Meaning": "Count of Strong Buy ratings on that snapshot."},
            {"Variable": "analystRatingsBuy", "Source": "Monthly analyst file", "Meaning": "Count of Buy ratings."},
            {"Variable": "analystRatingsHold", "Source": "Monthly analyst file", "Meaning": "Count of Hold ratings."},
            {"Variable": "analystRatingsSell", "Source": "Monthly analyst file", "Meaning": "Count of Sell ratings."},
            {"Variable": "analystRatingsStrongSell", "Source": "Monthly analyst file", "Meaning": "Count of Strong Sell ratings."},
        ]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Other variables that build the row")
    st.dataframe(
        pd.DataFrame([
            {"Variable": "TickerId", "Meaning": "Internal name id. Joins prices to earnings. Analysis is by this id, not by Symbol."},
            {"Variable": "Symbol", "Meaning": "Ticker (ABNB). Used to attach the monthly analyst file. The same symbol can appear on two ids."},
            {"Variable": "ShortName", "Meaning": "SYMBOL:EXCHANGE. The exchange picks the session calendar."},
            {"Variable": "InstrumentName", "Meaning": "Company name."},
            {"Variable": "Date", "Meaning": "Trading date on the daily price row."},
            {"Variable": "LastPrice", "Meaning": "Closing price used for stock returns. Adjustment convention is not stated in-file."},
            {"Variable": "Open, High, Low, Volume", "Meaning": "Session prices and share volume. Not in the default feature list."},
            {"Variable": "MarketCap", "Meaning": "Market value on that date. Used only if the benchmark is cap-weighted."},
            {"Variable": "CurrencyId", "Meaning": "840 = USD; a few rows are CAD."},
            {"Variable": "Id", "Meaning": "Vendor earnings-event id."},
            {"Variable": "MarketTimeId", "Meaning": "1 before open, 2 after close. Unknown values drop the event."},
            {"Variable": "BenchmarkAdjClose", "Meaning": "Benchmark level used to build every *_mkt return. Not stored on the event table."},
        ]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Models and scores")
    st.markdown(
        "Linear (OLS, ridge, lasso), trees, and a small network share the same "
        "target, features, and split. Ridge and lasso choose their penalty on 2023."
    )
    st.dataframe(
        pd.DataFrame([
            {"Score": "oos_r2", "Meaning": "How much 2024 variation is explained versus guessing the 2020–2022 average return. Primary score."},
            {"Score": "r2", "Meaning": "Same idea, but versus that fold’s own average."},
            {"Score": "ic", "Meaning": "Pearson correlation of the forecast with the realized return."},
            {"Score": "rank_ic", "Meaning": "Spearman (rank) correlation."},
            {"Score": "hit_rate", "Meaning": "Share of prints where the forecast and the realized return have the same sign. Zeros count as misses."},
            {"Score": "rmse, mae", "Meaning": "Typical size of the forecast error."},
            {"Score": "rmse_train_mean / rmse_zero", "Meaning": "Error if you always guess the training-period average, or always guess zero."},
            {"Score": "top_minus_bottom", "Meaning": "Average realized return in the highest predicted fifth minus the lowest."},
        ]),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "A useful model should beat those two guesses and show a non-negative "
        "out-of-sample R² on 2024. The linear models on this panel do not."
    )


def _render_what_changed() -> None:
    st.subheader("What changes when PAR is scored on Brian's panel")
    st.markdown(
        "The tab next to this one is the original write-up. The comparison uses "
        "the same PAR idea on a different clock, a different sample, and a "
        "held-out year. That is why the two UIs do not show the same numbers."
    )
    st.dataframe(
        pd.DataFrame([
            {
                "Piece": "Question",
                "Original PAR study": "Does PAR reverse into the announcement, and is that stronger than on fake dates?",
                "This comparison": "If you only use information from before the print, does PAR forecast 2024 better than QScore and analyst ratings?",
            },
            {
                "Piece": "Sample",
                "Original PAR study": "70 S&P 500 names, Yahoo prices, 2010–2025",
                "This comparison": "About 200 names, Brian's three Parquet files, 2020–2024",
            },
            {
                "Piece": "Benchmark",
                "Original PAR study": "SPY",
                "This comparison": "Equal-weight average of the same ~200 names",
            },
            {
                "Piece": "Day 0",
                "Original PAR study": "The calendar release date, even after the close",
                "This comparison": "Before the open: that session. After the close: the next session",
            },
            {
                "Piece": "PAR window",
                "Original PAR study": "Close at t=−5 through close at t=−2, around calendar day 0",
                "This comparison": "The same three sessions, but around Brian's shifted anchor. It still ends on the decision close",
            },
            {
                "Piece": "Train and test",
                "Original PAR study": "None. The whole 2010–2025 sample is described at once. Fake dates are a placebo, not a test year",
                "This comparison": "Fit on 2020–2022, check 2023, report 2024",
            },
            {
                "Piece": "Quintile rule",
                "Original PAR study": "Each stock's own earlier events, after 20 releases",
                "This comparison": "That rule never trades on a 2020–2024 history. The comparison uses the 20th and 80th percentile of every 2020–2022 PAR instead",
            },
        ]),
        width="stretch",
        hide_index=True,
    )
    st.markdown(
        "Brian's models never see the earnings print. They copy eight numbers from "
        "two sessions before the anchor: three QScore series and five analyst-rating "
        "counts. The target they share with the reversal is the market-adjusted "
        "announcement return."
    )


def main() -> None:
    st.set_page_config(page_title="PAR reversal vs IMP", layout="wide")
    st.title("PAR reversal vs IMP")
    st.caption(
        "Two research questions on one 2024 test year. "
        "**PAR methodology** and **IMP methodology** are the write-ups."
    )

    if not (REVERSAL / "metrics.csv").exists():
        st.info("No comparison output yet. From the repo root run: python script/run_reversal.py --with-lin")
        return

    summary, folds, strategy, corr, events = _load()
    comparison, stocks, par_method, imp_method, changed = st.tabs([
        "2024 comparison",
        "2024 by stock",
        "PAR methodology",
        "IMP methodology",
        "What changed for IMP",
    ])
    with comparison:
        _render_comparison(summary, folds, strategy, corr, events)
    with stocks:
        _render_stocks()
    with par_method:
        _render_methodology()
    with imp_method:
        _render_imp_methodology()
    with changed:
        _render_what_changed()


if __name__ == "__main__":
    main()
