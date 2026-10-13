"""Streamlit research dashboard: short-term return reversals around earnings.

Launch with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

from earnings_reversal.cache import DiskCache
from earnings_reversal.config import (
    DEFAULT_BENCHMARK,
    DEFAULT_EVENT_END,
    DEFAULT_EVENT_START,
    DEFAULT_REPETITIONS,
    DEFAULT_SEED,
    DEFAULT_TICKER,
    AnalysisConfig,
)
from earnings_reversal.earnings import parse_manual_csv
from earnings_reversal.pipeline import (
    AnalysisResult,
    Inputs,
    load_inputs,
    price_range,
    run_analysis,
)
from earnings_reversal.universe import ALL_TICKERS, GROUPS, StockGroup, display_name
from ui import (
    tab_data,
    tab_events,
    tab_inspector,
    tab_methodology,
    tab_pseudo,
    tab_strategy,
    tab_summary,
)

st.set_page_config(
    page_title="Earnings Return Reversals",
    page_icon="\U0001F4C9",
    layout="wide",
)

CACHE = DiskCache()

VIEW_SINGLE = "Single stock"
GROUP_BY_VIEW = {group.label: group for group in GROUPS}
OTHER_TICKER = "Other ticker\u2026"


@st.cache_data(show_spinner=False)
def cached_inputs(ticker: str, benchmark: str, event_start: str, event_end: str,
                  refresh_token: int) -> Inputs:
    """Retrieval only. Keyed on the fields that actually affect a download.

    A refresh is performed by emptying the disk cache before bumping
    ``refresh_token``, so this function never needs a force flag and a widget
    change can never trigger a download on its own.
    """
    config = AnalysisConfig(
        ticker=ticker, benchmark=benchmark,
        event_start=event_start, event_end=event_end,
    )
    return load_inputs(config, CACHE, force_refresh=False)


def invalidate_disk_cache(config: AnalysisConfig, tickers) -> None:
    """Drop the cached artefacts for these tickers so they are re-downloaded."""
    start, end = price_range(config)
    for ticker in list(tickers) + [config.benchmark]:
        CACHE.delete("prices", f"{ticker}__{start}__{end}")
    for ticker in tickers:
        CACHE.delete("earnings", ticker)


@st.cache_data(show_spinner=False)
def cached_analysis(config: AnalysisConfig, manual_csv_text: str, refresh_token: int):
    """Computation only, over whatever inputs are available."""
    inputs = cached_inputs(config.ticker, config.benchmark, config.event_start,
                           config.event_end, refresh_token)
    manual_records = pd.DataFrame()
    errors: list[dict] = []
    if manual_csv_text:
        manual_records, errors = parse_manual_csv(
            manual_csv_text, retrieved_at=datetime.now(timezone.utc)
        )
    for_ticker = 0
    if len(manual_records):
        for_ticker = int(
            (manual_records["ticker"].astype(str).str.upper() == config.ticker).sum()
        )
    result = run_analysis(inputs, manual_records if len(manual_records) else None, config)
    return result, errors, int(len(manual_records)), for_ticker


def stock_config(base: AnalysisConfig, ticker: str) -> AnalysisConfig:
    excluded = st.session_state["excluded_by_ticker"].get(ticker, [])
    return AnalysisConfig(
        ticker=ticker,
        benchmark=base.benchmark,
        event_start=base.event_start,
        event_end=base.event_end,
        seed=base.seed,
        n_repetitions=base.n_repetitions,
        excluded_release_dates=tuple(excluded),
    ).normalised()


def sidebar() -> tuple[str, AnalysisConfig, int]:  # noqa: C901 - linear UI layout
    st.sidebar.title("Research settings")
    view = st.sidebar.radio("View", [VIEW_SINGLE, *GROUP_BY_VIEW], key="view")
    group = GROUP_BY_VIEW.get(view)

    choice = st.sidebar.selectbox(
        f"Stock ({len(ALL_TICKERS)} available)",
        list(ALL_TICKERS) + [OTHER_TICKER],
        format_func=lambda t: t if t == OTHER_TICKER else display_name(t),
        key="selected_ticker",
        disabled=group is not None,
        help="Every single-stock page shows the stock chosen here. The first 30 are "
             "the original universe; the next 40 are the added stocks.",
    )
    if choice == OTHER_TICKER:
        ticker = st.sidebar.text_input(
            f"Ticker outside the {len(ALL_TICKERS)} stocks", key="other_ticker",
            disabled=group is not None,
        ).strip().upper() or DEFAULT_TICKER
    else:
        ticker = choice

    benchmark = st.sidebar.text_input(
        "Benchmark ticker", value=DEFAULT_BENCHMARK,
        help="SPY is a practical substitute for the market return.",
    ).strip().upper()

    default_start = date.fromisoformat(DEFAULT_EVENT_START)
    default_end = date.fromisoformat(DEFAULT_EVENT_END)
    start = st.sidebar.date_input(
        "Event range start", value=default_start,
        min_value=date(1990, 1, 1), max_value=date(2100, 12, 31),
    )
    end = st.sidebar.date_input(
        "Event range end", value=default_end,
        min_value=date(1990, 1, 1), max_value=date(2100, 12, 31),
    )
    if start > end:
        st.sidebar.error("The event range start is after its end; using the default range.")
        start, end = default_start, default_end

    seed = st.sidebar.number_input(
        "Random seed", value=DEFAULT_SEED, min_value=0, max_value=2**31 - 1, step=1,
    )
    repetitions = st.sidebar.number_input(
        "Pseudo-event repetitions", value=DEFAULT_REPETITIONS,
        min_value=1, max_value=5000, step=50,
        help="Monte Carlo sensitivity over which pseudo dates are drawn. More "
             "repetitions never increase the number of earnings observations.",
    )
    st.sidebar.caption(
        "Return windows are fixed in this first version to avoid an unnecessary "
        "parameter search. These settings apply to every stock."
    )

    base = AnalysisConfig(
        ticker=ticker,
        benchmark=benchmark or DEFAULT_BENCHMARK,
        event_start=pd.Timestamp(start).date().isoformat(),
        event_end=pd.Timestamp(end).date().isoformat(),
        seed=int(seed),
        n_repetitions=int(repetitions),
    )
    config = stock_config(base, ticker)

    st.sidebar.divider()
    if group is not None:
        label = f"Refresh data for all {len(group.members)} stocks"
        targets = list(group.tickers)
        help_text = ("Discard every cached download for the stocks in this view and "
                     "retrieve them again. This takes a minute or two.")
    else:
        label = f"Refresh data for {ticker}"
        targets = [ticker]
        help_text = "Discard the cached download for this stock and retrieve it again."
    if st.sidebar.button(label, help=help_text):
        invalidate_disk_cache(config, targets)
        st.session_state["refresh_token"] = st.session_state.get("refresh_token", 0) + 1
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(
        "Data is served from the local cache, so changing any widget above does not "
        "trigger a download."
    )
    return view, config, int(st.session_state.get("refresh_token", 0))


def render_single(config: AnalysisConfig, refresh_token: int) -> None:
    try:
        with st.spinner(f"Running the event analysis for {config.ticker}\u2026"):
            result, manual_errors, manual_total, manual_for_ticker = cached_analysis(
                config, st.session_state["manual_csv_text"], refresh_token
            )
    except Exception as exc:
        st.error(
            f"The analysis could not be built: {type(exc).__name__}: {exc}\n\n"
            "No results are shown. Nothing has been substituted for the missing data."
        )
        return

    st.session_state["manual_errors"] = manual_errors
    st.session_state["manual_accepted"] = manual_total
    st.session_state["manual_for_ticker"] = manual_for_ticker

    st.markdown(f"### {display_name(config.ticker)}")
    if result.blocking_error:
        st.error(result.blocking_error)

    tabs = st.tabs([
        "Data & Coverage",
        "Earnings Event Analysis",
        "Actual vs Pseudo",
        "PAR Quintile Strategy",
        "Event Inspector",
        "Methodology",
    ])
    with tabs[0]:
        tab_data.render(result, CACHE)
    with tabs[1]:
        tab_events.render(result)
    with tabs[2]:
        tab_pseudo.render(result)
    with tabs[3]:
        tab_strategy.render(result)
    with tabs[4]:
        tab_inspector.render(result)
    with tabs[5]:
        tab_methodology.render(result)


def universe_results(config: AnalysisConfig, refresh_token: int,
                     group: StockGroup) -> dict:
    results: dict = {}
    tickers = group.tickers
    progress = st.progress(0.0, text="Preparing the stocks\u2026")
    for index, ticker in enumerate(tickers):
        progress.progress(index / len(tickers),
                          text=f"Analysing {ticker} ({index + 1} of {len(tickers)})")
        try:
            result, _, _, _ = cached_analysis(
                stock_config(config, ticker), st.session_state["manual_csv_text"],
                refresh_token,
            )
        except Exception as exc:
            failed_config = stock_config(config, ticker)
            result = AnalysisResult(config=failed_config, inputs=Inputs(config=failed_config))
            result.blocking_error = f"{type(exc).__name__}: {exc}"
        results[ticker] = result
    progress.empty()
    return results


def open_stock(ticker: str) -> None:
    st.session_state["selected_ticker"] = ticker
    st.session_state["view"] = VIEW_SINGLE


def main() -> None:
    st.title("Short-term return reversals around earnings announcements")
    st.caption(
        "Descriptive single-company event studies, run one stock at a time for "
        f"{len(ALL_TICKERS)} long-standing S&P 500 companies and inspired by So & Wang "
        "(2014), *News-driven return reversals*. Not a replication, not a trading "
        "strategy, not evidence of a liquidity-provision mechanism, and not a "
        "machine-learning project. All results are exploratory and in-sample."
    )

    st.session_state.setdefault("manual_csv_text", "")
    st.session_state.setdefault("excluded_by_ticker", {})
    st.session_state.setdefault("refresh_token", 0)
    st.session_state.setdefault("view", VIEW_SINGLE)
    st.session_state.setdefault("selected_ticker", DEFAULT_TICKER)

    view, config, refresh_token = sidebar()

    group = GROUP_BY_VIEW.get(view)
    if group is not None:
        results = universe_results(config, refresh_token, group)
        tab_summary.render(results, config, open_stock, group)
    else:
        render_single(config, refresh_token)


if __name__ == "__main__":
    main()
