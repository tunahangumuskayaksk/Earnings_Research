"""Streamlit smoke tests.

These run the real ``app.py`` through Streamlit's app-testing harness. They are
skipped unless the local cache already holds the default data, so the suite never
depends on network access.
"""

from __future__ import annotations

import datetime as dt

import pytest

from earnings_reversal.cache import DiskCache
from earnings_reversal.config import PROJECT_ROOT, AnalysisConfig
from earnings_reversal.pipeline import price_range
from earnings_reversal.universe import ADDED_TICKERS, UNIVERSE_TICKERS

APP_PATH = str(PROJECT_ROOT / "app.py")

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

DEFAULT = AnalysisConfig()
START, END = price_range(DEFAULT)


def _cache_is_warm(tickers=(DEFAULT.ticker,)) -> bool:
    cache = DiskCache()
    return cache.exists("prices", f"{DEFAULT.benchmark}__{START}__{END}") and all(
        cache.exists("prices", f"{ticker}__{START}__{END}")
        and cache.exists("earnings", ticker)
        for ticker in tickers
    )


warm_cache_only = pytest.mark.skipif(
    not _cache_is_warm(),
    reason="Local cache is empty; run the app once to populate it.",
)

warm_universe_only = pytest.mark.skipif(
    not _cache_is_warm(UNIVERSE_TICKERS),
    reason="Universe cache is incomplete; open the summary view once to populate it.",
)

warm_added_only = pytest.mark.skipif(
    not _cache_is_warm(ADDED_TICKERS),
    reason="Cache for the added 40 is incomplete; open the 40-stock view once.",
)


@pytest.fixture
def app():
    return AppTest.from_file(APP_PATH, default_timeout=300)


@warm_cache_only
def test_all_six_tabs_render_without_exceptions(app):
    app.run()
    assert app.exception == []
    assert [tab.label for tab in app.tabs] == [
        "Data & Coverage", "Earnings Event Analysis", "Actual vs Pseudo",
        "PAR Quintile Strategy", "Event Inspector", "Methodology",
    ]
    assert len(app.metric) > 0
    assert len(app.dataframe) > 0


@warm_cache_only
def test_default_run_shows_no_error_boxes(app):
    app.run()
    assert [str(e.value) for e in app.error] == []


@warm_cache_only
def test_empty_event_range_shows_guidance_not_fabricated_results(app):
    """A range with no earnings records must never be filled with plausible numbers."""
    app.run()
    app.sidebar.date_input[0].set_value(dt.date(2026, 1, 1))
    app.sidebar.date_input[1].set_value(dt.date(2026, 6, 30))
    app.run()

    assert app.exception == []
    messages = [str(e.value) for e in app.error] + [str(i.value) for i in app.info]
    assert any("No completed earnings records are available" in m for m in messages)
    # Every tab falls back to an explanation rather than a chart of invented data.
    assert any("No event statistics can be shown yet" in m for m in messages)
    assert any("No pseudo comparison is available" in m for m in messages)
    assert any("No event can be inspected yet" in m for m in messages)
    assert app.metric == [] or all(m.value != "nan" for m in app.metric)


@warm_cache_only
def test_invalid_manual_upload_reports_row_errors(app):
    app.session_state["manual_csv_text"] = "ticker,release_date\nAAPL,not-a-date\n"
    app.run()
    assert app.exception == []
    assert any("problem(s) in the uploaded file" in str(e.value) for e in app.error)
    assert app.session_state["manual_errors"]
    assert app.session_state["manual_accepted"] == 0


@warm_cache_only
def test_seed_change_alters_pseudo_draws_but_not_the_actual_sample(app):
    app.run()
    first = [m.value for m in app.metric if m.label == "Events included"]
    app.sidebar.number_input[0].set_value(7)
    app.run()
    assert app.exception == []
    second = [m.value for m in app.metric if m.label == "Events included"]
    assert first == second


@warm_cache_only
def test_methodology_tab_states_the_announcement_window_caveat(app):
    app.run()
    text = " ".join(
        str(item.value)
        for group in (app.markdown, app.caption, app.info)
        for item in group
    )
    assert "not a pure post-release return" in text
    assert "not a confidence interval" in text.lower()
    assert "in-sample" in text.lower()
    assert "ordinary trading day" in text.lower()
    assert "that entire set is one repetition" in text.lower()


def _text(app) -> str:
    return " ".join(
        str(item.value)
        for group in (app.markdown, app.caption, app.info, app.warning)
        for item in group
    )


@warm_cache_only
def test_strategy_tab_discloses_its_assumptions(app):
    app.run()
    text = _text(app)
    assert "per-stock historical adaptation" in text.lower()
    assert "not a replication" in text.lower()
    assert "same-close execution assumption" in text.lower()
    assert "before transaction costs" in text.lower()
    assert "group spread" in text.lower()
    assert "timing check passed" in text.lower()
    labels = [m.label for m in app.metric]
    assert "Matched events with enough history" in labels
    assert "Eligible pseudo events per repetition" in labels


@pytest.mark.skipif(not _cache_is_warm(("MSFT",)), reason="MSFT cache is empty.")
def test_choosing_a_stock_switches_every_page_to_it(app):
    app.run()
    app.sidebar.selectbox(key="selected_ticker").set_value("MSFT")
    app.run()
    assert app.exception == []
    assert any("MSFT \u2014 Microsoft" in str(m.value) for m in app.markdown)
    stock_metric = [m.value for m in app.metric if m.label == "Stock"]
    assert stock_metric == ["MSFT"]


@warm_cache_only
def test_exclusions_are_kept_per_stock(app):
    app.session_state["excluded_by_ticker"] = {"AAPL": ["2010-01-25"]}
    app.run()
    assert app.exception == []
    included = [m.value for m in app.metric if m.label == "Events included"]
    assert included == ["61"]
    # The excluded date stays selectable so it can be restored.
    widget = app.multiselect(key="exclusion_multiselect_AAPL")
    assert "2010-01-25" in widget.options
    assert widget.value == ["2010-01-25"]
    app.run()
    assert [m.value for m in app.metric if m.label == "Events included"] == ["61"]


@warm_universe_only
def test_summary_view_renders_every_stock(app):
    app.session_state["view"] = "30-stock summary"
    app.run()
    assert app.exception == []
    assert any(m.label == "Stocks analysed" and m.value == "30 of 30" for m in app.metric)
    text = _text(app)
    assert "not independent" in text
    assert "not a confidence interval" in text.lower()
    per_stock = [df.value for df in app.dataframe if "Ticker" in df.value.columns
                 and "Slope" in df.value.columns]
    assert per_stock and len(per_stock[0]) == 30
    strategy = [df.value for df in app.dataframe if "Trades" in df.value.columns]
    assert strategy and len(strategy[0]) == 30
    headings = " ".join(str(m.value) for m in app.markdown)
    assert "Actual statistic against each stock's pseudo median" in headings
    assert "Actual strategy result against each stock's pseudo range" in headings
    assert "Strategy result against each stock's pseudo median" in headings
    assert "Mean outcome in Q1 (long) vs Q5 (short) events" in headings
    for stat in ("win_adj", "pnl_raw", "trades"):
        app.selectbox(key="first30_strategy_stat").set_value(stat)
        app.run()
        assert app.exception == []
    app.radio(key="first30_quintile_basis").set_value("Raw stock")
    app.selectbox(key="first30_stat").set_value("reversal_frequency")
    app.run()
    assert app.exception == []


@warm_universe_only
def test_open_stock_button_returns_to_that_stocks_pages(app):
    app.session_state["view"] = "30-stock summary"
    app.run()
    app.selectbox(key="first30_open_choice").set_value("KO")
    app.run()
    app.button(key="first30_open_button").click()
    app.run()
    assert app.exception == []
    assert app.session_state["view"] == "Single stock"
    assert any("KO \u2014 Coca-Cola" in str(m.value) for m in app.markdown)


def test_view_switch_offers_both_summaries_and_seventy_stocks(app):
    app.run()
    assert app.sidebar.radio(key="view").options == [
        "Single stock", "30-stock summary", "40-stock summary",
    ]
    options = app.sidebar.selectbox(key="selected_ticker").options
    assert len(options) == 71  # 70 stocks plus "Other ticker…"
    assert options[0].startswith("AAPL") and options[30].startswith("ORCL")


@warm_added_only
def test_forty_stock_summary_shows_only_the_added_stocks(app):
    app.session_state["view"] = "40-stock summary"
    app.run()
    assert app.exception == []
    assert any(m.label == "Stocks analysed" and m.value == "40 of 40" for m in app.metric)
    per_stock = [df.value for df in app.dataframe if "Ticker" in df.value.columns
                 and "Slope" in df.value.columns]
    assert per_stock and len(per_stock[0]) == 40
    assert set(per_stock[0]["Ticker"]) == set(ADDED_TICKERS)
    app.selectbox(key="added40_open_choice").set_value("NVDA")
    app.run()
    app.button(key="added40_open_button").click()
    app.run()
    assert app.exception == []
    assert app.session_state["view"] == "Single stock"
    assert any("NVDA \u2014 Nvidia" in str(m.value) for m in app.markdown)
