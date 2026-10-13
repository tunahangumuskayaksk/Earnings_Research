"""Synthetic fixtures. These exist for tests only and never reach the application."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.config import AnalysisConfig
from earnings_reversal.events import AlignedPanel
from earnings_reversal.prices import PriceData
from earnings_reversal.cache import Provenance

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)

SYNTHETIC_RELEASES = [
    "2021-01-27 16:30", "2021-04-28 16:30", "2021-07-27 16:30", "2021-10-28 16:30",
    "2022-01-27 16:30", "2022-04-28 16:30", "2022-07-28 16:30", "2022-10-27 16:30",
    "2023-02-02 16:30", "2023-05-04 16:30", "2023-08-03 16:30", "2023-11-02 16:30",
]


def long_synthetic_releases(first_year: int = 2014, last_year: int = 2021) -> list[str]:
    """Quarterly after-close releases on the first session on/after the 28th."""
    from earnings_reversal.trading_calendar import trading_sessions

    sessions = trading_sessions(f"{first_year}-01-01", f"{last_year}-12-31")
    releases = []
    for year in range(first_year, last_year + 1):
        for month in (1, 4, 7, 10):
            day = sessions[sessions >= pd.Timestamp(year, month, 28)][0]
            releases.append(f"{day.date()} 16:30")
    return releases


def build_synthetic_inputs(ticker: str = "TEST", price_seed: int = 99,
                           n_repetitions: int = 25, releases=None,
                           price_start: str = "2020-06-01", price_end: str = "2024-03-31",
                           event_start: str = "2021-01-01", event_end: str = "2023-12-31"):
    """Random-walk prices plus quarterly releases, wrapped as pipeline inputs."""
    from earnings_reversal.earnings import EarningsData, normalise_yahoo_earnings
    from earnings_reversal.pipeline import Inputs
    from earnings_reversal.trading_calendar import trading_sessions

    releases = SYNTHETIC_RELEASES if releases is None else releases
    sessions = trading_sessions(price_start, price_end)
    rng = np.random.default_rng(price_seed)
    stock = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.018, size=len(sessions)))
    bench = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.009, size=len(sessions)))

    index = pd.DatetimeIndex(
        [pd.Timestamp(s) for s in releases]
    ).tz_localize("America/New_York")
    frame = pd.DataFrame({
        "EPS Estimate": np.full(len(releases), 1.0),
        "Reported EPS": np.full(len(releases), 1.1),
        "Surprise(%)": np.full(len(releases), 2.0),
    }, index=index)

    records = normalise_yahoo_earnings(frame, ticker, NOW)
    provenance = Provenance(
        kind="earnings", tickers=[ticker], requested_start=None, requested_end=None,
        retrieved_at_utc="1970-01-01T00:00:00+00:00", data_source="synthetic fixture",
        adjustment_settings={}, package_versions={}, cache_key=ticker, cache_file="",
    )
    config = AnalysisConfig(ticker=ticker, benchmark="BENCH",
                            event_start=event_start, event_end=event_end,
                            seed=42, n_repetitions=n_repetitions)
    return Inputs(
        config=config,
        stock=make_price_data(ticker, sessions, stock),
        benchmark=make_price_data("BENCH", sessions, bench),
        earnings=EarningsData(records=records, provenance=provenance),
    )


def build_long_synthetic_inputs(ticker: str = "LONG", price_seed: int = 7,
                                n_repetitions: int = 40):
    """32 quarterly events (2014-2021): enough history for the quintile strategy."""
    return build_synthetic_inputs(
        ticker, price_seed=price_seed, n_repetitions=n_repetitions,
        releases=long_synthetic_releases(), price_start="2013-06-01",
        price_end="2022-06-30", event_start="2014-01-01", event_end="2021-12-31",
    )


def make_panel(sessions, stock_prices, benchmark_prices,
               stock_ticker="TEST", benchmark_ticker="BENCH") -> AlignedPanel:
    return AlignedPanel(
        sessions=pd.DatetimeIndex(pd.to_datetime(sessions), name="session"),
        stock_prices=np.asarray(stock_prices, dtype=float),
        benchmark_prices=np.asarray(benchmark_prices, dtype=float),
        stock_ticker=stock_ticker,
        benchmark_ticker=benchmark_ticker,
    )


def make_price_data(ticker, sessions, adj_close, close=None) -> PriceData:
    sessions = pd.DatetimeIndex(pd.to_datetime(sessions), name="session")
    frame = pd.DataFrame({
        "Open": np.nan, "High": np.nan, "Low": np.nan,
        "Close": close if close is not None else adj_close,
        "Adj Close": adj_close,
        "Volume": np.nan, "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=sessions)
    provenance = Provenance(
        kind="prices", tickers=[ticker], requested_start=str(sessions.min().date()),
        requested_end=str(sessions.max().date()), retrieved_at_utc="1970-01-01T00:00:00+00:00",
        data_source="synthetic test fixture", adjustment_settings={},
        package_versions={}, cache_key="test", cache_file="",
    )
    return PriceData(ticker=ticker, frame=frame, raw_frame=frame, provenance=provenance)


@pytest.fixture
def real_sessions():
    """Genuine NYSE sessions spanning a holiday and several weekends."""
    from earnings_reversal.trading_calendar import trading_sessions

    return trading_sessions("2021-06-01", "2021-08-31")


@pytest.fixture
def geometric_panel(real_sessions):
    """Prices that grow by fixed daily rates so window returns are exact."""
    n = len(real_sessions)
    stock = 100.0 * np.power(1.01, np.arange(n))
    bench = 50.0 * np.power(1.005, np.arange(n))
    return make_panel(real_sessions, stock, bench)
