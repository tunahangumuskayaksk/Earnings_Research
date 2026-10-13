"""Price normalisation, missing adjusted close and cache provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.cache import DiskCache, Provenance, package_versions, utc_now_iso
from earnings_reversal.prices import (
    PRICE_COLUMNS,
    _coverage_notes,
    _normalise_download,
    align_to_sessions,
    daily_returns,
    describe_adjustment,
)
from earnings_reversal.trading_calendar import trading_sessions
from conftest import make_price_data


def _raw_frame(index, adj_close, close=None):
    return pd.DataFrame({
        "Open": close if close is not None else adj_close,
        "High": close if close is not None else adj_close,
        "Low": close if close is not None else adj_close,
        "Close": close if close is not None else adj_close,
        "Adj Close": adj_close,
        "Volume": np.full(len(index), 1000.0),
        "Dividends": np.zeros(len(index)),
        "Stock Splits": np.zeros(len(index)),
    }, index=index)


def test_normalise_keeps_all_documented_columns():
    index = pd.DatetimeIndex(pd.date_range("2021-06-01", periods=5, freq="B"))
    frame = _normalise_download(_raw_frame(index, np.arange(1.0, 6.0)), "TEST")
    assert list(frame.columns) == PRICE_COLUMNS


def test_normalise_flattens_a_multiindex_download():
    index = pd.DatetimeIndex(pd.date_range("2021-06-01", periods=4, freq="B"))
    base = _raw_frame(index, np.arange(1.0, 5.0))
    multi = pd.concat({"TEST": base}, axis=1)
    frame = _normalise_download(multi, "TEST")
    assert list(frame.columns) == PRICE_COLUMNS
    assert frame["Adj Close"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_missing_adj_close_column_becomes_nan_not_the_close():
    index = pd.DatetimeIndex(pd.date_range("2021-06-01", periods=4, freq="B"))
    raw = _raw_frame(index, np.arange(1.0, 5.0)).drop(columns=["Adj Close"])
    frame = _normalise_download(raw, "TEST")
    assert frame["Adj Close"].isna().all()
    assert frame["Close"].notna().all()
    # The unadjusted close is never copied across.
    assert not frame["Adj Close"].equals(frame["Close"])


def test_coverage_notes_flag_absent_adjusted_close():
    sessions = trading_sessions("2021-06-01", "2021-06-15")
    raw = _raw_frame(sessions, np.full(len(sessions), np.nan),
                     close=np.arange(1.0, len(sessions) + 1.0))
    aligned = align_to_sessions(raw, "2021-06-01", "2021-06-15")
    notes = _coverage_notes("TEST", raw, aligned, "2021-06-01", "2021-06-15")
    assert any("adjusted close is entirely absent" in n for n in notes)


def test_coverage_notes_flag_partial_adjusted_close():
    sessions = trading_sessions("2021-06-01", "2021-06-15")
    adj = np.arange(1.0, len(sessions) + 1.0)
    adj[3] = np.nan
    raw = _raw_frame(sessions, adj, close=np.arange(1.0, len(sessions) + 1.0))
    aligned = align_to_sessions(raw, "2021-06-01", "2021-06-15")
    notes = _coverage_notes("TEST", raw, aligned, "2021-06-01", "2021-06-15")
    assert any("Unadjusted close is NOT substituted" in n for n in notes)
    assert any("left missing" in n for n in notes)


def test_coverage_notes_flag_a_short_history():
    sessions = trading_sessions("2021-06-10", "2021-06-20")
    raw = _raw_frame(sessions, np.arange(1.0, len(sessions) + 1.0))
    aligned = align_to_sessions(raw, "2021-06-01", "2021-06-30")
    notes = _coverage_notes("TEST", raw, aligned, "2021-06-01", "2021-06-30")
    assert any("later than the requested" in n for n in notes)


def test_daily_returns_preserve_gaps():
    series = pd.Series([100.0, np.nan, 102.0, 103.0])
    returns = daily_returns(series)
    assert np.isnan(returns.iloc[1])
    assert np.isnan(returns.iloc[2])
    assert returns.iloc[3] == pytest.approx(103.0 / 102.0 - 1.0)


def test_describe_adjustment_counts_real_sessions():
    sessions = trading_sessions("2021-06-01", "2021-06-30")
    adj = np.arange(1.0, len(sessions) + 1.0)
    adj[2] = np.nan
    price = make_price_data("TEST", sessions, adj)
    info = describe_adjustment(price)
    assert info["sessions_expected"] == len(sessions)
    assert info["sessions_with_adj_close"] == len(sessions) - 1
    assert info["adjusted_close_available"]


def test_cache_round_trip_records_provenance(tmp_path):
    cache = DiskCache(tmp_path)
    sessions = trading_sessions("2021-06-01", "2021-06-30")
    frame = _raw_frame(sessions, np.arange(1.0, len(sessions) + 1.0))
    provenance = Provenance(
        kind="prices", tickers=["TEST"], requested_start="2021-06-01",
        requested_end="2021-06-30", retrieved_at_utc=utc_now_iso(),
        data_source="test", adjustment_settings={"auto_adjust": False, "actions": True},
        package_versions=package_versions(), cache_key="TEST", cache_file="",
    )
    written = cache.write("prices", "TEST", frame, provenance)
    assert written.file_sha256
    assert written.row_count == len(frame)

    restored = cache.read("prices", "TEST")
    assert restored is not None
    frame_back, provenance_back = restored
    pd.testing.assert_frame_equal(frame_back, frame)
    assert provenance_back.adjustment_settings["auto_adjust"] is False
    assert provenance_back.file_sha256 == written.file_sha256
    assert provenance_back.package_versions["pandas"] == pd.__version__
    assert cache.age_seconds("prices", "TEST") is not None

    cache.delete("prices", "TEST")
    assert cache.read("prices", "TEST") is None


def test_cache_miss_returns_none(tmp_path):
    assert DiskCache(tmp_path).read("prices", "ABSENT") is None
    assert DiskCache(tmp_path).age_seconds("prices", "ABSENT") is None
