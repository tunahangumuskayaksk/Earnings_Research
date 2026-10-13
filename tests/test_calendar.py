"""Trading-calendar handling, weekend/holiday offsets and event alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.config import ANNOUNCEMENT_WINDOW, PAR_WINDOW
from earnings_reversal.events import build_event_table
from earnings_reversal.prices import align_to_sessions
from earnings_reversal.returns import evaluate_window
from earnings_reversal.trading_calendar import (
    classify_release_session,
    is_session,
    session_position,
    trading_sessions,
)
from conftest import make_panel, make_price_data


def _events(rows):
    frame = pd.DataFrame(rows)
    frame["release_date"] = pd.to_datetime(frame["release_date"])
    return frame


def test_calendar_excludes_weekends_and_holidays():
    sessions = trading_sessions("2021-06-30", "2021-07-07")
    dates = [s.date().isoformat() for s in sessions]
    assert "2021-07-05" not in dates   # Independence Day observed
    assert "2021-07-03" not in dates   # Saturday
    assert "2021-07-04" not in dates   # Sunday
    assert dates == ["2021-06-30", "2021-07-01", "2021-07-02", "2021-07-06", "2021-07-07"]


def test_offsets_count_sessions_across_a_weekend():
    """A Monday event's t=-1 is the previous Friday, not the previous calendar day."""
    sessions = trading_sessions("2021-06-01", "2021-07-31")
    monday = pd.Timestamp("2021-06-21")
    position = session_position(sessions, monday)
    assert sessions[position - 1] == pd.Timestamp("2021-06-18")  # Friday
    assert sessions[position - 1] != monday - pd.Timedelta(days=1)


def test_offsets_count_sessions_across_a_holiday():
    sessions = trading_sessions("2021-06-01", "2021-07-31")
    position = session_position(sessions, pd.Timestamp("2021-07-06"))
    assert sessions[position - 1] == pd.Timestamp("2021-07-02")
    assert sessions[position - 2] == pd.Timestamp("2021-07-01")


def test_par_window_spans_real_sessions_around_a_holiday():
    sessions = trading_sessions("2021-06-01", "2021-07-31")
    position = session_position(sessions, pd.Timestamp("2021-07-08"))
    prices = np.arange(1.0, len(sessions) + 1.0)
    result = evaluate_window(sessions, prices, prices, position, PAR_WINDOW)
    assert [d.date().isoformat() for d in result.sessions] == [
        "2021-07-01", "2021-07-02", "2021-07-06",
    ]
    assert result.start_price_date == pd.Timestamp("2021-06-30")
    assert result.end_price_date == pd.Timestamp("2021-07-06")


def test_align_to_sessions_preserves_gaps_and_never_forward_fills():
    sessions = trading_sessions("2021-06-01", "2021-06-30")
    kept = sessions.drop([sessions[5]])
    frame = pd.DataFrame(
        {"Adj Close": np.arange(1.0, len(kept) + 1.0), "Close": np.arange(1.0, len(kept) + 1.0)},
        index=kept,
    )
    aligned = align_to_sessions(frame, "2021-06-01", "2021-06-30")
    assert aligned.index.equals(sessions)
    assert np.isnan(aligned.loc[sessions[5], "Adj Close"])
    assert aligned["Adj Close"].isna().sum() == 1
    # The value after the gap is unchanged, i.e. nothing was carried forward.
    assert aligned.loc[sessions[6], "Adj Close"] == 6.0


def test_missing_session_does_not_shift_later_offsets():
    """A hole in the data must stay a hole rather than pulling offsets forward."""
    sessions = trading_sessions("2021-06-01", "2021-07-31")
    prices = np.arange(1.0, len(sessions) + 1.0)
    gapped = prices.copy()
    gap_index = 20
    gapped[gap_index] = np.nan
    assert session_position(sessions, sessions[25]) == 25  # positions are unchanged
    result = evaluate_window(sessions, gapped, prices, gap_index + 1, ANNOUNCEMENT_WINDOW)
    assert not result.valid
    far = evaluate_window(sessions, gapped, prices, gap_index + 12, ANNOUNCEMENT_WINDOW)
    assert far.valid
    assert far.start_price_date == sessions[gap_index + 10]


def test_after_close_release_keeps_its_own_calendar_date():
    sessions = trading_sessions("2021-06-01", "2021-07-31")
    prices = 100.0 * np.power(1.001, np.arange(len(sessions)))
    panel = make_panel(sessions, prices, prices)
    release = pd.Timestamp("2021-06-21")
    events = _events([{
        "ticker": "TEST", "release_date": release, "release_time": "16:30:00",
        "time_known": True, "session_class": "after_close", "source": "test",
        "verified": True, "source_url": "", "conflict_detail": "", "override_note": "",
    }])
    included, excluded = build_event_table(panel, events)
    assert len(excluded) == 0
    assert included.loc[0, "release_date"] == release
    position = int(included.loc[0, "session_position"])
    assert sessions[position] == release          # t=0 is the release date itself
    assert sessions[position] != sessions[position + 1]
    assert "after-close release kept on its own calendar date" in included.loc[0, "quality_flags"]


def test_release_on_a_non_trading_date_is_excluded_not_moved():
    sessions = trading_sessions("2021-06-01", "2021-07-31")
    prices = np.arange(1.0, len(sessions) + 1.0)
    panel = make_panel(sessions, prices, prices)
    saturday = pd.Timestamp("2021-06-19")
    assert not is_session(saturday)
    events = _events([{
        "ticker": "TEST", "release_date": saturday, "release_time": None,
        "time_known": False, "session_class": "non_trading_day", "source": "test",
        "verified": False, "source_url": "", "conflict_detail": "", "override_note": "",
    }])
    included, excluded = build_event_table(panel, events)
    assert len(included) == 0
    assert len(excluded) == 1
    assert excluded.loc[0, "reason"] == "Release date is not a trading session"
    assert excluded.loc[0, "release_date"] == saturday


def test_event_with_incomplete_data_is_excluded_with_a_reason():
    sessions = trading_sessions("2021-06-01", "2021-07-31")
    prices = np.arange(1.0, len(sessions) + 1.0)
    broken = prices.copy()
    release = pd.Timestamp("2021-06-21")
    position = session_position(sessions, release)
    broken[position + 3] = np.nan
    panel = make_panel(sessions, broken, prices)
    events = _events([{
        "ticker": "TEST", "release_date": release, "release_time": "16:30:00",
        "time_known": True, "session_class": "after_close", "source": "test",
        "verified": True, "source_url": "", "conflict_detail": "", "override_note": "",
    }])
    included, excluded = build_event_table(panel, events)
    assert len(included) == 0
    assert excluded.loc[0, "reason"] == "Incomplete stock or benchmark data in a required window"
    assert "missing adjusted close for stock" in excluded.loc[0, "detail"]


@pytest.mark.parametrize("release_time,expected", [
    ("07:30:00", "before_open"),
    ("09:29:00", "before_open"),
    ("09:30:00", "during_session"),
    ("12:00:00", "during_session"),
    ("16:00:00", "after_close"),
    ("16:30:00", "after_close"),
    (None, "unknown"),
])
def test_session_classification(release_time, expected):
    from datetime import time as dtime

    day = pd.Timestamp("2021-06-21").date()
    parsed = None if release_time is None else dtime.fromisoformat(release_time)
    assert classify_release_session(day, parsed) == expected


def test_early_close_day_is_classified_against_the_real_close():
    """2021-11-26 is a 13:00 ET half day."""
    from datetime import time as dtime

    half_day = pd.Timestamp("2021-11-26").date()
    assert classify_release_session(half_day, dtime(13, 30)) == "after_close"
    assert classify_release_session(half_day, dtime(12, 0)) == "during_session"
    full_day = pd.Timestamp("2021-11-29").date()
    assert classify_release_session(full_day, dtime(13, 30)) == "during_session"


def test_panel_alignment_uses_a_shared_session_axis():
    sessions = trading_sessions("2021-06-01", "2021-06-30")
    from earnings_reversal.events import AlignedPanel

    stock = make_price_data("A", sessions, np.arange(1.0, len(sessions) + 1.0))
    bench = make_price_data("B", sessions, np.arange(2.0, len(sessions) + 2.0))
    panel = AlignedPanel.from_price_data(stock, bench)
    assert panel.sessions.equals(sessions)
    assert panel.stock_prices.shape == panel.benchmark_prices.shape
