"""U.S. equity trading sessions taken from an explicit exchange calendar.

The calendar is never inferred from the set of dates that happened to come back
in a price download: doing that would hide missing observations and silently
shift every event offset.
"""

from __future__ import annotations

from datetime import date, time
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from .config import TRADING_CALENDAR_NAME

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


@lru_cache(maxsize=8)
def get_calendar(name: str = TRADING_CALENDAR_NAME):
    return xcals.get_calendar(name)


@lru_cache(maxsize=32)
def _sessions_cached(name: str, start: str, end: str) -> tuple[pd.Timestamp, ...]:
    calendar = get_calendar(name)
    bound_start = max(pd.Timestamp(start), pd.Timestamp(calendar.first_session))
    bound_end = min(pd.Timestamp(end), pd.Timestamp(calendar.last_session))
    if bound_start > bound_end:
        return ()
    sessions = calendar.sessions_in_range(bound_start, bound_end)
    return tuple(pd.Timestamp(s).normalize().tz_localize(None) for s in sessions)


def trading_sessions(start, end, name: str = TRADING_CALENDAR_NAME) -> pd.DatetimeIndex:
    """Timezone-naive midnight timestamps for every session in ``[start, end]``."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    sessions = _sessions_cached(name, start.isoformat(), end.isoformat())
    return pd.DatetimeIndex(sessions, name="session")


def is_session(day, name: str = TRADING_CALENDAR_NAME) -> bool:
    day = pd.Timestamp(day).normalize()
    calendar = get_calendar(name)
    if day < pd.Timestamp(calendar.first_session) or day > pd.Timestamp(calendar.last_session):
        return False
    return bool(calendar.is_session(day))


def session_close_time(day, name: str = TRADING_CALENDAR_NAME) -> time:
    """Local New York closing time for a session, honouring early closes."""
    day = pd.Timestamp(day).normalize()
    if not is_session(day, name):
        return REGULAR_CLOSE
    calendar = get_calendar(name)
    try:
        close = calendar.session_close(day)
    except Exception:  # pragma: no cover - defensive
        return REGULAR_CLOSE
    return pd.Timestamp(close).tz_convert("America/New_York").time()


def session_open_time(day, name: str = TRADING_CALENDAR_NAME) -> time:
    day = pd.Timestamp(day).normalize()
    if not is_session(day, name):
        return REGULAR_OPEN
    calendar = get_calendar(name)
    try:
        opening = calendar.session_open(day)
    except Exception:  # pragma: no cover - defensive
        return REGULAR_OPEN
    return pd.Timestamp(opening).tz_convert("America/New_York").time()


def classify_release_session(release_date: date, release_time: time | None,
                             name: str = TRADING_CALENDAR_NAME) -> str:
    """Classify a release as before-open / during-session / after-close / unknown."""
    if release_time is None:
        return "unknown"
    if not is_session(release_date, name):
        return "non_trading_day"
    open_time = session_open_time(release_date, name)
    close_time = session_close_time(release_date, name)
    if release_time < open_time:
        return "before_open"
    if release_time < close_time:
        return "during_session"
    return "after_close"


def session_position(sessions: pd.DatetimeIndex, day) -> int | None:
    """Index of ``day`` within ``sessions``; ``None`` if it is not a session."""
    day = pd.Timestamp(day).normalize()
    position = sessions.searchsorted(day)
    if position >= len(sessions) or sessions[position] != day:
        return None
    return int(position)
