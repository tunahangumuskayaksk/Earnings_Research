"""Align the merged panel onto exchange trading sessions."""

from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

from src.prepare import _row_bar

US_EXCHANGES = {"XNYS", "XNGS", "XNAS", "XNMS", "BATS"}
IDENTITY_COLUMNS = (
    "TickerId",
    "ShortName",
    "InstrumentName",
    "Symbol",
    "Exchange",
    "CurrencyId",
)
_CALENDARS: dict[str, object] = {}


def _calendar_code(exchange: str) -> str:
    if exchange == "XTSE":
        return "XTSE"
    if exchange in US_EXCHANGES:
        return "XNYS"
    raise ValueError(f"No trading calendar for exchange {exchange!r}.")


def _calendar(code: str):
    if code not in _CALENDARS:
        _CALENDARS[code] = xcals.get_calendar(code)
    return _CALENDARS[code]


def _sessions(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    raw = _calendar(code).sessions_in_range(start, end)
    sessions = pd.DatetimeIndex(raw)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def _exchange(frame: pd.DataFrame) -> str:
    if "Exchange" in frame.columns and frame["Exchange"].notna().any():
        return str(frame["Exchange"].dropna().iloc[0])
    return str(frame["ShortName"].dropna().iloc[0]).split(":")[-1]


def _place_earnings(grid: pd.DataFrame, earnings: pd.DataFrame) -> None:
    sessions = pd.DatetimeIndex(grid.index)
    for row in earnings.itertuples(index=False):
        day = pd.Timestamp(row.EventDate).normalize()
        if day < sessions[0] or day > sessions[-1]:
            continue
        if day in grid.index and pd.notna(grid.at[day, "EventDate"]):
            continue
        if day in grid.index:
            target = day
            time_id = row.MarketTimeId
        else:
            position = sessions.searchsorted(day, side="left")
            if position >= len(sessions):
                continue
            target = sessions[position]
            time_id = 1.0
        if pd.notna(grid.at[target, "EventDate"]):
            continue
        grid.at[target, "EventDate"] = pd.Timestamp(row.EventDate)
        grid.at[target, "MarketTimeId"] = time_id
        if "Id" in grid.columns and hasattr(row, "Id"):
            grid.at[target, "Id"] = row.Id


def align_trading_sessions(panel: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """Reindex each stock onto exchange sessions; map closed-day announcements forward."""
    pieces: list[pd.DataFrame] = []
    earnings = earnings.copy()
    earnings["EventDate"] = pd.to_datetime(earnings["EventDate"])
    groups = list(panel.groupby("TickerId", sort=False))
    with _row_bar(len(panel)) as bar:
        for _ticker, frame in groups:
            frame = frame.copy()
            frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
            frame = frame.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
            code = _calendar_code(_exchange(frame))
            sessions = _sessions(code, frame.index.min(), frame.index.max())
            grid = frame.reindex(sessions)
            for column in IDENTITY_COLUMNS:
                if column in frame.columns and frame[column].notna().any():
                    grid[column] = frame[column].dropna().iloc[0]
            owned = earnings.loc[earnings["TickerId"] == _ticker]
            if len(owned):
                _place_earnings(grid, owned)
            grid.index.name = "Date"
            pieces.append(grid.reset_index())
            bar.update(len(frame))
    aligned = pd.concat(pieces, ignore_index=True)
    return aligned.reindex(columns=panel.columns)
