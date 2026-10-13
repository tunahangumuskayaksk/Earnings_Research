"""Event alignment and the per-event audit table.

t=0 is the actual release calendar date when that date is a trading session.
All offsets count trading sessions, not calendar days. After-close releases keep
their own calendar date in this paper-style analysis and are never shifted to the
next session; releases falling on non-trading dates are excluded and reported
rather than silently moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ALL_WINDOWS,
    ANNOUNCEMENT_WINDOW,
    LATER_WINDOW,
    PAR_WINDOW,
    REQUIRED_MAX_OFFSET,
    REQUIRED_MIN_OFFSET,
    SUBSEQUENT_WINDOW,
)
from .prices import PriceData
from .returns import evaluate_window, window_returns_vectorized
from .trading_calendar import is_session, session_position


@dataclass
class AlignedPanel:
    """Stock and benchmark adjusted closes on one shared session axis."""

    sessions: pd.DatetimeIndex
    stock_prices: np.ndarray
    benchmark_prices: np.ndarray
    stock_ticker: str
    benchmark_ticker: str

    @classmethod
    def from_price_data(cls, stock: PriceData, benchmark: PriceData) -> "AlignedPanel":
        sessions = stock.frame.index
        if not sessions.equals(benchmark.frame.index):
            sessions = stock.frame.index.union(benchmark.frame.index).sort_values()
        stock_prices = stock.frame["Adj Close"].reindex(sessions).to_numpy(dtype=float)
        bench_prices = benchmark.frame["Adj Close"].reindex(sessions).to_numpy(dtype=float)
        return cls(
            sessions=pd.DatetimeIndex(sessions, name="session"),
            stock_prices=stock_prices,
            benchmark_prices=bench_prices,
            stock_ticker=stock.ticker,
            benchmark_ticker=benchmark.ticker,
        )

    def position_of(self, day) -> int | None:
        return session_position(self.sessions, day)

    def window_matrix(self) -> dict[str, np.ndarray]:
        """Market-adjusted window returns at every session position."""
        matrix: dict[str, np.ndarray] = {}
        for window in ALL_WINDOWS:
            stock = window_returns_vectorized(self.stock_prices, window)
            bench = window_returns_vectorized(self.benchmark_prices, window)
            matrix[window.key] = stock - bench
        return matrix

    def required_span_available(self) -> np.ndarray:
        """Boolean per position: every price from -5 to +8 present for both series."""
        n = len(self.sessions)
        finite = np.isfinite(self.stock_prices) & np.isfinite(self.benchmark_prices)
        prefix = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
        lo = np.arange(n) + REQUIRED_MIN_OFFSET
        hi = np.arange(n) + REQUIRED_MAX_OFFSET
        span = REQUIRED_MAX_OFFSET - REQUIRED_MIN_OFFSET + 1
        ok = (lo >= 0) & (hi < n)
        lo_ok = np.where(ok, lo, 0)
        hi_ok = np.where(ok, hi, 0)
        return ok & ((prefix[hi_ok + 1] - prefix[lo_ok]) == span)


EVENT_COLUMNS = [
    "ticker", "release_date", "session_class", "release_time", "time_known",
    "source", "verified", "source_url",
    "par", "announcement_adj", "subsequent_adj", "later_adj",
    "par_stock", "par_benchmark",
    "announcement_stock", "announcement_benchmark",
    "subsequent_stock", "subsequent_benchmark",
    "later_stock", "later_benchmark",
    "par_first_return_date", "par_start_price_date", "par_end_price_date",
    "par_start_price", "par_end_price",
    "announcement_first_return_date", "announcement_start_price_date",
    "announcement_end_price_date", "announcement_start_price", "announcement_end_price",
    "subsequent_first_return_date", "subsequent_start_price_date",
    "subsequent_end_price_date", "subsequent_start_price", "subsequent_end_price",
    "later_first_return_date", "later_start_price_date", "later_end_price_date",
    "later_start_price", "later_end_price",
    "session_position", "quality_flags",
]


def build_event_table(panel: AlignedPanel,
                      events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(included_events, excluded_events)`` with a reason for each exclusion."""
    included: list[dict] = []
    excluded: list[dict] = []

    if events is None or len(events) == 0:
        return (pd.DataFrame(columns=EVENT_COLUMNS),
                pd.DataFrame(columns=["release_date", "reason", "detail"]))

    for _, event in events.iterrows():
        release_date = pd.Timestamp(event["release_date"]).normalize()

        if not is_session(release_date):
            excluded.append({
                "release_date": release_date,
                "reason": "Release date is not a trading session",
                "detail": (
                    "Paper-style alignment requires t=0 to be a trading session. "
                    "The event is excluded rather than shifted to a neighbouring day."
                ),
            })
            continue

        position = panel.position_of(release_date)
        if position is None:
            excluded.append({
                "release_date": release_date,
                "reason": "Release date outside the loaded price window",
                "detail": (
                    f"No session at {release_date.date()} in the loaded range "
                    f"{panel.sessions.min().date()}..{panel.sessions.max().date()}."
                ),
            })
            continue

        results = {w.key: evaluate_window(panel.sessions, panel.stock_prices,
                                          panel.benchmark_prices, position, w)
                   for w in ALL_WINDOWS}
        invalid = [r for r in results.values() if not r.valid]
        if invalid:
            excluded.append({
                "release_date": release_date,
                "reason": "Incomplete stock or benchmark data in a required window",
                "detail": " ".join(r.reason for r in invalid),
            })
            continue

        flags: list[str] = []
        if not bool(event.get("time_known", False)):
            flags.append("release time unknown (date-only record)")
        if not bool(event.get("verified", False)):
            flags.append("not user-verified")
        if str(event.get("conflict_detail", "") or ""):
            flags.append(str(event["conflict_detail"]))
        if str(event.get("override_note", "") or ""):
            flags.append(str(event["override_note"]))
        if event.get("session_class") == "after_close":
            flags.append("after-close release kept on its own calendar date")

        row: dict = {
            "ticker": event.get("ticker"),
            "release_date": release_date,
            "session_class": event.get("session_class"),
            "release_time": event.get("release_time"),
            "time_known": bool(event.get("time_known", False)),
            "source": event.get("source"),
            "verified": bool(event.get("verified", False)),
            "source_url": event.get("source_url"),
            "session_position": position,
            "quality_flags": "; ".join(flags) if flags else "",
        }
        name_map = {
            PAR_WINDOW.key: "par",
            ANNOUNCEMENT_WINDOW.key: "announcement",
            SUBSEQUENT_WINDOW.key: "subsequent",
            LATER_WINDOW.key: "later",
        }
        for key, prefix in name_map.items():
            r = results[key]
            adj_name = "par" if prefix == "par" else f"{prefix}_adj"
            row[adj_name] = r.market_adjusted_return
            row[f"{prefix}_stock"] = r.stock_return
            row[f"{prefix}_benchmark"] = r.benchmark_return
            row[f"{prefix}_first_return_date"] = r.first_return_date
            row[f"{prefix}_start_price_date"] = r.start_price_date
            row[f"{prefix}_end_price_date"] = r.end_price_date
            row[f"{prefix}_start_price"] = r.stock_start_price
            row[f"{prefix}_end_price"] = r.stock_end_price
        included.append(row)

    included_frame = pd.DataFrame(included, columns=EVENT_COLUMNS) if included \
        else pd.DataFrame(columns=EVENT_COLUMNS)
    excluded_frame = pd.DataFrame(excluded, columns=["release_date", "reason", "detail"]) \
        if excluded else pd.DataFrame(columns=["release_date", "reason", "detail"])
    return included_frame, excluded_frame
