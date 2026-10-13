"""Central configuration for the earnings reversal research pilot.

Window definitions and pseudo-event offsets are intentionally fixed in this
first version so that the study does not turn into a parameter search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"

TRADING_CALENDAR_NAME = "XNYS"
EARNINGS_TIMEZONE = "America/New_York"
DATA_SOURCE = "Yahoo Finance via yfinance"

DEFAULT_TICKER = "AAPL"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_EVENT_START = "2010-01-01"
DEFAULT_EVENT_END = "2025-12-31"
DEFAULT_SEED = 42
DEFAULT_REPETITIONS = 500

# Yahoo's earnings-date endpoint is paged by a `limit` argument. Empirically it
# refuses to return more than ~100 rows, but we still ask for more so that any
# future server-side increase is picked up automatically.
EARNINGS_REQUEST_LIMIT = 400

MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.5

# Extra calendar padding around the requested event range so that every actual
# and pseudo window has its prices available.
PRICE_PAD_DAYS_BEFORE = 220
PRICE_PAD_DAYS_AFTER = 90


@dataclass(frozen=True)
class Window:
    """A return window measured in trading sessions relative to the event."""

    key: str
    label: str
    start: int
    end: int

    @property
    def baseline_offset(self) -> int:
        """Session whose close is the starting price for this window."""
        return self.start - 1

    def describe(self) -> str:
        return (
            f"{self.label}: sessions [{self.start:+d},{self.end:+d}] "
            f"(start price = close at {self.baseline_offset:+d}, "
            f"end price = close at {self.end:+d})"
        )


PAR_WINDOW = Window("par", "PAR", -4, -2)
ANNOUNCEMENT_WINDOW = Window("announcement", "Announcement return", -1, 1)
SUBSEQUENT_WINDOW = Window("subsequent", "Subsequent return", 2, 5)
LATER_WINDOW = Window("later", "Later return", 6, 8)

OUTCOME_WINDOWS = (ANNOUNCEMENT_WINDOW, SUBSEQUENT_WINDOW, LATER_WINDOW)
ALL_WINDOWS = (PAR_WINDOW,) + OUTCOME_WINDOWS

# Full span of sessions that must be available for an event to be analysable.
REQUIRED_MIN_OFFSET = PAR_WINDOW.baseline_offset  # -5 (baseline close)
REQUIRED_MAX_OFFSET = LATER_WINDOW.end  # +8

# Pseudo-event placement: 10 to 40 trading sessions before the actual event.
PSEUDO_MIN_OFFSET = 10
PSEUDO_MAX_OFFSET = 40

# How the -5 baseline close is treated when screening a pseudo window for
# earnings contamination. "after_baseline" screens every calendar date strictly
# after the baseline close date through the +8 session date, which covers every
# date that can influence a return in sessions [-4,+8].
CONTAMINATION_BOUNDARY = "after_baseline"

ANNOUNCEMENT_WINDOW_CAVEAT = (
    "The announcement window includes the trading day before, the release date, "
    "and the trading day after. It is not a pure post-release return."
)

PSEUDO_PERCENTILE_CAVEAT = (
    "Variation across pseudo-date draws\u2014not a confidence interval."
)


@dataclass(frozen=True)
class AnalysisConfig:
    """User-adjustable settings coming from the Streamlit sidebar."""

    ticker: str = DEFAULT_TICKER
    benchmark: str = DEFAULT_BENCHMARK
    event_start: str = DEFAULT_EVENT_START
    event_end: str = DEFAULT_EVENT_END
    seed: int = DEFAULT_SEED
    n_repetitions: int = DEFAULT_REPETITIONS
    excluded_release_dates: tuple[str, ...] = field(default=())

    def normalised(self) -> "AnalysisConfig":
        return AnalysisConfig(
            ticker=self.ticker.strip().upper(),
            benchmark=self.benchmark.strip().upper(),
            event_start=self.event_start,
            event_end=self.event_end,
            seed=int(self.seed),
            n_repetitions=int(self.n_repetitions),
            excluded_release_dates=tuple(sorted(self.excluded_release_dates)),
        )
