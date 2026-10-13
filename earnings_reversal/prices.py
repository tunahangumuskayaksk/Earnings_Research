"""Daily price retrieval, caching and session alignment.

Prices are requested with ``auto_adjust=False`` and ``actions=True`` so that the
raw OHLC, the separate adjusted close, volume, dividends and splits are all
preserved. Adjusted close is used for every return calculation and is never
silently replaced by the unadjusted close.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .cache import DiskCache, Provenance, package_versions, utc_now_iso
from .config import (
    DATA_SOURCE,
    MAX_DOWNLOAD_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
    TRADING_CALENDAR_NAME,
)
from .trading_calendar import trading_sessions

PRICE_COLUMNS = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "Dividends", "Stock Splits",
]

ADJUSTMENT_SETTINGS = {
    "interval": "1d",
    "auto_adjust": False,
    "actions": True,
    "back_adjust": False,
    "repair": False,
    "return_basis": "Adj Close (Yahoo split/dividend adjusted)",
    "double_adjustment_guard": "auto_adjust=False so Close stays raw; "
                              "only 'Adj Close' is used for returns",
}


class PriceDownloadError(RuntimeError):
    """Raised when price data could not be retrieved and no cache is usable."""


@dataclass
class PriceData:
    """Session-aligned adjusted prices for one ticker plus its provenance."""

    ticker: str
    frame: pd.DataFrame              # indexed by trading session, NaN preserved
    raw_frame: pd.DataFrame          # exactly what the source returned
    provenance: Provenance
    coverage_notes: list[str] = field(default_factory=list)

    @property
    def adj_close(self) -> pd.Series:
        return self.frame["Adj Close"]

    @property
    def actual_start(self) -> pd.Timestamp | None:
        valid = self.raw_frame.index
        return pd.Timestamp(valid.min()) if len(valid) else None

    @property
    def actual_end(self) -> pd.Timestamp | None:
        valid = self.raw_frame.index
        return pd.Timestamp(valid.max()) if len(valid) else None

    @property
    def missing_sessions(self) -> pd.DatetimeIndex:
        return self.frame.index[self.frame["Adj Close"].isna()]


def _normalise_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's column layout and keep the documented fields."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        levels = frame.columns.get_level_values(0)
        if ticker in set(levels):
            frame = frame.xs(ticker, axis=1, level=0)
        else:
            levels_1 = frame.columns.get_level_values(1)
            if ticker in set(levels_1):
                frame = frame.xs(ticker, axis=1, level=1)
            else:
                frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.DatetimeIndex(
        [pd.Timestamp(idx).tz_localize(None).normalize() if pd.Timestamp(idx).tzinfo
         else pd.Timestamp(idx).normalize() for idx in frame.index],
        name="session",
    )
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    for column in PRICE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[PRICE_COLUMNS]


def _download_with_retries(ticker: str, start: str, end: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if raw is None or raw.empty:
                raise PriceDownloadError(
                    f"{ticker}: the source returned no rows for {start}..{end}."
                )
            return raw
        except Exception as exc:  # network / parsing / empty response
            last_error = exc
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time_module.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise PriceDownloadError(
        f"{ticker}: download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def align_to_sessions(frame: pd.DataFrame, start, end,
                      calendar_name: str = TRADING_CALENDAR_NAME) -> pd.DataFrame:
    """Reindex to the expected sessions, keeping gaps as NaN (never forward-filled)."""
    sessions = trading_sessions(start, end, calendar_name)
    aligned = frame.reindex(sessions)
    aligned.index.name = "session"
    return aligned


def _coverage_notes(ticker: str, raw: pd.DataFrame, aligned: pd.DataFrame,
                    requested_start: str, requested_end: str) -> list[str]:
    notes: list[str] = []
    if raw.empty:
        notes.append(f"{ticker}: no rows returned at all.")
        return notes
    first, last = raw.index.min().date(), raw.index.max().date()
    notes.append(f"{ticker}: source returned {len(raw)} rows covering {first} to {last}.")
    if pd.Timestamp(first) > pd.Timestamp(requested_start):
        notes.append(
            f"{ticker}: history starts {first}, later than the requested {requested_start}."
        )
    if pd.Timestamp(last) < pd.Timestamp(requested_end) - pd.Timedelta(days=5):
        notes.append(
            f"{ticker}: history ends {last}, earlier than the requested {requested_end}."
        )
    missing = aligned.index[aligned["Adj Close"].isna()]
    inside = missing[(missing >= raw.index.min()) & (missing <= raw.index.max())]
    if len(inside):
        shown = ", ".join(d.date().isoformat() for d in inside[:8])
        suffix = "" if len(inside) <= 8 else f" (+{len(inside) - 8} more)"
        notes.append(
            f"{ticker}: {len(inside)} expected trading session(s) inside the covered "
            f"range have no adjusted close and are left missing: {shown}{suffix}."
        )
    if aligned["Adj Close"].notna().sum() == 0:
        notes.append(f"{ticker}: adjusted close is entirely absent.")
    elif aligned["Close"].notna().sum() > aligned["Adj Close"].notna().sum():
        notes.append(
            f"{ticker}: adjusted close is missing on some sessions where the raw "
            f"close exists. Unadjusted close is NOT substituted."
        )
    return notes


def load_prices(ticker: str, start: str, end: str, cache: DiskCache,
                force_refresh: bool = False,
                calendar_name: str = TRADING_CALENDAR_NAME) -> PriceData:
    """Load prices from cache when possible, otherwise download and cache them."""
    ticker = ticker.strip().upper()
    key = f"{ticker}__{start}__{end}"
    cached = None if force_refresh else cache.read("prices", key)

    download_error: Exception | None = None
    if cached is None:
        try:
            raw = _download_with_retries(ticker, start, end)
            normalised = _normalise_download(raw, ticker)
            provenance = Provenance(
                kind="prices",
                tickers=[ticker],
                requested_start=start,
                requested_end=end,
                retrieved_at_utc=utc_now_iso(),
                data_source=DATA_SOURCE,
                adjustment_settings=dict(ADJUSTMENT_SETTINGS),
                package_versions=package_versions(),
                cache_key=key,
                cache_file="",
                actual_start=str(normalised.index.min().date()) if len(normalised) else None,
                actual_end=str(normalised.index.max().date()) if len(normalised) else None,
            )
            provenance = cache.write("prices", key, normalised, provenance)
            cached = (normalised, provenance)
        except Exception as exc:
            download_error = exc
            fallback = cache.read("prices", key)
            if fallback is None:
                raise PriceDownloadError(str(exc)) from exc
            cached = fallback

    normalised, provenance = cached
    if download_error is not None:
        provenance.notes = list(provenance.notes) + [
            f"Download failed ({type(download_error).__name__}: {download_error}). "
            f"Serving the existing cache retrieved at {provenance.retrieved_at_utc}."
        ]

    aligned = align_to_sessions(normalised, start, end, calendar_name)
    notes = _coverage_notes(ticker, normalised, aligned, start, end)
    if download_error is not None:
        notes.insert(0, f"{ticker}: live refresh failed, using cached data. {download_error}")
    return PriceData(
        ticker=ticker,
        frame=aligned,
        raw_frame=normalised,
        provenance=provenance,
        coverage_notes=notes,
    )


def daily_returns(adj_close: pd.Series) -> pd.Series:
    """r_d = P_d / P_{d-1} - 1 with missing observations preserved as NaN."""
    values = adj_close.astype(float)
    return values / values.shift(1) - 1.0


def describe_adjustment(price: PriceData) -> dict[str, Any]:
    splits = price.frame.get("Stock Splits")
    dividends = price.frame.get("Dividends")
    return {
        "ticker": price.ticker,
        "adjusted_close_available": bool(price.frame["Adj Close"].notna().any()),
        "sessions_expected": int(len(price.frame)),
        "sessions_with_adj_close": int(price.frame["Adj Close"].notna().sum()),
        "splits_in_range": int((splits.fillna(0) != 0).sum()) if splits is not None else 0,
        "dividends_in_range": int((dividends.fillna(0) != 0).sum()) if dividends is not None else 0,
    }
