"""Attach a market benchmark series to a daily panel.

Sources:
- ``yahoo``: cached Yahoo adjusted close for ``MARKET_TICKER`` (default SPY)
- ``equal``: equal-weighted index from the panel's ``LastPrice``
- ``cap``: same-day ``MarketCap``-weighted index from the panel
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.prepare import DATA_DIR, _row_bar

BENCHMARK_COLUMN = "BenchmarkAdjClose"
BenchmarkSource = Literal["yahoo", "equal", "cap"]
VALID_BENCHMARK_SOURCES: frozenset[str] = frozenset(("yahoo", "equal", "cap"))
PRICE_COLUMN = "LastPrice"
CAP_COLUMN = "MarketCap"
INDEX_BASE = 100.0


def normalize_benchmark_source(source: str) -> BenchmarkSource:
    key = str(source).strip().lower()
    if key not in VALID_BENCHMARK_SOURCES:
        allowed = ", ".join(sorted(VALID_BENCHMARK_SOURCES))
        raise ValueError(f"BENCHMARK_SOURCE must be one of {{{allowed}}}, got {source!r}.")
    return key  # type: ignore[return-value]


def benchmark_paths(
    folder: Path,
    ticker: str = "SPY",
    *,
    source: str = "yahoo",
) -> tuple[Path, Path]:
    method = normalize_benchmark_source(source)
    if method == "yahoo":
        stem = f"benchmark_{ticker.strip().lower()}"
    else:
        stem = f"benchmark_{method}"
    return folder / f"{stem}.parquet", folder / f"{stem}.json"


def _download_benchmark(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No prices returned for {ticker}.")
    close = raw["Adj Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    dates = pd.to_datetime(close.index)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    frame = pd.DataFrame({
        "Date": dates.normalize(),
        "AdjClose": pd.to_numeric(close, errors="coerce").to_numpy(),
    })
    frame = frame.dropna(subset=["AdjClose"]).drop_duplicates("Date", keep="last")
    return frame.sort_values("Date").reset_index(drop=True)


def load_benchmark(folder: Path, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path, meta_path = benchmark_paths(folder, ticker, source="yahoo")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cached = pd.read_parquet(path)
        cached["Date"] = pd.to_datetime(cached["Date"])
        covers = (
            meta.get("ticker") == ticker
            and cached["Date"].min() <= start
            and cached["Date"].max() >= end
        )
        if covers:
            with _row_bar(len(cached)) as bar:
                bar.update(len(cached))
            return cached
    frame = _download_benchmark(ticker, start, end)
    if frame["Date"].min() > start or frame["Date"].max() < end:
        raise RuntimeError(
            f"{ticker} prices cover {frame['Date'].min():%Y-%m-%d} to "
            f"{frame['Date'].max():%Y-%m-%d}, short of {start:%Y-%m-%d} to {end:%Y-%m-%d}."
        )
    _write_benchmark_cache(
        folder,
        frame,
        source="yahoo",
        ticker=ticker,
        extra={"field": "Adj Close"},
    )
    with _row_bar(len(frame)) as bar:
        bar.update(len(frame))
    return frame


def build_universe_index(
    panel: pd.DataFrame,
    method: Literal["equal", "cap"],
) -> pd.DataFrame:
    """Build a daily close index from panel prices (and market caps for ``cap``).

    Membership each day: names with a valid close-to-close return that day
    (``LastPrice`` on t and t-1). For ``cap``, also require same-day
    ``MarketCap`` > 0. Weights are renormalized over the day's members.
    """
    method = normalize_benchmark_source(method)
    if method not in ("equal", "cap"):
        raise ValueError(f"Universe index method must be 'equal' or 'cap', got {method!r}.")

    required = ["Date", "TickerId", PRICE_COLUMN]
    if method == "cap":
        required.append(CAP_COLUMN)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(
            f"Panel missing columns for {method} benchmark: {missing}. "
            f"Need {required}."
        )

    cols = ["Date", "TickerId", PRICE_COLUMN] + ([CAP_COLUMN] if method == "cap" else [])
    work = panel.loc[:, cols].copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.normalize()
    work[PRICE_COLUMN] = pd.to_numeric(work[PRICE_COLUMN], errors="coerce")
    if method == "cap":
        work[CAP_COLUMN] = pd.to_numeric(work[CAP_COLUMN], errors="coerce")

    work = work.dropna(subset=["Date", "TickerId", PRICE_COLUMN])
    work = work.loc[work[PRICE_COLUMN] > 0]
    if method == "cap":
        work = work.loc[work[CAP_COLUMN].notna() & (work[CAP_COLUMN] > 0)]
    work = (
        work.sort_values(["TickerId", "Date"])
        .drop_duplicates(["TickerId", "Date"], keep="last")
        .reset_index(drop=True)
    )
    if work.empty:
        raise RuntimeError(f"No usable rows to build {method} benchmark.")

    work["ret"] = work.groupby("TickerId", sort=False)[PRICE_COLUMN].pct_change()
    rets = work.dropna(subset=["ret"])
    rets = rets.loc[np.isfinite(rets["ret"].to_numpy(dtype=float))]
    if rets.empty:
        raise RuntimeError(f"No daily returns available for {method} benchmark.")

    if method == "equal":
        daily = rets.groupby("Date", sort=True)["ret"].mean()
    else:
        weighted = rets["ret"] * rets[CAP_COLUMN]
        daily = weighted.groupby(rets["Date"], sort=True).sum() / rets.groupby(
            "Date", sort=True
        )[CAP_COLUMN].sum()

    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    if daily.empty:
        raise RuntimeError(f"No finite daily portfolio returns for {method} benchmark.")

    levels = INDEX_BASE * (1.0 + daily.to_numpy(dtype=float)).cumprod()
    frame = pd.DataFrame({"Date": pd.to_datetime(daily.index).normalize(), "AdjClose": levels})
    return frame.reset_index(drop=True)


def _write_benchmark_cache(
    folder: Path,
    frame: pd.DataFrame,
    *,
    source: str,
    ticker: str | None = None,
    extra: dict | None = None,
) -> tuple[Path, Path]:
    method = normalize_benchmark_source(source)
    path, meta_path = benchmark_paths(folder, ticker or method, source=method)
    folder.mkdir(parents=True, exist_ok=True)
    out = frame.loc[:, ["Date", "AdjClose"]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out.to_parquet(path, index=False)
    meta: dict = {
        "source": method,
        "field": "AdjClose",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": out["Date"].min().strftime("%Y-%m-%d"),
        "end": out["Date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(out)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if method == "yahoo":
        meta["ticker"] = ticker
    else:
        meta["method"] = method
        meta["price_field"] = PRICE_COLUMN
        if method == "cap":
            meta["weight_field"] = CAP_COLUMN
            meta["weight_timing"] = "same_day"
    if extra:
        meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2))
    return path, meta_path


def attach_benchmark(
    frame: pd.DataFrame,
    ticker: str = "SPY",
    data_dir: Path | None = None,
    column: str = BENCHMARK_COLUMN,
    source: str = "yahoo",
) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    method = normalize_benchmark_source(source)
    dates = pd.to_datetime(frame["Date"]).dt.normalize()

    if method == "yahoo":
        prices = load_benchmark(folder, ticker, dates.min(), dates.max())
    else:
        prices = build_universe_index(frame, method)
        _write_benchmark_cache(folder, prices, source=method)
        with _row_bar(len(prices)) as bar:
            bar.update(len(prices))

    lookup = prices.set_index(pd.to_datetime(prices["Date"]).dt.normalize())["AdjClose"]
    out = frame.copy()
    out[column] = dates.map(lookup).to_numpy()
    return out
