"""Window return mathematics.

For a window of trading sessions ``[a, b]`` measured relative to an event:

    daily return      r_d  = P_d / P_{d-1} - 1
    window return          = prod(1 + r_d) for d = a..b, minus 1
    market-adjusted        = stock window return - benchmark window return

The starting price is the close at session ``a-1`` and the ending price is the
close at session ``b``. The market adjustment is a difference between two
compounded returns; daily stock-minus-benchmark differences are never compounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Window


@dataclass
class WindowResult:
    """A single window return with everything needed to audit it by hand."""

    window: Window
    stock_return: float = float("nan")
    benchmark_return: float = float("nan")
    market_adjusted_return: float = float("nan")
    start_price_date: pd.Timestamp | None = None
    end_price_date: pd.Timestamp | None = None
    first_return_date: pd.Timestamp | None = None
    stock_start_price: float = float("nan")
    stock_end_price: float = float("nan")
    benchmark_start_price: float = float("nan")
    benchmark_end_price: float = float("nan")
    sessions: list[pd.Timestamp] = field(default_factory=list)
    valid: bool = False
    reason: str = ""


def compound_daily_returns(daily: np.ndarray | list[float]) -> float:
    """prod(1 + r) - 1, returning NaN if any input is missing."""
    values = np.asarray(daily, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.prod(1.0 + values) - 1.0)


def window_daily_returns(prices: np.ndarray, position: int, window: Window) -> np.ndarray:
    """The individual daily returns r_d for d = a..b around ``position``."""
    prices = np.asarray(prices, dtype=float)
    lo = position + window.baseline_offset
    hi = position + window.end
    if lo < 0 or hi >= len(prices):
        return np.array([])
    segment = prices[lo:hi + 1]
    if segment.size < 2:
        return np.array([])
    with np.errstate(invalid="ignore", divide="ignore"):
        return segment[1:] / segment[:-1] - 1.0


def window_return(prices: np.ndarray, position: int, window: Window) -> float:
    """Compounded window return, NaN unless every required price is present."""
    daily = window_daily_returns(prices, position, window)
    expected = window.end - window.start + 1
    if daily.size != expected:
        return float("nan")
    return compound_daily_returns(daily)


def window_returns_vectorized(prices: np.ndarray, window: Window) -> np.ndarray:
    """``window_return`` evaluated at every position, computed in one pass.

    Equivalent to compounding the daily returns: because the product telescopes,
    the value is ``P_b / P_{a-1} - 1`` whenever every price from ``a-1`` through
    ``b`` is present. The presence requirement is enforced explicitly so that a
    gap inside the window yields NaN instead of silently bridging it.
    """
    prices = np.asarray(prices, dtype=float)
    n = prices.size
    result = np.full(n, np.nan)
    if n == 0:
        return result

    finite = np.isfinite(prices) & (prices != 0.0)
    prefix = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])

    lo = np.arange(n) + window.baseline_offset
    hi = np.arange(n) + window.end
    span = window.end - window.baseline_offset + 1
    in_bounds = (lo >= 0) & (hi < n)
    if not np.any(in_bounds):
        return result

    lo_ok = np.where(in_bounds, lo, 0)
    hi_ok = np.where(in_bounds, hi, 0)
    complete = (prefix[hi_ok + 1] - prefix[lo_ok]) == span
    usable = in_bounds & complete
    with np.errstate(invalid="ignore", divide="ignore"):
        values = prices[hi_ok] / prices[lo_ok] - 1.0
    result[usable] = values[usable]
    return result


def evaluate_window(sessions: pd.DatetimeIndex, stock_prices: np.ndarray,
                    benchmark_prices: np.ndarray, position: int,
                    window: Window) -> WindowResult:
    """Full audit record for one window around one event position."""
    result = WindowResult(window=window)
    lo = position + window.baseline_offset
    hi = position + window.end
    n = len(sessions)
    if lo < 0 or hi >= n:
        result.reason = (
            f"{window.label}: required sessions {window.baseline_offset:+d}..{window.end:+d} "
            f"fall outside the loaded price history."
        )
        return result

    result.start_price_date = sessions[lo]
    result.end_price_date = sessions[hi]
    result.first_return_date = sessions[position + window.start]
    result.sessions = [sessions[i] for i in range(position + window.start, hi + 1)]
    result.stock_start_price = float(stock_prices[lo])
    result.stock_end_price = float(stock_prices[hi])
    result.benchmark_start_price = float(benchmark_prices[lo])
    result.benchmark_end_price = float(benchmark_prices[hi])

    stock = window_return(stock_prices, position, window)
    benchmark = window_return(benchmark_prices, position, window)
    result.stock_return = stock
    result.benchmark_return = benchmark
    if np.isnan(stock) or np.isnan(benchmark):
        missing = []
        if np.isnan(stock):
            missing.append("stock")
        if np.isnan(benchmark):
            missing.append("benchmark")
        result.reason = (
            f"{window.label}: missing adjusted close for {' and '.join(missing)} "
            f"between {sessions[lo].date()} and {sessions[hi].date()}."
        )
        return result

    result.market_adjusted_return = stock - benchmark
    result.valid = True
    return result


def daily_return_frame(sessions: pd.DatetimeIndex, stock_prices: np.ndarray,
                       benchmark_prices: np.ndarray, position: int,
                       lo_offset: int, hi_offset: int) -> pd.DataFrame:
    """Per-session detail table used by the Event Inspector."""
    n = len(sessions)
    lo = max(position + lo_offset, 0)
    hi = min(position + hi_offset, n - 1)
    rows = []
    for i in range(lo, hi + 1):
        prev_stock = stock_prices[i - 1] if i > 0 else np.nan
        prev_bench = benchmark_prices[i - 1] if i > 0 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            stock_ret = stock_prices[i] / prev_stock - 1.0 if np.isfinite(prev_stock) else np.nan
            bench_ret = benchmark_prices[i] / prev_bench - 1.0 if np.isfinite(prev_bench) else np.nan
        rows.append({
            "offset": i - position,
            "session": sessions[i],
            "stock_adj_close": stock_prices[i],
            "benchmark_adj_close": benchmark_prices[i],
            "stock_daily_return": stock_ret,
            "benchmark_daily_return": bench_ret,
            "daily_difference": stock_ret - bench_ret,
        })
    return pd.DataFrame(rows)
