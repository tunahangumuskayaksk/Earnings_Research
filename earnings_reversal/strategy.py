"""PAR quintile strategy: a per-stock historical adaptation.

For each event the 20th and 80th percentiles of PAR are estimated from the
stock's own *earlier* included earnings events only (at least
``MIN_PRIOR_EVENTS`` of them; the current event is never part of its own
threshold set). PAR below the 20th percentile (Q1) means long, PAR above the 80th
percentile (Q5) means short, anything else means no trade. Quintile membership
alone decides the direction, whatever the sign of the threshold.

This is not the original study's cross-sectional quintile sort, which ranks many
firms against each other in the same quarter. Here one stock is ranked against
its own history.

Timing. PAR covers sessions [-4,-2], so it is complete at the close of t=-2. The
announcement window [-1,+1] starts from that same close. Trades are therefore
assumed to be entered at the close that completes the signal (a same-close
execution assumption) and exited at the close of t=+1. Each earlier event used
in a threshold sits at a strictly earlier session, so its PAR was complete at an
earlier close than the current entry. All returns are before costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ANNOUNCEMENT_WINDOW, PAR_WINDOW
from .returns import window_returns_vectorized

LOW_PERCENTILE = 20
HIGH_PERCENTILE = 80
MIN_PRIOR_EVENTS = 20
DEFAULT_NOTIONAL = 10_000.0
STRATEGY_WINDOW = ANNOUNCEMENT_WINDOW

QUINTILE = "quintile"
SIGN_SAME = "sign_same"
SIGN_ALL = "sign_all"
STRATEGY_LABELS = {
    QUINTILE: "PAR quintile (Q1 long, Q5 short)",
    SIGN_SAME: "PAR sign (negative long, positive short), same eligible events",
    SIGN_ALL: "PAR sign (negative long, positive short), all events",
}

POSITION_LABELS = {1: "Long", -1: "Short", 0: "No trade"}
INSUFFICIENT = "Insufficient history"

ADAPTATION_NOTE = (
    "Per-stock historical adaptation: each stock's PAR is ranked against that "
    "stock's own earlier earnings events. This is not a replication of the "
    "original study's cross-sectional quintile sort across many firms."
)
SAME_CLOSE_NOTE = (
    "Same-close execution assumption: PAR is complete at the close of t=\u22122 and "
    "the trade is assumed to be entered at that same close, which is the start "
    "price of the announcement window, then exited at the close of t=+1. In "
    "practice the close is not known until the closing auction prints, so this "
    "assumes a market-on-close order sized from a near-close PAR estimate. It also "
    "assumes the release date was known by t=\u22122. Entering at the next open is "
    "not modelled."
)
COSTS_NOTE = (
    "All results are before transaction costs, bid-ask spreads, borrow fees, "
    "financing and market impact."
)
SPREAD_NOTE = (
    "Q1 mean minus Q5 mean is a group spread between two sets of events at "
    "different dates. It is not the average return per strategy trade, and it is "
    "not a simultaneously held long-short portfolio return."
)
RETURN_BASIS_NOTE = (
    "Raw stock strategy return: the position in the stock alone (a long earns the "
    "stock return, a short earns minus the stock return). Signed market-adjusted "
    "return: position \u00d7 (stock return \u2212 benchmark return), i.e. the stock "
    "position paired with an equal and opposite benchmark position."
)


def threshold_lookup(pool_par: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """20th/80th percentiles of the first ``k`` pool values, for k = 0..n.

    ``pool_par`` must be in chronological order. Entries with fewer than
    ``MIN_PRIOR_EVENTS`` values are NaN. Percentiles use numpy's default linear
    interpolation.
    """
    n = len(pool_par)
    low = np.full(n + 1, np.nan)
    high = np.full(n + 1, np.nan)
    for k in range(MIN_PRIOR_EVENTS, n + 1):
        low[k], high[k] = np.percentile(pool_par[:k], [LOW_PERCENTILE, HIGH_PERCENTILE])
    return low, high


def quintile_direction(par: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """+1 below the low threshold, -1 above the high threshold, 0 otherwise."""
    with np.errstate(invalid="ignore"):
        long_ = par < low
        short = par > high
    return np.where(long_, 1, np.where(short, -1, 0)).astype(int)


def sign_direction(par: np.ndarray) -> np.ndarray:
    """The existing negative/positive split: long after negative PAR, short after positive."""
    return (-np.sign(par)).astype(int)


def performance(direction: np.ndarray, raw: np.ndarray, adjusted: np.ndarray) -> dict:
    """Per-trade statistics for executed trades (direction != 0).

    Fixed-notional P&L is the plain sum of signed returns per unit of notional:
    every trade uses the same notional and nothing is compounded. A win is a
    strictly positive signed return.
    """
    traded = direction != 0
    signed_raw = direction[traded] * raw[traded]
    signed_adj = direction[traded] * adjusted[traded]
    n = int(traded.sum())
    return {
        "trades": n,
        "longs": int((direction == 1).sum()),
        "shorts": int((direction == -1).sum()),
        "mean_raw": float(signed_raw.mean()) if n else None,
        "mean_adj": float(signed_adj.mean()) if n else None,
        "win_raw": float((signed_raw > 0).mean()) if n else None,
        "win_adj": float((signed_adj > 0).mean()) if n else None,
        "pnl_raw": float(signed_raw.sum()),
        "pnl_adj": float(signed_adj.sum()),
    }


def group_spread(direction: np.ndarray, outcome: np.ndarray) -> float | None:
    """Mean outcome of long-group events minus mean outcome of short-group events."""
    longs = outcome[direction == 1]
    shorts = outcome[direction == -1]
    if longs.size == 0 or shorts.size == 0:
        return None
    return float(longs.mean() - shorts.mean())


PERFORMANCE_KEYS = ("trades", "longs", "shorts", "mean_raw", "mean_adj", "win_raw",
                    "win_adj", "pnl_raw", "pnl_adj")
SPREAD_KEYS = ("spread_raw", "spread_adj")


@dataclass
class QuintileStrategy:
    """Actual trades, actual performance and the pseudo-event comparison."""

    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance: dict[str, dict] = field(default_factory=dict)
    matched_performance: dict[str, dict] = field(default_factory=dict)
    pseudo_reps: dict[str, pd.DataFrame] = field(default_factory=dict)
    eligible_events: int = 0
    matched_events: int = 0
    matched_eligible_events: int = 0
    pseudo_eligible: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    availability_violations: int = 0
    notes: list[str] = field(default_factory=list)


def _with_spreads(direction, raw, adjusted) -> dict:
    stats = performance(direction, raw, adjusted)
    stats["spread_raw"] = group_spread(direction, raw)
    stats["spread_adj"] = group_spread(direction, adjusted)
    return stats


def _actual_trades(events: pd.DataFrame, low: np.ndarray, high: np.ndarray,
                   pool_positions: np.ndarray) -> pd.DataFrame:
    par = events["par"].to_numpy(dtype=float)
    positions = events["session_position"].to_numpy(dtype=int)
    prior = np.searchsorted(pool_positions, positions, side="left")
    eligible = prior >= MIN_PRIOR_EVENTS
    lo, hi = low[prior], high[prior]
    direction = np.where(eligible, quintile_direction(par, lo, hi), 0)

    pool_signal_dates = pd.DatetimeIndex(events["par_end_price_date"])
    latest_input = [pool_signal_dates[k - 1] if k > 0 else pd.NaT for k in prior]

    quintile = np.where(par < lo, "Q1", np.where(par > hi, "Q5", "Q2\u2013Q4"))
    trades = pd.DataFrame({
        "release_date": events["release_date"].to_numpy(),
        "signal_and_entry_close": pd.DatetimeIndex(events["par_end_price_date"]),
        "exit_close": pd.DatetimeIndex(events["announcement_end_price_date"]),
        "par": par,
        "prior_events": prior,
        "threshold_p20": lo,
        "threshold_p80": hi,
        "latest_threshold_input": pd.DatetimeIndex(latest_input),
        "eligible": eligible,
        "quintile": np.where(eligible, quintile, INSUFFICIENT),
        "position": [POSITION_LABELS[d] if ok else INSUFFICIENT
                     for d, ok in zip(direction, eligible)],
        "direction": direction,
        "sign_direction": sign_direction(par),
        "stock_return": events["announcement_stock"].to_numpy(dtype=float),
        "benchmark_return": events["announcement_benchmark"].to_numpy(dtype=float),
        "adjusted_return": events["announcement_adj"].to_numpy(dtype=float),
    })
    traded = trades["direction"] != 0
    trades["signed_raw"] = (trades["direction"] * trades["stock_return"]).where(traded)
    trades["signed_adj"] = (trades["direction"] * trades["adjusted_return"]).where(traded)
    return trades


def _pseudo_reps(draws: np.ndarray, par_matrix: np.ndarray, raw_matrix: np.ndarray,
                 adj_matrix: np.ndarray, pool_positions: np.ndarray, low: np.ndarray,
                 high: np.ndarray) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    par = par_matrix[draws]
    raw = raw_matrix[draws]
    adj = adj_matrix[draws]
    prior = np.searchsorted(pool_positions, draws, side="left")
    eligible = prior >= MIN_PRIOR_EVENTS
    quint = np.where(eligible, quintile_direction(par, low[prior], high[prior]), 0)
    sign_same = np.where(eligible, sign_direction(par), 0)
    sign_all = sign_direction(par)

    reps: dict[str, list[dict]] = {QUINTILE: [], SIGN_SAME: [], SIGN_ALL: []}
    for r in range(draws.shape[0]):
        for key, direction in ((QUINTILE, quint), (SIGN_SAME, sign_same),
                               (SIGN_ALL, sign_all)):
            row = _with_spreads(direction[r], raw[r], adj[r])
            reps[key].append(row)
    return ({k: pd.DataFrame(v) for k, v in reps.items()},
            pd.Series(eligible.sum(axis=1), name="eligible_pseudo_events"))


def run_quintile_strategy(result) -> QuintileStrategy:
    """Build the strategy for one analysed stock (an ``AnalysisResult``)."""
    strategy = QuintileStrategy()
    if not result.has_events or result.panel is None:
        return strategy

    events = result.included_events.sort_values("session_position").reset_index(drop=True)
    pool_positions = events["session_position"].to_numpy(dtype=int)
    pool_par = events["par"].to_numpy(dtype=float)
    low, high = threshold_lookup(pool_par)

    trades = _actual_trades(events, low, high, pool_positions)
    strategy.trades = trades
    strategy.eligible_events = int(trades["eligible"].sum())
    traded = trades[trades["direction"] != 0]
    strategy.availability_violations = int(
        (traded["latest_threshold_input"] >= traded["signal_and_entry_close"]).sum()
    )

    raw = trades["stock_return"].to_numpy()
    adj = trades["adjusted_return"].to_numpy()
    eligible = trades["eligible"].to_numpy()
    sign = trades["sign_direction"].to_numpy()
    strategy.performance = {
        QUINTILE: _with_spreads(trades["direction"].to_numpy(), raw, adj),
        SIGN_SAME: _with_spreads(np.where(eligible, sign, 0), raw, adj),
        SIGN_ALL: _with_spreads(sign, raw, adj),
    }
    if strategy.eligible_events == 0:
        strategy.notes.append(
            f"No event has {MIN_PRIOR_EVENTS} earlier included events, so the quintile "
            f"strategy never trades. Widen the event range or restore excluded events."
        )

    simulation = result.simulation
    if simulation is None or simulation.draws is None or not simulation.matched_positions:
        strategy.notes.append("No pseudo-event repetitions are available for comparison.")
        return strategy

    matched = set(simulation.matched_positions)
    in_matched = trades.index[events["session_position"].isin(matched)]
    m = trades.loc[in_matched]
    m_raw, m_adj = m["stock_return"].to_numpy(), m["adjusted_return"].to_numpy()
    m_eligible = m["eligible"].to_numpy()
    m_sign = m["sign_direction"].to_numpy()
    strategy.matched_events = len(m)
    strategy.matched_eligible_events = int(m_eligible.sum())
    strategy.matched_performance = {
        QUINTILE: _with_spreads(m["direction"].to_numpy(), m_raw, m_adj),
        SIGN_SAME: _with_spreads(np.where(m_eligible, m_sign, 0), m_raw, m_adj),
        SIGN_ALL: _with_spreads(m_sign, m_raw, m_adj),
    }

    matrix = result.panel.window_matrix()
    raw_matrix = window_returns_vectorized(result.panel.stock_prices, STRATEGY_WINDOW)
    strategy.pseudo_reps, strategy.pseudo_eligible = _pseudo_reps(
        simulation.draws, matrix[PAR_WINDOW.key], raw_matrix,
        matrix[STRATEGY_WINDOW.key], pool_positions, low, high,
    )
    return strategy


COMPARISON_STATS = {
    "trades": ("Executed trades", "count"),
    "mean_adj": ("Average signed market-adjusted return per trade", "pct"),
    "mean_raw": ("Average raw stock strategy return per trade", "pct"),
    "win_adj": ("Win rate, market-adjusted", "pct"),
    "win_raw": ("Win rate, raw stock", "pct"),
    "pnl_adj": ("Cumulative fixed-notional P&L, market-adjusted", "money"),
    "pnl_raw": ("Cumulative fixed-notional P&L, raw stock", "money"),
    "spread_adj": ("Group spread: long-group mean minus short-group mean, market-adjusted",
                   "pct"),
    "spread_raw": ("Group spread: long-group mean minus short-group mean, raw stock", "pct"),
}


def pseudo_comparison(strategy: QuintileStrategy, key: str) -> pd.DataFrame:
    """Actual matched-sample value beside pseudo median and 5th/95th percentiles.

    Percentiles describe variation across repetitions; repetitions are never
    pooled into one sample.
    """
    actual = strategy.matched_performance.get(key, {})
    reps = strategy.pseudo_reps.get(key, pd.DataFrame())
    rows = []
    for stat, (label, kind) in COMPARISON_STATS.items():
        values = reps[stat].to_numpy(dtype=float) if stat in reps else np.array([])
        valid = values[np.isfinite(values)]
        value = actual.get(stat)
        p5 = float(np.percentile(valid, 5)) if valid.size else None
        p95 = float(np.percentile(valid, 95)) if valid.size else None
        rows.append({
            "stat": stat,
            "Statistic": label,
            "kind": kind,
            "Actual (matched sample)": value,
            "Pseudo median": float(np.median(valid)) if valid.size else None,
            "Pseudo 5th percentile": p5,
            "Pseudo 95th percentile": p95,
            "Valid repetitions": int(valid.size),
            "Actual vs pseudo range": _position(value, p5, p95),
        })
    return pd.DataFrame(rows)


def _position(value, p5, p95) -> str:
    if value is None or p5 is None or not np.isfinite(value):
        return "not available"
    if value < p5:
        return "below pseudo 5th pct"
    if value > p95:
        return "above pseudo 95th pct"
    return "within pseudo 5th\u201395th pct"
