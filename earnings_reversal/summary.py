"""Cross-stock summary of the single-stock analyses.

Each stock is still analysed on its own, exactly as on the single-stock pages. This
module only lines the per-stock results up side by side. It is a collection of
separate time-series event studies, not a cross-sectional portfolio sort, and the
stocks are not independent: their earnings cluster in the same reporting seasons
and share market-wide shocks. Nothing here computes a pooled significance test.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .cache import DiskCache
from .config import AnalysisConfig
from .pipeline import AnalysisResult, Inputs, load_inputs, run_analysis
from .pseudo import actual_stat_values, pseudo_summary
from .stats import reversal_stats
from .strategy import COMPARISON_STATS, QUINTILE, SIGN_SAME, pseudo_comparison
from .universe import member

SUMMARY_STATS = (
    "correlation",
    "slope",
    "negative_mean",
    "positive_mean",
    "negative_minus_positive",
    "reversal_frequency",
)

STAT_TITLES = {
    "correlation": "PAR/outcome correlation",
    "slope": "OLS regression slope",
    "negative_mean": "Mean outcome after negative PAR",
    "positive_mean": "Mean outcome after positive PAR",
    "negative_minus_positive": "Negative-PAR minus positive-PAR mean difference",
    "reversal_frequency": "Reversal (opposite-sign) frequency",
}

PERCENT_SUMMARY_STATS = {"negative_mean", "positive_mean", "negative_minus_positive",
                         "reversal_frequency"}

# Which side of the pseudo range the reversal hypothesis predicts.
REVERSAL_SIDE = {
    "correlation": "below",
    "slope": "below",
    "negative_mean": "above",
    "positive_mean": "below",
    "negative_minus_positive": "above",
    "reversal_frequency": "above",
}

BELOW = "below pseudo 5th pct"
ABOVE = "above pseudo 95th pct"
WITHIN = "within pseudo 5th\u201395th pct"
UNAVAILABLE = "not available"


def pseudo_position(actual: float | None, p5: float | None, p95: float | None) -> str:
    if actual is None or p5 is None or p95 is None or not np.isfinite(actual):
        return UNAVAILABLE
    if actual < p5:
        return BELOW
    if actual > p95:
        return ABOVE
    return WITHIN


def stock_summary_row(ticker: str, result: AnalysisResult | None,
                      window_key: str) -> dict:
    """One row for one stock and one outcome window."""
    info = member(ticker)
    row: dict = {
        "ticker": ticker,
        "name": info.name if info else "",
        "sector": info.sector if info else "",
        "status": "ok",
        "events_in_range": 0,
        "events_included": 0,
        "events_excluded": 0,
        "first_event": None,
        "last_event": None,
        "unknown_release_time": 0,
        "matched_events": 0,
        "r_squared": None,
        "negative_count": None,
        "positive_count": None,
    }
    for stat in SUMMARY_STATS:
        row[stat] = None
        row[f"{stat}_matched"] = None
        row[f"{stat}_pseudo_median"] = None
        row[f"{stat}_pseudo_p5"] = None
        row[f"{stat}_pseudo_p95"] = None
        row[f"{stat}_position"] = UNAVAILABLE

    if result is None:
        row["status"] = "not analysed"
        return row

    row["events_in_range"] = int(result.diagnostics.get("completed_in_range", 0) or 0)
    row["events_included"] = int(len(result.included_events))
    row["events_excluded"] = int(len(result.excluded_events))
    if result.blocking_error:
        row["status"] = result.blocking_error
        return row

    events = result.included_events
    row["first_event"] = pd.Timestamp(events["release_date"].min()).date()
    row["last_event"] = pd.Timestamp(events["release_date"].max()).date()
    row["unknown_release_time"] = int((~events["time_known"].astype(bool)).sum())

    stats = result.stats_by_outcome.get(window_key)
    if stats is not None:
        values = actual_stat_values(stats)
        row["r_squared"] = values["r_squared"]
        row["negative_count"] = int(stats.negative_par.count)
        row["positive_count"] = int(stats.positive_par.count)
        for stat in SUMMARY_STATS:
            row[stat] = values[stat]

    simulation = result.simulation
    matched = result.matched_actual_stats.get(window_key)
    rep_stats = simulation.per_rep_stats.get(window_key) if simulation else None
    if simulation is not None:
        row["matched_events"] = len(simulation.matched_positions)
    if matched is not None and rep_stats is not None and not rep_stats.empty:
        matched_values = actual_stat_values(matched)
        for stat in SUMMARY_STATS:
            summary = pseudo_summary(rep_stats, stat)
            actual = matched_values[stat]
            row[f"{stat}_matched"] = actual
            row[f"{stat}_pseudo_median"] = summary["median"]
            row[f"{stat}_pseudo_p5"] = summary["p5"]
            row[f"{stat}_pseudo_p95"] = summary["p95"]
            row[f"{stat}_position"] = pseudo_position(actual, summary["p5"], summary["p95"])
    return row


def universe_table(results: dict[str, AnalysisResult | None],
                   window_key: str) -> pd.DataFrame:
    rows = [stock_summary_row(ticker, result, window_key)
            for ticker, result in results.items()]
    return pd.DataFrame(rows)


def cross_stock_overview(table: pd.DataFrame) -> pd.DataFrame:
    """How each statistic is distributed across the analysed stocks."""
    ok = table[table["status"] == "ok"]
    rows = []
    for stat in SUMMARY_STATS:
        values = pd.to_numeric(ok[stat], errors="coerce").dropna()
        positions = ok[f"{stat}_position"]
        reversal_side = BELOW if REVERSAL_SIDE[stat] == "below" else ABOVE
        opposite_side = ABOVE if reversal_side == BELOW else BELOW
        rows.append({
            "stat": stat,
            "Statistic": STAT_TITLES[stat],
            "Stocks with a value": int(values.size),
            "Median across stocks": float(values.median()) if values.size else None,
            "Equal-weight mean across stocks": float(values.mean()) if values.size else None,
            "Stocks below zero": int((values < 0).sum()),
            "Stocks above zero": int((values > 0).sum()),
            "Beyond pseudo range, reversal side": int((positions == reversal_side).sum()),
            "Beyond pseudo range, opposite side": int((positions == opposite_side).sum()),
            "Reversal side": reversal_side,
        })
    return pd.DataFrame(rows)


def pooled_events(results: dict[str, AnalysisResult | None]) -> pd.DataFrame:
    """Every included event from every stock, tagged with its ticker."""
    frames = []
    for ticker, result in results.items():
        if result is None or not result.has_events:
            continue
        frame = result.included_events.copy()
        info = member(ticker)
        frame["sector"] = info.sector if info else ""
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pooled_group_table(events: pd.DataFrame, outcome_column: str) -> pd.DataFrame:
    """Negative/positive PAR group means over all stocks' events together.

    Descriptive only: events from different stocks on the same dates are not
    independent observations.
    """
    if events is None or events.empty:
        return pd.DataFrame()
    stats = reversal_stats(events["par"], events[outcome_column])
    rows = []
    for group in (stats.negative_par, stats.positive_par):
        rows.append({
            "PAR group": group.label,
            "Events": group.count,
            "Mean outcome": group.mean.value,
            "Median outcome": group.median.value,
            "Opposite-sign frequency": group.opposite_sign_frequency.value,
        })
    rows.append({
        "PAR group": "Negative minus positive (mean difference)",
        "Events": stats.negative_par.count + stats.positive_par.count,
        "Mean outcome": stats.negative_minus_positive_mean.value,
        "Median outcome": None,
        "Opposite-sign frequency": None,
    })
    return pd.DataFrame(rows)


def strategy_universe_table(results: dict[str, AnalysisResult | None]) -> pd.DataFrame:
    """One row per stock: PAR quintile strategy results and pseudo comparison."""
    rows = []
    for ticker, result in results.items():
        info = member(ticker)
        row = {"ticker": ticker, "name": info.name if info else ticker,
               "sector": info.sector if info else ""}
        strategy = getattr(result, "quintile", None) if result is not None else None
        if strategy is None or strategy.trades.empty:
            row["status"] = (result.blocking_error if result is not None
                             and result.blocking_error else "not analysed")
            rows.append(row)
            continue
        row["status"] = "ok"
        row["events"] = len(strategy.trades)
        row["eligible_events"] = strategy.eligible_events
        for key, value in strategy.performance[QUINTILE].items():
            row[key] = value
        row["sign_same_mean_adj"] = strategy.performance[SIGN_SAME]["mean_adj"]
        row["sign_same_pnl_adj"] = strategy.performance[SIGN_SAME]["pnl_adj"]
        row["matched_eligible_events"] = strategy.matched_eligible_events
        trades = strategy.trades
        for side, direction in (("q1", 1), ("q5", -1)):
            group = trades[trades["direction"] == direction]
            row[f"{side}_count"] = len(group)
            row[f"{side}_mean_adj"] = float(group["adjusted_return"].mean()) if len(group) else None
            row[f"{side}_mean_raw"] = float(group["stock_return"].mean()) if len(group) else None
        if strategy.pseudo_reps:
            comparison = pseudo_comparison(strategy, QUINTILE).set_index("stat")
            for stat in COMPARISON_STATS:
                line = comparison.loc[stat]
                row[f"{stat}_matched"] = line["Actual (matched sample)"]
                row[f"{stat}_pseudo_median"] = line["Pseudo median"]
                row[f"{stat}_pseudo_p5"] = line["Pseudo 5th percentile"]
                row[f"{stat}_pseudo_p95"] = line["Pseudo 95th percentile"]
                row[f"{stat}_position"] = line["Actual vs pseudo range"]
            row["pseudo_trades_median"] = float(
                np.median(strategy.pseudo_reps[QUINTILE]["trades"])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def pseudo_median_gap(table: pd.DataFrame, stat: str, better: str) -> pd.DataFrame:
    """Actual matched-sample value minus the stock's own pseudo median.

    ``better`` is "below" or "above": the side of the pseudo median that the
    reversal hypothesis (or a profitable strategy) points to. With no effect,
    roughly half of the stocks would land on each side; the stocks are not
    independent, so the count is descriptive only.
    """
    if table.empty or f"{stat}_matched" not in table:
        return pd.DataFrame(columns=["ticker", "sector", "actual", "pseudo_median",
                                     "gap", "favourable"])
    data = table[table["status"] == "ok"] if "status" in table else table
    actual = pd.to_numeric(data[f"{stat}_matched"], errors="coerce")
    median = pd.to_numeric(data[f"{stat}_pseudo_median"], errors="coerce")
    gap = actual - median
    out = pd.DataFrame({
        "ticker": data["ticker"], "sector": data["sector"],
        "actual": actual, "pseudo_median": median, "gap": gap,
        "favourable": (gap < 0) if better == "below" else (gap > 0),
    })
    return out[out["gap"].notna()].sort_values("gap").reset_index(drop=True)


def volatility_ratios(results: dict[str, AnalysisResult | None],
                      window_key: str) -> pd.Series:
    """Std. dev. of matched actual outcomes divided by that of all pseudo draws.

    A ratio above one means earnings windows are noisier than the fake-date
    windows, so the pseudo 5th-95th range understates how far an earnings
    statistic can move by chance.
    """
    ratios = {}
    for ticker, result in results.items():
        if result is None or result.panel is None or result.simulation is None:
            continue
        simulation = result.simulation
        if simulation.draws is None or not simulation.matched_positions:
            continue
        outcomes = result.panel.window_matrix()[window_key]
        actual = np.nanstd(outcomes[simulation.matched_positions])
        pseudo = np.nanstd(outcomes[simulation.draws.ravel()])
        if np.isfinite(actual) and np.isfinite(pseudo) and pseudo > 0:
            ratios[ticker] = float(actual / pseudo)
    return pd.Series(ratios, dtype=float)


def analyse_universe(config: AnalysisConfig, tickers, cache: DiskCache,
                     manual_records: pd.DataFrame | None = None,
                     excluded_by_ticker: dict[str, tuple[str, ...]] | None = None,
                     force_refresh: bool = False,
                     progress: Callable[[int, int, str], None] | None = None
                     ) -> dict[str, AnalysisResult]:
    """Run the single-stock pipeline for every ticker with shared settings."""
    excluded_by_ticker = excluded_by_ticker or {}
    tickers = list(tickers)
    results: dict[str, AnalysisResult] = {}
    for index, ticker in enumerate(tickers):
        if progress is not None:
            progress(index, len(tickers), ticker)
        stock_config = AnalysisConfig(
            ticker=ticker,
            benchmark=config.benchmark,
            event_start=config.event_start,
            event_end=config.event_end,
            seed=config.seed,
            n_repetitions=config.n_repetitions,
            excluded_release_dates=tuple(excluded_by_ticker.get(ticker, ())),
        ).normalised()
        try:
            inputs = load_inputs(stock_config, cache, force_refresh)
            results[ticker] = run_analysis(inputs, manual_records, stock_config)
        except Exception as exc:
            failed = AnalysisResult(config=stock_config, inputs=Inputs(config=stock_config))
            failed.blocking_error = f"{type(exc).__name__}: {exc}"
            results[ticker] = failed
    if progress is not None:
        progress(len(tickers), len(tickers), "")
    return results
