"""Pseudo-event construction and the actual-versus-pseudo Monte Carlo comparison.

For each eligible actual earnings event, candidate pseudo-dates sit 10 to 40
trading sessions before it (inclusive). A candidate survives only if all required
stock and benchmark prices exist and its return-analysis interval contains no
known earnings release.

Repeating the draws is a sensitivity check on which pseudo dates were chosen. It
does not create additional independent observations, and the resulting spread is
not a confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CONTAMINATION_BOUNDARY,
    OUTCOME_WINDOWS,
    PAR_WINDOW,
    PSEUDO_MAX_OFFSET,
    PSEUDO_MIN_OFFSET,
    REQUIRED_MAX_OFFSET,
    REQUIRED_MIN_OFFSET,
)
from .events import AlignedPanel
from .stats import ReversalStats, fast_ols, reversal_stats

# Two pseudo windows overlap when their [-5,+8] session spans intersect.
WINDOW_SPAN = REQUIRED_MAX_OFFSET - REQUIRED_MIN_OFFSET  # 13 sessions apart

CONTAMINATION_RULE = (
    "A pseudo window is rejected if any known completed release date falls in the "
    "calendar interval that drives its returns. With the 'after_baseline' boundary "
    "the interval runs from the day after the session -5 baseline close through the "
    "session +8 date, so every non-trading date between sessions is screened while "
    "the baseline close date itself is not treated as contaminated."
)


@dataclass
class PseudoSimulation:
    candidates: dict[int, np.ndarray] = field(default_factory=dict)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    matched_positions: list[int] = field(default_factory=list)
    unmatched: pd.DataFrame = field(default_factory=pd.DataFrame)
    draws: np.ndarray | None = None
    per_rep_stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    reuse: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    n_repetitions: int = 0
    notes: list[str] = field(default_factory=list)


def contamination_interval(sessions: pd.DatetimeIndex, position: int,
                           boundary: str = CONTAMINATION_BOUNDARY
                           ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Calendar interval screened for earnings releases around ``position``."""
    lo = position + REQUIRED_MIN_OFFSET
    hi = position + REQUIRED_MAX_OFFSET
    if lo < 0 or hi >= len(sessions):
        return None
    baseline_date = pd.Timestamp(sessions[lo]).normalize()
    end_date = pd.Timestamp(sessions[hi]).normalize()
    first_return_date = pd.Timestamp(sessions[position + PAR_WINDOW.start]).normalize()
    if boundary == "after_baseline":
        start_date = baseline_date + pd.Timedelta(days=1)
    elif boundary == "sessions_only":
        start_date = first_return_date
    else:
        raise ValueError(f"Unknown contamination boundary '{boundary}'.")
    return start_date, end_date


def _has_release(release_dates: np.ndarray, start: pd.Timestamp,
                 end: pd.Timestamp) -> bool:
    if release_dates.size == 0:
        return False
    left = np.searchsorted(release_dates, np.datetime64(start.to_datetime64()), side="left")
    right = np.searchsorted(release_dates, np.datetime64(end.to_datetime64()), side="right")
    return bool(right > left)


def build_candidates(panel: AlignedPanel, event_positions: list[int],
                     release_dates: pd.DatetimeIndex,
                     event_labels: list[pd.Timestamp] | None = None,
                     boundary: str = CONTAMINATION_BOUNDARY
                     ) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Eligible pseudo positions per actual event, plus exclusion diagnostics."""
    sessions = panel.sessions
    available = panel.required_span_available()
    release_array = np.array(
        sorted(pd.DatetimeIndex(release_dates).normalize().to_numpy()),
        dtype="datetime64[ns]",
    ) if len(release_dates) else np.array([], dtype="datetime64[ns]")

    candidates: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    labels = event_labels or [sessions[p] for p in event_positions]

    for position, label in zip(event_positions, labels):
        considered = 0
        out_of_range = 0
        missing_data = 0
        contaminated = 0
        eligible: list[int] = []
        for offset in range(PSEUDO_MIN_OFFSET, PSEUDO_MAX_OFFSET + 1):
            candidate = position - offset
            considered += 1
            if candidate < 0 or candidate >= len(sessions):
                out_of_range += 1
                continue
            if not available[candidate]:
                missing_data += 1
                continue
            interval = contamination_interval(sessions, candidate, boundary)
            if interval is None:
                out_of_range += 1
                continue
            if _has_release(release_array, interval[0], interval[1]):
                contaminated += 1
                continue
            eligible.append(candidate)
        array = np.array(eligible, dtype=int)
        candidates[position] = array
        overlapping_actual = int(np.sum((position - array) <= WINDOW_SPAN)) if array.size else 0
        rows.append({
            "release_date": label,
            "event_position": position,
            "candidates_considered": considered,
            "rejected_out_of_range": out_of_range,
            "rejected_missing_data": missing_data,
            "rejected_earnings_contamination": contaminated,
            "eligible_candidates": int(array.size),
            "eligible_overlapping_actual_window": overlapping_actual,
        })
    return candidates, pd.DataFrame(rows)


def _rep_stats_row(par: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    usable = np.isfinite(par) & np.isfinite(outcome)
    par = par[usable]
    outcome = outcome[usable]
    n = par.size
    row: dict[str, float] = {
        "n": float(n),
        "correlation": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
        "negative_count": np.nan,
        "positive_count": np.nan,
        "negative_mean": np.nan,
        "positive_mean": np.nan,
        "negative_minus_positive": np.nan,
        "reversal_frequency": np.nan,
    }
    if n == 0:
        return row
    par_var = float(np.ptp(par))
    out_var = float(np.ptp(outcome))
    if n >= 3 and par_var > 0 and out_var > 0:
        row["correlation"] = float(np.corrcoef(par, outcome)[0, 1])
    if n >= 3 and par_var > 0:
        slope, intercept, r_squared = fast_ols(par, outcome)
        row["slope"] = slope
        row["intercept"] = intercept
        row["r_squared"] = r_squared

    negative = par < 0
    positive = par > 0
    row["negative_count"] = float(negative.sum())
    row["positive_count"] = float(positive.sum())
    if negative.any():
        row["negative_mean"] = float(outcome[negative].mean())
    if positive.any():
        row["positive_mean"] = float(outcome[positive].mean())
    if negative.any() and positive.any():
        row["negative_minus_positive"] = row["negative_mean"] - row["positive_mean"]

    both_nonzero = (par != 0) & (outcome != 0)
    if both_nonzero.any():
        row["reversal_frequency"] = float(
            ((par[both_nonzero] * outcome[both_nonzero]) < 0).mean()
        )
    return row


def run_simulation(panel: AlignedPanel, event_positions: list[int],
                   release_dates: pd.DatetimeIndex, seed: int, n_repetitions: int,
                   event_labels: list[pd.Timestamp] | None = None,
                   boundary: str = CONTAMINATION_BOUNDARY) -> PseudoSimulation:
    """Draw pseudo dates repeatedly and recompute the statistics each time."""
    simulation = PseudoSimulation(seed=int(seed), n_repetitions=int(n_repetitions))
    labels = event_labels or [panel.sessions[p] for p in event_positions]
    candidates, diagnostics = build_candidates(
        panel, event_positions, release_dates, labels, boundary
    )
    simulation.candidates = candidates
    simulation.diagnostics = diagnostics

    matched = [p for p in event_positions if candidates[p].size > 0]
    unmatched_mask = diagnostics["eligible_candidates"] == 0
    simulation.matched_positions = matched
    simulation.unmatched = diagnostics.loc[unmatched_mask].copy()

    if not matched:
        simulation.notes.append(
            "No actual event has an eligible pseudo-date candidate, so no "
            "actual-versus-pseudo comparison is possible."
        )
        return simulation
    if len(simulation.unmatched):
        simulation.notes.append(
            f"{len(simulation.unmatched)} actual event(s) had no eligible pseudo "
            f"candidate and are dropped from both the matched actual sample and the "
            f"pseudo sample."
        )

    matrix = panel.window_matrix()
    rng = np.random.default_rng(int(seed))
    candidate_lists = [candidates[p] for p in matched]
    n_events = len(matched)
    draws = np.empty((int(n_repetitions), n_events), dtype=int)
    for column, array in enumerate(candidate_lists):
        picks = rng.integers(0, array.size, size=int(n_repetitions))
        draws[:, column] = array[picks]
    simulation.draws = draws

    par_matrix = matrix[PAR_WINDOW.key]
    for window in OUTCOME_WINDOWS:
        outcome_matrix = matrix[window.key]
        rows = []
        for rep in range(int(n_repetitions)):
            positions = draws[rep]
            rows.append(_rep_stats_row(par_matrix[positions], outcome_matrix[positions]))
        simulation.per_rep_stats[window.key] = pd.DataFrame(rows)

    simulation.reuse = _reuse_diagnostics(draws)
    simulation.notes.append(CONTAMINATION_RULE)
    return simulation


def _reuse_diagnostics(draws: np.ndarray) -> dict[str, Any]:
    """Overlap and reuse statistics; pseudo windows are not independent draws."""
    n_reps, n_events = draws.shape
    overlapping_pairs = np.zeros(n_reps, dtype=int)
    unique_per_rep = np.zeros(n_reps, dtype=int)
    for rep in range(n_reps):
        row = np.sort(draws[rep])
        unique_per_rep[rep] = np.unique(row).size
        diffs = row[:, None] - row[None, :]
        mask = np.triu(np.ones((n_events, n_events), dtype=bool), k=1)
        overlapping_pairs[rep] = int(np.sum(np.abs(diffs[mask]) <= WINDOW_SPAN))
    flat = draws.ravel()
    unique_overall, counts = np.unique(flat, return_counts=True)
    return {
        "repetitions": int(n_reps),
        "events_per_repetition": int(n_events),
        "total_draws": int(flat.size),
        "distinct_pseudo_dates_used": int(unique_overall.size),
        "mean_unique_dates_per_repetition": float(unique_per_rep.mean()),
        "mean_overlapping_pairs_per_repetition": float(overlapping_pairs.mean()),
        "repetitions_with_any_overlap": int((overlapping_pairs > 0).sum()),
        "max_reuse_of_a_single_date": int(counts.max()) if counts.size else 0,
        "mean_reuse_per_distinct_date": float(flat.size / unique_overall.size)
        if unique_overall.size else float("nan"),
    }


def actual_matched_stats(panel: AlignedPanel, matched_positions: list[int]
                         ) -> dict[str, ReversalStats]:
    """Statistics for the actual events that survived pseudo matching."""
    matrix = panel.window_matrix()
    positions = np.array(matched_positions, dtype=int)
    result: dict[str, ReversalStats] = {}
    for window in OUTCOME_WINDOWS:
        result[window.key] = reversal_stats(
            matrix[PAR_WINDOW.key][positions],
            matrix[window.key][positions],
            outcome_label=window.label,
        )
    return result


STAT_LABELS = {
    "correlation": "PAR/outcome correlation",
    "slope": "OLS regression slope",
    "r_squared": "R-squared",
    "negative_count": "Negative-PAR count",
    "positive_count": "Positive-PAR count",
    "negative_mean": "Mean outcome after negative PAR",
    "positive_mean": "Mean outcome after positive PAR",
    "negative_minus_positive": "Negative-PAR minus positive-PAR mean difference",
    "reversal_frequency": "Reversal (opposite-sign) frequency",
}

PERCENT_STATS = {"negative_mean", "positive_mean", "negative_minus_positive",
                 "reversal_frequency"}


def actual_stat_values(stats: ReversalStats) -> dict[str, float | None]:
    """The statistics of ``stats`` keyed like the per-repetition pseudo columns."""
    return {
        "correlation": stats.correlation.value,
        "slope": stats.slope.value,
        "r_squared": stats.r_squared.value,
        "negative_count": float(stats.negative_par.count),
        "positive_count": float(stats.positive_par.count),
        "negative_mean": stats.negative_par.mean.value,
        "positive_mean": stats.positive_par.mean.value,
        "negative_minus_positive": stats.negative_minus_positive_mean.value,
        "reversal_frequency": stats.opposite_sign_frequency.value,
    }


def pseudo_summary(rep_stats: pd.DataFrame, key: str) -> dict[str, float | int | None]:
    """Median, 5th and 95th percentile of one statistic across repetitions."""
    series = rep_stats[key].to_numpy(dtype=float) if key in rep_stats else np.array([])
    valid = series[np.isfinite(series)]
    return {
        "median": float(np.median(valid)) if valid.size else None,
        "p5": float(np.percentile(valid, 5)) if valid.size else None,
        "p95": float(np.percentile(valid, 95)) if valid.size else None,
        "valid": int(valid.size),
    }


def comparison_table(actual: ReversalStats, rep_stats: pd.DataFrame) -> pd.DataFrame:
    """Actual matched-sample statistic beside the pseudo distribution summary."""
    actual_values = actual_stat_values(actual)
    rows = []
    for key, label in STAT_LABELS.items():
        summary = pseudo_summary(rep_stats, key)
        rows.append({
            "Statistic": label,
            "Actual (matched sample)": actual_values.get(key),
            "Pseudo median": summary["median"],
            "Pseudo 5th percentile": summary["p5"],
            "Pseudo 95th percentile": summary["p95"],
            "Valid repetitions": summary["valid"],
            "Is percentage": key in PERCENT_STATS,
        })
    return pd.DataFrame(rows)
