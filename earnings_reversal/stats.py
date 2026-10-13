"""Descriptive statistics for the PAR / outcome relationship.

Everything here is exploratory and in-sample. A fitted line summarises the
observations it was estimated on; it is not evidence of out-of-sample predictive
performance. Degenerate inputs return an explicit "not available" reason rather
than a misleading number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

MIN_CORRELATION_OBS = 3
MIN_REGRESSION_OBS = 3
SMALL_SAMPLE_THRESHOLD = 30


@dataclass
class Stat:
    """A statistic that may legitimately be unavailable."""

    value: float | None = None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None and np.isfinite(self.value)

    def format(self, digits: int = 4, percent: bool = False) -> str:
        if not self.available:
            return "not available"
        if percent:
            return f"{self.value * 100:.2f}%"
        return f"{self.value:.{digits}f}"


@dataclass
class GroupSummary:
    label: str
    count: int = 0
    mean: Stat = field(default_factory=Stat)
    median: Stat = field(default_factory=Stat)
    opposite_sign_frequency: Stat = field(default_factory=Stat)
    opposite_sign_count: int = 0
    both_nonzero_count: int = 0
    zero_outcome_count: int = 0


@dataclass
class ReversalStats:
    """All statistics for one (PAR, outcome) pair."""

    outcome_label: str
    n: int = 0
    correlation: Stat = field(default_factory=Stat)
    slope: Stat = field(default_factory=Stat)
    intercept: Stat = field(default_factory=Stat)
    r_squared: Stat = field(default_factory=Stat)
    opposite_sign_frequency: Stat = field(default_factory=Stat)
    opposite_sign_count: int = 0
    both_nonzero_count: int = 0
    zero_par_count: int = 0
    zero_outcome_count: int = 0
    negative_par: GroupSummary = field(default_factory=lambda: GroupSummary("Negative PAR"))
    positive_par: GroupSummary = field(default_factory=lambda: GroupSummary("Positive PAR"))
    zero_par: GroupSummary = field(default_factory=lambda: GroupSummary("Zero PAR"))
    negative_minus_positive_mean: Stat = field(default_factory=Stat)
    notes: list[str] = field(default_factory=list)

    @property
    def small_sample(self) -> bool:
        return 0 < self.n < SMALL_SAMPLE_THRESHOLD


def fast_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form simple OLS returning (slope, intercept, r_squared)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < MIN_REGRESSION_OBS:
        return float("nan"), float("nan"), float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx == 0.0:
        return float("nan"), float("nan"), float("nan")
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    syy = float(((y - y_mean) ** 2).sum())
    r_squared = (sxy ** 2) / (sxx * syy) if syy > 0 else float("nan")
    return slope, intercept, r_squared


def statsmodels_ols(x: np.ndarray, y: np.ndarray):
    """Full OLS fit used for the actual-event sample."""
    import statsmodels.api as sm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, design).fit()


def _group_summary(label: str, par: np.ndarray, outcome: np.ndarray) -> GroupSummary:
    summary = GroupSummary(label=label, count=int(par.size))
    if par.size == 0:
        reason = f"No events with {label.lower()}."
        summary.mean = Stat(None, reason)
        summary.median = Stat(None, reason)
        summary.opposite_sign_frequency = Stat(None, reason)
        return summary
    summary.mean = Stat(float(np.mean(outcome)))
    summary.median = Stat(float(np.median(outcome)))
    both_nonzero = (par != 0) & (outcome != 0)
    summary.both_nonzero_count = int(both_nonzero.sum())
    summary.zero_outcome_count = int((outcome == 0).sum())
    if summary.both_nonzero_count == 0:
        summary.opposite_sign_frequency = Stat(
            None,
            f"No events in this group have both a nonzero PAR and a nonzero outcome "
            f"({summary.zero_outcome_count} zero outcome(s)).",
        )
    else:
        opposite = (par[both_nonzero] * outcome[both_nonzero]) < 0
        summary.opposite_sign_count = int(opposite.sum())
        summary.opposite_sign_frequency = Stat(
            float(opposite.mean())
        )
    return summary


def reversal_stats(par: np.ndarray | pd.Series, outcome: np.ndarray | pd.Series,
                   outcome_label: str = "outcome",
                   use_statsmodels: bool = True) -> ReversalStats:
    """Correlation, OLS fit, reversal frequency and PAR-sign group summaries.

    A reversal is defined as ``PAR * outcome_return < 0``. Observations where
    either value is exactly zero are counted separately and never folded into the
    reversal or continuation count.
    """
    par = np.asarray(pd.Series(par).astype(float))
    outcome = np.asarray(pd.Series(outcome).astype(float))
    if par.size != outcome.size:
        raise ValueError("PAR and outcome must have the same length.")

    usable = np.isfinite(par) & np.isfinite(outcome)
    dropped = int((~usable).sum())
    par = par[usable]
    outcome = outcome[usable]

    result = ReversalStats(outcome_label=outcome_label, n=int(par.size))
    if dropped:
        result.notes.append(f"{dropped} observation(s) dropped for missing values.")
    if result.n == 0:
        empty = Stat(None, "No usable observations.")
        result.correlation = empty
        result.slope = empty
        result.intercept = empty
        result.r_squared = empty
        result.opposite_sign_frequency = empty
        result.negative_minus_positive_mean = empty
        result.negative_par = _group_summary("Negative PAR", par, outcome)
        result.positive_par = _group_summary("Positive PAR", par, outcome)
        result.zero_par = _group_summary("Zero PAR", par, outcome)
        return result

    par_constant = float(np.ptp(par)) == 0.0
    outcome_constant = float(np.ptp(outcome)) == 0.0

    if result.n < MIN_CORRELATION_OBS:
        reason = f"Only {result.n} observation(s); at least {MIN_CORRELATION_OBS} required."
        result.correlation = Stat(None, reason)
    elif par_constant or outcome_constant:
        which = "PAR" if par_constant else "the outcome"
        result.correlation = Stat(None, f"{which} is constant, so correlation is undefined.")
    else:
        corr, _ = scipy_stats.pearsonr(par, outcome)
        result.correlation = Stat(float(corr))

    if result.n < MIN_REGRESSION_OBS:
        reason = f"Only {result.n} observation(s); at least {MIN_REGRESSION_OBS} required."
        result.slope = Stat(None, reason)
        result.intercept = Stat(None, reason)
        result.r_squared = Stat(None, reason)
    elif par_constant:
        reason = "PAR is constant, so the slope is not identified."
        result.slope = Stat(None, reason)
        result.intercept = Stat(None, reason)
        result.r_squared = Stat(None, reason)
    else:
        if use_statsmodels:
            try:
                fit = statsmodels_ols(par, outcome)
                intercept, slope = float(fit.params[0]), float(fit.params[1])
                r_squared = float(fit.rsquared)
            except Exception as exc:  # pragma: no cover - defensive
                slope, intercept, r_squared = fast_ols(par, outcome)
                result.notes.append(f"statsmodels OLS failed ({exc}); used closed-form OLS.")
        else:
            slope, intercept, r_squared = fast_ols(par, outcome)
        result.slope = Stat(slope)
        result.intercept = Stat(intercept)
        result.r_squared = Stat(r_squared if np.isfinite(r_squared) else None,
                                "" if np.isfinite(r_squared) else
                                "The outcome is constant, so R-squared is undefined.")

    result.zero_par_count = int((par == 0).sum())
    result.zero_outcome_count = int((outcome == 0).sum())
    both_nonzero = (par != 0) & (outcome != 0)
    result.both_nonzero_count = int(both_nonzero.sum())
    if result.both_nonzero_count == 0:
        result.opposite_sign_frequency = Stat(
            None,
            "No events have both a nonzero PAR and a nonzero outcome, so the "
            "opposite-sign frequency is undefined.",
        )
    else:
        opposite = (par[both_nonzero] * outcome[both_nonzero]) < 0
        result.opposite_sign_count = int(opposite.sum())
        result.opposite_sign_frequency = Stat(float(opposite.mean()))

    negative = par < 0
    positive = par > 0
    zero = par == 0
    result.negative_par = _group_summary("Negative PAR", par[negative], outcome[negative])
    result.positive_par = _group_summary("Positive PAR", par[positive], outcome[positive])
    result.zero_par = _group_summary("Zero PAR", par[zero], outcome[zero])

    if result.negative_par.mean.available and result.positive_par.mean.available:
        result.negative_minus_positive_mean = Stat(
            result.negative_par.mean.value - result.positive_par.mean.value
        )
    else:
        missing = []
        if not result.negative_par.mean.available:
            missing.append("negative-PAR")
        if not result.positive_par.mean.available:
            missing.append("positive-PAR")
        result.negative_minus_positive_mean = Stat(
            None, f"Empty {' and '.join(missing)} group."
        )

    if result.small_sample:
        result.notes.append(
            f"Small sample: {result.n} events. Treat every statistic as exploratory."
        )
    return result


def stats_to_rows(stats: ReversalStats) -> list[dict[str, Any]]:
    """Flatten a ReversalStats into display rows for the UI."""
    rows = [
        {"Statistic": "Events included", "Value": str(stats.n), "Note": ""},
        {"Statistic": "Pearson correlation (PAR vs outcome)",
         "Value": stats.correlation.format(), "Note": stats.correlation.reason},
        {"Statistic": "OLS slope", "Value": stats.slope.format(),
         "Note": stats.slope.reason},
        {"Statistic": "OLS intercept", "Value": stats.intercept.format(percent=True),
         "Note": stats.intercept.reason},
        {"Statistic": "R-squared", "Value": stats.r_squared.format(),
         "Note": stats.r_squared.reason},
        {"Statistic": "Opposite-sign (reversal) frequency",
         "Value": stats.opposite_sign_frequency.format(percent=True),
         "Note": (f"{stats.opposite_sign_count} of {stats.both_nonzero_count} events "
                  f"with both returns nonzero"
                  if stats.opposite_sign_frequency.available
                  else stats.opposite_sign_frequency.reason)},
        {"Statistic": "Zero-PAR events", "Value": str(stats.zero_par_count),
         "Note": "Reported separately, never counted as reversals or continuations"},
        {"Statistic": "Zero-outcome events", "Value": str(stats.zero_outcome_count),
         "Note": "Reported separately, never counted as reversals or continuations"},
        {"Statistic": "Negative-PAR minus positive-PAR mean difference",
         "Value": stats.negative_minus_positive_mean.format(percent=True),
         "Note": stats.negative_minus_positive_mean.reason or
                 "Events for one company at different times; not a tradable long-short return"},
    ]
    return rows


def group_table(stats_by_outcome: dict[str, ReversalStats]) -> pd.DataFrame:
    """Negative/positive PAR summary across all outcome windows."""
    rows = []
    for label, stats in stats_by_outcome.items():
        for group in (stats.negative_par, stats.positive_par, stats.zero_par):
            if group.count == 0 and group.label == "Zero PAR":
                continue
            rows.append({
                "Outcome window": label,
                "PAR group": group.label,
                "Count": group.count,
                "Mean outcome": group.mean.value,
                "Median outcome": group.median.value,
                "Opposite-sign frequency": group.opposite_sign_frequency.value,
                "Both-nonzero events": group.both_nonzero_count,
                "Note": group.mean.reason or group.opposite_sign_frequency.reason,
            })
        rows.append({
            "Outcome window": label,
            "PAR group": "Negative minus positive (mean difference)",
            "Count": stats.negative_par.count + stats.positive_par.count,
            "Mean outcome": stats.negative_minus_positive_mean.value,
            "Median outcome": None,
            "Opposite-sign frequency": None,
            "Both-nonzero events": None,
            "Note": stats.negative_minus_positive_mean.reason,
        })
    return pd.DataFrame(rows)
