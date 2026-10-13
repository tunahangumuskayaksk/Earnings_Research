"""Orchestration: retrieval is kept separate from computation.

``load_inputs`` performs (or reuses) downloads; ``run_analysis`` consumes whatever
inputs are available — cached, freshly downloaded, or manually supplied — and runs
the identical computation path over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cache import DiskCache
from .config import (
    AnalysisConfig,
    OUTCOME_WINDOWS,
    PAR_WINDOW,
    PRICE_PAD_DAYS_AFTER,
    PRICE_PAD_DAYS_BEFORE,
)
from .earnings import (
    EarningsData,
    EarningsFetchError,
    coverage_diagnostics,
    confirmed_releases,
    load_earnings,
    reconcile_records,
    select_analysis_events,
)
from .events import AlignedPanel, build_event_table
from .prices import PriceData, PriceDownloadError, load_prices
from .pseudo import PseudoSimulation, actual_matched_stats, run_simulation
from .stats import ReversalStats, reversal_stats
from .strategy import QuintileStrategy, run_quintile_strategy


def price_range(config: AnalysisConfig) -> tuple[str, str]:
    """Padded download range covering every actual and pseudo window."""
    start = (pd.Timestamp(config.event_start) - pd.Timedelta(days=PRICE_PAD_DAYS_BEFORE)).date()
    end = (pd.Timestamp(config.event_end) + pd.Timedelta(days=PRICE_PAD_DAYS_AFTER)).date()
    return start.isoformat(), end.isoformat()


@dataclass
class Inputs:
    """Raw retrieved material plus any retrieval errors."""

    config: AnalysisConfig
    stock: PriceData | None = None
    benchmark: PriceData | None = None
    earnings: EarningsData | None = None
    price_error: str = ""
    earnings_error: str = ""
    requested_price_start: str = ""
    requested_price_end: str = ""

    @property
    def prices_available(self) -> bool:
        return self.stock is not None and self.benchmark is not None


@dataclass
class AnalysisResult:
    config: AnalysisConfig
    inputs: Inputs
    reconciled: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)
    selected_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    panel: AlignedPanel | None = None
    included_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats_by_outcome: dict[str, ReversalStats] = field(default_factory=dict)
    simulation: PseudoSimulation | None = None
    matched_actual_stats: dict[str, ReversalStats] = field(default_factory=dict)
    matched_event_dates: list[pd.Timestamp] = field(default_factory=list)
    quintile: QuintileStrategy | None = None
    messages: list[str] = field(default_factory=list)
    blocking_error: str = ""

    @property
    def has_events(self) -> bool:
        return len(self.included_events) > 0


def load_inputs(config: AnalysisConfig, cache: DiskCache,
                force_refresh: bool = False) -> Inputs:
    config = config.normalised()
    start, end = price_range(config)
    inputs = Inputs(config=config, requested_price_start=start, requested_price_end=end)

    try:
        inputs.stock = load_prices(config.ticker, start, end, cache, force_refresh)
    except PriceDownloadError as exc:
        inputs.price_error = str(exc)
    try:
        inputs.benchmark = load_prices(config.benchmark, start, end, cache, force_refresh)
    except PriceDownloadError as exc:
        inputs.price_error = (inputs.price_error + " | " if inputs.price_error else "") + str(exc)

    try:
        inputs.earnings = load_earnings(config.ticker, cache, force_refresh)
    except EarningsFetchError as exc:
        inputs.earnings_error = str(exc)
    return inputs


def run_analysis(inputs: Inputs, manual_records: pd.DataFrame | None = None,
                 config: AnalysisConfig | None = None) -> AnalysisResult:
    config = (config or inputs.config).normalised()
    result = AnalysisResult(config=config, inputs=inputs)

    downloaded = inputs.earnings.records if inputs.earnings is not None else None
    if manual_records is not None and len(manual_records):
        manual_records = manual_records[
            manual_records["ticker"].astype(str).str.upper() == config.ticker
        ].reset_index(drop=True)
    frames = [f for f in (downloaded, manual_records) if f is not None and len(f)]
    result.reconciled = reconcile_records(frames)
    result.diagnostics = coverage_diagnostics(
        result.reconciled, config.event_start, config.event_end
    )
    if inputs.earnings_error:
        result.messages.append(inputs.earnings_error)
    if manual_records is not None and len(manual_records):
        result.messages.append(
            f"{len(manual_records)} manually uploaded earnings record(s) merged; "
            f"{int(manual_records['verified'].sum())} of them are marked verified."
        )

    result.selected_events = select_analysis_events(
        result.reconciled, config.event_start, config.event_end,
        config.excluded_release_dates,
    )
    if config.excluded_release_dates:
        result.messages.append(
            f"{len(config.excluded_release_dates)} event(s) excluded by the user: "
            + ", ".join(config.excluded_release_dates)
        )

    if not inputs.prices_available:
        result.blocking_error = inputs.price_error or "Price data is unavailable."
        return result

    result.panel = AlignedPanel.from_price_data(inputs.stock, inputs.benchmark)
    result.included_events, result.excluded_events = build_event_table(
        result.panel, result.selected_events
    )
    if not result.has_events:
        if len(result.selected_events) == 0:
            result.blocking_error = (
                "No completed earnings records are available inside the selected "
                "event range, so no event statistics can be computed."
            )
        else:
            result.blocking_error = (
                "Every candidate event was excluded. See the exclusion table for "
                "the reason attached to each one."
            )
        return result

    par = result.included_events["par"].to_numpy(dtype=float)
    for window in OUTCOME_WINDOWS:
        column = f"{window.key}_adj"
        result.stats_by_outcome[window.key] = reversal_stats(
            par, result.included_events[column].to_numpy(dtype=float),
            outcome_label=window.label,
        )

    positions = result.included_events["session_position"].astype(int).tolist()
    labels = list(result.included_events["release_date"])
    releases = confirmed_releases(result.reconciled)
    result.simulation = run_simulation(
        result.panel, positions, releases, config.seed, config.n_repetitions,
        event_labels=labels,
    )
    if result.simulation.matched_positions:
        result.matched_actual_stats = actual_matched_stats(
            result.panel, result.simulation.matched_positions
        )
        position_to_date = dict(zip(positions, labels))
        result.matched_event_dates = [
            position_to_date[p] for p in result.simulation.matched_positions
        ]
    result.quintile = run_quintile_strategy(result)
    return result


def events_for_download(result: AnalysisResult) -> pd.DataFrame:
    """Audit table formatted for CSV export (decimals preserved)."""
    if not result.has_events:
        return pd.DataFrame()
    frame = result.included_events.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.date
    return frame


def matched_sample_note(result: AnalysisResult) -> str:
    if result.simulation is None or not result.simulation.matched_positions:
        return "No matched actual/pseudo sample is available."
    matched = len(result.simulation.matched_positions)
    total = len(result.included_events)
    if matched == total:
        return (
            f"All {total} included events have at least one eligible pseudo-date "
            f"candidate, so the matched actual sample equals the full event sample."
        )
    return (
        f"{matched} of {total} included events have at least one eligible pseudo-date "
        f"candidate. The actual-versus-pseudo comparison uses only those {matched} events."
    )
