"""Pseudo-date offsets, contamination screening, reproducibility and matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.config import (
    PSEUDO_MAX_OFFSET,
    PSEUDO_MIN_OFFSET,
    REQUIRED_MAX_OFFSET,
    REQUIRED_MIN_OFFSET,
)
from earnings_reversal.pseudo import (
    build_candidates,
    contamination_interval,
    run_simulation,
)
from earnings_reversal.trading_calendar import session_position, trading_sessions
from conftest import make_panel


@pytest.fixture
def panel():
    sessions = trading_sessions("2020-01-02", "2022-12-30")
    rng = np.random.default_rng(3)
    stock = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=len(sessions)))
    bench = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.006, size=len(sessions)))
    return make_panel(sessions, stock, bench)


def test_candidate_offsets_stay_within_ten_to_forty_sessions(panel):
    position = session_position(panel.sessions, pd.Timestamp("2022-06-15"))
    candidates, diagnostics = build_candidates(panel, [position], pd.DatetimeIndex([]))
    offsets = position - candidates[position]
    assert offsets.size > 0
    assert offsets.min() >= PSEUDO_MIN_OFFSET
    assert offsets.max() <= PSEUDO_MAX_OFFSET
    # Both endpoints are inclusive, so all 31 offsets are available here.
    assert sorted(offsets.tolist()) == list(range(PSEUDO_MIN_OFFSET, PSEUDO_MAX_OFFSET + 1))
    assert diagnostics.loc[0, "candidates_considered"] == 31


def test_contamination_interval_covers_non_trading_days_between_sessions(panel):
    position = session_position(panel.sessions, pd.Timestamp("2021-07-08"))
    start, end = contamination_interval(panel.sessions, position)
    baseline = panel.sessions[position + REQUIRED_MIN_OFFSET]
    last = panel.sessions[position + REQUIRED_MAX_OFFSET]
    assert start == baseline + pd.Timedelta(days=1)
    assert end == last
    # The span is a calendar range, so weekends and the holiday inside it are screened.
    span = pd.date_range(start, end, freq="D")
    assert pd.Timestamp("2021-07-05") in span      # Independence Day observed
    assert any(day.weekday() >= 5 for day in span)  # weekends included


def test_pseudo_window_containing_an_earnings_release_is_rejected(panel):
    position = session_position(panel.sessions, pd.Timestamp("2022-06-15"))
    candidate = position - 20
    start, end = contamination_interval(panel.sessions, candidate)
    contaminating = pd.DatetimeIndex([start + pd.Timedelta(days=2)])

    clean, _ = build_candidates(panel, [position], pd.DatetimeIndex([]))
    screened, diagnostics = build_candidates(panel, [position], contaminating)

    assert candidate in clean[position]
    assert candidate not in screened[position]
    assert diagnostics.loc[0, "rejected_earnings_contamination"] >= 1


def test_release_on_a_weekend_inside_the_window_is_still_detected(panel):
    position = session_position(panel.sessions, pd.Timestamp("2022-06-15"))
    candidate = position - 25
    start, end = contamination_interval(panel.sessions, candidate)
    weekend = [d for d in pd.date_range(start, end, freq="D") if d.weekday() >= 5]
    assert weekend, "the screened span should contain a weekend"
    screened, _ = build_candidates(panel, [position], pd.DatetimeIndex([weekend[0]]))
    assert candidate not in screened[position]


def test_baseline_close_date_boundary_is_handled_as_documented(panel):
    """The -5 baseline close date itself is outside the screened interval."""
    position = session_position(panel.sessions, pd.Timestamp("2022-06-15"))
    candidate = position - 20
    baseline_date = panel.sessions[candidate + REQUIRED_MIN_OFFSET]
    screened, _ = build_candidates(panel, [position], pd.DatetimeIndex([baseline_date]))
    assert candidate in screened[position]

    first_return_date = panel.sessions[candidate - 4]
    screened_inside, _ = build_candidates(
        panel, [position], pd.DatetimeIndex([first_return_date])
    )
    assert candidate not in screened_inside[position]


def test_candidates_with_missing_prices_are_rejected(panel):
    position = session_position(panel.sessions, pd.Timestamp("2022-06-15"))
    broken = panel.stock_prices.copy()
    broken[position - 20] = np.nan
    damaged = make_panel(panel.sessions, broken, panel.benchmark_prices)
    candidates, diagnostics = build_candidates(damaged, [position], pd.DatetimeIndex([]))
    offsets = position - candidates[position]
    # Any window covering the hole is rejected: offsets 12 through 25.
    assert 20 not in offsets
    assert diagnostics.loc[0, "rejected_missing_data"] == (
        REQUIRED_MAX_OFFSET - REQUIRED_MIN_OFFSET + 1
    )


def test_candidates_near_the_start_of_history_are_rejected():
    sessions = trading_sessions("2021-01-04", "2021-06-30")
    prices = np.arange(1.0, len(sessions) + 1.0)
    panel = make_panel(sessions, prices, prices)
    position = 20  # too close to the beginning for 40-session lookbacks
    candidates, diagnostics = build_candidates(panel, [position], pd.DatetimeIndex([]))
    assert candidates[position].size < 31
    assert diagnostics.loc[0, "rejected_out_of_range"] > 0
    assert all(c >= 0 for c in candidates[position])


def test_simulation_is_reproducible_with_a_fixed_seed(panel):
    positions = [session_position(panel.sessions, pd.Timestamp(d))
                 for d in ("2021-02-01", "2021-05-03", "2021-08-02", "2021-11-01")]
    releases = pd.DatetimeIndex([panel.sessions[p] for p in positions])
    first = run_simulation(panel, positions, releases, seed=42, n_repetitions=25)
    second = run_simulation(panel, positions, releases, seed=42, n_repetitions=25)
    third = run_simulation(panel, positions, releases, seed=43, n_repetitions=25)

    assert np.array_equal(first.draws, second.draws)
    pd.testing.assert_frame_equal(first.per_rep_stats["announcement"],
                                  second.per_rep_stats["announcement"])
    assert not np.array_equal(first.draws, third.draws)


def test_simulation_uses_the_matched_actual_sample(panel):
    sessions = trading_sessions("2021-01-04", "2021-12-31")
    prices = 100.0 * np.power(1.001, np.arange(len(sessions)))
    small = make_panel(sessions, prices, prices)
    # Position 12 is too close to the start: even the nearest candidate (offset 10)
    # would need a baseline close five sessions before index 2.
    early = 12
    late = session_position(sessions, pd.Timestamp("2021-09-01"))
    simulation = run_simulation(small, [early, late], pd.DatetimeIndex([]),
                                seed=42, n_repetitions=10)
    assert late in simulation.matched_positions
    assert early not in simulation.matched_positions
    assert len(simulation.unmatched) == 1
    assert simulation.draws.shape == (10, 1)
    assert any("no eligible pseudo candidate" in n for n in simulation.notes)


def test_draws_only_ever_come_from_eligible_candidates(panel):
    positions = [session_position(panel.sessions, pd.Timestamp("2022-03-01")),
                 session_position(panel.sessions, pd.Timestamp("2022-09-01"))]
    simulation = run_simulation(panel, positions, pd.DatetimeIndex([]),
                                seed=1, n_repetitions=50)
    for column, position in enumerate(simulation.matched_positions):
        allowed = set(simulation.candidates[position].tolist())
        assert set(simulation.draws[:, column].tolist()) <= allowed
        offsets = position - simulation.draws[:, column]
        assert offsets.min() >= PSEUDO_MIN_OFFSET
        assert offsets.max() <= PSEUDO_MAX_OFFSET


def test_repetitions_do_not_increase_the_sample_size(panel):
    positions = [session_position(panel.sessions, pd.Timestamp("2022-03-01")),
                 session_position(panel.sessions, pd.Timestamp("2022-09-01"))]
    simulation = run_simulation(panel, positions, pd.DatetimeIndex([]),
                                seed=1, n_repetitions=100)
    per_rep = simulation.per_rep_stats["announcement"]
    assert len(per_rep) == 100
    assert (per_rep["n"] == len(simulation.matched_positions)).all()
    assert simulation.reuse["total_draws"] == 100 * len(simulation.matched_positions)
    assert simulation.reuse["mean_reuse_per_distinct_date"] > 1.0


def test_no_eligible_candidates_at_all_is_reported_not_crashed():
    sessions = trading_sessions("2021-01-04", "2021-03-31")
    prices = np.arange(1.0, len(sessions) + 1.0)
    panel = make_panel(sessions, prices, prices)
    simulation = run_simulation(panel, [10], pd.DatetimeIndex([]), seed=42,
                                n_repetitions=10)
    assert simulation.matched_positions == []
    assert simulation.draws is None
    assert any("No actual event has an eligible pseudo-date candidate" in n
               for n in simulation.notes)
