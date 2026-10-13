"""End-to-end computation over synthetic inputs, with no network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.config import AnalysisConfig, ALL_WINDOWS, PAR_WINDOW
from earnings_reversal.earnings import normalise_yahoo_earnings, parse_manual_csv
from earnings_reversal.pipeline import Inputs, matched_sample_note, run_analysis
from conftest import NOW, SYNTHETIC_RELEASES as RELEASES, build_synthetic_inputs


@pytest.fixture
def synthetic_inputs():
    return build_synthetic_inputs()


def test_pipeline_produces_a_complete_audit_table(synthetic_inputs):
    result = run_analysis(synthetic_inputs)
    assert result.blocking_error == ""
    assert len(result.included_events) == len(RELEASES)
    assert len(result.excluded_events) == 0

    required = [
        "ticker", "release_date", "session_class", "par", "announcement_adj",
        "subsequent_adj", "later_adj", "par_stock", "par_benchmark",
        "announcement_stock", "announcement_benchmark", "quality_flags",
    ]
    for column in required:
        assert column in result.included_events.columns
    for window in ALL_WINDOWS:
        for suffix in ("first_return_date", "start_price_date", "end_price_date",
                       "start_price", "end_price"):
            assert f"{window.key}_{suffix}" in result.included_events.columns
    assert result.included_events["par"].notna().all()
    assert (result.included_events["session_class"] == "after_close").all()


def test_market_adjusted_column_equals_stock_minus_benchmark(synthetic_inputs):
    result = run_analysis(synthetic_inputs)
    events = result.included_events
    np.testing.assert_allclose(
        events["announcement_adj"],
        events["announcement_stock"] - events["announcement_benchmark"],
        rtol=1e-12,
    )


def test_audit_row_can_be_reproduced_by_hand(synthetic_inputs):
    """Recompute one event's PAR straight from the price series."""
    result = run_analysis(synthetic_inputs)
    event = result.included_events.iloc[3]
    panel = result.panel
    position = int(event["session_position"])

    start_index = panel.sessions.get_loc(pd.Timestamp(event["par_start_price_date"]))
    end_index = panel.sessions.get_loc(pd.Timestamp(event["par_end_price_date"]))
    assert start_index == position - 5
    assert end_index == position - 2

    stock = panel.stock_prices[end_index] / panel.stock_prices[start_index] - 1.0
    bench = panel.benchmark_prices[end_index] / panel.benchmark_prices[start_index] - 1.0
    assert event["par"] == pytest.approx(stock - bench)
    assert event["par_stock"] == pytest.approx(stock)
    assert event["par_start_price"] == pytest.approx(panel.stock_prices[start_index])


def test_window_matrix_agrees_with_the_event_table(synthetic_inputs):
    """The vectorised path used by the Monte Carlo matches the audited path."""
    result = run_analysis(synthetic_inputs)
    matrix = result.panel.window_matrix()
    for _, event in result.included_events.iterrows():
        position = int(event["session_position"])
        assert matrix[PAR_WINDOW.key][position] == pytest.approx(event["par"])
        assert matrix["announcement"][position] == pytest.approx(event["announcement_adj"])
        assert matrix["subsequent"][position] == pytest.approx(event["subsequent_adj"])
        assert matrix["later"][position] == pytest.approx(event["later_adj"])


def test_pipeline_reproducible_with_a_fixed_seed(synthetic_inputs):
    first = run_analysis(synthetic_inputs)
    second = run_analysis(synthetic_inputs)
    assert np.array_equal(first.simulation.draws, second.simulation.draws)
    pd.testing.assert_frame_equal(first.simulation.per_rep_stats["announcement"],
                                  second.simulation.per_rep_stats["announcement"])

    other = run_analysis(synthetic_inputs,
                         config=AnalysisConfig(**{**synthetic_inputs.config.__dict__,
                                                  "seed": 7}))
    assert not np.array_equal(first.simulation.draws, other.simulation.draws)


def test_matched_actual_sample_matches_the_pseudo_sample_size(synthetic_inputs):
    result = run_analysis(synthetic_inputs)
    simulation = result.simulation
    matched = len(simulation.matched_positions)
    assert matched > 0
    assert simulation.draws.shape == (result.config.n_repetitions, matched)
    for key, stats in result.matched_actual_stats.items():
        assert stats.n == matched
        assert (simulation.per_rep_stats[key]["n"] == matched).all()
    assert len(result.matched_event_dates) == matched
    assert "matched" in matched_sample_note(result)


def test_quarterly_releases_never_contaminate_their_own_pseudo_windows(synthetic_inputs):
    """With ~63 sessions between releases, the 10-40 session search window is clean."""
    result = run_analysis(synthetic_inputs)
    diagnostics = result.simulation.diagnostics
    assert (diagnostics["eligible_candidates"] > 0).all()
    assert int(diagnostics["rejected_earnings_contamination"].sum()) == 0


def test_contamination_screen_bites_when_releases_are_close_together(synthetic_inputs):
    """An extra release 20 sessions before an event removes candidates around it."""
    result = run_analysis(synthetic_inputs)
    panel = result.panel
    position = int(result.included_events.iloc[5]["session_position"])
    intruder = panel.sessions[position - 20]

    from earnings_reversal.pseudo import build_candidates

    clean, _ = build_candidates(panel, [position], pd.DatetimeIndex([]))
    screened, diagnostics = build_candidates(panel, [position],
                                             pd.DatetimeIndex([intruder]))
    assert screened[position].size < clean[position].size
    assert diagnostics.loc[0, "rejected_earnings_contamination"] > 0


def test_pipeline_without_prices_reports_a_blocking_error(synthetic_inputs):
    broken = Inputs(config=synthetic_inputs.config, stock=None, benchmark=None,
                    earnings=synthetic_inputs.earnings,
                    price_error="simulated download failure")
    result = run_analysis(broken)
    assert not result.has_events
    assert "simulated download failure" in result.blocking_error
    assert result.panel is None
    assert result.simulation is None


def test_pipeline_without_earnings_reports_a_blocking_error(synthetic_inputs):
    bare = Inputs(config=synthetic_inputs.config, stock=synthetic_inputs.stock,
                  benchmark=synthetic_inputs.benchmark, earnings=None,
                  earnings_error="simulated earnings failure")
    result = run_analysis(bare)
    assert not result.has_events
    assert "No completed earnings records" in result.blocking_error
    assert "simulated earnings failure" in " ".join(result.messages)
    # No statistics are invented in place of the missing events.
    assert result.stats_by_outcome == {}


def test_user_exclusion_flows_through_to_the_statistics(synthetic_inputs):
    base = run_analysis(synthetic_inputs)
    excluded_date = pd.Timestamp(base.included_events.iloc[0]["release_date"])
    config = AnalysisConfig(
        **{**synthetic_inputs.config.__dict__,
           "excluded_release_dates": (excluded_date.date().isoformat(),)}
    )
    trimmed = run_analysis(synthetic_inputs, config=config)
    assert len(trimmed.included_events) == len(base.included_events) - 1
    assert excluded_date not in set(trimmed.included_events["release_date"])
    assert trimmed.stats_by_outcome["announcement"].n == base.stats_by_outcome["announcement"].n - 1


def test_release_on_a_non_trading_day_is_excluded_by_the_pipeline(synthetic_inputs):
    saturday = pd.DatetimeIndex([pd.Timestamp("2022-03-05 16:30")]).tz_localize(
        "America/New_York"
    )
    extra = pd.DataFrame({"EPS Estimate": [1.0], "Reported EPS": [1.1],
                          "Surprise(%)": [2.0]}, index=saturday)
    merged = pd.concat([
        synthetic_inputs.earnings.records,
        normalise_yahoo_earnings(extra, "TEST", NOW),
    ], ignore_index=True)
    synthetic_inputs.earnings.records = merged

    result = run_analysis(synthetic_inputs)
    assert len(result.excluded_events) == 1
    assert result.excluded_events.loc[0, "reason"] == "Release date is not a trading session"
    assert len(result.included_events) == len(RELEASES)


def test_manual_rows_only_apply_to_their_own_ticker(synthetic_inputs):
    """An upload for one stock must not change another stock's events."""
    csv = (
        "ticker,release_date,release_time,timezone,source_url,verified\n"
        "OTHER,2022-04-29,16:30,America/New_York,https://example.com/other,true\n"
        "TEST,2022-01-28,16:30,America/New_York,https://example.com/test,true\n"
    )
    manual, errors = parse_manual_csv(csv, NOW)
    assert errors == []

    result = run_analysis(synthetic_inputs, manual)
    dates = set(result.included_events["release_date"].dt.date.astype(str))
    # The TEST row overrides its own 2022-01-27 release...
    assert "2022-01-28" in dates
    assert "2022-01-27" not in dates
    # ...while the OTHER row leaves TEST's 2022-04-28 release untouched.
    assert "2022-04-28" in dates
    assert "2022-04-29" not in dates
    assert not (result.reconciled["ticker"] == "OTHER").any()
