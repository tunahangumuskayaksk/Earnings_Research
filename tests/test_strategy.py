"""PAR quintile strategy: thresholds, timing, performance and pseudo comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.config import ANNOUNCEMENT_WINDOW, PAR_WINDOW
from earnings_reversal.pipeline import run_analysis
from earnings_reversal.returns import window_returns_vectorized
from earnings_reversal.strategy import (
    INSUFFICIENT,
    MIN_PRIOR_EVENTS,
    QUINTILE,
    SIGN_ALL,
    SIGN_SAME,
    group_spread,
    performance,
    pseudo_comparison,
    quintile_direction,
    sign_direction,
    threshold_lookup,
)
from earnings_reversal.summary import strategy_universe_table
from conftest import build_long_synthetic_inputs, build_synthetic_inputs


def test_thresholds_need_twenty_prior_values_and_use_only_those():
    values = np.arange(30, dtype=float)
    low, high = threshold_lookup(values)
    assert len(low) == 31
    assert np.isnan(low[:MIN_PRIOR_EVENTS]).all()
    assert low[20] == pytest.approx(np.percentile(values[:20], 20))
    assert high[20] == pytest.approx(np.percentile(values[:20], 80))
    assert high[25] == pytest.approx(np.percentile(values[:25], 80))


def test_quintile_membership_sets_direction_whatever_the_threshold_sign():
    # Both thresholds positive: a small positive PAR is still Q1, so long.
    assert quintile_direction(np.array([0.005]), np.array([0.01]), np.array([0.03]))[0] == 1
    # Both thresholds negative: a small negative PAR is still Q5, so short.
    assert quintile_direction(np.array([-0.005]), np.array([-0.03]), np.array([-0.01]))[0] == -1
    # Inside the band, or on a threshold exactly, there is no trade.
    assert quintile_direction(np.array([0.0, 0.01]), np.array([-0.01, 0.01]),
                              np.array([0.01, 0.02])).tolist() == [0, 0]
    # Missing thresholds (insufficient history) never trade.
    assert quintile_direction(np.array([-1.0]), np.array([np.nan]), np.array([np.nan]))[0] == 0


def test_sign_direction_matches_the_negative_positive_split():
    assert sign_direction(np.array([-0.02, 0.0, 0.03])).tolist() == [1, 0, -1]


def test_performance_by_hand():
    direction = np.array([1, -1, 0, 1])
    raw = np.array([0.02, 0.01, 0.50, -0.03])
    adjusted = np.array([0.01, -0.02, 0.30, 0.00])
    stats = performance(direction, raw, adjusted)
    assert stats["trades"] == 3 and stats["longs"] == 2 and stats["shorts"] == 1
    assert stats["mean_raw"] == pytest.approx((0.02 - 0.01 - 0.03) / 3)
    assert stats["mean_adj"] == pytest.approx((0.01 + 0.02 + 0.00) / 3)
    assert stats["win_raw"] == pytest.approx(1 / 3)
    assert stats["win_adj"] == pytest.approx(2 / 3)  # a zero return is not a win
    assert stats["pnl_raw"] == pytest.approx(-0.02)
    assert stats["pnl_adj"] == pytest.approx(0.03)


def test_group_spread_is_not_the_average_trade_return():
    direction = np.array([1, 1, 1, -1])
    outcome = np.array([0.01, 0.01, 0.01, -0.05])
    spread = group_spread(direction, outcome)
    assert spread == pytest.approx(0.06)
    average_trade = performance(direction, outcome, outcome)["mean_adj"]
    assert average_trade == pytest.approx((0.03 + 0.05) / 4)
    assert spread != pytest.approx(average_trade)
    assert group_spread(np.array([1, 0]), np.array([0.1, 0.2])) is None


def test_no_trades_returns_empty_statistics():
    stats = performance(np.array([0, 0]), np.array([0.1, 0.2]), np.array([0.1, 0.2]))
    assert stats["trades"] == 0
    assert stats["mean_adj"] is None and stats["win_adj"] is None
    assert stats["pnl_adj"] == 0.0


@pytest.fixture(scope="module")
def long_result():
    return run_analysis(build_long_synthetic_inputs())


def test_events_without_twenty_prior_events_are_skipped(long_result):
    trades = long_result.quintile.trades
    assert len(trades) == 32
    early = trades.iloc[:MIN_PRIOR_EVENTS]
    assert (early["position"] == INSUFFICIENT).all()
    assert (early["direction"] == 0).all()
    assert early["threshold_p20"].isna().all()
    assert trades.iloc[MIN_PRIOR_EVENTS:]["eligible"].all()
    assert long_result.quintile.eligible_events == 32 - MIN_PRIOR_EVENTS


def test_thresholds_exclude_the_current_event_and_later_events(long_result):
    trades = long_result.quintile.trades
    par = trades["par"].to_numpy()
    for i in range(MIN_PRIOR_EVENTS, len(trades)):
        earlier = par[:i]
        assert trades.loc[i, "prior_events"] == i
        assert trades.loc[i, "threshold_p20"] == pytest.approx(np.percentile(earlier, 20))
        assert trades.loc[i, "threshold_p80"] == pytest.approx(np.percentile(earlier, 80))
        expected = 1 if par[i] < np.percentile(earlier, 20) else (
            -1 if par[i] > np.percentile(earlier, 80) else 0)
        assert trades.loc[i, "direction"] == expected


def test_signal_inputs_are_complete_before_the_entry_close(long_result):
    strategy = long_result.quintile
    traded = strategy.trades[strategy.trades["direction"] != 0]
    assert len(traded) > 0
    assert strategy.availability_violations == 0
    assert (traded["latest_threshold_input"] < traded["signal_and_entry_close"]).all()
    # The entry close is the PAR end and the announcement window's start price.
    events = long_result.included_events.set_index("release_date")
    for _, row in traded.iterrows():
        event = events.loc[row["release_date"]]
        assert row["signal_and_entry_close"] == event["par_end_price_date"]
        assert row["signal_and_entry_close"] == event["announcement_start_price_date"]


def test_signed_returns_follow_the_position(long_result):
    trades = long_result.quintile.trades
    traded = trades[trades["direction"] != 0]
    assert np.allclose(traded["signed_raw"], traded["direction"] * traded["stock_return"])
    assert np.allclose(traded["signed_adj"],
                       traded["direction"] * (traded["stock_return"] - traded["benchmark_return"]))
    assert trades.loc[trades["direction"] == 0, "signed_adj"].isna().all()
    perf = long_result.quintile.performance[QUINTILE]
    assert perf["trades"] == len(traded)
    assert perf["mean_adj"] == pytest.approx(traded["signed_adj"].mean())
    assert perf["pnl_raw"] == pytest.approx(traded["signed_raw"].sum())


def test_sign_strategy_reproduces_the_existing_split(long_result):
    perf = long_result.quintile.performance
    stats = long_result.stats_by_outcome["announcement"]
    assert perf[SIGN_ALL]["spread_adj"] == pytest.approx(
        stats.negative_minus_positive_mean.value)
    assert perf[SIGN_ALL]["longs"] == stats.negative_par.count
    assert perf[SIGN_ALL]["shorts"] == stats.positive_par.count
    assert perf[SIGN_SAME]["trades"] <= long_result.quintile.eligible_events


def test_pseudo_thresholds_use_only_actual_events_before_each_fake_date(long_result):
    strategy = long_result.quintile
    simulation = long_result.simulation
    panel = long_result.panel
    events = long_result.included_events.sort_values("session_position")
    pool_pos = events["session_position"].to_numpy()
    pool_par = events["par"].to_numpy()
    matrix = panel.window_matrix()
    raw_matrix = window_returns_vectorized(panel.stock_prices, ANNOUNCEMENT_WINDOW)

    for rep in (0, 7, len(simulation.draws) - 1):
        direction = []
        for p in simulation.draws[rep]:
            earlier = pool_par[pool_pos < p]
            if earlier.size < MIN_PRIOR_EVENTS:
                direction.append(0)
                continue
            lo, hi = np.percentile(earlier, [20, 80])
            par = matrix[PAR_WINDOW.key][p]
            direction.append(1 if par < lo else (-1 if par > hi else 0))
        direction = np.array(direction)
        expected = performance(direction, raw_matrix[simulation.draws[rep]],
                               matrix["announcement"][simulation.draws[rep]])
        got = strategy.pseudo_reps[QUINTILE].iloc[rep]
        assert got["trades"] == expected["trades"]
        if expected["trades"]:
            assert got["mean_adj"] == pytest.approx(expected["mean_adj"])
            assert got["mean_raw"] == pytest.approx(expected["mean_raw"])
            assert got["pnl_adj"] == pytest.approx(expected["pnl_adj"])


def test_pseudo_counts_and_comparison_table(long_result):
    strategy = long_result.quintile
    n_reps = long_result.config.n_repetitions
    assert len(strategy.pseudo_reps[QUINTILE]) == n_reps
    assert len(strategy.pseudo_eligible) == n_reps
    assert strategy.matched_events == len(long_result.simulation.matched_positions)
    assert strategy.matched_eligible_events <= strategy.matched_events
    assert (strategy.pseudo_eligible <= strategy.matched_events).all()

    table = pseudo_comparison(strategy, QUINTILE).set_index("stat")
    reps = strategy.pseudo_reps[QUINTILE]["mean_adj"].dropna()
    assert table.loc["mean_adj", "Pseudo median"] == pytest.approx(reps.median())
    assert table.loc["mean_adj", "Valid repetitions"] == len(reps)
    assert table.loc["trades", "Actual (matched sample)"] == \
        strategy.matched_performance[QUINTILE]["trades"]
    assert "Group spread" in table.loc["spread_adj", "Statistic"]


def test_short_history_never_trades_and_explains_why():
    result = run_analysis(build_synthetic_inputs())
    strategy = result.quintile
    assert strategy.eligible_events == 0
    assert strategy.performance[QUINTILE]["trades"] == 0
    assert any("never trades" in note for note in strategy.notes)
    assert (strategy.pseudo_reps[QUINTILE]["trades"] == 0).all()


def test_strategy_universe_table(long_result):
    table = strategy_universe_table({"LONG": long_result, "NONE": None})
    ok = table.set_index("ticker").loc["LONG"]
    assert ok["status"] == "ok"
    assert ok["trades"] == long_result.quintile.performance[QUINTILE]["trades"]
    assert ok["eligible_events"] == 32 - MIN_PRIOR_EVENTS
    assert table.set_index("ticker").loc["NONE", "status"] == "not analysed"

    trades = long_result.quintile.trades
    q1 = trades.loc[trades["direction"] == 1, "adjusted_return"]
    q5 = trades.loc[trades["direction"] == -1, "adjusted_return"]
    assert ok["q1_count"] == len(q1) and ok["q5_count"] == len(q5)
    if len(q1) and len(q5):
        assert ok["q1_mean_adj"] - ok["q5_mean_adj"] == pytest.approx(
            long_result.quintile.performance[QUINTILE]["spread_adj"])
    comparison = pseudo_comparison(long_result.quintile, QUINTILE).set_index("stat")
    for stat in ("mean_raw", "win_adj", "pnl_raw", "spread_adj", "trades"):
        assert ok[f"{stat}_pseudo_median"] == pytest.approx(
            comparison.loc[stat, "Pseudo median"], nan_ok=True)
