"""The stock universes and the cross-stock summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.config import DEFAULT_BENCHMARK, AnalysisConfig
from earnings_reversal.pipeline import AnalysisResult, Inputs, run_analysis
from earnings_reversal import summary as summary_module
from earnings_reversal.summary import (
    ABOVE,
    BELOW,
    SUMMARY_STATS,
    UNAVAILABLE,
    WITHIN,
    analyse_universe,
    cross_stock_overview,
    pooled_events,
    pooled_group_table,
    pseudo_median_gap,
    pseudo_position,
    stock_summary_row,
    universe_table,
    volatility_ratios,
)
from earnings_reversal.universe import (
    ADDED_40,
    ALL_TICKERS,
    FIRST_30,
    GROUPS,
    UNIVERSE,
    UNIVERSE_TICKERS,
    display_name,
    group_of,
    member,
)
from conftest import build_synthetic_inputs


def test_universe_has_thirty_distinct_stocks():
    assert len(UNIVERSE) == 30
    assert len(set(UNIVERSE_TICKERS)) == 30
    assert DEFAULT_BENCHMARK not in UNIVERSE_TICKERS
    assert all(m.name and m.sector for m in UNIVERSE)


def test_original_thirty_are_unchanged_and_first():
    assert FIRST_30.tickers == UNIVERSE_TICKERS
    assert FIRST_30.tickers[:3] == ("AAPL", "MSFT", "IBM")
    assert ALL_TICKERS[:30] == UNIVERSE_TICKERS
    assert GROUPS == (FIRST_30, ADDED_40)


def test_added_forty_are_distinct_from_the_first_thirty():
    added = ADDED_40.tickers
    assert len(added) == 40 and len(set(added)) == 40
    assert not set(added) & set(UNIVERSE_TICKERS)
    assert DEFAULT_BENCHMARK not in added
    assert len(ALL_TICKERS) == len(set(ALL_TICKERS)) == 70
    assert all(m.name and m.sector for m in ADDED_40.members)
    # Screened out during selection; see the universe module docstring.
    for excluded in ("DHR", "VTR", "HPQ", "MU", "F", "MRK", "COST", "CVX", "T"):
        assert excluded not in ALL_TICKERS


def test_group_labels_and_lookup():
    assert FIRST_30.label == "30-stock summary"
    assert ADDED_40.label == "40-stock summary"
    assert group_of("aapl") is FIRST_30
    assert group_of("NVDA") is ADDED_40
    assert group_of("NOTREAL") is None
    assert display_name("JNJ") == "JNJ \u2014 Johnson & Johnson (Health Care)"


def test_universe_lookup_helpers():
    assert member("aapl").name == "Apple"
    assert member("NOTREAL") is None
    assert display_name("MSFT").startswith("MSFT \u2014 Microsoft")
    assert display_name("NOTREAL") == "NOTREAL"


@pytest.mark.parametrize("actual,expected", [
    (-0.5, BELOW), (0.0, WITHIN), (0.5, ABOVE), (None, UNAVAILABLE),
    (float("nan"), UNAVAILABLE),
])
def test_pseudo_position_labels(actual, expected):
    assert pseudo_position(actual, -0.2, 0.2) == expected


def test_pseudo_position_without_a_range_is_unavailable():
    assert pseudo_position(0.1, None, None) == UNAVAILABLE


@pytest.fixture(scope="module")
def two_results():
    results = {}
    for ticker, seed in (("AAA", 1), ("BBB", 2)):
        results[ticker] = run_analysis(build_synthetic_inputs(ticker, price_seed=seed))
    return results


def test_summary_row_matches_the_single_stock_result(two_results):
    result = two_results["AAA"]
    row = stock_summary_row("AAA", result, "announcement")
    stats = result.stats_by_outcome["announcement"]
    assert row["status"] == "ok"
    assert row["events_included"] == len(result.included_events)
    assert row["slope"] == pytest.approx(stats.slope.value)
    assert row["correlation"] == pytest.approx(stats.correlation.value)
    assert row["negative_mean"] == pytest.approx(stats.negative_par.mean.value)
    assert row["positive_mean"] == pytest.approx(stats.positive_par.mean.value)
    assert row["negative_count"] + row["positive_count"] <= row["events_included"]


def test_summary_row_pseudo_columns_are_consistent(two_results):
    row = stock_summary_row("AAA", two_results["AAA"], "announcement")
    for stat in SUMMARY_STATS:
        p5, median, p95 = (row[f"{stat}_pseudo_p5"], row[f"{stat}_pseudo_median"],
                           row[f"{stat}_pseudo_p95"])
        assert p5 <= median <= p95
        assert row[f"{stat}_position"] == pseudo_position(row[f"{stat}_matched"], p5, p95)


def test_summary_row_for_a_failed_stock_carries_its_reason():
    config = AnalysisConfig(ticker="FAIL")
    failed = AnalysisResult(config=config, inputs=Inputs(config=config))
    failed.blocking_error = "download failed"
    row = stock_summary_row("FAIL", failed, "announcement")
    assert row["status"] == "download failed"
    assert row["slope"] is None
    assert row["slope_position"] == UNAVAILABLE
    assert stock_summary_row("NONE", None, "announcement")["status"] == "not analysed"


def test_overview_counts_only_successful_stocks(two_results):
    config = AnalysisConfig(ticker="FAIL")
    failed = AnalysisResult(config=config, inputs=Inputs(config=config))
    failed.blocking_error = "download failed"
    results = dict(two_results, FAIL=failed)

    table = universe_table(results, "announcement")
    assert list(table["ticker"]) == ["AAA", "BBB", "FAIL"]
    overview = cross_stock_overview(table).set_index("stat")
    slope = overview.loc["slope"]
    assert slope["Stocks with a value"] == 2
    assert slope["Stocks below zero"] + slope["Stocks above zero"] <= 2
    expected_median = float(np.median(table.loc[table["status"] == "ok", "slope"]))
    assert slope["Median across stocks"] == pytest.approx(expected_median)
    assert slope["Reversal side"] == BELOW
    assert overview.loc["negative_minus_positive", "Reversal side"] == ABOVE


def test_pooled_events_keep_every_stocks_events(two_results):
    events = pooled_events(two_results)
    expected = sum(len(r.included_events) for r in two_results.values())
    assert len(events) == expected
    assert set(events["ticker"]) == {"AAA", "BBB"}

    pooled = pooled_group_table(events, "announcement_adj").set_index("PAR group")
    neg = pooled.loc["Negative PAR"]
    pos = pooled.loc["Positive PAR"]
    assert neg["Events"] == int((events["par"] < 0).sum())
    assert neg["Mean outcome"] == pytest.approx(
        events.loc[events["par"] < 0, "announcement_adj"].mean()
    )
    assert pooled.loc["Negative minus positive (mean difference)", "Mean outcome"] == \
        pytest.approx(neg["Mean outcome"] - pos["Mean outcome"])


def test_pseudo_median_gap(two_results):
    table = universe_table(two_results, "announcement")
    gaps = pseudo_median_gap(table, "slope", "below").set_index("ticker")
    for ticker in ("AAA", "BBB"):
        row = table.set_index("ticker").loc[ticker]
        expected = row["slope_matched"] - row["slope_pseudo_median"]
        assert gaps.loc[ticker, "gap"] == pytest.approx(expected)
        assert gaps.loc[ticker, "favourable"] == (expected < 0)
    above = pseudo_median_gap(table, "slope", "above").set_index("ticker")
    assert (above["favourable"] != gaps["favourable"]).all()
    assert list(pseudo_median_gap(table, "slope", "below")["gap"]) == sorted(gaps["gap"])
    assert pseudo_median_gap(pd.DataFrame(), "slope", "below").empty


def test_volatility_ratio_compares_actual_and_pseudo_outcomes(two_results):
    ratios = volatility_ratios(two_results, "announcement")
    assert set(ratios.index) == {"AAA", "BBB"}
    result = two_results["AAA"]
    outcomes = result.panel.window_matrix()["announcement"]
    simulation = result.simulation
    expected = (np.nanstd(outcomes[simulation.matched_positions])
                / np.nanstd(outcomes[simulation.draws.ravel()]))
    assert ratios["AAA"] == pytest.approx(expected)
    assert volatility_ratios({"NONE": None}, "announcement").empty


def test_pooled_group_table_handles_no_events():
    assert pooled_group_table(pd.DataFrame(), "announcement_adj").empty
    assert pooled_events({}).empty


def test_analyse_universe_uses_shared_settings_and_isolates_failures(monkeypatch):
    seen = []

    def fake_load_inputs(config, cache, force_refresh=False):
        seen.append(config)
        if config.ticker == "BAD":
            raise RuntimeError("simulated outage")
        return build_synthetic_inputs(config.ticker, price_seed=len(seen))

    monkeypatch.setattr(summary_module, "load_inputs", fake_load_inputs)
    base = AnalysisConfig(ticker="IGNORED", benchmark="BENCH",
                          event_start="2021-01-01", event_end="2023-12-31",
                          seed=7, n_repetitions=10)
    progress_calls = []
    results = analyse_universe(
        base, ["AAA", "BAD", "CCC"], cache=None,
        excluded_by_ticker={"CCC": ("2022-01-27",)},
        progress=lambda i, n, t: progress_calls.append((i, n, t)),
    )

    assert list(results) == ["AAA", "BAD", "CCC"]
    assert [c.ticker for c in seen] == ["AAA", "BAD", "CCC"]
    assert all(c.seed == 7 and c.n_repetitions == 10 for c in seen)
    assert results["AAA"].blocking_error == ""
    assert "simulated outage" in results["BAD"].blocking_error
    assert len(results["CCC"].included_events) == len(results["AAA"].included_events) - 1
    assert progress_calls[-1] == (3, 3, "")
