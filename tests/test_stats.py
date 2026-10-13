"""Statistics: reversal definition, degenerate inputs and empty groups."""

from __future__ import annotations

import numpy as np
import pytest

from earnings_reversal.stats import fast_ols, reversal_stats, statsmodels_ols


def test_slope_and_correlation_on_a_known_sample():
    par = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
    outcome = -0.5 * par + 0.001
    stats = reversal_stats(par, outcome)
    assert stats.n == 5
    assert stats.slope.value == pytest.approx(-0.5)
    assert stats.intercept.value == pytest.approx(0.001)
    assert stats.r_squared.value == pytest.approx(1.0)
    assert stats.correlation.value == pytest.approx(-1.0)


def test_fast_ols_matches_statsmodels():
    rng = np.random.default_rng(5)
    par = rng.normal(0, 0.03, size=40)
    outcome = -0.4 * par + rng.normal(0, 0.02, size=40)
    slope, intercept, r_squared = fast_ols(par, outcome)
    fit = statsmodels_ols(par, outcome)
    assert slope == pytest.approx(float(fit.params[1]))
    assert intercept == pytest.approx(float(fit.params[0]))
    assert r_squared == pytest.approx(float(fit.rsquared))


def test_reversal_is_opposite_sign_product_below_zero():
    par = np.array([-0.02, 0.03, -0.01, 0.04])
    outcome = np.array([0.01, -0.02, -0.03, 0.05])  # reversals: 1st and 2nd
    stats = reversal_stats(par, outcome)
    assert stats.both_nonzero_count == 4
    assert stats.opposite_sign_count == 2
    assert stats.opposite_sign_frequency.value == pytest.approx(0.5)


def test_zero_values_are_reported_separately_and_never_counted():
    par = np.array([-0.02, 0.0, 0.03, 0.01])
    outcome = np.array([0.01, 0.05, 0.0, -0.02])
    stats = reversal_stats(par, outcome)
    assert stats.zero_par_count == 1
    assert stats.zero_outcome_count == 1
    assert stats.both_nonzero_count == 2       # only the 1st and 4th observations
    assert stats.opposite_sign_count == 2
    assert stats.opposite_sign_frequency.value == pytest.approx(1.0)
    assert stats.zero_par.count == 1


def test_negative_minus_positive_mean_difference():
    par = np.array([-0.02, -0.01, 0.01, 0.02])
    outcome = np.array([0.03, 0.01, -0.01, -0.03])
    stats = reversal_stats(par, outcome)
    assert stats.negative_par.count == 2
    assert stats.positive_par.count == 2
    assert stats.negative_par.mean.value == pytest.approx(0.02)
    assert stats.positive_par.mean.value == pytest.approx(-0.02)
    assert stats.negative_minus_positive_mean.value == pytest.approx(0.04)


def test_empty_group_returns_not_available_with_a_reason():
    par = np.array([0.01, 0.02, 0.03])       # no negative-PAR events
    outcome = np.array([0.01, -0.01, 0.02])
    stats = reversal_stats(par, outcome)
    assert stats.negative_par.count == 0
    assert not stats.negative_par.mean.available
    assert "No events" in stats.negative_par.mean.reason
    assert not stats.negative_minus_positive_mean.available
    assert "Empty negative-PAR group" in stats.negative_minus_positive_mean.reason
    assert stats.negative_minus_positive_mean.format() == "not available"


def test_insufficient_observations_block_correlation_and_regression():
    stats = reversal_stats(np.array([0.01, -0.02]), np.array([0.0, 0.01]))
    assert stats.n == 2
    assert not stats.correlation.available
    assert "at least 3 required" in stats.correlation.reason
    assert not stats.slope.available
    assert stats.slope.format() == "not available"


def test_constant_par_is_not_identified():
    par = np.full(6, 0.02)
    outcome = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.04])
    stats = reversal_stats(par, outcome)
    assert not stats.correlation.available
    assert "constant" in stats.correlation.reason
    assert not stats.slope.available
    assert "not identified" in stats.slope.reason


def test_constant_outcome_blocks_correlation_but_not_the_slope():
    par = np.array([-0.02, -0.01, 0.01, 0.02])
    outcome = np.zeros(4)
    stats = reversal_stats(par, outcome)
    assert not stats.correlation.available
    assert stats.slope.value == pytest.approx(0.0)
    assert stats.zero_outcome_count == 4
    assert not stats.opposite_sign_frequency.available
    assert "undefined" in stats.opposite_sign_frequency.reason


def test_empty_sample_is_handled_gracefully():
    stats = reversal_stats(np.array([]), np.array([]))
    assert stats.n == 0
    assert not stats.correlation.available
    assert not stats.slope.available
    assert not stats.negative_minus_positive_mean.available
    assert stats.correlation.format() == "not available"


def test_missing_values_are_dropped_and_reported():
    par = np.array([-0.02, np.nan, 0.03, 0.01])
    outcome = np.array([0.01, 0.05, np.nan, -0.02])
    stats = reversal_stats(par, outcome)
    assert stats.n == 2
    assert any("dropped for missing values" in note for note in stats.notes)


def test_small_sample_is_flagged():
    rng = np.random.default_rng(1)
    stats = reversal_stats(rng.normal(size=12), rng.normal(size=12))
    assert stats.small_sample
    assert any("Small sample" in note for note in stats.notes)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        reversal_stats(np.array([0.1, 0.2]), np.array([0.1]))
