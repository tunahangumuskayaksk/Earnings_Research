"""Return endpoints, compounding and missing-data behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.config import (
    ANNOUNCEMENT_WINDOW,
    LATER_WINDOW,
    PAR_WINDOW,
    SUBSEQUENT_WINDOW,
    Window,
)
from earnings_reversal.returns import (
    compound_daily_returns,
    evaluate_window,
    window_daily_returns,
    window_return,
    window_returns_vectorized,
)
from conftest import make_panel


def test_window_uses_close_at_a_minus_one_and_b():
    """[a,b] starts from the close at a-1 and ends at the close at b."""
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    position = 5
    window = Window("w", "W", -2, 1)
    # Start price is the close at offset -3 (index 2), end price at offset +1 (index 6).
    expected = prices[6] / prices[2] - 1.0
    assert window_return(prices, position, window) == pytest.approx(expected)


def test_par_window_endpoints_are_minus_five_and_minus_two():
    prices = np.arange(1.0, 31.0)
    position = 10
    result = window_return(prices, position, PAR_WINDOW)
    assert PAR_WINDOW.baseline_offset == -5
    expected = prices[position - 2] / prices[position - 5] - 1.0
    assert result == pytest.approx(expected)


def test_par_includes_exactly_three_daily_returns():
    prices = np.arange(1.0, 31.0)
    daily = window_daily_returns(prices, 10, PAR_WINDOW)
    assert daily.size == 3  # t=-4, -3, -2


@pytest.mark.parametrize("window", [PAR_WINDOW, ANNOUNCEMENT_WINDOW,
                                    SUBSEQUENT_WINDOW, LATER_WINDOW])
def test_compounded_equals_product_of_daily_returns(window):
    rng = np.random.default_rng(7)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=60))
    position = 30
    daily = window_daily_returns(prices, position, window)
    assert daily.size == window.end - window.start + 1
    assert window_return(prices, position, window) == pytest.approx(
        float(np.prod(1.0 + daily) - 1.0)
    )


def test_market_adjusted_is_difference_of_compounded_returns(geometric_panel):
    """Not the compounding of daily stock-minus-benchmark differences."""
    panel = geometric_panel
    position = 30
    result = evaluate_window(panel.sessions, panel.stock_prices,
                             panel.benchmark_prices, position, PAR_WINDOW)
    stock = 1.01 ** 3 - 1.0
    bench = 1.005 ** 3 - 1.0
    assert result.stock_return == pytest.approx(stock)
    assert result.benchmark_return == pytest.approx(bench)
    assert result.market_adjusted_return == pytest.approx(stock - bench)

    wrong = np.prod(1.0 + np.full(3, 0.01 - 0.005)) - 1.0
    assert result.market_adjusted_return != pytest.approx(wrong)


def test_vectorized_matches_scalar_everywhere():
    rng = np.random.default_rng(11)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=80))
    vectorized = window_returns_vectorized(prices, ANNOUNCEMENT_WINDOW)
    for position in range(len(prices)):
        scalar = window_return(prices, position, ANNOUNCEMENT_WINDOW)
        if np.isnan(scalar):
            assert np.isnan(vectorized[position])
        else:
            assert vectorized[position] == pytest.approx(scalar)


def test_missing_price_inside_window_yields_nan_not_a_bridge():
    """A gap must not be bridged by skipping over it."""
    prices = np.array([10.0, 11.0, 12.0, np.nan, 14.0, 15.0, 16.0, 17.0, 18.0])
    position = 5
    window = Window("w", "W", -3, 0)
    assert np.isnan(window_return(prices, position, window))
    assert np.isnan(window_returns_vectorized(prices, window)[position])
    # The naive endpoint ratio would have silently produced a number.
    assert np.isfinite(prices[position] / prices[position - 4] - 1.0)


def test_no_forward_fill_through_missing_observations():
    prices = np.array([100.0, np.nan, 102.0, 103.0, 104.0, 105.0])
    daily = window_daily_returns(prices, 4, Window("w", "W", -3, 0))
    assert np.isnan(daily).sum() == 2  # the gap affects both adjacent returns
    assert np.isnan(compound_daily_returns(daily))


def test_window_outside_loaded_history_is_invalid():
    prices = np.arange(1.0, 11.0)
    assert np.isnan(window_return(prices, 1, PAR_WINDOW))
    assert np.isnan(window_return(prices, 9, LATER_WINDOW))


def test_evaluate_window_reports_dates_and_prices(geometric_panel):
    panel = geometric_panel
    position = 30
    result = evaluate_window(panel.sessions, panel.stock_prices,
                             panel.benchmark_prices, position, ANNOUNCEMENT_WINDOW)
    assert result.start_price_date == panel.sessions[position - 2]
    assert result.end_price_date == panel.sessions[position + 1]
    assert result.first_return_date == panel.sessions[position - 1]
    assert result.stock_start_price == pytest.approx(panel.stock_prices[position - 2])
    assert result.stock_end_price == pytest.approx(panel.stock_prices[position + 1])
    assert result.valid


def test_missing_benchmark_invalidates_the_event(geometric_panel):
    panel = geometric_panel
    bench = panel.benchmark_prices.copy()
    bench[30] = np.nan
    broken = make_panel(panel.sessions, panel.stock_prices, bench)
    result = evaluate_window(broken.sessions, broken.stock_prices,
                             broken.benchmark_prices, 30, ANNOUNCEMENT_WINDOW)
    assert not result.valid
    assert "benchmark" in result.reason


def test_adjusted_prices_absorb_a_split_with_no_double_adjustment():
    """A 4-for-1 split effective at index 3, priced three ways."""
    split_factor = 4.0
    raw_close = np.array([100.0, 101.0, 102.0, 25.75, 26.0, 26.5])
    # Correct: only the pre-split closes are divided once.
    adj_close = np.array([25.0, 25.25, 25.5, 25.75, 26.0, 26.5])
    # The double-adjustment bug: pre-split closes divided a second time.
    double_adjusted = np.array([6.25, 6.3125, 6.375, 25.75, 26.0, 26.5])

    window = Window("w", "W", -1, 0)   # spans the split: sessions 2->3
    position = 3

    correct = window_return(adj_close, position, window)
    assert correct == pytest.approx(25.75 / 25.25 - 1.0)
    assert abs(correct) < 0.05

    # The unadjusted series shows an artificial crash across the split.
    assert window_return(raw_close, position, window) < -0.5
    # Adjusting an already-adjusted series produces an equally artificial spike.
    assert window_return(double_adjusted, position, window) > 1.0
    assert double_adjusted[0] == pytest.approx(raw_close[0] / split_factor ** 2)


def test_post_split_window_is_unaffected_by_the_adjustment_choice():
    """Away from the split the two series give the same return, as they must."""
    raw_close = np.array([100.0, 101.0, 102.0, 25.75, 26.0, 26.5])
    adj_close = np.array([25.0, 25.25, 25.5, 25.75, 26.0, 26.5])
    window = Window("w", "W", -1, 0)
    position = 5
    assert window_return(adj_close, position, window) == pytest.approx(
        window_return(raw_close, position, window)
    )


def test_dividend_adjusted_return_includes_the_distribution():
    """Adjusted close carries the dividend; the raw close understates the return."""
    raw_close = np.array([100.0, 100.0, 99.0, 99.5])  # $1 dividend goes ex on day 2
    adj_close = np.array([99.0, 99.0, 99.0, 99.5])
    window = Window("w", "W", -1, 0)
    position = 3
    assert window_return(adj_close, position, window) > window_return(raw_close, position, window)


def test_configured_adjustment_settings_prevent_double_adjustment():
    from earnings_reversal.prices import ADJUSTMENT_SETTINGS

    assert ADJUSTMENT_SETTINGS["auto_adjust"] is False
    assert ADJUSTMENT_SETTINGS["actions"] is True
    assert ADJUSTMENT_SETTINGS["back_adjust"] is False
    assert "Adj Close" in ADJUSTMENT_SETTINGS["return_basis"]
