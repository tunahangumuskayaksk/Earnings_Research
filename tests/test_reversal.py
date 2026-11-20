"""PAR uses the IMP anchor and ends on the decision close."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reversal import PAR_END_OFFSET, assign_folds, attach_par


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=30)
    prices = 100.0 * (1.01 ** np.arange(30))
    rows = []
    for i, day in enumerate(dates):
        rows.append({
            "TickerId": 1,
            "Date": day,
            "LastPrice": prices[i],
            "BenchmarkAdjClose": 100.0,
            "EventDate": pd.NaT,
            "MarketTimeId": np.nan,
            "Symbol": "AAA",
        })
    frame = pd.DataFrame(rows)
    # Before-open on session 20. After-close on session 12 (anchor becomes 13).
    frame.loc[20, "EventDate"] = dates[20]
    frame.loc[20, "MarketTimeId"] = 1
    frame.loc[12, "EventDate"] = dates[12]
    frame.loc[12, "MarketTimeId"] = 2
    return frame, prices, dates


def test_par_ends_on_the_decision_session_for_both_release_times():
    frame, prices, dates = _panel()
    par = attach_par(frame).set_index("EventDate")

    bmo = par.loc[pd.Timestamp(dates[20])]
    # Anchor 20, PAR close[15] -> close[18]. Decision is anchor-2 = 18.
    assert bmo["par_end"] == pd.Timestamp(dates[20 + PAR_END_OFFSET])
    assert bmo["par"] == prices[18] / prices[15] - 1
    assert bmo["par_bench"] == 0.0
    assert bmo["par_mkt"] == bmo["par"]

    amc = par.loc[pd.Timestamp(dates[12])]
    # After close: anchor is the next session, 13. PAR ends at 11.
    assert amc["par_end"] == pd.Timestamp(dates[11])
    assert amc["par"] == prices[11] / prices[8] - 1


def test_folds_follow_the_imp_cuts():
    events = pd.DataFrame({
        "DecisionDate": pd.to_datetime(["2022-06-01", "2023-06-01", "2024-06-01"]),
        "TickerId": [1, 1, 1],
    })
    folded = assign_folds(events)
    assert list(folded.sort_values("DecisionDate")["fold"]) == ["train", "val", "test"]
