"""Score the pre-announcement reversal on the IMP panel and splits.

Reuses ``script/run_feature.py``'s loaders (``load_*``, ``merge_panel``,
``build_aligned_panel``, ``build_event_table``). Does not change the ML
feature list. Writes ``output/reversal/``.

    python script/run_reversal.py
    python script/run_reversal.py --with-lin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.default import (
    BENCHMARK_SOURCE,
    DATA_DIR,
    MARKET_TICKER,
    PROJECT_ROOT,
    TARGET,
    TRAIN_END,
    VAL_END,
)
from src.features import build_aligned_panel, build_event_table
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.reversal import (
    assign_folds,
    fit_positioning_ols,
    fit_reversal_ols,
    par_outcome_correlation,
    strategy_table,
    with_par,
    write_reversal_outputs,
)


def build_events() -> pd.DataFrame:
    stock = load_stock(DATA_DIR)
    earnings = load_earnings(DATA_DIR)
    sentiment = load_sentiment(DATA_DIR)
    df_all = merge_panel(stock, earnings, sentiment)
    df_align = build_aligned_panel(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_DIR,
        benchmark_source=BENCHMARK_SOURCE,
    )
    events = build_event_table(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_DIR,
        df_align=df_align,
        benchmark_source=BENCHMARK_SOURCE,
    )
    events = with_par(events, df_align)
    events = assign_folds(events)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-lin",
        action="store_true",
        help="Also fit IMP's linear family (ols / ridge / lasso) into output/lin.",
    )
    args = parser.parse_args()

    events = build_events()
    reversal_folds, reversal_summary, coef = fit_reversal_ols(events)
    positioning_folds, positioning_summary = fit_positioning_ols(events)
    folds = pd.concat([reversal_folds, positioning_folds], ignore_index=True)
    summaries = pd.DataFrame([reversal_summary, positioning_summary])
    strategies = strategy_table(events)
    correlations = par_outcome_correlation(events)

    out = PROJECT_ROOT / "output" / "reversal"
    write_reversal_outputs(out, events, folds, summaries, strategies, correlations)

    print(f"Benchmark: {BENCHMARK_SOURCE}    target: {TARGET}")
    print(f"Split on DecisionDate: train <= {TRAIN_END}, val <= {VAL_END}, test after")
    print(f"Events: {len(events)}    names: {events['TickerId'].nunique()}")
    print(f"Reversal OLS slope (train): {coef[1]:.4f}")
    print()
    print(summaries.to_string(index=False))
    print()
    print(correlations.to_string(index=False))
    print()
    print(strategies.to_string(index=False))
    print()
    print(f"Wrote {out}")

    if args.with_lin:
        from src.pipeline import run_family_by_name
        run_family_by_name(("lin",))


if __name__ == "__main__":
    main()
