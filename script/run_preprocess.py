"""Build the event table, run train-only preprocessing, and print a summary.

Shared defaults come from ``config/default.py``. Local overrides below are
script-only (e.g. STANDARDIZE); model runs use ``config/config_*.py``.

Usage (from repo root):
    python script/run_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.default import (
    BENCHMARK_SOURCE,
    DATA_DIR,
    FEATURE_COLUMNS as CONFIG_FEATURE_COLUMNS,
    IMPUTE_STRATEGY,
    MARKET_TICKER,
    SPLIT_COLUMN,
    TARGET,
    TRAIN_END,
    VAL_END,
)
from src.features import build_event_table
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.preprocess import run_preprocess
from src.report import print_preprocess_report

# ============================== [LOCAL OVERRIDES] ==============================

# Feature list. None uses config.default.FEATURE_COLUMNS.
FEATURE_COLUMNS = None

# Z-score features with train mean/std. Allowed: True | False
STANDARDIZE = True

# ==============================================================================


def _resolved_features() -> list[str]:
    if FEATURE_COLUMNS is None:
        return list(CONFIG_FEATURE_COLUMNS)
    return list(FEATURE_COLUMNS)


def main() -> None:
    features = _resolved_features()

    stock_data = load_stock(DATA_DIR)
    earnings = load_earnings(DATA_DIR)
    historical_sentiment_scores = load_sentiment(DATA_DIR)
    df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)
    df = build_event_table(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_DIR,
        feature_columns=features,
        benchmark_source=BENCHMARK_SOURCE,
    )

    result = run_preprocess(
        df,
        target=TARGET,
        feature_columns=features,
        split_column=SPLIT_COLUMN,
        train_end=TRAIN_END,
        val_end=VAL_END,
        impute_strategy=IMPUTE_STRATEGY,
        standardize=STANDARDIZE,
    )
    print_preprocess_report(
        result,
        target=TARGET,
        split_column=SPLIT_COLUMN,
    )


if __name__ == "__main__":
    main()
