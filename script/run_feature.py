"""Build the event table and print its descriptive report (no files written)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.default import BENCHMARK_SOURCE, DATA_DIR, FEATURE_COLUMNS, MARKET_TICKER
from src.features import build_aligned_panel, build_event_table
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.report import print_benchmark_preview, print_event_report


def main() -> None:
    features = list(FEATURE_COLUMNS)
    stock_data = load_stock(DATA_DIR)
    earnings = load_earnings(DATA_DIR)
    historical_sentiment_scores = load_sentiment(DATA_DIR)
    df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)

    df_align = build_aligned_panel(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_DIR,
        benchmark_source=BENCHMARK_SOURCE,
    )
    print_benchmark_preview(df_align)

    df = build_event_table(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_DIR,
        df_align=df_align,
        feature_columns=features,
        benchmark_source=BENCHMARK_SOURCE,
    )
    print_event_report(df, feature_columns=features)


if __name__ == "__main__":
    main()
