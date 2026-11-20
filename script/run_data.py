"""Load/merge the daily panel and print table previews (no files written)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.default import DATA_DIR
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def main() -> None:
    stock_data = load_stock(DATA_DIR)
    print("stock_data")
    print(stock_data.head())
    print()

    earnings = load_earnings(DATA_DIR)
    print("earnings")
    print(earnings.head())
    print()

    historical_sentiment_scores = load_sentiment(DATA_DIR)
    print("historical_sentiment_scores")
    print(historical_sentiment_scores.head())
    print()

    df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)
    print("df_all")
    print(df_all.head())
    print(df_all.shape)
    print()
    summary = df_all.describe(include="all").T
    summary["dtype"] = df_all.dtypes
    print("df_all.describe")
    print(summary)


if __name__ == "__main__":
    main()
