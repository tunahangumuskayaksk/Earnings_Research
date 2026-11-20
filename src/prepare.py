"""Load and merge the earnings-movements parquet tables (no modeling prep)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from config.default import (
    DATA_DIR,
    EARNINGS_FILE,
    SENTIMENT_FILE,
    SENTIMENT_TOLERANCE_DAYS,
    STOCK_FILE,
)

ROOT = Path(__file__).resolve().parents[1]

_CHUNK = 50_000


def _row_bar(total: int) -> tqdm:
    return tqdm(
        total=total,
        leave=False,
        position=0,
        dynamic_ncols=True,
        desc="",
        unit="",
        miniters=1,
        mininterval=0,
        bar_format="{bar}| {n_fmt}/{total_fmt}",
    )


def _read_parquet(path: Path) -> pd.DataFrame:
    file = pq.ParquetFile(path)
    batches: list[pa.RecordBatch] = []
    with _row_bar(file.metadata.num_rows) as bar:
        for batch in file.iter_batches(batch_size=_CHUNK):
            batches.append(batch)
            bar.update(batch.num_rows)
    if not batches:
        return pd.DataFrame()
    return pa.Table.from_batches(batches).to_pandas()


def load_stock(data_dir: Path | None = None) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    return _read_parquet(folder / STOCK_FILE)


def load_earnings(data_dir: Path | None = None) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    return _read_parquet(folder / EARNINGS_FILE)


def load_sentiment(data_dir: Path | None = None) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    sentiment = _read_parquet(folder / SENTIMENT_FILE)
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment = sentiment.drop_duplicates(["symbol", "date"], keep="last")
    sentiment["symbol"] = sentiment["symbol"].astype("string")
    return sentiment.sort_values("date")


def _merge_sentiment(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    tolerance_days: int = SENTIMENT_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """Attach the latest rating snapshot on or before each price date.

    Snapshots are monthly. A rating older than the tolerance is left missing.
    """
    frame = prices.copy()
    frame["_row"] = range(len(frame))
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Symbol"] = frame["Symbol"].astype("string")

    ratings = sentiment.copy()
    ratings["date"] = pd.to_datetime(ratings["date"])
    ratings["symbol"] = ratings["symbol"].astype("string")
    ratings = ratings.drop_duplicates(["symbol", "date"], keep="last")
    ratings = ratings.sort_values("date")

    ordered = frame.sort_values("Date")
    merged = pd.merge_asof(
        ordered,
        ratings,
        left_on="Date",
        right_on="date",
        left_by="Symbol",
        right_by="symbol",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    return merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def merge_panel(
    stock_data: pd.DataFrame,
    earnings: pd.DataFrame,
    historical_sentiment_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Merge earnings on Date/TickerId, then asof-carry monthly ratings."""
    earnings = earnings.copy()
    earnings["EventDate"] = pd.to_datetime(earnings["EventDate"])

    parts: list[pd.DataFrame] = []
    with _row_bar(len(stock_data)) as bar:
        for start in range(0, len(stock_data), _CHUNK):
            chunk = stock_data.iloc[start : start + _CHUNK]
            parts.append(
                chunk.merge(
                    earnings,
                    how="left",
                    left_on=["Date", "TickerId"],
                    right_on=["EventDate", "TickerId"],
                )
            )
            bar.update(len(chunk))
    temp = pd.concat(parts, ignore_index=True)
    temp[["Symbol", "Exchange"]] = temp["ShortName"].str.split(":", expand=True)
    return _merge_sentiment(temp, historical_sentiment_scores)
