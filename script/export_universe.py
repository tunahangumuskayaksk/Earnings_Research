"""Export the ~200-name stock universe to ``data/stock_universe.csv``.

One row per ``TickerId`` from ``stock-data_us.parquet``: identifiers and
price date coverage for WRDS / external downloads.

Usage (from repo root):
    python script/export_universe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.default import DATA_DIR
from src.prepare import load_stock

OUTPUT_NAME = "stock_universe.csv"

UNIVERSE_COLUMNS = [
    "TickerId",
    "Symbol",
    "Exchange",
    "InstrumentName",
    "price_start",
    "price_end",
]


def _split_short_name(short_name: object) -> tuple[object, object]:
    text = "" if short_name is None or (isinstance(short_name, float) and pd.isna(short_name)) else str(short_name)
    if ":" in text:
        symbol, exchange = text.split(":", 1)
        return symbol, exchange
    return (text or pd.NA), pd.NA


def _mode_or_last(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    modes = clean.mode(dropna=True)
    if len(modes) > 0:
        return modes.iloc[0]
    return clean.iloc[-1]


def build_universe(stock: pd.DataFrame) -> pd.DataFrame:
    """Collapse the daily stock panel to one universe row per ``TickerId``."""
    required = ["TickerId", "Date", "ShortName", "InstrumentName"]
    missing = [c for c in required if c not in stock.columns]
    if missing:
        raise KeyError(f"stock panel missing columns: {missing}")

    frame = stock.loc[:, required].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.dropna(subset=["TickerId", "Date"])
    frame = frame.sort_values(["TickerId", "Date"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for ticker_id, group in frame.groupby("TickerId", sort=True):
        short_name = _mode_or_last(group["ShortName"])
        symbol, exchange = _split_short_name(short_name)
        rows.append(
            {
                "TickerId": int(ticker_id) if pd.notna(ticker_id) else pd.NA,
                "Symbol": symbol,
                "Exchange": exchange,
                "InstrumentName": _mode_or_last(group["InstrumentName"]),
                "price_start": group["Date"].min().strftime("%Y-%m-%d"),
                "price_end": group["Date"].max().strftime("%Y-%m-%d"),
            }
        )

    universe = pd.DataFrame(rows)
    return universe.loc[:, UNIVERSE_COLUMNS].sort_values(["Symbol", "TickerId"]).reset_index(drop=True)


def export_universe(
    data_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    folder = DATA_DIR if data_dir is None else data_dir
    out = folder / OUTPUT_NAME if output_path is None else output_path
    stock = load_stock(folder)
    universe = build_universe(stock)
    out.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(out, index=False)
    return out


def main() -> None:
    path = export_universe()
    universe = pd.read_csv(path)
    print(f"wrote {path}  rows={len(universe)}  cols={len(universe.columns)}")
    print(
        f"symbols={universe['Symbol'].nunique(dropna=True)}  "
        f"exchanges={sorted(universe['Exchange'].dropna().unique().tolist())}"
    )
    print(universe.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
