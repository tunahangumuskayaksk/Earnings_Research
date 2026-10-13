"""Run the analysis for every stock in the universe and write the summary tables.

Usage:  python scripts/run_universe.py [--group first30|added40] [--window announcement]
                                      [--out results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from earnings_reversal.cache import DiskCache  # noqa: E402
from earnings_reversal.config import OUTCOME_WINDOWS, AnalysisConfig  # noqa: E402
from earnings_reversal.summary import (  # noqa: E402
    analyse_universe,
    cross_stock_overview,
    pooled_events,
    pooled_group_table,
    strategy_universe_table,
    universe_table,
    volatility_ratios,
)
from earnings_reversal.universe import GROUPS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="announcement", choices=[w.key for w in OUTCOME_WINDOWS])
    parser.add_argument("--out", default="results", help="Output directory for CSV files.")
    parser.add_argument("--group", default=GROUPS[0].key, choices=[g.key for g in GROUPS],
                        help="first30 = the original 30 stocks, added40 = the 40 added stocks.")
    parser.add_argument("--refresh", action="store_true", help="Re-download instead of using the cache.")
    args = parser.parse_args()

    def progress(done: int, total: int, ticker: str) -> None:
        if ticker:
            print(f"[{done + 1}/{total}] {ticker}", file=sys.stderr)

    tickers = {g.key: g.tickers for g in GROUPS}[args.group]
    results = analyse_universe(AnalysisConfig(), tickers, DiskCache(),
                               force_refresh=args.refresh, progress=progress)
    table = universe_table(results, args.window)
    overview = cross_stock_overview(table)
    events = pooled_events(results)
    pooled = pooled_group_table(events, f"{args.window}_adj")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"universe_{args.window}.csv", index=False)
    overview.to_csv(out / f"overview_{args.window}.csv", index=False)
    events.to_csv(out / "pooled_events.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(overview.drop(columns="stat").to_string(index=False))
        print()
        print(pooled.to_string(index=False))
    strategy = strategy_universe_table(results)
    strategy.to_csv(out / "par_quintile_strategy.csv", index=False)
    ok = strategy[strategy["status"] == "ok"]
    if len(ok):
        print("\nPAR quintile strategy, announcement window, before costs "
              "(per-stock historical adaptation, same-close execution at t=-2):")
        cols = ["ticker", "trades", "longs", "shorts", "mean_adj", "win_adj", "pnl_adj",
                "mean_raw", "spread_adj", "sign_same_mean_adj", "mean_adj_pseudo_p5",
                "mean_adj_pseudo_p95", "mean_adj_position"]
        print(ok[[c for c in cols if c in ok]].to_string(index=False))
    ratios = volatility_ratios(results, args.window)
    if len(ratios):
        print(f"\nActual/pseudo outcome std. dev. ratio: median {ratios.median():.2f}, "
              f"range {ratios.min():.2f}-{ratios.max():.2f} over {len(ratios)} stocks")
    failed = table[table["status"] != "ok"]
    if len(failed):
        print("\nNot analysed:", ", ".join(f"{t} ({s})" for t, s in zip(failed["ticker"], failed["status"])))


if __name__ == "__main__":
    main()
