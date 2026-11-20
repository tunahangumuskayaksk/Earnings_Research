"""Terminal descriptive summaries for the event training table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.benchmark import BENCHMARK_COLUMN
from config.default import FEATURE_COLUMNS
from src.preprocess import PreprocessResult

TARGET_COLUMNS = [
    "ret_ann",
    "ret_ann_mkt",
    "ret_next",
    "ret_next_mkt",
    "ret_post",
    "ret_post_mkt",
    "dir_ann",
    "vol_ann",
    "vol_post",
]
PRIMARY_TARGET = "ret_ann_mkt"
DATE_COLUMNS = (
    "EventDate",
    "ReleaseDate",
    "AnchorDate",
    "DecisionDate",
)


def _configure_display() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def _section(title: str) -> None:
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def _day(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _for_print(df: pd.DataFrame) -> pd.DataFrame:
    """Copy for terminal display: dates as YYYY-MM-DD, cleaner integer ids."""
    out = df.copy()
    for column in DATE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d")
    if "MarketTimeId" in out.columns:
        out["MarketTimeId"] = pd.to_numeric(out["MarketTimeId"], errors="coerce").astype("Int64")
    if "Id" in out.columns:
        out["Id"] = pd.to_numeric(out["Id"], errors="coerce").astype("Int64")
    if "dir_ann" in out.columns:
        out["dir_ann"] = pd.to_numeric(out["dir_ann"], errors="coerce").astype("Int64")
    return out


def _coverage(df: pd.DataFrame) -> None:
    print(f"rows={len(df)}  cols={df.shape[1]}  tickers={df['TickerId'].nunique()}")
    print(f"EventDate        {_day(df['EventDate'].min())} .. {_day(df['EventDate'].max())}")
    print(
        f"DecisionDate     {_day(df['DecisionDate'].min())} .. "
        f"{_day(df['DecisionDate'].max())}"
    )
    if "MarketTimeId" in df.columns:
        counts = (
            pd.to_numeric(df["MarketTimeId"], errors="coerce")
            .astype("Int64")
            .value_counts(dropna=False)
            .sort_index()
        )
        print(counts.to_string())


def _as_day(value):
    """Format timestamps as YYYY-MM-DD; leave other values unchanged."""
    if value is pd.NA or value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return value
    if pd.isna(ts):
        return value
    return ts.strftime("%Y-%m-%d")


def _column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact column stats: count/unique/min/max/mean/std/dtype + missingness."""
    desc = df.describe(include="all").T
    summary = pd.DataFrame(index=df.columns)
    summary["count"] = desc["count"] if "count" in desc.columns else df.count()
    summary["unique"] = df.nunique(dropna=True)
    for column in ("min", "max", "mean", "std"):
        summary[column] = desc[column] if column in desc.columns else pd.NA
    date_cols = [c for c in DATE_COLUMNS if c in summary.index]
    if date_cols:
        for column in ("min", "max", "mean"):
            summary.loc[date_cols, column] = summary.loc[date_cols, column].map(_as_day)
        summary.loc[date_cols, "std"] = pd.NA
    summary["dtype"] = df.dtypes.astype(str)
    summary["missing_n"] = df.isna().sum()
    summary["missing_rate"] = df.isna().mean().map(lambda x: f"{x:.2%}")
    return summary


def _corr_with_target(
    df: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str] | None = None,
) -> None:
    feature_cols = list(FEATURE_COLUMNS if feature_columns is None else feature_columns)
    cols = [c for c in feature_cols + TARGET_COLUMNS if c in df.columns and c != target]
    if target not in df.columns:
        print(f"{target} not in df")
        return
    pair = df[[target] + cols].apply(pd.to_numeric, errors="coerce")
    corr = (
        pair.corr(method="pearson")[target]
        .drop(labels=[target])
        .sort_index(key=lambda idx: idx.str.lower())
    )
    print(f"pearson vs {target}")
    print(corr.round(4).to_string())


def print_notebook_config(
    *,
    target: str,
    feature_columns: Sequence[str],
    split_column: str,
    train_end: str,
    val_end: str,
    impute_strategy: str | None,
    random_state: int,
    feature_importance: bool,
    feature_importance_repeats: int,
    standardize: bool | None = None,
    active_families: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    benchmark_source: str | None = None,
    market_ticker: str | None = None,
) -> None:
    """Print notebook-local CONFIG (shared knobs + optional per-family blocks).

    Unlike ``src.pipeline.print_family_config``, this does not read
    ``config.config_*`` modules — only the values passed from the notebook.
    """
    _section("CONFIG")
    if active_families is not None:
        print(f"ACTIVE_FAMILIES  {[name for name, _ in active_families]}")
    print(f"TARGET            {target}")
    if benchmark_source is not None:
        print(f"BENCHMARK_SOURCE  {benchmark_source}")
    if market_ticker is not None:
        print(f"MARKET_TICKER     {market_ticker}")
    print(f"FEATURE_COLUMNS   ({len(feature_columns)})")
    for name in feature_columns:
        print(f"                  - {name}")
    print(f"SPLIT_COLUMN      {split_column}")
    print(f"TRAIN_END         {train_end}")
    print(f"VAL_END           {val_end}")
    print(f"IMPUTE_STRATEGY   {impute_strategy}")
    if standardize is not None:
        print(f"STANDARDIZE (step5 preview) {standardize}")
    print(f"RANDOM_STATE      {random_state}")
    print(f"FEATURE_IMPORTANCE {feature_importance}")
    print(f"FEATURE_IMPORTANCE_REPEATS {feature_importance_repeats}")
    if active_families:
        for family, block in active_families:
            print()
            print(f"[{family}]")
            for key, value in block.items():
                print(f"  {key:<22} {value}")


def print_benchmark_preview(df_align: pd.DataFrame, n: int = 5) -> None:
    """Print the session-aligned panel after benchmark attach (compact check)."""
    _configure_display()
    if BENCHMARK_COLUMN not in df_align.columns:
        raise KeyError(f"{BENCHMARK_COLUMN!r} missing from aligned panel.")

    dates = pd.to_datetime(df_align["Date"])
    bench = pd.to_numeric(df_align[BENCHMARK_COLUMN], errors="coerce")
    _section("benchmark")
    print(f"rows={len(df_align)}  tickers={df_align['TickerId'].nunique()}")
    print(f"Date             {_day(dates.min())} .. {_day(dates.max())}")
    print(
        f"{BENCHMARK_COLUMN}  "
        f"null={bench.isna().mean():.2%}  "
        f"min={bench.min():.4f}  max={bench.max():.4f}"
    )
    cols = [
        c
        for c in ("Date", "TickerId", "Symbol", "LastPrice", BENCHMARK_COLUMN)
        if c in df_align.columns
    ]
    preview = df_align.loc[:, cols].copy()
    preview["Date"] = pd.to_datetime(preview["Date"]).dt.strftime("%Y-%m-%d")
    print(preview.head(n).to_string(index=False))


def print_event_report(
    df: pd.DataFrame,
    primary_target: str = PRIMARY_TARGET,
    feature_columns: Sequence[str] | None = None,
) -> None:
    """Print coverage, head, compact describe, correlations, and target summary."""
    _configure_display()
    view = _for_print(df)

    _section("df coverage")
    _coverage(df)

    _section("df head")
    print(view.head())

    _section("df describe")
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    print(_column_summary(df[feature_cols]))

    _section(f"df correlations ({primary_target})")
    _corr_with_target(df, primary_target, feature_columns=feature_columns)

    _section("df target summary")
    present = sorted(
        [c for c in TARGET_COLUMNS if c in df.columns],
        key=str.lower,
    )
    print(_column_summary(df[present]))


def print_preprocess_report(
    result: PreprocessResult,
    *,
    target: str,
    split_column: str,
) -> None:
    """Print fold coverage, target stats, train X head, and per-fold feature summary."""
    _configure_display()

    _section("folds")
    for name, fold in result.folds.items():
        dates = pd.to_datetime(fold.frame[split_column])
        print(
            f"{name:5s} n={len(fold.frame):5d}  "
            f"{dates.min().date()} .. {dates.max().date()}"
        )
    train_y_mean = float(result.folds["train"].y.to_numpy(dtype=float).mean())
    print("train_y_mean", train_y_mean)

    _section("target summary")
    print(
        f"target={target}  events_in={result.n_before_target_drop}  "
        f"dropped_target={result.n_dropped_target}"
    )
    rows: list[dict] = []
    for name, fold in result.folds.items():
        y = fold.y
        rows.append(
            {
                "fold": name,
                "count": int(y.notna().sum()),
                "unique": int(y.nunique(dropna=True)),
                "min": float(y.min()) if len(y) else float("nan"),
                "max": float(y.max()) if len(y) else float("nan"),
                "mean": float(y.mean()) if len(y) else float("nan"),
                "std": float(y.std(ddof=0)) if len(y) else float("nan"),
                "missing_n": int(y.isna().sum()),
                "missing_rate": f"{y.isna().mean():.2%}" if len(fold.frame) else "n/a",
            }
        )
    print(pd.DataFrame(rows).set_index("fold"))

    _section("train X head")
    print(result.folds["train"].X.head())

    _section("feature summary")
    print(
        f"impute={result.fitted.impute_strategy}  "
        f"standardize={result.fitted.standardize}  "
        f"n_features={len(result.fitted.feature_columns)}"
    )
    for name, fold in result.folds.items():
        print()
        print(f"[{name}] rows={len(fold.X)}")
        print(_column_summary(fold.X))
