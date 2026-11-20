"""Pre-announcement reversal on the IMP event panel.

PAR uses the same anchor as ``src.features`` (before-open keeps the release
session; after-close moves to the next session). The window is close at
anchor−5 through close at anchor−2, so the signal is finished on
``DecisionDate``. The announcement target ``ret_ann_mkt`` starts at that same
close. That is a same-close execution assumption, and every figure is before
costs.

This is not a new price panel. It reads ``df_align`` from
``build_aligned_panel`` and joins onto ``build_event_table``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.default import (
    FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET,
    TRAIN_END,
    VAL_END,
)
from src.benchmark import BENCHMARK_COLUMN
from src.features import (
    DECISION_OFFSET,
    EVENT_COLUMN,
    PRICE_COLUMN,
    _event_anchors,
    _window_return,
)
from src.metrics import evaluate_regression, quintile_table, summary_row_from_fold_metrics
from src.preprocess import chronological_split

# Close at anchor-5 -> close at anchor-2. Ends on DecisionDate (anchor-2).
PAR_START_OFFSET = -4
PAR_END_OFFSET = -2
MIN_PRIOR_EVENTS = 20
LOW_PERCENTILE = 20
HIGH_PERCENTILE = 80

OUTCOME_COLUMNS = ("ret_ann_mkt", "ret_ann", "ret_post_mkt")


def attach_par(df_align: pd.DataFrame) -> pd.DataFrame:
    """One row per announcement: stock PAR, benchmark PAR, and the difference."""
    rows: list[dict] = []
    for ticker, frame in df_align.groupby("TickerId", sort=False):
        frame = frame.sort_values("Date").reset_index(drop=True)
        prices = frame[PRICE_COLUMN].to_numpy(dtype=float)
        benchmark = frame[BENCHMARK_COLUMN].to_numpy(dtype=float)
        sessions = pd.to_datetime(frame["Date"]).to_numpy()
        for announcement, anchor in _event_anchors(frame):
            if anchor is None:
                continue
            decision_at = anchor + DECISION_OFFSET
            if decision_at < 0 or decision_at >= len(frame):
                continue
            par = _window_return(prices, anchor, PAR_START_OFFSET, PAR_END_OFFSET)
            par_bench = _window_return(benchmark, anchor, PAR_START_OFFSET, PAR_END_OFFSET)
            par_mkt = (
                par - par_bench if np.isfinite(par) and np.isfinite(par_bench) else np.nan
            )
            release = frame.iloc[announcement]
            rows.append({
                "TickerId": ticker,
                "EventDate": pd.Timestamp(release[EVENT_COLUMN]).normalize(),
                "par": par,
                "par_bench": par_bench,
                "par_mkt": par_mkt,
                "par_end": pd.Timestamp(sessions[anchor + PAR_END_OFFSET]).normalize()
                if 0 <= anchor + PAR_END_OFFSET < len(sessions) else pd.NaT,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["EventDate"] = pd.to_datetime(out["EventDate"]).dt.normalize()
    return out


def with_par(events: pd.DataFrame, df_align: pd.DataFrame) -> pd.DataFrame:
    """Left-join PAR onto an IMP event table. DecisionDate must match the PAR end."""
    par = attach_par(df_align)
    if par.empty:
        frame = events.copy()
        for column in ("par", "par_bench", "par_mkt", "par_end"):
            frame[column] = np.nan
        return frame
    merged = events.merge(par, on=["TickerId", "EventDate"], how="left")
    return merged


def assign_folds(
    events: pd.DataFrame,
    split_column: str = SPLIT_COLUMN,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> pd.DataFrame:
    """Tag each event train / val / test with IMP's chronological cuts."""
    parts = chronological_split(events, split_column, train_end, val_end)
    frames = []
    for name, part in parts.items():
        tagged = part.copy()
        tagged["fold"] = name
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True)


def _design(frame: pd.DataFrame, columns: list[str], fill: pd.Series | None) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if fill is not None:
        values = values.fillna(fill)
    clean = values.to_numpy(dtype=float)
    return np.column_stack([np.ones(len(clean)), clean])


def _fit_ols(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
    return coef


def _prediction_frame(events: pd.DataFrame, y_pred: np.ndarray, model: str) -> pd.DataFrame:
    return pd.DataFrame({
        "model": model,
        "fold": events["fold"].to_numpy(),
        "TickerId": events["TickerId"].to_numpy(),
        "DecisionDate": pd.to_datetime(events["DecisionDate"]).to_numpy(),
        "y_true": events[TARGET].to_numpy(dtype=float),
        "y_pred": y_pred,
    })


def _score(model: str, predictions: pd.DataFrame, train_mean: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    for fold in ("train", "val", "test"):
        part = predictions.loc[predictions["fold"] == fold]
        metrics = evaluate_regression(part["y_true"], part["y_pred"], train_y_mean=train_mean)
        metrics["fold"] = fold
        metrics["model"] = model
        rows.append(metrics)
    fold_metrics = pd.DataFrame(rows).set_index("fold")
    summary = summary_row_from_fold_metrics(
        model, fold_metrics, quintiles=quintile_table(predictions, folds=("train", "val", "test"))
    )
    return fold_metrics.reset_index(), summary


def fit_reversal_ols(events: pd.DataFrame) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Train-only OLS: ret_ann_mkt ~ par_mkt. A negative slope is a reversal."""
    usable = events.dropna(subset=[TARGET, "par_mkt"]).copy()
    train = usable.loc[usable["fold"] == "train"]
    if len(train) < 3:
        raise ValueError("Not enough training rows with PAR and the announcement target.")
    coef = _fit_ols(
        _design(train, ["par_mkt"], None),
        train[TARGET].to_numpy(dtype=float),
    )
    pred = _design(usable, ["par_mkt"], None) @ coef
    predictions = _prediction_frame(usable, pred, "reversal_ols")
    train_mean = float(train[TARGET].mean())
    fold_metrics, summary = _score("reversal_ols", predictions, train_mean)
    summary["slope"] = float(coef[1])
    summary["intercept"] = float(coef[0])
    return fold_metrics, summary, coef


def fit_positioning_ols(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train-only OLS on IMP's QScore and analyst-rating features.

    Median fill is fit on train only. This is the linear baseline the reversal
    model is compared with; ridge, trees, and networks stay in ``main.py``.
    """
    columns = [c for c in FEATURE_COLUMNS if c in events.columns]
    usable = events.dropna(subset=[TARGET]).copy()
    train = usable.loc[usable["fold"] == "train"]
    fill = train[columns].apply(pd.to_numeric, errors="coerce").median()
    coef = _fit_ols(
        _design(train, columns, fill),
        train[TARGET].to_numpy(dtype=float),
    )
    pred = _design(usable, columns, fill) @ coef
    predictions = _prediction_frame(usable, pred, "positioning_ols")
    fold_metrics, summary = _score("positioning_ols", predictions, float(train[TARGET].mean()))
    return fold_metrics, summary


def _expanding_quintile(par: np.ndarray) -> np.ndarray:
    direction = np.zeros(len(par), dtype=int)
    finite_prefix: list[float] = []
    for i, value in enumerate(par):
        if len(finite_prefix) >= MIN_PRIOR_EVENTS and np.isfinite(value):
            lo, hi = np.percentile(finite_prefix, [LOW_PERCENTILE, HIGH_PERCENTILE])
            if value < lo:
                direction[i] = 1
            elif value > hi:
                direction[i] = -1
        if np.isfinite(value):
            finite_prefix.append(float(value))
    return direction


def _train_cutoff_direction(par: np.ndarray, fold: np.ndarray) -> np.ndarray:
    train = par[(fold == "train") & np.isfinite(par)]
    direction = np.zeros(len(par), dtype=int)
    if train.size < MIN_PRIOR_EVENTS:
        return direction
    lo, hi = np.percentile(train, [LOW_PERCENTILE, HIGH_PERCENTILE])
    for i, value in enumerate(par):
        if fold[i] == "train" or not np.isfinite(value):
            continue
        if value < lo:
            direction[i] = 1
        elif value > hi:
            direction[i] = -1
    return direction


def _pooled_train_direction(frame: pd.DataFrame) -> np.ndarray:
    """20th/80th of every training-fold PAR, applied to every name.

    A single name does not reach 20 prior events inside 2020–2022, so the
    per-stock cutoff never trades. This cutoff is the cross-section IMP uses
    when it buckets predictions.
    """
    train = frame.loc[frame["fold"] == "train", "par_mkt"].to_numpy(dtype=float)
    train = train[np.isfinite(train)]
    direction = np.zeros(len(frame), dtype=int)
    if train.size < MIN_PRIOR_EVENTS:
        return direction
    lo, hi = np.percentile(train, [LOW_PERCENTILE, HIGH_PERCENTILE])
    par = frame["par_mkt"].to_numpy(dtype=float)
    for i, value in enumerate(par):
        if not np.isfinite(value):
            continue
        if value < lo:
            direction[i] = 1
        elif value > hi:
            direction[i] = -1
    return direction


def strategy_table(events: pd.DataFrame) -> pd.DataFrame:
    """Signed announcement returns for the sign rule and the quintile rules.

    Long when PAR is low, short when PAR is high. ``mean_signed_mkt`` is the
    average return per executed trade, not a long-minus-short group spread.
    """
    frame = events.dropna(subset=["par_mkt", "ret_ann_mkt"]).sort_values(
        ["TickerId", "DecisionDate"]
    )
    pieces = []
    for _, group in frame.groupby("TickerId", sort=False):
        group = group.copy()
        par = group["par_mkt"].to_numpy(dtype=float)
        group["sign_direction"] = np.where(np.isfinite(par), -np.sign(par), 0).astype(int)
        group["expanding_direction"] = _expanding_quintile(par)
        group["train_cutoff_direction"] = _train_cutoff_direction(
            par, group["fold"].to_numpy()
        )
        pieces.append(group)
    traded = pd.concat(pieces, ignore_index=True) if pieces else frame
    traded["pooled_direction"] = _pooled_train_direction(traded)
    rows = []
    rules = (
        ("sign", "sign_direction"),
        ("quintile_expanding", "expanding_direction"),
        ("quintile_train_cutoff", "train_cutoff_direction"),
        ("quintile_pooled_train", "pooled_direction"),
    )
    for rule, column in rules:
        for fold, part in traded.groupby("fold"):
            direction = part[column].to_numpy(dtype=int)
            active = direction != 0
            signed_mkt = direction[active] * part.loc[active, "ret_ann_mkt"].to_numpy(dtype=float)
            signed_raw = direction[active] * part.loc[active, "ret_ann"].to_numpy(dtype=float)
            rows.append({
                "rule": rule,
                "fold": fold,
                "events": int(len(part)),
                "trades": int(active.sum()),
                "longs": int((direction == 1).sum()),
                "shorts": int((direction == -1).sum()),
                "mean_signed_mkt": float(signed_mkt.mean()) if active.any() else np.nan,
                "mean_signed_raw": float(signed_raw.mean()) if active.any() else np.nan,
                "hit_rate_mkt": float((signed_mkt > 0).mean()) if active.any() else np.nan,
            })
    return pd.DataFrame(rows)


def par_outcome_correlation(events: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman correlation of PAR with each IMP outcome, by fold."""
    rows = []
    for fold, part in events.groupby("fold"):
        for outcome in OUTCOME_COLUMNS:
            pair = part[["par_mkt", outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3 or pair["par_mkt"].nunique() < 2:
                pearson = spearman = np.nan
            else:
                pearson = float(pair["par_mkt"].corr(pair[outcome], method="pearson"))
                spearman = float(pair["par_mkt"].corr(pair[outcome], method="spearman"))
            rows.append({
                "fold": fold,
                "outcome": outcome,
                "n": int(len(pair)),
                "pearson": pearson,
                "spearman": spearman,
            })
    return pd.DataFrame(rows)


def write_reversal_outputs(
    folder: Path,
    events: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    strategies: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    events.to_parquet(folder / "events.parquet", index=False)
    fold_metrics.to_csv(folder / "fold_metrics.csv", index=False)
    summaries.to_csv(folder / "metrics.csv", index=False)
    strategies.to_csv(folder / "strategy.csv", index=False)
    correlations.to_csv(folder / "correlation.csv", index=False)
