"""Chronological split and train-only feature preprocessing for the event table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.default import FEATURE_COLUMNS as CONFIG_FEATURE_COLUMNS

ALLOWED_IMPUTE = {None, "median", "mean", "drop"}
ALLOWED_SPLIT = {
    "EventDate",
    "DecisionDate",
    "AnchorDate",
    "ReleaseDate",
}


@dataclass
class FittedPreprocessor:
    """Statistics fit on the training fold only."""

    feature_columns: list[str]
    impute_strategy: str | None
    standardize: bool
    fill_values: pd.Series | None
    mean: pd.Series | None
    scale: pd.Series | None


@dataclass
class FoldBundle:
    """One split after target filtering and feature transform."""

    name: str
    frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


@dataclass
class PreprocessResult:
    folds: dict[str, FoldBundle]
    fitted: FittedPreprocessor
    n_dropped_target: int
    n_before_target_drop: int


def chronological_split(
    df: pd.DataFrame,
    split_column: str,
    train_end: str,
    val_end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Split events by calendar cut dates on ``split_column``.

    train:  date <= train_end
    val:    train_end < date <= val_end   (only if val_end is set)
    test:   date > val_end if val_end else date > train_end
    """
    if split_column not in df.columns:
        raise KeyError(f"Split column {split_column!r} not in df.")
    if split_column not in ALLOWED_SPLIT:
        raise ValueError(f"split_column must be one of {sorted(ALLOWED_SPLIT)}.")

    dates = pd.to_datetime(df[split_column]).dt.normalize()
    train_cut = pd.Timestamp(train_end).normalize()
    train = df.loc[dates <= train_cut].copy()

    if val_end is None:
        test = df.loc[dates > train_cut].copy()
        if len(train) == 0 or len(test) == 0:
            raise ValueError(
                f"Empty fold after split: train={len(train)} test={len(test)} "
                f"({split_column} train_end={train_end})."
            )
        return {"train": train, "test": test}

    val_cut = pd.Timestamp(val_end).normalize()
    if val_cut <= train_cut:
        raise ValueError(f"VAL_END ({val_end}) must be after TRAIN_END ({train_end}).")

    val = df.loc[(dates > train_cut) & (dates <= val_cut)].copy()
    test = df.loc[dates > val_cut].copy()
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Empty fold after split: train={len(train)} val={len(val)} "
            f"test={len(test)} ({split_column} train_end={train_end} val_end={val_end})."
        )
    return {"train": train, "val": val, "test": test}


def drop_missing_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Remove rows whose target is missing. Returns (filtered_df, n_dropped)."""
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not in df.")
    before = len(df)
    out = df.loc[df[target].notna()].copy()
    return out, before - len(out)


def _resolve_features(feature_columns: list[str] | None, df: pd.DataFrame) -> list[str]:
    cols = list(CONFIG_FEATURE_COLUMNS if feature_columns is None else feature_columns)
    if not cols:
        raise ValueError("FEATURE_COLUMNS must be non-empty.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns missing from df: {missing}.")
    return cols


def fit_preprocessor(
    train: pd.DataFrame,
    feature_columns: list[str],
    impute_strategy: str | None,
    standardize: bool,
) -> FittedPreprocessor:
    """Fit impute / scale statistics on the training fold only."""
    if impute_strategy not in ALLOWED_IMPUTE:
        raise ValueError(
            f"IMPUTE_STRATEGY must be one of {sorted(ALLOWED_IMPUTE, key=lambda x: str(x))}."
        )

    frame = train[feature_columns]
    if impute_strategy == "drop":
        frame = frame.dropna()
        if len(frame) == 0:
            raise ValueError("No training rows left after dropping feature NaNs.")

    fill_values = None
    if impute_strategy in {"median", "mean"}:
        fill_values = (
            frame.median(numeric_only=True)
            if impute_strategy == "median"
            else frame.mean(numeric_only=True)
        )
        fill_values = fill_values.reindex(feature_columns)
        if fill_values.isna().any():
            # All-NaN feature on train: fall back to 0 after reporting via fill.
            fill_values = fill_values.fillna(0.0)

    mean = None
    scale = None
    if standardize:
        values = frame.copy()
        if fill_values is not None:
            values = values.fillna(fill_values)
        mean = values.mean(numeric_only=True).reindex(feature_columns)
        scale = values.std(numeric_only=True, ddof=0).reindex(feature_columns)
        scale = scale.replace(0.0, 1.0).fillna(1.0)
        mean = mean.fillna(0.0)

    return FittedPreprocessor(
        feature_columns=list(feature_columns),
        impute_strategy=impute_strategy,
        standardize=standardize,
        fill_values=fill_values,
        mean=mean,
        scale=scale,
    )


def transform_features(df: pd.DataFrame, fitted: FittedPreprocessor) -> pd.DataFrame:
    """Apply train-fitted impute / scale. May still contain NaNs if strategy is None."""
    X = df[fitted.feature_columns].copy()
    for column in fitted.feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    if fitted.impute_strategy == "drop":
        X = X.dropna()
    elif fitted.fill_values is not None:
        X = X.fillna(fitted.fill_values)

    if fitted.standardize:
        assert fitted.mean is not None and fitted.scale is not None
        X = (X - fitted.mean) / fitted.scale

    return X


def transform_fold(
    name: str,
    df: pd.DataFrame,
    target: str,
    fitted: FittedPreprocessor,
) -> FoldBundle:
    """Build X, y for one fold; align y to rows kept after feature drop."""
    X = transform_features(df, fitted)
    y = pd.to_numeric(df.loc[X.index, target], errors="coerce")
    keep = y.notna()
    X = X.loc[keep]
    y = y.loc[keep]
    frame = df.loc[X.index].copy()
    return FoldBundle(name=name, frame=frame, X=X, y=y)


def run_preprocess(
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str] | None = None,
    split_column: str = "DecisionDate",
    train_end: str = "2022-12-31",
    val_end: str | None = None,
    impute_strategy: str | None = "median",
    standardize: bool = True,
) -> PreprocessResult:
    """Full pipeline: drop missing targets → split → fit on train → transform folds."""
    features = _resolve_features(feature_columns, df)
    filtered, n_dropped = drop_missing_target(df, target)
    splits = chronological_split(filtered, split_column, train_end, val_end)
    fitted = fit_preprocessor(
        splits["train"],
        features,
        impute_strategy=impute_strategy,
        standardize=standardize,
    )
    folds = {
        name: transform_fold(name, part, target, fitted) for name, part in splits.items()
    }
    return PreprocessResult(
        folds=folds,
        fitted=fitted,
        n_dropped_target=n_dropped,
        n_before_target_drop=len(df),
    )


def summarize_preprocess(result: PreprocessResult) -> dict[str, Any]:
    """Compact counts for logging / scripts."""
    summary: dict[str, Any] = {
        "n_before_target_drop": result.n_before_target_drop,
        "n_dropped_target": result.n_dropped_target,
        "impute_strategy": result.fitted.impute_strategy,
        "standardize": result.fitted.standardize,
        "n_features": len(result.fitted.feature_columns),
        "folds": {},
    }
    for name, fold in result.folds.items():
        summary["folds"][name] = {
            "rows": len(fold.X),
            "y_mean": float(fold.y.mean()) if len(fold.y) else np.nan,
            "y_std": float(fold.y.std(ddof=0)) if len(fold.y) else np.nan,
            "feature_nan": int(fold.X.isna().sum().sum()),
        }
    return summary
