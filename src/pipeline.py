"""End-to-end experiment pipeline: load → preprocess → evaluate → artifacts.

Family-specific tune / fit adapters live in ``src/train.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_event_table
from src.metrics import (
    infer_task,
    metrics_table,
    permutation_importance_regression,
    print_model_fit_report,
    print_summary_table,
    quintile_table,
    summary_row_from_fold_metrics,
)
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.preprocess import PreprocessResult, run_preprocess
from src.train import (
    FamilyName,
    FamilySpec,
    PredictFn,
    get_family_spec,
)

PREDICTION_META_COLS = (
    "TickerId",
    "EventDate",
    "DecisionDate",
    "AnchorDate",
    "ReleaseDate",
)
BANNER = "=" * 120


def print_banner(title: str) -> None:
    print()
    print(BANNER)
    print()
    print(title.upper())
    print()
    print(BANNER)


def fold_date_range(result: PreprocessResult, split_column: str) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for name, fold in result.folds.items():
        dates = pd.to_datetime(fold.frame[split_column])
        summary[name] = {
            "rows": int(len(fold.frame)),
            "date_min": dates.min().strftime("%Y-%m-%d"),
            "date_max": dates.max().strftime("%Y-%m-%d"),
            "y_mean": float(fold.y.mean()),
            "y_std": float(fold.y.std(ddof=0)),
            "feature_nan": int(fold.X.isna().sum().sum()),
        }
    return summary


def build_predictions(
    result: PreprocessResult,
    *,
    model_type: str,
    predict: PredictFn,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for name, fold in result.folds.items():
        meta_cols = [c for c in PREDICTION_META_COLS if c in fold.frame.columns]
        block = fold.frame.loc[fold.X.index, meta_cols].copy()
        block["model"] = model_type
        block["fold"] = name
        block["y_true"] = fold.y.to_numpy(dtype=float)
        block["y_pred"] = np.asarray(predict(fold.X), dtype=float)
        block["residual"] = block["y_true"] - block["y_pred"]
        blocks.append(block)
    return pd.concat(blocks, axis=0).reset_index(drop=True)


def save_run_artifacts(
    *,
    output_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    run_config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    stale = output_dir / "quintiles.csv"
    if stale.exists():
        stale.unlink()
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)


def print_family_config(spec: FamilySpec) -> None:
    print("-" * 60)
    print("CONFIG")
    print("-" * 60)
    print(f"MODEL_LABEL       {spec.config.MODEL_LABEL}")
    for key, value in spec.config_extra_lines:
        print(f"{key:<18}{value}")
    print(f"TARGET            {spec.config.TARGET}")
    print(f"BENCHMARK_SOURCE  {spec.config.BENCHMARK_SOURCE}")
    print(f"MARKET_TICKER     {spec.config.MARKET_TICKER}")
    print(f"DATA_ROOT         {spec.config.DATA_ROOT}")
    print(f"FEATURE_COLUMNS   ({len(spec.config.FEATURE_COLUMNS)})")
    for name in spec.config.FEATURE_COLUMNS:
        print(f"                  - {name}")
    print(f"SPLIT_COLUMN      {spec.config.SPLIT_COLUMN}")
    print(f"TRAIN_END         {spec.config.TRAIN_END}")
    print(f"VAL_END           {spec.config.VAL_END}")
    print(f"IMPUTE_STRATEGY   {spec.config.IMPUTE_STRATEGY}")
    print(f"STANDARDIZE       {spec.config.STANDARDIZE}")
    print(f"RANDOM_STATE      {spec.config.RANDOM_STATE}")
    print(f"FEATURE_IMPORTANCE {spec.config.FEATURE_IMPORTANCE}")
    print(f"FEATURE_IMPORTANCE_REPEATS {spec.config.FEATURE_IMPORTANCE_REPEATS}")


def run_one_model(
    spec: FamilySpec,
    *,
    model_type: str,
    result: PreprocessResult,
    train_y_mean: float,
    task: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Fit one model via the family adapter, then shared metrics / report."""
    bundle = spec.fit_one(model_type, result)
    cfg = spec.config

    fold_predictions = {
        name: (fold.y, bundle.predict(fold.X)) for name, fold in result.folds.items()
    }
    fold_metrics = metrics_table(
        fold_predictions, task=task, train_y_mean=train_y_mean
    )

    importance = None
    if cfg.FEATURE_IMPORTANCE:
        test = result.folds["test"]
        importance = permutation_importance_regression(
            bundle.predict,
            test.X,
            test.y,
            n_repeats=cfg.FEATURE_IMPORTANCE_REPEATS,
            random_state=cfg.RANDOM_STATE,
            train_y_mean=train_y_mean,
        )

    predictions = build_predictions(
        result, model_type=model_type, predict=bundle.predict
    )
    quintiles = quintile_table(predictions)
    print_model_fit_report(
        model_type,
        fold_metrics=fold_metrics,
        quintiles=quintiles,
        tuning=bundle.tuning,
        selected=bundle.selected,
        tuning_label=bundle.tuning_label,
        best_iteration=bundle.best_iteration,
        max_iter=bundle.max_iter,
        importance=importance,
    )
    summary = summary_row_from_fold_metrics(
        model_type,
        fold_metrics,
        quintiles=quintiles,
        **bundle.summary_extras,
    )
    return summary, predictions


def run_family(spec: FamilySpec) -> None:
    """Full batch run for one model family (config → data → fit → artifacts)."""
    cfg = spec.config
    print_family_config(spec)

    task = infer_task(cfg.TARGET)
    if task != "regression":
        raise NotImplementedError(
            "Classification metrics are not implemented yet; "
            f"use a regression TARGET (got TARGET={cfg.TARGET!r})."
        )

    model_types = tuple(spec.model_types)
    if not model_types:
        raise ValueError(f"{spec.model_types_key} must be non-empty.")

    stock_data = load_stock(cfg.DATA_ROOT)
    earnings = load_earnings(cfg.DATA_ROOT)
    historical_sentiment_scores = load_sentiment(cfg.DATA_ROOT)
    df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)
    df = build_event_table(
        df_all,
        market_ticker=cfg.MARKET_TICKER,
        data_dir=cfg.DATA_ROOT,
        feature_columns=cfg.FEATURE_COLUMNS,
        benchmark_source=cfg.BENCHMARK_SOURCE,
    )

    result = run_preprocess(
        df,
        target=cfg.TARGET,
        feature_columns=cfg.FEATURE_COLUMNS,
        split_column=cfg.SPLIT_COLUMN,
        train_end=cfg.TRAIN_END,
        val_end=cfg.VAL_END,
        impute_strategy=cfg.IMPUTE_STRATEGY,
        standardize=cfg.STANDARDIZE,
    )
    if "val" not in result.folds or "test" not in result.folds:
        raise RuntimeError(
            "Expected train/val/test folds; check TRAIN_END / VAL_END in config."
        )

    train_y_mean = float(np.asarray(result.folds["train"].y, dtype=float).mean())
    summary_rows: list[dict[str, float | int | str]] = []
    prediction_blocks: list[pd.DataFrame] = []

    for model_type in model_types:
        summary, predictions = run_one_model(
            spec,
            model_type=model_type,
            result=result,
            train_y_mean=train_y_mean,
            task=task,
        )
        summary_rows.append(summary)
        prediction_blocks.append(predictions)

    metrics = pd.DataFrame(summary_rows)
    predictions_all = pd.concat(prediction_blocks, axis=0).reset_index(drop=True)
    print_banner("SUMMARY")
    print_summary_table(metrics)

    run_config = {
        "MODEL_LABEL": cfg.MODEL_LABEL,
        spec.model_types_key: list(model_types),
        **spec.run_config_extra,
        "TARGET": cfg.TARGET,
        "BENCHMARK_SOURCE": cfg.BENCHMARK_SOURCE,
        "MARKET_TICKER": cfg.MARKET_TICKER,
        "DATA_ROOT": str(cfg.DATA_ROOT),
        "FEATURE_COLUMNS": list(cfg.FEATURE_COLUMNS),
        "SPLIT_COLUMN": cfg.SPLIT_COLUMN,
        "TRAIN_END": cfg.TRAIN_END,
        "VAL_END": cfg.VAL_END,
        "IMPUTE_STRATEGY": cfg.IMPUTE_STRATEGY,
        "STANDARDIZE": cfg.STANDARDIZE,
        "RANDOM_STATE": cfg.RANDOM_STATE,
        "train_y_mean": train_y_mean,
        "n_before_target_drop": result.n_before_target_drop,
        "n_dropped_target": result.n_dropped_target,
        "folds": fold_date_range(result, cfg.SPLIT_COLUMN),
    }
    save_run_artifacts(
        output_dir=Path(cfg.OUTPUT_DIR),
        metrics=metrics,
        predictions=predictions_all,
        run_config=run_config,
    )


def run_family_by_name(family: FamilyName | Sequence[FamilyName]) -> None:
    """Run one or more families sequentially (``main.py`` / ``main_*.py``).

    Each family keeps its own preprocess (``STANDARDIZE``), SUMMARY, and
    ``output/{lin,tr,nn}/`` artifacts.
    """
    if isinstance(family, str):
        families: tuple[FamilyName, ...] = (family,)
    else:
        families = tuple(family)
    if not families:
        raise ValueError("families must be non-empty; expected lin / tr / nn.")

    unknown = [name for name in families if name not in ("lin", "tr", "nn")]
    if unknown:
        raise ValueError(
            f"Unknown family name(s) {unknown!r}; expected lin / tr / nn."
        )

    for name in families:
        run_family(get_family_spec(name))
