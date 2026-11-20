"""Tree estimators for the IMP event table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.metrics import evaluate_regression

ALLOWED_TREE_TYPES = frozenset(
    {"decision_tree", "random_forest", "hist_gradient_boosting"}
)
# All batch members select max_depth on the chronological val fold.
DEPTH_TUNABLE_TYPES = ALLOWED_TREE_TYPES
EARLY_STOP_TYPES = frozenset({"hist_gradient_boosting"})


def create_tree_model(
    model_type: str,
    *,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> Any:
    """Build an unfitted sklearn tree / ensemble regressor.

    HistGradientBoosting does **not** use sklearn's random ``validation_fraction``
    early stopping; call ``fit_tree_model`` with the chronological val fold.
    """
    if model_type not in ALLOWED_TREE_TYPES:
        raise ValueError(
            f"Unknown TREE_MODEL_TYPE {model_type!r}; "
            f"allowed: {sorted(ALLOWED_TREE_TYPES)}"
        )
    if model_type == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        warm_start=False,
        random_state=random_state,
    )


def _fit_hgb_on_chronological_val(
    *,
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    min_samples_leaf: int,
    random_state: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_y_mean: float,
    early_stopping_patience: int,
) -> tuple[Any, int]:
    """Grow boosters with warm_start; stop using chronological val OOS R²."""
    probe = HistGradientBoostingRegressor(
        max_iter=1,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        warm_start=True,
        random_state=random_state,
    )
    best_iter = 1
    best_oos = float("-inf")
    best_rmse = float("inf")
    stale = 0
    for n_iter in range(1, int(n_estimators) + 1):
        probe.set_params(max_iter=n_iter)
        probe.fit(X_train, y_train)
        metrics = evaluate_regression(
            y_val, probe.predict(X_val), train_y_mean=train_y_mean
        )
        oos = float(metrics["oos_r2"])
        rmse = float(metrics["rmse"])
        improved = oos > best_oos + 1e-12 or (
            np.isclose(oos, best_oos) and rmse < best_rmse
        )
        if improved:
            best_iter = n_iter
            best_oos = oos
            best_rmse = rmse
            stale = 0
        else:
            stale += 1
            if stale >= early_stopping_patience:
                break

    final = HistGradientBoostingRegressor(
        max_iter=best_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        warm_start=False,
        random_state=random_state,
    )
    final.fit(X_train, y_train)
    return final, best_iter


def fit_tree_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    early_stopping_patience: int = 10,
) -> tuple[Any, dict[str, float | int]]:
    """Fit on train; HistGBDT early-stops on chronological val when provided."""
    X_tr = X_train.to_numpy(dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    info: dict[str, float | int] = {}

    if isinstance(model, HistGradientBoostingRegressor):
        if X_val is None or y_val is None:
            raise ValueError(
                "hist_gradient_boosting requires chronological X_val / y_val "
                "for early stopping."
            )
        X_va = X_val.to_numpy(dtype=float)
        y_va = np.asarray(y_val, dtype=float)
        train_mean = float(y_tr.mean())
        fitted, best_iter = _fit_hgb_on_chronological_val(
            max_depth=int(model.max_depth),
            n_estimators=int(model.max_iter),
            learning_rate=float(model.learning_rate),
            min_samples_leaf=int(model.min_samples_leaf),
            random_state=int(model.random_state),
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            train_y_mean=train_mean,
            early_stopping_patience=int(early_stopping_patience),
        )
        info["best_iteration"] = int(best_iter)
        return fitted, info

    model.fit(X_tr, y_tr)
    return model, info


def predict_tree(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Continuous predictions."""
    return model.predict(X.to_numpy(dtype=float))


def select_max_depth_on_val(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    depth_grid: tuple[int, ...] | list[int],
    default_max_depth: int,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    early_stopping_patience: int = 10,
    random_state: int = 42,
) -> tuple[int, pd.DataFrame]:
    """Pick max_depth on chronological val by maximizing ``oos_r2``.

    Returns ``(best_depth, tuning_table)``. Empty grid -> ``default_max_depth``.
    """
    if model_type not in DEPTH_TUNABLE_TYPES or not depth_grid:
        return int(default_max_depth), pd.DataFrame()

    train_mean = float(np.asarray(y_train, dtype=float).mean())
    rows: list[dict[str, float | int]] = []
    for depth in depth_grid:
        model = create_tree_model(
            model_type,
            n_estimators=n_estimators,
            max_depth=int(depth),
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        fitted, _ = fit_tree_model(
            model,
            X_train,
            y_train,
            X_val=X_val if model_type in EARLY_STOP_TYPES else None,
            y_val=y_val if model_type in EARLY_STOP_TYPES else None,
            early_stopping_patience=early_stopping_patience,
        )
        metrics = evaluate_regression(
            y_val,
            predict_tree(fitted, X_val),
            train_y_mean=train_mean,
        )
        rows.append(
            {
                "max_depth": int(depth),
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_ic": metrics["ic"],
                "val_oos_r2": metrics["oos_r2"],
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["val_oos_r2", "val_rmse"], ascending=[False, True]
    ).reset_index(drop=True)
    best_depth = int(table.loc[0, "max_depth"])
    return best_depth, table
