"""Linear estimators for the IMP event table (ols / ridge / lasso / logistic)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

from src.metrics import evaluate_regression

ALLOWED_LINEAR_TYPES = frozenset({"ols", "ridge", "lasso", "logistic"})
CLASSIFICATION_TYPES = frozenset({"logistic"})
ALPHA_TUNABLE_TYPES = frozenset({"ridge", "lasso"})


def create_linear_model(
    model_type: str,
    *,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Any:
    """Build an unfitted sklearn linear estimator."""
    if model_type not in ALLOWED_LINEAR_TYPES:
        raise ValueError(
            f"Unknown LINEAR_MODEL_TYPE {model_type!r}; "
            f"allowed: {sorted(ALLOWED_LINEAR_TYPES)}"
        )
    if model_type == "ols":
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=alpha, random_state=random_state)
    if model_type == "lasso":
        return Lasso(alpha=alpha, random_state=random_state, max_iter=10_000)
    return LogisticRegression(random_state=random_state, max_iter=2_000)


def fit_linear_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """Fit ``model`` on training features / target."""
    model.fit(X.to_numpy(dtype=float), np.asarray(y))
    return model


def predict_linear(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Point predictions (class labels for logistic, continuous otherwise)."""
    return model.predict(X.to_numpy(dtype=float))


def select_alpha_on_val(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    alpha_grid: tuple[float, ...] | list[float],
    default_alpha: float,
    random_state: int = 42,
) -> tuple[float, pd.DataFrame]:
    """Pick alpha on validation by maximizing ``oos_r2`` (fallback: min RMSE).

    Returns ``(best_alpha, tuning_table)``. For non-tunable models, returns
    ``default_alpha`` and an empty table.
    """
    if model_type not in ALPHA_TUNABLE_TYPES or not alpha_grid:
        return float(default_alpha), pd.DataFrame()

    train_mean = float(np.asarray(y_train, dtype=float).mean())
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        model = create_linear_model(
            model_type, alpha=float(alpha), random_state=random_state
        )
        fit_linear_model(model, X_train, y_train)
        metrics = evaluate_regression(
            y_val,
            predict_linear(model, X_val),
            train_y_mean=train_mean,
        )
        rows.append(
            {
                "alpha": float(alpha),
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_ic": metrics["ic"],
                "val_oos_r2": metrics["oos_r2"],
            }
        )
    table = pd.DataFrame(rows)
    # Prefer highest OOS R²; break ties with lower RMSE.
    table = table.sort_values(
        ["val_oos_r2", "val_rmse"], ascending=[False, True]
    ).reset_index(drop=True)
    best_alpha = float(table.loc[0, "alpha"])
    return best_alpha, table
