"""Neural estimators for the IMP event table (sklearn MLPRegressor)."""

from __future__ import annotations

from typing import Any
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from src.metrics import evaluate_regression

ALLOWED_NN_TYPES = frozenset({"mlp_shallow", "mlp", "mlp_deep"})
ALPHA_TUNABLE_TYPES = ALLOWED_NN_TYPES

DEFAULT_HIDDEN_SIZES: dict[str, tuple[int, ...]] = {
    "mlp_shallow": (64,),
    "mlp": (64, 32),
    "mlp_deep": (128, 64, 32),
}


def create_nn_model(
    model_type: str,
    *,
    hidden_sizes: dict[str, tuple[int, ...]] | None = None,
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    batch_size: int = 64,
    max_iter: int = 200,
    random_state: int = 42,
) -> Any:
    """Build an unfitted sklearn MLPRegressor.

    Does **not** use sklearn's random ``validation_fraction`` early stopping;
    call ``fit_nn_model`` with the chronological val fold.
    """
    if model_type not in ALLOWED_NN_TYPES:
        raise ValueError(
            f"Unknown NN_MODEL_TYPE {model_type!r}; "
            f"allowed: {sorted(ALLOWED_NN_TYPES)}"
        )
    sizes_map = DEFAULT_HIDDEN_SIZES if hidden_sizes is None else hidden_sizes
    if model_type not in sizes_map:
        raise KeyError(f"HIDDEN_SIZES missing entry for {model_type!r}")
    hidden = tuple(int(x) for x in sizes_map[model_type])
    if not hidden or any(h < 1 for h in hidden):
        raise ValueError(f"Invalid hidden sizes for {model_type!r}: {hidden}")

    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=False,
        warm_start=False,
        random_state=random_state,
    )


def _clone_mlp_template(model: MLPRegressor) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=model.hidden_layer_sizes,
        activation=model.activation,
        solver=model.solver,
        alpha=model.alpha,
        batch_size=model.batch_size,
        learning_rate_init=model.learning_rate_init,
        max_iter=1,
        early_stopping=False,
        warm_start=True,
        random_state=model.random_state,
    )


def fit_nn_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_patience: int = 8,
) -> tuple[Any, dict[str, float | int]]:
    """Fit on train; early-stop by chronological val OOS R² (restore best weights)."""
    X_tr = X_train.to_numpy(dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    X_va = X_val.to_numpy(dtype=float)
    y_va = np.asarray(y_val, dtype=float)
    train_mean = float(y_tr.mean())
    max_epochs = int(model.max_iter)

    probe = _clone_mlp_template(model)
    best_epoch = 1
    best_oos = float("-inf")
    best_rmse = float("inf")
    best_coefs: list[np.ndarray] | None = None
    best_intercepts: list[np.ndarray] | None = None
    stale = 0

    # warm_start + rising max_iter intentionally stops before tol; silence that noise.
    with catch_warnings():
        filterwarnings("ignore", category=ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            probe.set_params(max_iter=epoch)
            probe.fit(X_tr, y_tr)
            metrics = evaluate_regression(
                y_va, probe.predict(X_va), train_y_mean=train_mean
            )
            oos = float(metrics["oos_r2"])
            rmse = float(metrics["rmse"])
            improved = oos > best_oos + 1e-12 or (
                np.isclose(oos, best_oos) and rmse < best_rmse
            )
            if improved:
                best_epoch = epoch
                best_oos = oos
                best_rmse = rmse
                best_coefs = [c.copy() for c in probe.coefs_]
                best_intercepts = [b.copy() for b in probe.intercepts_]
                stale = 0
            else:
                stale += 1
                if stale >= early_stopping_patience:
                    break

    if best_coefs is None or best_intercepts is None:
        raise RuntimeError("MLP early stopping failed to record any weights.")

    probe.coefs_ = best_coefs
    probe.intercepts_ = best_intercepts
    probe.n_iter_ = best_epoch
    probe.set_params(max_iter=max_epochs, warm_start=False)
    return probe, {"best_iteration": int(best_epoch)}


def predict_nn(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Continuous predictions."""
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
    hidden_sizes: dict[str, tuple[int, ...]] | None = None,
    learning_rate_init: float = 1e-3,
    batch_size: int = 64,
    max_iter: int = 200,
    early_stopping_patience: int = 8,
    random_state: int = 42,
) -> tuple[float, pd.DataFrame]:
    """Pick L2 ``alpha`` on chronological val by maximizing ``oos_r2``."""
    if model_type not in ALPHA_TUNABLE_TYPES or not alpha_grid:
        return float(default_alpha), pd.DataFrame()

    train_mean = float(np.asarray(y_train, dtype=float).mean())
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        model = create_nn_model(
            model_type,
            hidden_sizes=hidden_sizes,
            alpha=float(alpha),
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
        )
        fitted, _ = fit_nn_model(
            model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_patience=early_stopping_patience,
        )
        metrics = evaluate_regression(
            y_val,
            predict_nn(fitted, X_val),
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
    table = table.sort_values(
        ["val_oos_r2", "val_rmse"], ascending=[False, True]
    ).reset_index(drop=True)
    best_alpha = float(table.loc[0, "alpha"])
    return best_alpha, table
