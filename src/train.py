"""Family-specific tune / fit adapters (lin / tr / nn).

Pipeline orchestration lives in ``src/pipeline.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.preprocess import PreprocessResult

FamilyName = Literal["lin", "tr", "nn", "custom"]

PredictFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class FitBundle:
    """Result of family-specific tune + fit for one ``model_type``."""

    model: Any
    predict: PredictFn
    tuning: pd.DataFrame | None = None
    selected: float | int | None = None
    tuning_label: str = "ALPHA"
    best_iteration: int | None = None
    max_iter: int | None = None
    summary_extras: dict[str, float | int | str] = field(default_factory=dict)


FitOneFn = Callable[[str, PreprocessResult], FitBundle]


@dataclass(frozen=True)
class FamilySpec:
    name: FamilyName
    config: Any
    model_types: tuple[str, ...]
    model_types_key: str
    fit_one: FitOneFn
    config_extra_lines: tuple[tuple[str, Any], ...]
    run_config_extra: dict[str, Any]


def fit_lin(
    model_type: str,
    result: PreprocessResult,
    config: ModuleType | None = None,
) -> FitBundle:
    if config is None:
        import config.config_lin as config
    C = config
    from model.model_lin import (
        ALPHA_TUNABLE_TYPES,
        CLASSIFICATION_TYPES,
        create_linear_model,
        fit_linear_model,
        predict_linear,
        select_alpha_on_val,
    )

    if model_type in CLASSIFICATION_TYPES:
        raise ValueError(
            f"model_type={model_type!r} needs classification support; "
            "use ols | ridge | lasso."
        )

    train = result.folds["train"]
    val = result.folds["val"]
    selected_alpha = float(C.ALPHA)
    tuning: pd.DataFrame | None = None
    if model_type in ALPHA_TUNABLE_TYPES:
        selected_alpha, tuning = select_alpha_on_val(
            model_type,
            train.X,
            train.y,
            val.X,
            val.y,
            alpha_grid=tuple(C.ALPHA_GRID),
            default_alpha=float(C.ALPHA),
            random_state=C.RANDOM_STATE,
        )

    model = create_linear_model(
        model_type,
        alpha=selected_alpha,
        random_state=C.RANDOM_STATE,
    )
    fit_linear_model(model, train.X, train.y)
    return FitBundle(
        model=model,
        predict=lambda X, m=model: predict_linear(m, X),
        tuning=tuning,
        selected=selected_alpha if tuning is not None else None,
        tuning_label="ALPHA",
        summary_extras={"alpha": float(selected_alpha)},
    )


def fit_tr(
    model_type: str,
    result: PreprocessResult,
    config: ModuleType | None = None,
) -> FitBundle:
    if config is None:
        import config.config_tr as config
    C = config
    from model.model_tr import (
        EARLY_STOP_TYPES,
        create_tree_model,
        fit_tree_model,
        predict_tree,
        select_max_depth_on_val,
    )

    train = result.folds["train"]
    val = result.folds["val"]
    selected_depth, tuning = select_max_depth_on_val(
        model_type,
        train.X,
        train.y,
        val.X,
        val.y,
        depth_grid=tuple(C.MAX_DEPTH_GRID),
        default_max_depth=int(C.MAX_DEPTH),
        n_estimators=C.N_ESTIMATORS,
        learning_rate=C.LEARNING_RATE,
        min_samples_leaf=C.MIN_SAMPLES_LEAF,
        early_stopping_patience=C.EARLY_STOPPING_PATIENCE,
        random_state=C.RANDOM_STATE,
    )

    model = create_tree_model(
        model_type,
        n_estimators=C.N_ESTIMATORS,
        max_depth=selected_depth,
        learning_rate=C.LEARNING_RATE,
        min_samples_leaf=C.MIN_SAMPLES_LEAF,
        random_state=C.RANDOM_STATE,
    )
    use_val_stop = model_type in EARLY_STOP_TYPES
    model, fit_info = fit_tree_model(
        model,
        train.X,
        train.y,
        X_val=val.X if use_val_stop else None,
        y_val=val.y if use_val_stop else None,
        early_stopping_patience=C.EARLY_STOPPING_PATIENCE,
    )
    best_iteration = (
        int(fit_info["best_iteration"]) if "best_iteration" in fit_info else None
    )
    extras: dict[str, float | int | str] = {
        "max_depth": int(selected_depth),
        "n_estimators": int(C.N_ESTIMATORS),
    }
    if best_iteration is not None:
        extras["best_iteration"] = int(best_iteration)

    return FitBundle(
        model=model,
        predict=lambda X, m=model: predict_tree(m, X),
        tuning=tuning,
        selected=int(selected_depth),
        tuning_label="MAX_DEPTH",
        best_iteration=best_iteration,
        max_iter=int(C.N_ESTIMATORS) if best_iteration is not None else None,
        summary_extras=extras,
    )


def fit_nn(
    model_type: str,
    result: PreprocessResult,
    config: ModuleType | None = None,
) -> FitBundle:
    if config is None:
        import config.config_nn as config
    C = config
    from model.model_nn import (
        create_nn_model,
        fit_nn_model,
        predict_nn,
        select_alpha_on_val,
    )

    train = result.folds["train"]
    val = result.folds["val"]
    selected_alpha, tuning = select_alpha_on_val(
        model_type,
        train.X,
        train.y,
        val.X,
        val.y,
        alpha_grid=tuple(C.ALPHA_GRID),
        default_alpha=float(C.ALPHA),
        hidden_sizes=dict(C.HIDDEN_SIZES),
        learning_rate_init=float(C.LR),
        batch_size=int(C.BATCH_SIZE),
        max_iter=int(C.MAX_ITER),
        early_stopping_patience=int(C.EARLY_STOPPING_PATIENCE),
        random_state=C.RANDOM_STATE,
    )

    model = create_nn_model(
        model_type,
        hidden_sizes=dict(C.HIDDEN_SIZES),
        alpha=float(selected_alpha),
        learning_rate_init=float(C.LR),
        batch_size=int(C.BATCH_SIZE),
        max_iter=int(C.MAX_ITER),
        random_state=C.RANDOM_STATE,
    )
    model, fit_info = fit_nn_model(
        model,
        train.X,
        train.y,
        X_val=val.X,
        y_val=val.y,
        early_stopping_patience=int(C.EARLY_STOPPING_PATIENCE),
    )
    best_iteration = int(fit_info["best_iteration"])
    hidden = C.HIDDEN_SIZES.get(model_type, ())
    return FitBundle(
        model=model,
        predict=lambda X, m=model: predict_nn(m, X),
        tuning=tuning,
        selected=float(selected_alpha),
        tuning_label="ALPHA",
        best_iteration=best_iteration,
        max_iter=int(C.MAX_ITER),
        summary_extras={
            "hidden": str(tuple(hidden)),
            "alpha": float(selected_alpha),
            "best_iteration": int(best_iteration),
        },
    )


def _build_lin_spec() -> FamilySpec:
    import config.config_lin as C

    def fit_one(model_type: str, result: PreprocessResult) -> FitBundle:
        return fit_lin(model_type, result, C)

    return FamilySpec(
        name="lin",
        config=C,
        model_types=tuple(C.LINEAR_MODEL_TYPES),
        model_types_key="LINEAR_MODEL_TYPES",
        fit_one=fit_one,
        config_extra_lines=(
            ("LINEAR_MODEL_TYPES", tuple(C.LINEAR_MODEL_TYPES)),
            ("ALPHA", C.ALPHA),
            ("ALPHA_GRID", tuple(C.ALPHA_GRID)),
        ),
        run_config_extra={
            "ALPHA": C.ALPHA,
            "ALPHA_GRID": list(C.ALPHA_GRID),
        },
    )


def _build_tr_spec() -> FamilySpec:
    import config.config_tr as C

    def fit_one(model_type: str, result: PreprocessResult) -> FitBundle:
        return fit_tr(model_type, result, C)

    return FamilySpec(
        name="tr",
        config=C,
        model_types=tuple(C.TREE_MODEL_TYPES),
        model_types_key="TREE_MODEL_TYPES",
        fit_one=fit_one,
        config_extra_lines=(
            ("TREE_MODEL_TYPES", tuple(C.TREE_MODEL_TYPES)),
            ("N_ESTIMATORS", C.N_ESTIMATORS),
            ("MAX_DEPTH", C.MAX_DEPTH),
            ("MAX_DEPTH_GRID", tuple(C.MAX_DEPTH_GRID)),
            ("LEARNING_RATE", C.LEARNING_RATE),
            ("MIN_SAMPLES_LEAF", C.MIN_SAMPLES_LEAF),
            ("EARLY_STOPPING_PATIENCE", C.EARLY_STOPPING_PATIENCE),
        ),
        run_config_extra={
            "N_ESTIMATORS": C.N_ESTIMATORS,
            "MAX_DEPTH": C.MAX_DEPTH,
            "MAX_DEPTH_GRID": list(C.MAX_DEPTH_GRID),
            "LEARNING_RATE": C.LEARNING_RATE,
            "MIN_SAMPLES_LEAF": C.MIN_SAMPLES_LEAF,
            "EARLY_STOPPING_PATIENCE": C.EARLY_STOPPING_PATIENCE,
        },
    )


def _build_nn_spec() -> FamilySpec:
    import config.config_nn as C

    def fit_one(model_type: str, result: PreprocessResult) -> FitBundle:
        return fit_nn(model_type, result, C)

    return FamilySpec(
        name="nn",
        config=C,
        model_types=tuple(C.NN_MODEL_TYPES),
        model_types_key="NN_MODEL_TYPES",
        fit_one=fit_one,
        config_extra_lines=(
            ("NN_MODEL_TYPES", tuple(C.NN_MODEL_TYPES)),
            ("HIDDEN_SIZES", dict(C.HIDDEN_SIZES)),
            ("ALPHA", C.ALPHA),
            ("ALPHA_GRID", tuple(C.ALPHA_GRID)),
            ("LR", C.LR),
            ("BATCH_SIZE", C.BATCH_SIZE),
            ("MAX_ITER", C.MAX_ITER),
            ("EARLY_STOPPING_PATIENCE", C.EARLY_STOPPING_PATIENCE),
        ),
        run_config_extra={
            "HIDDEN_SIZES": {k: list(v) for k, v in C.HIDDEN_SIZES.items()},
            "ALPHA": C.ALPHA,
            "ALPHA_GRID": list(C.ALPHA_GRID),
            "LR": C.LR,
            "BATCH_SIZE": C.BATCH_SIZE,
            "MAX_ITER": C.MAX_ITER,
            "EARLY_STOPPING_PATIENCE": C.EARLY_STOPPING_PATIENCE,
        },
    )


_SPEC_BUILDERS: dict[FamilyName, Callable[[], FamilySpec]] = {
    "lin": _build_lin_spec,
    "tr": _build_tr_spec,
    "nn": _build_nn_spec,
}


def get_family_spec(family: FamilyName) -> FamilySpec:
    try:
        return _SPEC_BUILDERS[family]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown family {family!r}; expected one of {sorted(_SPEC_BUILDERS)}."
        ) from exc


@dataclass(frozen=True)
class DemoSetup:
    """Resolved notebook CONFIG: active families, step-5 standardize, data dir."""

    active_families: tuple[tuple[FamilyName, dict[str, Any]], ...]
    standardize: bool
    data_dir: Path


def resolve_demo_setup(
    families: Sequence[str],
    *,
    lin: Mapping[str, Any],
    tr: Mapping[str, Any],
    nn: Mapping[str, Any],
    root: Path,
    custom: Mapping[str, Any] | None = None,
) -> DemoSetup:
    """Validate ``FAMILIES`` + family dicts, check parquet inputs under ``root/data``.

    ``custom`` is demo-only. The notebook may define the real block later in §5+;
    if ``custom`` is omitted but ``\"custom\"`` is in ``FAMILIES``, a stub is used
    until §5+ replaces it.
    """
    from config.default import EARNINGS_FILE, SENTIMENT_FILE, STOCK_FILE

    blocks: dict[str, Mapping[str, Any]] = {"lin": lin, "tr": tr, "nn": nn}
    if "custom" in families:
        blocks["custom"] = (
            dict(custom)
            if custom is not None
            else {"MODEL_TYPES": ("pending",), "STANDARDIZE": True}
        )
    allowed = set(blocks)
    unknown = [name for name in families if name not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown FAMILIES entry(ies) {unknown!r}; "
            "expected lin / tr / nn / custom."
        )
    if not families:
        raise ValueError(
            "FAMILIES must be non-empty; pick at least one of lin / tr / nn / custom."
        )

    active: list[tuple[FamilyName, dict[str, Any]]] = []
    for name in families:
        block = blocks[name]
        if not isinstance(block, Mapping):
            raise TypeError(
                f"{name!r} config block must be a mapping, got {type(block)!r}."
            )
        if "STANDARDIZE" not in block:
            raise KeyError(f"{name!r} config block must include STANDARDIZE.")
        if "MODEL_TYPES" not in block:
            raise KeyError(f"{name!r} config block must include MODEL_TYPES.")
        active.append((name, dict(block)))  # type: ignore[arg-type]

    data_dir = Path(root) / "data"
    required = (STOCK_FILE, EARNINGS_FILE, SENTIMENT_FILE)
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing under {data_dir}: {missing}. "
            "Obtain the parquet inputs and place them in data/."
        )

    standardize = bool(active[0][1]["STANDARDIZE"])
    return DemoSetup(
        active_families=tuple(active),
        standardize=standardize,
        data_dir=data_dir,
    )


def build_demo_custom_spec(
    block: Mapping[str, Any],
    *,
    fit_one: FitOneFn,
    feature_importance: bool,
    feature_importance_repeats: int,
    random_state: int,
) -> FamilySpec:
    """Build a demo-only ``FamilySpec`` from a notebook ``fit_one`` (no config_*)."""
    model_types = tuple(block["MODEL_TYPES"])
    if not model_types:
        raise ValueError("custom MODEL_TYPES must be non-empty.")
    cfg = SimpleNamespace(
        FEATURE_IMPORTANCE=bool(feature_importance),
        FEATURE_IMPORTANCE_REPEATS=int(feature_importance_repeats),
        RANDOM_STATE=int(random_state),
        STANDARDIZE=bool(block["STANDARDIZE"]),
        MODEL_LABEL="demo_custom",
    )
    extras = tuple(
        (key, block[key])
        for key in block
        if key not in {"MODEL_TYPES", "STANDARDIZE"}
    )
    return FamilySpec(
        name="custom",
        config=cfg,
        model_types=model_types,
        model_types_key="MODEL_TYPES",
        fit_one=fit_one,
        config_extra_lines=(("MODEL_TYPES", model_types),) + extras,
        run_config_extra={k: v for k, v in block.items() if k != "MODEL_TYPES"},
    )


def apply_demo_block(
    family: FamilyName,
    block: dict[str, Any],
    *,
    feature_importance: bool,
    feature_importance_repeats: int,
    random_state: int,
) -> FamilySpec:
    """Write a demo family dict into ``config.config_*`` and return a fresh spec."""
    if family == "lin":
        import config.config_lin as C

        C.LINEAR_MODEL_TYPES = tuple(block["MODEL_TYPES"])
        C.ALPHA = float(block["ALPHA"])
        C.ALPHA_GRID = tuple(block["ALPHA_GRID"])
        C.STANDARDIZE = bool(block["STANDARDIZE"])
    elif family == "tr":
        import config.config_tr as C

        C.TREE_MODEL_TYPES = tuple(block["MODEL_TYPES"])
        C.N_ESTIMATORS = int(block["N_ESTIMATORS"])
        C.MAX_DEPTH = int(block["MAX_DEPTH"])
        C.MAX_DEPTH_GRID = tuple(block["MAX_DEPTH_GRID"])
        C.LEARNING_RATE = float(block["LEARNING_RATE"])
        C.MIN_SAMPLES_LEAF = int(block["MIN_SAMPLES_LEAF"])
        C.EARLY_STOPPING_PATIENCE = int(block["EARLY_STOPPING_PATIENCE"])
        C.STANDARDIZE = bool(block["STANDARDIZE"])
    elif family == "nn":
        import config.config_nn as C

        C.NN_MODEL_TYPES = tuple(block["MODEL_TYPES"])
        C.HIDDEN_SIZES = dict(block["HIDDEN_SIZES"])
        C.ALPHA = float(block["ALPHA"])
        C.ALPHA_GRID = tuple(block["ALPHA_GRID"])
        C.LR = float(block["LR"])
        C.BATCH_SIZE = int(block["BATCH_SIZE"])
        C.MAX_ITER = int(block["MAX_ITER"])
        C.EARLY_STOPPING_PATIENCE = int(block["EARLY_STOPPING_PATIENCE"])
        C.STANDARDIZE = bool(block["STANDARDIZE"])
    else:
        raise ValueError(f"Unknown family {family!r}.")

    C.FEATURE_IMPORTANCE = bool(feature_importance)
    C.FEATURE_IMPORTANCE_REPEATS = int(feature_importance_repeats)
    C.RANDOM_STATE = int(random_state)
    return get_family_spec(family)
