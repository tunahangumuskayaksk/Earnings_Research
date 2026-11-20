"""Linear model settings for the IMP event table.

Shared paths / data / split / target defaults: ``config/default.py``.
"""

from __future__ import annotations

from config.default import (
    BENCHMARK_SOURCE,
    DATA_ROOT,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE,
    FEATURE_IMPORTANCE_REPEATS,
    IMPUTE_STRATEGY,
    MARKET_TICKER,
    PROJECT_ROOT,
    RANDOM_STATE,
    SPLIT_COLUMN,
    TARGET,
    TRAIN_END,
    VAL_END,
)

# Run outputs for this family.
OUTPUT_DIR = PROJECT_ROOT / "output" / "lin"

# Batch label for run_config / logs. Any non-empty string.
MODEL_LABEL = "lin_batch"

# Models to run under the same preprocess config (one section each).
# Allowed entries: "ols" | "ridge" | "lasso"
# ("logistic" reserved until classification metrics exist.)
LINEAR_MODEL_TYPES = ("ols", "ridge", "lasso")

# Default L2/L1 strength when ALPHA_GRID is empty / unused.
# Allowed: float > 0. Ignored for ols.
ALPHA = 1.0

# Candidate alphas tuned on validation only (ridge/lasso).
# Allowed: non-empty sequence of floats > 0, or () to use ALPHA as-is.
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# Z-score features with train-set mean/std. Allowed: True | False
STANDARDIZE = True
