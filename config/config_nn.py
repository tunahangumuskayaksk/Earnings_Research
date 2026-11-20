"""Neural model settings for the IMP event table.

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
OUTPUT_DIR = PROJECT_ROOT / "output" / "nn"

# Batch label for run_config / logs. Any non-empty string.
MODEL_LABEL = "nn_batch"

# Models to run under the same preprocess config (one section each).
# Allowed: "mlp_shallow" | "mlp" | "mlp_deep"
NN_MODEL_TYPES = ("mlp_shallow", "mlp", "mlp_deep")

# Hidden widths keyed by model type (sklearn MLPRegressor).
HIDDEN_SIZES = {
    "mlp_shallow": (64,),
    "mlp": (64, 32),
    "mlp_deep": (128, 64, 32),
}

# Default L2 penalty when ALPHA_GRID is empty.
# Allowed: float >= 0
ALPHA = 1e-4

# Candidate L2 penalties tuned on chronological validation (all MLP types).
# Allowed: non-empty sequence of floats >= 0, or () to use ALPHA as-is.
ALPHA_GRID = (1e-5, 1e-4, 1e-3, 1e-2)

# Adam learning rate. Allowed: float > 0
LR = 1e-3

# Mini-batch size. Allowed: int >= 1
BATCH_SIZE = 64

# Maximum training iterations (epochs). Allowed: int >= 1
MAX_ITER = 200

# Stop if chronological val OOS R² does not improve for this many epochs.
# Allowed: int >= 1
EARLY_STOPPING_PATIENCE = 8

# Z-score features with train-set mean/std. Allowed: True | False
STANDARDIZE = True
