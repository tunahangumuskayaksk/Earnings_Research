"""Tree model settings for the IMP event table.

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
OUTPUT_DIR = PROJECT_ROOT / "output" / "tr"

# Batch label for run_config / logs. Any non-empty string.
MODEL_LABEL = "tr_batch"

# Models to run under the same preprocess config (one section each).
# Allowed: "decision_tree" | "random_forest" | "hist_gradient_boosting"
TREE_MODEL_TYPES = (
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
)

# Number of trees / boosting stages (cap for HistGBDT early stopping).
# Allowed: int >= 1. Ignored for decision_tree.
N_ESTIMATORS = 200

# Default max depth when MAX_DEPTH_GRID is empty.
# Allowed: int >= 1
MAX_DEPTH = 4

# Candidate depths tuned on chronological validation (all tree types).
# Allowed: non-empty sequence of ints >= 1, or () to use MAX_DEPTH as-is.
MAX_DEPTH_GRID = (2, 3, 4, 6, 8)

# Boosting step size. Allowed: float in (0, 1]. Ignored for non-boosted trees.
LEARNING_RATE = 0.05

# Minimum samples in a leaf. Allowed: int >= 1
MIN_SAMPLES_LEAF = 20

# HistGBDT: stop if chronological val OOS R² does not improve.
# Allowed: int >= 1
EARLY_STOPPING_PATIENCE = 10

# Z-score features. Allowed: True | False (usually False for trees)
STANDARDIZE = False
