"""Train IMP models for one or more families (linear / tree / neural).

Edit ``FAMILIES`` below, then run from the repo root:

    python main.py

Family-specific hyperparameters stay in ``config/config_{lin,tr,nn}.py``.
Shared defaults: ``config/default.py``.
Pipeline: ``src/pipeline.py``; family fit adapters: ``src/train.py``.
"""

from __future__ import annotations

from src.pipeline import run_family_by_name
from src.train import FamilyName

# ======================================== [CONFIG] ========================================
# Which model families to run (in order). Comment out entries you do not want.
FAMILIES: tuple[FamilyName, ...] = (
    "lin",
    # "tr",
    # "nn",
)
# ========================================================================================


def main() -> None:
    run_family_by_name(FAMILIES)


if __name__ == "__main__":
    main()
