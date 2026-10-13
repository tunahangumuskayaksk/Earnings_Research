"""Short-term return reversals around earnings announcements.

A descriptive, single-company research pilot inspired by So & Wang (2014),
"News-driven return reversals". It is not a replication of that multi-company
study, not a trading strategy, and not evidence of any causal mechanism.
"""

from .config import (  # noqa: F401
    ALL_WINDOWS,
    ANNOUNCEMENT_WINDOW,
    AnalysisConfig,
    LATER_WINDOW,
    OUTCOME_WINDOWS,
    PAR_WINDOW,
    SUBSEQUENT_WINDOW,
    Window,
)

__version__ = "0.1.0"
