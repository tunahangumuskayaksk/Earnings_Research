"""Display helpers. Returns are stored as decimals and shown as percentages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from earnings_reversal.stats import Stat


def pct(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def stat_text(stat: Stat, percent: bool = False, digits: int = 4) -> str:
    return stat.format(digits=digits, percent=percent)


def percent_frame(frame: pd.DataFrame, columns, digits: int = 2) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(lambda v: pct(v, digits))
    return out


def date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.date
    return out


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return date_frame(frame).to_csv(index=False).encode("utf-8")


def download(st, frame: pd.DataFrame, label: str, filename: str, key: str) -> None:
    """Optional CSV export. Never required to understand what is on screen."""
    if frame is None or len(frame) == 0:
        return
    st.download_button(label, data=to_csv_bytes(frame), file_name=filename,
                       mime="text/csv", key=key)
