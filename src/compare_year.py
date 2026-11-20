"""Stock- and quarter-level comparison of PAR vs IMP on one held-out year."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.sectors import attach_sectors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVERSAL_EVENTS = PROJECT_ROOT / "output" / "reversal" / "events.parquet"
LIN_PREDICTIONS = PROJECT_ROOT / "output" / "lin" / "predictions.parquet"


def _signed(direction: np.ndarray, realized: np.ndarray) -> np.ndarray:
    out = np.where(direction == 0, np.nan, direction * realized)
    return out.astype(float)


def year_events(
    events: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    fold: str = "test",
    imp_model: str = "ols",
) -> pd.DataFrame:
    """One row per announcement: realized return, PAR trade, IMP trade."""
    if events is None:
        events = pd.read_parquet(REVERSAL_EVENTS)
    if predictions is None:
        predictions = pd.read_parquet(LIN_PREDICTIONS)

    year = events.loc[events["fold"] == fold].copy()
    year = attach_sectors(year)
    year["EventDate"] = pd.to_datetime(year["EventDate"])
    year["Quarter"] = year["EventDate"].dt.to_period("Q").astype(str)

    pred = predictions.loc[
        (predictions["fold"] == fold) & (predictions["model"] == imp_model),
        ["TickerId", "EventDate", "y_pred"],
    ].copy()
    pred["EventDate"] = pd.to_datetime(pred["EventDate"])
    year = year.merge(pred, on=["TickerId", "EventDate"], how="left")

    par = year["par_mkt"].to_numpy(dtype=float)
    realized = year["ret_ann_mkt"].to_numpy(dtype=float)
    year["par_direction"] = np.where(np.isfinite(par), -np.sign(par), 0).astype(int)
    year["imp_direction"] = np.where(np.isfinite(year["y_pred"]), np.sign(year["y_pred"]), 0).astype(int)
    year["par_return"] = _signed(year["par_direction"].to_numpy(), realized)
    year["imp_return"] = _signed(year["imp_direction"].to_numpy(), realized)
    year["par_minus_imp"] = year["par_return"] - year["imp_return"]
    year["ahead"] = np.where(
        year["par_minus_imp"] > 0,
        "PAR",
        np.where(year["par_minus_imp"] < 0, "IMP", "tie"),
    )
    return year.sort_values(["Quarter", "Symbol", "EventDate"]).reset_index(drop=True)


def stock_table(year: pd.DataFrame) -> pd.DataFrame:
    """One row per name: average 2024 trade return under each rule."""
    rows = []
    for _, part in year.groupby(["TickerId", "Symbol", "Company", "Sector", "Industry"], dropna=False):
        rows.append({
            "Symbol": part["Symbol"].iloc[0],
            "Company": part["Company"].iloc[0],
            "Sector": part["Sector"].iloc[0],
            "Industry": part["Industry"].iloc[0],
            "quarters": int(part["Quarter"].nunique()),
            "events": int(len(part)),
            "avg_realized": float(part["ret_ann_mkt"].mean()),
            "avg_par": float(part["par_return"].mean()),
            "avg_imp": float(part["imp_return"].mean()),
            "par_minus_imp": float((part["par_return"] - part["imp_return"]).mean()),
            "par_ahead_events": int((part["ahead"] == "PAR").sum()),
            "imp_ahead_events": int((part["ahead"] == "IMP").sum()),
        })
    out = pd.DataFrame(rows)
    out["ahead"] = np.where(out["par_minus_imp"] > 0, "PAR", np.where(out["par_minus_imp"] < 0, "IMP", "tie"))
    return out.sort_values("par_minus_imp", ascending=False).reset_index(drop=True)


def sector_table(stocks: pd.DataFrame) -> pd.DataFrame:
    """One row per GICS sector: how many names PAR beats."""
    rows = []
    for sector, part in stocks.groupby("Sector"):
        rows.append({
            "Sector": sector,
            "stocks": int(len(part)),
            "events": int(part["events"].sum()),
            "avg_par": float(part["avg_par"].mean()),
            "avg_imp": float(part["avg_imp"].mean()),
            "par_minus_imp": float(part["par_minus_imp"].mean()),
            "par_ahead_stocks": int((part["ahead"] == "PAR").sum()),
            "imp_ahead_stocks": int((part["ahead"] == "IMP").sum()),
        })
    return pd.DataFrame(rows).sort_values("par_minus_imp", ascending=False).reset_index(drop=True)


def quarter_table(year: pd.DataFrame) -> pd.DataFrame:
    """Symbol × quarter: the table the UI shows."""
    frame = year.copy()
    frame["EventDate"] = pd.to_datetime(frame["EventDate"]).dt.strftime("%Y-%m-%d")
    return frame[[
        "Quarter",
        "EventDate",
        "Symbol",
        "Company",
        "Sector",
        "Industry",
        "ret_ann_mkt",
        "par_mkt",
        "par_return",
        "y_pred",
        "imp_return",
        "par_minus_imp",
        "ahead",
    ]].rename(columns={
        "ret_ann_mkt": "realized",
        "par_mkt": "par",
        "y_pred": "imp_forecast",
    })
