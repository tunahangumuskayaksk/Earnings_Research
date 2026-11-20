"""GICS sector and industry labels for the IMP universe.

``src/gics_sectors.csv`` is the S&P 500 list plus eleven names that are not
on it. Aecon (ARE on XTSE) shares a symbol with Alexandria and is labelled
separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SECTOR_FILE = Path(__file__).with_name("gics_sectors.csv")


def sector_table() -> pd.DataFrame:
    return pd.read_csv(SECTOR_FILE)


def attach_sectors(events: pd.DataFrame) -> pd.DataFrame:
    """Add Company, Sector, Industry. Aecon (ARE on XTSE) is construction, not the REIT."""
    mapped = events.merge(sector_table(), on="Symbol", how="left")
    if "ShortName" in mapped.columns:
        aecon = mapped["ShortName"].astype(str).str.contains("XTSE", na=False) & mapped["Symbol"].eq("ARE")
        mapped.loc[aecon, ["Company", "Sector", "Industry"]] = (
            "Aecon Group", "Industrials", "Construction & Engineering"
        )
    mapped["Sector"] = mapped["Sector"].fillna("Unclassified")
    mapped["Industry"] = mapped["Industry"].fillna("Unclassified")
    if "Symbol" in mapped.columns:
        mapped["Company"] = mapped["Company"].fillna(mapped["Symbol"])
    return mapped
