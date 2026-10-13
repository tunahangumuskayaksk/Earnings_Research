"""The stock universes: the original 30 stocks and 40 added stocks (70 in total).

Each stock is a long-standing S&P 500 constituent that was in the index before
the 2010 start of the event range. Membership comes from general knowledge of the
index, not from a point-in-time constituent dataset, so the universe is chosen
with hindsight: these are companies that are still large today, which is a
survivorship bias the summary pages disclose.

Data coverage was checked before selection: every stock below has daily prices
from 2009 onward and 62 completed earnings records in 2010-2025 (four in every
calendar year 2010-2024), all 62 of which pass the alignment and data checks.

First 30 (2026-10-10 check): candidates with a missing quarter (MRK 2012,
COST 2014) or a release dated on a non-trading day (CVX 2023-07-23,
T 2016-10-22) were left out so that every stock enters on the same footing.

Added 40 (2026-10-11 check, 101 candidates screened, 87 passed): left out were
candidates with an uneven quarterly count (MU), a release on a non-trading day
(F, ADM, VLO, ECL on 2012-10-30, the Hurricane Sandy closure), downloads that
failed during the check (EA, BK, MMC, EQR, AVB), and unadjusted or mis-adjusted
spin-off days in the adjusted close (DHR 2016-07-05 +61%, VTR 2015-08-18 -24%,
HPQ 2015-11-02 +13%). From the passing candidates, 40 were picked for sector
spread, including Materials and Real Estate, without looking at their results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseMember:
    ticker: str
    name: str
    sector: str


@dataclass(frozen=True)
class StockGroup:
    key: str
    members: tuple[UniverseMember, ...]
    selection_notes: str

    @property
    def label(self) -> str:
        return f"{len(self.members)}-stock summary"

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(m.ticker for m in self.members)


UNIVERSE: tuple[UniverseMember, ...] = (
    UniverseMember("AAPL", "Apple", "Information Technology"),
    UniverseMember("MSFT", "Microsoft", "Information Technology"),
    UniverseMember("IBM", "IBM", "Information Technology"),
    UniverseMember("INTC", "Intel", "Information Technology"),
    UniverseMember("CSCO", "Cisco Systems", "Information Technology"),
    UniverseMember("GOOGL", "Alphabet", "Communication Services"),
    UniverseMember("VZ", "Verizon", "Communication Services"),
    UniverseMember("DIS", "Walt Disney", "Communication Services"),
    UniverseMember("AMZN", "Amazon", "Consumer Discretionary"),
    UniverseMember("HD", "Home Depot", "Consumer Discretionary"),
    UniverseMember("MCD", "McDonald's", "Consumer Discretionary"),
    UniverseMember("NKE", "Nike", "Consumer Discretionary"),
    UniverseMember("PG", "Procter & Gamble", "Consumer Staples"),
    UniverseMember("KO", "Coca-Cola", "Consumer Staples"),
    UniverseMember("PEP", "PepsiCo", "Consumer Staples"),
    UniverseMember("WMT", "Walmart", "Consumer Staples"),
    UniverseMember("UNH", "UnitedHealth Group", "Health Care"),
    UniverseMember("PFE", "Pfizer", "Health Care"),
    UniverseMember("ABT", "Abbott Laboratories", "Health Care"),
    UniverseMember("LLY", "Eli Lilly", "Health Care"),
    UniverseMember("JPM", "JPMorgan Chase", "Financials"),
    UniverseMember("BAC", "Bank of America", "Financials"),
    UniverseMember("WFC", "Wells Fargo", "Financials"),
    UniverseMember("AXP", "American Express", "Financials"),
    UniverseMember("XOM", "Exxon Mobil", "Energy"),
    UniverseMember("COP", "ConocoPhillips", "Energy"),
    UniverseMember("CAT", "Caterpillar", "Industrials"),
    UniverseMember("HON", "Honeywell", "Industrials"),
    UniverseMember("BA", "Boeing", "Industrials"),
    UniverseMember("NEE", "NextEra Energy", "Utilities"),
)

ADDED_UNIVERSE: tuple[UniverseMember, ...] = (
    UniverseMember("ORCL", "Oracle", "Information Technology"),
    UniverseMember("QCOM", "Qualcomm", "Information Technology"),
    UniverseMember("TXN", "Texas Instruments", "Information Technology"),
    UniverseMember("ADBE", "Adobe", "Information Technology"),
    UniverseMember("NVDA", "Nvidia", "Information Technology"),
    UniverseMember("CMCSA", "Comcast", "Communication Services"),
    UniverseMember("OMC", "Omnicom Group", "Communication Services"),
    UniverseMember("LOW", "Lowe's", "Consumer Discretionary"),
    UniverseMember("SBUX", "Starbucks", "Consumer Discretionary"),
    UniverseMember("TGT", "Target", "Consumer Discretionary"),
    UniverseMember("TJX", "TJX Companies", "Consumer Discretionary"),
    UniverseMember("BKNG", "Booking Holdings", "Consumer Discretionary"),
    UniverseMember("MO", "Altria", "Consumer Staples"),
    UniverseMember("CL", "Colgate-Palmolive", "Consumer Staples"),
    UniverseMember("KMB", "Kimberly-Clark", "Consumer Staples"),
    UniverseMember("GIS", "General Mills", "Consumer Staples"),
    UniverseMember("JNJ", "Johnson & Johnson", "Health Care"),
    UniverseMember("BMY", "Bristol-Myers Squibb", "Health Care"),
    UniverseMember("AMGN", "Amgen", "Health Care"),
    UniverseMember("GILD", "Gilead Sciences", "Health Care"),
    UniverseMember("MDT", "Medtronic", "Health Care"),
    UniverseMember("GS", "Goldman Sachs", "Financials"),
    UniverseMember("MS", "Morgan Stanley", "Financials"),
    UniverseMember("C", "Citigroup", "Financials"),
    UniverseMember("USB", "U.S. Bancorp", "Financials"),
    UniverseMember("TRV", "Travelers", "Financials"),
    UniverseMember("SLB", "SLB (Schlumberger)", "Energy"),
    UniverseMember("EOG", "EOG Resources", "Energy"),
    UniverseMember("OXY", "Occidental Petroleum", "Energy"),
    UniverseMember("MMM", "3M", "Industrials"),
    UniverseMember("UNP", "Union Pacific", "Industrials"),
    UniverseMember("UPS", "United Parcel Service", "Industrials"),
    UniverseMember("LMT", "Lockheed Martin", "Industrials"),
    UniverseMember("DE", "Deere", "Industrials"),
    UniverseMember("DUK", "Duke Energy", "Utilities"),
    UniverseMember("SO", "Southern Company", "Utilities"),
    UniverseMember("SHW", "Sherwin-Williams", "Materials"),
    UniverseMember("FCX", "Freeport-McMoRan", "Materials"),
    UniverseMember("SPG", "Simon Property Group", "Real Estate"),
    UniverseMember("PSA", "Public Storage", "Real Estate"),
)

_COMMON_NOTES = (
    "- Long-standing S&P 500 constituents that were in the index before the 2010 "
    "start of the event range. Index membership comes from general knowledge of the "
    "index, not from a point-in-time constituent dataset.\n"
    "- Every stock was checked for daily prices from 2009 onward and for 62 "
    "completed earnings records in 2010\u20132025 (four in every year 2010\u20132024), "
    "all passing the alignment and data checks.\n"
)

FIRST_30 = StockGroup(
    key="first30",
    members=UNIVERSE,
    selection_notes=_COMMON_NOTES + (
        "- Spread across all sectors except Real Estate and Materials.\n"
        "- MRK and COST were left out because each is missing one quarter; CVX and "
        "T because each has one release dated on a non-trading day. Including them "
        "would not break anything, but would put stocks on unequal footing.\n"
        "- Large one-day price moves were screened for unadjusted spin-offs; the "
        "ones found are genuine earnings reactions or the March 2020 crash."
    ),
)

ADDED_40 = StockGroup(
    key="added40",
    members=ADDED_UNIVERSE,
    selection_notes=_COMMON_NOTES + (
        "- 101 further candidates were screened and 87 passed. The 40 were picked "
        "from those for sector spread, including Materials and Real Estate, before "
        "any of their results were looked at.\n"
        "- Left out: MU (uneven quarterly count); F, ADM, VLO and ECL (a release on "
        "2012-10-30, when the exchange was closed for Hurricane Sandy); EA, BK, MMC, "
        "EQR and AVB (downloads failed during the check).\n"
        "- Spin-off screen: the adjusted close was checked on known spin-off dates. "
        "DHR (2016-07-05, +61% adjusted against +4% raw), VTR (2015-08-18, \u221224%) "
        "and HPQ (2015-11-02, +13%) showed unadjusted or mis-adjusted moves and were "
        "left out. Spin-offs at EBAY, EXC, MMM, KMB, SPG, MET and others were "
        "properly adjusted."
    ),
)

GROUPS: tuple[StockGroup, ...] = (FIRST_30, ADDED_40)
ALL_MEMBERS: tuple[UniverseMember, ...] = UNIVERSE + ADDED_UNIVERSE

UNIVERSE_TICKERS: tuple[str, ...] = FIRST_30.tickers
ADDED_TICKERS: tuple[str, ...] = ADDED_40.tickers
ALL_TICKERS: tuple[str, ...] = tuple(m.ticker for m in ALL_MEMBERS)
_BY_TICKER = {m.ticker: m for m in ALL_MEMBERS}


def member(ticker: str) -> UniverseMember | None:
    return _BY_TICKER.get(ticker.strip().upper())


def group_of(ticker: str) -> StockGroup | None:
    for group in GROUPS:
        if ticker.strip().upper() in group.tickers:
            return group
    return None


def display_name(ticker: str) -> str:
    found = member(ticker)
    if found is None:
        return ticker
    return f"{found.ticker} \u2014 {found.name} ({found.sector})"
