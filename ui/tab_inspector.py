"""Tab D: Event Inspector — manual verification of individual announcements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from earnings_reversal.config import ALL_WINDOWS, ANNOUNCEMENT_WINDOW_CAVEAT
from earnings_reversal.returns import daily_return_frame
from .formatting import date_frame, download, pct

CHART_LO = -12
CHART_HI = 14

WINDOW_COLORS = {
    "par": "rgba(242, 201, 76, 0.22)",
    "announcement": "rgba(194, 59, 34, 0.18)",
    "subsequent": "rgba(42, 111, 181, 0.16)",
    "later": "rgba(86, 160, 109, 0.16)",
}


def _price_chart(panel, position: int, release_date: pd.Timestamp) -> go.Figure:
    lo = max(position + CHART_LO, 0)
    hi = min(position + CHART_HI, len(panel.sessions) - 1)
    sessions = panel.sessions[lo:hi + 1]
    stock = panel.stock_prices[lo:hi + 1]
    bench = panel.benchmark_prices[lo:hi + 1]

    base_index = position + ALL_WINDOWS[0].baseline_offset  # close at -5
    base_stock = panel.stock_prices[base_index]
    base_bench = panel.benchmark_prices[base_index]
    if not np.isfinite(base_stock) or not np.isfinite(base_bench):
        base_stock = stock[np.isfinite(stock)][0] if np.isfinite(stock).any() else np.nan
        base_bench = bench[np.isfinite(bench)][0] if np.isfinite(bench).any() else np.nan

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=sessions, y=100.0 * stock / base_stock, mode="lines+markers",
        name=f"{panel.stock_ticker} (normalised)", line=dict(color="#2a6fb5", width=2),
    ))
    figure.add_trace(go.Scatter(
        x=sessions, y=100.0 * bench / base_bench, mode="lines+markers",
        name=f"{panel.benchmark_ticker} (normalised)", line=dict(color="#888", width=2,
                                                                 dash="dot"),
    ))

    for window in ALL_WINDOWS:
        start_index = position + window.start
        end_index = position + window.end
        if start_index < 0 or end_index >= len(panel.sessions):
            continue
        figure.add_vrect(
            x0=panel.sessions[start_index] - pd.Timedelta(hours=10),
            x1=panel.sessions[end_index] + pd.Timedelta(hours=10),
            fillcolor=WINDOW_COLORS[window.key], line_width=0, layer="below",
            annotation_text=f"{window.label} [{window.start:+d},{window.end:+d}]",
            annotation_position="top left", annotation_font_size=10,
        )
    figure.add_vline(
        x=release_date, line_width=2, line_color="#c23b22",
        annotation_text=f"Release {pd.Timestamp(release_date).date()} (t=0)",
        annotation_position="bottom right",
    )
    figure.update_layout(
        yaxis_title=f"Adjusted close, rebased to 100 at session {ALL_WINDOWS[0].baseline_offset:+d}",
        xaxis_title="Trading session",
        height=480, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="x unified",
    )
    return figure


def _calculation_block(event: pd.Series) -> pd.DataFrame:
    rows = []
    name_map = {"par": "par", "announcement": "announcement_adj",
                "subsequent": "subsequent_adj", "later": "later_adj"}
    for window in ALL_WINDOWS:
        prefix = window.key
        adjusted = event[name_map[prefix]]
        rows.append({
            "Window": f"{window.label} [{window.start:+d},{window.end:+d}]",
            "First return date (session "
            f"{window.start:+d})": pd.Timestamp(event[f"{prefix}_first_return_date"]).date(),
            f"Start price date (session {window.baseline_offset:+d})":
                pd.Timestamp(event[f"{prefix}_start_price_date"]).date(),
            "End price date (session "
            f"{window.end:+d})": pd.Timestamp(event[f"{prefix}_end_price_date"]).date(),
            "Start adjusted close": f"{event[f'{prefix}_start_price']:,.4f}",
            "End adjusted close": f"{event[f'{prefix}_end_price']:,.4f}",
            "Stock return": pct(event[f"{prefix}_stock"]),
            "Benchmark return": pct(event[f"{prefix}_benchmark"]),
            "Market-adjusted": pct(adjusted),
        })
    return pd.DataFrame(rows)


def render(result) -> None:
    st.subheader("Event Inspector")
    if not result.has_events or result.panel is None:
        st.info(
            "No event can be inspected yet. "
            + (result.blocking_error or "No eligible events are available.")
        )
        return

    events = result.included_events
    st.markdown("#### Event table")
    st.caption("Sort by any column; use the search box below to filter by date or flag.")
    query = st.text_input(
        "Filter events (matches release date, session classification, source or flags)",
        value="", key=f"inspector_query_{result.config.ticker}",
    )
    table = events.copy()
    table["release_date_str"] = pd.to_datetime(table["release_date"]).dt.date.astype(str)
    if query.strip():
        needle = query.strip().lower()
        mask = (
            table["release_date_str"].str.lower().str.contains(needle)
            | table["session_class"].astype(str).str.lower().str.contains(needle)
            | table["source"].astype(str).str.lower().str.contains(needle)
            | table["quality_flags"].astype(str).str.lower().str.contains(needle)
        )
        table = table[mask]
    if table.empty:
        st.warning("No event matches that filter.")
        return

    summary = pd.DataFrame({
        "Release date": pd.to_datetime(table["release_date"]).dt.date,
        "Session class": table["session_class"],
        "Release time": table["release_time"].fillna("unknown"),
        "Verified": table["verified"],
        "PAR": table["par"].map(pct),
        "Announcement": table["announcement_adj"].map(pct),
        "Subsequent": table["subsequent_adj"].map(pct),
        "Later": table["later_adj"].map(pct),
        "Flags": table["quality_flags"],
    })
    st.dataframe(summary, width="stretch", hide_index=True, height=300)

    options = list(table["release_date_str"])
    chosen = st.selectbox("Select an announcement to inspect", options, index=0,
                          key=f"inspector_choice_{result.config.ticker}")
    event = table[table["release_date_str"] == chosen].iloc[0]
    position = int(event["session_position"])
    release_date = pd.Timestamp(event["release_date"])

    st.divider()
    columns = st.columns(4)
    columns[0].metric("PAR [-4,-2]", pct(event["par"]))
    columns[1].metric("Announcement [-1,+1]", pct(event["announcement_adj"]))
    columns[2].metric("Subsequent [+2,+5]", pct(event["subsequent_adj"]))
    columns[3].metric("Later [+6,+8]", pct(event["later_adj"]))
    st.caption(ANNOUNCEMENT_WINDOW_CAVEAT)

    st.markdown("#### Window calculations")
    st.caption(
        "Each window's return is the compounded stock return minus the compounded "
        "benchmark return over the same sessions. The start price is the close at "
        "session a-1 and the end price is the close at session b, so the market-"
        "adjusted figure can be reproduced by hand from the two price columns."
    )
    st.dataframe(_calculation_block(event), width="stretch", hide_index=True)

    st.markdown("#### Daily adjusted closes and returns around the release")
    detail = daily_return_frame(
        result.panel.sessions, result.panel.stock_prices,
        result.panel.benchmark_prices, position, CHART_LO, CHART_HI,
    )
    display = detail.copy()
    display["session"] = display["session"].dt.date
    display["stock_adj_close"] = display["stock_adj_close"].map(
        lambda v: f"{v:,.4f}" if np.isfinite(v) else "missing")
    display["benchmark_adj_close"] = display["benchmark_adj_close"].map(
        lambda v: f"{v:,.4f}" if np.isfinite(v) else "missing")
    for column in ("stock_daily_return", "benchmark_daily_return", "daily_difference"):
        display[column] = display[column].map(pct)
    display = display.rename(columns={
        "offset": "Session offset", "session": "Session",
        "stock_adj_close": f"{result.panel.stock_ticker} adj close",
        "benchmark_adj_close": f"{result.panel.benchmark_ticker} adj close",
        "stock_daily_return": "Stock daily return",
        "benchmark_daily_return": "Benchmark daily return",
        "daily_difference": "Daily difference (not compounded)",
    })
    st.dataframe(display, width="stretch", hide_index=True, height=380)
    st.caption(
        "The daily difference column is shown for intuition only. The market "
        "adjustment used in the analysis is a difference between two compounded "
        "returns, never the compounding of these daily differences."
    )
    download(st, detail, "Download daily detail (CSV)",
             f"{result.config.ticker}_{chosen}_daily.csv", "dl_daily")

    st.markdown("#### Price chart around the release")
    st.plotly_chart(_price_chart(result.panel, position, release_date),
                    width="stretch")

    st.markdown("#### Source records for this release")
    reconciled = result.reconciled
    cluster = reconciled[
        (pd.to_datetime(reconciled["release_date"]).dt.normalize()
         >= release_date - pd.Timedelta(days=5))
        & (pd.to_datetime(reconciled["release_date"]).dt.normalize()
           <= release_date + pd.Timedelta(days=5))
    ]
    if len(cluster):
        st.dataframe(date_frame(cluster), width="stretch", hide_index=True)
    if str(event["quality_flags"]):
        st.warning(f"Data-quality flags: {event['quality_flags']}")
