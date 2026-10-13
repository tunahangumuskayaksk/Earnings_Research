"""Tab B: Earnings Event Analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from earnings_reversal.config import (
    ANNOUNCEMENT_WINDOW,
    ANNOUNCEMENT_WINDOW_CAVEAT,
    OUTCOME_WINDOWS,
    PAR_WINDOW,
)
from earnings_reversal.stats import SMALL_SAMPLE_THRESHOLD, group_table, stats_to_rows
from .formatting import download, pct


def _scatter(events: pd.DataFrame, stats) -> go.Figure:
    hover = [
        f"Release {pd.Timestamp(row.release_date).date()}<br>"
        f"PAR {pct(row.par)}<br>"
        f"Announcement {pct(row.announcement_adj)}<br>"
        f"Stock {pct(row.announcement_stock)} / Benchmark {pct(row.announcement_benchmark)}"
        for row in events.itertuples()
    ]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=events["par"], y=events["announcement_adj"], mode="markers",
        marker=dict(size=10, color="#2a6fb5", line=dict(width=1, color="white")),
        text=hover, hoverinfo="text", name="Earnings events",
    ))
    figure.add_hline(y=0, line_width=1, line_color="#888")
    figure.add_vline(x=0, line_width=1, line_color="#888")

    if stats.slope.available and stats.intercept.available and len(events) >= 2:
        x_min, x_max = float(events["par"].min()), float(events["par"].max())
        pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.01
        line_x = np.array([x_min - pad, x_max + pad])
        line_y = stats.intercept.value + stats.slope.value * line_x
        figure.add_trace(go.Scatter(
            x=line_x, y=line_y, mode="lines", name="In-sample OLS fit",
            line=dict(color="#c23b22", width=2, dash="dash"),
        ))
    else:
        figure.add_annotation(
            text=f"No fitted line: {stats.slope.reason}", showarrow=False,
            xref="paper", yref="paper", x=0.5, y=1.05,
        )

    figure.update_layout(
        xaxis_title="PAR: market-adjusted return, sessions [-4,-2]",
        yaxis_title="Announcement market-adjusted return, sessions [-1,+1]",
        xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"),
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="closest",
    )
    return figure


def render(result) -> None:
    st.subheader("Earnings Event Analysis")
    if not result.has_events:
        st.info(
            "No event statistics can be shown yet. "
            + (result.blocking_error or "No eligible events are available.")
        )
        st.markdown(
            "**What to do next:** check the *Data & Coverage* tab for the download "
            "errors or exclusion reasons, widen the event date range, or upload a "
            "verified earnings-date CSV."
        )
        return

    events = result.included_events
    stats = result.stats_by_outcome[ANNOUNCEMENT_WINDOW.key]

    st.info(ANNOUNCEMENT_WINDOW_CAVEAT)

    columns = st.columns(4)
    columns[0].metric("Events included", stats.n)
    columns[1].metric("Correlation (PAR vs announcement)", stats.correlation.format())
    columns[2].metric("OLS slope", stats.slope.format())
    columns[3].metric("R-squared", stats.r_squared.format())

    if stats.small_sample:
        st.warning(
            f"Small sample: {stats.n} events, below the {SMALL_SAMPLE_THRESHOLD}-event "
            f"threshold at which these statistics start to be even loosely informative. "
            f"Roughly four announcements a year is the hard limit for one company."
        )
    st.caption(
        "All statistics are exploratory and estimated in-sample on these events. The "
        "fitted line summarises the points it was estimated from; it is not evidence "
        "of out-of-sample predictive performance."
    )

    st.plotly_chart(_scatter(events, stats), width="stretch")

    st.markdown("#### Announcement-window statistics")
    st.dataframe(pd.DataFrame(stats_to_rows(stats)), width="stretch",
                 hide_index=True)
    for note in stats.notes:
        st.caption(note)

    st.markdown("#### PAR-sign groups across all three outcome windows")
    st.caption(
        "A reversal is defined as PAR \u00d7 outcome return < 0. Events where either "
        "value is exactly zero are reported separately and never counted as a "
        "reversal or a continuation."
    )
    labels = {w.key: f"{w.label} [{w.start:+d},{w.end:+d}]" for w in OUTCOME_WINDOWS}
    table = group_table({labels[k]: s for k, s in result.stats_by_outcome.items()})
    display = table.copy()
    for column in ("Mean outcome", "Median outcome"):
        display[column] = display[column].map(lambda v: pct(v) if v is not None else "n/a")
    display["Opposite-sign frequency"] = display["Opposite-sign frequency"].map(
        lambda v: pct(v) if v is not None and np.isfinite(v) else "n/a"
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "The negative-PAR minus positive-PAR mean difference compares events for one "
        "company occurring at different times. It is **not** a simultaneously tradable "
        "long-short portfolio return."
    )
    download(st, table, "Download group summary (CSV)", "par_group_summary.csv",
             "dl_groups")

    with st.expander("Per-outcome statistics for the subsequent and later windows"):
        for window in OUTCOME_WINDOWS[1:]:
            stat = result.stats_by_outcome[window.key]
            st.markdown(f"**{window.label} — sessions [{window.start:+d},{window.end:+d}]**")
            st.dataframe(pd.DataFrame(stats_to_rows(stat)), width="stretch",
                         hide_index=True)

    st.markdown("#### Event-level audit table")
    st.caption(
        "One row per included event with every window's first return date, starting "
        "and ending price dates and the adjusted prices used."
    )
    st.dataframe(_audit_view(events), width="stretch", hide_index=True,
                 height=360)
    download(st, events, "Download full event audit table (CSV)",
             f"{result.config.ticker}_event_audit.csv", "dl_audit")


def _audit_view(events: pd.DataFrame) -> pd.DataFrame:
    view = events.copy()
    view["release_date"] = pd.to_datetime(view["release_date"]).dt.date
    for column in view.columns:
        if pd.api.types.is_datetime64_any_dtype(events[column]):
            view[column] = pd.to_datetime(events[column]).dt.date
    percent_columns = [c for c in view.columns
                       if c == "par" or c.endswith(("_adj", "_stock", "_benchmark"))]
    for column in percent_columns:
        view[column] = view[column].map(lambda v: pct(v))
    price_columns = [c for c in view.columns if c.endswith("_price")]
    for column in price_columns:
        view[column] = view[column].map(lambda v: f"{v:,.4f}" if pd.notna(v) else "n/a")
    return view
