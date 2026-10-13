"""Tab C: Actual vs Pseudo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from earnings_reversal.config import (
    ANNOUNCEMENT_WINDOW,
    OUTCOME_WINDOWS,
    PAR_WINDOW,
    PSEUDO_MAX_OFFSET,
    PSEUDO_MIN_OFFSET,
    PSEUDO_PERCENTILE_CAVEAT,
)
from earnings_reversal.pipeline import matched_sample_note
from earnings_reversal.pseudo import CONTAMINATION_RULE, comparison_table
from earnings_reversal.returns import evaluate_window
from .formatting import date_frame, download, num, pct


def _histogram(values: np.ndarray, actual: float | None, title: str,
               x_title: str, percent: bool) -> go.Figure:
    valid = values[np.isfinite(values)]
    figure = go.Figure()
    figure.add_trace(go.Histogram(
        x=valid, nbinsx=40, marker=dict(color="#7fa6cc", line=dict(width=1, color="white")),
        name="Pseudo repetitions",
    ))
    if actual is not None and np.isfinite(actual):
        figure.add_vline(
            x=actual, line_width=3, line_color="#c23b22",
            annotation_text=f"Actual: {pct(actual) if percent else num(actual)}",
            annotation_position="top",
        )
    if valid.size:
        for percentile, dash in ((5, "dot"), (95, "dot")):
            figure.add_vline(x=float(np.percentile(valid, percentile)),
                             line_width=1, line_dash=dash, line_color="#555")
    figure.update_layout(
        title=title, xaxis_title=x_title, yaxis_title="Repetitions",
        height=400, margin=dict(l=10, r=10, t=60, b=10), bargap=0.05,
        xaxis=dict(tickformat=".1%") if percent else dict(),
        showlegend=False,
    )
    return figure


def render(result) -> None:
    st.subheader("Actual vs Pseudo")
    simulation = result.simulation
    if simulation is None or not result.has_events:
        st.info(
            "No pseudo comparison is available. "
            + (result.blocking_error or "No eligible events are available.")
        )
        return
    if not simulation.matched_positions:
        st.warning(
            "No actual event has an eligible pseudo-date candidate, so there is "
            "nothing to compare."
        )
        if len(simulation.diagnostics):
            st.dataframe(date_frame(simulation.diagnostics), width="stretch",
                         hide_index=True)
        return

    columns = st.columns(4)
    columns[0].metric("Matched actual events", len(simulation.matched_positions))
    columns[1].metric("Repetitions", simulation.n_repetitions)
    columns[2].metric("Random seed", simulation.seed)
    columns[3].metric("Pseudo offsets (sessions before)",
                      f"{PSEUDO_MIN_OFFSET}\u2013{PSEUDO_MAX_OFFSET}")

    st.info(matched_sample_note(result))
    st.warning(
        "Repeated draws are a Monte Carlo sensitivity check on *which* pseudo dates "
        "were chosen. They are not additional independent earnings observations and "
        "do not increase the sample size. The percentile range below is "
        f"\u201c{PSEUDO_PERCENTILE_CAVEAT}\u201d The share of repetitions exceeding "
        "the actual statistic is deliberately not reported as a p-value, and the "
        "repetitions are never pooled into one large sample."
    )

    outcome_labels = {f"{w.label} [{w.start:+d},{w.end:+d}]": w.key for w in OUTCOME_WINDOWS}
    chosen_label = st.selectbox("Outcome window", list(outcome_labels.keys()), index=0)
    key = outcome_labels[chosen_label]

    actual = result.matched_actual_stats.get(key)
    rep_stats = simulation.per_rep_stats.get(key)
    if actual is None or rep_stats is None or rep_stats.empty:
        st.warning("No repetition statistics are available for this outcome window.")
        return

    table = comparison_table(actual, rep_stats)
    display = table.copy()

    def _format(row, column):
        value = row[column]
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "not available"
        if row["Is percentage"]:
            return pct(value)
        if "count" in row["Statistic"].lower():
            return f"{value:.1f}"
        return num(value)

    for column in ("Actual (matched sample)", "Pseudo median",
                   "Pseudo 5th percentile", "Pseudo 95th percentile"):
        display[column] = display.apply(lambda r, c=column: _format(r, c), axis=1)
    display = display.drop(columns=["Is percentage"])
    display = display.rename(columns={
        "Pseudo 5th percentile": "Pseudo 5th pct",
        "Pseudo 95th percentile": "Pseudo 95th pct",
    })
    st.markdown(f"#### Comparison table — {chosen_label}")
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(PSEUDO_PERCENTILE_CAVEAT)
    download(st, table, "Download comparison table (CSV)",
             f"actual_vs_pseudo_{key}.csv", f"dl_cmp_{key}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _histogram(
                rep_stats["negative_mean"].to_numpy(dtype=float),
                actual.negative_par.mean.value,
                "Pseudo mean outcome after negative PAR",
                "Mean outcome after negative PAR",
                percent=True,
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            _histogram(
                rep_stats["positive_mean"].to_numpy(dtype=float),
                actual.positive_par.mean.value,
                "Pseudo mean outcome after positive PAR",
                "Mean outcome after positive PAR",
                percent=True,
            ),
            width="stretch",
        )
    st.caption(
        "Each blue bar is one repetition: the same number of fake dates as actual "
        "earnings events, split into negative-PAR and positive-PAR groups, then "
        "averaged. Dotted lines mark the 5th and 95th percentiles across those "
        "repetitions. The solid red line is the actual matched-sample mean. "
        f"{PSEUDO_PERCENTILE_CAVEAT}"
    )

    _explain_pseudo(result, chosen_label)

    st.markdown("#### Candidate screening and exclusions")
    st.caption(CONTAMINATION_RULE)
    diagnostics = simulation.diagnostics
    if len(diagnostics):
        summary = pd.DataFrame([{
            "Candidates considered per event": int(diagnostics["candidates_considered"].iloc[0]),
            "Total rejected: out of range": int(diagnostics["rejected_out_of_range"].sum()),
            "Total rejected: missing price data": int(diagnostics["rejected_missing_data"].sum()),
            "Total rejected: earnings contamination":
                int(diagnostics["rejected_earnings_contamination"].sum()),
            "Events with zero eligible candidates": int(
                (diagnostics["eligible_candidates"] == 0).sum()),
            "Median eligible candidates per event": float(
                diagnostics["eligible_candidates"].median()),
        }])
        st.dataframe(summary.T.rename(columns={0: "Value"}), width="stretch")
        with st.expander("Per-event candidate diagnostics"):
            st.dataframe(date_frame(diagnostics), width="stretch", hide_index=True)
            download(st, diagnostics, "Download candidate diagnostics (CSV)",
                     "pseudo_candidates.csv", "dl_cand")
        overlapping = int(diagnostics["eligible_overlapping_actual_window"].sum())
        if overlapping:
            st.caption(
                f"{overlapping} eligible candidate(s) across all events sit 10\u201313 "
                f"sessions before their actual event, so their [-5,+8] interval "
                f"partly overlaps that event's own analysis window. This follows "
                f"directly from the fixed 10\u201340 session rule and is disclosed "
                f"rather than silently filtered."
            )

    st.markdown("#### Reuse and overlap diagnostics")
    reuse = simulation.reuse
    if reuse:
        st.dataframe(
            pd.DataFrame([reuse]).T.rename(columns={0: "Value"}),
            width="stretch",
        )
        st.caption(
            "Pseudo windows are drawn with replacement across repetitions and reused "
            "many times. They must not be described as independent observations."
        )

    if len(simulation.unmatched):
        st.markdown("#### Actual events dropped from the matched sample")
        st.dataframe(date_frame(simulation.unmatched), width="stretch",
                     hide_index=True)

    for note in simulation.notes:
        st.caption(note)

    st.caption(
        "This pilot does not target or claim replication of any particular effect "
        "size from the original study."
    )


def _dates_for_offsets(sessions, position: int, start: int, end: int) -> list[pd.Timestamp]:
    return [sessions[position + offset] for offset in range(start, end + 1)]


def _format_dates(dates) -> str:
    return ", ".join(pd.Timestamp(d).date().isoformat() for d in dates)


def _closed_days_between(sessions, first, last) -> list[pd.Timestamp]:
    """Calendar dates strictly between first and last that are not sessions."""
    days = pd.date_range(pd.Timestamp(first).normalize(),
                         pd.Timestamp(last).normalize(), freq="D")
    session_set = set(pd.DatetimeIndex(sessions).normalize())
    return [d for d in days if d not in session_set
            and d > pd.Timestamp(first).normalize()
            and d < pd.Timestamp(last).normalize()]


def _worked_example(result) -> dict | None:
    """One real fake date from the first repetition, with real window dates and returns."""
    simulation = result.simulation
    panel = result.panel
    if simulation is None or simulation.draws is None or panel is None:
        return None
    if not simulation.matched_positions:
        return None

    dates = list(result.matched_event_dates)
    positions = list(simulation.matched_positions)
    preferred = pd.Timestamp("2024-05-02")
    index = 0
    for i, date in enumerate(dates):
        if pd.Timestamp(date).normalize() == preferred:
            index = i
            break
    else:
        index = len(positions) - 1

    actual_position = positions[index]
    fake_position = int(simulation.draws[0, index])
    offset = actual_position - fake_position
    actual_date = pd.Timestamp(panel.sessions[actual_position])
    fake_date = pd.Timestamp(panel.sessions[fake_position])

    par_dates = _dates_for_offsets(panel.sessions, fake_position,
                                   PAR_WINDOW.start, PAR_WINDOW.end)
    ann_dates = _dates_for_offsets(panel.sessions, fake_position,
                                   ANNOUNCEMENT_WINDOW.start, ANNOUNCEMENT_WINDOW.end)
    par = evaluate_window(panel.sessions, panel.stock_prices, panel.benchmark_prices,
                          fake_position, PAR_WINDOW)
    announcement = evaluate_window(
        panel.sessions, panel.stock_prices, panel.benchmark_prices,
        fake_position, ANNOUNCEMENT_WINDOW,
    )
    closed = _closed_days_between(panel.sessions, par_dates[0], ann_dates[-1])
    return {
        "actual_date": actual_date,
        "fake_date": fake_date,
        "offset": offset,
        "par_dates": par_dates,
        "ann_dates": ann_dates,
        "par": par.market_adjusted_return if par.valid else None,
        "announcement": announcement.market_adjusted_return if announcement.valid else None,
        "closed_days": closed,
        "ticker": panel.stock_ticker,
        "benchmark": panel.benchmark_ticker,
        "n_events": len(positions),
        "n_reps": simulation.n_repetitions,
    }


def _explain_pseudo(result, chosen_label: str) -> None:
    st.markdown("#### What a pseudo date is")
    n_events = len(result.simulation.matched_positions) if result.simulation else 0
    n_reps = result.simulation.n_repetitions if result.simulation else 0
    st.markdown(
        "A **pseudo date** is an ordinary trading day that we *pretend* is an "
        "earnings announcement. The program does not invent prices. It only moves "
        "the same return windows to a nearby non-earnings day and repeats the "
        "identical calculations."
    )
    st.markdown(
        f"For each of the **{n_events}** matched actual earnings events, the program "
        f"looks 10 to 40 trading sessions earlier and keeps only dates whose "
        f"full analysis interval has complete prices and no known earnings release. "
        f"It then draws one of those dates at random. Doing that for every actual "
        f"event produces one fake sample of the same size. That whole set is "
        f"**one repetition**. The sidebar currently asks for **{n_reps}** repetitions, "
        f"each with a new random draw of fake dates."
    )
    st.markdown(
        "We then compare the actual earnings statistics with the cloud of statistics "
        f"from those {n_reps} fake samples. The histograms above do that for the "
        f"**{chosen_label}** means after negative PAR and after positive PAR."
    )
    st.markdown(
        f"The {n_reps} repetitions do **not** create {n_reps * max(n_events, 1):,} "
        f"independent observations. We still have only {n_events} actual earnings "
        f"events. The repetitions only show how much the comparison moves when a "
        f"different set of ordinary dates is chosen. The dotted percentile range is "
        f"variation across those draws, not a confidence interval, and it is not a "
        f"p-value."
    )

    example = _worked_example(result)
    with st.expander("One example of a repetition", expanded=True):
        if example is None:
            st.info("A worked example needs a completed simulation with at least one draw.")
            return
        weekday = example["actual_date"].strftime("%A")
        st.markdown(
            f"Take {example['ticker']}'s actual earnings date: "
            f"**{weekday}, {example['actual_date'].date()}**."
        )
        st.markdown(
            f"In the first repetition the program randomly chose an eligible fake "
            f"date **{example['offset']} trading sessions earlier**: "
            f"**{example['fake_date'].date()}**."
        )
        st.markdown(
            f"It treats {example['fake_date'].date()} as if it were an earnings date:"
        )
        rows = [
            {
                "Window": f"Before fake event (PAR) [{PAR_WINDOW.start:+d} to {PAR_WINDOW.end:+d}]",
                "Trading sessions": _format_dates(example["par_dates"]),
                "What we calculate": (
                    f"{example['ticker']} compounded return minus "
                    f"{example['benchmark']} compounded return"
                ),
                "Result in this draw": pct(example["par"]),
            },
            {
                "Window": (
                    f"Around fake event (announcement) "
                    f"[{ANNOUNCEMENT_WINDOW.start:+d} to {ANNOUNCEMENT_WINDOW.end:+d}]"
                ),
                "Trading sessions": _format_dates(example["ann_dates"]),
                "What we calculate": (
                    f"{example['ticker']} compounded return minus "
                    f"{example['benchmark']} compounded return"
                ),
                "Result in this draw": pct(example["announcement"]),
            },
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if example["closed_days"]:
            shown = ", ".join(d.date().isoformat() for d in example["closed_days"][:6])
            extra = "" if len(example["closed_days"]) <= 6 else (
                f" and {len(example['closed_days']) - 6} more"
            )
            st.caption(
                f"The market was closed on {shown}{extra}. Offsets skip those days "
                f"because they count trading sessions, not calendar days."
            )
        if example["par"] is not None and example["announcement"] is not None:
            if example["par"] * example["announcement"] < 0:
                pattern = "a reversal (the two returns have opposite signs)"
            elif example["par"] * example["announcement"] > 0:
                pattern = "a continuation (the two returns have the same sign)"
            else:
                pattern = "neither a reversal nor a continuation, because one return is zero"
            st.markdown(
                f"In this one fake window the pre-event return was "
                f"**{pct(example['par'])}** and the around-event return was "
                f"**{pct(example['announcement'])}**. That is {pattern} on an "
                f"ordinary, non-earnings date."
            )
        st.markdown(
            f"The program makes one such random selection for each of the "
            f"**{example['n_events']}** actual earnings announcements, then "
            f"calculates the same statistics across those {example['n_events']} "
            f"fake events. That entire set is one repetition. The next repetition "
            f"selects a new set of fake dates."
        )
