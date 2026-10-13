"""Tab F: PAR Quintile Strategy (per-stock historical adaptation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from earnings_reversal.config import ANNOUNCEMENT_WINDOW_CAVEAT, PSEUDO_PERCENTILE_CAVEAT
from earnings_reversal.strategy import (
    ADAPTATION_NOTE,
    COMPARISON_STATS,
    COSTS_NOTE,
    DEFAULT_NOTIONAL,
    HIGH_PERCENTILE,
    INSUFFICIENT,
    LOW_PERCENTILE,
    MIN_PRIOR_EVENTS,
    QUINTILE,
    RETURN_BASIS_NOTE,
    SAME_CLOSE_NOTE,
    SIGN_ALL,
    SIGN_SAME,
    SPREAD_NOTE,
    STRATEGY_LABELS,
    pseudo_comparison,
)
from .formatting import date_frame, download, pct

POSITION_COLOURS = {
    "Long": "#1f77b4",
    "Short": "#ff7f0e",
    "No trade": "#9aa5b1",
    INSUFFICIENT: "#d0d4d9",
}


def _money(value, notional: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"${value * notional:,.0f}"


def _format(value, kind: str, notional: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if kind == "count":
        return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
    if kind == "money":
        return _money(value, notional)
    return pct(value)


PERFORMANCE_ROWS = [
    ("trades", "Executed trades", "count"),
    ("longs", "Long trades (Q1 or negative PAR)", "count"),
    ("shorts", "Short trades (Q5 or positive PAR)", "count"),
    ("mean_adj", "Average signed market-adjusted return per trade", "pct"),
    ("win_adj", "Win rate, market-adjusted", "pct"),
    ("pnl_adj", "Cumulative fixed-notional P&L, market-adjusted", "money"),
    ("mean_raw", "Average raw stock strategy return per trade", "pct"),
    ("win_raw", "Win rate, raw stock", "pct"),
    ("pnl_raw", "Cumulative fixed-notional P&L, raw stock", "money"),
    ("spread_adj", "Group spread (not a strategy return): long-group mean minus "
                   "short-group mean, market-adjusted", "pct"),
    ("spread_raw", "Group spread (not a strategy return): long-group mean minus "
                   "short-group mean, raw stock", "pct"),
]


def _performance_table(performance: dict, notional: float) -> pd.DataFrame:
    rows = []
    for key, label, kind in PERFORMANCE_ROWS:
        row = {"Statistic": label}
        for strategy in (QUINTILE, SIGN_SAME, SIGN_ALL):
            row[STRATEGY_LABELS[strategy]] = _format(
                performance[strategy].get(key), kind, notional
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cumulative_chart(trades: pd.DataFrame, basis: str, notional: float) -> go.Figure:
    returns = trades["stock_return"] if basis == "raw" else trades["adjusted_return"]
    dates = trades["release_date"]
    eligible = trades["eligible"].to_numpy()
    lines = {
        STRATEGY_LABELS[QUINTILE]: trades["direction"].to_numpy(),
        STRATEGY_LABELS[SIGN_SAME]: np.where(eligible, trades["sign_direction"], 0),
        STRATEGY_LABELS[SIGN_ALL]: trades["sign_direction"].to_numpy(),
    }
    styles = {
        STRATEGY_LABELS[QUINTILE]: dict(color="#c23b22", width=3),
        STRATEGY_LABELS[SIGN_SAME]: dict(color="#4c78a8", width=2),
        STRATEGY_LABELS[SIGN_ALL]: dict(color="#4c78a8", width=1.5, dash="dot"),
    }
    figure = go.Figure()
    for label, direction in lines.items():
        pnl = np.cumsum(direction * returns.to_numpy() * notional)
        figure.add_trace(go.Scatter(
            x=dates, y=pnl, mode="lines", name=label, line=styles[label],
            line_shape="hv",
        ))
    figure.add_hline(y=0, line_color="#999", line_width=1)
    first_eligible = trades.loc[trades["eligible"], "release_date"]
    if len(first_eligible):
        figure.add_vline(x=first_eligible.iloc[0], line_dash="dash", line_color="#777")
        figure.add_annotation(
            x=first_eligible.iloc[0], y=1, yref="paper", showarrow=False,
            text=f"first event with {MIN_PRIOR_EVENTS} prior events", xanchor="left",
        )
    figure.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title=f"Cumulative P&L, ${notional:,.0f} per trade",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        legend=dict(orientation="h", y=-0.15),
    )
    return figure


def _threshold_chart(trades: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=trades["release_date"], y=trades["threshold_p20"], mode="lines",
        line=dict(color="#1f77b4", dash="dash", shape="hv"),
        name=f"{LOW_PERCENTILE}th percentile of earlier PARs",
    ))
    figure.add_trace(go.Scatter(
        x=trades["release_date"], y=trades["threshold_p80"], mode="lines",
        line=dict(color="#ff7f0e", dash="dash", shape="hv"),
        name=f"{HIGH_PERCENTILE}th percentile of earlier PARs",
    ))
    for position, colour in POSITION_COLOURS.items():
        subset = trades[trades["position"] == position]
        if subset.empty:
            continue
        figure.add_trace(go.Scatter(
            x=subset["release_date"], y=subset["par"], mode="markers", name=position,
            marker=dict(color=colour, size=9, line=dict(width=1, color="white")),
            customdata=np.stack([subset["prior_events"]], axis=-1),
            hovertemplate="%{x|%Y-%m-%d}<br>PAR %{y:.2%}<br>prior events %{customdata[0]}"
                          "<extra>" + position + "</extra>",
        ))
    figure.add_hline(y=0, line_color="#999", line_width=1)
    figure.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="PAR [-4,-2], market-adjusted", tickformat=".1%"),
        legend=dict(orientation="h", y=-0.15),
    )
    return figure


def _pseudo_histogram(values: np.ndarray, actual, kind: str, notional: float,
                      title: str) -> go.Figure:
    scale = notional if kind == "money" else 1.0
    valid = values[np.isfinite(values)] * scale
    figure = go.Figure(go.Histogram(
        x=valid, nbinsx=40, marker=dict(color="#7fa6cc", line=dict(width=1, color="white")),
    ))
    if actual is not None and np.isfinite(actual):
        figure.add_vline(x=actual * scale, line_width=3, line_color="#c23b22",
                         annotation_text=f"Actual: {_format(actual, kind, notional)}",
                         annotation_position="top")
    if valid.size:
        for q in (5, 95):
            figure.add_vline(x=float(np.percentile(valid, q)), line_width=1,
                             line_dash="dot", line_color="#555")
    tick = dict(tickformat=".1%") if kind == "pct" else (
        dict(tickprefix="$", tickformat=",.0f") if kind == "money" else {})
    figure.update_layout(title=title, height=380, showlegend=False, bargap=0.05,
                         margin=dict(l=10, r=10, t=60, b=10), xaxis=tick,
                         yaxis_title="Repetitions")
    return figure


def _trade_table(trades: pd.DataFrame) -> pd.DataFrame:
    view = trades[[
        "release_date", "signal_and_entry_close", "exit_close", "par", "prior_events",
        "threshold_p20", "threshold_p80", "latest_threshold_input", "quintile",
        "position", "stock_return", "benchmark_return", "adjusted_return",
        "signed_raw", "signed_adj",
    ]].copy()
    view = date_frame(view)
    for column in ("par", "threshold_p20", "threshold_p80", "stock_return",
                   "benchmark_return", "adjusted_return", "signed_raw", "signed_adj"):
        view[column] = view[column].map(pct)
    return view.rename(columns={
        "release_date": "Release date",
        "signal_and_entry_close": "Signal complete & entry close (t=\u22122)",
        "exit_close": "Exit close (t=+1)",
        "par": "PAR",
        "prior_events": "Earlier events used",
        "threshold_p20": f"{LOW_PERCENTILE}th pct threshold",
        "threshold_p80": f"{HIGH_PERCENTILE}th pct threshold",
        "latest_threshold_input": "Latest threshold input (its t=\u22122)",
        "quintile": "Quintile",
        "position": "Position",
        "stock_return": "Stock return [-1,+1]",
        "benchmark_return": "Benchmark return [-1,+1]",
        "adjusted_return": "Market-adjusted return [-1,+1]",
        "signed_raw": "Raw stock strategy return",
        "signed_adj": "Signed market-adjusted return",
    })


def render(result) -> None:
    st.subheader("PAR Quintile Strategy")
    strategy = result.quintile
    if strategy is None or strategy.trades.empty:
        st.info("No strategy results are available. "
                + (result.blocking_error or "No eligible events are available."))
        return
    ticker = result.config.ticker

    st.info(ADAPTATION_NOTE)
    st.markdown(
        f"**Rule.** For each earnings event, take the stock's own **earlier** included "
        f"events (at least {MIN_PRIOR_EVENTS}; the current event is never included) and "
        f"compute the {LOW_PERCENTILE}th and {HIGH_PERCENTILE}th percentiles of their PAR. "
        f"If the current PAR is below the {LOW_PERCENTILE}th percentile (Q1), go **long**; "
        f"above the {HIGH_PERCENTILE}th (Q5), go **short**; otherwise do not trade. "
        f"Quintile membership alone sets the direction, even when a threshold is "
        f"positive or negative. Events with fewer than {MIN_PRIOR_EVENTS} earlier events "
        f"are skipped. The position is held over the announcement window [-1,+1]."
    )
    st.warning(SAME_CLOSE_NOTE + " " + COSTS_NOTE)
    st.caption(ANNOUNCEMENT_WINDOW_CAVEAT)

    if strategy.availability_violations:
        st.error(
            f"{strategy.availability_violations} trade(s) use a threshold input dated on "
            f"or after the entry close. Signal availability is not guaranteed for them."
        )
    else:
        st.caption(
            "Timing check passed: for every executed trade, the latest PAR used in its "
            "thresholds was complete at a strictly earlier close than the entry close."
        )
    for note in strategy.notes:
        st.warning(note)

    notional = float(st.number_input(
        "Fixed notional per trade ($)", min_value=100.0, value=DEFAULT_NOTIONAL,
        step=1000.0, key="strategy_notional",
        help="Every trade uses this notional. P&L is summed, never compounded.",
    ))

    perf = strategy.performance[QUINTILE]
    total = len(strategy.trades)
    columns = st.columns(5)
    columns[0].metric("Events with enough history", f"{strategy.eligible_events} of {total}")
    columns[1].metric("Executed trades", perf["trades"],
                      help=f"{perf['longs']} long, {perf['shorts']} short")
    columns[2].metric("Avg signed market-adj. return / trade", pct(perf["mean_adj"]))
    columns[3].metric("Win rate (market-adj.)", pct(perf["win_adj"], 1))
    columns[4].metric("Cumulative P&L (market-adj.)", _money(perf["pnl_adj"], notional))

    st.markdown("#### Results, before costs")
    st.dataframe(_performance_table(strategy.performance, notional), width="stretch",
                 hide_index=True)
    st.caption(RETURN_BASIS_NOTE)
    st.caption(
        SPREAD_NOTE + " The average return per trade weights every trade equally, so it "
        "equals the group spread divided by two only when long and short counts match."
    )
    st.caption(
        "The PAR sign strategy is the existing negative/positive PAR split traded "
        "directly. \u201cSame eligible events\u201d restricts it to events that have "
        f"{MIN_PRIOR_EVENTS} earlier events, so it can be compared with the quintile "
        "strategy on the same dates."
    )

    st.markdown("#### Cumulative fixed-notional P&L")
    basis_label = st.radio(
        "Return basis", ["Signed market-adjusted", "Raw stock"], horizontal=True,
        key=f"strategy_basis_{ticker}",
    )
    basis = "raw" if basis_label == "Raw stock" else "adj"
    st.plotly_chart(_cumulative_chart(strategy.trades, basis, notional),
                    width="stretch", key=f"strategy_cum_{ticker}_{basis}")
    st.caption(
        "Each step is one event. Flat stretches are events with no trade. Before costs."
    )

    st.markdown("#### Thresholds and positions")
    st.plotly_chart(_threshold_chart(strategy.trades), width="stretch",
                    key=f"strategy_thresholds_{ticker}")
    st.caption(
        "Dashed lines are the thresholds that applied to each event, estimated only from "
        "earlier events. Points are each event's PAR, coloured by the position taken."
    )

    _pseudo_section(strategy, notional, ticker)

    st.markdown("#### Event-level trades")
    st.dataframe(_trade_table(strategy.trades), width="stretch", hide_index=True)
    download(st, strategy.trades, "Download trades (CSV)",
             f"{ticker}_par_quintile_trades.csv", f"dl_strategy_{ticker}")


def _pseudo_section(strategy, notional: float, ticker: str) -> None:
    st.markdown("#### Actual vs pseudo-event repetitions")
    if not strategy.pseudo_reps:
        st.info("No pseudo-event repetitions are available for this stock.")
        return
    st.markdown(
        "The same rule is applied to every pseudo-event repetition from the *Actual vs "
        "Pseudo* tab. For each fake date, the thresholds come only from **actual** "
        "earnings events before that fake date, again requiring "
        f"{MIN_PRIOR_EVENTS} of them. The actual column uses the matched sample, the "
        "events that have pseudo-date candidates."
    )
    st.warning(
        "Each repetition is one alternative history. Repetitions are not pooled into one "
        "sample and are not independent observations; the percentile range is "
        f"\u201c{PSEUDO_PERCENTILE_CAVEAT}\u201d Earnings-window returns are usually more "
        "volatile than ordinary-day returns, so actual results can land outside this "
        "range on either side without implying a reversal effect."
    )

    eligible = strategy.pseudo_eligible
    quint_trades = strategy.pseudo_reps[QUINTILE]["trades"]
    columns = st.columns(4)
    columns[0].metric("Matched actual events", strategy.matched_events)
    columns[1].metric("Matched events with enough history",
                      strategy.matched_eligible_events)
    columns[2].metric(
        "Eligible pseudo events per repetition",
        f"{int(eligible.min())}" if eligible.min() == eligible.max()
        else f"{int(eligible.min())}\u2013{int(eligible.max())}",
    )
    columns[3].metric(
        "Pseudo quintile trades per repetition",
        f"{int(np.median(quint_trades))} (5th\u201395th: "
        f"{int(np.percentile(quint_trades, 5))}\u2013{int(np.percentile(quint_trades, 95))})",
    )

    choice = st.selectbox(
        "Strategy", [QUINTILE, SIGN_SAME], format_func=STRATEGY_LABELS.get,
        key=f"strategy_pseudo_choice_{ticker}",
    )
    table = pseudo_comparison(strategy, choice)
    display = table.copy()
    for column in ("Actual (matched sample)", "Pseudo median", "Pseudo 5th percentile",
                   "Pseudo 95th percentile"):
        display[column] = [_format(v, k, notional) for v, k in zip(table[column], table["kind"])]
    st.dataframe(display.drop(columns=["stat", "kind"]), width="stretch", hide_index=True)
    st.caption(
        "Valid repetitions can fall below the total when a repetition has no trades "
        "(no average or win rate) or no trade on one side (no group spread). "
        + SPREAD_NOTE + " " + COSTS_NOTE
    )

    stat = st.selectbox(
        "Histogram statistic", [s for s in COMPARISON_STATS if s != "trades"],
        format_func=lambda s: COMPARISON_STATS[s][0],
        key=f"strategy_hist_stat_{ticker}",
    )
    label, kind = COMPARISON_STATS[stat]
    values = strategy.pseudo_reps[choice][stat].to_numpy(dtype=float)
    actual = strategy.matched_performance[choice].get(stat)
    st.plotly_chart(_pseudo_histogram(values, actual, kind, notional, label),
                    width="stretch", key=f"strategy_hist_{ticker}_{choice}_{stat}")
    st.caption("Red line: actual matched-sample value. Dotted lines: pseudo 5th and "
               "95th percentiles.")
