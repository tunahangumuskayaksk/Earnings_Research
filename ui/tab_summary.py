"""Multi-stock summary page."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from earnings_reversal.config import (
    ANNOUNCEMENT_WINDOW_CAVEAT,
    OUTCOME_WINDOWS,
    PSEUDO_PERCENTILE_CAVEAT,
)
from earnings_reversal.summary import (
    PERCENT_SUMMARY_STATS,
    REVERSAL_SIDE,
    STAT_TITLES,
    SUMMARY_STATS,
    cross_stock_overview,
    pooled_events,
    pooled_group_table,
    pseudo_median_gap,
    strategy_universe_table,
    universe_table,
    volatility_ratios,
)
from earnings_reversal.strategy import (
    ADAPTATION_NOTE,
    COMPARISON_STATS,
    COSTS_NOTE,
    DEFAULT_NOTIONAL,
    MIN_PRIOR_EVENTS,
)
from earnings_reversal.universe import StockGroup, display_name
from .formatting import download, num, pct


def _fmt(value, percent: bool) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return pct(value) if percent else num(value)


def _volatility_note(results: dict, window_key: str) -> str:
    ratios = volatility_ratios(results, window_key)
    if ratios.empty:
        return "Volatility comparison unavailable: no stock has pseudo draws."
    noisier = int((ratios > 1).sum())
    return (
        f"Measured here: the outcome return's standard deviation on actual event "
        f"dates divided by its standard deviation on the fake dates has a median of "
        f"{ratios.median():.2f} across {len(ratios)} stocks (range "
        f"{ratios.min():.2f}\u2013{ratios.max():.2f}); {noisier} of {len(ratios)} "
        f"stocks have a ratio above 1."
    )


def _band_chart(table: pd.DataFrame, stat: str) -> go.Figure:
    """Actual statistic per stock against its own pseudo 5th-95th range."""
    percent = stat in PERCENT_SUMMARY_STATS
    data = table[table["status"] == "ok"].copy()
    data = data[pd.to_numeric(data[f"{stat}_matched"], errors="coerce").notna()]
    data = data.sort_values(f"{stat}_matched")
    figure = go.Figure()
    if data.empty:
        figure.add_annotation(text="No stock has this statistic available.",
                              showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return figure

    labels = data["ticker"].tolist()
    for _, row in data.iterrows():
        figure.add_trace(go.Scatter(
            x=[row[f"{stat}_pseudo_p5"], row[f"{stat}_pseudo_p95"]],
            y=[row["ticker"], row["ticker"]],
            mode="lines", line=dict(color="#b8c7d9", width=6),
            hoverinfo="skip", showlegend=False,
        ))
    figure.add_trace(go.Scatter(
        x=data[f"{stat}_pseudo_median"], y=labels, mode="markers",
        marker=dict(color="#5b7fa6", size=7, symbol="line-ns-open", line=dict(width=2)),
        name="Pseudo median",
        hovertemplate="%{y}: pseudo median %{x}<extra></extra>",
    ))
    side = REVERSAL_SIDE[stat]
    colors = []
    for _, row in data.iterrows():
        position = row[f"{stat}_position"]
        if (side == "below" and position.startswith("below")) or \
                (side == "above" and position.startswith("above")):
            colors.append("#c23b22")
        elif position.startswith(("below", "above")):
            colors.append("#7a4fb3")
        else:
            colors.append("#333333")
    hover = [
        f"{row.ticker} ({row.sector})<br>Actual: {_fmt(getattr(row, stat + '_matched'), percent)}"
        f"<br>Pseudo 5th\u201395th: {_fmt(getattr(row, stat + '_pseudo_p5'), percent)} to "
        f"{_fmt(getattr(row, stat + '_pseudo_p95'), percent)}"
        f"<br>{getattr(row, stat + '_position')}"
        for row in data.itertuples()
    ]
    figure.add_trace(go.Scatter(
        x=data[f"{stat}_matched"], y=labels, mode="markers",
        marker=dict(color=colors, size=11, line=dict(width=1, color="white")),
        name="Actual (matched sample)", text=hover, hoverinfo="text",
    ))
    figure.add_vline(x=0, line_width=1, line_color="#888")
    figure.update_layout(
        height=max(420, 22 * len(labels) + 120),
        xaxis_title=STAT_TITLES[stat],
        xaxis=dict(tickformat=".1%") if percent else dict(),
        yaxis=dict(categoryorder="array", categoryarray=labels),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return figure


def _dumbbell_chart(table: pd.DataFrame) -> go.Figure:
    """Mean outcome after negative PAR vs after positive PAR, per stock."""
    data = table[table["status"] == "ok"].copy()
    data = data[pd.to_numeric(data["negative_minus_positive"], errors="coerce").notna()]
    data = data.sort_values("negative_minus_positive")
    figure = go.Figure()
    if data.empty:
        figure.add_annotation(text="No stock has both PAR groups available.",
                              showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return figure
    labels = data["ticker"].tolist()
    for _, row in data.iterrows():
        figure.add_trace(go.Scatter(
            x=[row["negative_mean"], row["positive_mean"]],
            y=[row["ticker"], row["ticker"]],
            mode="lines", line=dict(color="#cccccc", width=2),
            hoverinfo="skip", showlegend=False,
        ))
    figure.add_trace(go.Scatter(
        x=data["negative_mean"], y=labels, mode="markers",
        marker=dict(color="#2a6fb5", size=11),
        name="After negative PAR",
        text=[f"{t}: after negative PAR {pct(v)} (n={n})"
              for t, v, n in zip(labels, data["negative_mean"], data["negative_count"])],
        hoverinfo="text",
    ))
    figure.add_trace(go.Scatter(
        x=data["positive_mean"], y=labels, mode="markers",
        marker=dict(color="#e08a2b", size=11),
        name="After positive PAR",
        text=[f"{t}: after positive PAR {pct(v)} (n={n})"
              for t, v, n in zip(labels, data["positive_mean"], data["positive_count"])],
        hoverinfo="text",
    ))
    figure.add_vline(x=0, line_width=1, line_color="#888")
    figure.update_layout(
        height=max(420, 22 * len(labels) + 120),
        xaxis_title="Mean market-adjusted outcome return",
        xaxis=dict(tickformat=".1%"),
        yaxis=dict(categoryorder="array", categoryarray=labels),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return figure


def _per_stock_view(table: pd.DataFrame) -> pd.DataFrame:
    view = pd.DataFrame({
        "Ticker": table["ticker"],
        "Company": table["name"],
        "Sector": table["sector"],
        "Events": table["events_included"],
        "Correlation": table["correlation"].map(lambda v: _fmt(v, False)),
        "Slope": table["slope"].map(lambda v: _fmt(v, False)),
        "R-squared": table["r_squared"].map(lambda v: _fmt(v, False)),
        "Neg-PAR n": table["negative_count"],
        "Mean after neg PAR": table["negative_mean"].map(lambda v: _fmt(v, True)),
        "Pos-PAR n": table["positive_count"],
        "Mean after pos PAR": table["positive_mean"].map(lambda v: _fmt(v, True)),
        "Neg minus pos": table["negative_minus_positive"].map(lambda v: _fmt(v, True)),
        "Reversal freq": table["reversal_frequency"].map(lambda v: _fmt(v, True)),
        "Slope vs pseudo": table["slope_position"],
        "Neg minus pos vs pseudo": table["negative_minus_positive_position"],
        "Status": table["status"],
    })
    return view


STRATEGY_CHART_STATS = [s for s in COMPARISON_STATS if s != "trades"] + ["trades"]
FAVOURABLE = "#c23b22"
UNFAVOURABLE = "#7a4fb3"


def _axis(kind: str) -> dict:
    if kind == "pct":
        return dict(tickformat=".1%")
    if kind == "money":
        return dict(tickprefix="$", tickformat=",.0f")
    return {}


def _scaled(values, kind: str):
    return values * DEFAULT_NOTIONAL if kind == "money" else values


def _format_kind(value, kind: str) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if kind == "pct":
        return pct(value)
    if kind == "money":
        return f"${value * DEFAULT_NOTIONAL:,.0f}"
    if kind == "count":
        return f"{value:.1f}"
    return num(value)


def _strategy_band_chart(table: pd.DataFrame, stat: str) -> go.Figure:
    """Actual strategy statistic per stock against its own pseudo 5th-95th range."""
    label, kind = COMPARISON_STATS[stat]
    data = table.copy()
    data[f"{stat}_matched"] = pd.to_numeric(data[f"{stat}_matched"], errors="coerce")
    data = data.dropna(subset=[f"{stat}_matched"]).sort_values(f"{stat}_matched")
    figure = go.Figure()
    if data.empty:
        figure.add_annotation(text="No stock has this statistic available.",
                              showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return figure
    neutral = stat == "trades"
    colours = [
        "#333" if neutral or pos.startswith("within") or pos == "not available"
        else FAVOURABLE if pos.startswith("above") else UNFAVOURABLE
        for pos in data[f"{stat}_position"].fillna("not available")
    ]
    p5 = _scaled(pd.to_numeric(data[f"{stat}_pseudo_p5"], errors="coerce"), kind)
    p95 = _scaled(pd.to_numeric(data[f"{stat}_pseudo_p95"], errors="coerce"), kind)
    figure.add_trace(go.Bar(
        y=data["ticker"], x=p95 - p5, base=p5, orientation="h",
        marker=dict(color="rgba(127,166,204,0.45)"), name="Pseudo 5th\u201395th pct",
        hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        y=data["ticker"], x=_scaled(data[f"{stat}_pseudo_median"], kind), mode="markers",
        marker=dict(symbol="line-ns-open", size=12, color="#4c78a8"), name="Pseudo median",
        hoverinfo="skip",
    ))
    hover = [
        f"{row.ticker}: actual {_format_kind(getattr(row, stat + '_matched'), kind)}, "
        f"pseudo median {_format_kind(getattr(row, stat + '_pseudo_median'), kind)}, "
        f"{row.trades} trades"
        for row in data.itertuples()
    ]
    figure.add_trace(go.Scatter(
        y=data["ticker"], x=_scaled(data[f"{stat}_matched"], kind), mode="markers",
        marker=dict(size=10, color=colours, line=dict(width=1, color="white")),
        name="Actual (matched sample)", text=hover, hoverinfo="text",
    ))
    if kind != "count":
        figure.add_vline(x=0, line_color="#999", line_width=1)
    figure.update_layout(
        height=max(420, 22 * len(data) + 120), margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title=f"{label}, before costs", **_axis(kind)),
        yaxis=dict(categoryorder="array", categoryarray=data["ticker"].tolist()),
        legend=dict(orientation="h", y=1.04), barmode="overlay",
    )
    return figure


def _median_gap_chart(gaps: pd.DataFrame, kind: str, x_title: str,
                      favourable_label: str, other_label: str) -> go.Figure:
    """Horizontal bars: actual minus pseudo median, coloured by side."""
    figure = go.Figure()
    if gaps.empty:
        figure.add_annotation(text="No stock has this statistic available.",
                              showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return figure
    for favourable, colour, name in ((True, FAVOURABLE, favourable_label),
                                     (False, UNFAVOURABLE, other_label)):
        subset = gaps[gaps["favourable"] == favourable]
        if subset.empty:
            continue
        figure.add_trace(go.Bar(
            y=subset["ticker"], x=_scaled(subset["gap"], kind), orientation="h",
            marker=dict(color=colour), name=name,
            text=[f"{t}: actual {_format_kind(a, kind)}, pseudo median "
                  f"{_format_kind(m, kind)}, gap {_format_kind(g, kind)}"
                  for t, a, m, g in zip(subset["ticker"], subset["actual"],
                                        subset["pseudo_median"], subset["gap"])],
            hoverinfo="text", textposition="none",
        ))
    figure.add_vline(x=0, line_color="#666", line_width=1)
    figure.update_layout(
        height=max(420, 22 * len(gaps) + 120), margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title=x_title, **_axis(kind)),
        yaxis=dict(categoryorder="array", categoryarray=gaps["ticker"].tolist()),
        legend=dict(orientation="h", y=1.04), barmode="overlay",
    )
    return figure


def _median_gap_caption(gaps: pd.DataFrame, favourable_text: str, kind: str) -> str:
    if gaps.empty:
        return ""
    n = len(gaps)
    k = int(gaps["favourable"].sum())
    return (
        f"{k} of {n} stocks {favourable_text}. The median gap across stocks is "
        f"{_format_kind(float(gaps['gap'].median()), kind)}. Each bar is the actual "
        "matched-sample value minus that stock's own pseudo median. If earnings dates "
        "behaved like the fake dates, about half of the stocks would land on each "
        "side; unlike the 5th\u201395th range, this comparison is not pushed outward "
        "by the extra volatility of earnings windows. The stocks are not independent, "
        "so the count is descriptive, not a test."
    )


def _quintile_dumbbell(table: pd.DataFrame, basis: str) -> go.Figure:
    """Mean outcome of Q1 (long) events vs Q5 (short) events, per stock."""
    q1, q5 = f"q1_mean_{basis}", f"q5_mean_{basis}"
    data = table.copy()
    for column in (q1, q5):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[q1, q5])
    data["gap"] = data[q1] - data[q5]
    data = data.sort_values("gap")
    figure = go.Figure()
    if data.empty:
        figure.add_annotation(text="No stock has trades in both Q1 and Q5.",
                              showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return figure
    labels = data["ticker"].tolist()
    for _, row in data.iterrows():
        figure.add_trace(go.Scatter(
            x=[row[q1], row[q5]], y=[row["ticker"], row["ticker"]], mode="lines",
            line=dict(color="#cccccc", width=2), hoverinfo="skip", showlegend=False,
        ))
    figure.add_trace(go.Scatter(
        x=data[q1], y=labels, mode="markers", marker=dict(color="#2a6fb5", size=11),
        name="Q1 events (PAR below 20th pct, long)",
        text=[f"{t}: Q1 mean {pct(v)} (n={n})"
              for t, v, n in zip(labels, data[q1], data["q1_count"])],
        hoverinfo="text",
    ))
    figure.add_trace(go.Scatter(
        x=data[q5], y=labels, mode="markers", marker=dict(color="#e08a2b", size=11),
        name="Q5 events (PAR above 80th pct, short)",
        text=[f"{t}: Q5 mean {pct(v)} (n={n})"
              for t, v, n in zip(labels, data[q5], data["q5_count"])],
        hoverinfo="text",
    ))
    figure.add_vline(x=0, line_width=1, line_color="#888")
    figure.update_layout(
        height=max(420, 22 * len(labels) + 120),
        xaxis=dict(title="Mean announcement-window return [-1,+1] (unsigned)",
                   tickformat=".1%"),
        yaxis=dict(categoryorder="array", categoryarray=labels),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return figure


def _strategy_section(results: dict, prefix: str) -> None:
    st.markdown("#### PAR quintile strategy across stocks")
    table = strategy_universe_table(results)
    ok = table[table["status"] == "ok"] if "status" in table else table.iloc[0:0]
    if ok.empty:
        st.info("No stock has strategy results.")
        return
    st.info(
        f"{ADAPTATION_NOTE} Announcement window [-1,+1]; each stock is ranked against "
        f"its own earlier events and needs {MIN_PRIOR_EVENTS} of them before trading. "
        f"{COSTS_NOTE} Same-close execution at t=\u22122 is assumed; see the stock's "
        "*PAR Quintile Strategy* tab for details."
    )
    mean_adj = pd.to_numeric(ok["mean_adj"], errors="coerce")
    positions = ok.get("mean_adj_position", pd.Series(dtype=str)).fillna("")
    columns = st.columns(4)
    columns[0].metric("Trades across all stocks", int(ok["trades"].sum()),
                      help=f"{int(ok['longs'].sum())} long, {int(ok['shorts'].sum())} short")
    columns[1].metric("Stocks with positive avg return / trade",
                      f"{int((mean_adj > 0).sum())} of {int(mean_adj.notna().sum())}")
    columns[2].metric("Median of per-stock avg return / trade",
                      _fmt(float(mean_adj.median()), True))
    columns[3].metric(
        "Above / below pseudo range",
        f"{int(positions.str.startswith('above').sum())} / "
        f"{int(positions.str.startswith('below').sum())}",
        help="Actual matched-sample average per trade against each stock's own pseudo "
             "5th\u201395th percentile range.",
    )
    st.markdown("##### Actual strategy result against each stock's pseudo range")
    stat = st.selectbox(
        "Strategy statistic", STRATEGY_CHART_STATS,
        format_func=lambda s: COMPARISON_STATS[s][0], key=f"{prefix}strategy_stat",
    )
    label, kind = COMPARISON_STATS[stat]
    st.plotly_chart(_strategy_band_chart(ok, stat), width="stretch",
                    key=f"{prefix}strategy_band_{stat}")
    st.caption(
        "Grey bar: that stock's pseudo 5th\u201395th percentile range, from the same rule "
        "applied to every fake-date repetition. Tick: pseudo median. Dot: the actual "
        "matched-sample value. Red: above the 95th percentile (better than on fake "
        "dates); purple: below the 5th. With "
        f"{int(ok['trades'].min())}\u2013{int(ok['trades'].max())} trades per stock, "
        "every dot is very noisy, and earnings-window returns are more volatile than "
        "fake-date returns, so dots land outside the range on both sides more often "
        f"than on fake dates. {PSEUDO_PERCENTILE_CAVEAT}"
        + (" Money values use $" f"{DEFAULT_NOTIONAL:,.0f} per trade." if kind == "money" else "")
    )

    if stat != "trades":
        st.markdown("##### Strategy result against each stock's pseudo median")
        gaps = pseudo_median_gap(ok, stat, "above")
        st.plotly_chart(
            _median_gap_chart(gaps, kind, f"{label}: actual minus pseudo median",
                              "Above pseudo median", "Below pseudo median"),
            width="stretch", key=f"{prefix}strategy_gap_{stat}",
        )
        st.caption(_median_gap_caption(gaps, "did better than their pseudo median", kind))

    st.markdown("##### Mean outcome in Q1 (long) vs Q5 (short) events")
    basis_label = st.radio("Return basis", ["Market-adjusted", "Raw stock"],
                           horizontal=True, key=f"{prefix}quintile_basis")
    basis = "raw" if basis_label == "Raw stock" else "adj"
    st.plotly_chart(_quintile_dumbbell(ok, basis), width="stretch",
                    key=f"{prefix}quintile_dumbbell_{basis}")
    st.caption(
        "Blue: average announcement-window return of the events the strategy went long "
        "(PAR below the stock's own historical 20th percentile). Orange: average return "
        "of the events it shorted (PAR above the 80th). These are plain, unsigned "
        "returns; a reversal shows blue right of zero and orange left of it. Stocks are "
        "sorted by Q1 minus Q5, which is a group spread, not the average return per "
        "trade. Each dot averages only "
        f"{int(min(ok['q1_count'].min(), ok['q5_count'].min()))}\u2013"
        f"{int(max(ok['q1_count'].max(), ok['q5_count'].max()))} events. Before costs."
    )
    view = ok[[
        "ticker", "name", "eligible_events", "trades", "longs", "shorts", "mean_adj",
        "win_adj", "pnl_adj", "mean_raw", "win_raw", "pnl_raw", "spread_adj",
        "sign_same_mean_adj", "mean_adj_pseudo_median", "mean_adj_pseudo_p5",
        "mean_adj_pseudo_p95", "mean_adj_position",
    ]].copy()
    for column in ("mean_adj", "win_adj", "mean_raw", "win_raw", "spread_adj",
                   "sign_same_mean_adj", "mean_adj_pseudo_median", "mean_adj_pseudo_p5",
                   "mean_adj_pseudo_p95"):
        view[column] = view[column].map(lambda v: _fmt(v, True))
    for column in ("pnl_adj", "pnl_raw"):
        view[column] = view[column].map(
            lambda v: "n/a" if v is None or not np.isfinite(v)
            else f"${v * DEFAULT_NOTIONAL:,.0f}"
        )
    st.dataframe(view.rename(columns={
        "ticker": "Ticker", "name": "Company", "eligible_events": "Events with history",
        "trades": "Trades", "longs": "Long", "shorts": "Short",
        "mean_adj": "Avg signed mkt-adj. / trade", "win_adj": "Win rate (mkt-adj.)",
        "pnl_adj": f"P&L mkt-adj. (${DEFAULT_NOTIONAL:,.0f}/trade)",
        "mean_raw": "Avg raw stock / trade", "win_raw": "Win rate (raw)",
        "pnl_raw": f"P&L raw (${DEFAULT_NOTIONAL:,.0f}/trade)",
        "spread_adj": "Q1 mean \u2212 Q5 mean (group spread)",
        "sign_same_mean_adj": "PAR sign strategy avg / trade (same events)",
        "mean_adj_pseudo_median": "Pseudo median avg / trade",
        "mean_adj_pseudo_p5": "Pseudo 5th pct", "mean_adj_pseudo_p95": "Pseudo 95th pct",
        "mean_adj_position": "Actual vs pseudo range",
    }), width="stretch", hide_index=True, height=520)
    st.caption(
        "All figures before costs. P&L is the sum of per-trade returns at a fixed "
        "notional, not compounded. The group spread compares Q1 and Q5 events at "
        "different dates and is not a strategy return. Stocks are not independent, so "
        "the count of profitable stocks is not a test."
    )
    download(st, table, "Download strategy summary (CSV)",
             f"{prefix}par_quintile_strategy.csv", f"{prefix}dl_strategy")


def render(results: dict, config, open_stock: Callable[[str], None],
           group: StockGroup) -> None:
    n = len(group.members)
    prefix = f"{group.key}_"
    st.subheader(group.label)
    st.markdown(
        "Every stock below is analysed **on its own**, with exactly the same windows, "
        "pseudo-date rules, seed and repetition count as the single-stock pages. "
        "This page only lines those separate results up side by side."
    )
    st.warning(
        f"These are {n} separate time-series event studies, not a cross-sectional "
        "portfolio sort. The stocks are **not independent**: most of them report in "
        f"the same weeks of each earnings season and share market-wide shocks, so {n} "
        f"stocks with similar results is weaker evidence than {n} unrelated studies. "
        "No pooled p-value is computed. The universe is chosen with hindsight from "
        "companies that are still large S&P 500 members, which is a survivorship bias."
    )

    failed = {t: r for t, r in results.items() if r is None or r.blocking_error}
    if failed:
        st.error(
            f"{len(failed)} stock(s) could not be analysed and are shown with their "
            f"reason in the tables below: " + ", ".join(failed)
        )

    labels = {f"{w.label} [{w.start:+d},{w.end:+d}]": w.key for w in OUTCOME_WINDOWS}
    chosen_label = st.selectbox("Outcome window", list(labels.keys()), index=0,
                                key=f"{prefix}window")
    window_key = labels[chosen_label]
    if window_key == "announcement":
        st.caption(ANNOUNCEMENT_WINDOW_CAVEAT)

    table = universe_table(results, window_key)
    ok = table[table["status"] == "ok"]

    columns = st.columns(4)
    columns[0].metric("Stocks analysed", f"{len(ok)} of {len(table)}")
    columns[1].metric("Earnings events included", int(ok["events_included"].sum()))
    slopes = pd.to_numeric(ok["slope"], errors="coerce").dropna()
    columns[2].metric("Stocks with a negative slope", f"{int((slopes < 0).sum())} of {slopes.size}")
    beyond = int((ok["slope_position"].astype(str).str.startswith("below")).sum())
    columns[3].metric("Slope below its pseudo 5th pct", f"{beyond} of {len(ok)}")

    st.markdown("#### How each statistic looks across the stocks")
    overview = cross_stock_overview(table)
    display = overview.drop(columns=["stat"]).copy()
    for column in ("Median across stocks", "Equal-weight mean across stocks"):
        display[column] = [
            _fmt(value, stat in PERCENT_SUMMARY_STATS)
            for value, stat in zip(overview[column], overview["stat"])
        ]
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "The reversal hypothesis predicts a negative correlation and slope, a "
        "positive mean after negative PAR, a negative mean after positive PAR, a "
        "positive negative-minus-positive difference and a reversal frequency above "
        "one half. \u201cBeyond pseudo range\u201d counts stocks whose actual statistic "
        "falls outside the 5th\u201395th percentile of that stock's own fake-date "
        "repetitions. Do not read that count as \u201cmore than chance\u201d: when "
        "returns on event dates are more volatile than on the fake dates (typical for "
        "the announcement window), statistics built from them stray outside the "
        "fake-date range more often on both sides. Compare the reversal-side count "
        "with the opposite-side count instead; similar counts on both sides point to "
        "extra noise, not reversal. Because the stocks are not independent, neither "
        "count is a test."
    )
    st.caption(_volatility_note(results, window_key))

    st.markdown("#### Actual statistic against each stock's pseudo range")
    stat = st.selectbox(
        "Statistic", list(SUMMARY_STATS), index=list(SUMMARY_STATS).index("slope"),
        format_func=lambda s: STAT_TITLES[s], key=f"{prefix}stat",
    )
    st.plotly_chart(_band_chart(table, stat), width="stretch")
    side = "below the 5th" if REVERSAL_SIDE[stat] == "below" else "above the 95th"
    st.caption(
        "Grey bar: that stock's pseudo 5th\u201395th percentile range. Tick: pseudo "
        "median. Dot: the actual matched-sample statistic. Red dots are stocks where "
        f"the actual value is {side} percentile, the side the reversal hypothesis "
        "predicts; purple dots are beyond the range on the opposite side. "
        f"{PSEUDO_PERCENTILE_CAVEAT}"
    )

    st.markdown("#### Actual statistic against each stock's pseudo median")
    gaps = pseudo_median_gap(table, stat, REVERSAL_SIDE[stat])
    percent = stat in PERCENT_SUMMARY_STATS
    toward = "below" if REVERSAL_SIDE[stat] == "below" else "above"
    st.plotly_chart(
        _median_gap_chart(gaps, "pct" if percent else "number",
                          f"{STAT_TITLES[stat]}: actual minus pseudo median",
                          "On the reversal side", "On the opposite side"),
        width="stretch", key=f"{prefix}gap_{stat}",
    )
    st.caption(_median_gap_caption(
        gaps, f"are {toward} their pseudo median, the side the reversal hypothesis "
              "predicts", "pct" if percent else "number",
    ))

    st.markdown("#### Mean outcome after negative PAR vs after positive PAR")
    st.plotly_chart(_dumbbell_chart(table), width="stretch")
    st.caption(
        "Blue: mean outcome return when the stock had a negative PAR. Orange: mean "
        "outcome return when PAR was positive. A clean reversal shows blue to the "
        "right of zero and orange to the left. Stocks are sorted by the gap between "
        "the two. Each dot is an average of roughly 30 events, so individual dots "
        "are noisy."
    )

    st.markdown("#### Per-stock results")
    st.dataframe(_per_stock_view(table), width="stretch", hide_index=True, height=520)
    download(st, table, "Download full per-stock summary (CSV)",
             f"{prefix}summary_{window_key}.csv", f"{prefix}dl_{window_key}")

    st.markdown("#### All events pooled together (descriptive)")
    events = pooled_events(results)
    outcome_column = f"{window_key}_adj"
    pooled = pooled_group_table(events, outcome_column)
    if pooled.empty:
        st.info("No events are available to pool.")
    else:
        view = pooled.copy()
        for column in ("Mean outcome", "Median outcome", "Opposite-sign frequency"):
            view[column] = view[column].map(lambda v: _fmt(v, True))
        st.dataframe(view, width="stretch", hide_index=True)
        neg = pd.to_numeric(ok["negative_mean"], errors="coerce").dropna()
        pos = pd.to_numeric(ok["positive_mean"], errors="coerce").dropna()
        st.caption(
            f"{len(events)} events from {events['ticker'].nunique()} stocks treated as one "
            "list. Equal-weight averages of the per-stock means are "
            f"{_fmt(float(neg.mean()) if neg.size else None, True)} after negative PAR and "
            f"{_fmt(float(pos.mean()) if pos.size else None, True)} after positive PAR. "
            "Events from different stocks on the same days are not independent, so "
            "the pooled count overstates how much evidence these numbers contain."
        )
        download(st, events, "Download all pooled events (CSV)",
                 f"{prefix}events.csv", f"{prefix}dl_events")

    _strategy_section(results, prefix)

    with st.expander("Data coverage per stock"):
        coverage = table[[
            "ticker", "name", "sector", "events_in_range", "events_included",
            "events_excluded", "matched_events", "first_event", "last_event",
            "unknown_release_time", "status",
        ]].rename(columns={
            "ticker": "Ticker", "name": "Company", "sector": "Sector",
            "events_in_range": "Completed events in range",
            "events_included": "Included", "events_excluded": "Excluded",
            "matched_events": "Matched with pseudo dates",
            "first_event": "First event", "last_event": "Last event",
            "unknown_release_time": "Unknown release time", "status": "Status",
        })
        st.dataframe(coverage, width="stretch", hide_index=True)
        st.caption(
            "Earnings dates come from Yahoo Finance, which returns at most 100 records "
            "per company. Coverage was checked for every stock before it was added to "
            "the universe. Recent quarters may be missing; the last event column shows "
            "where each stock's data actually ends."
        )

    with st.expander(f"How these {n} stocks were chosen"):
        st.markdown(group.selection_notes)
        st.dataframe(
            pd.DataFrame([{"Ticker": m.ticker, "Company": m.name, "Sector": m.sector}
                          for m in group.members]),
            width="stretch", hide_index=True,
        )

    st.markdown("#### Open one stock")
    left, right = st.columns([3, 1])
    target = left.selectbox("Stock", list(group.tickers),
                            format_func=display_name, key=f"{prefix}open_choice")
    right.button("Open its pages", on_click=open_stock, args=(target,),
                 key=f"{prefix}open_button", width="stretch")
