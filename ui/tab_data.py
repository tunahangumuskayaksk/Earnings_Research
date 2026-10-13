"""Tab A: Data & Coverage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from earnings_reversal.cache import humanise_age
from earnings_reversal.earnings import (
    COMPLETED_RULE,
    MANUAL_TEMPLATE_COLUMNS,
    manual_template_csv,
    select_analysis_events,
)
from earnings_reversal.prices import describe_adjustment
from .formatting import date_frame, download


def _coverage_cards(result) -> None:
    inputs = result.inputs
    diagnostics = result.diagnostics
    columns = st.columns(4)
    columns[0].metric("Stock", result.config.ticker)
    columns[1].metric("Benchmark", result.config.benchmark)
    columns[2].metric("Completed events in range", diagnostics.get("completed_in_range", 0))
    columns[3].metric("Events used in analysis", len(result.included_events))

    rows = []
    for price in (inputs.stock, inputs.benchmark):
        if price is None:
            continue
        info = describe_adjustment(price)
        rows.append({
            "Ticker": price.ticker,
            "Requested from": price.provenance.requested_start,
            "Requested to": price.provenance.requested_end,
            "Actual first row": price.provenance.actual_start,
            "Actual last row": price.provenance.actual_end,
            "Expected sessions": info["sessions_expected"],
            "Sessions with adjusted close": info["sessions_with_adj_close"],
            "Splits in range": info["splits_in_range"],
            "Dividend dates in range": info["dividends_in_range"],
            "Retrieved (UTC)": price.provenance.retrieved_at_utc,
        })
    if rows:
        st.markdown("#### Price coverage actually obtained")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _provenance_section(result) -> None:
    with st.expander("Provenance, adjustment settings and package versions"):
        for price in (result.inputs.stock, result.inputs.benchmark):
            if price is None:
                continue
            st.markdown(f"**{price.ticker} prices**")
            st.json(price.provenance.to_dict(), expanded=False)
        if result.inputs.earnings is not None:
            st.markdown("**Earnings dates**")
            st.json(result.inputs.earnings.provenance.to_dict(), expanded=False)
            st.caption(
                f"Installed earnings endpoint signature: "
                f"`get_earnings_dates{result.inputs.earnings.api_signature}`"
            )


def _cache_section(result, cache) -> None:
    st.markdown("#### Cache freshness")
    rows = []
    for price in (result.inputs.stock, result.inputs.benchmark):
        if price is None:
            continue
        rows.append({
            "Artefact": f"{price.ticker} prices",
            "Cache key": price.provenance.cache_key,
            "Retrieved (UTC)": price.provenance.retrieved_at_utc,
            "Age": humanise_age(cache.age_seconds("prices", price.provenance.cache_key)),
            "Rows": price.provenance.row_count,
            "SHA-256 (first 16)": price.provenance.file_sha256[:16],
        })
    if result.inputs.earnings is not None:
        provenance = result.inputs.earnings.provenance
        rows.append({
            "Artefact": f"{result.config.ticker} earnings dates",
            "Cache key": provenance.cache_key,
            "Retrieved (UTC)": provenance.retrieved_at_utc,
            "Age": humanise_age(cache.age_seconds("earnings", provenance.cache_key)),
            "Rows": provenance.row_count,
            "SHA-256 (first 16)": provenance.file_sha256[:16],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Data is served from the local cache so that changing a widget does not "
        "trigger a new download. Use **Refresh data** in the sidebar to force one."
    )


def _earnings_section(result) -> None:
    diagnostics = result.diagnostics
    st.markdown("#### Earnings-date coverage")

    columns = st.columns(4)
    columns[0].metric("Earliest completed record",
                      str(diagnostics.get("earliest_available") or "none"))
    columns[1].metric("Latest completed record",
                      str(diagnostics.get("latest_available") or "none"))
    columns[2].metric("Unverified in range", diagnostics.get("unverified_count", 0))
    columns[3].metric("Conflicting records", diagnostics.get("conflicting_count", 0))

    for warning in diagnostics.get("warnings", []):
        st.warning(warning)

    by_year = diagnostics.get("counts_by_year")
    if by_year is not None and len(by_year):
        st.markdown("**Completed events by year (years with zero records are shown)**")
        st.dataframe(by_year, width="stretch", hide_index=True)
        download(st, by_year, "Download counts by year (CSV)",
                 "earnings_counts_by_year.csv", "dl_year_counts")

    st.markdown("**Source and verification status**")
    reconciled = result.reconciled
    if reconciled is not None and len(reconciled):
        view_columns = [
            "release_date", "release_time", "time_known", "session_class", "source",
            "verified", "status", "status_reason", "conflict", "conflict_detail",
            "override_note", "source_url",
        ]
        view = date_frame(reconciled[[c for c in view_columns if c in reconciled.columns]])
        st.dataframe(view, width="stretch", hide_index=True, height=320)
        download(st, reconciled, "Download all earnings records (CSV)",
                 "earnings_records.csv", "dl_earnings_records")
    else:
        st.info("No earnings records are available from any source yet.")

    excluded = diagnostics.get("excluded_records")
    if excluded is not None and len(excluded):
        with st.expander(
            f"Records not used as events ({len(excluded)}) — future, unconfirmed, "
            f"duplicate, superseded or conflicting"
        ):
            st.dataframe(date_frame(excluded), width="stretch", hide_index=True)

    st.caption(f"**Completed-release rule.** {COMPLETED_RULE}")
    st.caption(
        "Four announcements per year is a diagnostic expectation only. No event is "
        "ever created from a calendar rule, assumed 90-day spacing or interpolation."
    )


def _upload_section(ticker: str) -> None:
    st.markdown("#### Verified earnings-date upload")
    st.markdown(
        "Upload a CSV to add or override release dates you have checked against a "
        "company release or another source. Expected columns: "
        + ", ".join(f"`{c}`" for c in MANUAL_TEMPLATE_COLUMNS)
        + ". `release_date` must be `YYYY-MM-DD`; `release_time` is optional; "
        "`timezone` defaults to `America/New_York`; `verified` must be a boolean and "
        "requires a `source_url`. A verified manual record wins over a downloaded "
        "record describing the same release, and the override is disclosed."
    )
    columns = st.columns([1, 2])
    columns[0].download_button(
        "Download empty template (CSV)",
        data=manual_template_csv().encode("utf-8"),
        file_name="verified_earnings_template.csv",
        mime="text/csv",
        key="dl_template",
    )
    uploaded = columns[1].file_uploader(
        "Upload verified earnings dates", type=["csv"], key="manual_upload"
    )
    if uploaded is not None:
        st.session_state["manual_csv_text"] = uploaded.getvalue().decode(
            "utf-8-sig", errors="replace"
        )
    if st.session_state.get("manual_csv_text"):
        if st.button("Clear uploaded earnings file", key="clear_manual"):
            st.session_state["manual_csv_text"] = ""
            st.rerun()

    errors = st.session_state.get("manual_errors") or []
    if errors:
        st.error(f"{len(errors)} problem(s) in the uploaded file. No invalid row was used.")
        st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)
    accepted = st.session_state.get("manual_accepted", 0)
    for_ticker = st.session_state.get("manual_for_ticker", 0)
    if accepted:
        st.success(
            f"{accepted} uploaded row(s) accepted. {for_ticker} of them are for "
            f"{ticker} and are merged on this page. Rows for other tickers are used "
            f"when you select those stocks and in the multi-stock summary."
        )


def _exclusion_section(result) -> None:
    st.markdown("#### Inspect and exclude questionable events")
    ticker = result.config.ticker
    # Options come from the unfiltered candidates so an excluded date stays listed.
    candidates = select_analysis_events(
        result.reconciled, result.config.event_start, result.config.event_end
    )
    if candidates is None or len(candidates) == 0:
        st.info("No candidate events are available to review.")
        return
    options = [pd.Timestamp(d).date().isoformat() for d in candidates["release_date"]]
    by_ticker = st.session_state.setdefault("excluded_by_ticker", {})
    stored = [d for d in by_ticker.get(ticker, []) if d in options]
    chosen = st.multiselect(
        f"{ticker} release dates to exclude from the analysis",
        options=options,
        default=stored,
        help="Excluded events are removed before any statistic is computed, on this "
             "stock's pages and in the multi-stock summary.",
        key=f"exclusion_multiselect_{ticker}",
    )
    if list(chosen) != list(by_ticker.get(ticker, [])):
        by_ticker[ticker] = list(chosen)
        st.rerun()
    if chosen:
        st.info(f"{len(chosen)} {ticker} event(s) excluded by you: " + ", ".join(chosen))


def render(result, cache) -> None:
    st.subheader("Data & Coverage")
    st.caption(
        "Everything on this page reflects what was actually retrieved. Nothing is "
        "filled in, interpolated or simulated."
    )

    if result.inputs.price_error:
        st.error(f"Price download problem: {result.inputs.price_error}")
    if result.inputs.earnings_error:
        st.error(f"Earnings-date problem: {result.inputs.earnings_error}")
    for message in result.messages:
        st.info(message)

    _coverage_cards(result)

    notes = []
    for price in (result.inputs.stock, result.inputs.benchmark):
        if price is not None:
            notes.extend(price.coverage_notes)
    if notes:
        with st.expander("Price coverage notes and missing observations", expanded=False):
            for note in notes:
                st.markdown(f"- {note}")

    for note in (result.inputs.earnings.notes if result.inputs.earnings else []):
        st.warning(note)

    _provenance_section(result)
    _cache_section(result, cache)
    st.divider()
    _earnings_section(result)
    st.divider()
    _upload_section(result.config.ticker)
    st.divider()
    _exclusion_section(result)

    if len(result.excluded_events):
        st.divider()
        st.markdown("#### Events excluded by the alignment and data checks")
        st.dataframe(date_frame(result.excluded_events), width="stretch",
                     hide_index=True)
        download(st, result.excluded_events, "Download exclusions (CSV)",
                 "excluded_events.csv", "dl_exclusions")
