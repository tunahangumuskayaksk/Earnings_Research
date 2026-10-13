"""Earnings-record normalisation, time zones, duplicates, conflicts and uploads."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from earnings_reversal.earnings import (
    confirmed_releases,
    coverage_diagnostics,
    manual_template_csv,
    normalise_yahoo_earnings,
    parse_manual_csv,
    reconcile_records,
    select_analysis_events,
)

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _yahoo_frame(entries):
    """entries: list of (timestamp string in NY, reported_eps, surprise)."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(stamp) for stamp, _, _ in entries]
    ).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "EPS Estimate": [1.0] * len(entries),
            "Reported EPS": [eps for _, eps, _ in entries],
            "Surprise(%)": [surprise for _, _, surprise in entries],
        },
        index=index,
    )


def test_future_records_are_excluded():
    frame = _yahoo_frame([
        ("2024-02-01 16:30", 1.2, 3.0),
        ("2024-08-01 16:30", np.nan, np.nan),   # scheduled, in the future
    ])
    records = normalise_yahoo_earnings(frame, "TEST", NOW)
    statuses = dict(zip(records["release_date"].dt.date.astype(str), records["status"]))
    assert statuses["2024-02-01"] == "completed"
    assert statuses["2024-08-01"] == "excluded"
    assert "future" in records.loc[
        records["release_date"].dt.date.astype(str) == "2024-08-01", "status_reason"
    ].iloc[0].lower()


def test_past_record_without_a_realised_figure_is_excluded():
    frame = _yahoo_frame([("2024-02-01 16:30", np.nan, np.nan)])
    records = normalise_yahoo_earnings(frame, "TEST", NOW)
    assert records.loc[0, "status"] == "excluded"
    assert "cannot be established as a completed release" in records.loc[0, "status_reason"]


def test_midnight_placeholder_is_not_a_known_release_time():
    frame = _yahoo_frame([("2024-02-01 00:00", 1.2, 3.0)])
    records = normalise_yahoo_earnings(frame, "TEST", NOW)
    assert records.loc[0, "time_known"] is False or records.loc[0, "time_known"] == False  # noqa: E712
    assert records.loc[0, "release_time"] is None
    assert records.loc[0, "session_class"] == "unknown"
    assert records.loc[0, "verified"] == False  # noqa: E712
    assert records.loc[0, "status"] == "completed"  # date-only, but still a real release


def test_timezone_conversion_near_a_date_boundary():
    """A 23:30 UTC release on 2024-02-02 is 18:30 on 2024-02-01 in New York."""
    index = pd.DatetimeIndex([pd.Timestamp("2024-02-02 23:30", tz="UTC")])
    frame = pd.DataFrame(
        {"EPS Estimate": [1.0], "Reported EPS": [1.1], "Surprise(%)": [2.0]}, index=index
    )
    records = normalise_yahoo_earnings(frame, "TEST", NOW)
    assert records.loc[0, "release_date"] == pd.Timestamp("2024-02-02")
    assert records.loc[0, "release_time"] == "18:30:00"
    assert records.loc[0, "session_class"] == "after_close"


def test_timezone_conversion_can_move_the_date_backwards():
    """00:30 UTC on 2024-02-03 is 19:30 on 2024-02-02 in New York."""
    index = pd.DatetimeIndex([pd.Timestamp("2024-02-03 00:30", tz="UTC")])
    frame = pd.DataFrame(
        {"EPS Estimate": [1.0], "Reported EPS": [1.1], "Surprise(%)": [2.0]}, index=index
    )
    records = normalise_yahoo_earnings(frame, "TEST", NOW)
    assert records.loc[0, "release_date"] == pd.Timestamp("2024-02-02")
    assert records.loc[0, "release_time"] == "19:30:00"


def test_manual_csv_timezone_conversion_crosses_midnight():
    csv = (
        "ticker,release_date,release_time,timezone,source_url,verified\n"
        "TEST,2024-02-05,22:30,America/Los_Angeles,https://example.com/pr,true\n"
    )
    records, errors = parse_manual_csv(csv, NOW)
    assert errors == []
    # 22:30 Pacific on Monday Feb 5 is 01:30 Eastern on Tuesday Feb 6.
    assert records.loc[0, "release_date"] == pd.Timestamp("2024-02-06")
    assert records.loc[0, "release_time"] == "01:30:00"
    assert records.loc[0, "session_class"] == "before_open"


def test_manual_csv_timezone_shift_onto_a_weekend_is_classified_as_non_trading():
    csv = (
        "ticker,release_date,release_time,timezone,source_url,verified\n"
        "TEST,2024-02-02,22:30,America/Los_Angeles,https://example.com/pr,true\n"
    )
    records, errors = parse_manual_csv(csv, NOW)
    assert errors == []
    # 22:30 Pacific on Friday Feb 2 is 01:30 Eastern on Saturday Feb 3.
    assert records.loc[0, "release_date"] == pd.Timestamp("2024-02-03")
    assert records.loc[0, "session_class"] == "non_trading_day"


def test_exact_duplicates_are_removed():
    frame = _yahoo_frame([("2024-02-01 16:30", 1.2, 3.0)])
    records = normalise_yahoo_earnings(frame, "TEST", NOW)
    reconciled = reconcile_records([records, records.copy()])
    assert len(reconciled) == 1
    assert reconciled.attrs["exact_duplicates_removed"] == 1
    assert bool(reconciled.loc[0, "is_primary"])


def test_conflicting_dates_are_flagged_not_silently_resolved():
    yahoo = normalise_yahoo_earnings(_yahoo_frame([("2024-02-01 16:30", 1.2, 3.0)]),
                                     "TEST", NOW)
    other = yahoo.copy()
    other["release_date"] = pd.Timestamp("2024-02-02")
    other["release_timestamp_ny"] = "2024-02-02T16:30:00-05:00"
    other["source"] = "second_source"

    reconciled = reconcile_records([yahoo, other])
    assert len(reconciled) == 2
    assert reconciled["conflict"].all()
    assert (reconciled["status"] == "conflict").all()
    assert not reconciled["is_primary"].any()
    assert "disagree" in reconciled.loc[0, "conflict_detail"]
    # A flagged conflict never reaches the analysis.
    assert len(select_analysis_events(reconciled, "2024-01-01", "2024-12-31")) == 0


def test_verified_manual_record_resolves_a_conflict_and_discloses_the_override():
    yahoo = normalise_yahoo_earnings(_yahoo_frame([("2024-02-01 16:30", 1.2, 3.0)]),
                                     "TEST", NOW)
    csv = (
        "ticker,release_date,release_time,timezone,source_url,verified\n"
        "TEST,2024-02-02,16:30,America/New_York,https://example.com/pr,true\n"
    )
    manual, errors = parse_manual_csv(csv, NOW)
    assert errors == []

    reconciled = reconcile_records([yahoo, manual])
    primary = reconciled[reconciled["is_primary"]]
    assert len(primary) == 1
    assert primary.iloc[0]["source"] == "manual_upload"
    assert primary.iloc[0]["release_date"] == pd.Timestamp("2024-02-02")
    assert "overrides" in primary.iloc[0]["override_note"]
    assert "disagreed on the date" in primary.iloc[0]["override_note"]
    superseded = reconciled[reconciled["source"] == "yahoo_get_earnings_dates"]
    assert superseded.iloc[0]["status"] == "superseded"


def test_time_conflict_on_an_agreed_date_is_reported():
    yahoo = normalise_yahoo_earnings(_yahoo_frame([("2024-02-01 16:30", 1.2, 3.0)]),
                                     "TEST", NOW)
    other = yahoo.copy()
    other["release_time"] = "16:05:00"
    other["release_timestamp_ny"] = "2024-02-01T16:05:00-05:00"
    other["source"] = "second_source"
    reconciled = reconcile_records([yahoo, other])
    primary = reconciled[reconciled["is_primary"]].iloc[0]
    assert "different release times" in primary["conflict_detail"]
    assert len(select_analysis_events(reconciled, "2024-01-01", "2024-12-31")) == 1


def test_manual_csv_row_level_errors():
    csv = (
        "ticker,release_date,release_time,timezone,source_url,verified\n"
        "TEST,2024-13-45,16:30,America/New_York,https://example.com,true\n"
        "TEST,2024-02-01,25:99,America/New_York,https://example.com,true\n"
        ",2024-02-02,,America/New_York,https://example.com,true\n"
        "TEST,2024-02-05,,Mars/Olympus,https://example.com,true\n"
        "TEST,2024-02-06,,America/New_York,,true\n"
        "TEST,2024-02-07,16:30,America/New_York,https://example.com,true\n"
    )
    records, errors = parse_manual_csv(csv, NOW)
    fields = {e["field"] for e in errors}
    assert "release_date" in fields
    assert "release_time" in fields
    assert "ticker" in fields
    assert "timezone" in fields
    assert "source_url" in fields
    assert len(records) == 1  # only the final valid row survives
    assert records.loc[0, "release_date"] == pd.Timestamp("2024-02-07")


def test_manual_csv_missing_required_column():
    records, errors = parse_manual_csv("ticker,when\nTEST,2024-02-01\n", NOW)
    assert len(records) == 0
    assert "release_date" in errors[0]["field"]


def test_manual_future_record_is_excluded():
    csv = (
        "ticker,release_date,release_time,timezone,source_url,verified\n"
        "TEST,2025-02-01,16:30,America/New_York,https://example.com,true\n"
    )
    records, errors = parse_manual_csv(csv, NOW)
    assert errors == []
    assert records.loc[0, "status"] == "excluded"
    assert "future" in records.loc[0, "status_reason"].lower()


def test_template_is_empty_with_no_invented_dates():
    template = manual_template_csv()
    assert template.strip() == "ticker,release_date,release_time,timezone,source_url,verified"
    assert len(template.strip().splitlines()) == 1


def test_coverage_diagnostics_reports_zero_years_and_gaps():
    frame = _yahoo_frame([
        ("2022-02-01 16:30", 1.0, 1.0),
        ("2022-05-01 16:30", 1.0, 1.0),
        ("2022-08-01 16:30", 1.0, 1.0),
        ("2022-11-01 16:30", 1.0, 1.0),
        ("2024-02-01 16:30", 1.0, 1.0),
    ])
    reconciled = reconcile_records([normalise_yahoo_earnings(frame, "TEST", NOW)])
    diagnostics = coverage_diagnostics(reconciled, "2022-01-01", "2024-12-31")
    by_year = diagnostics["counts_by_year"].set_index("year")["completed_events"].to_dict()
    assert by_year == {2022: 4, 2023: 0, 2024: 1}
    assert any("zero completed records" in w for w in diagnostics["warnings"])
    assert any("2023" in w for w in diagnostics["warnings"])
    assert any("gap(s) longer than 150 days" in w for w in diagnostics["warnings"])


def test_confirmed_releases_includes_dates_outside_the_selected_range():
    frame = _yahoo_frame([
        ("2021-02-01 16:30", 1.0, 1.0),
        ("2022-02-01 16:30", 1.0, 1.0),
    ])
    reconciled = reconcile_records([normalise_yahoo_earnings(frame, "TEST", NOW)])
    releases = confirmed_releases(reconciled)
    assert len(releases) == 2
    selected = select_analysis_events(reconciled, "2022-01-01", "2022-12-31")
    assert len(selected) == 1
    # Screening must still see the 2021 release even though it is not an event.
    assert pd.Timestamp("2021-02-01") in releases


def test_user_exclusions_are_applied():
    frame = _yahoo_frame([
        ("2022-02-01 16:30", 1.0, 1.0),
        ("2022-05-01 16:30", 1.0, 1.0),
    ])
    reconciled = reconcile_records([normalise_yahoo_earnings(frame, "TEST", NOW)])
    selected = select_analysis_events(reconciled, "2022-01-01", "2022-12-31",
                                      excluded_dates=("2022-02-01",))
    assert len(selected) == 1
    assert selected.loc[0, "release_date"] == pd.Timestamp("2022-05-01")


def test_empty_sources_produce_an_empty_reconciliation():
    reconciled = reconcile_records([])
    assert len(reconciled) == 0
    assert len(confirmed_releases(reconciled)) == 0
    diagnostics = coverage_diagnostics(reconciled, "2020-01-01", "2020-12-31")
    assert diagnostics["completed_in_range"] == 0
    assert any("No earnings records" in w for w in diagnostics["warnings"])
