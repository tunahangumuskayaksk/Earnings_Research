"""Historical earnings release dates: retrieval, normalisation and reconciliation.

The event studied here is the *first public earnings release*, which may precede
the conference call. Nothing in this module invents a date: records are only ever
taken from the data source or from a user-supplied verified CSV, and anything that
cannot be established as a completed release is excluded and reported.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .cache import DiskCache, Provenance, package_versions, utc_now_iso
from .config import (
    DATA_SOURCE,
    EARNINGS_REQUEST_LIMIT,
    EARNINGS_TIMEZONE,
    MAX_DOWNLOAD_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
)
from .trading_calendar import classify_release_session, is_session

RECORD_COLUMNS = [
    "ticker", "release_date", "release_time", "timezone", "release_timestamp_ny",
    "source_timestamp_raw", "time_known", "session_class", "source", "source_url",
    "verified", "status", "status_reason", "eps_estimate", "reported_eps",
    "surprise_pct",
]

# Records whose timestamp is exactly midnight are date-only rows padded by the
# provider; they must not be treated as a known release time.
PLACEHOLDER_TIME = time(0, 0)

# Two records within this many calendar days are assumed to describe the same
# underlying release when checking for conflicts between sources.
CLUSTER_TOLERANCE_DAYS = 5

COMPLETED_RULE = (
    "A record is retained as a completed release only if (a) its release "
    "timestamp is in the past relative to the retrieval time, and (b) it carries "
    "at least one realised figure (reported EPS or surprise %) or it is a "
    "user-verified manual record. Date-only rows are evaluated at the end of the "
    "release day. Future estimates and past-dated rows with no realised figure "
    "are excluded and listed."
)


class EarningsFetchError(RuntimeError):
    """Raised when earnings dates could not be retrieved and no cache exists."""


@dataclass
class EarningsData:
    records: pd.DataFrame
    provenance: Provenance
    notes: list[str] = field(default_factory=list)
    api_signature: str = ""


def _empty_records() -> pd.DataFrame:
    frame = pd.DataFrame({col: pd.Series(dtype="object") for col in RECORD_COLUMNS})
    return frame


def _fetch_raw(ticker: str, limit: int) -> tuple[pd.DataFrame, str, list[str]]:
    """Call the installed yfinance earnings endpoint, inspecting its real signature."""
    import inspect

    handle = yf.Ticker(ticker)
    notes: list[str] = []
    getter = getattr(handle, "get_earnings_dates", None)
    if getter is None:
        raise EarningsFetchError(
            "The installed yfinance build exposes no get_earnings_dates(); "
            "use the verified-CSV upload instead."
        )
    try:
        signature = str(inspect.signature(getter))
    except (TypeError, ValueError):
        signature = "(unavailable)"
    kwargs: dict[str, Any] = {}
    if "limit" in signature:
        kwargs["limit"] = limit
    else:
        notes.append(
            "The installed get_earnings_dates() takes no 'limit' argument, so only "
            "its default page of history could be requested."
        )

    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            frame = getter(**kwargs)
            if frame is None:
                raise EarningsFetchError(f"{ticker}: earnings endpoint returned None.")
            if len(frame) == 0:
                raise EarningsFetchError(f"{ticker}: earnings endpoint returned no rows.")
            if "limit" in kwargs and len(frame) < limit:
                notes.append(
                    f"Requested up to {limit} earnings rows; the source returned "
                    f"{len(frame)}. This is the provider's ceiling, not a local "
                    f"truncation, so history may be incomplete."
                )
            return frame, signature, notes
        except Exception as exc:
            last_error = exc
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time_module.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise EarningsFetchError(
        f"{ticker}: earnings-date retrieval failed after {MAX_DOWNLOAD_ATTEMPTS} "
        f"attempts. Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _column(frame: pd.DataFrame, *candidates: str) -> pd.Series | None:
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return frame[lowered[candidate.lower()]]
    return None


def normalise_yahoo_earnings(frame: pd.DataFrame, ticker: str,
                             retrieved_at: datetime | None = None) -> pd.DataFrame:
    """Turn the provider's frame into the canonical record layout."""
    if frame is None or len(frame) == 0:
        return _empty_records()
    retrieved_at = retrieved_at or datetime.now(timezone.utc)

    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        localized = index.tz_localize(EARNINGS_TIMEZONE)
        tz_note = "naive"
    else:
        localized = index.tz_convert(EARNINGS_TIMEZONE)
        tz_note = str(index.tz)

    eps_estimate = _column(frame, "EPS Estimate")
    reported_eps = _column(frame, "Reported EPS")
    surprise = _column(frame, "Surprise(%)", "Surprise (%)", "surprise")

    rows: list[dict[str, Any]] = []
    for position, stamp in enumerate(localized):
        stamp = pd.Timestamp(stamp)
        local_time = stamp.time()
        time_known = local_time != PLACEHOLDER_TIME
        release_date = stamp.date()
        realised = []
        for series in (reported_eps, surprise):
            if series is not None:
                value = series.iloc[position]
                realised.append(value is not None and not pd.isna(value))
        has_realised = any(realised)

        # Date-only rows are judged at the end of their release day.
        effective = stamp if time_known else stamp.replace(hour=23, minute=59, second=59)
        in_past = effective.tz_convert("UTC").to_pydatetime() <= retrieved_at

        if not in_past:
            status, reason = "excluded", "Scheduled in the future at retrieval time."
        elif not has_realised:
            status, reason = (
                "excluded",
                "Past-dated row with no reported EPS or surprise; cannot be "
                "established as a completed release.",
            )
        else:
            status, reason = "completed", "Past-dated row with a realised figure."

        rows.append({
            "ticker": ticker,
            "release_date": pd.Timestamp(release_date),
            "release_time": local_time.strftime("%H:%M:%S") if time_known else None,
            "timezone": EARNINGS_TIMEZONE,
            "release_timestamp_ny": stamp.isoformat(),
            "source_timestamp_raw": f"{pd.Timestamp(frame.index[position])} (tz={tz_note})",
            "time_known": bool(time_known),
            "session_class": classify_release_session(
                release_date, local_time if time_known else None
            ),
            "source": "yahoo_get_earnings_dates",
            "source_url": f"https://finance.yahoo.com/calendar/earnings?symbol={ticker}",
            "verified": False,
            "status": status,
            "status_reason": reason,
            "eps_estimate": _safe_float(eps_estimate, position),
            "reported_eps": _safe_float(reported_eps, position),
            "surprise_pct": _safe_float(surprise, position),
        })

    records = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    return records.sort_values("release_date").reset_index(drop=True)


def _safe_float(series: pd.Series | None, position: int) -> float | None:
    if series is None:
        return None
    value = series.iloc[position]
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_earnings(ticker: str, cache: DiskCache, force_refresh: bool = False,
                  limit: int = EARNINGS_REQUEST_LIMIT) -> EarningsData:
    """Load earnings records from cache when possible, otherwise fetch them."""
    ticker = ticker.strip().upper()
    key = f"{ticker}"
    cached = None if force_refresh else cache.read("earnings", key)
    notes: list[str] = []
    signature = ""

    download_error: Exception | None = None
    if cached is None:
        try:
            raw, signature, fetch_notes = _fetch_raw(ticker, limit)
            notes.extend(fetch_notes)
            retrieved_at = datetime.now(timezone.utc)
            records = normalise_yahoo_earnings(raw, ticker, retrieved_at)
            provenance = Provenance(
                kind="earnings",
                tickers=[ticker],
                requested_start=None,
                requested_end=None,
                retrieved_at_utc=retrieved_at.isoformat(timespec="seconds"),
                data_source=DATA_SOURCE,
                adjustment_settings={
                    "endpoint": "Ticker.get_earnings_dates",
                    "signature": signature,
                    "requested_limit": limit,
                    "completed_rule": COMPLETED_RULE,
                },
                package_versions=package_versions(),
                cache_key=key,
                cache_file="",
                actual_start=str(records["release_date"].min().date()) if len(records) else None,
                actual_end=str(records["release_date"].max().date()) if len(records) else None,
                notes=list(notes),
            )
            provenance = cache.write("earnings", key, records, provenance)
            cached = (records, provenance)
        except Exception as exc:
            download_error = exc
            fallback = cache.read("earnings", key)
            if fallback is None:
                raise EarningsFetchError(str(exc)) from exc
            cached = fallback

    records, provenance = cached
    notes = list(provenance.notes) + [n for n in notes if n not in provenance.notes]
    if download_error is not None:
        notes.insert(0, (
            f"Earnings refresh failed ({type(download_error).__name__}: "
            f"{download_error}). Showing the cache retrieved at "
            f"{provenance.retrieved_at_utc}."
        ))
    records = records.copy()
    records["release_date"] = pd.to_datetime(records["release_date"])
    return EarningsData(
        records=records,
        provenance=provenance,
        notes=notes,
        api_signature=provenance.adjustment_settings.get("signature", signature),
    )


# --------------------------------------------------------------------------
# Manual verified CSV
# --------------------------------------------------------------------------

MANUAL_TEMPLATE_COLUMNS = [
    "ticker", "release_date", "release_time", "timezone", "source_url", "verified",
]


def manual_template_csv() -> str:
    """An empty template: headers only, no invented sample dates."""
    return ",".join(MANUAL_TEMPLATE_COLUMNS) + "\n"


def parse_manual_csv(data: str | bytes,
                     retrieved_at: datetime | None = None
                     ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate an uploaded CSV, returning (records, row-level errors)."""
    import io

    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig", errors="replace")
    errors: list[dict[str, Any]] = []
    try:
        frame = pd.read_csv(io.StringIO(data), dtype=str).fillna("")
    except Exception as exc:
        return _empty_records(), [{"row": "-", "field": "file",
                                   "problem": f"Could not parse CSV: {exc}"}]

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in ("ticker", "release_date") if c not in frame.columns]
    if missing:
        return _empty_records(), [{
            "row": "-", "field": ",".join(missing),
            "problem": "Required column(s) missing. Expected headers: "
                       + ", ".join(MANUAL_TEMPLATE_COLUMNS),
        }]

    rows: list[dict[str, Any]] = []
    for position, row in frame.iterrows():
        line = int(position) + 2  # account for the header line
        row_errors: list[dict[str, Any]] = []

        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            row_errors.append({"row": line, "field": "ticker", "problem": "Empty ticker."})

        raw_date = str(row.get("release_date", "")).strip()
        release_date = None
        if not raw_date:
            row_errors.append({"row": line, "field": "release_date",
                               "problem": "Empty release_date."})
        else:
            try:
                release_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError:
                row_errors.append({
                    "row": line, "field": "release_date",
                    "problem": f"'{raw_date}' is not a YYYY-MM-DD date.",
                })

        raw_time = str(row.get("release_time", "")).strip()
        release_time: time | None = None
        if raw_time:
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    release_time = datetime.strptime(raw_time, fmt).time()
                    break
                except ValueError:
                    continue
            if release_time is None:
                row_errors.append({
                    "row": line, "field": "release_time",
                    "problem": f"'{raw_time}' is not HH:MM or HH:MM:SS.",
                })

        tz_name = str(row.get("timezone", "")).strip() or EARNINGS_TIMEZONE
        try:
            import zoneinfo
            zoneinfo.ZoneInfo(tz_name)
        except Exception:
            row_errors.append({"row": line, "field": "timezone",
                               "problem": f"Unknown timezone '{tz_name}'."})
            tz_name = EARNINGS_TIMEZONE

        source_url = str(row.get("source_url", "")).strip()
        verified_raw = str(row.get("verified", "")).strip().lower()
        verified = verified_raw in {"true", "yes", "y", "1"}
        if verified and not source_url:
            row_errors.append({
                "row": line, "field": "source_url",
                "problem": "verified=true requires a source_url identifying the release.",
            })
        if verified_raw and verified_raw not in {"true", "false", "yes", "no", "y", "n", "0", "1"}:
            row_errors.append({"row": line, "field": "verified",
                               "problem": f"'{verified_raw}' is not a boolean."})

        if row_errors:
            errors.extend(row_errors)
            continue

        local_time = release_time if release_time is not None else PLACEHOLDER_TIME
        stamp = pd.Timestamp(datetime.combine(release_date, local_time)).tz_localize(tz_name)
        stamp_ny = stamp.tz_convert(EARNINGS_TIMEZONE)
        time_known = release_time is not None
        # The calendar date that matters is the local New York release date.
        ny_date = stamp_ny.date()
        effective = stamp_ny if time_known else stamp_ny.replace(hour=23, minute=59, second=59)
        in_past = effective.tz_convert("UTC").to_pydatetime() <= retrieved_at
        if not in_past:
            status, reason = "excluded", "Release timestamp is in the future."
        elif not verified:
            status, reason = (
                "completed",
                "Manual record accepted as completed, but marked unverified by the uploader.",
            )
        else:
            status, reason = "completed", "User-verified manual record with a source URL."

        rows.append({
            "ticker": ticker,
            "release_date": pd.Timestamp(ny_date),
            "release_time": stamp_ny.time().strftime("%H:%M:%S") if time_known else None,
            "timezone": tz_name,
            "release_timestamp_ny": stamp_ny.isoformat(),
            "source_timestamp_raw": f"{raw_date} {raw_time} ({tz_name})".strip(),
            "time_known": bool(time_known),
            "session_class": classify_release_session(
                ny_date, stamp_ny.time() if time_known else None
            ),
            "source": "manual_upload",
            "source_url": source_url,
            "verified": bool(verified),
            "status": status,
            "status_reason": reason,
            "eps_estimate": None,
            "reported_eps": None,
            "surprise_pct": None,
        })

    records = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    if len(records):
        records = records.sort_values("release_date").reset_index(drop=True)
    return records, errors


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------

def reconcile_records(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge sources, drop exact duplicates and flag conflicting records.

    Records whose release dates fall within ``CLUSTER_TOLERANCE_DAYS`` of each
    other are treated as describing the same underlying release. Within a cluster
    a verified manual record wins and the override is disclosed; otherwise a
    disagreement about the date is flagged for review rather than resolved
    arbitrarily.
    """
    usable = [f for f in frames if f is not None and len(f)]
    if not usable:
        out = _empty_records()
        for column in ("cluster_id", "is_primary", "conflict", "conflict_detail",
                       "override_note", "duplicate_of"):
            out[column] = pd.Series(dtype="object")
        return out

    numeric = ["eps_estimate", "reported_eps", "surprise_pct"]
    usable = [f.astype({c: "float64" for c in numeric if c in f.columns})
              for f in usable]
    merged = pd.concat(usable, ignore_index=True)
    merged["release_date"] = pd.to_datetime(merged["release_date"])

    # Exact duplicates: identical source, timestamp and date.
    before = len(merged)
    merged = merged.drop_duplicates(
        subset=["ticker", "source", "release_timestamp_ny", "release_date"], keep="first"
    ).reset_index(drop=True)
    exact_dupes_removed = before - len(merged)

    merged = merged.sort_values(
        ["ticker", "release_date", "source"], kind="mergesort"
    ).reset_index(drop=True)

    cluster_ids: list[int] = []
    counter = -1
    last_date: pd.Timestamp | None = None
    last_ticker: str | None = None
    for _, row in merged.iterrows():
        same_ticker = row["ticker"] == last_ticker
        close_enough = (
            last_date is not None
            and abs((row["release_date"] - last_date).days) <= CLUSTER_TOLERANCE_DAYS
        )
        if not (same_ticker and close_enough):
            counter += 1
        cluster_ids.append(counter)
        last_date = row["release_date"]
        last_ticker = row["ticker"]
    merged["cluster_id"] = cluster_ids

    merged["is_primary"] = False
    merged["conflict"] = False
    merged["conflict_detail"] = ""
    merged["override_note"] = ""
    merged["duplicate_of"] = ""

    for cluster_id, group in merged.groupby("cluster_id"):
        indices = list(group.index)
        distinct_dates = sorted({d.date().isoformat() for d in group["release_date"]})
        verified_manual = group[(group["source"] == "manual_upload") & (group["verified"])]

        if len(verified_manual):
            primary = verified_manual.index[0]
            merged.loc[primary, "is_primary"] = True
            others = [i for i in indices if i != primary]
            if others:
                other_dates = sorted({
                    merged.loc[i, "release_date"].date().isoformat() for i in others
                })
                note = (
                    "Verified manual record overrides "
                    + ", ".join(
                        f"{merged.loc[i, 'source']} @ {merged.loc[i, 'release_date'].date()}"
                        for i in others
                    )
                )
                merged.loc[primary, "override_note"] = note
                for i in others:
                    merged.loc[i, "status"] = "superseded"
                    merged.loc[i, "status_reason"] = (
                        "Superseded by a user-verified manual record in the same cluster."
                    )
                    merged.loc[i, "duplicate_of"] = str(primary)
                if len(set(other_dates) | {merged.loc[primary, 'release_date'].date().isoformat()}) > 1:
                    merged.loc[primary, "override_note"] += (
                        " (sources disagreed on the date; the verified record was used)"
                    )
            continue

        if len(distinct_dates) > 1:
            detail = "Sources disagree on the release date: " + ", ".join(
                f"{merged.loc[i, 'source']}={merged.loc[i, 'release_date'].date()}"
                for i in indices
            )
            for i in indices:
                merged.loc[i, "conflict"] = True
                merged.loc[i, "conflict_detail"] = detail
                merged.loc[i, "status"] = "conflict"
                merged.loc[i, "status_reason"] = (
                    "Conflicting release dates flagged for review; not analysed "
                    "until resolved by a verified record."
                )
            continue

        # One agreed date. Prefer a record with a known release time.
        with_time = group[group["time_known"]]
        primary = with_time.index[0] if len(with_time) else indices[0]
        merged.loc[primary, "is_primary"] = True
        distinct_times = sorted({
            str(merged.loc[i, "release_time"]) for i in indices
            if merged.loc[i, "time_known"]
        })
        if len(distinct_times) > 1:
            merged.loc[primary, "conflict_detail"] = (
                "Sources agree on the date but report different release times: "
                + ", ".join(distinct_times)
            )
        for i in indices:
            if i != primary:
                merged.loc[i, "duplicate_of"] = str(primary)
                if merged.loc[i, "status"] == "completed":
                    merged.loc[i, "status"] = "duplicate"
                    merged.loc[i, "status_reason"] = (
                        "Same release already represented by another record in the cluster."
                    )

    merged.attrs["exact_duplicates_removed"] = exact_dupes_removed
    return merged


def confirmed_releases(reconciled: pd.DataFrame) -> pd.DatetimeIndex:
    """All completed primary release dates, used for contamination screening."""
    if reconciled is None or len(reconciled) == 0:
        return pd.DatetimeIndex([], name="release_date")
    mask = (reconciled["status"] == "completed") & reconciled["is_primary"]
    dates = pd.to_datetime(reconciled.loc[mask, "release_date"]).dt.normalize()
    return pd.DatetimeIndex(sorted(set(dates)), name="release_date")


def select_analysis_events(reconciled: pd.DataFrame, start, end,
                           excluded_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Completed, primary, in-range records the user has not excluded."""
    if reconciled is None or len(reconciled) == 0:
        return reconciled if reconciled is not None else _empty_records()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    excluded = {pd.Timestamp(d).normalize() for d in excluded_dates}
    frame = reconciled.copy()
    frame["release_date"] = pd.to_datetime(frame["release_date"]).dt.normalize()
    mask = (
        (frame["status"] == "completed")
        & frame["is_primary"]
        & (frame["release_date"] >= start)
        & (frame["release_date"] <= end)
        & ~frame["release_date"].isin(excluded)
    )
    return frame.loc[mask].sort_values("release_date").reset_index(drop=True)


def coverage_diagnostics(reconciled: pd.DataFrame, start, end) -> dict[str, Any]:
    """Counts by year (including zero years), gaps and suspicious coverage."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    result: dict[str, Any] = {
        "total_records": int(len(reconciled)) if reconciled is not None else 0,
        "earliest_available": None,
        "latest_available": None,
        "completed_in_range": 0,
        "counts_by_year": pd.DataFrame(columns=["year", "completed_events"]),
        "unverified_count": 0,
        "conflicting_count": 0,
        "excluded_records": pd.DataFrame(),
        "warnings": [],
    }
    if reconciled is None or len(reconciled) == 0:
        result["warnings"].append("No earnings records are available from any source.")
        return result

    frame = reconciled.copy()
    frame["release_date"] = pd.to_datetime(frame["release_date"]).dt.normalize()
    completed = frame[(frame["status"] == "completed") & frame["is_primary"]]
    if len(completed):
        result["earliest_available"] = completed["release_date"].min().date()
        result["latest_available"] = completed["release_date"].max().date()

    in_range = completed[(completed["release_date"] >= start) & (completed["release_date"] <= end)]
    result["completed_in_range"] = int(len(in_range))

    years = list(range(start.year, end.year + 1))
    counts = in_range["release_date"].dt.year.value_counts().to_dict()
    by_year = pd.DataFrame({
        "year": years,
        "completed_events": [int(counts.get(y, 0)) for y in years],
    })
    by_year["verified_events"] = [
        int(((in_range["release_date"].dt.year == y) & (in_range["verified"])).sum())
        for y in years
    ]
    by_year["known_release_time"] = [
        int(((in_range["release_date"].dt.year == y) & (in_range["time_known"])).sum())
        for y in years
    ]
    result["counts_by_year"] = by_year

    result["unverified_count"] = int((~in_range["verified"]).sum())
    result["conflicting_count"] = int(frame["conflict"].sum())
    result["excluded_records"] = frame[frame["status"] != "completed"][
        ["release_date", "source", "status", "status_reason", "release_timestamp_ny"]
    ].sort_values("release_date")

    zero_years = by_year.loc[by_year["completed_events"] == 0, "year"].tolist()
    if zero_years:
        result["warnings"].append(
            "Years inside the requested range with zero completed records: "
            + ", ".join(str(y) for y in zero_years)
        )
    thin_years = by_year.loc[
        (by_year["completed_events"] > 0) & (by_year["completed_events"] < 4), "year"
    ].tolist()
    if thin_years:
        result["warnings"].append(
            "Years with fewer than four completed records (a diagnostic signal only, "
            "never a licence to fabricate events): "
            + ", ".join(str(y) for y in thin_years)
        )
    crowded = by_year.loc[by_year["completed_events"] > 4, "year"].tolist()
    if crowded:
        result["warnings"].append(
            "Years with more than four completed records, worth reviewing for "
            "duplicates: " + ", ".join(str(y) for y in crowded)
        )
    if len(in_range) >= 2:
        gaps = in_range["release_date"].diff().dt.days.dropna()
        big = gaps[gaps > 150]
        if len(big):
            positions = in_range["release_date"].iloc[big.index].dt.date.tolist()
            result["warnings"].append(
                f"{len(big)} gap(s) longer than 150 days between consecutive records, "
                f"ending at: " + ", ".join(str(d) for d in positions[:6])
            )
    unknown_time = int((~in_range["time_known"]).sum())
    if unknown_time:
        result["warnings"].append(
            f"{unknown_time} of {len(in_range)} in-range records have no known release "
            f"time (date-only or midnight placeholder) and are classified 'unknown'."
        )
    if result["earliest_available"] and pd.Timestamp(result["earliest_available"]) > start:
        result["warnings"].append(
            f"Earliest available completed release is {result['earliest_available']}, "
            f"after the requested range start {start.date()}."
        )
    if result["latest_available"] and pd.Timestamp(result["latest_available"]) < end - pd.Timedelta(days=120):
        result["warnings"].append(
            f"Latest available completed release is {result['latest_available']}, "
            f"well before the requested range end {end.date()}. Recent quarters are "
            f"not covered."
        )
    return result
