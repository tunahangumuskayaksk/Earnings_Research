# Data QA notes

Findings from comparing [`dataset.md`](dataset.md) / [`fields.md`](fields.md) to a run of
`script/run_data.py`, `script/run_feature.py`, and `script/run_preprocess.py`.
Observed types and counts: [`data/swagger.yaml`](../data/swagger.yaml).

Each item keeps the original observation, then a short **Response** (doc fix and/or open follow-up).
This file replaced `data/inconsistencies.md` (same audit, clearer name, lives with other docs).

---

## 1. Not every raw earnings row becomes an event row

**Claim (docs):** Pipeline yields one row per earnings announcement.

**Observed:** Raw earnings 3,928 → `df_all` event days 3,841 → `df` 3,837. Drops include ticker `16577` (no price rows), weekend `EventDate`s, and other exact-join misses.

**Response:** Documented. [`dataset.md`](dataset.md) now states that `df` is one row per **joined** announcement and explains the exact `Date`/`EventDate` join. No code change yet; changing to asof/snap-before-join would alter the sample and needs an explicit decision.

---

## 2. Closed-day snap rarely applies

**Claim (docs):** A closed `EventDate` is moved to the next session (`ReleaseDate`).

**Observed:** `ReleaseDate == EventDate` on all 3,837 `df` rows. Weekend / missing-price events never enter `df_all`, so snap never runs for them.

**Response:** Documented. [`fields.md`](fields.md) now describes `ReleaseDate` as the panel trading day for events that already match a price date, and notes that non-joined events are not snapped. Behavior unchanged.

---

## 3. `Exchange` / `BenchmarkAdjClose` are not on `df`

**Claim (docs):** `Exchange` listed with event identity helpers; benchmark series easy to misread as a `df` column.

**Observed:** `Exchange` and `BenchmarkAdjClose` live on `df_all` / `df_align`. Event table `df` has neither.

**Response:** Documented. [`fields.md`](fields.md) splits identity helpers into **on `df`** vs **on panel / `df_align`**, and states that `BenchmarkAdjClose` is not a `df` column.

---

## 4. `ret_ann` / `ret_post` window naming

**Claim (docs):** `ret_ann` is the window `[-1, +1]` (easy to read as a 2-day move).

**Observed:** Cumulative return uses close at `start_offset − 1` → close at `end_offset`, so `[-1, +1]` is close at `AnchorDate−2` → `AnchorDate+1`. `ret_post` `[+1, +5]` starts at the anchor close; the two windows overlap on day `+1`.

**Response:** Documented. [`fields.md`](fields.md) spells out the close-to-close endpoints and the intentional overlap. Offset constants in `src/features.py` unchanged.

---

## 5. `dir_ann` is almost never 0

**Claim (docs):** `dir_ann` is `+1` / `0` / `−1`.

**Observed:** Only `+1` (2,027) and `−1` (1,807) in sample; 3 nulls; 0 does not appear.

**Response:** Documented. [`fields.md`](fields.md) now says the label is `sign(ret_ann_mkt)` and is almost only `±1` here. Reserved for classification as before.

---

## 6. Sentiment `tickerId` vs stock `TickerId`

**Claim (docs):** Sentiment `tickerId` is the same entity as stock `TickerId`.

**Observed:** Some stock ids missing from sentiment and vice versa; `ANET:XNYS` maps to two stock `TickerId`s. Join still works because the pipeline uses **`symbol`**, not `tickerId`.

**Response:** Documented. [`dataset.md`](dataset.md) notes the id mismatch and that asof merge keys on `symbol`. No join-key change.

---

## 7. `ShortName_before` is not a useful alias

**Claim (docs):** Historical short name / alias for matching.

**Observed:** Equals `symbol` wherever non-null (5 nulls).

**Response:** Documented. [`dataset.md`](dataset.md) describes the field as present in source but not a separate alias in this sample.

---

## 8. `QScoreLong` is Trend + Speed

**Claim (docs):** Combined long-horizon score (sounded like an independent factor).

**Observed:** Exactly `QScoreTrendLong + QScoreSpeedLong` on every non-null row → perfect collinearity if all three enter the model.

**Response:** Documented in [`dataset.md`](dataset.md) / [`fields.md`](fields.md). Default `FEATURE_COLUMNS` still lists all three for backward compatibility; prefer dropping `QScoreLong` when Trend and Speed are both used (demo / config edit). Code default left unchanged for now.

---

## 9. Non-USD rows in the “US” price file

**Claim (docs):** `CurrencyId` commonly `840` for U.S. listings.

**Observed:** `ARE:XTSE` priced in CAD (`124`), 1,255 rows, mixed into the same price/mcap columns as USD names.

**Response:** Documented. [`dataset.md`](dataset.md) notes the CAD exception. Optional filter of non-USD names is an open modeling choice, not applied in the default pipeline.

---

## 10. Coverage wording (~200 names, 2020–2024)

**Claim (README / docs):** On the order of ~200 names, 2020–2024.

**Observed:** 200 `TickerId`s, 199 `ShortName`s, 198 symbols; sentiment dates span 2019–2026, but the price panel used is 2020–2024.

**Response:** Documented. [`dataset.md`](dataset.md) and README keep the approximate coverage language and note the symbol-count nuance / sentiment date span.

---

## Other data-quality notes (no separate doc bugs)

| Note                                                  | Response                                                         |
| ----------------------------------------------------- | ---------------------------------------------------------------- |
| Raw `EventDate` is string; `MarketTimeId` is float    | Accepted source quirks; loaders coerce types.                    |
| `QScore*` nulls in raw / rare nulls on `df`           | Expected; impute on train in preprocess.                         |
| Analyst nulls on ~8.7% of events                      | Documented under decision-day features; asof miss.               |
| `df_all` duplicate-ish columns (`symbol`/`Symbol`, …) | Merge artifacts; dropped where needed in feature build.          |
| Stock file sorted newest-first                        | Harmless after sort in pipeline steps.                           |
| Val/test feature drift vs train                       | Real distribution shift; monitor in metrics, not a schema error. |
