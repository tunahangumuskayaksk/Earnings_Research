# Fields

Columns built (or snapped) when constructing the announcement table `df` from `df_all`.  
Source Parquet columns are documented in [`dataset.md`](dataset.md).

Pipeline: `df_all` → `df_align` (trading-day align + benchmark) → **decision-day features** + **targets** → `df`  
(`df`: one row per earnings announcement that joined to a price day; see [`dataset.md`](dataset.md)).

---

## Calendar fields

Built from `EventDate`, the trading calendar, and `MarketTimeId`.

| Column         | Description                                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `EventDate`    | Original earnings calendar date from the earnings table (carried onto `df`).                                                                 |
| `ReleaseDate`  | Trading day of the announcement on the aligned panel. For events that already match a price `Date`, this equals `EventDate` in the sample.   |
| `AnchorDate`   | Return-window origin (`t = 0`). Before open (`MarketTimeId = 1`): same as `ReleaseDate`. After close (`MarketTimeId = 2`): next trading day. |
| `DecisionDate` | Feature as-of date: `AnchorDate` minus 2 trading days (`DECISION_OFFSET = -2`). Predictors must not use information after this close.        |
| `MarketTimeId` | Release timing flag from earnings: `1` = before open, `2` = after close.                                                                     |

Weekend / missing-price `EventDate` values never enter `df_all` (exact join), so they are not snapped forward into `df`.

Chronological split (`SPLIT_COLUMN`) may use any of: `EventDate`, `DecisionDate`, `AnchorDate`, `ReleaseDate`.

---

## Identity helpers

On **`df`** (event table):

| Column      | Description                                                                      |
| ----------- | -------------------------------------------------------------------------------- |
| `TickerId`  | Internal ticker id (from the panel).                                             |
| `Symbol`    | Ticker symbol split from `ShortName` (e.g. `ABNB`). Used for analyst asof merge. |
| `ShortName` | Full `SYMBOL:EXCHANGE` string on the event row.                                  |
| `Id`        | Earnings-event id when present in the source table.                              |

On **`df_all` / `df_align`** (not copied onto `df`):

| Column     | Description                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------- |
| `Exchange` | Exchange code split from `ShortName` (e.g. `XNGS`). Selects the trading calendar in alignment. |

---

## Intermediate market series

| Column              | Description                                                                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `BenchmarkAdjClose` | SPY adjusted close (Yahoo), attached on `df_align` and used only to build market-adjusted returns. Cached under `data/`. Not a column of `df`. |

Stock return legs use `LastPrice`; market legs use `BenchmarkAdjClose`. Treat `*_mkt` columns as approximate excess returns across vendors/fields.

---

## Target / label columns

All windows are in **trading days relative to `AnchorDate`**, unless noted.  
Cumulative window return: **close at `start_offset − 1` through close at `end_offset`** (see `src/features.py`).

| Column         | Description                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `ret_ann`      | Announcement window labeled **[-1, +1]** in offsets: close at `AnchorDate−2` → close at `AnchorDate+1` (three trading-day span of closes). |
| `ret_ann_mkt`  | `ret_ann` minus the same window on SPY. Default modeling target.                                                                           |
| `ret_post`     | Post window **[+1, +5]**: close at `AnchorDate` → close at `AnchorDate+5`. Overlaps the ann window on day `+1` by construction.            |
| `ret_post_mkt` | `ret_post` minus the same window on SPY.                                                                                                   |
| `ret_next`     | One-step forward stock return from the **announcement** (`ReleaseDate`) close to the next trading day.                                     |
| `ret_next_mkt` | `ret_next` minus the same one-step move on SPY.                                                                                            |
| `dir_ann`      | `sign(ret_ann_mkt)` for classification experiments; in this sample almost only `+1` / `−1` (exact zero is rare).                           |
| `vol_ann`      | Sample standard deviation of daily stock returns for offsets in `[-1, +1]`.                                                                |
| `vol_post`     | Sample standard deviation of daily stock returns for offsets in `[+1, +5]`.                                                                |

Offsets are fixed in `src/features.py`:  
`ANN_START_OFFSET = -1`, `ANN_END_OFFSET = 1`, `POST_START_OFFSET = 1`, `POST_END_OFFSET = 5`.

---

## Decision-day feature columns (not newly engineered)

These are **copied** from the `DecisionDate` row of `df_align` (not recomputed by IMP). They are the model inputs in `FEATURE_COLUMNS`.

| Column                     | Source          | Description                                                                                       |
| -------------------------- | --------------- | ------------------------------------------------------------------------------------------------- |
| `QScoreTrendLong`          | `stock-data_us` | Long-horizon trend technical score at decision.                                                   |
| `QScoreSpeedLong`          | `stock-data_us` | Long-horizon speed / momentum score at decision.                                                  |
| `QScoreLong`               | `stock-data_us` | `Trend + Speed` at decision; omit from `FEATURE_COLUMNS` if Trend and Speed are already included. |
| `analystRatingsStrongBuy`  | sentiment asof  | Strong Buy count as of decision.                                                                  |
| `analystRatingsBuy`        | sentiment asof  | Buy count as of decision.                                                                         |
| `analystRatingsHold`       | sentiment asof  | Hold count as of decision.                                                                        |
| `analystRatingsSell`       | sentiment asof  | Sell count as of decision.                                                                        |
| `analystRatingsStrongSell` | sentiment asof  | Strong Sell count as of decision.                                                                 |

Analyst columns exist on the daily panel only after the backward `merge_asof` in `src/prepare.py` (31-day tolerance). A non-trivial share of events can still be null when no snapshot falls in the window.

---

## Timing sketch

```
DecisionDate              ReleaseDate (= EventDate when joined)
        |                            |
        |<-  (depends on MarketTimeId)
        |                            v
        *-----*-----*=====ann=====*-----*-----*-----*  post
                    ^             ^
         close Anchor-2    AnchorDate / Anchor+1
                    |             |
         ret_ann: close[-2] → close[+1]; ret_post: close[0] → close[+5]
```

Features: values on `DecisionDate` only.  
Labels: returns / vols around `AnchorDate` (and `ret_next` from `ReleaseDate`).
