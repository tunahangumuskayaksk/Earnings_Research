# Dataset

Source Parquet tables under `data/`. They are merged into a daily panel (`df_all`), then collapsed into an event table (`df`).

`df` has **one row per earnings announcement that survives the exact price-day join** (see join note below)—not every raw earnings row.

| File                                     | Granularity              | Role                                |
| ---------------------------------------- | ------------------------ | ----------------------------------- |
| `stock-data_us.parquet`                  | Ticker × trading day     | Prices and technical scores         |
| `earnings_us.parquet`                    | One row per announcement | Event date and pre/post-market flag |
| `historical_sentiment_scores_us.parquet` | Ticker × month           | Analyst rating counts               |

Coverage in the development sample is on the order of **~200 U.S. names** and price dates **2020–2024** (ticker / symbol counts are not exactly equal; sentiment snapshots extend outside that price window but only overlapping dates are used).

---

## `stock-data_us.parquet`

Daily equity panel (prices, volume, market cap, and long-horizon QScore features).

| Column            | Description                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| `TickerId`        | Internal numeric ticker identifier; join key with earnings.                                             |
| `ShortName`       | Symbol and exchange, e.g. `ABNB:XNGS`.                                                                  |
| `InstrumentName`  | Company / instrument display name.                                                                      |
| `Date`            | Trading calendar date for the row.                                                                      |
| `LastPrice`       | Closing price used for return construction.                                                             |
| `Open`            | Session open price.                                                                                     |
| `High`            | Session high price.                                                                                     |
| `Low`             | Session low price.                                                                                      |
| `Volume`          | Traded share volume for the session.                                                                    |
| `CurrencyId`      | Currency for prices: almost all rows are `840` (USD); a small CAD book exists (`124`, e.g. `ARE:XTSE`). |
| `QScoreTrendLong` | Long-horizon trend technical score.                                                                     |
| `QScoreSpeedLong` | Long-horizon speed / momentum technical score.                                                          |
| `QScoreLong`      | Equals `QScoreTrendLong + QScoreSpeedLong` on observed rows (redundant if all three are model inputs).  |
| `MarketCap`       | Market capitalization on that date (in `CurrencyId` units).                                             |

---

## `earnings_us.parquet`

Earnings announcement events (calendar date and market timing).

| Column         | Description                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `Id`           | Unique earnings-event identifier.                                                                       |
| `TickerId`     | Internal ticker id; intended to match `stock-data_us.TickerId` (a few earnings ids have no price rows). |
| `EventDate`    | Calendar date of the earnings release (stored as string in the parquet; parsed on load).                |
| `MarketTimeId` | Release timing vs the session: `1` = before open, `2` = after close. Sets the return anchor session.    |

Joined to the price panel with an **exact** left join on  
`Date` + `TickerId` ↔ `EventDate` + `TickerId`.

Events with no matching price row are dropped before event-table construction (no price ticker, weekend `EventDate`, or other calendar days without a stock row). Closed-day “snap to next session” only applies to events that already sit on a price date in `df_all`.

---

## `historical_sentiment_scores_us.parquet`

Monthly analyst-rating snapshots (counts by recommendation bucket).

| Column                     | Description                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| `symbol`                   | Ticker symbol string, e.g. `ABNB` (join key after splitting stock `ShortName`).                       |
| `date`                     | Snapshot date (typically month-start).                                                                |
| `analystRatingsStrongBuy`  | Number of Strong Buy ratings in the snapshot.                                                         |
| `analystRatingsBuy`        | Number of Buy ratings.                                                                                |
| `analystRatingsHold`       | Number of Hold ratings.                                                                               |
| `analystRatingsSell`       | Number of Sell ratings.                                                                               |
| `analystRatingsStrongSell` | Number of Strong Sell ratings.                                                                        |
| `ShortName_before`         | Present in source; where non-null it equals `symbol` in this sample (not a separate alias field).     |
| `tickerId`                 | Internal id with lowercase `t`. Not always 1:1 with stock `TickerId`; the pipeline joins on `symbol`. |

Attached to the daily panel with a backward `merge_asof` on symbol and date (latest snapshot on or before each price date, within a 31-day tolerance).

---

## How the three tables connect

```
stock-data_us
    ├─ left join earnings_us          (Date/TickerId = EventDate/TickerId)
    └─ asof join historical_sentiment (Symbol/Date ← symbol/date)
                ↓
             df_all   (daily panel)
                ↓
        build_event_table → df   (one row per joined announcement)
```

Observed schema / counts: [`data/swagger.yaml`](../data/swagger.yaml).  
QA notes vs docs: [`data_qa.md`](data_qa.md).
