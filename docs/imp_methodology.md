# IMP methodology (one page)

**Identifying Market Positioning Ahead of Earnings.** The question is predictive: using only information known **before** the announcement, can a model forecast the stock’s market-adjusted return in the announcement window?

The sample is Brian’s three Parquet files: about 200 U.S. names, prices 2020–2024. One row is one earnings print that already sits on a price day. Features are copied from two sessions before the return clock starts. Nothing from the release itself enters the model.

## How a print is placed on the calendar

`MarketTimeId` says when the company reported. **1 = before the open:** the return clock (`AnchorDate`) is that session. **2 = after the close:** the clock moves to the **next** session. `EventDate` is the vendor’s calendar date. `ReleaseDate` is that date on the trading calendar. `DecisionDate` is `AnchorDate` minus two trading sessions. Predictors must not use anything after that close.

```
DecisionDate          ReleaseDate              AnchorDate (t = 0)
     |                     |                        |
     close −2              print                    before-open: same day
                                                    after-close: next session
     features taken here   not used in the model    announcement window
                                                    close(−2) → close(+1)
```

Train: `DecisionDate` through 31 Dec 2022. Validation: 2023 (ridge / lasso / trees pick penalties here). Test: 2024. Medians and scaling are fit on train only.

## What the model is trying to forecast

Windows are trading sessions relative to `AnchorDate`. A window `[a, b]` is the close at session `a−1` through the close at session `b`. The default target is **`ret_ann_mkt`**. The market subtracted is the configured benchmark (equal-weight universe index by default, not SPY).

| Variable | Meaning |
| --- | --- |
| `ret_ann` | Stock return, announcement window `[−1, +1]`: close at anchor−2 → close at anchor+1. |
| `ret_ann_mkt` | `ret_ann` minus the same window on the benchmark. **Default target.** |
| `ret_post` | Stock return after the print, `[+1, +5]`: close at the anchor → close at anchor+5. Overlaps the announcement window on day +1. |
| `ret_post_mkt` | `ret_post` minus the same window on the benchmark. |
| `ret_next` | One-session stock return from the `ReleaseDate` close to the next session (not from the anchor). |
| `ret_next_mkt` | `ret_next` minus the same one-session move on the benchmark. |
| `dir_ann` | Sign of `ret_ann_mkt` (`+1` / `−1`). Reserved for classification; not the current target. |
| `vol_ann` | Standard deviation of daily stock returns inside `[−1, +1]`. |
| `vol_post` | Standard deviation of daily stock returns inside `[+1, +5]`. |

## Variables the model is allowed to see

These eight numbers are **copied** from the `DecisionDate` row. IMP does not recompute them. `QScoreLong` equals `QScoreTrendLong + QScoreSpeedLong` on observed rows.

| Variable | Source | Meaning |
| --- | --- | --- |
| `QScoreTrendLong` | Daily price file | Vendor long-horizon **trend** score at the decision close. |
| `QScoreSpeedLong` | Daily price file | Vendor long-horizon **speed / momentum** score at the decision close. |
| `QScoreLong` | Daily price file | Trend + speed at the decision close. |
| `analystRatingsStrongBuy` | Monthly analyst file | Count of Strong Buy ratings on the latest snapshot on or before the decision date (31-day lookback). |
| `analystRatingsBuy` | Monthly analyst file | Count of Buy ratings on that snapshot. |
| `analystRatingsHold` | Monthly analyst file | Count of Hold ratings. |
| `analystRatingsSell` | Monthly analyst file | Count of Sell ratings. |
| `analystRatingsStrongSell` | Monthly analyst file | Count of Strong Sell ratings. |

If no analyst snapshot falls inside 31 days, those five counts are missing. Train-only median fill is used. A snapshot after `DecisionDate` is never used.

## Other variables that build the row (not model inputs)

| Variable | Meaning |
| --- | --- |
| `TickerId` | Internal name id. Joins prices to earnings. Analysis is by this id, not by `Symbol`. |
| `Symbol` | Ticker (`ABNB`). Used to attach the monthly analyst file. The same symbol can appear on two ids (ANET, ARE). |
| `ShortName` | `SYMBOL:EXCHANGE` (e.g. `ABNB:XNGS`). The exchange picks the session calendar (US names use NYSE; `ARE:XTSE` uses Toronto). |
| `InstrumentName` | Company name. |
| `Date` | Trading date on the daily price row. |
| `LastPrice` | Closing price used for stock returns. Adjustment convention is not stated in-file. |
| `Open`, `High`, `Low`, `Volume` | Session prices and share volume. Not in the default feature list. |
| `MarketCap` | Market value on that date. Used only if the benchmark is cap-weighted. |
| `CurrencyId` | `840` = USD; a few rows are CAD. |
| `Id` | Vendor earnings-event id. |
| `EventDate` | Calendar date of the release. |
| `ReleaseDate` | That date on the trading calendar. |
| `AnchorDate` | `t = 0` for the return windows (after-close is shifted). |
| `DecisionDate` | Feature as-of date: two sessions before the anchor. **Split date.** |
| `MarketTimeId` | `1` before open, `2` after close. Unknown values drop the event. |
| `BenchmarkAdjClose` | Benchmark level on the aligned panel. Builds every `*_mkt` return. Not a column of the event table. |
| `symbol`, `date` | Keys on the monthly analyst file (`symbol` + month-start `date`). |

## Models and scores

Linear (OLS, ridge, lasso), trees, and a small network share the same target, features, and split. Ridge and lasso choose their penalty on 2023 by maximizing out-of-sample R².

| Score | Meaning |
| --- | --- |
| `oos_r2` | Fraction of 2024 (or 2023) variation explained versus guessing the **2020–2022 average** return. Primary score. |
| `r2` | Same idea, but versus that fold’s own average (in-sample for train). |
| `ic` | Pearson correlation of the forecast with the realized return. |
| `rank_ic` | Spearman (rank) correlation. |
| `hit_rate` | Share of prints where the forecast and the realized return have the same sign. Zeros count as misses. |
| `rmse`, `mae` | Typical size of the forecast error. |
| `rmse_train_mean`, `mae_train_mean` | Error if you always guess the training-period average. |
| `rmse_zero`, `mae_zero` | Error if you always guess zero. |
| `top_minus_bottom` | Average realized return in the highest predicted fifth minus the lowest. |

A useful model should beat the training-mean and zero guesses and show a non-negative `oos_r2` on 2024. The linear models on this panel do not: 2024 `oos_r2` is slightly negative, and lasso collapses to the training average.
