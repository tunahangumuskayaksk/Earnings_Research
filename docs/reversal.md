# Reversal on the IMP panel

The reversal study uses IMP's three Parquet files and the same event builder as `script/run_feature.py`. It does not download Yahoo prices and it does not keep the calendar-date `t = 0` from the earlier 70-stock app.

## What is shared

| Piece | Where it comes from |
| --- | --- |
| Prices, earnings, analyst ratings | `stock-data_us.parquet`, `earnings_us.parquet`, `historical_sentiment_scores_us.parquet` |
| Universe | Every `TickerId` in that panel (the ~200-name file behind `data/stock_universe.csv`) |
| Session clock | `src/sessions.py` |
| Before-open / after-close anchor | `MarketTimeId`: 1 stays on the release session, 2 moves to the next session |
| Announcement target | `ret_ann_mkt`: close at anchor−2 through close at anchor+1, stock minus benchmark |
| Benchmark | `config.default.BENCHMARK_SOURCE` (equal-weight universe index, not SPY) |
| Split | `DecisionDate`: train ≤ 2022-12-31, val through 2023-12-31, test 2024 |
| Scores | `src.metrics.evaluate_regression` and `quintile_table` |

## What the reversal adds

PAR is the stock's market-adjusted return from the close at anchor−5 to the close at anchor−2. That end date is `DecisionDate`, so the signal is known at the close the announcement window starts from. Joining it does not put PAR into `FEATURE_COLUMNS`, so `main.py` is unchanged until someone chooses to add it.

Two predictors are scored like IMP models:

- `reversal_ols`: `ret_ann_mkt ~ par_mkt`, coefficients from the training fold only. A negative slope is the reversal.
- `positioning_ols`: the same regression on IMP's eight QScore and analyst-rating columns, with medians filled from train only.

`python script/run_reversal.py --with-lin` also runs the existing linear family (`ols`, `ridge`, `lasso`) into `output/lin/`. Trees and networks stay on `main.py`.

Trading rules are reported separately. Their average signed return is not an IMP R². The sign rule trades every event. The per-stock quintiles need 20 earlier events for that name; 2020–2022 does not contain 20, so those rules record zero trades. `quintile_pooled_train` uses the 20th and 80th percentile of all training-fold PAR values, which is the same kind of cross-section as IMP's prediction buckets.

## How this differs from the 70-stock Yahoo study

The Yahoo app kept `t = 0` on the calendar release date even for after-close prints, used SPY, and did not impose the 2022 / 2023 / 2024 cut. Those results are not comparable to this table. This table is.

## Fitting it into the main pipeline

PAR can become one more decision-day column: compute it in `build_event_table` and append `par_mkt` to `FEATURE_COLUMNS`. Until Edward wants that, leave the feature list alone and read `output/reversal/metrics.csv` next to `output/lin/metrics.csv`. `compare_app.py` does that.

Do not average the reversal sample with the model sample without matching `DecisionDate` and dropping rows that lack PAR. The script already scores each model on the rows it can use.
