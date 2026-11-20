# IMP

**Identifying Market Positioning Ahead of Earnings.**

Predict post-earnings stock behavior from information available **before** the announcement—not from the release itself. The repo builds an event panel from daily price, earnings, and analyst-rating Parquet files, then trains linear, tree, and neural baselines under one chronological split.

## Features

- Merge daily prices, earnings events, and monthly analyst ratings
- NYSE session alignment with BMO / AMC return anchors; decision day = anchor − 2
- Fixed announcement / post windows and SPY market-adjusted returns
- Train / val / test by calendar date; impute and scale **on train only**
- Multi-model batches via `main.py` (`FAMILIES`) with shared metrics and `output/` artifacts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Global defaults

Shared paths, source filenames, benchmark ticker, target, and split cuts live in [`config/default.py`](config/default.py).  
`config/config_*.py` only keeps family-specific knobs (model types, grids, `STANDARDIZE`, …).

## Data

Place provider Parquet files in `data/` (not redistributed here). Column-level docs:

- [`docs/dataset.md`](docs/dataset.md) — source tables
- [`docs/fields.md`](docs/fields.md) — event-table dates, labels, decision-day features
- [`docs/metrics.md`](docs/metrics.md) — evaluation scores and baselines
- [`docs/data_qa.md`](docs/data_qa.md) — doc vs data QA notes and resolutions

| File                                     | Role                                      |
| ---------------------------------------- | ----------------------------------------- |
| `stock-data_us.parquet`                  | Daily prices and QScore features          |
| `earnings_us.parquet`                    | Event dates and `MarketTimeId`            |
| `historical_sentiment_scores_us.parquet` | Monthly analyst rating counts             |
| `stock_universe.csv`                     | One row per `TickerId` for WRDS / external data mapping (see below) |
| `benchmark_spy.parquet` / `.json`        | Optional Yahoo SPY cache (when `BENCHMARK_SOURCE=yahoo`) |
| `benchmark_equal|cap.parquet` / `.json`  | Optional universe-index cache (when `BENCHMARK_SOURCE=equal|cap`) |

Sample panel used in development: on the order of **~200 U.S. names** with price dates **2020–2024** (ticker/symbol counts are not exact equals; see `docs/dataset.md`). Stock returns use `LastPrice` (adjustment convention not stated in-file).

### Stock universe CSV

[`data/stock_universe.csv`](data/stock_universe.csv) is a minimal download list (one row per `TickerId`) for WRDS / other external pulls: identifiers plus price-panel date coverage. Regenerate with `python script/export_universe.py`.

| Column | Notes |
| ------ | ----- |
| `TickerId` | Join key back to IMP |
| `Symbol`, `Exchange` | Split from `ShortName` (e.g. `ABNB` / `XNGS`) |
| `InstrumentName` | Company name for manual checks |
| `price_start`, `price_end` | Suggested download window |

Same `Symbol` may appear on more than one row (see `docs/data_qa.md`); keep those rows when mapping vendor ids.

## Quick start

```bash
python script/run_data.py          # preview df_all
python script/run_feature.py       # build df + descriptive report
python script/run_preprocess.py    # split + train-only impute / scale
python script/export_universe.py   # write data/stock_universe.csv

python main.py                     # edit FAMILIES = ("lin",) or ("lin", "tr", "nn")
python script/run_reversal.py      # PAR reversal on the same panel and splits
python script/run_reversal.py --with-lin
streamlit run compare_app.py       # reversal vs linear IMP models
```

Reversal vs the positioning models: [docs/reversal.md](docs/reversal.md).  
IMP methodology (one page, every variable): [docs/imp_methodology.md](docs/imp_methodology.md).

Notebook walkthrough: `notebook/demo.ipynb` (set `FAMILIES` in CONFIG, then run in order).

## Pipeline

```text
data/*.parquet
    -> prepare      load + merge              -> df_all
    -> sessions     trading calendar
    -> benchmark    market Adj Close (yahoo / equal / cap)
    -> features     event rows                -> df
    -> preprocess   train / val / test + train-only impute / scale
    -> main.py      fit, metrics, output/
```

| Name     | Meaning                                       |
| -------- | --------------------------------------------- |
| `df_all` | Daily panel after earnings and analyst merges |
| `df`     | One row per earnings announcement             |

Design summary (details in `docs/`):

- Earnings: exact left join on `Date`/`TickerId` ↔ `EventDate`/`TickerId`
- Analyst ratings: backward `merge_asof` by symbol, 31-day tolerance
- Anchor from `MarketTimeId` (1 = BMO same day, 2 = AMC next session)
- Decision features snapped at `DecisionDate`; default target `ret_ann_mkt` (announcement [-1, +1] vs configured benchmark)

## Preprocessing

1. Drop rows with missing `TARGET` (never impute the label).
2. Chronological split on `SPLIT_COLUMN` via `TRAIN_END` / `VAL_END` (defaults `2022-12-31` / `2023-12-31`).
3. Fit impute / standardize **on train only**, then transform other folds.

Val is used for **selection and early stopping** (maximize OOS R²; tie-break lower RMSE)—not only for reporting:

| Family | Selected on val         | Early stopping       |
| ------ | ----------------------- | -------------------- |
| lin    | `alpha` (ridge / lasso) | —                    |
| tr     | `max_depth`             | HistGBDT: val OOS R² |
| nn     | L2 `alpha`              | all MLPs: val OOS R² |

HistGBDT / MLP do not use sklearn’s random `validation_fraction` holdout from train.

## Models

| Family | Config                 | Types                                                      |
| ------ | ---------------------- | ---------------------------------------------------------- |
| lin    | `config/config_lin.py` | `ols`, `ridge`, `lasso`                                    |
| tr     | `config/config_tr.py`  | `decision_tree`, `random_forest`, `hist_gradient_boosting` |
| nn     | `config/config_nn.py`  | `mlp_shallow`, `mlp`, `mlp_deep`                           |

Select families in `main.py` (`FAMILIES`). Classification (`dir_ann`) is reserved. Shared target / features / cuts: `config/default.py`.

## Metrics and outputs

Definitions: [`docs/metrics.md`](docs/metrics.md) (OOS R², baselines, IC, quintiles, permutation importance).

Terminal blocks: **METRICS**, optional **FEATURE**, **QUINTILES**, then **SUMMARY** (test only; includes `top_minus_bottom`).  
Artifacts under `output/{lin,tr,nn}/` (gitignored): `metrics.csv`, `predictions.parquet`, `run_config.json`.

## Layout

```text
config/
  default.py       global data / benchmark / split defaults
  config_*.py      family-specific model knobs
model/           estimators
src/             prepare, sessions, benchmark, features, preprocess, metrics, report, pipeline, train
script/          run_data / run_feature / run_preprocess / export_universe
docs/            dataset.md, fields.md, metrics.md, data_qa.md
notebook/        demo.ipynb
main.py
data/            local Parquet (+ benchmark caches); stock_universe.csv tracked
output/          run artifacts (gitignored)
```

## License and attribution

Data remain under their provider terms and are not included here. Cite So and Wang (2014) and Aboody et al. (2010) if you rely on the window design motivated by those papers.
