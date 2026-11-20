# Metrics

Evaluation definitions used by lin / tr / nn (`src/metrics.py`).  
Data columns are in `dataset.md` / `fields.md`; this file covers **how predictions are scored**.

Regression only for now. Classification (`dir_ann`) is reserved.

Notation: \(y\) = true target, \(\hat{y}\) = model prediction, \(n\) = rows in the fold.

---

## Core scores

| Name       | Definition                                                              | Notes                                                 |
| ---------- | ----------------------------------------------------------------------- | ----------------------------------------------------- |
| `n`        | Number of evaluated rows                                                | Integer in displays                                   |
| `rmse`     | \(\sqrt{\mathrm{mean}((y-\hat{y})^2)}\)                                 | Lower is better                                       |
| `mae`      | \(\mathrm{mean}(\lvert y-\hat{y}\rvert)\)                               | Lower is better                                       |
| `r2`       | sklearn `r2_score` vs **this fold’s** mean of \(y\)                     | In-fold R²; not the primary OOS score                 |
| `oos_r2`   | \(1 - \mathrm{SSE}(\hat{y}) / \mathrm{SSE}(\bar{y}\_{\mathrm{train}})\) | Primary OOS fit vs train-mean baseline                |
| `ic`       | Pearson corr\((y,\hat{y})\)                                             | `n/a` if \(y\) or \(\hat{y}\) is constant             |
| `rank_ic`  | Spearman corr\((y,\hat{y})\)                                            | Same constant-prediction rule                         |
| `hit_rate` | \(\mathrm{mean}(\mathrm{sign}(y)=\mathrm{sign}(\hat{y}))\)              | Directional agreement; zeros in \(y\) count as misses |

\[
\mathrm{SSE}(a) = \sum_i (y_i - a_i)^2
\]

For `oos_r2`, the baseline \(a*i\) is the constant **training-fold** mean \(\bar{y}*{\mathrm{train}}\) (known at train time). Prefer **`oos_r2` over `r2`** when judging val/test.

---

## Baselines (shown next to RMSE / MAE)

| Name       | Prediction                               | Reported as                         |
| ---------- | ---------------------------------------- | ----------------------------------- |
| Train-mean | \(\hat{y}_i = \bar{y}_{\mathrm{train}}\) | `rmse_train_mean`, `mae_train_mean` |
| Zero       | \(\hat{y}\_i = 0\)                       | `rmse_zero`, `mae_zero`             |

Terminal display packs them as:

```text
rmse (mean/zero)  ->  model_rmse (rmse_train_mean / rmse_zero)
mae  (mean/zero)  ->  model_mae  (mae_train_mean / mae_zero)
r2   (oos)        ->  sklearn_r2 (oos_r2)
```

A useful model should beat both baselines on RMSE/MAE and show non-negative `oos_r2` when possible.

---

## Model selection

On the chronological **val** fold, tuneables (e.g. ridge `alpha`, tree `max_depth`, MLP L2) maximize **`oos_r2`**, tie-break **lower RMSE**.  
HistGBDT / MLP early stopping also monitor val `oos_r2`.

---

## FEATURE (permutation importance)

Optional; computed on the **test** fold after fit (config `FEATURE_IMPORTANCE`).

For each feature, shuffle that column `FEATURE_IMPORTANCE_REPEATS` times and measure metric change vs the unshuffled prediction:

| Column           | Meaning (positive ⇒ feature helps)    |
| ---------------- | ------------------------------------- |
| `delta_rmse`     | shuffled RMSE − baseline RMSE         |
| `delta_mae`      | shuffled MAE − baseline MAE           |
| `delta_ic`       | baseline IC − shuffled IC             |
| `delta_rank_ic`  | baseline rank IC − shuffled rank IC   |
| `delta_hit_rate` | baseline hit rate − shuffled hit rate |
| `delta_oos_r2`   | baseline OOS R² − shuffled OOS R²     |

Table is sorted by `delta_ic` descending. Importance is **model-conditional** (depends on the fitted model and feature set).

---

## QUINTILES

On val/test predictions, sort by \(\hat{y}\) into five equal-count buckets (`pd.qcut`):

| Column             | Meaning                                                  |
| ------------------ | -------------------------------------------------------- |
| `q1` … `q5`        | Mean realized \(y\) in lowest … highest predicted bucket |
| `top_minus_bottom` | `q5 − q1` (spread of realized target; higher is better)  |
| `n`                | Rows in that fold                                        |

Skipped when predictions are constant or have too few unique values.

The **test**-fold `top_minus_bottom` is also copied into **SUMMARY** / `metrics.csv`
(as `NaN` when quintiles were skipped for that model).

---

## Terminal blocks and artifacts

Per model run (typical order):

1. **ALPHA** / depth grid / **EARLY_STOP** (if applicable)
2. **METRICS** — train / val / test rows
3. **FEATURE** — permutation table (optional)
4. **QUINTILES** — val / test
5. **SUMMARY** — one test-fold row per model (fit metrics + `top_minus_bottom`)

Written under `output/{lin,tr,nn}/` (gitignored):

| File                  | Contents                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| `metrics.csv`         | SUMMARY rows (numeric), including test `top_minus_bottom` when available |
| `predictions.parquet` | Per-event `y_true`, `y_pred`, `residual`, `fold`, `model`, dates         |
| `run_config.json`     | Resolved settings and fold date ranges                                   |

Implementation: `src/metrics.py`.
