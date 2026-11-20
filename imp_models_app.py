"""IMP model explorer: every trained model side by side, by sector and by stock.

Reads the artifacts written by ``main.py`` / ``script/run_reversal.py``:

    output/lin|tr|nn/{metrics.csv, predictions.parquet, run_config.json}
    output/reversal/{events.parquet, metrics.csv}

Run from the repo root:

    python script/run_reversal.py --with-lin
    python -c "from src.pipeline import run_family_by_name; run_family_by_name(('tr','nn'))"
    streamlit run imp_models_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config import default as cfg
from src.sectors import attach_sectors

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FAMILIES = ("lin", "tr", "nn")
FOLDS = ("train", "val", "test")

FOLD_LABEL = {
    "train": f"Train (≤ {cfg.TRAIN_END})",
    "val": f"Validation (> {cfg.TRAIN_END}, ≤ {cfg.VAL_END})",
    "test": f"Test (> {cfg.VAL_END})",
}

FAMILY_LABEL = {"lin": "Linear", "tr": "Tree", "nn": "Neural net", "rev": "Reversal"}

MODEL_INFO = {
    "ols": ("Linear", "Ordinary least squares",
            "Straight-line fit of the target on all 8 features. No penalty, so every "
            "feature gets a coefficient. Baseline for everything else."),
    "ridge": ("Linear", "Ridge regression",
              "OLS with an L2 penalty that shrinks coefficients toward zero. Penalty "
              "strength `alpha` chosen on the validation year."),
    "lasso": ("Linear", "Lasso regression",
              "OLS with an L1 penalty that can set coefficients exactly to zero. If "
              "every coefficient is zeroed the forecast is a constant (the training "
              "mean) and the model cannot rank events."),
    "decision_tree": ("Tree", "Decision tree",
                      "One tree of yes/no splits on the features. `max_depth` chosen on "
                      "validation; leaves need ≥ 20 events."),
    "random_forest": ("Tree", "Random forest",
                      "Average of 200 trees, each grown on a bootstrap sample with random "
                      "feature subsets. Less noisy than one tree."),
    "hist_gradient_boosting": ("Tree", "Histogram gradient boosting",
                               "Trees added one at a time, each fixing the previous "
                               "errors (learning rate 0.05). Stops early when validation "
                               "R² stops improving."),
    "mlp_shallow": ("Neural net", "MLP, 1 hidden layer (64)",
                    "Small neural network. Features standardized on train. L2 `alpha` "
                    "chosen on validation, early stopping on validation R²."),
    "mlp": ("Neural net", "MLP, 2 hidden layers (64, 32)",
            "Same as above with two layers."),
    "mlp_deep": ("Neural net", "MLP, 3 hidden layers (128, 64, 32)",
                 "Same as above with three layers."),
    "reversal_ols": ("Reversal", "PAR reversal (OLS on PAR)",
                     "Your model on IMP's clock: target regressed on one feature, the "
                     "market-adjusted pre-announcement return from close anchor−5 to "
                     "close anchor−2. A negative slope is a reversal: forecasts a bounce "
                     "after a drop and a give-back after a run-up."),
}

SHORT = {
    "ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "decision_tree": "Tree",
    "random_forest": "Forest", "hist_gradient_boosting": "Boosting",
    "mlp_shallow": "MLP-1", "mlp": "MLP-2", "mlp_deep": "MLP-3", "reversal_ols": "PAR",
}

FEATURE_INFO = [
    ("QScoreTrendLong", "Stock-data panel",
     "Long-horizon trend score (technical), −5 to +5. Higher = stronger uptrend."),
    ("QScoreSpeedLong", "Stock-data panel",
     "Long-horizon speed / momentum score, −5 to +5. Higher = accelerating."),
    ("QScoreLong", "Stock-data panel",
     "Trend + Speed (−10 to +10). Exactly the sum of the two columns above, so it adds "
     "no new information and makes OLS coefficients unstable."),
    ("analystRatingsStrongBuy", "Monthly analyst snapshot",
     "Number of Strong Buy ratings in the latest monthly snapshot ≤ 31 days old."),
    ("analystRatingsBuy", "Monthly analyst snapshot", "Number of Buy ratings."),
    ("analystRatingsHold", "Monthly analyst snapshot", "Number of Hold ratings."),
    ("analystRatingsSell", "Monthly analyst snapshot", "Number of Sell ratings."),
    ("analystRatingsStrongSell", "Monthly analyst snapshot", "Number of Strong Sell ratings."),
    ("par_mkt", "Computed from prices (reversal only)",
     "Pre-announcement return: stock minus benchmark, close anchor−5 → close anchor−2. "
     "Used only by the PAR reversal model, not by the 9 IMP models."),
]

METRIC_HELP = {
    "OOS R²": "1 − error / error of always predicting the training average. "
              "Positive = beats that constant guess. Above ~0.01 would be notable here.",
    "IC": "Pearson correlation of forecast with actual return.",
    "Rank IC": "Spearman (rank) correlation. Less sensitive to a few huge moves.",
    "Hit rate": "Share of events where the model's call (long if its forecast is above "
                "its own median for the period, short if below) had the right sign.",
    "Long–short": "Average return of taking that call on every event: long the top half "
                  "of forecasts, short the bottom half. Before costs.",
    "Top–bottom quintile": "Average actual return of the top 20% of forecasts minus "
                           "the bottom 20%.",
}


# ------------------------------------------------------------------ data
@st.cache_data
def load_events() -> pd.DataFrame:
    events = pd.read_parquet(OUT / "reversal" / "events.parquet")
    events["EventDate"] = pd.to_datetime(events["EventDate"])
    return attach_sectors(events)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    frames = []
    for family in FAMILIES:
        path = OUT / family / "predictions.parquet"
        if path.exists():
            part = pd.read_parquet(path)
            part["family"] = family
            frames.append(part[["TickerId", "EventDate", "model", "fold", "y_true", "y_pred", "family"]])
    events = load_events()
    rev_path = OUT / "reversal" / "metrics.csv"
    if rev_path.exists():
        rev = pd.read_csv(rev_path).set_index("model").loc["reversal_ols"]
        usable = events.dropna(subset=["par_mkt", cfg.TARGET])
        frames.append(pd.DataFrame({
            "TickerId": usable["TickerId"], "EventDate": usable["EventDate"],
            "model": "reversal_ols", "fold": usable["fold"],
            "y_true": usable[cfg.TARGET],
            "y_pred": rev["intercept"] + rev["slope"] * usable["par_mkt"],
            "family": "rev",
        }))
    preds = pd.concat(frames, ignore_index=True)
    preds["EventDate"] = pd.to_datetime(preds["EventDate"])
    keep = ["TickerId", "EventDate", "Symbol", "Company", "Sector", "Industry",
            "AnchorDate", "DecisionDate", "MarketTimeId", "par_mkt"]
    preds = preds.merge(events[keep].drop_duplicates(["TickerId", "EventDate"]),
                        on=["TickerId", "EventDate"], how="left")
    # The model's call: long above its own median forecast for that fold, short below.
    median = preds.groupby(["model", "fold"])["y_pred"].transform("median")
    preds["call"] = np.sign(preds["y_pred"] - median).astype(int)
    preds["signed"] = preds["call"] * preds["y_true"]
    preds["label"] = preds["model"].map(lambda m: MODEL_INFO.get(m, ("", m, ""))[1])
    return preds


@st.cache_data
def load_configs() -> dict:
    configs = {}
    for family in FAMILIES:
        path = OUT / family / "run_config.json"
        if path.exists():
            configs[family] = json.loads(path.read_text())
    return configs


@st.cache_data
def load_saved_metrics() -> pd.DataFrame:
    frames = []
    for family in FAMILIES:
        path = OUT / family / "metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["family"] = family
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------ scoring
def score(part: pd.DataFrame, train_mean: float, quintiles: bool = True) -> dict:
    y = part["y_true"].to_numpy(float)
    p = part["y_pred"].to_numpy(float)
    n = len(y)
    out = {"Events": n}
    if n == 0:
        return out
    sse = np.sum((y - p) ** 2)
    sse0 = np.sum((y - train_mean) ** 2)
    out["OOS R²"] = 1 - sse / sse0 if sse0 > 0 else np.nan
    varies = np.nanstd(p) > 1e-12 and n > 2
    out["IC"] = float(np.corrcoef(p, y)[0, 1]) if varies else np.nan
    out["Rank IC"] = float(pd.Series(p).rank().corr(pd.Series(y).rank())) if varies else np.nan
    called = part.loc[part["call"] != 0]
    out["Hit rate"] = float((called["signed"] > 0).mean()) if len(called) else np.nan
    out["Long–short"] = float(called["signed"].mean()) if len(called) else np.nan
    if quintiles and varies and n >= 25:
        q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
        out["Top–bottom quintile"] = float(y[q.to_numpy() == 4].mean() - y[q.to_numpy() == 0].mean())
    return out


def train_means(preds: pd.DataFrame) -> dict:
    return preds.loc[preds["fold"] == "train"].groupby("model")["y_true"].mean().to_dict()


def leaderboard(preds: pd.DataFrame, fold: str) -> pd.DataFrame:
    means = train_means(preds)
    rows = []
    for model, part in preds.loc[preds["fold"] == fold].groupby("model"):
        info = MODEL_INFO.get(model, ("", model, ""))
        rows.append({"Family": info[0], "Model": info[1], "model": model,
                     **score(part, means.get(model, 0.0))})
    return pd.DataFrame(rows)


def group_scores(preds: pd.DataFrame, fold: str, by: str, models: list[str]) -> pd.DataFrame:
    means = train_means(preds)
    sub = preds.loc[(preds["fold"] == fold) & preds["model"].isin(models)]
    rows = []
    for (group, model), part in sub.groupby([by, "model"]):
        rows.append({by: group, "model": model,
                     "Model": MODEL_INFO.get(model, ("", model, ""))[1],
                     "Stocks": part["TickerId"].nunique(),
                     **score(part, means.get(model, 0.0), quintiles=False)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ formatting
PCT_COLS = ("Hit rate", "Long–short", "Top–bottom quintile")


def show_table(frame: pd.DataFrame, hide: tuple[str, ...] = ("model",)) -> None:
    config = {}
    for col in frame.columns:
        if col in PCT_COLS:
            config[col] = st.column_config.NumberColumn(col, format="percent", help=METRIC_HELP.get(col))
        elif col in ("OOS R²", "IC", "Rank IC"):
            config[col] = st.column_config.NumberColumn(col, format="%.4f", help=METRIC_HELP.get(col))
    st.dataframe(frame.drop(columns=[c for c in hide if c in frame.columns]),
                 hide_index=True, column_config=config)


def metric_bar(board: pd.DataFrame, metric: str) -> alt.Chart:
    data = board.dropna(subset=[metric])
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric}:Q", title=metric,
                    axis=alt.Axis(format="%" if metric in PCT_COLS else ".3f")),
            y=alt.Y("Model:N", sort="-x", title=None),
            color=alt.Color("Family:N", legend=alt.Legend(orient="bottom")),
            tooltip=["Model", "Family", alt.Tooltip(f"{metric}:Q", format=".4f"), "Events"],
        )
        .properties(height=34 * len(data) + 20)
    )


# ------------------------------------------------------------------ pages
def page_overview(preds: pd.DataFrame, fold: str) -> None:
    st.subheader(f"All models on {FOLD_LABEL[fold]}")
    st.write(
        "Every model forecasts the same thing for the same events: the stock's "
        f"**market-adjusted return over the announcement window** (`{cfg.TARGET}`, close "
        "anchor−2 → close anchor+1), using only information known at the close of "
        "anchor−2. Parameters are fitted on train, settings picked on validation, and "
        "**test is the only honest out-of-sample score**."
    )
    board = leaderboard(preds, fold)
    best = board.dropna(subset=["OOS R²"]).sort_values("OOS R²", ascending=False).head(1)
    beats = int((board["OOS R²"] > 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", len(board))
    c2.metric("Events scored", int(board["Events"].max()))
    c3.metric("Beat the constant guess", f"{beats} / {len(board)}")
    if len(best):
        c4.metric("Best OOS R²", f"{best['OOS R²'].iloc[0]:.4f}", best["Model"].iloc[0], delta_color="off")

    show_table(board.sort_values("OOS R²", ascending=False))
    with st.expander("What each score means"):
        for name, text in METRIC_HELP.items():
            st.markdown(f"- **{name}** — {text}")

    metric = st.radio("Chart", list(METRIC_HELP), horizontal=True, index=0)
    st.altair_chart(metric_bar(board, metric), width="stretch")

    st.markdown("#### Does a model hold up across periods?")
    st.caption("Same score on train, validation and test. A model that is good on train and bad "
               "on test is overfitting.")
    metric2 = st.selectbox("Score", ["OOS R²", "Rank IC", "Hit rate", "Long–short"], key="m2")
    rows = [leaderboard(preds, f).assign(Period=f) for f in FOLDS]
    allp = pd.concat(rows)
    chart = (
        alt.Chart(allp.dropna(subset=[metric2]))
        .mark_bar()
        .encode(
            x=alt.X(f"{metric2}:Q", axis=alt.Axis(format="%" if metric2 in PCT_COLS else ".3f")),
            y=alt.Y("Period:N", sort=list(FOLDS), title=None),
            color=alt.Color("Period:N", sort=list(FOLDS), legend=None),
            row=alt.Row("Model:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=["Model", "Period", alt.Tooltip(f"{metric2}:Q", format=".4f")],
        )
        .properties(height=60, width=420)
    )
    st.altair_chart(chart)


def page_sectors(preds: pd.DataFrame, fold: str) -> None:
    st.subheader(f"By sector and industry — {FOLD_LABEL[fold]}")
    st.write(
        "Each model's call is fixed for the whole period (long above its own median "
        "forecast, short below). Here those calls are scored inside each group, so a "
        "sector number says how well the model ranked *that sector's* events."
    )
    models = sorted(preds["model"].unique(), key=list(MODEL_INFO).index)
    chosen = st.multiselect("Models", models, default=[m for m in models if m != "lasso"],
                            format_func=lambda m: MODEL_INFO[m][1])
    if not chosen:
        st.info("Pick at least one model.")
        return
    level = st.radio("Group by", ["Sector", "Industry"], horizontal=True)
    metric = st.radio("Score", ["Long–short", "Hit rate", "Rank IC", "OOS R²"], horizontal=True)
    scores = group_scores(preds, fold, level, chosen)
    min_events = st.slider("Hide groups with fewer events than", 0, 60, 20 if level == "Sector" else 12)
    scores = scores.loc[scores["Events"] >= min_events]
    scores["Short"] = scores["model"].map(SHORT)
    if scores.empty:
        st.info("No group has that many events.")
        return

    domain_mid = 0.5 if metric == "Hit rate" else 0.0
    heat = (
        alt.Chart(scores)
        .mark_rect()
        .encode(
            x=alt.X("Short:N", sort=[SHORT[m] for m in chosen], title=None,
                    axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{level}:N", title=None),
            color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="redblue", domainMid=domain_mid),
                            legend=alt.Legend(format="%" if metric in PCT_COLS else ".2f")),
            tooltip=[level, "Model", "Events", "Stocks",
                     alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .properties(height=max(220, 26 * scores[level].nunique()))
    )
    text = heat.transform_filter(f"isValid(datum['{metric}'])").mark_text(fontSize=10).encode(
        text=alt.Text(f"{metric}:Q", format=".1%" if metric in PCT_COLS else ".2f"),
        color=alt.value("black"),
    )
    st.altair_chart(heat + text, width="stretch")
    st.caption("Blue = model did well in that group, red = badly. "
               f"Hit rate is centred at 50%; the others at 0. Hover for event counts.")

    winners = (scores.sort_values(metric, ascending=False).groupby(level).head(1)
               [[level, "Model", metric, "Events"]].sort_values(metric, ascending=False))
    st.markdown(f"#### Best model in each {level.lower()} by {metric}")
    show_table(winners)
    st.warning(
        f"Small samples: a sector in one year has roughly 25–140 events. A long–short "
        "gap of a few percent between two models in one sector is usually within noise. "
        "Look for a model that leads in the same group in **both** validation and test.",
    )
    with st.expander("Full table"):
        show_table(scores.sort_values([level, metric], ascending=[True, False]))


def page_stocks(preds: pd.DataFrame, fold: str) -> None:
    st.subheader(f"By stock — {FOLD_LABEL[fold]}")
    sub = preds.loc[preds["fold"] == fold]
    st.write(
        f"About {sub.groupby('TickerId').size().median():.0f} announcements per stock in "
        "this period, so per-stock results are mostly luck. Use this page to inspect "
        "individual events, not to rank models."
    )
    sectors = ["All"] + sorted(sub["Sector"].dropna().unique())
    sector = st.selectbox("Sector", sectors)
    names = sub if sector == "All" else sub.loc[sub["Sector"] == sector]
    options = (names[["Symbol", "Company"]].drop_duplicates().sort_values("Symbol"))
    labels = (options["Symbol"] + " — " + options["Company"].astype(str)).tolist()
    choice = st.selectbox("Stock", labels)
    symbol = choice.split(" — ")[0]
    stock = sub.loc[sub["Symbol"] == symbol]

    wide = stock.pivot_table(index=["EventDate", "y_true"], columns="label", values="y_pred").reset_index()
    wide = wide.rename(columns={"EventDate": "Earnings date", "y_true": "Actual"})
    st.markdown(f"#### {symbol}: forecast vs actual market-adjusted announcement return")
    config = {c: st.column_config.NumberColumn(c, format="percent") for c in wide.columns if c != "Earnings date"}
    config["Earnings date"] = st.column_config.DateColumn("Earnings date")
    st.dataframe(wide, hide_index=True, column_config=config)

    tally = (stock.loc[stock["call"] != 0].groupby("label")
             .agg(Events=("signed", "size"), Right=("signed", lambda s: int((s > 0).sum())),
                  **{"Avg signed return": ("signed", "mean")})
             .reset_index().rename(columns={"label": "Model"})
             .sort_values("Avg signed return", ascending=False))
    st.markdown("#### Calls on this stock")
    st.dataframe(tally, hide_index=True, column_config={
        "Avg signed return": st.column_config.NumberColumn(format="percent")})

    with st.expander("All stocks: which model had the best average call"):
        per = (sub.loc[sub["call"] != 0].groupby(["Symbol", "Sector", "label"])["signed"].mean()
               .reset_index())
        best = per.sort_values("signed", ascending=False).groupby("Symbol").head(1)
        counts = best["label"].value_counts().rename_axis("Model").reset_index(name="Stocks led")
        st.dataframe(counts, hide_index=True)
        st.caption("With ~4 events per stock, these counts are close to what random "
                   "forecasts would produce.")


def page_models(preds: pd.DataFrame) -> None:
    st.subheader("Models")
    saved = load_saved_metrics()
    configs = load_configs()
    rows = []
    for model, (family, name, text) in MODEL_INFO.items():
        chosen = ""
        if not saved.empty and model in set(saved["model"]):
            row = saved.loc[saved["model"] == model].iloc[0]
            bits = []
            for col in ("alpha", "max_depth", "hidden", "best_iteration"):
                if col in row and pd.notna(row[col]):
                    bits.append(f"{col}={row[col]}")
            chosen = ", ".join(bits)
        rows.append({"Family": family, "Model": name, "What it does": text,
                     "Chosen on validation": chosen})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "What it does": st.column_config.TextColumn(width="large")})

    st.markdown("#### How settings are chosen")
    st.markdown(
        "- Fit on **train** only. Missing features filled with the **train** median; "
        "linear and neural models also standardize features with **train** mean/std.\n"
        "- Try each setting in the grid, keep the one with the best **validation** OOS R² "
        "(tie-break: lower RMSE). Trees and MLPs also stop early on validation.\n"
        "- Report **test** once. Nothing on test influences any choice."
    )
    grids = {
        "Linear": "ridge / lasso `alpha` from `config/config_lin.py`",
        "Tree": f"`max_depth` ∈ {configs.get('tr', {}).get('MAX_DEPTH_GRID', '?')}, "
                f"{configs.get('tr', {}).get('N_ESTIMATORS', '?')} trees, min leaf "
                f"{configs.get('tr', {}).get('MIN_SAMPLES_LEAF', '?')}",
        "Neural net": f"L2 `alpha` ∈ {configs.get('nn', {}).get('ALPHA_GRID', '?')}, "
                      f"batch {configs.get('nn', {}).get('BATCH_SIZE', '?')}, "
                      f"max {configs.get('nn', {}).get('MAX_ITER', '?')} epochs",
        "Reversal": "No tuning: one OLS slope on PAR, fitted on train.",
    }
    for family, text in grids.items():
        st.markdown(f"- **{family}:** {text}")
    with st.expander("Raw run configs"):
        st.json(configs)


def page_features() -> None:
    st.subheader("Features (what the models see)")
    st.write("All features are snapshotted at the **decision date = anchor − 2 sessions**, "
             "the close the announcement window starts from. Nothing after that is used.")
    st.dataframe(pd.DataFrame(FEATURE_INFO, columns=["Feature", "Source", "Meaning"]),
                 hide_index=True, column_config={"Meaning": st.column_config.TextColumn(width="large")})

    events = load_events()
    cols = [f for f, *_ in FEATURE_INFO]
    st.markdown("#### Coverage and correlation with the target")
    fold = st.radio("Period", FOLDS, horizontal=True, format_func=FOLD_LABEL.get, key="ffold")
    part = events.loc[events["fold"] == fold]
    rows = []
    for col in cols:
        x = pd.to_numeric(part[col], errors="coerce")
        y = part[cfg.TARGET]
        ok = x.notna() & y.notna()
        rows.append({
            "Feature": col,
            "Missing": float(x.isna().mean()),
            "Median": float(x.median()),
            "Corr. with target": float(x[ok].corr(y[ok])) if ok.sum() > 2 else np.nan,
            "Rank corr.": float(x[ok].rank().corr(y[ok].rank())) if ok.sum() > 2 else np.nan,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Missing": st.column_config.NumberColumn(format="percent"),
        "Corr. with target": st.column_config.NumberColumn(format="%.3f"),
        "Rank corr.": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption("A single feature's correlation is the simplest test of whether it carries "
               "any signal. Values under ±0.05 are weak for ~800 events.")

    feature = st.selectbox("Look at one feature", cols)
    data = part[[feature, cfg.TARGET]].dropna().rename(columns={cfg.TARGET: "Announcement return"})
    scatter = (
        alt.Chart(data).mark_circle(opacity=0.35, size=18)
        .encode(x=alt.X(f"{feature}:Q"),
                y=alt.Y("Announcement return:Q", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[-0.3, 0.3], clamp=True)))
    )
    st.altair_chart(scatter + scatter.transform_regression(feature, "Announcement return")
                    .mark_line(color="firebrick"), width="stretch")


def page_periods() -> None:
    st.subheader("Periods and windows")
    events = load_events()
    configs = load_configs()
    fold_info = next(iter(configs.values()), {}).get("folds", {})

    st.markdown("#### 1. Calendar split (which years train / tune / test)")
    rows = []
    for fold in FOLDS:
        info = fold_info.get(fold, {})
        rows.append({"Period": FOLD_LABEL[fold], "Used for": {
            "train": "Fit model parameters", "val": "Pick settings, early stopping",
            "test": "Final score only"}[fold],
            "Events": info.get("rows"), "First decision date": info.get("date_min"),
            "Last decision date": info.get("date_max"),
            "Avg target": info.get("y_mean"), "Std target": info.get("y_std")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Avg target": st.column_config.NumberColumn(format="percent"),
        "Std target": st.column_config.NumberColumn(format="percent")})
    st.caption(f"Split on `{cfg.SPLIT_COLUMN}`. Data covers "
               f"{events['EventDate'].min():%Y-%m-%d} → {events['EventDate'].max():%Y-%m-%d}, "
               f"{events['TickerId'].nunique()} stocks.")
    monthly = (events.assign(Month=events["EventDate"].dt.to_period("M").dt.to_timestamp())
               .groupby(["Month", "fold"]).size().reset_index(name="Events"))
    st.altair_chart(
        alt.Chart(monthly).mark_bar().encode(
            x=alt.X("Month:T", title=None), y="Events:Q",
            color=alt.Color("fold:N", sort=list(FOLDS), title="Period"),
            tooltip=["Month:T", "fold", "Events"]).properties(height=200),
        width="stretch")

    st.markdown("#### 2. Event window (trading sessions around each announcement)")
    st.markdown(
        "**Anchor (day 0)** is the first session whose price can react to the news. "
        "Before-open release (`MarketTimeId = 1`): the release day. After-close release "
        "(`MarketTimeId = 2`): the next session. "
        f"In this data {int((events['MarketTimeId'] == 2).sum())} events are after-close "
        f"and {int((events['MarketTimeId'] == 1).sum())} before-open."
    )
    spans = pd.DataFrame([
        ("PAR (reversal input)", -5, -2, "Signal"),
        ("Decision: features snapshotted", -2, -2, "Signal"),
        ("Announcement return — the TARGET", -2, 1, "Outcome"),
        ("Post window (ret_post, not used)", 1, 5, "Other"),
    ], columns=["Window", "start", "end", "Kind"])
    base = alt.Chart(spans).encode(y=alt.Y("Window:N", sort=None, title=None,
                                           axis=alt.Axis(labelLimit=320)))
    bars = base.mark_bar(height=16).encode(
        x=alt.X("start:Q", title="Trading sessions relative to anchor (close-to-close)",
                scale=alt.Scale(domain=[-6, 6]), axis=alt.Axis(tickMinStep=1)),
        x2="end:Q", color=alt.Color("Kind:N", legend=None))
    dots = base.mark_point(filled=True, size=90).encode(x="end:Q", color=alt.Color("Kind:N", legend=None))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    st.altair_chart((bars + dots + zero).properties(height=170), width="stretch")
    st.caption("Bars run from the starting close to the ending close. The target therefore "
               "covers the returns of days −1, 0 and +1. Day −1 is before the news, so the "
               "target is not a pure reaction return.")
    st.markdown(
        f"- **Market adjustment:** stock return minus the `{cfg.BENCHMARK_SOURCE}` benchmark "
        f"({'equal-weighted index of the panel' if cfg.BENCHMARK_SOURCE == 'equal' else cfg.MARKET_TICKER}) "
        "over the same window.\n"
        "- **Your 70-stock app** puts day 0 on the calendar release date even for after-close "
        "releases and uses SPY, so its numbers are not directly comparable. The PAR model "
        "on this site is rebuilt on IMP's anchor and benchmark, so it is."
    )


def page_caveats() -> None:
    st.subheader("Things to keep in mind")
    st.markdown(
        "1. **One test year.** Test is 2024 only (~800 events). Differences of a few "
        "thousandths in R² or a couple of points of hit rate are noise. A walk-forward "
        "split (test 2022, 2023, 2024 in turn) would triple the out-of-sample events.\n"
        "2. **Train includes 2020–2021.** COVID-era moves are large and may dominate the fit.\n"
        "3. **Per-stock results ≈ 4 events each.** Compare sectors, not single names.\n"
        "4. **QScoreLong = Trend + Speed.** Perfectly collinear; OLS coefficients on those "
        "three are not interpretable.\n"
        "5. **Benchmark is built from the same ~200 stocks.** Re-running with "
        "`BENCHMARK_SOURCE = \"yahoo\"` (SPY) checks the results are not an artefact of it.\n"
        "6. **Hit rate and long–short here** use each model's call relative to its own "
        "median forecast, so a model that always forecasts a small positive number is not "
        "credited for the market's average drift. The R² / IC numbers match `metrics.csv`.\n"
        "7. **Lasso** shrinks to the training mean (constant forecast): no ranking, no calls."
    )


# ------------------------------------------------------------------ main
def main() -> None:
    st.set_page_config(page_title="IMP model explorer", layout="wide")
    st.title("IMP model explorer")
    st.caption("Identifying Market Positioning ahead of earnings — 9 IMP models plus the "
               "PAR reversal, scored on the same events.")
    missing = [f for f in FAMILIES if not (OUT / f / "predictions.parquet").exists()]
    if not (OUT / "reversal" / "events.parquet").exists():
        st.error("Run `python script/run_reversal.py --with-lin` first.")
        return
    if missing:
        st.warning(f"No results yet for: {', '.join(missing)}. Run those families in `main.py`.")

    preds = load_predictions()
    with st.sidebar:
        st.header("Settings")
        fold = st.radio("Period to score", FOLDS, index=2, format_func=FOLD_LABEL.get)
        st.markdown(f"**Target:** `{cfg.TARGET}`  \n**Benchmark:** `{cfg.BENCHMARK_SOURCE}`  \n"
                    f"**Split column:** `{cfg.SPLIT_COLUMN}`")
        if fold != "test":
            st.info("Train and validation were used to fit and tune. Only test is out-of-sample.")

    tabs = st.tabs(["Leaderboard", "By sector / industry", "By stock",
                    "Models", "Features", "Periods & windows", "Caveats"])
    with tabs[0]:
        page_overview(preds, fold)
    with tabs[1]:
        page_sectors(preds, fold)
    with tabs[2]:
        page_stocks(preds, fold)
    with tabs[3]:
        page_models(preds)
    with tabs[4]:
        page_features()
    with tabs[5]:
        page_periods()
    with tabs[6]:
        page_caveats()


main()
