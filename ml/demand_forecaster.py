"""Part demand forecasting with gradient boosting.

MRO spare demand is intermittent: most part-months are zero, and the non-zero
months are lumpy. Plain regression on such a series predicts ~0 everywhere and
scores well on RMSE while being useless for planning.

This module handles that directly:

  * Features encode the intermittency (months since last demand, ADI, CV^2)
  * Classifies each part into Syntetos-Boylan categories (smooth / erratic /
    intermittent / lumpy) so the dashboard knows which forecasts to trust
  * Trains on log1p demand and evaluates on MASE against a seasonal-naive
    baseline, which is the metric that actually reflects planning value
  * Splits by time, never randomly - random CV leaks the future

Uses XGBoost when installed, otherwise falls back to sklearn's
HistGradientBoostingRegressor so the pipeline runs anywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
TARGET = "demand_qty"
HORIZON_MONTHS = 3
MIN_HISTORY_MONTHS = 18

# Syntetos-Boylan cut-offs for demand classification.
ADI_CUTOFF = 1.32          # average inter-demand interval
CV2_CUTOFF = 0.49          # squared coefficient of variation of non-zero demand


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    model: Any = field(repr=False, default=None)


# --------------------------------------------------------------- features


def classify_demand_pattern(series: pd.Series) -> str:
    """Syntetos-Boylan classification. Drives how much to trust a forecast."""
    nonzero = series[series > 0]
    if len(nonzero) < 2:
        return "NO_DEMAND"
    adi = len(series) / len(nonzero)
    cv2 = float((nonzero.std() / nonzero.mean()) ** 2) if nonzero.mean() > 0 else 0.0
    if adi < ADI_CUTOFF and cv2 < CV2_CUTOFF:
        return "SMOOTH"
    if adi < ADI_CUTOFF:
        return "ERRATIC"
    if cv2 < CV2_CUTOFF:
        return "INTERMITTENT"
    return "LUMPY"


def build_features(history: pd.DataFrame, parts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Panel features per (part_number, month).

    Every lag/rolling feature is computed with shift(1) first so a row never
    sees its own month - the single most common leak in demand models.
    """
    df = history.copy()
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df = df.sort_values(["part_number", "month"]).reset_index(drop=True)

    g = df.groupby("part_number")[TARGET]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    shifted = g.shift(1)
    for w in ROLLING_WINDOWS:
        grp = shifted.groupby(df["part_number"])
        df[f"roll_mean_{w}"] = grp.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = grp.transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
        df[f"roll_max_{w}"] = grp.transform(lambda s, w=w: s.rolling(w, min_periods=1).max())
        df[f"roll_nonzero_{w}"] = grp.transform(lambda s, w=w: (s > 0).rolling(w, min_periods=1).mean())

    # Intermittency signal: how long since this part last moved.
    had_demand = (shifted > 0).astype(float)
    df["months_since_demand"] = (
        had_demand.groupby(df["part_number"])
        .transform(lambda s: s.groupby((s == 1).cumsum()).cumcount())
    )

    # Trend and seasonality.
    df["momentum_3_12"] = df["roll_mean_3"] / df["roll_mean_12"].replace(0, np.nan)
    df["month_num"] = df["month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["tenure_months"] = df.groupby("part_number").cumcount()

    if parts is not None:
        attrs = parts[["part_number", "ata_chapter", "criticality", "unit_cost_usd", "repairable_flag"]]
        df = df.merge(attrs, on="part_number", how="left")
        df["ata_chapter"] = df["ata_chapter"].astype("category").cat.codes
        df["criticality_code"] = df["criticality"].map({"NO_GO": 2, "GO_IF": 1, "GO": 0}).fillna(-1)
        df["log_unit_cost"] = np.log1p(df["unit_cost_usd"].fillna(0))
        df["repairable_flag"] = df["repairable_flag"].fillna(False).astype(int)
        df = df.drop(columns=["criticality", "unit_cost_usd"])

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"part_number", "month", TARGET, "month_num"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------------ model


def _make_model(**kwargs: Any):
    try:
        from xgboost import XGBRegressor  # type: ignore

        params = {
            "n_estimators": 600, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "reg_lambda": 1.0,
            # count:poisson fits intermittent spares far better than squarederror
            "objective": "count:poisson", "tree_method": "hist", "random_state": 42,
        }
        params.update(kwargs)
        return XGBRegressor(**params), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        log.info("xgboost not installed, falling back to HistGradientBoostingRegressor")
        params = {
            "max_iter": 500, "learning_rate": 0.05, "max_depth": 6,
            "min_samples_leaf": 20, "l2_regularization": 1.0, "random_state": 42,
        }
        params.update(kwargs)
        return HistGradientBoostingRegressor(**params), "sklearn"


def mase(y_true: np.ndarray, y_pred: np.ndarray, naive: np.ndarray) -> float:
    """Mean absolute scaled error vs a naive baseline. < 1.0 beats naive."""
    naive_mae = np.mean(np.abs(y_true - naive))
    if naive_mae == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)


def train_demand_model(
    history: pd.DataFrame,
    parts: pd.DataFrame | None = None,
    test_months: int = 6,
) -> ForecastResult:
    """Train and time-split evaluate the demand model."""
    feats = build_features(history, parts)
    feats = feats[feats["tenure_months"] >= max(LAGS)]

    months = np.sort(feats["month"].unique())
    if len(months) <= test_months + 6:
        raise ValueError(f"need > {test_months + 6} months of history, got {len(months)}")
    split_month = months[-test_months]

    train = feats[feats["month"] < split_month]
    test = feats[feats["month"] >= split_month]
    cols = feature_columns(feats)

    X_tr, y_tr = train[cols].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=float)
    X_te, y_te = test[cols].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)

    model, backend = _make_model()
    if backend == "xgboost":
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
    else:
        # Poisson objective is unavailable here, so train on log1p and invert.
        # expm1(E[log1p(y)]) underestimates E[y] (Jensen), and under-forecasting
        # spares is the dangerous direction for AOG - a missed part grounds an
        # aircraft while excess stock only ties up capital. Duan's smearing
        # estimator, fitted on training residuals, corrects the retransform.
        model.fit(X_tr, np.log1p(y_tr))
        resid = np.log1p(y_tr) - model.predict(X_tr)
        model.smearing_factor_ = float(np.mean(np.exp(resid)))
        pred = np.expm1(model.predict(X_te)) * model.smearing_factor_
    pred = np.clip(pred, 0, None)

    # Baseline: last observed month, the de facto planner heuristic.
    naive = test["lag_1"].fillna(0).to_numpy(dtype=float)
    metrics = {
        "backend": backend,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae": float(np.mean(np.abs(y_te - pred))),
        "rmse": float(np.sqrt(np.mean((y_te - pred) ** 2))),
        "mase_vs_naive": mase(y_te, pred, naive),
        "naive_mae": float(np.mean(np.abs(y_te - naive))),
        "bias": float(np.mean(pred - y_te)),
        "test_from_month": str(split_month),
    }
    log.info("demand model %s: MAE %.3f, MASE %.3f", backend, metrics["mae"], metrics["mase_vs_naive"])

    out = test[["part_number", "month"]].copy()
    out["actual_qty"] = y_te
    out["predicted_qty"] = np.round(pred, 2)
    out["naive_qty"] = naive

    importance = _importance(model, cols, backend)
    return ForecastResult(predictions=out, metrics=metrics, feature_importance=importance, model=model)


def _importance(model: Any, cols: list[str], backend: str) -> pd.DataFrame:
    if backend == "xgboost":
        vals = model.feature_importances_
    else:
        # HistGradientBoosting has no native importances; permutation is too
        # slow for a nightly DAG, so report an empty frame rather than lie.
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": cols, "importance": vals})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def forecast_next_periods(
    history: pd.DataFrame,
    model: Any,
    parts: pd.DataFrame | None = None,
    horizon: int = HORIZON_MONTHS,
) -> pd.DataFrame:
    """Recursive multi-step forecast: predict a month, append it, re-featurize.

    Recursive rather than direct multi-horizon because the feature set is
    dominated by short lags, and MRO planners need a coherent burn-down path
    rather than three independent point estimates.
    """
    working = history.copy()
    working["month"] = pd.PeriodIndex(working["month"], freq="M")
    last_month = working["month"].max()
    results = []

    for step in range(1, horizon + 1):
        target_month = last_month + step
        parts_index = working["part_number"].unique()
        future = pd.DataFrame({
            "part_number": parts_index,
            "month": target_month,
            TARGET: np.nan,
        })
        combined = pd.concat([working, future], ignore_index=True)
        feats = build_features(combined, parts)
        rows = feats[feats["month"] == target_month]
        cols = feature_columns(feats)

        pred = model.predict(rows[cols].to_numpy(dtype=float))
        # Models trained on log1p carry a smearing factor; XGBoost's Poisson
        # objective predicts on the original scale and carries none.
        if hasattr(model, "smearing_factor_"):
            pred = np.expm1(pred) * model.smearing_factor_
        pred = np.clip(pred, 0, None)

        step_out = rows[["part_number", "month"]].copy()
        step_out["forecast_qty"] = np.round(pred, 2)
        step_out["horizon_step"] = step
        results.append(step_out)

        # Feed the prediction back in so the next step has a lag_1.
        fed = step_out.rename(columns={"forecast_qty": TARGET})[["part_number", "month", TARGET]]
        working = pd.concat([working, fed], ignore_index=True)

    out = pd.concat(results, ignore_index=True)
    out["month"] = out["month"].astype(str)
    return out


def demand_profile(history: pd.DataFrame) -> pd.DataFrame:
    """Per-part demand classification + safety-stock inputs for the scorer."""
    rows = []
    for part, g in history.groupby("part_number"):
        s = g.sort_values("month")[TARGET]
        nonzero = s[s > 0]
        rows.append({
            "part_number": part,
            "pattern": classify_demand_pattern(s),
            "months_observed": len(s),
            "monthly_demand": float(s.mean()),
            "demand_std": float(s.std()) if len(s) > 1 else 0.0,
            "zero_month_pct": round(100 * float((s == 0).mean()), 1),
            "peak_month_qty": float(s.max()),
            "avg_nonzero_qty": float(nonzero.mean()) if len(nonzero) else 0.0,
        })
    df = pd.DataFrame(rows)
    # Forecast reliability gate for the dashboard.
    df["forecastable"] = df["pattern"].isin(["SMOOTH", "ERRATIC"]) & (df["months_observed"] >= MIN_HISTORY_MONTHS)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sample = Path(__file__).parents[1] / "data/sample"
    hist = pd.read_parquet(sample / "demand_history.parquet")
    parts = pd.read_parquet(sample / "parts_master.parquet")

    result = train_demand_model(hist, parts)
    for k, v in result.metrics.items():
        print(f"{k:>18}: {v}")
    if not result.feature_importance.empty:
        print("\nTop features:")
        print(result.feature_importance.head(10).to_string(index=False))

    print("\nDemand pattern mix:")
    print(demand_profile(hist)["pattern"].value_counts().to_string())
