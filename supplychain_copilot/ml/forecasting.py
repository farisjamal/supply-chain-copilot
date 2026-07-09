"""Demand forecasting with scikit-learn.

Approach: supervised regression on engineered time-series features
(lags, rolling mean, calendar features). A RandomForest learns the
weekly seasonality and trend per SKU. Forecasts are produced
recursively: each predicted day is fed back in as the next day's lag.

Evaluation: the last HOLDOUT_DAYS of history are held out and MAE is
reported per SKU, so the model is judged on data it never saw.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

HOLDOUT_DAYS = 56
LAGS = [1, 7, 14]
ROLLING_WINDOW = 7


def build_features(series: pd.DataFrame) -> pd.DataFrame:
    """series: DataFrame with columns [date, units_sold] for a single SKU."""
    df = series.sort_values("date").copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)
    df[f"rolling_{ROLLING_WINDOW}"] = (
        df["units_sold"].shift(1).rolling(ROLLING_WINDOW).mean()
    )
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df.dropna().reset_index(drop=True)


FEATURE_COLS = [f"lag_{lag}" for lag in LAGS] + [f"rolling_{ROLLING_WINDOW}", "dayofweek", "month"]


@dataclass
class ForecastModel:
    """A trained per-SKU forecaster plus its holdout evaluation metrics."""

    models: dict = field(default_factory=dict)   # sku -> fitted regressor
    metrics: dict = field(default_factory=dict)  # sku -> {"mae": float, "mean_demand": float}


def train(orders: pd.DataFrame) -> ForecastModel:
    """Train one RandomForest per SKU; report MAE on a time-based holdout."""
    result = ForecastModel()
    for sku, grp in orders.groupby("sku"):
        feats = build_features(grp[["date", "units_sold"]])
        train_df = feats.iloc[:-HOLDOUT_DAYS]
        test_df = feats.iloc[-HOLDOUT_DAYS:]

        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0)
        model.fit(train_df[FEATURE_COLS], train_df["units_sold"])

        preds = model.predict(test_df[FEATURE_COLS])
        result.models[sku] = model
        result.metrics[sku] = {
            "mae": float(mean_absolute_error(test_df["units_sold"], preds)),
            "mean_demand": float(grp["units_sold"].mean()),
        }
    return result


def forecast(model: ForecastModel, orders: pd.DataFrame, sku: str, horizon_days: int = 14) -> pd.DataFrame:
    """Recursive multi-step forecast: predictions become future lag inputs."""
    if sku not in model.models:
        raise KeyError(f"No model for {sku}")
    history = (
        orders[orders["sku"] == sku][["date", "units_sold"]]
        .sort_values("date")
        .copy()
    )
    reg = model.models[sku]
    rows = []
    for step in range(1, horizon_days + 1):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        values = history["units_sold"].to_numpy()
        x = {f"lag_{lag}": values[-lag] for lag in LAGS}
        x[f"rolling_{ROLLING_WINDOW}"] = values[-ROLLING_WINDOW:].mean()
        x["dayofweek"] = next_date.dayofweek
        x["month"] = next_date.month
        pred = max(0.0, float(reg.predict(pd.DataFrame([x])[FEATURE_COLS])[0]))
        rows.append({"date": next_date, "forecast_units": round(pred, 1)})
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "units_sold": [pred]})],
            ignore_index=True,
        )
    return pd.DataFrame(rows)
