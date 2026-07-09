import pandas as pd

from supplychain_copilot import data_store
from supplychain_copilot.ml import forecasting


def test_train_produces_model_and_metrics_per_sku():
    orders = data_store.load_orders()
    model = forecasting.train(orders)
    skus = set(orders["sku"])
    assert set(model.models) == skus
    for sku in skus:
        m = model.metrics[sku]
        assert m["mae"] > 0
        # model should beat wild guessing: MAE well under mean demand
        assert m["mae"] < m["mean_demand"] * 0.5


def test_forecast_shape_and_values():
    orders = data_store.load_orders()
    model = forecasting.train(orders)
    fc = forecasting.forecast(model, orders, "SKU-1001", horizon_days=14)
    assert len(fc) == 14
    assert fc["forecast_units"].notna().all()
    assert (fc["forecast_units"] >= 0).all()
    assert fc["date"].iloc[0] == orders["date"].max() + pd.Timedelta(days=1)
