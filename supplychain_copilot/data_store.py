"""Lazy, cached access to the CSV dataset. Auto-generates data on first use."""

from functools import lru_cache

import pandas as pd

from .config import DATA_DIR


def _ensure_data() -> None:
    if not (DATA_DIR / "orders.csv").exists():
        from .data_gen import generate_all

        generate_all()


@lru_cache(maxsize=1)
def load_orders() -> pd.DataFrame:
    _ensure_data()
    return pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["date"])


@lru_cache(maxsize=1)
def load_inventory() -> pd.DataFrame:
    _ensure_data()
    return pd.read_csv(DATA_DIR / "inventory.csv")


@lru_cache(maxsize=1)
def load_suppliers() -> pd.DataFrame:
    _ensure_data()
    return pd.read_csv(DATA_DIR / "suppliers.csv")


@lru_cache(maxsize=1)
def load_shipments() -> pd.DataFrame:
    _ensure_data()
    return pd.read_csv(
        DATA_DIR / "shipments.csv",
        parse_dates=["order_date", "promised_date", "delivered_date"],
    )
