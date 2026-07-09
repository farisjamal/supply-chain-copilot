"""Synthetic supply chain dataset generator.

Produces four CSVs under data/:
  orders.csv     - 2 years of daily unit sales per SKU (trend + weekly seasonality + promos + noise)
  inventory.csv  - current stock position per SKU with reorder points
  suppliers.csv  - supplier master data
  shipments.csv  - inbound purchase orders with promised vs actual delivery dates

Deterministic (seeded) so results are reproducible.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import DATA_DIR

RNG_SEED = 42

SKUS = [
    # sku, product_name, base_daily_demand, trend_per_day, unit_cost, supplier_id
    ("SKU-1001", "USB-C Cable 1m", 120, 0.05, 1.80, "SUP-01"),
    ("SKU-1002", "Wireless Mouse", 80, 0.02, 6.50, "SUP-01"),
    ("SKU-1003", "Mechanical Keyboard", 35, 0.03, 28.00, "SUP-02"),
    ("SKU-1004", "27in Monitor", 18, 0.01, 145.00, "SUP-02"),
    ("SKU-1005", "Laptop Stand", 55, -0.01, 12.00, "SUP-03"),
    ("SKU-1006", "Webcam 1080p", 42, 0.00, 22.50, "SUP-03"),
    ("SKU-1007", "USB Hub 4-port", 95, 0.04, 7.20, "SUP-04"),
    ("SKU-1008", "Headset with Mic", 60, 0.02, 18.90, "SUP-04"),
]

SUPPLIERS = [
    # supplier_id, name, country, avg_lead_days, on_time_delivery_rate
    ("SUP-01", "Acme Components Ltd", "Vietnam", 12, 0.94),
    ("SUP-02", "Shenzhen Electro Co", "China", 18, 0.82),
    ("SUP-03", "Nordic Peripherals AB", "Sweden", 8, 0.97),
    ("SUP-04", "Delta Devices Inc", "Malaysia", 15, 0.88),
]

HISTORY_DAYS = 730


def generate_orders(rng: np.random.Generator, end: date) -> pd.DataFrame:
    dates = pd.date_range(end=end, periods=HISTORY_DAYS, freq="D")
    rows = []
    # weekday multipliers: weekends dip for B2B-style demand
    dow_factor = np.array([1.05, 1.1, 1.1, 1.05, 1.0, 0.6, 0.5])
    for sku, _name, base, trend, _cost, _sup in SKUS:
        t = np.arange(HISTORY_DAYS)
        level = base + trend * t
        seasonal = level * dow_factor[dates.dayofweek]
        noise = rng.normal(0, level * 0.12)
        units = seasonal + noise
        # occasional promo spikes
        promo_days = rng.choice(HISTORY_DAYS, size=10, replace=False)
        units[promo_days] *= rng.uniform(1.6, 2.4, size=10)
        units = np.clip(units, 0, None).round().astype(int)
        rows.append(pd.DataFrame({"date": dates, "sku": sku, "units_sold": units}))
    return pd.concat(rows, ignore_index=True)


def generate_inventory(rng: np.random.Generator, orders: pd.DataFrame) -> pd.DataFrame:
    recent = orders[orders["date"] >= orders["date"].max() - pd.Timedelta(days=28)]
    avg_daily = recent.groupby("sku")["units_sold"].mean()
    rows = []
    for sku, name, _base, _trend, cost, sup in SKUS:
        daily = avg_daily[sku]
        lead = next(s[3] for s in SUPPLIERS if s[0] == sup)
        safety = int(daily * 5)
        reorder_point = int(daily * lead + safety)
        # a few SKUs deliberately near/below reorder point so the agent has problems to find
        cover_days = rng.uniform(3, 45)
        on_hand = int(daily * cover_days)
        rows.append(
            {
                "sku": sku,
                "product_name": name,
                "on_hand": on_hand,
                "reorder_point": reorder_point,
                "safety_stock": safety,
                "lead_time_days": lead,
                "unit_cost": cost,
                "supplier_id": sup,
            }
        )
    return pd.DataFrame(rows)


def generate_shipments(rng: np.random.Generator, end: date) -> pd.DataFrame:
    rows = []
    ship_id = 5000
    for sup_id, _name, _country, lead, otd in SUPPLIERS:
        sup_skus = [s[0] for s in SKUS if s[5] == sup_id]
        for _ in range(30):
            ship_id += 1
            sku = rng.choice(sup_skus)
            order_dt = end - timedelta(days=int(rng.integers(5, 180)))
            promised = order_dt + timedelta(days=lead)
            if rng.random() < otd:
                delay = int(rng.integers(-2, 1))
            else:
                delay = int(rng.integers(2, 15))
            delivered = promised + timedelta(days=delay)
            status = "delivered" if delivered <= end else "in_transit"
            rows.append(
                {
                    "shipment_id": f"SHP-{ship_id}",
                    "supplier_id": sup_id,
                    "sku": sku,
                    "order_date": order_dt,
                    "promised_date": promised,
                    "delivered_date": delivered if status == "delivered" else pd.NaT,
                    "status": status,
                    "quantity": int(rng.integers(200, 2000)),
                }
            )
    return pd.DataFrame(rows)


def generate_all(seed: int = RNG_SEED) -> None:
    """Generate all CSVs into data/."""
    rng = np.random.default_rng(seed)
    end = date.today()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    orders = generate_orders(rng, end)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    inventory = generate_inventory(rng, orders)
    inventory.to_csv(DATA_DIR / "inventory.csv", index=False)

    suppliers = pd.DataFrame(
        SUPPLIERS,
        columns=["supplier_id", "name", "country", "avg_lead_days", "on_time_delivery_rate"],
    )
    suppliers.to_csv(DATA_DIR / "suppliers.csv", index=False)

    shipments = generate_shipments(rng, end)
    shipments.to_csv(DATA_DIR / "shipments.csv", index=False)

    print(f"Generated {len(orders)} order rows, {len(inventory)} SKUs, "
          f"{len(suppliers)} suppliers, {len(shipments)} shipments -> {DATA_DIR}")


if __name__ == "__main__":
    generate_all()
