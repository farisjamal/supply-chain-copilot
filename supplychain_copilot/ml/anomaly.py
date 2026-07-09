"""Shipment anomaly detection with IsolationForest.

Flags shipments whose (delay, quantity) pattern is unusual relative to
the rest of the inbound flow — e.g. a normally punctual supplier
suddenly delivering 12 days late, or an abnormally large late order.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


def score_shipments(shipments: pd.DataFrame) -> pd.DataFrame:
    """Return delivered shipments with delay_days, is_late and is_anomaly columns."""
    df = shipments[shipments["status"] == "delivered"].copy()
    df["delay_days"] = (df["delivered_date"] - df["promised_date"]).dt.days
    df["is_late"] = df["delay_days"] > 0

    iso = IsolationForest(contamination=0.08, random_state=0)
    df["anomaly_score"] = iso.fit_predict(df[["delay_days", "quantity"]])
    df["is_anomaly"] = df["anomaly_score"] == -1
    return df.drop(columns=["anomaly_score"])
