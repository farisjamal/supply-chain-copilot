"""Agent tools — the actions the LLM can take.

Each tool has a docstring the LLM reads to decide when to call it
(tool calling). Tools return compact, structured text the model can
reason over; heavy lifting (pandas, sklearn, retrieval) stays in Python.
"""

from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from .. import data_store
from ..ml import forecasting
from ..ml.anomaly import score_shipments
from ..rag.pipeline import get_retriever


@lru_cache(maxsize=1)
def _forecast_model() -> forecasting.ForecastModel:
    return forecasting.train(data_store.load_orders())


@tool
def check_inventory(sku: str) -> str:
    """Get the current stock position for a SKU: on-hand units, reorder point,
    days of cover, lead time and supplier. Use SKU codes like 'SKU-1003'.
    Pass 'ALL' to get a summary table of every SKU."""
    inv = data_store.load_inventory()
    orders = data_store.load_orders()
    recent = orders[orders["date"] >= orders["date"].max() - pd.Timedelta(days=28)]
    avg_daily = recent.groupby("sku")["units_sold"].mean()

    def describe(row) -> str:
        daily = avg_daily.get(row["sku"], 0.0)
        cover = row["on_hand"] / daily if daily else float("inf")
        flag = "BELOW REORDER POINT" if row["on_hand"] <= row["reorder_point"] else "ok"
        return (
            f"{row['sku']} ({row['product_name']}): on_hand={row['on_hand']}, "
            f"reorder_point={row['reorder_point']}, days_of_cover={cover:.1f}, "
            f"lead_time={row['lead_time_days']}d, supplier={row['supplier_id']}, status={flag}"
        )

    if sku.strip().upper() == "ALL":
        return "\n".join(describe(r) for _, r in inv.iterrows())
    match = inv[inv["sku"].str.upper() == sku.strip().upper()]
    if match.empty:
        return f"Unknown SKU {sku!r}. Valid SKUs: {', '.join(inv['sku'])}"
    return describe(match.iloc[0])


@tool
def forecast_demand(sku: str, horizon_days: int = 14) -> str:
    """Forecast daily demand for a SKU over the next N days using the trained
    ML model (RandomForest on lag/seasonality features). Returns the daily
    forecast, the total, and the model's holdout MAE so you can convey
    uncertainty."""
    orders = data_store.load_orders()
    model = _forecast_model()
    try:
        fc = forecasting.forecast(model, orders, sku.strip().upper(), horizon_days)
    except KeyError:
        return f"Unknown SKU {sku!r}. Valid SKUs: {', '.join(model.models)}"
    m = model.metrics[sku.strip().upper()]
    lines = [f"{r.date.date()}: {r.forecast_units}" for r in fc.itertuples()]
    return (
        f"Demand forecast for {sku} next {horizon_days} days "
        f"(holdout MAE={m['mae']:.1f} units/day vs mean demand {m['mean_demand']:.1f}):\n"
        + "\n".join(lines)
        + f"\nTOTAL: {fc['forecast_units'].sum():.0f} units"
    )


@tool
def shipment_status(supplier_id: str = "ALL") -> str:
    """Review inbound shipments: late deliveries and ML-flagged anomalies
    (IsolationForest over delay and quantity). Optionally filter by supplier
    id like 'SUP-02'. Use this for questions about delivery performance,
    late POs, or supplier reliability."""
    scored = score_shipments(data_store.load_shipments())
    sups = data_store.load_suppliers().set_index("supplier_id")
    if supplier_id.strip().upper() != "ALL":
        scored = scored[scored["supplier_id"].str.upper() == supplier_id.strip().upper()]
        if scored.empty:
            return f"No delivered shipments for supplier {supplier_id!r}."
    late = scored[scored["is_late"]].sort_values("delay_days", ascending=False)
    lines = [
        f"Delivered shipments: {len(scored)}, late: {len(late)} "
        f"({len(late) / len(scored):.0%}), anomalies flagged: {int(scored['is_anomaly'].sum())}"
    ]
    for _, r in late.head(10).iterrows():
        sup_name = sups.loc[r["supplier_id"], "name"]
        tag = " [ANOMALY]" if r["is_anomaly"] else ""
        lines.append(
            f"{r['shipment_id']} {r['sku']} from {sup_name} ({r['supplier_id']}): "
            f"{r['delay_days']}d late, qty={r['quantity']}{tag}"
        )
    return "\n".join(lines)


@tool
def search_supplier_docs(query: str) -> str:
    """Search the internal knowledge base (supplier contracts, inventory policy,
    logistics SOPs, incoterms guide) and return the most relevant passages.
    Use this for questions about payment terms, penalties, MOQs, escalation
    rules, reorder policy, or incoterms. This is a RAG retrieval step —
    ground your answer in the returned passages and cite the source file."""
    hits = get_retriever().search(query, k=3)
    if not hits:
        return "No relevant passages found."
    out = []
    for chunk, score in hits:
        out.append(f"[source: {chunk.source} — {chunk.heading} (score {score:.2f})]\n{chunk.text}")
    return "\n\n---\n\n".join(out)


@tool
def low_stock_report() -> str:
    """List every SKU at or below its reorder point, with supplier and lead
    time, ordered by urgency (days of cover). Use this when asked what needs
    reordering or where stockout risk is."""
    inv = data_store.load_inventory()
    orders = data_store.load_orders()
    recent = orders[orders["date"] >= orders["date"].max() - pd.Timedelta(days=28)]
    avg_daily = recent.groupby("sku")["units_sold"].mean()
    inv = inv.assign(days_of_cover=inv.apply(
        lambda r: r["on_hand"] / avg_daily.get(r["sku"], 1.0), axis=1
    ))
    low = inv[inv["on_hand"] <= inv["reorder_point"]].sort_values("days_of_cover")
    if low.empty:
        return "No SKUs are at or below their reorder point."
    lines = ["SKUs at/below reorder point (most urgent first):"]
    for _, r in low.iterrows():
        risk = "STOCKOUT RISK" if r["days_of_cover"] < r["lead_time_days"] else "reorder now"
        lines.append(
            f"{r['sku']} ({r['product_name']}): {r['on_hand']} on hand, "
            f"{r['days_of_cover']:.1f} days cover vs {r['lead_time_days']}d lead time "
            f"from {r['supplier_id']} -> {risk}"
        )
    return "\n".join(lines)


TOOLS = [check_inventory, forecast_demand, shipment_status, search_supplier_docs, low_stock_report]
