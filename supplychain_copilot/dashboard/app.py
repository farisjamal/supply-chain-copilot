"""Streamlit dashboard: operations analytics + chat with the agent.

Run:  streamlit run supplychain_copilot/dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from supplychain_copilot import data_store
from supplychain_copilot.agent.tools import _forecast_model
from supplychain_copilot.ml import forecasting
from supplychain_copilot.ml.anomaly import score_shipments

st.set_page_config(page_title="Supply Chain Copilot", page_icon="📦", layout="wide")
st.title("📦 Supply Chain Copilot")

orders = data_store.load_orders()
inventory = data_store.load_inventory()
shipments = score_shipments(data_store.load_shipments())

tab_overview, tab_forecast, tab_chat = st.tabs(["Operations overview", "Demand forecast", "Ask the agent"])

with tab_overview:
    recent = orders[orders["date"] >= orders["date"].max() - pd.Timedelta(days=28)]
    avg_daily = recent.groupby("sku")["units_sold"].mean()
    low = inventory[inventory["on_hand"] <= inventory["reorder_point"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs tracked", len(inventory))
    c2.metric("SKUs below reorder point", len(low))
    c3.metric("Late shipments", int(shipments["is_late"].sum()))
    c4.metric("Anomalous shipments", int(shipments["is_anomaly"].sum()))

    st.subheader("Daily demand (last 120 days)")
    window = orders[orders["date"] >= orders["date"].max() - pd.Timedelta(days=120)]
    fig = px.line(window, x="date", y="units_sold", color="sku")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Stock vs reorder point")
        inv_plot = inventory.melt(
            id_vars="sku",
            value_vars=["on_hand", "reorder_point"],
            var_name="measure",
            value_name="units",
        )
        fig = px.bar(inv_plot, x="sku", y="units", color="measure", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Shipment delays by supplier")
        fig = px.strip(
            shipments,
            x="supplier_id",
            y="delay_days",
            color="is_anomaly",
            hover_data=["shipment_id", "sku", "quantity"],
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_forecast:
    sku = st.selectbox("SKU", inventory["sku"])
    horizon = st.slider("Forecast horizon (days)", 7, 60, 28)
    model = _forecast_model()
    fc = forecasting.forecast(model, orders, sku, horizon)
    hist = orders[orders["sku"] == sku].tail(90)

    m = model.metrics[sku]
    st.caption(
        f"RandomForest on lag/seasonality features - holdout MAE "
        f"{m['mae']:.1f} units/day (mean demand {m['mean_demand']:.1f})"
    )
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["units_sold"], name="history")
    fig.add_scatter(x=fc["date"], y=fc["forecast_units"], name="forecast", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.metric(f"Total forecast demand, next {horizon} days", f"{fc['forecast_units'].sum():.0f} units")

with tab_chat:
    st.caption(
        "LangGraph ReAct agent with tool calling, conversation memory, and RAG "
        "over supplier contracts. Needs an LLM provider configured in .env."
    )
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    if question := st.chat_input("e.g. Which SKUs risk stockout, and what do our contracts say about expediting?"):
        st.chat_message("user").write(question)
        st.session_state.history.append(("user", question))
        try:
            if "agent" not in st.session_state:
                from supplychain_copilot.agent.graph import build_agent

                st.session_state.agent = build_agent()
            from supplychain_copilot.agent.graph import ask

            with st.spinner("thinking..."):
                answer = ask(st.session_state.agent, question, thread_id="streamlit")
        except Exception as exc:  # missing API key, provider down, etc.
            answer = f"Agent unavailable: {exc}\n\nConfigure a provider in .env (see .env.example)."
        st.chat_message("assistant").write(answer)
        st.session_state.history.append(("assistant", answer))
