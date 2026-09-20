# Demo Guide

This guide gives a short, repeatable demonstration for interviews.

## Start the project

Make sure Ollama is running and the `llama3.2` model is installed. Then double-click `START_DASHBOARD.bat` on Windows, or run:

```powershell
.venv\Scripts\python.exe -m streamlit run supplychain_copilot/dashboard/app.py
```

Open the **Ask the agent** tab after the dashboard appears.

## Three useful demo questions

### 1. Inventory decision

> Which SKUs need reordering right now?

What this demonstrates: the agent calls the low-stock tool, compares stock coverage with supplier lead time, and identifies stockout risk.

### 2. Multi-source planning

> Do we need to worry about monitor stock? Include the demand forecast.

What this demonstrates: the agent combines current inventory with a machine-learning forecast before recommending an action.

### 3. Document retrieval

> What are the payment terms in the Acme contract?

What this demonstrates: the agent searches supplier documents and answers from the retrieved contract section.

Follow with:

> What products does that contract cover?

This shows conversation memory because the agent must understand that "that contract" means the Acme contract.

## Thirty-second explanation

> I built a supply-chain assistant that lets planners ask operational questions in plain English. The LangGraph agent decides which Python tools to call, such as checking inventory, forecasting demand, analysing shipments, or searching supplier contracts. Python calculates the facts, and the language model turns those facts into a clear recommendation. I also built a Streamlit dashboard so a planner can review the data and chat with the agent in one place.

## Limitations to mention honestly

- The data and contracts are synthetic, so this is a portfolio demonstration rather than a production system.
- Conversation memory is stored in memory and resets when the application restarts.
- TF-IDF works well for this small knowledge base but would need embeddings and a vector database for a large document collection.
- Forecast quality would need additional validation before it supports real purchasing decisions.
