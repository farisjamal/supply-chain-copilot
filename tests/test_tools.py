from supplychain_copilot.agent.tools import (
    check_inventory,
    low_stock_report,
    search_supplier_docs,
    shipment_status,
)


def test_check_inventory_known_sku():
    out = check_inventory.invoke({"sku": "SKU-1003"})
    assert "SKU-1003" in out
    assert "reorder_point=" in out


def test_check_inventory_unknown_sku():
    out = check_inventory.invoke({"sku": "SKU-9999"})
    assert "Unknown SKU" in out


def test_check_inventory_all():
    out = check_inventory.invoke({"sku": "ALL"})
    assert out.count("SKU-") >= 8


def test_low_stock_report_runs():
    out = low_stock_report.invoke({})
    assert "reorder" in out.lower()


def test_shipment_status_summary():
    out = shipment_status.invoke({"supplier_id": "ALL"})
    assert "late:" in out


def test_rag_tool_cites_source():
    out = search_supplier_docs.invoke({"query": "minimum order quantity"})
    assert "[source:" in out
