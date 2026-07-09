# Inventory Management Policy

## Reorder Policy
We operate a continuous-review (s, Q) reorder point system. When on-hand inventory for a SKU falls to or below its reorder point, a replenishment purchase order must be raised within 1 business day.

Reorder point formula: average daily demand × supplier lead time (days) + safety stock.

## Safety Stock
Safety stock is set at 5 days of average demand for all SKUs. High-variability SKUs (coefficient of variation above 0.5) may be raised to 10 days with supply chain manager approval.

## Stockout Handling
If projected days of cover falls below the supplier lead time, the SKU is classified as at risk of stockout. Mitigations, in order of preference: expedite an in-transit shipment, place an emergency air-freight order, or substitute with an equivalent SKU.

## Excess Stock
On-hand above 60 days of cover is classified as excess. Excess stock should be flagged for promotion or channel rebalancing before the next replenishment cycle.

## ABC Classification
A-class SKUs (top 80% of revenue) are counted monthly; B-class quarterly; C-class twice a year.
