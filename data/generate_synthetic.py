"""Generate synthetic MRO supply-chain data for local demos.

Creates in data/sample/:
  parts_master.parquet        - part catalog (ATA chapters, criticality)
  suppliers.parquet           - supplier master with base lead times
  purchase_orders.parquet     - PO history (promised vs actual delivery)
  inventory.parquet           - on-hand stock per warehouse
  work_orders.parquet         - MRO work orders consuming parts
  demand_history.parquet      - monthly part demand (for forecasting)
  customs_shipments.xml       - customs clearance feed (XML)
  edi_850_orders.txt          - EDI 850 purchase orders (flat segments)
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
random.seed(42)

OUT = Path(__file__).parent / "sample"

ATA_CHAPTERS = {
    "21": "Air Conditioning", "24": "Electrical Power", "27": "Flight Controls",
    "29": "Hydraulic Power", "32": "Landing Gear", "49": "APU",
    "71": "Power Plant", "72": "Engine", "73": "Engine Fuel",
}
CRITICALITY = ["NO_GO", "GO_IF", "GO"]  # NO_GO = grounds the aircraft
COUNTRIES = ["US", "DE", "FR", "GB", "SG", "IN", "JP"]
WAREHOUSES = ["DEL-MRO-01", "BLR-MRO-02", "HYD-MRO-03"]

N_PARTS = 2000
N_SUPPLIERS = 60
N_POS = 5000
N_WORK_ORDERS = 1500
TODAY = date(2026, 7, 31)


def gen_parts() -> pd.DataFrame:
    chapters = RNG.choice(list(ATA_CHAPTERS), N_PARTS)
    return pd.DataFrame({
        "part_number": [f"PN-{c}-{i:05d}" for i, c in enumerate(chapters)],
        "description": [f"{ATA_CHAPTERS[c]} component {i}" for i, c in enumerate(chapters)],
        "ata_chapter": chapters,
        "criticality": RNG.choice(CRITICALITY, N_PARTS, p=[0.15, 0.35, 0.50]),
        "unit_cost_usd": np.round(RNG.lognormal(7, 1.5, N_PARTS), 2),
        "shelf_life_months": RNG.choice([0, 12, 24, 36, 60], N_PARTS),
        "repairable_flag": RNG.choice([True, False], N_PARTS, p=[0.4, 0.6]),
    })


def gen_suppliers() -> pd.DataFrame:
    return pd.DataFrame({
        "supplier_id": [f"SUP-{i:04d}" for i in range(N_SUPPLIERS)],
        "supplier_name": [f"Aero Supplier {i}" for i in range(N_SUPPLIERS)],
        "country": RNG.choice(COUNTRIES, N_SUPPLIERS),
        "base_lead_time_days": RNG.integers(30, 540, N_SUPPLIERS),
        "quality_rating": np.round(RNG.uniform(2.5, 5.0, N_SUPPLIERS), 1),
        "approved_vendor": RNG.choice([True, False], N_SUPPLIERS, p=[0.9, 0.1]),
    })


def gen_purchase_orders(parts: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    part_idx = RNG.integers(0, N_PARTS, N_POS)
    sup_idx = RNG.integers(0, N_SUPPLIERS, N_POS)
    order_dates = [TODAY - timedelta(days=int(d)) for d in RNG.integers(1, 900, N_POS)]
    base_lt = suppliers["base_lead_time_days"].to_numpy()[sup_idx]
    promised = base_lt + RNG.integers(-10, 10, N_POS)
    # Each supplier has its own reliability: p(late) and how bad the slip is.
    # This produces a realistic spread of on-time rates (roughly 40%-98%)
    # so the scorecard has something to actually discriminate on.
    sup_late_prob = RNG.beta(2, 6, N_SUPPLIERS)[sup_idx]
    sup_slip_scale = RNG.gamma(2, 9, N_SUPPLIERS)[sup_idx]
    is_late = RNG.random(N_POS) < sup_late_prob
    slip = np.where(is_late, RNG.exponential(sup_slip_scale), -RNG.integers(0, 12, N_POS))
    actual = (promised + slip).astype(int).clip(5)
    rows = pd.DataFrame({
        "po_number": [f"PO-{i:06d}" for i in range(N_POS)],
        "part_number": parts["part_number"].to_numpy()[part_idx],
        "supplier_id": suppliers["supplier_id"].to_numpy()[sup_idx],
        "order_date": order_dates,
        "promised_lead_time_days": promised,
        "quantity": RNG.integers(1, 25, N_POS),
        "unit_price_usd": np.round(parts["unit_cost_usd"].to_numpy()[part_idx] * RNG.uniform(0.9, 1.2, N_POS), 2),
    })
    rows["promised_date"] = [d + timedelta(days=int(p)) for d, p in zip(order_dates, promised)]
    rows["actual_delivery_date"] = [d + timedelta(days=int(a)) for d, a in zip(order_dates, actual)]
    # open POs: those whose actual delivery is in the future
    rows.loc[rows["actual_delivery_date"] > pd.Timestamp(TODAY).date(), "status"] = "OPEN"
    rows["status"] = rows["status"].fillna("DELIVERED") if "status" in rows else "DELIVERED"
    rows.loc[rows["status"] == "OPEN", "actual_delivery_date"] = pd.NaT
    return rows


def gen_inventory(parts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wh in WAREHOUSES:
        sub = parts.sample(frac=0.6, random_state=hash(wh) % 2**31)
        n = len(sub)
        # Most bins are stocked to a reorder point; a realistic minority have
        # run down or stocked out. A generator where everything is already at
        # zero would make the early-warning KPI meaningless - the point is to
        # catch parts on the way down, not after they hit bottom.
        qty = RNG.poisson(9, n)
        stocked_out = RNG.random(n) < 0.08
        qty = np.where(stocked_out, 0, qty)
        # Cycle-count discipline decays: many records go uncounted for months,
        # which is exactly the stale-data problem the accuracy KPI measures.
        days_since_count = RNG.gamma(shape=2.0, scale=45, size=n).astype(int).clip(0, 400)
        rows.append(pd.DataFrame({
            "warehouse_id": wh,
            "part_number": sub["part_number"].to_numpy(),
            "qty_on_hand": qty,
            "qty_reserved": np.minimum(RNG.poisson(1, n), qty),
            "last_counted": pd.Timestamp(TODAY) - pd.to_timedelta(days_since_count, unit="D"),
        }))
    return pd.concat(rows, ignore_index=True)


def gen_work_orders(parts: pd.DataFrame) -> pd.DataFrame:
    open_dates = [TODAY - timedelta(days=int(d)) for d in RNG.integers(0, 365, N_WORK_ORDERS)]
    return pd.DataFrame({
        "work_order_id": [f"WO-{i:06d}" for i in range(N_WORK_ORDERS)],
        "aircraft_reg": RNG.choice([f"VT-{c}{d}" for c in "ABI" for d in "XYZQW"], N_WORK_ORDERS),
        "part_number": parts["part_number"].sample(N_WORK_ORDERS, replace=True, random_state=7).to_numpy(),
        "qty_required": RNG.integers(1, 4, N_WORK_ORDERS),
        "wo_type": RNG.choice(["SCHEDULED", "UNSCHEDULED", "AOG"], N_WORK_ORDERS, p=[0.6, 0.3, 0.1]),
        "opened_date": open_dates,
        "status": RNG.choice(["OPEN", "CLOSED"], N_WORK_ORDERS, p=[0.25, 0.75]),
        "labor_hours": np.round(RNG.gamma(3, 4, N_WORK_ORDERS), 1),
    })


def gen_demand_history(parts: pd.DataFrame) -> pd.DataFrame:
    """36 months of demand per part with seasonality + trend for forecasting."""
    months = pd.period_range(end=pd.Period(TODAY, "M"), periods=36)
    month_nums = np.array([m.month for m in months])
    n = len(parts)
    # Spares demand is intermittent: most parts move rarely. base rate is
    # lognormal so a few fast movers dominate, and an intermittency mask
    # zeroes out months entirely for slow movers.
    base = RNG.lognormal(-0.3, 1.1, n)
    trend = RNG.uniform(-0.02, 0.05, n)
    seasonal = RNG.uniform(0, 0.5, n)
    move_prob = RNG.beta(3, 2, n)  # probability the part moves in a given month

    t = np.arange(len(months))
    # lam[part, month]
    lam = (
        base[:, None]
        * (1 + trend[:, None] * t[None, :])
        * (1 + seasonal[:, None] * np.sin(2 * np.pi * month_nums[None, :] / 12))
    ).clip(0.02)
    moved = RNG.random((n, len(months))) < move_prob[:, None]
    qty = RNG.poisson(lam) * moved

    return pd.DataFrame({
        "part_number": np.repeat(parts["part_number"].to_numpy(), len(months)),
        "month": np.tile([str(m) for m in months], n),
        "demand_qty": qty.ravel(),
    })


def gen_customs_xml(pos: pd.DataFrame) -> str:
    open_pos = pos[pos["status"] == "OPEN"].head(200)
    items = []
    for _, r in open_pos.iterrows():
        stat = random.choice(["IN_TRANSIT", "AT_CUSTOMS", "CLEARED", "HELD"])
        items.append(
            f'  <Shipment awb="AWB{random.randint(10**7, 10**8)}" poRef="{r.po_number}">\n'
            f"    <PartNumber>{r.part_number}</PartNumber>\n"
            f"    <Quantity>{r.quantity}</Quantity>\n"
            f"    <ClearanceStatus>{stat}</ClearanceStatus>\n"
            f"    <EstimatedClearanceDays>{random.randint(1, 21)}</EstimatedClearanceDays>\n"
            f"  </Shipment>"
        )
    return '<?xml version="1.0" encoding="UTF-8"?>\n<CustomsManifest generated="2026-07-31">\n' + "\n".join(items) + "\n</CustomsManifest>\n"


def gen_edi_850(pos: pd.DataFrame) -> str:
    lines = []
    for _, r in pos.head(50).iterrows():
        lines += [
            f"ST*850*{r.po_number[-6:]}",
            f"BEG*00*NE*{r.po_number}**{r.order_date:%Y%m%d}",
            f"PO1*1*{r.quantity}*EA*{r.unit_price_usd}**BP*{r.part_number}",
            f"SE*4*{r.po_number[-6:]}",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = gen_parts()
    suppliers = gen_suppliers()
    pos = gen_purchase_orders(parts, suppliers)
    parts.to_parquet(OUT / "parts_master.parquet", index=False)
    suppliers.to_parquet(OUT / "suppliers.parquet", index=False)
    pos.to_parquet(OUT / "purchase_orders.parquet", index=False)
    gen_inventory(parts).to_parquet(OUT / "inventory.parquet", index=False)
    gen_work_orders(parts).to_parquet(OUT / "work_orders.parquet", index=False)
    gen_demand_history(parts).to_parquet(OUT / "demand_history.parquet", index=False)
    (OUT / "customs_shipments.xml").write_text(gen_customs_xml(pos), encoding="utf-8")
    (OUT / "edi_850_orders.txt").write_text(gen_edi_850(pos), encoding="utf-8")
    print(f"Wrote synthetic data to {OUT}")


if __name__ == "__main__":
    main()
