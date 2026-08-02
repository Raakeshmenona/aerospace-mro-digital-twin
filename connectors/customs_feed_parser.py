"""Customs / logistics clearance XML feed parser.

Imported aerospace parts stall at customs more often than anywhere else in the
chain - an airworthiness release (EASA Form 1 / FAA 8130-3) mismatch can hold a
$400k engine part for weeks. This module turns the broker's XML manifest into a
clearance-risk table the AOG scorer can consume.

Uses iterparse so a 500 MB nightly manifest streams in constant memory.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

log = logging.getLogger(__name__)

# Broker feeds vary; these are the status values we normalize onto.
CLEARANCE_STATES = {
    "IN_TRANSIT": 0,
    "AT_CUSTOMS": 1,
    "HELD": 2,
    "CLEARED": 3,
    "DELIVERED": 4,
}
# Days of extra delay we assume per state when projecting arrival.
STATE_DELAY_DAYS = {"IN_TRANSIT": 3, "AT_CUSTOMS": 5, "HELD": 14, "CLEARED": 1, "DELIVERED": 0}


@dataclass
class Shipment:
    awb: str
    po_number: str
    part_number: str
    quantity: float | None
    clearance_status: str
    estimated_clearance_days: float | None
    hold_reason: str = ""


def _text(elem: ET.Element, tag: str) -> str:
    """Namespace-tolerant child lookup - broker feeds sometimes ship a default
    namespace and sometimes don't, for the same customer, on alternate days."""
    for child in elem:
        local = child.tag.rsplit("}", 1)[-1]
        if local.lower() == tag.lower():
            return (child.text or "").strip()
    return ""


def iter_shipments(source: str | Path) -> Iterator[Shipment]:
    """Stream Shipment records from a customs manifest XML file."""
    for _event, elem in ET.iterparse(str(source), events=("end",)):
        if elem.tag.rsplit("}", 1)[-1] != "Shipment":
            continue
        status = (_text(elem, "ClearanceStatus") or "IN_TRANSIT").upper()
        if status not in CLEARANCE_STATES:
            log.warning("unknown clearance status %r, treating as IN_TRANSIT", status)
            status = "IN_TRANSIT"
        yield Shipment(
            awb=elem.get("awb", ""),
            po_number=elem.get("poRef", ""),
            part_number=_text(elem, "PartNumber"),
            quantity=_to_float(_text(elem, "Quantity")),
            clearance_status=status,
            estimated_clearance_days=_to_float(_text(elem, "EstimatedClearanceDays")),
            hold_reason=_text(elem, "HoldReason"),
        )
        elem.clear()  # free the subtree; this is what keeps memory flat


def parse_customs_feed(source: str | Path) -> pd.DataFrame:
    """Parse a manifest into a DataFrame with projected clearance dates."""
    df = pd.DataFrame([s.__dict__ for s in iter_shipments(source)])
    if df.empty:
        return df
    now = pd.Timestamp(datetime.utcnow().date())
    # Trust the broker's ETA when given; otherwise fall back to the state prior.
    fallback = df["clearance_status"].map(STATE_DELAY_DAYS)
    days = df["estimated_clearance_days"].fillna(fallback)
    df["projected_clearance_date"] = now + pd.to_timedelta(days, unit="D")
    df["clearance_stage"] = df["clearance_status"].map(CLEARANCE_STATES)
    df["is_held"] = df["clearance_status"].eq("HELD")
    df["parsed_at"] = datetime.utcnow()
    return df


def customs_delay_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Per-PO customs risk contribution, 0-1, for the AOG scorer.

    Held shipments are the top signal; long projected clearance is the second.
    """
    if df.empty:
        return pd.DataFrame(columns=["po_number", "part_number", "customs_risk", "projected_clearance_date"])
    out = df.copy()
    days = (out["projected_clearance_date"] - pd.Timestamp(datetime.utcnow().date())).dt.days.clip(lower=0)
    # 21 days is the practical worst case before an MRO re-sources the part.
    out["customs_risk"] = (days / 21).clip(0, 1)
    out.loc[out["is_held"], "customs_risk"] = 1.0
    out.loc[out["clearance_status"].isin(["CLEARED", "DELIVERED"]), "customs_risk"] = 0.0
    grouped = (
        out.groupby(["po_number", "part_number"], as_index=False)
        .agg(customs_risk=("customs_risk", "max"), projected_clearance_date=("projected_clearance_date", "max"))
    )
    return grouped


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "data/sample/customs_shipments.xml"
    frame = parse_customs_feed(path)
    print(frame.head(10).to_string(index=False))
    print(f"\n{len(frame)} shipments, {int(frame['is_held'].sum())} held at customs")
