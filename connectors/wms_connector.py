"""Warehouse Management System adapter.

MRO stores rarely run one WMS. A typical operator has a modern cloud WMS at the
main base and a legacy ODBC-backed system at line stations. This adapter
normalizes both onto one inventory schema and adds the two things spreadsheets
always get wrong:

  * available vs on-hand (reserved and quarantined stock is NOT available)
  * shelf-life expiry (a part in the bin can still be unusable tomorrow)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

INVENTORY_COLUMNS = [
    "warehouse_id",
    "part_number",
    "qty_on_hand",
    "qty_reserved",
    "qty_quarantine",
    "qty_available",
    "bin_location",
    "condition_code",
    "expiry_date",
    "last_counted",
    "source_system",
]

# Only new/overhauled/serviceable stock can be issued against a work order.
ISSUABLE_CONDITIONS = {"NE", "OH", "SV"}


class WMSAdapter(ABC):
    """Interface every warehouse backend implements."""

    source_system: str = "WMS"

    @abstractmethod
    def _fetch_raw(self) -> pd.DataFrame:
        ...

    def fetch_inventory(self) -> pd.DataFrame:
        df = self._fetch_raw()
        return normalize_inventory(df, self.source_system)


class RestWMSAdapter(WMSAdapter):
    """Cloud WMS with a JSON stock endpoint."""

    source_system = "WMS_CLOUD"

    def __init__(self, base_url: str, api_key: str, warehouse_ids: list[str], session: Any = None):
        import requests

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.warehouse_ids = warehouse_ids
        self.session = session or requests.Session()

    def _fetch_raw(self) -> pd.DataFrame:
        frames = []
        for wh in self.warehouse_ids:
            resp = self.session.get(
                f"{self.base_url}/v2/inventory",
                params={"warehouseId": wh},
                headers={"X-API-Key": self.api_key},
                timeout=60,
            )
            resp.raise_for_status()
            frames.append(pd.DataFrame(resp.json().get("items", [])))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class OdbcWMSAdapter(WMSAdapter):
    """Legacy line-station WMS reachable only over ODBC."""

    source_system = "WMS_LEGACY"

    QUERY = """
        SELECT WHSE_ID   AS warehouse_id,
               PART_NO   AS part_number,
               QTY_OH    AS qty_on_hand,
               QTY_RESV  AS qty_reserved,
               QTY_QUAR  AS qty_quarantine,
               BIN_LOC   AS bin_location,
               COND_CD   AS condition_code,
               EXP_DT    AS expiry_date,
               CYCLE_DT  AS last_counted
        FROM   STOCK_MASTER
        WHERE  QTY_OH > 0
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    def _fetch_raw(self) -> pd.DataFrame:
        try:
            import pyodbc  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyodbc required for OdbcWMSAdapter") from exc
        with pyodbc.connect(self.dsn) as conn:
            return pd.read_sql(self.QUERY, conn)


class ParquetWMSAdapter(WMSAdapter):
    """Local demo adapter reading the synthetic inventory extract."""

    source_system = "WMS_DEMO"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _fetch_raw(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


def normalize_inventory(df: pd.DataFrame, source_system: str) -> pd.DataFrame:
    """Coerce any WMS payload onto INVENTORY_COLUMNS and derive availability."""
    if df.empty:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    out = df.rename(columns={
        "warehouseId": "warehouse_id", "partNumber": "part_number",
        "quantityOnHand": "qty_on_hand", "quantityReserved": "qty_reserved",
        "quantityQuarantine": "qty_quarantine", "binLocation": "bin_location",
        "condition": "condition_code", "expiryDate": "expiry_date",
        "lastCycleCount": "last_counted",
    }).copy()

    for col in INVENTORY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    for col in ("qty_on_hand", "qty_reserved", "qty_quarantine"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in ("expiry_date", "last_counted"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    out["condition_code"] = out["condition_code"].fillna("NE").astype(str).str.upper().str.strip()

    available = out["qty_on_hand"] - out["qty_reserved"] - out["qty_quarantine"]
    # Non-issuable conditions (AR = as-removed, scrap) count as zero available.
    available = available.where(out["condition_code"].isin(ISSUABLE_CONDITIONS), 0)
    # Expired shelf-life stock is physically present but legally unusable.
    expired = out["expiry_date"].notna() & (out["expiry_date"] < pd.Timestamp(datetime.utcnow().date()))
    available = available.where(~expired, 0)
    out["qty_available"] = available.clip(lower=0).astype(int)

    out["source_system"] = source_system
    return out[INVENTORY_COLUMNS]


def stock_accuracy_report(inventory: pd.DataFrame, stale_after_days: int = 90) -> dict[str, float]:
    """Cycle-count freshness - the driver behind the 70% -> 95% accuracy KPI.

    Records not counted recently are the ones spreadsheets get wrong, so this
    is the metric that justifies the pipeline to a supply chain manager.
    """
    if inventory.empty:
        return {"records": 0, "stale_pct": 0.0, "trusted_pct": 0.0}
    cutoff = pd.Timestamp(datetime.utcnow().date()) - timedelta(days=stale_after_days)
    stale = inventory["last_counted"].isna() | (inventory["last_counted"] < cutoff)
    return {
        "records": int(len(inventory)),
        "stale_pct": round(100 * stale.mean(), 1),
        "trusted_pct": round(100 * (~stale).mean(), 1),
    }


def fetch_all(adapters: list[WMSAdapter]) -> pd.DataFrame:
    """Union inventory across every warehouse backend."""
    frames = []
    for a in adapters:
        try:
            frames.append(a.fetch_inventory())
        except Exception:
            # One dead line-station WMS must not fail the nightly sync.
            log.exception("WMS adapter %s failed, continuing", a.source_system)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INVENTORY_COLUMNS)
