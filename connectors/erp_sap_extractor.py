"""SAP PM/MM extractor for MRO master and transactional data.

Reads the SAP tables that matter for MRO planning:

  MARA / MAKT  - material master + descriptions (part catalog)
  MARC         - plant-level material data (reorder point, lead time)
  MARD         - storage-location stock (on-hand)
  EKKO / EKPO  - purchasing document header / item (purchase orders)
  AUFK / AFKO  - maintenance order header / operations (work orders)

Two transport modes:

  RFC   - pyrfc against a live SAP gateway (production)
  FILE  - flat-file extract drop (the common reality for MRO shops that
          only get a nightly IDoc/CSV dump from the ERP team)

The FILE mode is the default so the pipeline runs end-to-end locally
without SAP credentials.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# SAP field name -> our canonical name. Keeping this explicit (rather than
# lowercasing SAP names) means downstream code never sees MATNR/LIFNR.
MARA_MAP = {
    "MATNR": "part_number",
    "MAKTX": "description",
    "MTART": "material_type",
    "MATKL": "material_group",
    "NTGEW": "net_weight_kg",
    "MHDHB": "shelf_life_months",
}
EKPO_MAP = {
    "EBELN": "po_number",
    "EBELP": "po_line",
    "MATNR": "part_number",
    "MENGE": "quantity",
    "NETPR": "unit_price",
    "WAERS": "currency",
    "EINDT": "promised_date",
}
AUFK_MAP = {
    "AUFNR": "work_order_id",
    "AUART": "wo_type",
    "EQUNR": "equipment_id",
    "ERDAT": "opened_date",
    "PHAS3": "closed_flag",
}

# SAP stores material numbers zero-padded to 18 chars in older releases.
_SAP_MATNR_WIDTH = 18


@dataclass
class SAPConnection:
    """Connection parameters. In production these come from Azure Key Vault."""

    ashost: str = ""
    sysnr: str = "00"
    client: str = "100"
    user: str = ""
    passwd: str = field(default="", repr=False)
    extract_dir: Path | None = None

    @property
    def mode(self) -> str:
        return "FILE" if self.extract_dir else "RFC"


def strip_matnr(value: str) -> str:
    """SAP pads numeric material numbers with leading zeros; MRO part numbers
    are alphanumeric so only strip when the payload is purely numeric."""
    v = str(value).strip()
    if len(v) == _SAP_MATNR_WIDTH and v.lstrip("0").isdigit():
        return v.lstrip("0")
    return v


class SAPExtractor:
    """Pull SAP PM/MM data into canonical DataFrames."""

    def __init__(self, conn: SAPConnection):
        self.conn = conn
        self._rfc = None

    # ------------------------------------------------------------------ RFC

    def _connect_rfc(self):
        if self._rfc is not None:
            return self._rfc
        try:
            from pyrfc import Connection  # type: ignore
        except ImportError as exc:  # pragma: no cover - needs SAP NW RFC SDK
            raise RuntimeError(
                "pyrfc not installed. Use extract_dir= for file-based extracts, "
                "or install pyrfc plus the SAP NetWeaver RFC SDK."
            ) from exc
        self._rfc = Connection(
            ashost=self.conn.ashost,
            sysnr=self.conn.sysnr,
            client=self.conn.client,
            user=self.conn.user,
            passwd=self.conn.passwd,
        )
        return self._rfc

    def _read_table_rfc(
        self,
        table: str,
        fields: Iterable[str],
        where: str = "",
        batch_size: int = 50_000,
    ) -> pd.DataFrame:
        """RFC_READ_TABLE with paging.

        RFC_READ_TABLE returns fixed-width rows in a single WA field, so we
        slice by the offsets SAP reports back rather than splitting on the
        delimiter - MRO descriptions frequently contain the default '|'.
        """
        conn = self._connect_rfc()
        fields = list(fields)
        rows: list[dict[str, str]] = []
        skip = 0
        while True:
            result = conn.call(
                "RFC_READ_TABLE",
                QUERY_TABLE=table,
                DELIMITER="",
                FIELDS=[{"FIELDNAME": f} for f in fields],
                OPTIONS=[{"TEXT": where}] if where else [],
                ROWSKIPS=skip,
                ROWCOUNT=batch_size,
            )
            offsets = [(f["FIELDNAME"], int(f["OFFSET"]), int(f["LENGTH"])) for f in result["FIELDS"]]
            batch = result["DATA"]
            for rec in batch:
                wa = rec["WA"]
                rows.append({name: wa[off : off + length].strip() for name, off, length in offsets})
            if len(batch) < batch_size:
                break
            skip += batch_size
            log.info("%s: fetched %d rows", table, skip)
        return pd.DataFrame(rows, columns=fields)

    # ----------------------------------------------------------------- FILE

    def _read_table_file(self, table: str, fields: Iterable[str]) -> pd.DataFrame:
        assert self.conn.extract_dir is not None
        path = self.conn.extract_dir / f"{table.lower()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"SAP extract missing: {path}")
        df = pd.read_csv(path, dtype=str).fillna("")
        missing = set(fields) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing SAP fields: {sorted(missing)}")
        return df[list(fields)]

    def _read_table(self, table: str, fields: Iterable[str], where: str = "") -> pd.DataFrame:
        if self.conn.mode == "FILE":
            return self._read_table_file(table, fields)
        return self._read_table_rfc(table, fields, where)

    # ------------------------------------------------------------- extracts

    def extract_material_master(self) -> pd.DataFrame:
        """MARA + MAKT -> part catalog."""
        df = self._read_table("MARA", MARA_MAP.keys())
        df = df.rename(columns=MARA_MAP)
        df["part_number"] = df["part_number"].map(strip_matnr)
        for col in ("net_weight_kg", "shelf_life_months"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source_system"] = "SAP_MM"
        df["extracted_at"] = datetime.utcnow()
        return df.drop_duplicates(subset=["part_number"])

    def extract_purchase_orders(self, changed_since: date | None = None) -> pd.DataFrame:
        """EKPO purchase order lines, optionally delta-filtered on EKKO.AEDAT."""
        where = ""
        if changed_since and self.conn.mode == "RFC":
            where = f"AEDAT GE '{changed_since:%Y%m%d}'"
        df = self._read_table("EKPO", EKPO_MAP.keys(), where)
        df = df.rename(columns=EKPO_MAP)
        df["part_number"] = df["part_number"].map(strip_matnr)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        df["promised_date"] = pd.to_datetime(df["promised_date"], format="%Y%m%d", errors="coerce")
        df["source_system"] = "SAP_MM"
        return df

    def extract_work_orders(self, changed_since: date | None = None) -> pd.DataFrame:
        """AUFK maintenance orders (SAP PM)."""
        where = ""
        if changed_since and self.conn.mode == "RFC":
            where = f"ERDAT GE '{changed_since:%Y%m%d}'"
        df = self._read_table("AUFK", AUFK_MAP.keys(), where)
        df = df.rename(columns=AUFK_MAP)
        df["opened_date"] = pd.to_datetime(df["opened_date"], format="%Y%m%d", errors="coerce")
        df["status"] = df["closed_flag"].map(lambda v: "CLOSED" if str(v).strip() == "X" else "OPEN")
        df["source_system"] = "SAP_PM"
        return df.drop(columns=["closed_flag"])

    def close(self) -> None:
        if self._rfc is not None:
            self._rfc.close()
            self._rfc = None

    def __enter__(self) -> SAPExtractor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
