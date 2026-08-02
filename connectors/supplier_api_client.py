"""Supplier portal REST clients + EDI 850/856 parsing.

Aerospace suppliers expose part availability three ways, and a real MRO
integration has to handle all of them:

  1. Modern REST/JSON portals (Boeing PFM-style, OAuth2 client credentials)
  2. Legacy EDI X12 850 (purchase order) / 856 (advance ship notice) drops
  3. Nothing at all - a human emails a spreadsheet (out of scope here)

`SupplierPortalClient` covers (1) with retry, backoff and rate-limit handling.
`parse_edi_850` / `parse_edi_856` cover (2) without an EDI translator licence.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 4
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class SupplierEndpoint:
    supplier_id: str
    base_url: str
    client_id: str
    client_secret: str
    token_url: str = ""
    page_size: int = 500


class SupplierPortalClient:
    """OAuth2 client-credentials REST client for a supplier parts portal."""

    def __init__(self, endpoint: SupplierEndpoint, session: requests.Session | None = None):
        self.ep = endpoint
        self.session = session or requests.Session()
        self._token: str | None = None
        self._token_expires = datetime.min

    # ---------------------------------------------------------------- auth

    def _ensure_token(self) -> str:
        # Refresh 60s early so a long page loop never fails mid-flight.
        if self._token and datetime.utcnow() < self._token_expires - timedelta(seconds=60):
            return self._token
        url = self.ep.token_url or f"{self.ep.base_url.rstrip('/')}/oauth2/token"
        resp = self.session.post(
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.ep.client_id,
                "client_secret": self.ep.client_secret,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expires = datetime.utcnow() + timedelta(seconds=int(payload.get("expires_in", 3600)))
        return self._token

    # --------------------------------------------------------------- fetch

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.ep.base_url.rstrip('/')}/{path.lstrip('/')}"
        for attempt in range(MAX_RETRIES):
            resp = self.session.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {self._ensure_token()}", "Accept": "application/json"},
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code == 401 and attempt == 0:
                self._token = None  # token revoked early; force one refresh
                continue
            if resp.status_code in RETRYABLE_STATUS:
                # Honour Retry-After when the portal sends it, else exponential.
                wait = float(resp.headers.get("Retry-After", 2**attempt))
                log.warning("%s %s -> %s, retrying in %.0fs", self.ep.supplier_id, path, resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"{self.ep.supplier_id}: {path} failed after {MAX_RETRIES} attempts")

    def _paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        page = 1
        while True:
            body = self._get(path, {**params, "page": page, "pageSize": self.ep.page_size})
            items = body.get("items") or body.get("data") or []
            yield from items
            # Portals disagree on pagination metadata; support both shapes.
            total_pages = body.get("totalPages")
            if total_pages is not None:
                if page >= int(total_pages):
                    return
            elif len(items) < self.ep.page_size:
                return
            page += 1

    # -------------------------------------------------------------- public

    def fetch_part_availability(self, part_numbers: list[str] | None = None) -> pd.DataFrame:
        """Current stock + quoted lead time per part."""
        params: dict[str, Any] = {}
        if part_numbers:
            params["partNumbers"] = ",".join(part_numbers)
        rows = [
            {
                "supplier_id": self.ep.supplier_id,
                "part_number": it.get("partNumber") or it.get("pn"),
                "qty_available": it.get("quantityAvailable", 0),
                "quoted_lead_time_days": it.get("leadTimeDays"),
                "unit_price_usd": it.get("unitPrice"),
                "condition_code": it.get("condition", "NE"),  # NE/OH/SV/AR
                "retrieved_at": datetime.utcnow(),
            }
            for it in self._paginate("v1/parts/availability", params)
        ]
        return pd.DataFrame(rows)

    def fetch_order_status(self, since: datetime) -> pd.DataFrame:
        """Open order milestones - what actually drives AOG risk."""
        rows = [
            {
                "supplier_id": self.ep.supplier_id,
                "po_number": it.get("purchaseOrderNumber"),
                "part_number": it.get("partNumber"),
                "quantity": it.get("quantity"),
                "status": it.get("status"),
                "promised_date": it.get("promisedShipDate"),
                "revised_date": it.get("revisedShipDate"),
                "tracking_number": it.get("trackingNumber"),
                "retrieved_at": datetime.utcnow(),
            }
            for it in self._paginate("v1/orders", {"modifiedSince": since.strftime("%Y-%m-%dT%H:%M:%SZ")})
        ]
        df = pd.DataFrame(rows)
        for col in ("promised_date", "revised_date"):
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
        return df


# ----------------------------------------------------------------- EDI X12


def _split_segments(raw: str) -> list[list[str]]:
    """Split X12 into element lists. Handles both '~'-terminated single-line
    files and newline-delimited segment dumps."""
    text = raw.replace("~", "\n")
    return [line.strip().split("*") for line in text.splitlines() if line.strip()]


def parse_edi_850(raw: str) -> pd.DataFrame:
    """EDI 850 Purchase Order -> one row per PO1 line item.

    Segments used: BEG (PO header), PO1 (line item), DTM (dates), REF.
    """
    rows: list[dict[str, Any]] = []
    po_number = ""
    order_date: datetime | None = None
    for seg in _split_segments(raw):
        tag = seg[0].upper()
        if tag == "BEG":
            po_number = seg[3] if len(seg) > 3 else ""
            order_date = _edi_date(seg[5]) if len(seg) > 5 else None
        elif tag == "PO1":
            # PO1*line*qty*uom*price**product-id-qualifier*product-id
            qualifiers = seg[6:] if len(seg) > 6 else []
            part_number = ""
            for i in range(0, len(qualifiers) - 1, 2):
                if qualifiers[i].upper() in {"BP", "VP", "MG", "PN"}:
                    part_number = qualifiers[i + 1]
                    break
            rows.append({
                "po_number": po_number,
                "po_line": seg[1] if len(seg) > 1 else "",
                "quantity": _to_float(seg[2]) if len(seg) > 2 else None,
                "uom": seg[3] if len(seg) > 3 else "EA",
                "unit_price_usd": _to_float(seg[4]) if len(seg) > 4 else None,
                "part_number": part_number,
                "order_date": order_date,
                "source_system": "EDI_850",
            })
    return pd.DataFrame(rows)


def parse_edi_856(raw: str) -> pd.DataFrame:
    """EDI 856 Advance Ship Notice -> shipment rows.

    The 856 hierarchy (HL shipment > order > item) is what tells us a part is
    physically moving - the earliest reliable AOG de-risking signal.
    """
    rows: list[dict[str, Any]] = []
    shipment_id = ""
    ship_date: datetime | None = None
    carrier = ""
    current_po = ""
    for seg in _split_segments(raw):
        tag = seg[0].upper()
        if tag == "BSN":
            shipment_id = seg[2] if len(seg) > 2 else ""
            ship_date = _edi_date(seg[3]) if len(seg) > 3 else None
        elif tag == "TD5":
            carrier = seg[3] if len(seg) > 3 else ""
        elif tag == "PRF":
            current_po = seg[1] if len(seg) > 1 else ""
        elif tag == "LIN":
            qualifiers = seg[2:] if len(seg) > 2 else []
            part_number = ""
            for i in range(0, len(qualifiers) - 1, 2):
                if qualifiers[i].upper() in {"BP", "VP", "PN"}:
                    part_number = qualifiers[i + 1]
                    break
            rows.append({
                "shipment_id": shipment_id,
                "po_number": current_po,
                "part_number": part_number,
                "ship_date": ship_date,
                "carrier": carrier,
                "source_system": "EDI_856",
            })
        elif tag == "SN1" and rows:
            rows[-1]["quantity_shipped"] = _to_float(seg[2]) if len(seg) > 2 else None
    return pd.DataFrame(rows)


def _edi_date(value: str) -> datetime | None:
    value = value.strip()
    for fmt in ("%Y%m%d", "%y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
