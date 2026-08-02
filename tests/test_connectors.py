"""Connector tests: EDI parsing, customs XML, WMS normalization, SAP mapping."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from connectors.customs_feed_parser import customs_delay_risk, parse_customs_feed
from connectors.erp_sap_extractor import SAPConnection, SAPExtractor, strip_matnr
from connectors.supplier_api_client import (
    SupplierEndpoint,
    SupplierPortalClient,
    parse_edi_850,
    parse_edi_856,
)
from connectors.wms_connector import normalize_inventory, stock_accuracy_report

# --------------------------------------------------------------------- EDI

EDI_850 = """ST*850*0001
BEG*00*NE*PO-000123**20260715
PO1*1*4*EA*1250.50**BP*HTL-4471-002
PO1*2*10*EA*89.99**VP*NAS1234-5
SE*5*0001"""


def test_parse_edi_850_extracts_lines():
    df = parse_edi_850(EDI_850)
    assert len(df) == 2
    assert df.loc[0, "part_number"] == "HTL-4471-002"
    assert df.loc[0, "quantity"] == 4
    assert df.loc[0, "unit_price_usd"] == 1250.50
    assert df.loc[0, "po_number"] == "PO-000123"
    assert df.loc[0, "order_date"] == datetime(2026, 7, 15)
    # second line uses the VP qualifier rather than BP
    assert df.loc[1, "part_number"] == "NAS1234-5"


def test_parse_edi_850_handles_tilde_terminated():
    """Real X12 arrives as one line with ~ terminators, not newlines."""
    single_line = EDI_850.replace("\n", "~")
    assert parse_edi_850(single_line).equals(parse_edi_850(EDI_850))


def test_parse_edi_850_ignores_unknown_qualifier():
    """An unrecognized product-id qualifier must not crash or mis-assign."""
    bad = "BEG*00*NE*PO-9**20260101\nPO1*1*2*EA*10**ZZ*SOMETHING"
    df = parse_edi_850(bad)
    assert len(df) == 1
    assert df.loc[0, "part_number"] == ""


EDI_856 = """ST*856*0002
BSN*00*SHP-778*20260720*1030
TD5**2*FEDEX
PRF*PO-000123
LIN**BP*HTL-4471-002
SN1**4*EA"""


def test_parse_edi_856_links_shipment_to_po():
    df = parse_edi_856(EDI_856)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["shipment_id"] == "SHP-778"
    assert row["po_number"] == "PO-000123"
    assert row["part_number"] == "HTL-4471-002"
    assert row["carrier"] == "FEDEX"
    assert row["quantity_shipped"] == 4


# ----------------------------------------------------------------- customs

CUSTOMS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<CustomsManifest generated="2026-07-31">
  <Shipment awb="AWB123" poRef="PO-1">
    <PartNumber>HTL-4471-002</PartNumber>
    <Quantity>4</Quantity>
    <ClearanceStatus>HELD</ClearanceStatus>
    <EstimatedClearanceDays>10</EstimatedClearanceDays>
    <HoldReason>Missing EASA Form 1</HoldReason>
  </Shipment>
  <Shipment awb="AWB124" poRef="PO-2">
    <PartNumber>NAS1234-5</PartNumber>
    <Quantity>10</Quantity>
    <ClearanceStatus>CLEARED</ClearanceStatus>
    <EstimatedClearanceDays>1</EstimatedClearanceDays>
  </Shipment>
</CustomsManifest>
"""


@pytest.fixture
def customs_file(tmp_path):
    p = tmp_path / "manifest.xml"
    p.write_text(CUSTOMS_XML, encoding="utf-8")
    return p


def test_parse_customs_feed(customs_file):
    df = parse_customs_feed(customs_file)
    assert len(df) == 2
    held = df[df["awb"] == "AWB123"].iloc[0]
    assert held["is_held"]
    assert held["hold_reason"] == "Missing EASA Form 1"
    assert held["part_number"] == "HTL-4471-002"


def test_customs_held_shipment_is_max_risk(customs_file):
    risk = customs_delay_risk(parse_customs_feed(customs_file))
    held = risk[risk["po_number"] == "PO-1"].iloc[0]
    cleared = risk[risk["po_number"] == "PO-2"].iloc[0]
    assert held["customs_risk"] == 1.0
    # A cleared shipment carries no customs risk regardless of its ETA field.
    assert cleared["customs_risk"] == 0.0


def test_parse_customs_handles_namespaced_feed(tmp_path):
    """Same broker, alternate day, default namespace present."""
    ns_xml = CUSTOMS_XML.replace(
        "<CustomsManifest ", '<CustomsManifest xmlns="urn:broker:manifest" '
    )
    p = tmp_path / "ns.xml"
    p.write_text(ns_xml, encoding="utf-8")
    df = parse_customs_feed(p)
    assert len(df) == 2
    assert df.iloc[0]["part_number"] == "HTL-4471-002"


def test_unknown_clearance_status_defaults_safely(tmp_path):
    weird = CUSTOMS_XML.replace("<ClearanceStatus>HELD</ClearanceStatus>",
                                "<ClearanceStatus>SCHRODINGER</ClearanceStatus>")
    p = tmp_path / "weird.xml"
    p.write_text(weird, encoding="utf-8")
    df = parse_customs_feed(p)
    assert df.iloc[0]["clearance_status"] == "IN_TRANSIT"


def test_empty_manifest_returns_empty_frame(tmp_path):
    p = tmp_path / "empty.xml"
    p.write_text('<?xml version="1.0"?><CustomsManifest></CustomsManifest>', encoding="utf-8")
    assert parse_customs_feed(p).empty
    assert customs_delay_risk(parse_customs_feed(p)).empty


# --------------------------------------------------------------------- WMS


def test_normalize_inventory_subtracts_reserved_and_quarantine():
    raw = pd.DataFrame({
        "warehouse_id": ["W1"], "part_number": ["P1"],
        "qty_on_hand": [10], "qty_reserved": [3], "qty_quarantine": [2],
        "condition_code": ["NE"],
    })
    out = normalize_inventory(raw, "TEST")
    assert out.loc[0, "qty_available"] == 5


def test_normalize_inventory_zeroes_non_issuable_condition():
    """As-removed stock is physically present but cannot be issued."""
    raw = pd.DataFrame({
        "warehouse_id": ["W1"], "part_number": ["P1"],
        "qty_on_hand": [10], "qty_reserved": [0], "qty_quarantine": [0],
        "condition_code": ["AR"],
    })
    assert normalize_inventory(raw, "TEST").loc[0, "qty_available"] == 0


def test_normalize_inventory_zeroes_expired_stock():
    raw = pd.DataFrame({
        "warehouse_id": ["W1"], "part_number": ["P1"],
        "qty_on_hand": [10], "qty_reserved": [0], "qty_quarantine": [0],
        "condition_code": ["NE"],
        "expiry_date": [datetime.utcnow() - timedelta(days=1)],
    })
    assert normalize_inventory(raw, "TEST").loc[0, "qty_available"] == 0


def test_normalize_inventory_never_returns_negative():
    """Reserved exceeding on-hand is a real WMS data bug; clamp, don't propagate."""
    raw = pd.DataFrame({
        "warehouse_id": ["W1"], "part_number": ["P1"],
        "qty_on_hand": [2], "qty_reserved": [9], "qty_quarantine": [0],
        "condition_code": ["NE"],
    })
    assert normalize_inventory(raw, "TEST").loc[0, "qty_available"] == 0


def test_normalize_inventory_accepts_camelcase_payload():
    """Cloud WMS sends camelCase JSON keys."""
    raw = pd.DataFrame({
        "warehouseId": ["W1"], "partNumber": ["P1"],
        "quantityOnHand": [5], "quantityReserved": [1],
        "quantityQuarantine": [0], "condition": ["ne"],
    })
    out = normalize_inventory(raw, "TEST")
    assert out.loc[0, "qty_available"] == 4
    assert out.loc[0, "condition_code"] == "NE"


def test_stock_accuracy_report_flags_stale():
    inv = pd.DataFrame({
        "last_counted": [
            pd.Timestamp(datetime.utcnow()),
            pd.Timestamp(datetime.utcnow() - timedelta(days=200)),
        ]
    })
    rep = stock_accuracy_report(inv)
    assert rep["records"] == 2
    assert rep["stale_pct"] == 50.0


# --------------------------------------------------------------------- SAP


def test_strip_matnr_removes_sap_padding():
    assert strip_matnr("000000000000123456") == "123456"


def test_strip_matnr_preserves_alphanumeric_part_numbers():
    """MRO part numbers are alphanumeric - stripping zeros would corrupt them."""
    assert strip_matnr("HTL-4471-002") == "HTL-4471-002"
    assert strip_matnr("00HTL4471002000000") == "00HTL4471002000000"


def test_sap_file_mode_reads_extract(tmp_path):
    (tmp_path / "mara.csv").write_text(
        "MATNR,MAKTX,MTART,MATKL,NTGEW,MHDHB\n"
        "000000000000123456,Hydraulic pump,ERSA,MRO,12.5,24\n",
        encoding="utf-8",
    )
    ex = SAPExtractor(SAPConnection(extract_dir=tmp_path))
    df = ex.extract_material_master()
    assert len(df) == 1
    assert df.loc[0, "part_number"] == "123456"
    assert df.loc[0, "description"] == "Hydraulic pump"
    assert df.loc[0, "shelf_life_months"] == 24


def test_sap_file_mode_rejects_incomplete_extract(tmp_path):
    (tmp_path / "mara.csv").write_text("MATNR,MAKTX\nX,Y\n", encoding="utf-8")
    ex = SAPExtractor(SAPConnection(extract_dir=tmp_path))
    with pytest.raises(ValueError, match="missing SAP fields"):
        ex.extract_material_master()


def test_sap_connection_mode_detection(tmp_path):
    assert SAPConnection(extract_dir=tmp_path).mode == "FILE"
    assert SAPConnection(ashost="sap.internal").mode == "RFC"


# ---------------------------------------------------------- supplier REST


class _FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Records calls and replays queued responses."""

    def __init__(self, get_responses):
        self.get_responses = list(get_responses)
        self.get_calls = []
        self.post_calls = 0

    def post(self, url, data=None, timeout=None):
        self.post_calls += 1
        return _FakeResponse({"access_token": "tok", "expires_in": 3600})

    def get(self, url, params=None, headers=None, timeout=None):
        self.get_calls.append((url, params))
        return self.get_responses.pop(0)


def _endpoint():
    return SupplierEndpoint(
        supplier_id="SUP-1", base_url="https://portal.example.com",
        client_id="id", client_secret="secret", page_size=2,
    )


def test_supplier_client_paginates_and_normalizes():
    session = _FakeSession([
        _FakeResponse({"items": [
            {"partNumber": "P1", "quantityAvailable": 3, "leadTimeDays": 45, "unitPrice": 100},
            {"partNumber": "P2", "quantityAvailable": 0, "leadTimeDays": 120, "unitPrice": 50},
        ], "totalPages": 2}),
        _FakeResponse({"items": [
            {"partNumber": "P3", "quantityAvailable": 9, "leadTimeDays": 30, "unitPrice": 10},
        ], "totalPages": 2}),
    ])
    client = SupplierPortalClient(_endpoint(), session=session)
    df = client.fetch_part_availability()

    assert len(df) == 3
    assert list(df["part_number"]) == ["P1", "P2", "P3"]
    assert df["supplier_id"].unique().tolist() == ["SUP-1"]
    assert session.post_calls == 1  # token fetched once and reused across pages


def test_supplier_client_reuses_token_across_calls():
    session = _FakeSession([
        _FakeResponse({"items": [], "totalPages": 1}),
        _FakeResponse({"items": [], "totalPages": 1}),
    ])
    client = SupplierPortalClient(_endpoint(), session=session)
    client.fetch_part_availability()
    client.fetch_part_availability()
    assert session.post_calls == 1


def test_supplier_client_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("connectors.supplier_api_client.time.sleep", lambda s: None)
    session = _FakeSession([
        _FakeResponse({}, status=429, headers={"Retry-After": "0"}),
        _FakeResponse({"items": [{"partNumber": "P1", "quantityAvailable": 1}], "totalPages": 1}),
    ])
    client = SupplierPortalClient(_endpoint(), session=session)
    df = client.fetch_part_availability()
    assert len(df) == 1


def test_supplier_client_stops_on_short_page_without_total():
    """Portals that omit totalPages must still terminate."""
    session = _FakeSession([
        _FakeResponse({"items": [{"partNumber": "P1", "quantityAvailable": 1}]}),
    ])
    client = SupplierPortalClient(_endpoint(), session=session)
    assert len(client.fetch_part_availability()) == 1
