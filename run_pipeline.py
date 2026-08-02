"""End-to-end local pipeline run - no Snowflake, Spark or SAP required.

Executes the same logic the Airflow DAG orchestrates, against the synthetic
sample data, and prints the three portfolio KPIs.

    python data/generate_synthetic.py
    python run_pipeline.py
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from connectors.customs_feed_parser import customs_delay_risk, parse_customs_feed
from connectors.supplier_api_client import parse_edi_850
from connectors.wms_connector import ParquetWMSAdapter, fetch_all, stock_accuracy_report
from ml.demand_forecaster import demand_profile, forecast_next_periods, train_demand_model
from processing.aog_risk_scorer import (
    AOG_ALERT_LEAD_HOURS,
    ScoringInputs,
    aog_summary,
    early_warning_report,
    recommend_actions,
    score_aog_risk,
)
from processing.lead_time_calculator import supplier_scorecard

SAMPLE = Path(__file__).parent / "data/sample"
AS_OF = datetime(2026, 7, 31)

log = logging.getLogger("mro")


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}")


def main(out_dir: Path | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if not (SAMPLE / "parts_master.parquet").exists():
        raise SystemExit("No sample data. Run: python data/generate_synthetic.py")

    # ---------------------------------------------------------- 1. ingest
    banner("1. INGEST - five source systems")
    parts = pd.read_parquet(SAMPLE / "parts_master.parquet")
    suppliers = pd.read_parquet(SAMPLE / "suppliers.parquet")
    pos = pd.read_parquet(SAMPLE / "purchase_orders.parquet")
    work_orders = pd.read_parquet(SAMPLE / "work_orders.parquet")
    history = pd.read_parquet(SAMPLE / "demand_history.parquet")
    print(f"  ERP        {len(parts):>7,} parts, {len(pos):>7,} PO lines, {len(work_orders):>6,} work orders")

    inventory = fetch_all([ParquetWMSAdapter(SAMPLE / "inventory.parquet")])
    accuracy = stock_accuracy_report(inventory)
    print(f"  WMS        {accuracy['records']:>7,} stock records, "
          f"{accuracy['trusted_pct']}% cycle-counted within 90d")

    customs_raw = parse_customs_feed(SAMPLE / "customs_shipments.xml")
    customs = customs_delay_risk(customs_raw)
    print(f"  Customs    {len(customs_raw):>7,} shipments, {int(customs_raw['is_held'].sum()):>3} held at border")

    edi = parse_edi_850((SAMPLE / "edi_850_orders.txt").read_text(encoding="utf-8"))
    print(f"  EDI 850    {len(edi):>7,} order lines parsed")
    print("  Supplier   (REST portals mocked - see tests/test_connectors.py)")

    # ------------------------------------------------- 2. supplier scorecard
    banner("2. SUPPLIER SCORECARD - observed vs contracted lead time")
    card = supplier_scorecard(pos, suppliers, as_of=AS_OF)
    print(card[["supplier_id", "supplier_name", "n_deliveries", "base_lead_time_days",
                "p80_days", "contract_gap_days", "on_time_rate", "composite_score",
                "risk_tier"]].head(10).to_string(index=False))
    tier_mix = card["risk_tier"].value_counts().to_dict()
    print(f"\n  Tier mix: {tier_mix}")
    print(f"  Median contracted-vs-observed gap: {card['contract_gap_days'].median():.0f} days")

    # ------------------------------------------------------ 3. demand model
    banner("3. DEMAND FORECAST - gradient boosting on 36 months of MRO cycles")
    result = train_demand_model(history, parts)
    m = result.metrics
    print(f"  backend {m['backend']}  |  train {m['n_train']:,}  test {m['n_test']:,} (from {m['test_from_month']})")
    print(f"  MAE {m['mae']:.3f}   naive MAE {m['naive_mae']:.3f}   MASE {m['mase_vs_naive']:.3f}")
    verdict = "beats" if m["mase_vs_naive"] < 1 else "LOSES TO"
    print(f"  -> model {verdict} the planner's last-month heuristic "
          f"({(1 - m['mase_vs_naive']) * 100:.1f}% error reduction)")

    profile = demand_profile(history)
    print(f"\n  Demand pattern mix: {profile['pattern'].value_counts().to_dict()}")
    print(f"  Confidently forecastable: {profile['forecastable'].sum()}/{len(profile)} parts")

    forecast = forecast_next_periods(history, result.model, parts, horizon=3)
    demand = forecast[forecast["horizon_step"] == 1][["part_number", "forecast_qty"]]

    # -------------------------------------------------------- 4. AOG risk
    banner("4. AOG RISK SCORING - criticality x availability x supply timing")
    open_orders = pos[pos["status"] == "OPEN"].rename(columns={"promised_date": "expected_date"})
    scored = score_aog_risk(ScoringInputs(
        parts=parts,
        inventory=inventory,
        open_orders=open_orders,
        demand=demand,
        supplier_scores=card,
        customs_risk=customs,
        open_work_orders=work_orders[work_orders["status"] == "OPEN"],
    ), as_of=AS_OF)

    print(recommend_actions(scored, 12).to_string(index=False))
    summary = aog_summary(scored)
    print(f"\n  Risk band mix: {scored['risk_band'].value_counts().to_dict()}")
    print(f"  Primary drivers: {scored[scored['aog_risk_score'] >= 60]['primary_driver'].value_counts().to_dict()}")

    # ------------------------------------------------------------- 5. KPIs
    banner("5. BUSINESS KPIs")
    horizon = int(scored["horizon_days"].iloc[0])
    at_risk = scored[scored["aog_risk_score"] >= 60]
    early = early_warning_report(scored, horizon)

    print("  AOG early warning")
    print(f"    {len(at_risk)} parts flagged at/above HIGH on a {horizon}-day horizon")
    print(f"    {len(early)} parts still have stock today but burn down to zero "
          f"inside the horizon")
    if not early.empty:
        median_hours = early["warning_hours"].median()
        over_72h = int((early["warning_hours"] >= AOG_ALERT_LEAD_HOURS).sum())
        print(f"    median warning: {median_hours:,.0f} hours "
              f"({median_hours / 24:.0f} days) vs 0 for reactive discovery")
        print(f"    {over_72h}/{len(early)} flagged with >= 72h notice "
              f"({100 * over_72h / len(early):.0f}%)")
        crit = early[early["criticality"] == "NO_GO"]
        print(f"    {len(crit)} of them are NO_GO parts - each one a potential grounding")

    print("\n  Supplier on-time delivery")
    print(f"    {len(card)} suppliers scored automatically from {len(pos):,} PO lines")
    print(f"    {int((card['risk_tier'] == 'CRITICAL').sum())} flagged CRITICAL for escalation")
    print(f"    fleet-wide OTD: {100 * card['on_time_rate'].mean():.1f}%")

    print("\n  Parts availability accuracy")
    print(f"    {accuracy['trusted_pct']}% of stock records cycle-counted within 90 days")
    print(f"    {accuracy['stale_pct']}% stale (the gap manual spreadsheets carry silently)")
    unified = scored["part_number"].nunique()
    print(f"    {unified:,} parts with a unified cross-system view "
          f"(ERP + WMS + supplier + customs)")

    print(f"\n  Value at risk: ${summary['value_at_risk_usd']:,.0f} "
          f"across {summary['critical_count'] + summary['high_count']} at-risk parts")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(out_dir / "aog_risk.parquet", index=False)
        card.to_parquet(out_dir / "supplier_scorecard.parquet", index=False)
        forecast.to_parquet(out_dir / "demand_forecast.parquet", index=False)
        print(f"\n  Outputs written to {out_dir}")

    return {"aog": summary, "model": m, "accuracy": accuracy}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the MRO pipeline end to end locally")
    ap.add_argument("--out", type=Path, default=None, help="write outputs to this directory")
    args = ap.parse_args()
    main(args.out)
