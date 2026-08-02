"""AOG risk rule engine: criticality x availability x supply timing.

The core deliverable. Every open demand for a part is scored 0-100 for the
risk that it grounds an aircraft, using five weighted factors:

  1. Criticality      - is the part NO-GO under the MEL? (a NO-GO shortage
                        grounds the aircraft; a GO shortage is paperwork)
  2. Coverage         - available stock vs projected demand over the horizon
  3. Replenishment    - will open POs land before stock runs out?
  4. Supplier         - is the incoming supplier reliable? (lead-time module)
  5. Customs          - is the inbound shipment stuck at a border?

The 72-hour-earlier KPI comes from factor 3 and 5: reactive discovery happens
when a tech pulls a bin and finds it empty. This scores the *projection*, so
the alert fires when the burn-down crosses the horizon, not on stockout day.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Weights sum to 1.0. Criticality dominates because a GO part shortage is
# never an AOG regardless of how bad the supply picture looks.
WEIGHTS = {
    "criticality": 0.30,
    "coverage": 0.28,
    "replenishment": 0.20,
    "supplier": 0.12,
    "customs": 0.10,
}

# MEL-derived: how directly a shortage grounds the aircraft.
CRITICALITY_SCORE = {"NO_GO": 1.0, "GO_IF": 0.55, "GO": 0.15}

RISK_BANDS = [(80, "CRITICAL"), (60, "HIGH"), (40, "MEDIUM"), (0, "LOW")]

# Planning horizon. 45 days covers the window in which a planner can still
# act (expedite, borrow, re-source) before a shortage becomes an AOG.
DEFAULT_HORIZON_DAYS = 45
# Alert lead time the KPI claims.
AOG_ALERT_LEAD_HOURS = 72


@dataclass
class ScoringInputs:
    parts: pd.DataFrame            # part_number, criticality, ata_chapter, unit_cost_usd
    inventory: pd.DataFrame        # part_number, qty_available (per warehouse)
    open_orders: pd.DataFrame      # part_number, quantity, expected_date, supplier_id
    demand: pd.DataFrame           # part_number, monthly_demand
    supplier_scores: pd.DataFrame | None = None   # supplier_id, on_time_rate, p80_days
    customs_risk: pd.DataFrame | None = None      # part_number, customs_risk
    open_work_orders: pd.DataFrame | None = None  # part_number, qty_required, wo_type


def _band(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "LOW"


def coverage_factor(available: pd.Series, daily_demand: pd.Series, horizon_days: int) -> pd.Series:
    """1.0 = no stock at all, 0.0 = fully covered through the horizon.

    Expressed as days-of-cover rather than a raw ratio so parts with very
    different demand rates compare on the same scale.
    """
    demand_over_horizon = daily_demand * horizon_days
    # A part with zero forecast demand cannot cause a shortage from demand
    # alone - but it can still be a NO-GO part with zero stock, which the
    # criticality factor handles.
    shortfall = (demand_over_horizon - available).clip(lower=0)
    return (shortfall / demand_over_horizon.where(demand_over_horizon > 0)).fillna(0).clip(0, 1)


def replenishment_factor(
    open_orders: pd.DataFrame, available: pd.Series, daily_demand: pd.Series,
    as_of: pd.Timestamp, horizon_days: int,
) -> pd.Series:
    """Risk that replenishment lands after stock runs out.

    Days-of-cover vs days-to-next-receipt. If the next PO arrives after the
    burn-down hits zero, there is an exposure window - that window, normalized
    by the horizon, is the factor.
    """
    days_of_cover = (available / daily_demand.where(daily_demand > 0)).fillna(np.inf)

    if open_orders.empty:
        next_receipt = pd.Series(np.inf, index=available.index)
    else:
        oo = open_orders.copy()
        oo["expected_date"] = pd.to_datetime(oo["expected_date"], errors="coerce")
        oo = oo[oo["expected_date"].notna()]
        earliest = oo.groupby("part_number")["expected_date"].min()
        next_receipt = ((earliest - as_of).dt.days).reindex(available.index).fillna(np.inf)

    exposure_days = (next_receipt - days_of_cover).clip(lower=0)
    # Parts already covered through the horizon carry no replenishment risk.
    exposure_days = exposure_days.where(days_of_cover < horizon_days, 0)
    return (exposure_days / horizon_days).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0, 1)


def score_aog_risk(inp: ScoringInputs, as_of: datetime | None = None,
                   horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    """Score every part in the catalog for AOG risk."""
    as_of_ts = pd.Timestamp(as_of or datetime.utcnow())

    base = inp.parts[["part_number", "criticality"]].drop_duplicates("part_number").set_index("part_number")
    if "ata_chapter" in inp.parts:
        base["ata_chapter"] = inp.parts.set_index("part_number")["ata_chapter"].reindex(base.index)
    if "unit_cost_usd" in inp.parts:
        base["unit_cost_usd"] = inp.parts.set_index("part_number")["unit_cost_usd"].reindex(base.index)

    # --- availability across all warehouses -------------------------------
    if inp.inventory.empty:
        available = pd.Series(0, index=base.index, dtype=float)
    else:
        available = inp.inventory.groupby("part_number")["qty_available"].sum().reindex(base.index).fillna(0)
    base["qty_available"] = available

    # --- demand rate ------------------------------------------------------
    demand_col = "forecast_qty" if "forecast_qty" in inp.demand.columns else "monthly_demand"
    monthly = inp.demand.groupby("part_number")[demand_col].mean().reindex(base.index).fillna(0)
    daily_demand = monthly / 30.0
    base["monthly_demand"] = monthly
    base["days_of_cover"] = (available / daily_demand.where(daily_demand > 0)).replace(np.inf, 9999).fillna(9999).round(1)

    # --- open work order pull (firm demand beats forecast demand) ---------
    if inp.open_work_orders is not None and not inp.open_work_orders.empty:
        wo = inp.open_work_orders
        firm = wo.groupby("part_number")["qty_required"].sum().reindex(base.index).fillna(0)
        base["firm_wo_demand"] = firm
        aog_wo = (
            wo[wo.get("wo_type", pd.Series(dtype=str)).eq("AOG")]
            .groupby("part_number")["qty_required"].sum().reindex(base.index).fillna(0)
        )
        base["aog_wo_demand"] = aog_wo
    else:
        base["firm_wo_demand"] = 0.0
        base["aog_wo_demand"] = 0.0

    # --- the five factors -------------------------------------------------
    f_crit = base["criticality"].map(CRITICALITY_SCORE).fillna(0.5)
    f_cov = coverage_factor(available, daily_demand, horizon_days)
    f_repl = replenishment_factor(inp.open_orders, available, daily_demand, as_of_ts, horizon_days)

    if inp.supplier_scores is not None and not inp.supplier_scores.empty and not inp.open_orders.empty:
        # Attribute each part's supplier risk to the supplier on its next PO.
        oo = inp.open_orders.copy()
        oo["expected_date"] = pd.to_datetime(oo["expected_date"], errors="coerce")
        next_sup = oo.sort_values("expected_date").drop_duplicates("part_number").set_index("part_number")["supplier_id"]
        sup_risk = (1 - inp.supplier_scores.set_index("supplier_id")["on_time_rate"]).clip(0, 1)
        f_sup = next_sup.map(sup_risk).reindex(base.index).fillna(0.5)
    else:
        f_sup = pd.Series(0.5, index=base.index)  # unknown supplier = neutral

    if inp.customs_risk is not None and not inp.customs_risk.empty:
        f_cust = inp.customs_risk.groupby("part_number")["customs_risk"].max().reindex(base.index).fillna(0)
    else:
        f_cust = pd.Series(0.0, index=base.index)

    raw = (
        WEIGHTS["criticality"] * f_crit
        + WEIGHTS["coverage"] * f_cov
        + WEIGHTS["replenishment"] * f_repl
        + WEIGHTS["supplier"] * f_sup
        + WEIGHTS["customs"] * f_cust
    )
    base["aog_risk_score"] = (raw * 100).round(1)

    # An open AOG work order is not a prediction - the aircraft is already
    # down. Force those to the top so the dashboard never buries them.
    base.loc[(base["aog_wo_demand"] > 0) & (base["qty_available"] < base["aog_wo_demand"]), "aog_risk_score"] = 100.0

    base["risk_band"] = base["aog_risk_score"].map(_band)
    base["factor_criticality"] = f_crit.round(3)
    base["factor_coverage"] = f_cov.round(3)
    base["factor_replenishment"] = f_repl.round(3)
    base["factor_supplier"] = f_sup.round(3)
    base["factor_customs"] = f_cust.round(3)
    base["primary_driver"] = _primary_driver(f_crit, f_cov, f_repl, f_sup, f_cust)
    base["scored_at"] = as_of_ts
    base["horizon_days"] = horizon_days

    return base.reset_index().sort_values("aog_risk_score", ascending=False)


def _primary_driver(*factors: pd.Series) -> pd.Series:
    """Which weighted factor contributes most - drives the 'why' column in
    Power BI so planners get an action, not just a number."""
    names = list(WEIGHTS)
    contrib = pd.DataFrame({n: f * WEIGHTS[n] for n, f in zip(names, factors)})
    return contrib.idxmax(axis=1)


def recommend_actions(scored: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Turn scores into the specific action a planner should take."""
    actions = {
        "criticality": "Review MEL coverage; confirm alternate/PMA part approved",
        "coverage": "Raise replenishment PO; check sister-station stock for loan",
        "replenishment": "Expedite open PO with supplier; request partial shipment",
        "supplier": "Escalate to supplier quality; dual-source this part number",
        "customs": "Chase broker on clearance; verify EASA Form 1 / FAA 8130-3 paperwork",
    }
    top = scored.head(top_n).copy()
    top["recommended_action"] = top["primary_driver"].map(actions)
    top["alert_lead_hours"] = AOG_ALERT_LEAD_HOURS
    cols = ["part_number", "criticality", "aog_risk_score", "risk_band", "qty_available",
            "days_of_cover", "primary_driver", "recommended_action"]
    return top[cols]


def early_warning_report(scored: pd.DataFrame, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    """Parts that still have stock today but burn down to zero inside the horizon.

    This is the population the 'identified 72 hours earlier' KPI is actually
    about. Reactive discovery finds a part when a technician opens an empty
    bin - by definition, zero hours of warning. Every row here was flagged
    while stock was still on the shelf, and `warning_hours` is the lead time
    the pipeline bought the planner.
    """
    df = scored[(scored["qty_available"] > 0) & (scored["days_of_cover"] < horizon_days)].copy()
    if df.empty:
        return df.assign(warning_hours=[], projected_stockout_date=[])
    df["warning_hours"] = (df["days_of_cover"] * 24).round(0)
    df["projected_stockout_date"] = (
        pd.to_datetime(df["scored_at"]) + pd.to_timedelta(df["days_of_cover"], unit="D")
    ).dt.date
    return df.sort_values("days_of_cover")


def aog_summary(scored: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the dashboard tiles."""
    total = len(scored)
    return {
        "parts_scored": total,
        "critical_count": int((scored["risk_band"] == "CRITICAL").sum()),
        "high_count": int((scored["risk_band"] == "HIGH").sum()),
        "no_go_at_risk": int(((scored["criticality"] == "NO_GO") & (scored["aog_risk_score"] >= 60)).sum()),
        "pct_at_risk": round(100 * (scored["aog_risk_score"] >= 60).mean(), 1) if total else 0.0,
        "value_at_risk_usd": float(
            scored.loc[scored["aog_risk_score"] >= 60, "unit_cost_usd"].sum()
        ) if "unit_cost_usd" in scored else 0.0,
    }


if __name__ == "__main__":
    from pathlib import Path

    sample = Path(__file__).parents[1] / "data/sample"
    parts = pd.read_parquet(sample / "parts_master.parquet")
    inv = pd.read_parquet(sample / "inventory.parquet")
    inv["qty_available"] = (inv["qty_on_hand"] - inv["qty_reserved"]).clip(lower=0)
    pos = pd.read_parquet(sample / "purchase_orders.parquet")
    wos = pd.read_parquet(sample / "work_orders.parquet")
    hist = pd.read_parquet(sample / "demand_history.parquet")

    open_orders = pos[pos["status"] == "OPEN"].rename(columns={"promised_date": "expected_date"})
    demand = hist.groupby("part_number", as_index=False)["demand_qty"].mean().rename(
        columns={"demand_qty": "monthly_demand"})

    scored = score_aog_risk(ScoringInputs(
        parts=parts, inventory=inv, open_orders=open_orders, demand=demand,
        open_work_orders=wos[wos["status"] == "OPEN"],
    ), as_of=datetime(2026, 7, 31))

    print(recommend_actions(scored, 15).to_string(index=False))
    print()
    for k, v in aog_summary(scored).items():
        print(f"{k:>20}: {v}")
