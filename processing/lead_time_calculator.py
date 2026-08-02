"""Dynamic supplier lead-time calculation and on-time delivery scorecards.

The contracted lead time on a supplier agreement is a negotiating artefact, not
a planning input. What planners need is the *observed* distribution, weighted
toward recent behaviour, expressed as a percentile they can commit to.

Key outputs
-----------
effective_lead_time(P50/P80/P95)  - what to actually plan with
on_time_delivery_rate             - the supplier scorecard headline
lead_time_volatility              - drives safety-stock sizing
trend_days_per_year               - is this supplier getting worse?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Deliveries older than this stop reflecting current supplier capability.
DEFAULT_LOOKBACK_DAYS = 730
# Exponential recency weighting: a delivery this old counts half as much.
RECENCY_HALFLIFE_DAYS = 180
# Aerospace convention: on-time means within the promise window, and early
# delivery beyond this is also a miss (it consumes bonded storage).
ON_TIME_LATE_TOLERANCE_DAYS = 0
ON_TIME_EARLY_TOLERANCE_DAYS = 30
MIN_DELIVERIES_FOR_STATS = 5


@dataclass
class LeadTimeProfile:
    supplier_id: str
    part_number: str | None
    n_deliveries: int
    p50_days: float
    p80_days: float
    p95_days: float
    mean_days: float
    volatility_days: float
    on_time_rate: float
    avg_slip_days: float
    trend_days_per_year: float
    confidence: str


def _recency_weights(order_dates: pd.Series, as_of: pd.Timestamp) -> np.ndarray:
    age_days = (as_of - pd.to_datetime(order_dates)).dt.days.clip(lower=0).to_numpy(dtype=float)
    return 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile. numpy has no built-in that accepts weights."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return float(np.interp(q, cum, v))


def prepare_deliveries(purchase_orders: pd.DataFrame, as_of: datetime | None = None) -> pd.DataFrame:
    """Filter to completed deliveries and compute realized lead time + slip."""
    as_of_ts = pd.Timestamp(as_of or datetime.utcnow())
    df = purchase_orders.copy()
    for col in ("order_date", "promised_date", "actual_delivery_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df[df["actual_delivery_date"].notna() & df["order_date"].notna()]
    df = df[df["order_date"] >= as_of_ts - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS)]

    df["actual_lead_time_days"] = (df["actual_delivery_date"] - df["order_date"]).dt.days
    df["slip_days"] = (df["actual_delivery_date"] - df["promised_date"]).dt.days
    # Guard against ERP data entry errors producing negative lead times.
    bad = df["actual_lead_time_days"] < 0
    if bad.any():
        log.warning("dropping %d POs with negative lead time", int(bad.sum()))
        df = df[~bad]
    df["on_time"] = df["slip_days"].between(-ON_TIME_EARLY_TOLERANCE_DAYS, ON_TIME_LATE_TOLERANCE_DAYS)
    return df


def _profile_group(g: pd.DataFrame, supplier_id: str, part_number: str | None, as_of: pd.Timestamp) -> LeadTimeProfile:
    values = g["actual_lead_time_days"].to_numpy(dtype=float)
    weights = _recency_weights(g["order_date"], as_of)
    n = len(values)

    if n >= 3:
        # Days of lead-time drift per year: positive means degrading supplier.
        x = (pd.to_datetime(g["order_date"]) - as_of).dt.days.to_numpy(dtype=float) / 365.0
        slope = float(np.polyfit(x, values, 1)[0])
    else:
        slope = 0.0

    confidence = "HIGH" if n >= 20 else "MEDIUM" if n >= MIN_DELIVERIES_FOR_STATS else "LOW"
    return LeadTimeProfile(
        supplier_id=supplier_id,
        part_number=part_number,
        n_deliveries=n,
        p50_days=_weighted_quantile(values, weights, 0.50),
        p80_days=_weighted_quantile(values, weights, 0.80),
        p95_days=_weighted_quantile(values, weights, 0.95),
        mean_days=float(np.average(values, weights=weights)),
        volatility_days=float(np.sqrt(np.average((values - np.average(values, weights=weights)) ** 2, weights=weights))),
        on_time_rate=float(np.average(g["on_time"].to_numpy(dtype=float), weights=weights)),
        avg_slip_days=float(np.average(g["slip_days"].fillna(0).to_numpy(dtype=float), weights=weights)),
        trend_days_per_year=slope,
        confidence=confidence,
    )


def supplier_lead_time_profiles(purchase_orders: pd.DataFrame, as_of: datetime | None = None) -> pd.DataFrame:
    """One profile row per supplier."""
    as_of_ts = pd.Timestamp(as_of or datetime.utcnow())
    df = prepare_deliveries(purchase_orders, as_of)
    profiles = [
        _profile_group(g, sup, None, as_of_ts)
        for sup, g in df.groupby("supplier_id", sort=False)
    ]
    return pd.DataFrame([p.__dict__ for p in profiles])


def supplier_part_lead_times(purchase_orders: pd.DataFrame, as_of: datetime | None = None,
                             min_deliveries: int = MIN_DELIVERIES_FOR_STATS) -> pd.DataFrame:
    """Supplier x part lead times, falling back to the supplier-level profile
    when a specific part has too little history to stand on its own."""
    as_of_ts = pd.Timestamp(as_of or datetime.utcnow())
    df = prepare_deliveries(purchase_orders, as_of)
    supplier_level = supplier_lead_time_profiles(purchase_orders, as_of).set_index("supplier_id")

    rows: list[dict] = []
    for (sup, part), g in df.groupby(["supplier_id", "part_number"], sort=False):
        if len(g) >= min_deliveries:
            rows.append(_profile_group(g, sup, part, as_of_ts).__dict__)
        elif sup in supplier_level.index:
            fallback = supplier_level.loc[sup].to_dict()
            fallback.update({"supplier_id": sup, "part_number": part, "confidence": "FALLBACK"})
            rows.append(fallback)
    return pd.DataFrame(rows)


def supplier_scorecard(purchase_orders: pd.DataFrame, suppliers: pd.DataFrame | None = None,
                       as_of: datetime | None = None) -> pd.DataFrame:
    """Ranked supplier scorecard - the Power BI supplier page feeds off this.

    Composite score (0-100) blends the three things a supply chain manager
    actually escalates on: did it arrive on time, how predictable is it, and
    is the relationship trending the wrong way.
    """
    prof = supplier_lead_time_profiles(purchase_orders, as_of)
    if prof.empty:
        return prof

    otd = prof["on_time_rate"] * 100
    # Volatility relative to the supplier's own lead time - 20 days of variance
    # is disastrous on a 30-day part and tolerable on a 400-day one.
    rel_vol = (prof["volatility_days"] / prof["p50_days"].clip(lower=1)).clip(0, 1)
    consistency = (1 - rel_vol) * 100
    # +/- 30 days per year of drift maps onto the full penalty range.
    trend_score = (1 - (prof["trend_days_per_year"].clip(-30, 30) + 30) / 60) * 100

    prof["otd_score"] = otd.round(1)
    prof["consistency_score"] = consistency.round(1)
    prof["trend_score"] = trend_score.round(1)
    prof["composite_score"] = (0.55 * otd + 0.30 * consistency + 0.15 * trend_score).round(1)
    prof["risk_tier"] = pd.cut(
        prof["composite_score"],
        bins=[-0.1, 50, 70, 85, 100.1],
        labels=["CRITICAL", "WATCH", "ACCEPTABLE", "PREFERRED"],
    )

    if suppliers is not None:
        prof = prof.merge(
            suppliers[["supplier_id", "supplier_name", "country", "base_lead_time_days"]],
            on="supplier_id", how="left",
        )
        # Contracted vs observed - the gap planners are unknowingly absorbing.
        prof["contract_gap_days"] = (prof["p80_days"] - prof["base_lead_time_days"]).round(1)

    return prof.sort_values("composite_score")


if __name__ == "__main__":
    from pathlib import Path

    sample = Path(__file__).parents[1] / "data/sample"
    pos = pd.read_parquet(sample / "purchase_orders.parquet")
    sup = pd.read_parquet(sample / "suppliers.parquet")
    card = supplier_scorecard(pos, sup, as_of=datetime(2026, 7, 31))
    cols = ["supplier_id", "supplier_name", "n_deliveries", "p80_days", "on_time_rate", "composite_score", "risk_tier"]
    print(card[cols].head(15).to_string(index=False))
