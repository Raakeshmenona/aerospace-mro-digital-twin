"""Tests for lead-time analytics, AOG scoring and demand forecasting."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.demand_forecaster import (
    build_features,
    classify_demand_pattern,
    demand_profile,
    mase,
    train_demand_model,
)
from processing.aog_risk_scorer import (
    CRITICALITY_SCORE,
    WEIGHTS,
    ScoringInputs,
    aog_summary,
    coverage_factor,
    early_warning_report,
    score_aog_risk,
)
from processing.lead_time_calculator import (
    _weighted_quantile,
    prepare_deliveries,
    supplier_lead_time_profiles,
    supplier_scorecard,
)

AS_OF = datetime(2026, 7, 31)


def make_pos(supplier_id="SUP-1", n=40, promised=100, slip=0, part="P1"):
    """PO history with a controllable slip so assertions are exact."""
    order_dates = [AS_OF - timedelta(days=200 + i * 5) for i in range(n)]
    return pd.DataFrame({
        "po_number": [f"PO-{i}" for i in range(n)],
        "part_number": part,
        "supplier_id": supplier_id,
        "order_date": order_dates,
        "promised_date": [d + timedelta(days=promised) for d in order_dates],
        "actual_delivery_date": [d + timedelta(days=promised + slip) for d in order_dates],
        "quantity": 1,
        "unit_price_usd": 100.0,
        "status": "DELIVERED",
    })


# ------------------------------------------------------------- lead times


def test_weighted_quantile_matches_unweighted_when_uniform():
    values = np.arange(1, 101, dtype=float)
    weights = np.ones_like(values)
    assert _weighted_quantile(values, weights, 0.5) == pytest.approx(50.5, abs=1.0)


def test_on_time_supplier_scores_full_otd():
    prof = supplier_lead_time_profiles(make_pos(slip=0), as_of=AS_OF)
    assert prof.loc[0, "on_time_rate"] == pytest.approx(1.0)
    assert prof.loc[0, "avg_slip_days"] == pytest.approx(0.0, abs=1e-6)


def test_chronically_late_supplier_scores_zero_otd():
    prof = supplier_lead_time_profiles(make_pos(slip=30), as_of=AS_OF)
    assert prof.loc[0, "on_time_rate"] == pytest.approx(0.0)
    assert prof.loc[0, "avg_slip_days"] == pytest.approx(30.0, abs=1e-6)


def test_p80_exceeds_p50_for_variable_supplier():
    pos = make_pos(n=60)
    rng = np.random.default_rng(0)
    pos["actual_delivery_date"] = pos["order_date"] + pd.to_timedelta(
        100 + rng.integers(0, 60, len(pos)), unit="D")
    prof = supplier_lead_time_profiles(pos, as_of=AS_OF)
    assert prof.loc[0, "p80_days"] > prof.loc[0, "p50_days"]
    assert prof.loc[0, "p95_days"] >= prof.loc[0, "p80_days"]


def test_negative_lead_times_are_dropped():
    """ERP data-entry errors produce deliveries before the order date."""
    pos = make_pos(n=10)
    pos.loc[0, "actual_delivery_date"] = pos.loc[0, "order_date"] - timedelta(days=5)
    clean = prepare_deliveries(pos, as_of=AS_OF)
    assert len(clean) == 9
    assert (clean["actual_lead_time_days"] >= 0).all()


def test_open_pos_excluded_from_lead_time():
    pos = make_pos(n=10)
    pos.loc[0, "actual_delivery_date"] = pd.NaT
    assert len(prepare_deliveries(pos, as_of=AS_OF)) == 9


def test_recency_weighting_favours_recent_behaviour():
    """A supplier that was late historically but is on time now should score
    better than the raw average implies."""
    n = 40
    order_dates = [AS_OF - timedelta(days=30 + i * 15) for i in range(n)]
    # Recent half on time, older half 60 days late.
    slips = [0] * (n // 2) + [60] * (n // 2)
    pos = pd.DataFrame({
        "po_number": [f"PO-{i}" for i in range(n)],
        "part_number": "P1", "supplier_id": "SUP-1",
        "order_date": order_dates,
        "promised_date": [d + timedelta(days=100) for d in order_dates],
        "actual_delivery_date": [d + timedelta(days=100 + s) for d, s in zip(order_dates, slips)],
        "quantity": 1, "unit_price_usd": 1.0, "status": "DELIVERED",
    })
    prof = supplier_lead_time_profiles(pos, as_of=AS_OF)
    # Unweighted OTD would be exactly 0.5; recency weighting must beat that.
    assert prof.loc[0, "on_time_rate"] > 0.5


def test_scorecard_ranks_reliable_above_unreliable():
    good = make_pos("SUP-GOOD", slip=0)
    bad = make_pos("SUP-BAD", slip=45)
    card = supplier_scorecard(pd.concat([good, bad]), as_of=AS_OF).set_index("supplier_id")
    assert card.loc["SUP-GOOD", "composite_score"] > card.loc["SUP-BAD", "composite_score"]
    assert card.loc["SUP-GOOD", "risk_tier"] == "PREFERRED"


def test_scorecard_computes_contract_gap():
    pos = make_pos("SUP-1", promised=100, slip=20)
    suppliers = pd.DataFrame({
        "supplier_id": ["SUP-1"], "supplier_name": ["Test"],
        "country": ["US"], "base_lead_time_days": [100],
    })
    card = supplier_scorecard(pos, suppliers, as_of=AS_OF)
    # Observed P80 is ~120 against a contracted 100.
    assert card.loc[0, "contract_gap_days"] == pytest.approx(20, abs=2)


def test_confidence_reflects_sample_size():
    assert supplier_lead_time_profiles(make_pos(n=30), as_of=AS_OF).loc[0, "confidence"] == "HIGH"
    assert supplier_lead_time_profiles(make_pos(n=6), as_of=AS_OF).loc[0, "confidence"] == "MEDIUM"
    assert supplier_lead_time_profiles(make_pos(n=3), as_of=AS_OF).loc[0, "confidence"] == "LOW"


# -------------------------------------------------------------- AOG scorer


def base_inputs(**overrides):
    parts = pd.DataFrame({
        "part_number": ["P1", "P2"],
        "criticality": ["NO_GO", "GO"],
        "ata_chapter": ["32", "21"],
        "unit_cost_usd": [10000.0, 100.0],
    })
    inventory = pd.DataFrame({
        "warehouse_id": ["W1", "W1"],
        "part_number": ["P1", "P2"],
        "qty_available": [0, 100],
    })
    open_orders = pd.DataFrame(columns=["part_number", "quantity", "expected_date", "supplier_id"])
    demand = pd.DataFrame({"part_number": ["P1", "P2"], "monthly_demand": [10.0, 10.0]})
    kwargs = {"parts": parts, "inventory": inventory, "open_orders": open_orders, "demand": demand}
    kwargs.update(overrides)
    return ScoringInputs(**kwargs)


def test_no_go_part_with_zero_stock_scores_higher_than_go_part_with_stock():
    scored = score_aog_risk(base_inputs(), as_of=AS_OF).set_index("part_number")
    assert scored.loc["P1", "aog_risk_score"] > scored.loc["P2", "aog_risk_score"]
    assert scored.loc["P1", "risk_band"] in ("CRITICAL", "HIGH")


def test_weights_sum_to_one():
    """A drifting weight set would silently rescale every historic score."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_criticality_ordering_is_monotonic():
    assert CRITICALITY_SCORE["NO_GO"] > CRITICALITY_SCORE["GO_IF"] > CRITICALITY_SCORE["GO"]


def test_score_is_bounded():
    scored = score_aog_risk(base_inputs(), as_of=AS_OF)
    assert scored["aog_risk_score"].between(0, 100).all()


def test_coverage_factor_zero_when_fully_stocked():
    available = pd.Series([1000.0], index=["P1"])
    daily = pd.Series([1.0], index=["P1"])
    assert coverage_factor(available, daily, 45).iloc[0] == 0.0


def test_coverage_factor_one_when_no_stock():
    available = pd.Series([0.0], index=["P1"])
    daily = pd.Series([1.0], index=["P1"])
    assert coverage_factor(available, daily, 45).iloc[0] == 1.0


def test_coverage_factor_ignores_parts_with_no_demand():
    """No forecast demand means no shortage risk from demand alone."""
    available = pd.Series([0.0], index=["P1"])
    daily = pd.Series([0.0], index=["P1"])
    assert coverage_factor(available, daily, 45).iloc[0] == 0.0


def test_open_aog_work_order_forces_max_score():
    """An aircraft already on the ground is not a prediction."""
    wos = pd.DataFrame({"part_number": ["P2"], "qty_required": [500], "wo_type": ["AOG"]})
    scored = score_aog_risk(base_inputs(open_work_orders=wos), as_of=AS_OF).set_index("part_number")
    assert scored.loc["P2", "aog_risk_score"] == 100.0


def test_aog_work_order_with_sufficient_stock_does_not_force_max():
    wos = pd.DataFrame({"part_number": ["P2"], "qty_required": [1], "wo_type": ["AOG"]})
    scored = score_aog_risk(base_inputs(open_work_orders=wos), as_of=AS_OF).set_index("part_number")
    assert scored.loc["P2", "aog_risk_score"] < 100.0


def test_customs_hold_raises_score():
    without = score_aog_risk(base_inputs(), as_of=AS_OF).set_index("part_number")
    customs = pd.DataFrame({"part_number": ["P2"], "customs_risk": [1.0]})
    with_hold = score_aog_risk(base_inputs(customs_risk=customs), as_of=AS_OF).set_index("part_number")
    assert with_hold.loc["P2", "aog_risk_score"] > without.loc["P2", "aog_risk_score"]


def test_incoming_po_reduces_replenishment_risk():
    soon = pd.DataFrame({
        "part_number": ["P1"], "quantity": [50],
        "expected_date": [AS_OF + timedelta(days=2)], "supplier_id": ["S1"],
    })
    late = soon.assign(expected_date=[AS_OF + timedelta(days=300)])
    early_score = score_aog_risk(base_inputs(open_orders=soon), as_of=AS_OF).set_index("part_number")
    late_score = score_aog_risk(base_inputs(open_orders=late), as_of=AS_OF).set_index("part_number")
    assert early_score.loc["P1", "factor_replenishment"] < late_score.loc["P1", "factor_replenishment"]


def test_primary_driver_identifies_dominant_factor():
    customs = pd.DataFrame({"part_number": ["P2"], "customs_risk": [1.0]})
    scored = score_aog_risk(base_inputs(customs_risk=customs), as_of=AS_OF).set_index("part_number")
    assert scored.loc["P2", "primary_driver"] in WEIGHTS


def test_unreliable_supplier_raises_score():
    orders = pd.DataFrame({
        "part_number": ["P1"], "quantity": [10],
        "expected_date": [AS_OF + timedelta(days=10)], "supplier_id": ["S-BAD"],
    })
    good = pd.DataFrame({"supplier_id": ["S-BAD"], "on_time_rate": [1.0]})
    bad = pd.DataFrame({"supplier_id": ["S-BAD"], "on_time_rate": [0.0]})
    s_good = score_aog_risk(base_inputs(open_orders=orders, supplier_scores=good), as_of=AS_OF).set_index("part_number")
    s_bad = score_aog_risk(base_inputs(open_orders=orders, supplier_scores=bad), as_of=AS_OF).set_index("part_number")
    assert s_bad.loc["P1", "aog_risk_score"] > s_good.loc["P1", "aog_risk_score"]


def test_early_warning_excludes_already_stocked_out_parts():
    """The KPI is about advance notice, so a part already at zero doesn't count."""
    scored = score_aog_risk(base_inputs(), as_of=AS_OF)
    early = early_warning_report(scored, horizon_days=45)
    assert "P1" not in set(early["part_number"])  # P1 has zero stock already
    assert (early["qty_available"] > 0).all()


def test_early_warning_flags_part_burning_down():
    inventory = pd.DataFrame({
        "warehouse_id": ["W1"], "part_number": ["P1"], "qty_available": [5],
    })
    parts = pd.DataFrame({
        "part_number": ["P1"], "criticality": ["NO_GO"],
        "ata_chapter": ["32"], "unit_cost_usd": [1.0],
    })
    demand = pd.DataFrame({"part_number": ["P1"], "monthly_demand": [15.0]})  # 0.5/day
    inp = ScoringInputs(
        parts=parts, inventory=inventory,
        open_orders=pd.DataFrame(columns=["part_number", "quantity", "expected_date", "supplier_id"]),
        demand=demand,
    )
    early = early_warning_report(score_aog_risk(inp, as_of=AS_OF), horizon_days=45)
    assert len(early) == 1
    # 5 units at 0.5/day = 10 days = 240 hours of warning.
    assert early.iloc[0]["warning_hours"] == pytest.approx(240, abs=1)


def test_aog_summary_counts_bands():
    summary = aog_summary(score_aog_risk(base_inputs(), as_of=AS_OF))
    assert summary["parts_scored"] == 2
    assert summary["critical_count"] + summary["high_count"] >= 1


def test_empty_inventory_does_not_crash():
    inp = base_inputs(inventory=pd.DataFrame(columns=["warehouse_id", "part_number", "qty_available"]))
    scored = score_aog_risk(inp, as_of=AS_OF)
    assert len(scored) == 2
    assert (scored["qty_available"] == 0).all()


# ---------------------------------------------------------------- ML


def synthetic_history(n_parts=30, n_months=36, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range(end=pd.Period("2026-07", freq="M"), periods=n_months)
    rows = []
    for p in range(n_parts):
        base = rng.uniform(1, 10)
        for m in months:
            lam = base * (1 + 0.3 * np.sin(2 * np.pi * m.month / 12))
            rows.append((f"P{p}", str(m), rng.poisson(lam)))
    return pd.DataFrame(rows, columns=["part_number", "month", "demand_qty"])


def test_classify_demand_pattern_smooth():
    steady = pd.Series([5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6])
    assert classify_demand_pattern(steady) == "SMOOTH"


def test_classify_demand_pattern_intermittent():
    sparse = pd.Series([0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3])
    assert classify_demand_pattern(sparse) in ("INTERMITTENT", "LUMPY")


def test_classify_demand_pattern_no_demand():
    assert classify_demand_pattern(pd.Series([0] * 12)) == "NO_DEMAND"


def test_build_features_does_not_leak_current_month():
    """lag_1 for month t must equal actual demand at t-1, never t."""
    hist = synthetic_history(n_parts=2, n_months=12)
    feats = build_features(hist)
    one = feats[feats["part_number"] == "P0"].sort_values("month").reset_index(drop=True)
    assert pd.isna(one.loc[0, "lag_1"])
    for i in range(1, len(one)):
        assert one.loc[i, "lag_1"] == one.loc[i - 1, "demand_qty"]


def test_rolling_features_exclude_current_month():
    hist = synthetic_history(n_parts=1, n_months=24)
    feats = build_features(hist).sort_values("month").reset_index(drop=True)
    # roll_mean_3 at row i averages rows i-3..i-1, never row i.
    expected = feats.loc[7:9, "demand_qty"].mean()
    assert feats.loc[10, "roll_mean_3"] == pytest.approx(expected)


def test_mase_below_one_means_better_than_naive():
    y = np.array([10.0, 12.0, 11.0])
    good = np.array([10.0, 12.0, 11.0])
    naive = np.array([5.0, 5.0, 5.0])
    assert mase(y, good, naive) == 0.0


def test_train_demand_model_beats_naive_on_seasonal_data():
    hist = synthetic_history(n_parts=40, n_months=36)
    result = train_demand_model(hist, test_months=6)
    assert result.metrics["mase_vs_naive"] < 1.0
    assert result.metrics["n_test"] > 0


def test_train_demand_model_uses_time_split_not_random():
    """Every test row must post-date every training row."""
    hist = synthetic_history(n_parts=10, n_months=36)
    result = train_demand_model(hist, test_months=6)
    split = pd.Period(result.metrics["test_from_month"], freq="M")
    assert (pd.PeriodIndex(result.predictions["month"], freq="M") >= split).all()


def test_train_demand_model_rejects_short_history():
    hist = synthetic_history(n_parts=5, n_months=8)
    with pytest.raises(ValueError, match="months of history"):
        train_demand_model(hist, test_months=6)


def test_demand_profile_marks_sparse_parts_unforecastable():
    sparse = pd.DataFrame({
        "part_number": ["P0"] * 36,
        "month": [str(m) for m in pd.period_range(end=pd.Period("2026-07", freq="M"), periods=36)],
        "demand_qty": [0] * 35 + [1],
    })
    prof = demand_profile(sparse)
    assert not prof.loc[0, "forecastable"]
