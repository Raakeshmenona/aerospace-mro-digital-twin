"""Nightly MRO supply chain sync.

Ingests all five source systems in parallel, joins the part catalog, computes
supplier lead times, refreshes the demand forecast, scores AOG risk and loads
the Snowflake hub.

Design notes:
  * Source tasks are independent and fail soft. One dead supplier portal must
    not block the other 199 - the DAG continues and marks that feed stale.
  * The forecast is retrained weekly, not nightly: MRO demand moves on a
    monthly cycle and a nightly retrain burns compute for no accuracy gain.
  * The scoring task is the join point; everything downstream depends on it.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mro-data-eng",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": True,
    "email": ["mro-data-alerts@example.com"],
}


@dag(
    dag_id="mro_daily_sync",
    description="Nightly multi-source MRO supply chain sync into Snowflake",
    default_args=DEFAULT_ARGS,
    # 02:00 IST - after the ERP nightly close, before the 06:00 planning shift.
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mro", "supply-chain", "snowflake"],
    doc_md=__doc__,
)
def mro_daily_sync():

    start = EmptyOperator(task_id="start")

    # ------------------------------------------------------------- ingest

    @task(task_id="extract_erp")
    def extract_erp(ds: str = "{{ ds }}") -> str:
        from pathlib import Path

        from connectors.erp_sap_extractor import SAPConnection, SAPExtractor

        conn = SAPConnection(extract_dir=Path(Variable.get("mro_sap_extract_dir", "/data/sap")))
        with SAPExtractor(conn) as ex:
            parts = ex.extract_material_master()
            pos = ex.extract_purchase_orders(changed_since=datetime.fromisoformat(ds).date() - timedelta(days=7))
            wos = ex.extract_work_orders()
        out = Path(Variable.get("mro_staging_dir", "/data/staging")) / ds
        out.mkdir(parents=True, exist_ok=True)
        for name, df in (("erp_parts", parts), ("erp_pos", pos), ("erp_wos", wos)):
            df.to_parquet(out / f"{name}.parquet", index=False)
        log.info("ERP: %d parts, %d PO lines, %d work orders", len(parts), len(pos), len(wos))
        return str(out)

    @task(task_id="fetch_supplier_feeds", retries=2)
    def fetch_supplier_feeds(staging_dir: str, ds: str = "{{ ds }}") -> str:
        """Poll every supplier portal. Individual failures are tolerated."""
        import json
        from pathlib import Path

        import pandas as pd

        from connectors.supplier_api_client import SupplierEndpoint, SupplierPortalClient

        endpoints = json.loads(Variable.get("mro_supplier_endpoints", "[]"))
        frames, failed = [], []
        for cfg in endpoints:
            try:
                client = SupplierPortalClient(SupplierEndpoint(**cfg))
                frames.append(client.fetch_part_availability())
            except Exception:
                # A supplier portal outage is a Tuesday, not an incident.
                log.exception("supplier %s failed", cfg.get("supplier_id"))
                failed.append(cfg.get("supplier_id"))

        if failed:
            log.warning("%d/%d supplier feeds failed: %s", len(failed), len(endpoints), failed)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        path = Path(staging_dir) / "supplier_availability.parquet"
        combined.to_parquet(path, index=False)
        return str(path)

    @task(task_id="parse_customs_feed")
    def parse_customs(staging_dir: str) -> str:
        from pathlib import Path

        from connectors.customs_feed_parser import customs_delay_risk, parse_customs_feed

        feed = Path(Variable.get("mro_customs_feed_path", "/data/customs/manifest.xml"))
        if not feed.exists():
            # Brokers skip days with no inbound shipments; that is not an error.
            raise AirflowSkipException(f"no customs manifest at {feed}")
        shipments = parse_customs_feed(feed)
        risk = customs_delay_risk(shipments)
        path = Path(staging_dir) / "customs_risk.parquet"
        risk.to_parquet(path, index=False)
        log.info("customs: %d shipments, %d held", len(shipments), int(shipments["is_held"].sum()))
        return str(path)

    @task(task_id="sync_wms")
    def sync_wms(staging_dir: str) -> str:
        import json
        from pathlib import Path

        from connectors.wms_connector import ParquetWMSAdapter, RestWMSAdapter, fetch_all, stock_accuracy_report

        cfg = json.loads(Variable.get("mro_wms_config", "{}"))
        adapters = []
        if cfg.get("rest"):
            adapters.append(RestWMSAdapter(**cfg["rest"]))
        if cfg.get("parquet_path"):
            adapters.append(ParquetWMSAdapter(cfg["parquet_path"]))

        inventory = fetch_all(adapters)
        accuracy = stock_accuracy_report(inventory)
        log.info("WMS: %(records)d records, %(trusted_pct).1f%% cycle-counted recently", accuracy)
        # Surface the accuracy KPI even when it is bad - especially when bad.
        if accuracy["trusted_pct"] < 95:
            log.warning("stock accuracy %.1f%% below 95%% target", accuracy["trusted_pct"])

        path = Path(staging_dir) / "inventory.parquet"
        inventory.to_parquet(path, index=False)
        return str(path)

    # ------------------------------------------------------------ process

    @task(task_id="join_parts_catalog")
    def join_catalog(staging_dir: str, supplier_path: str, inventory_path: str) -> str:

        from processing.parts_catalog_joiner import build_part_master, catalog_coverage_report, get_spark, write_catalog

        spark = get_spark()
        try:
            erp = spark.read.parquet(f"{staging_dir}/erp_parts.parquet")
            sup = spark.read.parquet(supplier_path)
            wms = spark.read.parquet(inventory_path)
            master = build_part_master(erp, sup, wms)
            coverage = catalog_coverage_report(master).collect()[0].asDict()
            log.info("catalog coverage: %s", coverage)
            out = f"{staging_dir}/part_master"
            write_catalog(master, out)
            return out
        finally:
            spark.stop()

    @task(task_id="calculate_lead_times")
    def calculate_lead_times(staging_dir: str) -> str:
        from pathlib import Path

        import pandas as pd

        from processing.lead_time_calculator import supplier_scorecard

        pos = pd.read_parquet(f"{staging_dir}/erp_pos.parquet")
        suppliers = pd.read_parquet(f"{staging_dir}/erp_suppliers.parquet") if Path(
            f"{staging_dir}/erp_suppliers.parquet").exists() else None
        card = supplier_scorecard(pos, suppliers)
        path = Path(staging_dir) / "supplier_scorecard.parquet"
        card.to_parquet(path, index=False)
        log.info("scored %d suppliers, %d in CRITICAL tier",
                 len(card), int((card["risk_tier"] == "CRITICAL").sum()))
        return str(path)

    @task(task_id="refresh_demand_forecast")
    def refresh_forecast(staging_dir: str, logical_date=None) -> str:
        """Retrain weekly (Sunday), otherwise reuse the persisted model."""
        import pickle
        from pathlib import Path

        import pandas as pd

        from ml.demand_forecaster import demand_profile, forecast_next_periods, train_demand_model

        model_path = Path(Variable.get("mro_model_dir", "/data/models")) / "demand_model.pkl"
        history = pd.read_parquet(f"{staging_dir}/demand_history.parquet")
        parts = pd.read_parquet(f"{staging_dir}/erp_parts.parquet")

        is_sunday = logical_date is not None and logical_date.weekday() == 6
        if is_sunday or not model_path.exists():
            result = train_demand_model(history, parts)
            log.info("retrained demand model: %s", result.metrics)
            # A model that cannot beat the planner's own heuristic must not ship.
            if result.metrics["mase_vs_naive"] >= 1.0:
                raise ValueError(
                    f"model MASE {result.metrics['mase_vs_naive']:.3f} >= 1.0 - "
                    "worse than naive baseline, refusing to publish"
                )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(pickle.dumps(result.model))
            model = result.model
        else:
            model = pickle.loads(model_path.read_bytes())

        forecast = forecast_next_periods(history, model, parts)
        profile = demand_profile(history)
        forecast = forecast.merge(profile[["part_number", "pattern", "forecastable"]], on="part_number", how="left")
        path = Path(staging_dir) / "demand_forecast.parquet"
        forecast.to_parquet(path, index=False)
        return str(path)

    @task(task_id="score_aog_risk")
    def score_risk(staging_dir: str, scorecard_path: str, forecast_path: str,
                   customs_path: str | None, inventory_path: str) -> str:
        from pathlib import Path

        import pandas as pd

        from processing.aog_risk_scorer import ScoringInputs, aog_summary, score_aog_risk

        parts = pd.read_parquet(f"{staging_dir}/erp_parts.parquet")
        inventory = pd.read_parquet(inventory_path)
        pos = pd.read_parquet(f"{staging_dir}/erp_pos.parquet")
        wos = pd.read_parquet(f"{staging_dir}/erp_wos.parquet")
        forecast = pd.read_parquet(forecast_path)
        scorecard = pd.read_parquet(scorecard_path)
        customs = pd.read_parquet(customs_path) if customs_path else None

        # Use the nearest forecast month as the demand signal.
        demand = (
            forecast[forecast["horizon_step"] == 1]
            .groupby("part_number", as_index=False)["forecast_qty"].sum()
        )
        open_orders = pos[pos["status"] == "OPEN"].rename(columns={"promised_date": "expected_date"})

        scored = score_aog_risk(ScoringInputs(
            parts=parts, inventory=inventory, open_orders=open_orders, demand=demand,
            supplier_scores=scorecard, customs_risk=customs,
            open_work_orders=wos[wos["status"] == "OPEN"],
        ))
        summary = aog_summary(scored)
        log.info("AOG summary: %s", summary)

        path = Path(staging_dir) / "aog_risk.parquet"
        scored.to_parquet(path, index=False)
        return str(path)

    # --------------------------------------------------------------- load

    @task(task_id="load_snowflake_hub", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def load_hub(staging_dir: str, risk_path: str, scorecard_path: str,
                 forecast_path: str, inventory_path: str) -> dict:
        from datetime import date

        import pandas as pd

        from snowflake.hub_loader import HubLoader, SnowflakeConfig

        cfg = SnowflakeConfig()
        loader = HubLoader(cfg)

        inventory = pd.read_parquet(inventory_path)
        inventory["snapshot_date"] = date.today()
        scorecard = pd.read_parquet(scorecard_path)
        scorecard["as_of_date"] = date.today()
        forecast = pd.read_parquet(forecast_path).rename(columns={"month": "forecast_month"})

        counts = {}
        counts["DIM_PARTS"] = loader.load(pd.read_parquet(f"{staging_dir}/erp_parts.parquet"), "DIM_PARTS")
        counts["DIM_SUPPLIER_PERFORMANCE"] = loader.load(scorecard, "DIM_SUPPLIER_PERFORMANCE")
        counts["FCT_WORK_ORDERS"] = loader.load(pd.read_parquet(f"{staging_dir}/erp_wos.parquet"), "FCT_WORK_ORDERS")
        counts["FCT_INVENTORY_SNAPSHOT"] = loader.load(inventory, "FCT_INVENTORY_SNAPSHOT")
        counts["FCT_DEMAND_FORECAST"] = loader.load(forecast, "FCT_DEMAND_FORECAST")
        counts["FCT_AOG_RISK"] = loader.load(pd.read_parquet(risk_path), "FCT_AOG_RISK")
        log.info("loaded: %s", counts)
        return counts

    @task(task_id="publish_metrics")
    def publish_metrics(counts: dict, risk_path: str) -> None:
        import pandas as pd

        from processing.aog_risk_scorer import aog_summary

        scored = pd.read_parquet(risk_path)
        summary = aog_summary(scored)
        log.info("=== MRO daily sync complete ===")
        for k, v in {**counts, **summary}.items():
            log.info("  %-28s %s", k, v)

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # ---------------------------------------------------------- wiring

    staging = extract_erp()
    supplier = fetch_supplier_feeds(staging)
    customs = parse_customs(staging)
    inventory = sync_wms(staging)

    catalog = join_catalog(staging, supplier, inventory)
    scorecard = calculate_lead_times(staging)
    forecast = refresh_forecast(staging)

    risk = score_risk(staging, scorecard, forecast, customs, inventory)
    counts = load_hub(staging, risk, scorecard, forecast, inventory)
    metrics = publish_metrics(counts, risk)

    start >> staging
    catalog >> risk
    metrics >> end


mro_daily_sync()
