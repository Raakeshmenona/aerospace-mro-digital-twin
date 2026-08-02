"""Event-triggered AOG alerting.

The nightly sync is too slow for an aircraft that goes down at 09:00. This DAG
runs on a short interval and on external trigger (a webhook from the
maintenance system when an AOG work order opens), re-scores only the affected
parts, and pages the right people.

Why a separate DAG rather than a shorter schedule on the main one: re-running
the full multi-source sync every 15 minutes would hammer 200 supplier portals
and blow the Snowflake credit budget. This one reads the already-loaded hub and
only re-polls the specific suppliers holding at-risk parts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Param, Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# A part must clear this score to page a human. Set high deliberately - an
# alerting system that cries wolf gets muted, and a muted AOG alert is worse
# than no alert at all.
ALERT_THRESHOLD = 75.0
# Don't re-page on the same part within this window.
ALERT_COOLDOWN_HOURS = 12


@dag(
    dag_id="aog_alert_trigger",
    description="Event-driven AOG risk re-scoring and alerting",
    default_args={
        "owner": "mro-data-eng",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    schedule="*/15 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mro", "aog", "alerting"],
    params={
        "part_numbers": Param([], type="array", description="Specific parts to re-score (empty = scan all open AOG WOs)"),
        "work_order_id": Param("", type="string", description="Triggering work order, if any"),
        "force_alert": Param(False, type="boolean"),
    },
    doc_md=__doc__,
)
def aog_alert_trigger():

    start = EmptyOperator(task_id="start")

    @task(task_id="identify_at_risk_parts")
    def identify_parts(params: dict = None) -> list[str]:
        """Which parts need re-scoring right now."""
        from snowflake.hub_loader import SnowflakeConfig, snowflake_connection

        params = params or {}
        explicit = params.get("part_numbers") or []
        if explicit:
            log.info("re-scoring %d explicitly requested parts", len(explicit))
            return list(explicit)

        # Otherwise: every part blocking an open work order, plus anything
        # already sitting near the alert threshold.
        sql = f"""
            SELECT DISTINCT PART_NUMBER FROM V_BLOCKED_WORK_ORDERS
            WHERE WO_TYPE = 'AOG' OR AOG_RISK_SCORE >= {ALERT_THRESHOLD - 15}
        """
        with snowflake_connection(SnowflakeConfig()) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            parts = [r[0] for r in cur.fetchall()]
        log.info("identified %d parts to re-score", len(parts))
        return parts

    @task(task_id="refresh_supplier_status")
    def refresh_supplier_status(part_numbers: list[str]) -> str:
        """Re-poll only the suppliers who hold these specific parts."""
        import json
        from pathlib import Path

        import pandas as pd

        from connectors.supplier_api_client import SupplierEndpoint, SupplierPortalClient

        if not part_numbers:
            return ""

        endpoints = json.loads(Variable.get("mro_supplier_endpoints", "[]"))
        frames = []
        for cfg in endpoints:
            try:
                client = SupplierPortalClient(SupplierEndpoint(**cfg))
                # Scoped query: 200 portals x 5 parts, not x 2M parts.
                frames.append(client.fetch_part_availability(part_numbers=part_numbers))
            except Exception:
                log.exception("supplier %s unreachable during AOG check", cfg.get("supplier_id"))

        out = Path(Variable.get("mro_staging_dir", "/data/staging")) / "aog_supplier_snapshot.parquet"
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined.to_parquet(out, index=False)
        return str(out)

    @task(task_id="rescore_parts")
    def rescore(part_numbers: list[str], supplier_path: str) -> list[dict]:
        """Re-score against live stock and supplier availability."""
        import pandas as pd

        from processing.aog_risk_scorer import ScoringInputs, recommend_actions, score_aog_risk
        from snowflake.hub_loader import SnowflakeConfig, snowflake_connection

        if not part_numbers:
            return []

        placeholders = ", ".join(["%s"] * len(part_numbers))
        with snowflake_connection(SnowflakeConfig()) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM V_DIM_PARTS_CURRENT WHERE PART_NUMBER IN ({placeholders})", part_numbers)
            parts = cur.fetch_pandas_all()
            cur.execute(
                f"""SELECT * FROM FCT_INVENTORY_SNAPSHOT
                    WHERE PART_NUMBER IN ({placeholders})
                    QUALIFY SNAPSHOT_DATE = MAX(SNAPSHOT_DATE) OVER ()""", part_numbers)
            inventory = cur.fetch_pandas_all()
            cur.execute(
                f"""SELECT PART_NUMBER, SUPPLIER_ID, QUANTITY,
                           COALESCE(PROJECTED_ARRIVAL_DATE, PROMISED_DATE) AS EXPECTED_DATE
                    FROM FCT_PURCHASE_ORDERS
                    WHERE STATUS = 'OPEN' AND PART_NUMBER IN ({placeholders})""", part_numbers)
            open_orders = cur.fetch_pandas_all()
            cur.execute(
                f"""SELECT PART_NUMBER, FORECAST_QTY FROM FCT_DEMAND_FORECAST
                    WHERE HORIZON_STEP = 1 AND PART_NUMBER IN ({placeholders})""", part_numbers)
            demand = cur.fetch_pandas_all()
            cur.execute("SELECT SUPPLIER_ID, ON_TIME_RATE, P80_DAYS FROM V_SUPPLIER_SCORECARD")
            scorecard = cur.fetch_pandas_all()
            cur.execute(
                f"""SELECT PART_NUMBER, QTY_REQUIRED, WO_TYPE FROM FCT_WORK_ORDERS
                    WHERE STATUS = 'OPEN' AND PART_NUMBER IN ({placeholders})""", part_numbers)
            work_orders = cur.fetch_pandas_all()

        for df in (parts, inventory, open_orders, demand, scorecard, work_orders):
            df.columns = [c.lower() for c in df.columns]

        # Live supplier stock beats last night's snapshot when available.
        if supplier_path:
            live = pd.read_parquet(supplier_path)
            if not live.empty:
                extra = live.groupby("part_number", as_index=False)["qty_available"].sum()
                extra["warehouse_id"] = "SUPPLIER_STOCK"
                inventory = pd.concat([inventory, extra], ignore_index=True)

        scored = score_aog_risk(ScoringInputs(
            parts=parts, inventory=inventory, open_orders=open_orders, demand=demand,
            supplier_scores=scorecard, open_work_orders=work_orders,
        ))
        actions = recommend_actions(scored, top_n=len(scored))
        return actions[actions["aog_risk_score"] >= ALERT_THRESHOLD].to_dict("records")

    @task(task_id="filter_cooldown")
    def filter_cooldown(alerts: list[dict], params: dict = None) -> list[dict]:
        """Suppress parts already paged recently, unless forced."""
        params = params or {}
        if params.get("force_alert") or not alerts:
            return alerts

        from snowflake.hub_loader import SnowflakeConfig, snowflake_connection

        parts = [a["part_number"] for a in alerts]
        placeholders = ", ".join(["%s"] * len(parts))
        with snowflake_connection(SnowflakeConfig()) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""SELECT DISTINCT PART_NUMBER FROM FCT_AOG_ALERT_LOG
                    WHERE PART_NUMBER IN ({placeholders})
                      AND ALERTED_AT > DATEADD('hour', -{ALERT_COOLDOWN_HOURS}, CURRENT_TIMESTAMP())""",
                parts,
            )
            recent = {r[0] for r in cur.fetchall()}
        fresh = [a for a in alerts if a["part_number"] not in recent]
        log.info("%d alerts, %d suppressed by cooldown", len(alerts), len(alerts) - len(fresh))
        return fresh

    @task(task_id="dispatch_alerts")
    def dispatch(alerts: list[dict], params: dict = None) -> int:
        """Page the planning desk and log the alert for the lead-time KPI."""
        from datetime import datetime as dt

        import pandas as pd

        from snowflake.hub_loader import HubLoader, SnowflakeConfig

        if not alerts:
            log.info("no parts above alert threshold")
            return 0

        params = params or {}
        lines = [
            f"[{a['risk_band']}] {a['part_number']} ({a['criticality']}) "
            f"score {a['aog_risk_score']}, {a['qty_available']:.0f} available, "
            f"{a['days_of_cover']:.0f}d cover -> {a['recommended_action']}"
            for a in alerts
        ]
        body = "\n".join(lines)
        log.warning("AOG ALERT (%d parts)\n%s", len(alerts), body)

        _send_notifications(
            subject=f"AOG risk: {len(alerts)} part(s) above threshold",
            body=body,
            work_order_id=params.get("work_order_id", ""),
        )

        log_df = pd.DataFrame(alerts)
        log_df["alerted_at"] = dt.utcnow()
        log_df["triggering_wo"] = params.get("work_order_id", "")
        HubLoader(SnowflakeConfig()).load(
            log_df, "FCT_AOG_ALERT_LOG", merge_keys=["part_number", "alerted_at"]
        )
        return len(alerts)

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    parts = identify_parts()
    supplier_snapshot = refresh_supplier_status(parts)
    alerts = rescore(parts, supplier_snapshot)
    fresh = filter_cooldown(alerts)
    sent = dispatch(fresh)

    start >> parts
    sent >> end


def _send_notifications(subject: str, body: str, work_order_id: str = "") -> None:
    """Fan out to Teams and email.

    Notification failure must never fail the DAG - a dead webhook should not
    hide the fact that we successfully identified an AOG risk.
    """
    import json

    from airflow.models import Variable

    webhook = Variable.get("mro_teams_webhook", "")
    if webhook:
        try:
            import requests

            requests.post(
                webhook,
                json={"title": subject, "text": body, "themeColor": "D93F3F"},
                timeout=15,
            ).raise_for_status()
        except Exception:
            log.exception("Teams notification failed (alert still recorded)")

    recipients = json.loads(Variable.get("mro_alert_recipients", "[]"))
    if recipients:
        try:
            from airflow.utils.email import send_email

            send_email(to=recipients, subject=subject, html_content=f"<pre>{body}</pre>")
        except Exception:
            log.exception("email notification failed (alert still recorded)")


aog_alert_trigger()
