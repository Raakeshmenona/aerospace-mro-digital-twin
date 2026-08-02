# Aerospace MRO Supply Chain Digital Twin

A multi-source data pipeline that unifies aircraft parts data from five legacy
systems into a Snowflake supply chain hub, forecasts spare-part demand, and
scores every part for **AOG (Aircraft on Ground) risk** before the bin runs empty.

## The Problem

MRO operations manage parts across 200+ suppliers with lead times of 6–18 months.
When an aircraft goes AOG, downtime costs $10,000–$150,000 per hour. The data
that would predict it is scattered across five systems that don't talk:

| System | Holds | Format |
|---|---|---|
| SAP PM/MM (ERP) | part master, POs, work orders | RFC / nightly flat-file extract |
| Supplier portals | availability, order milestones | REST/JSON (OAuth2), EDI 850/856 |
| Customs broker | clearance status of inbound parts | XML manifest |
| Warehouse WMS | on-hand, reserved, quarantine, shelf life | REST + legacy ODBC |
| Technical logs | work order demand | ERP-linked |

Engineers reconcile this by hand in spreadsheets. By the time a shortage is
visible, the aircraft is already down.

## Architecture

```
SAP PM/MM ────► erp_sap_extractor ──┐
Supplier APIs ► supplier_api_client ┤
EDI 850/856 ──► parse_edi_850/856   ├──► parts_catalog_joiner (PySpark)
Customs XML ──► customs_feed_parser ┤         │ normalize PN, resolve supersessions
WMS ──────────► wms_connector ──────┘         ▼
                                        unified part master
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────┐
        ▼                                     ▼                             ▼
 lead_time_calculator            demand_forecaster (GBM)            customs_delay_risk
 observed P50/P80/P95, OTD       3-month part demand                clearance risk
        └─────────────────────────────────────┼─────────────────────────────┘
                                              ▼
                                    aog_risk_scorer (rule engine)
                                  criticality × coverage × replenishment
                                       × supplier × customs
                                              │
                                              ▼
                              Snowflake hub (hub-and-spoke star schema)
                                              │
                                              ▼
                          Power BI: AOG heatmap, supplier scorecard, availability
```

Orchestrated by two Airflow DAGs: `mro_daily_sync` (nightly, full refresh) and
`aog_alert_trigger` (every 15 min + webhook, scoped re-scoring and paging).

## Quickstart — runs locally, no Snowflake/Spark/SAP needed

```bash
pip install -r requirements.txt
```

```bash
python data/generate_synthetic.py
```

```bash
python run_pipeline.py
```

That executes ingestion → supplier scorecard → demand model → AOG scoring and
prints the KPIs. Individual stages are runnable too:

```bash
python -m processing.aog_risk_scorer
```

```bash
python -m processing.lead_time_calculator
```

```bash
python -m ml.demand_forecaster
```

```bash
python -m pytest tests -q
```

## Project Layout

| Path | Purpose |
|---|---|
| `connectors/erp_sap_extractor.py` | SAP PM/MM via RFC (`RFC_READ_TABLE`) or flat-file extract; MARA/EKPO/AUFK → canonical names |
| `connectors/supplier_api_client.py` | OAuth2 REST client with backoff + `Retry-After`; EDI 850/856 parsers |
| `connectors/customs_feed_parser.py` | Streaming XML (`iterparse`) → clearance risk per PO |
| `connectors/wms_connector.py` | REST / ODBC / Parquet adapters → one inventory schema |
| `processing/parts_catalog_joiner.py` | PySpark: PN normalization, supersession chains, salted skew joins |
| `processing/lead_time_calculator.py` | Recency-weighted lead-time percentiles + supplier scorecard |
| `processing/aog_risk_scorer.py` | Five-factor weighted risk engine + recommended actions |
| `ml/demand_forecaster.py` | Gradient boosting on intermittent spares demand; MASE-gated |
| `snowflake/hub_loader.py` | Staged Parquet + `MERGE` upserts, batch tracking, rollback |
| `snowflake/snowflake_models/*.sql` | SCD2 `dim_parts`, supplier performance snapshots, fact tables, BI views |
| `airflow_dags/mro_daily_sync_dag.py` | Nightly multi-source sync |
| `airflow_dags/aog_alert_trigger_dag.py` | Event-driven re-scoring, cooldown, alerting |
| `run_pipeline.py` | End-to-end local run against synthetic data |
| `azure-pipelines.yml` | CI/CD: lint → DAG-import gate → tests → Snowflake DDL + DAG deploy |

## The AOG Risk Model

Every part scores 0–100 from five weighted factors:

| Factor | Weight | Signal |
|---|---|---|
| Criticality | 0.30 | MEL classification — NO_GO grounds the aircraft, GO is paperwork |
| Coverage | 0.28 | available stock vs forecast demand across a 45-day horizon |
| Replenishment | 0.20 | will the next PO land before stock hits zero? |
| Supplier | 0.12 | observed on-time rate of the supplier on the inbound PO |
| Customs | 0.10 | is the shipment held at a border? |

Two deliberate overrides: an **open AOG work order with insufficient stock forces
100** (the aircraft is already down — that's not a prediction), and parts with no
forecast demand carry no coverage risk regardless of stock level.

Each row keeps its full factor breakdown and a `primary_driver`, so the dashboard
shows a planner *why* a part scored high and what to do about it — not just a number.

## Business KPIs

Measured on the synthetic dataset by `run_pipeline.py`:

**AOG risk identified earlier.** The pipeline flags parts that still have stock
today but burn down to zero inside the horizon. On the sample run: 24 such parts,
**100% flagged with ≥ 72 hours of notice**, median warning 728 hours (30 days),
against zero hours for reactive discovery — a technician opening an empty bin.
Four were NO_GO parts, each a potential grounding.

**Supplier on-time delivery scorecard.** 60 suppliers scored automatically from
5,000 PO lines: recency-weighted lead-time percentiles, on-time rate, volatility,
and a degradation trend. Composite 0–100 with CRITICAL/WATCH/ACCEPTABLE/PREFERRED
tiers. The `contract_gap_days` column surfaces the difference between the
contracted lead time and the observed P80 — the slack planners absorb unknowingly.

**Parts availability accuracy.** The `stock_accuracy_report` measures cycle-count
freshness (59.8% counted within 90 days on the sample data) — the gap manual
spreadsheets carry silently. Availability is computed properly: on-hand minus
reserved minus quarantine, zeroed for non-issuable conditions and expired
shelf-life stock. 2,000 parts get a unified cross-system view.

**Demand forecast quality.** MASE 0.86 against the planner's last-month heuristic
(~14% error reduction). The DAG refuses to publish a model with MASE ≥ 1.0 —
a forecast that can't beat the heuristic it replaces shouldn't ship.

## Engineering Notes

**Intermittent demand is the real ML problem.** Most part-months are zero, so
plain regression predicts ~0 everywhere and scores well on RMSE while being
useless. The model encodes intermittency directly (months-since-last-demand, ADI,
CV²), classifies parts by Syntetos-Boylan category so the dashboard knows which
forecasts to trust, and evaluates on MASE. XGBoost uses a Poisson objective; the
sklearn fallback trains on `log1p` with a **Duan smearing correction**, because
naive back-transform under-forecasts — and under-forecasting spares is the
dangerous direction.

**Time-based splits only.** Random CV on a demand panel leaks the future and
inflates every metric. Tests assert that every test row post-dates every training
row, and that lag/rolling features never see their own month.

**Part number reconciliation.** The same physical part appears as `HTL-4471-002`,
`HTL4471002`, `HTL 4471-002 REV B`, and `HTL-4471-002/B` across systems.
`normalize_part_number` canonicalizes, supersession chains resolve iteratively to
the terminal orderable part, and `part_uid` is a deterministic SHA-256 prefix
(stable across runs, unlike `monotonically_increasing_id`).

**Skew is the Spark problem here, not volume.** A handful of common fasteners
dominate the row count, so `skew_aware_join` salts the hot side and replicates
the small side across buckets rather than letting one task inherit every row.

**Idempotency.** MRO pipelines get re-run constantly — late supplier files,
customs corrections. Every load is a `MERGE` on the natural key, and every row
carries `_batch_id` so a bad load can be rolled back without a full refresh.

**Fail-soft ingestion.** One dead supplier portal out of 200 marks that feed
stale and lets the DAG continue. A supplier outage is a Tuesday, not an incident.

**Alert discipline.** Alerting fires at score ≥ 75 with a 12-hour per-part
cooldown. An alerting system that cries wolf gets muted, and a muted AOG alert is
worse than no alert at all.

## Power BI Layer

Three pages, backed by the SQL views:

- **AOG risk heatmap** — `V_AOG_HEATMAP`, ATA chapter × criticality, coloured by
  at-risk count with value-at-risk in the tooltip
- **Supplier scorecard** — `V_SUPPLIER_SCORECARD` and `V_SUPPLIER_TREND`, with
  quarter-over-quarter movement to trigger corrective action requests
- **Part availability** — `V_BLOCKED_WORK_ORDERS`, open work orders that can't be
  completed, with shortfall and next receipt date

`V_AOG_EARLY_WARNING` backs the headline KPI tile by joining first-alert timestamps
against first-stockout dates — the audit trail proving the alert preceded the event.

## Production Configuration

Airflow Variables: `mro_sap_extract_dir`, `mro_staging_dir`, `mro_model_dir`,
`mro_supplier_endpoints` (JSON), `mro_wms_config` (JSON), `mro_customs_feed_path`,
`mro_teams_webhook`, `mro_alert_recipients` (JSON).

Snowflake auth uses key-pair for service accounts (`SNOWFLAKE_PRIVATE_KEY_PATH`),
with password auth as the local-dev fallback only. Credentials come from Azure
Key Vault in production — never from the repo.
"# aerospace-mro-digital-twin" 
"# aerospace-mro-digital-twin" 
"# aerospace-mro-digital-twin" 
"# aerospace-mro-digital-twin" 
