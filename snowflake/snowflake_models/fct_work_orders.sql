-- fct_work_orders and the AOG risk fact - the hub of the star schema.
--
-- Work orders are the demand signal. An open work order waiting on a part is
-- the difference between "we're low on stock" (a planning issue) and "an
-- aircraft cannot fly" (a $10k-150k/hour issue).

CREATE TABLE IF NOT EXISTS FCT_WORK_ORDERS (
    WORK_ORDER_ID      VARCHAR(30)  NOT NULL PRIMARY KEY,
    AIRCRAFT_REG       VARCHAR(12),
    EQUIPMENT_ID       VARCHAR(30),
    PART_NUMBER        VARCHAR(50),
    QTY_REQUIRED       NUMBER(10,2),
    WO_TYPE            VARCHAR(20),      -- SCHEDULED / UNSCHEDULED / AOG
    STATUS             VARCHAR(20),      -- OPEN / CLOSED
    OPENED_DATE        DATE,
    CLOSED_DATE        DATE,
    LABOR_HOURS        NUMBER(8,1),
    GROUND_HOURS       NUMBER(10,1),     -- hours aircraft unavailable
    SOURCE_SYSTEM      VARCHAR(20),
    _LOADED_AT         TIMESTAMP_NTZ,
    _BATCH_ID          VARCHAR(12)
)
CLUSTER BY (OPENED_DATE, STATUS)
COMMENT = 'Maintenance work orders from SAP PM - the part demand signal'
;

CREATE TABLE IF NOT EXISTS FCT_PURCHASE_ORDERS (
    PO_NUMBER              VARCHAR(30) NOT NULL,
    PO_LINE                VARCHAR(10) NOT NULL,
    PART_NUMBER            VARCHAR(50),
    SUPPLIER_ID            VARCHAR(20),
    QUANTITY               NUMBER(12,2),
    UNIT_PRICE_USD         NUMBER(14,2),
    ORDER_DATE             DATE,
    PROMISED_DATE          DATE,
    REVISED_DATE           DATE,
    ACTUAL_DELIVERY_DATE   DATE,
    STATUS                 VARCHAR(20),   -- OPEN / DELIVERED / CANCELLED
    CLEARANCE_STATUS       VARCHAR(20),   -- from the customs feed
    PROJECTED_ARRIVAL_DATE DATE,
    SOURCE_SYSTEM          VARCHAR(20),
    _LOADED_AT             TIMESTAMP_NTZ,
    _BATCH_ID              VARCHAR(12),
    CONSTRAINT PK_PO PRIMARY KEY (PO_NUMBER, PO_LINE)
)
CLUSTER BY (ORDER_DATE, STATUS)
;

CREATE TABLE IF NOT EXISTS FCT_INVENTORY_SNAPSHOT (
    WAREHOUSE_ID     VARCHAR(20) NOT NULL,
    PART_NUMBER      VARCHAR(50) NOT NULL,
    SNAPSHOT_DATE    DATE        NOT NULL,
    QTY_ON_HAND      NUMBER(12,2),
    QTY_RESERVED     NUMBER(12,2),
    QTY_QUARANTINE   NUMBER(12,2),
    QTY_AVAILABLE    NUMBER(12,2),
    CONDITION_CODE   VARCHAR(4),
    EXPIRY_DATE      DATE,
    LAST_COUNTED     DATE,
    SOURCE_SYSTEM    VARCHAR(20),
    _LOADED_AT       TIMESTAMP_NTZ,
    _BATCH_ID        VARCHAR(12),
    CONSTRAINT PK_INV PRIMARY KEY (WAREHOUSE_ID, PART_NUMBER, SNAPSHOT_DATE)
)
CLUSTER BY (SNAPSHOT_DATE)
;

CREATE TABLE IF NOT EXISTS FCT_AOG_RISK (
    PART_NUMBER           VARCHAR(50)   NOT NULL,
    SCORED_AT             TIMESTAMP_NTZ NOT NULL,
    AOG_RISK_SCORE        NUMBER(5,1),
    RISK_BAND             VARCHAR(12),
    CRITICALITY           VARCHAR(10),
    QTY_AVAILABLE         NUMBER(12,2),
    DAYS_OF_COVER         NUMBER(10,1),
    MONTHLY_DEMAND        NUMBER(12,2),
    FIRM_WO_DEMAND        NUMBER(12,2),
    AOG_WO_DEMAND         NUMBER(12,2),
    FACTOR_CRITICALITY    NUMBER(5,3),
    FACTOR_COVERAGE       NUMBER(5,3),
    FACTOR_REPLENISHMENT  NUMBER(5,3),
    FACTOR_SUPPLIER       NUMBER(5,3),
    FACTOR_CUSTOMS        NUMBER(5,3),
    PRIMARY_DRIVER        VARCHAR(20),
    HORIZON_DAYS          NUMBER(5,0),
    _LOADED_AT            TIMESTAMP_NTZ,
    _BATCH_ID             VARCHAR(12),
    CONSTRAINT PK_AOG PRIMARY KEY (PART_NUMBER, SCORED_AT)
)
CLUSTER BY (SCORED_AT, RISK_BAND)
COMMENT = 'Time series of AOG risk scores - retains the factor breakdown so a
           planner can see WHY a part scored high, not just that it did'
;

CREATE TABLE IF NOT EXISTS FCT_DEMAND_FORECAST (
    PART_NUMBER      VARCHAR(50) NOT NULL,
    FORECAST_MONTH   VARCHAR(7)  NOT NULL,   -- YYYY-MM
    FORECAST_QTY     NUMBER(12,2),
    HORIZON_STEP     NUMBER(3,0),
    DEMAND_PATTERN   VARCHAR(15),            -- SMOOTH/ERRATIC/INTERMITTENT/LUMPY
    FORECASTABLE     BOOLEAN,
    MODEL_VERSION    VARCHAR(20),
    _LOADED_AT       TIMESTAMP_NTZ,
    _BATCH_ID        VARCHAR(12),
    CONSTRAINT PK_FCST PRIMARY KEY (PART_NUMBER, FORECAST_MONTH)
)
;

-- Every alert ever dispatched. Two jobs: cooldown suppression (don't page the
-- same part twice in 12h) and evidence for the 72-hour early-warning KPI -
-- without this log there is no way to prove the alert preceded the stockout.
CREATE TABLE IF NOT EXISTS FCT_AOG_ALERT_LOG (
    PART_NUMBER        VARCHAR(50)   NOT NULL,
    ALERTED_AT         TIMESTAMP_NTZ NOT NULL,
    AOG_RISK_SCORE     NUMBER(5,1),
    RISK_BAND          VARCHAR(12),
    CRITICALITY        VARCHAR(10),
    QTY_AVAILABLE      NUMBER(12,2),
    DAYS_OF_COVER      NUMBER(10,1),
    PRIMARY_DRIVER     VARCHAR(20),
    RECOMMENDED_ACTION VARCHAR(300),
    TRIGGERING_WO      VARCHAR(30),
    ACKNOWLEDGED_AT    TIMESTAMP_NTZ,
    ACKNOWLEDGED_BY    VARCHAR(100),
    _LOADED_AT         TIMESTAMP_NTZ,
    _BATCH_ID          VARCHAR(12),
    CONSTRAINT PK_ALERT PRIMARY KEY (PART_NUMBER, ALERTED_AT)
)
CLUSTER BY (ALERTED_AT)
;

-- ---------------------------------------------------------------- views

-- The AOG heatmap source: current risk by ATA chapter x criticality.
CREATE OR REPLACE VIEW V_AOG_HEATMAP AS
WITH current_scores AS (
    SELECT *
    FROM   FCT_AOG_RISK
    QUALIFY SCORED_AT = MAX(SCORED_AT) OVER ()
)
SELECT p.ATA_CHAPTER,
       p.ATA_CHAPTER_NAME,
       r.CRITICALITY,
       COUNT(*)                                          AS PARTS_SCORED,
       SUM(IFF(r.RISK_BAND = 'CRITICAL', 1, 0))          AS CRITICAL_PARTS,
       SUM(IFF(r.RISK_BAND IN ('CRITICAL','HIGH'), 1, 0)) AS AT_RISK_PARTS,
       ROUND(AVG(r.AOG_RISK_SCORE), 1)                   AS AVG_RISK_SCORE,
       ROUND(MAX(r.AOG_RISK_SCORE), 1)                   AS MAX_RISK_SCORE,
       ROUND(SUM(IFF(r.AOG_RISK_SCORE >= 60, p.UNIT_COST_USD, 0)), 2) AS VALUE_AT_RISK_USD
FROM   current_scores r
JOIN   V_DIM_PARTS_CURRENT p ON p.PART_NUMBER = r.PART_NUMBER
GROUP BY 1, 2, 3
;

-- Open work orders blocked on a part shortage. This is the alert list.
CREATE OR REPLACE VIEW V_BLOCKED_WORK_ORDERS AS
WITH current_scores AS (
    SELECT * FROM FCT_AOG_RISK QUALIFY SCORED_AT = MAX(SCORED_AT) OVER ()
),
stock AS (
    SELECT PART_NUMBER, SUM(QTY_AVAILABLE) AS QTY_AVAILABLE
    FROM   FCT_INVENTORY_SNAPSHOT
    QUALIFY SNAPSHOT_DATE = MAX(SNAPSHOT_DATE) OVER ()
    GROUP BY PART_NUMBER
)
SELECT w.WORK_ORDER_ID,
       w.AIRCRAFT_REG,
       w.WO_TYPE,
       w.OPENED_DATE,
       DATEDIFF('day', w.OPENED_DATE, CURRENT_DATE())  AS DAYS_OPEN,
       w.PART_NUMBER,
       p.DESCRIPTION,
       p.CRITICALITY,
       w.QTY_REQUIRED,
       COALESCE(s.QTY_AVAILABLE, 0)                    AS QTY_AVAILABLE,
       w.QTY_REQUIRED - COALESCE(s.QTY_AVAILABLE, 0)   AS SHORTFALL,
       r.AOG_RISK_SCORE,
       r.RISK_BAND,
       r.PRIMARY_DRIVER,
       po.PROJECTED_ARRIVAL_DATE                       AS NEXT_RECEIPT_DATE
FROM   FCT_WORK_ORDERS w
JOIN   V_DIM_PARTS_CURRENT p  ON p.PART_NUMBER = w.PART_NUMBER
LEFT   JOIN stock s           ON s.PART_NUMBER = w.PART_NUMBER
LEFT   JOIN current_scores r  ON r.PART_NUMBER = w.PART_NUMBER
LEFT   JOIN FCT_PURCHASE_ORDERS po
       ON po.PART_NUMBER = w.PART_NUMBER AND po.STATUS = 'OPEN'
WHERE  w.STATUS = 'OPEN'
  AND  w.QTY_REQUIRED > COALESCE(s.QTY_AVAILABLE, 0)
QUALIFY ROW_NUMBER() OVER (
        PARTITION BY w.WORK_ORDER_ID ORDER BY po.PROJECTED_ARRIVAL_DATE) = 1
;

-- KPI tile source: how much earlier the pipeline flags risk vs. a stockout.
-- Compares the first date a part crossed the HIGH band against the date its
-- available stock actually hit zero.
CREATE OR REPLACE VIEW V_AOG_EARLY_WARNING AS
WITH first_alert AS (
    SELECT PART_NUMBER, MIN(SCORED_AT) AS FIRST_HIGH_RISK_AT
    FROM   FCT_AOG_RISK
    WHERE  RISK_BAND IN ('CRITICAL', 'HIGH')
    GROUP BY PART_NUMBER
),
first_stockout AS (
    SELECT PART_NUMBER, MIN(SNAPSHOT_DATE) AS FIRST_STOCKOUT_DATE
    FROM   FCT_INVENTORY_SNAPSHOT
    WHERE  QTY_AVAILABLE <= 0
    GROUP BY PART_NUMBER
)
SELECT a.PART_NUMBER,
       a.FIRST_HIGH_RISK_AT,
       s.FIRST_STOCKOUT_DATE,
       DATEDIFF('hour', a.FIRST_HIGH_RISK_AT, s.FIRST_STOCKOUT_DATE) AS WARNING_LEAD_HOURS
FROM   first_alert a
JOIN   first_stockout s USING (PART_NUMBER)
WHERE  a.FIRST_HIGH_RISK_AT < s.FIRST_STOCKOUT_DATE
;
