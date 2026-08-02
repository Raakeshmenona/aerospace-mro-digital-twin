-- dim_supplier_performance: the automated supplier scorecard.
--
-- Snapshot-per-day rather than current-state-only. Supplier performance
-- conversations are always retrospective ("you were at 62% in Q1, what
-- changed?"), so overwriting yesterday's score destroys the only evidence
-- that matters in a supplier review meeting.

CREATE TABLE IF NOT EXISTS DIM_SUPPLIERS (
    SUPPLIER_SK          NUMBER IDENTITY(1,1) PRIMARY KEY,
    SUPPLIER_ID          VARCHAR(20) NOT NULL,
    SUPPLIER_NAME        VARCHAR(200),
    COUNTRY              VARCHAR(4),
    BASE_LEAD_TIME_DAYS  NUMBER(6,0),      -- contracted, not observed
    QUALITY_RATING       NUMBER(3,1),
    APPROVED_VENDOR      BOOLEAN,
    _LOADED_AT           TIMESTAMP_NTZ,
    _BATCH_ID            VARCHAR(12)
)
COMMENT = 'Supplier master from ERP vendor records'
;

CREATE TABLE IF NOT EXISTS DIM_SUPPLIER_PERFORMANCE (
    SUPPLIER_ID           VARCHAR(20)  NOT NULL,
    AS_OF_DATE            DATE         NOT NULL,
    N_DELIVERIES          NUMBER(8,0),
    -- Observed lead-time percentiles, recency-weighted. P80 is the planning
    -- number: P50 strands half of all orders, P95 over-buys safety stock.
    P50_DAYS              NUMBER(8,1),
    P80_DAYS              NUMBER(8,1),
    P95_DAYS              NUMBER(8,1),
    MEAN_DAYS             NUMBER(8,1),
    VOLATILITY_DAYS       NUMBER(8,1),
    ON_TIME_RATE          NUMBER(5,4),
    AVG_SLIP_DAYS         NUMBER(8,1),
    TREND_DAYS_PER_YEAR   NUMBER(8,2),     -- positive = degrading
    CONTRACT_GAP_DAYS     NUMBER(8,1),     -- observed P80 minus contracted
    OTD_SCORE             NUMBER(5,1),
    CONSISTENCY_SCORE     NUMBER(5,1),
    TREND_SCORE           NUMBER(5,1),
    COMPOSITE_SCORE       NUMBER(5,1),
    RISK_TIER             VARCHAR(12),     -- CRITICAL/WATCH/ACCEPTABLE/PREFERRED
    CONFIDENCE            VARCHAR(10),     -- HIGH/MEDIUM/LOW/FALLBACK
    _LOADED_AT            TIMESTAMP_NTZ,
    _BATCH_ID             VARCHAR(12),
    CONSTRAINT PK_SUPPLIER_PERF PRIMARY KEY (SUPPLIER_ID, AS_OF_DATE)
)
CLUSTER BY (AS_OF_DATE, RISK_TIER)
COMMENT = 'Daily supplier scorecard snapshots (observed lead time + OTD)'
;

-- Latest scorecard for the Power BI supplier page.
CREATE OR REPLACE VIEW V_SUPPLIER_SCORECARD AS
WITH latest AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY SUPPLIER_ID ORDER BY AS_OF_DATE DESC) AS RN
    FROM   DIM_SUPPLIER_PERFORMANCE
)
SELECT l.SUPPLIER_ID,
       s.SUPPLIER_NAME,
       s.COUNTRY,
       s.APPROVED_VENDOR,
       l.AS_OF_DATE,
       l.N_DELIVERIES,
       l.P80_DAYS,
       s.BASE_LEAD_TIME_DAYS,
       l.CONTRACT_GAP_DAYS,
       ROUND(l.ON_TIME_RATE * 100, 1)      AS ON_TIME_PCT,
       l.VOLATILITY_DAYS,
       l.TREND_DAYS_PER_YEAR,
       l.COMPOSITE_SCORE,
       l.RISK_TIER,
       l.CONFIDENCE
FROM   latest l
JOIN   DIM_SUPPLIERS s USING (SUPPLIER_ID)
WHERE  l.RN = 1
;

-- Quarter-over-quarter movement: who is actually deteriorating, which is the
-- question that triggers a supplier corrective action request (SCAR).
CREATE OR REPLACE VIEW V_SUPPLIER_TREND AS
WITH quarterly AS (
    SELECT SUPPLIER_ID,
           DATE_TRUNC('QUARTER', AS_OF_DATE)      AS QTR,
           AVG(COMPOSITE_SCORE)                   AS AVG_SCORE,
           AVG(ON_TIME_RATE) * 100                AS AVG_OTD_PCT
    FROM   DIM_SUPPLIER_PERFORMANCE
    GROUP BY 1, 2
)
SELECT SUPPLIER_ID,
       QTR,
       ROUND(AVG_SCORE, 1)                                              AS AVG_SCORE,
       ROUND(AVG_OTD_PCT, 1)                                            AS AVG_OTD_PCT,
       ROUND(AVG_SCORE - LAG(AVG_SCORE) OVER (
             PARTITION BY SUPPLIER_ID ORDER BY QTR), 1)                 AS SCORE_DELTA_QOQ
FROM   quarterly
;

-- Single-source exposure: parts where the only approved supplier is in the
-- bottom tier. This is the report that justifies a dual-sourcing programme.
CREATE OR REPLACE VIEW V_SINGLE_SOURCE_EXPOSURE AS
SELECT p.PART_NUMBER,
       p.DESCRIPTION,
       p.CRITICALITY,
       p.UNIT_COST_USD,
       sc.SUPPLIER_ID,
       sc.SUPPLIER_NAME,
       sc.COMPOSITE_SCORE,
       sc.RISK_TIER,
       sc.P80_DAYS
FROM   V_DIM_PARTS_CURRENT p
JOIN   FCT_PURCHASE_ORDERS po ON po.PART_NUMBER = p.PART_NUMBER
JOIN   V_SUPPLIER_SCORECARD sc ON sc.SUPPLIER_ID = po.SUPPLIER_ID
WHERE  p.SINGLE_SOURCE_FLAG = TRUE
  AND  p.CRITICALITY IN ('NO_GO', 'GO_IF')
  AND  sc.RISK_TIER IN ('CRITICAL', 'WATCH')
QUALIFY ROW_NUMBER() OVER (PARTITION BY p.PART_NUMBER ORDER BY po.ORDER_DATE DESC) = 1
;
