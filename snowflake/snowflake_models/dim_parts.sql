-- dim_parts: the spoke every MRO query starts from.
--
-- SCD Type 2 on the attributes that change planning behaviour (criticality,
-- unit cost, supersession). An engineer re-classifying a part from GO to NO_GO
-- changes historic AOG exposure, so we keep the version history rather than
-- overwriting - otherwise last quarter's risk numbers silently rewrite
-- themselves and the dashboard loses its audit trail.

CREATE TABLE IF NOT EXISTS DIM_PARTS (
    PART_SK              NUMBER IDENTITY(1,1) PRIMARY KEY,
    PART_NUMBER          VARCHAR(50)   NOT NULL,
    PART_UID             VARCHAR(32),              -- stable hash of master key
    MASTER_PART_NUMBER   VARCHAR(50),              -- terminal supersession part
    DESCRIPTION          VARCHAR(500),
    ATA_CHAPTER          VARCHAR(4),
    ATA_CHAPTER_NAME     VARCHAR(100),
    CRITICALITY          VARCHAR(10),              -- NO_GO / GO_IF / GO
    UNIT_COST_USD        NUMBER(14,2),
    SHELF_LIFE_MONTHS    NUMBER(5,0),
    REPAIRABLE_FLAG      BOOLEAN,
    IS_SUPERSEDED        BOOLEAN     DEFAULT FALSE,
    SINGLE_SOURCE_FLAG   BOOLEAN     DEFAULT FALSE,
    SUPPLIER_COUNT       NUMBER(5,0) DEFAULT 0,
    SOURCE_SYSTEM        VARCHAR(20),
    -- SCD2 control columns
    VALID_FROM           TIMESTAMP_NTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    VALID_TO             TIMESTAMP_NTZ,
    IS_CURRENT           BOOLEAN       NOT NULL DEFAULT TRUE,
    ROW_HASH             VARCHAR(64),              -- change detection
    _LOADED_AT           TIMESTAMP_NTZ,
    _BATCH_ID            VARCHAR(12)
)
CLUSTER BY (PART_NUMBER, IS_CURRENT)
COMMENT = 'Unified part master across ERP, supplier portals and WMS (SCD2)'
;

-- Close out versions whose tracked attributes changed, then insert the new
-- version. Two statements rather than one MERGE because Snowflake forbids
-- updating and inserting the same target row in a single MERGE pass.

CREATE OR REPLACE PROCEDURE SP_LOAD_DIM_PARTS(STAGING_TABLE VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    -- 1. Expire changed current rows
    LET expire_sql := 'UPDATE DIM_PARTS t
        SET VALID_TO = CURRENT_TIMESTAMP(), IS_CURRENT = FALSE
        FROM ' || :STAGING_TABLE || ' s
        WHERE t.PART_NUMBER = s.PART_NUMBER
          AND t.IS_CURRENT = TRUE
          AND t.ROW_HASH <> s.ROW_HASH';
    EXECUTE IMMEDIATE :expire_sql;

    -- 2. Insert new and changed versions
    LET insert_sql := 'INSERT INTO DIM_PARTS (
            PART_NUMBER, PART_UID, MASTER_PART_NUMBER, DESCRIPTION, ATA_CHAPTER,
            CRITICALITY, UNIT_COST_USD, SHELF_LIFE_MONTHS, REPAIRABLE_FLAG,
            IS_SUPERSEDED, SINGLE_SOURCE_FLAG, SUPPLIER_COUNT, SOURCE_SYSTEM,
            ROW_HASH, VALID_FROM, IS_CURRENT, _LOADED_AT, _BATCH_ID)
        SELECT s.PART_NUMBER, s.PART_UID, s.MASTER_PART_NUMBER, s.DESCRIPTION,
               s.ATA_CHAPTER, s.CRITICALITY, s.UNIT_COST_USD, s.SHELF_LIFE_MONTHS,
               s.REPAIRABLE_FLAG, s.IS_SUPERSEDED, s.SINGLE_SOURCE_FLAG,
               s.SUPPLIER_COUNT, s.SOURCE_SYSTEM, s.ROW_HASH,
               CURRENT_TIMESTAMP(), TRUE, s._LOADED_AT, s._BATCH_ID
        FROM ' || :STAGING_TABLE || ' s
        LEFT JOIN DIM_PARTS t
               ON t.PART_NUMBER = s.PART_NUMBER AND t.IS_CURRENT = TRUE
        WHERE t.PART_NUMBER IS NULL';
    EXECUTE IMMEDIATE :insert_sql;

    RETURN 'DIM_PARTS load complete';
END;
$$
;

-- Current-state view: what every dashboard and downstream model joins to.
CREATE OR REPLACE VIEW V_DIM_PARTS_CURRENT AS
SELECT *
FROM   DIM_PARTS
WHERE  IS_CURRENT = TRUE
;

-- ATA chapter lookup keeps the heatmap axis readable without a Power BI hack.
CREATE OR REPLACE VIEW V_PARTS_BY_ATA AS
SELECT ATA_CHAPTER,
       ANY_VALUE(ATA_CHAPTER_NAME)                            AS ATA_CHAPTER_NAME,
       COUNT(*)                                               AS PART_COUNT,
       SUM(IFF(CRITICALITY = 'NO_GO', 1, 0))                  AS NO_GO_PARTS,
       SUM(IFF(SINGLE_SOURCE_FLAG, 1, 0))                     AS SINGLE_SOURCED_PARTS,
       ROUND(AVG(UNIT_COST_USD), 2)                           AS AVG_UNIT_COST_USD
FROM   V_DIM_PARTS_CURRENT
GROUP BY ATA_CHAPTER
;
