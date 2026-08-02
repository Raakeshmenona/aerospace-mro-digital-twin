"""PySpark: unify the 10M+ row part master across five source systems.

The hard part of an MRO digital twin is not volume, it is that the same
physical part appears under different numbers in every system:

  ERP        HTL-4471-002
  Supplier   HTL4471002
  WMS        HTL 4471-002 REV B
  Customs    HTL-4471-002/B

This module normalizes part numbers, resolves interchangeability chains
(supersession: part A superseded by B superseded by C) and produces a single
`part_uid` every downstream table joins on.

Broadcast-joins the small dimensions and salts the skewed catalog join -
a handful of common fasteners account for a large share of all rows.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

log = logging.getLogger(__name__)

# Rows below this per-key count join normally; above it we salt.
SKEW_THRESHOLD = 100_000
SALT_BUCKETS = 32
# Dimensions under this row count are broadcast rather than shuffled.
BROADCAST_ROW_LIMIT = 2_000_000


def get_spark(app_name: str = "mro_parts_catalog") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.sql.autoBroadcastJoinThreshold", str(64 * 1024 * 1024))
        .getOrCreate()
    )


def normalize_part_number(col: F.Column) -> F.Column:
    """Canonical part number key.

    Upper-cases, strips revision suffixes and drops every non-alphanumeric
    character. Revision suffixes are deliberately dropped: a REV B part is the
    same part for planning purposes, and interchangeability is handled
    separately by the supersession chain.
    """
    cleaned = F.upper(F.trim(col))
    cleaned = F.regexp_replace(cleaned, r"\s*(REV|REVISION)\s*[A-Z0-9]+$", "")
    cleaned = F.regexp_replace(cleaned, r"/[A-Z0-9]{1,3}$", "")
    return F.regexp_replace(cleaned, r"[^A-Z0-9]", "")


def add_part_key(df: DataFrame, source_col: str = "part_number") -> DataFrame:
    return df.withColumn("part_key", normalize_part_number(F.col(source_col)))


def resolve_supersessions(supersessions: DataFrame, max_depth: int = 10) -> DataFrame:
    """Walk supersession chains to their terminal part.

    Input:  old_part_key -> new_part_key
    Output: part_key -> master_part_key (the current, orderable part)

    Iterative rather than recursive because Spark has no graph traversal
    primitive here, and chains in real catalogs are short (< 10 hops).
    """
    chain = supersessions.select(
        F.col("old_part_key").alias("part_key"),
        F.col("new_part_key").alias("master_part_key"),
    ).distinct()

    for depth in range(max_depth):
        nxt = supersessions.select(
            F.col("old_part_key").alias("master_part_key"),
            F.col("new_part_key").alias("next_key"),
        )
        joined = chain.join(F.broadcast(nxt), on="master_part_key", how="left")
        advanced = joined.withColumn(
            "master_part_key", F.coalesce(F.col("next_key"), F.col("master_part_key"))
        ).drop("next_key")
        # Stop early once nothing moved - avoids max_depth wasted shuffles.
        if advanced.join(chain, ["part_key", "master_part_key"], "left_anti").isEmpty():
            log.info("supersession chains converged at depth %d", depth)
            chain = advanced
            break
        chain = advanced.checkpoint(eager=False) if depth % 3 == 2 else advanced

    # A part that supersedes nothing is its own master; that is added by the
    # left join in build_part_master, not here.
    return chain.dropDuplicates(["part_key"])


def _salt(df: DataFrame, key: str, buckets: int = SALT_BUCKETS) -> DataFrame:
    return df.withColumn("_salt", (F.rand(seed=17) * buckets).cast("int"))


def _explode_salt(df: DataFrame, buckets: int = SALT_BUCKETS) -> DataFrame:
    return df.withColumn("_salt", F.explode(F.array([F.lit(i) for i in range(buckets)])))


def skew_aware_join(left: DataFrame, right: DataFrame, key: str, how: str = "left") -> DataFrame:
    """Join where `left` has heavy key skew and `right` is the small side.

    Salts the hot side and replicates the small side across salt buckets so no
    single Spark task inherits every row for a common fastener part number.
    """
    hot_keys = (
        left.groupBy(key).count().filter(F.col("count") > SKEW_THRESHOLD).select(key).cache()
    )
    if hot_keys.isEmpty():
        hot_keys.unpersist()
        return left.join(right, on=key, how=how)

    log.info("salting join on %d skewed keys", hot_keys.count())
    hot_left = left.join(F.broadcast(hot_keys), on=key, how="left_semi")
    cold_left = left.join(F.broadcast(hot_keys), on=key, how="left_anti")

    hot_joined = _salt(hot_left, key).join(_explode_salt(right), on=[key, "_salt"], how=how).drop("_salt")
    cold_joined = cold_left.join(right, on=key, how=how)
    hot_keys.unpersist()
    return hot_joined.unionByName(cold_joined, allowMissingColumns=True)


def build_part_master(
    erp_parts: DataFrame,
    supplier_parts: DataFrame,
    wms_parts: DataFrame,
    supersessions: DataFrame | None = None,
) -> DataFrame:
    """Produce the unified part master with a stable `part_uid`.

    Source precedence: ERP is the system of record for engineering attributes,
    supplier feeds win on price and lead time, WMS wins on physical location.
    """
    erp = add_part_key(erp_parts).withColumn("source_system", F.lit("ERP"))
    sup = add_part_key(supplier_parts).withColumn("source_system", F.lit("SUPPLIER"))
    wms = add_part_key(wms_parts).withColumn("source_system", F.lit("WMS"))

    all_keys = (
        erp.select("part_key").union(sup.select("part_key")).union(wms.select("part_key")).distinct()
    )

    if supersessions is not None:
        resolved = resolve_supersessions(add_supersession_keys(supersessions))
        all_keys = (
            all_keys.join(F.broadcast(resolved), on="part_key", how="left")
            .withColumn("master_part_key", F.coalesce("master_part_key", "part_key"))
        )
    else:
        all_keys = all_keys.withColumn("master_part_key", F.col("part_key"))

    # Deterministic surrogate key - stable across runs, unlike monotonically_increasing_id.
    all_keys = all_keys.withColumn("part_uid", F.sha2(F.col("master_part_key"), 256).substr(1, 16))

    erp_attrs = erp.select(
        "part_key",
        F.col("part_number").alias("erp_part_number"),
        "description", "ata_chapter", "criticality", "unit_cost_usd",
        "shelf_life_months", "repairable_flag",
    ).dropDuplicates(["part_key"])

    sup_attrs = (
        sup.groupBy("part_key")
        .agg(
            F.min("quoted_lead_time_days").alias("best_quoted_lead_time_days"),
            F.min("unit_price_usd").alias("best_supplier_price_usd"),
            F.countDistinct("supplier_id").alias("supplier_count"),
            F.sum("qty_available").alias("supplier_qty_available"),
        )
    )

    wms_attrs = (
        wms.groupBy("part_key")
        .agg(
            F.sum("qty_available").alias("qty_available"),
            F.sum("qty_on_hand").alias("qty_on_hand"),
            F.countDistinct("warehouse_id").alias("stocking_locations"),
        )
    )

    master = (
        all_keys
        .join(F.broadcast(erp_attrs) if _is_small(erp_attrs) else erp_attrs, "part_key", "left")
        .join(sup_attrs, "part_key", "left")
        .join(wms_attrs, "part_key", "left")
        .withColumn("is_superseded", F.col("part_key") != F.col("master_part_key"))
        .withColumn("single_source_flag", F.coalesce(F.col("supplier_count"), F.lit(0)) <= 1)
        .withColumn("ingested_at", F.current_timestamp())
    )

    for col in ("qty_available", "qty_on_hand", "supplier_qty_available", "supplier_count", "stocking_locations"):
        master = master.withColumn(col, F.coalesce(F.col(col), F.lit(0)))

    return master


def add_supersession_keys(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("old_part_key", normalize_part_number(F.col("old_part_number")))
        .withColumn("new_part_key", normalize_part_number(F.col("new_part_number")))
        .select("old_part_key", "new_part_key")
        .filter(F.col("old_part_key") != F.col("new_part_key"))
    )


def _is_small(df: DataFrame) -> bool:
    """Cheap broadcast-eligibility check using Catalyst's size estimate rather
    than an expensive count() on a 10M row frame."""
    try:
        size = df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()
        return size < 64 * 1024 * 1024
    except Exception:
        return False


def catalog_coverage_report(master: DataFrame) -> DataFrame:
    """Which parts are missing attributes - the data quality gate for the
    95% parts availability accuracy KPI."""
    total = F.count("*")
    return master.agg(
        total.alias("total_parts"),
        F.sum(F.col("description").isNull().cast("int")).alias("missing_description"),
        F.sum(F.col("criticality").isNull().cast("int")).alias("missing_criticality"),
        F.sum((F.col("supplier_count") == 0).cast("int")).alias("no_supplier_link"),
        F.sum(F.col("single_source_flag").cast("int")).alias("single_sourced"),
        F.sum(F.col("is_superseded").cast("int")).alias("superseded_parts"),
        F.round(100 * F.avg(F.col("criticality").isNotNull().cast("double")), 2).alias("criticality_coverage_pct"),
    )


def write_catalog(master: DataFrame, path: str, partition_by: Iterable[str] = ("ata_chapter",)) -> None:
    (
        master.repartition(200, "part_uid")
        .write.mode("overwrite")
        .partitionBy(*partition_by)
        .parquet(path)
    )
    log.info("wrote unified part catalog to %s", path)
