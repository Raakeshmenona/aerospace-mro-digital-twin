"""Snowflake hub loader: staged Parquet + MERGE upserts.

Loading strategy, and why:

  * Write Parquet to an internal stage with PUT, then COPY INTO a transient
    staging table. Row-by-row inserts through the Python connector are two
    orders of magnitude slower and burn warehouse credits idling.
  * MERGE from staging into the target so re-running a DAG task is idempotent.
    An MRO pipeline gets re-run constantly (late supplier files, customs
    corrections) and duplicate work orders corrupt the AOG numbers.
  * Every target table carries a `_loaded_at` and `_batch_id` so a bad load
    can be identified and rolled back without a full refresh.
"""
from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Natural keys used for MERGE matching, per target table.
MERGE_KEYS: dict[str, list[str]] = {
    "DIM_PARTS": ["part_number"],
    "DIM_SUPPLIERS": ["supplier_id"],
    "DIM_SUPPLIER_PERFORMANCE": ["supplier_id", "as_of_date"],
    "FCT_PURCHASE_ORDERS": ["po_number", "po_line"],
    "FCT_WORK_ORDERS": ["work_order_id"],
    "FCT_INVENTORY_SNAPSHOT": ["warehouse_id", "part_number", "snapshot_date"],
    "FCT_AOG_RISK": ["part_number", "scored_at"],
    "FCT_DEMAND_FORECAST": ["part_number", "forecast_month"],
}

CHUNK_ROWS = 250_000


@dataclass
class SnowflakeConfig:
    account: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_ACCOUNT", ""))
    user: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_PASSWORD", ""), repr=False)
    role: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_ROLE", "MRO_LOADER"))
    warehouse: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_WAREHOUSE", "MRO_WH"))
    database: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_DATABASE", "MRO_HUB"))
    schema: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_SCHEMA", "CORE"))
    private_key_path: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", ""))

    def connect_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "account": self.account, "user": self.user, "role": self.role,
            "warehouse": self.warehouse, "database": self.database, "schema": self.schema,
        }
        # Key-pair auth is the norm for service accounts; password is the
        # local-dev fallback only.
        if self.private_key_path:
            args["private_key"] = _load_private_key(self.private_key_path)
        else:
            args["password"] = self.password
        return args


def _load_private_key(path: str) -> bytes:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    with open(path, "rb") as fh:
        key = serialization.load_pem_private_key(
            fh.read(),
            password=passphrase.encode() if passphrase else None,
            backend=default_backend(),
        )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


@contextmanager
def snowflake_connection(cfg: SnowflakeConfig) -> Iterator[Any]:
    import snowflake.connector

    conn = snowflake.connector.connect(**cfg.connect_args())
    try:
        yield conn
    finally:
        conn.close()


class HubLoader:
    """Batch upsert loader for the MRO supply chain hub."""

    def __init__(self, cfg: SnowflakeConfig, conn: Any = None):
        self.cfg = cfg
        self._conn = conn
        self.batch_id = uuid.uuid4().hex[:12]

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        if self._conn is not None:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
        else:
            with snowflake_connection(self.cfg) as conn:
                cur = conn.cursor()
                try:
                    yield cur
                finally:
                    cur.close()

    # ------------------------------------------------------------ loading

    def load(self, df: pd.DataFrame, table: str, merge_keys: Sequence[str] | None = None,
             mode: str = "merge") -> int:
        """Load a DataFrame into `table`.

        mode='merge'   idempotent upsert on the natural key (default)
        mode='append'  insert only, for immutable event/snapshot facts
        """
        if df.empty:
            log.info("%s: nothing to load", table)
            return 0

        keys = list(merge_keys or MERGE_KEYS.get(table.upper(), []))
        if mode == "merge" and not keys:
            raise ValueError(f"no merge keys configured for {table}; pass merge_keys= or use mode='append'")

        payload = self._stamp(df)
        with self._cursor() as cur:
            if mode == "append":
                rows = self._write_dataframe(cur, payload, table)
            else:
                staging = f"STG_{table.upper()}_{self.batch_id}"
                self._create_staging_like(cur, staging, table)
                self._write_dataframe(cur, payload, staging)
                rows = self._merge(cur, staging, table, keys, payload.columns)
                cur.execute(f"DROP TABLE IF EXISTS {staging}")
        log.info("%s: %s %d rows (batch %s)", table, mode, rows, self.batch_id)
        return rows

    def _stamp(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [c.upper() for c in out.columns]
        out["_LOADED_AT"] = datetime.utcnow()
        out["_BATCH_ID"] = self.batch_id
        # Snowflake's Parquet writer rejects pandas Period and tz-aware mixes.
        for col in out.columns:
            if isinstance(out[col].dtype, pd.PeriodDtype):
                out[col] = out[col].astype(str)
        return out

    def _create_staging_like(self, cur: Any, staging: str, target: str) -> None:
        # TRANSIENT: no fail-safe storage cost for data that lives 30 seconds.
        cur.execute(f"CREATE OR REPLACE TRANSIENT TABLE {staging} LIKE {target}")

    def _write_dataframe(self, cur: Any, df: pd.DataFrame, table: str) -> int:
        from snowflake.connector.pandas_tools import write_pandas

        total = 0
        for start in range(0, len(df), CHUNK_ROWS):
            chunk = df.iloc[start : start + CHUNK_ROWS]
            success, _nchunks, nrows, _ = write_pandas(
                cur.connection, chunk, table.upper(),
                quote_identifiers=False, auto_create_table=False,
            )
            if not success:
                raise RuntimeError(f"write_pandas failed for {table}")
            total += nrows
        return total

    def _merge(self, cur: Any, staging: str, target: str, keys: Sequence[str], columns: Sequence[str]) -> int:
        cols = [c.upper() for c in columns]
        key_cols = [k.upper() for k in keys]
        on_clause = " AND ".join(f"t.{k} = s.{k}" for k in key_cols)
        update_cols = [c for c in cols if c not in key_cols]
        set_clause = ", ".join(f"t.{c} = s.{c}" for c in update_cols)
        insert_cols = ", ".join(cols)
        insert_vals = ", ".join(f"s.{c}" for c in cols)

        sql = f"""
            MERGE INTO {target} t
            USING {staging} s
              ON {on_clause}
            WHEN MATCHED THEN UPDATE SET {set_clause}
            WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """
        cur.execute(sql)
        # Snowflake returns (rows_inserted, rows_updated) for MERGE.
        result = cur.fetchone() or (0, 0)
        return int(sum(int(v or 0) for v in result))

    # ---------------------------------------------------------- utilities

    def execute_sql_file(self, path: str | Path) -> None:
        """Run a .sql model file. Statements split on ';' at line ends."""
        text = Path(path).read_text(encoding="utf-8")
        statements = [s.strip() for s in text.split(";\n") if s.strip() and not s.strip().startswith("--")]
        with self._cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        log.info("applied %s (%d statements)", Path(path).name, len(statements))

    def deploy_models(self, models_dir: str | Path) -> None:
        """Apply every .sql model in dependency order (dim_ before fct_)."""
        files = sorted(Path(models_dir).glob("*.sql"), key=lambda p: (not p.name.startswith("dim_"), p.name))
        for f in files:
            self.execute_sql_file(f)

    def rollback_batch(self, table: str, batch_id: str) -> int:
        """Remove a bad load without a full table refresh."""
        with self._cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE _BATCH_ID = %s", (batch_id,))
            return int(cur.rowcount or 0)


def load_all(cfg: SnowflakeConfig, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Load a full nightly batch. Dimensions first so facts join cleanly."""
    order = ["DIM_PARTS", "DIM_SUPPLIERS", "DIM_SUPPLIER_PERFORMANCE",
             "FCT_PURCHASE_ORDERS", "FCT_WORK_ORDERS", "FCT_INVENTORY_SNAPSHOT",
             "FCT_DEMAND_FORECAST", "FCT_AOG_RISK"]
    loader = HubLoader(cfg)
    counts: dict[str, int] = {}
    for table in order:
        if table in frames:
            counts[table] = loader.load(frames[table], table)
    return counts
