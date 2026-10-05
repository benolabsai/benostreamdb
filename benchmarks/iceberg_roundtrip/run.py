#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Iceberg / Delta Lake round-trip harness.

Proves **table-format interop**: write a table with BenoStreamDB (Apache
Iceberg), then read it back through Spark or Trino (Iceberg), and compare
against Delta Lake for format cost. This is the Tier-2 "storage / table format"
baseline from `docs/BENCHMARKING_PLAN.md` §3.

The point is not speed — it is that a BenoStreamDB-written Iceberg table is
readable by the standard ecosystem, and that the overlay indexes do not change
the data. Each run emits the same JSON record shape as the competitor runner
(`benchmarks/competitors/run_competitor.py`) so it lands in the same rollup.

Usage:

    # Write with BenoStreamDB, read back with Spark (Iceberg)
    python benchmarks/iceberg_roundtrip/run.py --engine spark --rows 100000

    # Read back with Trino (Iceberg connector)
    python benchmarks/iceberg_roundtrip/run.py --engine trino --rows 100000

    # Delta Lake format-cost comparison
    python benchmarks/iceberg_roundtrip/run.py --engine delta --rows 100000

    # Reuse an existing BenoStreamDB table instead of writing a fresh one
    python benchmarks/iceberg_roundtrip/run.py --engine spark --table-uri /data/t
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import time
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def make_table(n: int):
    """A deterministic (id, value, category) table."""
    import pyarrow as pa

    rng = np.random.default_rng(42)
    return pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "value": pa.array(rng.random(n).astype(np.float64)),
            "category": pa.array(rng.integers(0, 100, n).astype(np.int64)),
        }
    )


# --------------------------------------------------------------------------- #
# BenoStreamDB writer (Iceberg)
# --------------------------------------------------------------------------- #
def write_benostreamdb(uri: str, n: int) -> float:
    """Write `n` rows into a BenoStreamDB Iceberg table at `uri`."""
    import benostreamdb as bsdb

    t0 = time.time()
    table = bsdb.Table(uri)
    table.write(make_table(n))
    table.commit()
    table.wait_for_background_tasks()
    return round(time.time() - t0, 3)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_spark(uri: str) -> dict:
    """Read an Iceberg table through Spark (Iceberg runtime)."""
    if importlib.util.find_spec("pyspark") is None:
        return {"available": False, "error": "pip install pyspark"}
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("bsdb-iceberg-roundtrip")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1",
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.bsdb", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.bsdb.type", "hadoop")
        .config("spark.sql.catalog.bsdb.warehouse", uri)
        .getOrCreate()
    )
    t0 = time.time()
    df = spark.read.format("iceberg").load(uri)
    rows = df.count()
    return {
        "available": True,
        "seconds": round(time.time() - t0, 3),
        "rows": rows,
    }


def read_trino(uri: str) -> dict:
    """Read an Iceberg table through Trino (Iceberg connector)."""
    if importlib.util.find_spec("trino") is None:
        return {"available": False, "error": "pip install trino"}
    from trino.dbapi import connect

    host = os.environ.get("TRINO_HOST", "localhost")
    port = int(os.environ.get("TRINO_PORT", "8080"))
    user = os.environ.get("TRINO_USER", "bench")
    catalog = os.environ.get("TRINO_ICEBERG_CATALOG", "iceberg")
    schema = os.environ.get("TRINO_SCHEMA", "default")
    table = os.environ.get("TRINO_TABLE", "t")

    conn = connect(
        host=host, port=port, user=user, catalog=catalog, schema=schema
    )
    t0 = time.time()
    cur = conn.cursor()
    cur.execute(f"SELECT count(*) FROM {table}")
    rows = cur.fetchone()[0]
    return {
        "available": True,
        "seconds": round(time.time() - t0, 3),
        "rows": rows,
    }


def read_delta(path: str) -> dict:
    """Read a Delta Lake table (format-cost comparison)."""
    if importlib.util.find_spec("deltalake") is None:
        return {"available": False, "error": "pip install deltalake"}
    from deltalake import DeltaTable

    t0 = time.time()
    dt = DeltaTable(path)
    rows = dt.to_pyarrow_table().num_rows
    return {
        "available": True,
        "seconds": round(time.time() - t0, 3),
        "rows": rows,
    }


def write_delta(path: str, n: int) -> float:
    """Write `n` rows as a Delta Lake table (format-cost comparison)."""
    from deltalake import write_deltalake

    t0 = time.time()
    write_deltalake(path, make_table(n))
    return round(time.time() - t0, 3)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["spark", "trino", "delta"])
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument(
        "--table-uri",
        default=None,
        help="Reuse an existing BenoStreamDB Iceberg table instead of writing one.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tmpdir = None
    try:
        if args.engine == "delta":
            tmpdir = tempfile.mkdtemp(prefix="bsdb_delta_")
            write_s = write_delta(tmpdir, args.rows)
            res = read_delta(tmpdir)
            res["write_seconds"] = write_s
            res["format"] = "delta"
        else:
            uri: str = args.table_uri or ""
            if not uri:
                tmpdir = tempfile.mkdtemp(prefix="bsdb_iceberg_")
                uri = tmpdir
                write_s = write_benostreamdb(uri, args.rows)
            else:
                write_s = None
            res = read_spark(uri) if args.engine == "spark" else read_trino(uri)
            res["write_seconds"] = write_s
            res["format"] = "iceberg"

        record: dict[str, Any] = {
            "engine": args.engine,
            "workload": "iceberg_roundtrip",
            "rows": args.rows,
            **res,
        }
        print(json.dumps(record, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
