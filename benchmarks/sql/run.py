#!/usr/bin/env python3
"""SQL Benchmark Runner: ClickBench & OLAP query evaluation across engines.

Supported Engines:
- benostreamdb (Native Arrow/DataFusion-powered analytical SQL engine)
- duckdb (In-process analytical SQL database)
- datafusion (Apache DataFusion reference)
- clickhouse (ClickHouse analytical DBMS)

Queries:
Standard ClickBench query subset (Q0 - Q9) covering aggregations, filters,
group-by, distinct counts, and order-by/limit operations.

Usage:
    python benchmarks/sql/run.py --rows 500000 --engines benostreamdb,duckdb,datafusion
    python benchmarks/sql/run.py --parquet data/hits.parquet --out results/clickbench.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")

CLICKBENCH_QUERIES = [
    ("Q0 (count)", 'SELECT count(*) FROM t'),
    ("Q1 (filter count)", 'SELECT count(*) FROM t WHERE "AdvEngineID" <> 0'),
    ("Q2 (multi-agg)", 'SELECT sum("AdvEngineID"), count(*), avg("ResolutionWidth") FROM t'),
    ("Q3 (avg int)", 'SELECT avg("UserID") FROM t'),
    ("Q4 (distinct user)", 'SELECT count(DISTINCT "UserID") FROM t'),
    ("Q5 (distinct phrase)", 'SELECT count(DISTINCT "SearchPhrase") FROM t'),
    ("Q6 (min/max date)", 'SELECT min("EventDate"), max("EventDate") FROM t'),
    ("Q7 (group by agg)", 'SELECT "AdvEngineID", count(*) AS c FROM t WHERE "AdvEngineID" <> 0 GROUP BY "AdvEngineID" ORDER BY c DESC LIMIT 10'),
    ("Q8 (group by distinct)", 'SELECT "RegionID", count(DISTINCT "UserID") AS u FROM t GROUP BY "RegionID" ORDER BY u DESC LIMIT 10'),
    ("Q9 (string filter group)", 'SELECT "MobilePhoneModel", count(DISTINCT "UserID") AS u FROM t WHERE "MobilePhoneModel" <> \'\' GROUP BY "MobilePhoneModel" ORDER BY u DESC LIMIT 10'),
]


def ensure_clickbench_dataset(parquet_path: Optional[str], rows: int = 500_000, seed: int = 42) -> str:
    """Returns path to a Parquet dataset matching the ClickBench schema."""
    if parquet_path and os.path.exists(parquet_path):
        return parquet_path

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"synth_hits_{rows}_s{seed}.parquet")
    if os.path.exists(cache_path):
        return cache_path

    print(f"Generating synthetic ClickBench hits dataset ({rows:,} rows)...", flush=True)
    rng = np.random.default_rng(seed)

    # Synthetic realistic ClickBench data distributions
    watch_ids = rng.integers(1, 100_000_000, size=rows, dtype=np.int64)
    adv_engine_ids = rng.choice([0, 1, 2, 3, 4, 5], size=rows, p=[0.7, 0.1, 0.08, 0.05, 0.04, 0.03])
    res_widths = rng.choice([1024, 1280, 1366, 1440, 1920, 2560, 3840], size=rows)
    user_ids = rng.integers(1, max(100, rows // 5), size=rows, dtype=np.int64)
    
    phrases = ["", "search query", "online store", "benostreamdb speed", "high performance vector db", "graph analytics", "clickbench olap"]
    search_phrases = rng.choice(phrases, size=rows)
    
    base_date = np.datetime64("2024-01-01")
    event_dates = (base_date + rng.integers(0, 365, size=rows)).astype("datetime64[s]")
    region_ids = rng.integers(1, 200, size=rows, dtype=np.int32)

    phones = ["", "iPhone 15", "Samsung Galaxy S24", "Google Pixel 8", "Xiaomi 14", "OnePlus 12"]
    mobile_models = rng.choice(phones, size=rows)

    table = pa.Table.from_arrays(
        [
            pa.array(watch_ids),
            pa.array(adv_engine_ids),
            pa.array(res_widths),
            pa.array(user_ids),
            pa.array(search_phrases),
            pa.array(event_dates),
            pa.array(region_ids),
            pa.array(mobile_models),
        ],
        names=[
            "WatchID",
            "AdvEngineID",
            "ResolutionWidth",
            "UserID",
            "SearchPhrase",
            "EventDate",
            "RegionID",
            "MobilePhoneModel",
        ],
    )

    pq.write_table(table, cache_path, compression="snappy")
    print(f"Saved synthetic ClickBench parquet to {cache_path} ({os.path.getsize(cache_path) / 1e6:.1f} MB)", flush=True)
    return cache_path


def run_duckdb(parquet_path: str, queries: List[tuple[str, str]], warm_runs: int = 2) -> Dict[str, Any]:
    if not importlib.util.find_spec("duckdb"):
        return {"engine": "duckdb", "available": False, "error": "duckdb not installed"}
    import duckdb

    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{parquet_path}')")

    results = []
    for q_name, sql in queries:
        # Cold run
        t0 = time.perf_counter()
        res = con.execute(sql).fetchall()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm runs
        warm_times = []
        for _ in range(warm_runs):
            t1 = time.perf_counter()
            _ = con.execute(sql).fetchall()
            warm_times.append((time.perf_counter() - t1) * 1000)

        warm_ms = float(np.median(warm_times)) if warm_times else cold_ms
        results.append({
            "name": q_name,
            "cold_ms": round(cold_ms, 2),
            "warm_ms": round(warm_ms, 2),
            "rows": len(res),
        })

    return {"engine": "duckdb", "available": True, "queries": results}


def run_datafusion(parquet_path: str, queries: List[tuple[str, str]], warm_runs: int = 2) -> Dict[str, Any]:
    if not importlib.util.find_spec("datafusion"):
        return {"engine": "datafusion", "available": False, "error": "datafusion not installed"}
    import datafusion

    ctx = datafusion.SessionContext()
    ctx.register_parquet("t", parquet_path)

    results = []
    for q_name, sql in queries:
        try:
            t0 = time.perf_counter()
            batches = ctx.sql(sql).collect()
            cold_ms = (time.perf_counter() - t0) * 1000
            total_rows = sum(b.num_rows for b in batches)

            warm_times = []
            for _ in range(warm_runs):
                t1 = time.perf_counter()
                _ = ctx.sql(sql).collect()
                warm_times.append((time.perf_counter() - t1) * 1000)

            warm_ms = float(np.median(warm_times)) if warm_times else cold_ms
            results.append({
                "name": q_name,
                "cold_ms": round(cold_ms, 2),
                "warm_ms": round(warm_ms, 2),
                "rows": total_rows,
            })
        except Exception as e:
            results.append({"name": q_name, "error": str(e)})

    return {"engine": "datafusion", "available": True, "queries": results}


def run_benostreamdb(parquet_path: str, queries: List[tuple[str, str]], warm_runs: int = 2) -> Dict[str, Any]:
    import benostreamdb as bsdb

    tmpdir = tempfile.mkdtemp(prefix="bsdb_sql_")
    try:
        t_ingest_0 = time.perf_counter()
        table = bsdb.Table(tmpdir)
        pa_table = pq.read_table(parquet_path)
        table.write(pa_table)
        table.commit()
        table.wait_for_background_tasks()
        ingest_s = time.perf_counter() - t_ingest_0

        results = []
        for q_name, sql in queries:
            try:
                t0 = time.perf_counter()
                res = table.sql(sql)
                cold_ms = (time.perf_counter() - t0) * 1000

                warm_times = []
                for _ in range(warm_runs):
                    t1 = time.perf_counter()
                    _ = table.sql(sql)
                    warm_times.append((time.perf_counter() - t1) * 1000)

                warm_ms = float(np.median(warm_times)) if warm_times else cold_ms
                results.append({
                    "name": q_name,
                    "cold_ms": round(cold_ms, 2),
                    "warm_ms": round(warm_ms, 2),
                    "rows": len(res),
                })
            except Exception as e:
                results.append({"name": q_name, "error": str(e)})

        return {
            "engine": "benostreamdb",
            "available": True,
            "ingest_s": round(ingest_s, 2),
            "queries": results,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="ClickBench & SQL Benchmark Runner")
    parser.add_argument("--parquet", default=None, help="Path to parquet table")
    parser.add_argument("--rows", type=int, default=500_000, help="Row count for synthetic ClickBench dataset")
    parser.add_argument("--engines", default="benostreamdb,duckdb,datafusion", help="Comma-separated engines")
    parser.add_argument("--warm-runs", type=int, default=2, help="Number of warm iterations")
    parser.add_argument("--out", default=None, help="Output markdown path")
    args = parser.parse_args()

    parquet_path = ensure_clickbench_dataset(args.parquet, args.rows)
    file_mb = os.path.getsize(parquet_path) / 1e6

    print(f"Dataset ready: {parquet_path} ({file_mb:.1f} MB)", flush=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    runners = {
        "benostreamdb": run_benostreamdb,
        "duckdb": run_duckdb,
        "datafusion": run_datafusion,
    }

    engine_results = {}
    for engine in engines:
        runner = runners.get(engine)
        if not runner:
            print(f"Skipping unknown engine: {engine}")
            continue
        print(f"Running SQL benchmark on {engine}...", flush=True)
        res = runner(parquet_path, CLICKBENCH_QUERIES, warm_runs=args.warm_runs)
        engine_results[engine] = res
        if res.get("available"):
            print(f"  -> {engine} completed {len(res['queries'])} queries")
        else:
            print(f"  -> {engine} unavailable: {res.get('error')}")

    # Build Comparative Markdown Report
    lines = [
        "# ClickBench SQL Benchmark Results",
        "",
        f"- **Rows**: {args.rows:,}",
        f"- **Dataset File**: `{os.path.basename(parquet_path)}` ({file_mb:.1f} MB)",
        f"- **Warm Iterations**: {args.warm_runs}",
        "",
        "### Warm Query Latency (ms) — Median",
        "",
    ]

    header = "| Query | " + " | ".join(engines) + " |"
    sep = "|---| " + " | ".join(["---"] * len(engines)) + " |"
    lines.extend([header, sep])

    for i, (q_name, _) in enumerate(CLICKBENCH_QUERIES):
        row = [f"**{q_name}**"]
        for eng in engines:
            e_data = engine_results.get(eng, {})
            if not e_data.get("available"):
                row.append("N/A")
                continue
            q_res = e_data["queries"][i]
            if "error" in q_res:
                row.append("Error")
            else:
                row.append(f"{q_res['warm_ms']:.1f} ms")
        lines.append("| " + " | ".join(row) + " |")

    # Ingest / Total time summary
    lines.extend([
        "",
        "### Ingestion & Total Execution Time",
        "",
        "| Engine | Ingestion (s) | Total Warm SQL Time (ms) | Status |",
        "|---|---|---|---|",
    ])

    for eng in engines:
        e_data = engine_results.get(eng, {})
        if not e_data.get("available"):
            lines.append(f"| {eng} | - | - | ❌ {e_data.get('error')} |")
            continue
        ingest_str = f"{e_data.get('ingest_s', 0):.2f}s" if "ingest_s" in e_data else "Direct Parquet Scan"
        total_warm_ms = sum(q.get("warm_ms", 0) for q in e_data.get("queries", []) if "warm_ms" in q)
        lines.append(f"| {eng} | {ingest_str} | {total_warm_ms:.1f} ms | ✅ Pass |")

    report_md = "\n".join(lines) + "\n"
    print("\n" + report_md)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Wrote SQL benchmark report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
