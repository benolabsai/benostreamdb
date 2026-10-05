#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Production Workload Harness (Item 9).

Measures concurrency scaling, throughput (QPS), p99 inflation, and stability
under production multi-threaded conditions.

Usage:
  python benchmarks/production_workload.py --workload concurrent --sweep
  python benchmarks/production_workload.py --workload maintenance
"""

import argparse
import concurrent.futures
import json
import os
import platform
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa

try:
    import benostreamdb
except ImportError:
    pass


def create_benchmark_table(db_path: str, n_rows: int = 20000, dim: int = 128) -> None:
    """Creates a temporary benchmark table with HNSW-TQ8 index."""
    schema = pa.schema([
        ("id", pa.int64()),
        ("embedding", pa.list_(pa.float32(), dim)),
    ])
    table = benostreamdb.Table.create(f"file://{db_path}", schema)
    table.add_index(
        "embedding",
        {"type": "hnsw_tq8", "complexity": 16, "quality": 200, "metric": "l2"},
    )

    chunk_size = 5000
    for i in range(0, n_rows, chunk_size):
        chunk_n = min(chunk_size, n_rows - i)
        vecs = np.random.rand(chunk_n, dim).astype(np.float32)
        # Normalize
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        ids = np.arange(i, i + chunk_n, dtype=np.int64)

        batch = pa.RecordBatch.from_arrays([
            pa.array(ids),
            pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), dim),
        ], schema=schema)
        table.write(pa.Table.from_batches([batch]))
        table.commit()

    table.wait_for_background_tasks()


def query_worker(db_path: str, num_queries: int, stop_event: threading.Event, latencies: list, dim: int = 128):
    """Executes vector search queries concurrently and records latency."""
    try:
        table = benostreamdb.Table(f"file://{db_path}")
        count = 0
        while not stop_event.is_set() and count < num_queries:
            q = np.random.rand(dim).astype(np.float32)
            q /= np.linalg.norm(q)

            t0 = time.perf_counter()
            _ = table.search("embedding", q.tolist(), k=10)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # ms
            count += 1
    except Exception as e:
        print(f"Worker error: {e}")


def run_concurrency_level(db_path: str, concurrency: int, queries_total: int, dim: int = 128) -> Dict[str, Any]:
    """Runs concurrent query workers at a specific concurrency level."""
    queries_per_thread = max(1, queries_total // concurrency)
    stop_event = threading.Event()
    latencies = []

    # Warmup
    table = benostreamdb.Table(f"file://{db_path}")
    for _ in range(10):
        q = np.random.rand(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        _ = table.search("embedding", q.tolist(), k=10)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(query_worker, db_path, queries_per_thread, stop_event, latencies, dim)
            for _ in range(concurrency)
        ]
        concurrent.futures.wait(futures)
    total_time = time.perf_counter() - t0

    actual_queries = len(latencies)
    qps = actual_queries / total_time if total_time > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_queries": actual_queries,
        "duration_s": round(total_time, 3),
        "qps": round(qps, 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        "p90_ms": round(float(np.percentile(latencies, 90)), 2) if latencies else 0.0,
        "p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0,
        "mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
    }


def run_concurrent_sweep(args) -> Dict[str, Any]:
    """Sweeps concurrency from 1 to 32 worker threads."""
    created_temp = False
    db_path = args.db
    dim = 128

    if not db_path or not os.path.exists(db_path):
        temp_dir = tempfile.mkdtemp(prefix="bsdb_conc_")
        print(f"Creating benchmark table in {temp_dir} (20,000 vectors, 128-d, HNSW-TQ8)...", flush=True)
        t_build_0 = time.time()
        create_benchmark_table(temp_dir, n_rows=20000, dim=dim)
        print(f"Benchmark table created in {time.time() - t_build_0:.2f}s", flush=True)
        db_path = temp_dir
        created_temp = True

    concurrency_levels = [1, 2, 4, 8, 16, 32]
    total_queries_per_level = args.queries_total or 1000

    print(f"\nBenchmarking multi-client concurrency scaling across {concurrency_levels} threads...", flush=True)
    results_by_level = []

    try:
        base_qps = 1.0
        for idx, c in enumerate(concurrency_levels):
            print(f"  Testing Concurrency = {c} threads ({total_queries_per_level} queries)...", end="", flush=True)
            res = run_concurrency_level(db_path, c, total_queries_per_level, dim=dim)
            if idx == 0:
                base_qps = max(res["qps"], 1.0)
            res["speedup"] = round(res["qps"] / base_qps, 2)
            print(f" -> QPS: {res['qps']:.1f}, p50: {res['p50_ms']:.2f}ms, p99: {res['p99_ms']:.2f}ms, speedup: {res['speedup']:.2f}x", flush=True)
            results_by_level.append(res)

        report = {
            "engine": "benostreamdb",
            "workload": "concurrent_scaling",
            "host": f"{platform.processor() or platform.machine()} ({platform.system()})",
            "levels": results_by_level,
        }

        # Write markdown table
        md_lines = [
            "# Multi-Client Concurrency Scaling Benchmark",
            "",
            "- **Engine**: BenoStreamDB",
            f"- **Host**: {report['host']}",
            "- **Workload**: Concurrent ANN Vector Queries (HNSW-TQ8, Top-10)",
            f"- **Queries per Concurrency Tier**: {total_queries_per_level:,}",
            "",
            "| Concurrency (Threads) | Throughput (QPS) | Scaling Speedup | p50 Latency | p90 Latency | p99 Latency |",
            "|---|---|---|---|---|---|",
        ]
        for row in results_by_level:
            md_lines.append(
                f"| **{row['concurrency']}** | **{row['qps']:.1f}** | **{row['speedup']:.2f}x** | "
                f"{row['p50_ms']:.2f} ms | {row['p90_ms']:.2f} ms | {row['p99_ms']:.2f} ms |"
            )

        md_report = "\n".join(md_lines) + "\n"
        out_md = "benchmarks/results/production_concurrency.md"
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w") as f:
            f.write(md_report)
        print(f"\nWrote concurrency scaling report to {out_md}")

        return report

    finally:
        if created_temp:
            shutil.rmtree(db_path, ignore_errors=True)


def run_maintenance(args) -> Dict[str, Any]:
    temp_dir = tempfile.mkdtemp(prefix="bsdb_maint_")
    print(f"Generating disposable synthetic dataset for maintenance test in {temp_dir}...")

    dim = 384
    schema = pa.schema([
        ("id", pa.int64()),
        ("embedding", pa.list_(pa.float32(), dim)),
    ])

    t = benostreamdb.Table.create(f"file://{temp_dir}", schema)
    t.add_index("embedding", {"type": "hnsw", "device": "cpu"})

    # Write 50 small segments (simulate a heavily fragmented table)
    for i in range(50):
        vecs = np.random.rand(2000, dim).astype(np.float32)
        ids = np.arange(i * 2000, (i + 1) * 2000)
        batch = pa.RecordBatch.from_arrays([
            pa.array(ids),
            pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), dim),
        ], schema=schema)
        t.write(pa.Table.from_batches([batch]))
        t.commit()

    print(f"Synthetic dataset created with {len(t)} rows.")

    print("Gathering baseline latency...")
    baseline_latencies = []
    stop_event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for _ in range(args.concurrency):
            futures.append(executor.submit(query_worker, temp_dir, 50, stop_event, baseline_latencies, dim))
        concurrent.futures.wait(futures)

    base_p99 = np.percentile(baseline_latencies, 99) if baseline_latencies else 0.0
    print(f"Baseline p99: {base_p99:.2f} ms")

    print("Triggering background compaction while saturating reads...")
    maint_latencies = []
    stop_maint = threading.Event()

    def compact_worker():
        t0 = time.time()
        tbl = benostreamdb.Table(f"file://{temp_dir}")
        tbl.rewrite_data_files(2_000_000_000)
        print(f"Compaction finished in {time.time() - t0:.2f}s")
        stop_maint.set()

    maint_thread = threading.Thread(target=compact_worker)
    maint_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for _ in range(args.concurrency):
            futures.append(executor.submit(query_worker, temp_dir, 1000000, stop_maint, maint_latencies, dim))

        maint_thread.join()
        concurrent.futures.wait(futures)

    maint_p99 = np.percentile(maint_latencies, 99) if maint_latencies else 0.0
    print(f"Under-maintenance p99: {maint_p99:.2f} ms")

    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "engine": "benostreamdb",
        "workload": "maintenance_saturation",
        "concurrency": args.concurrency,
        "baseline_p99_ms": round(base_p99, 2),
        "maintenance_p99_ms": round(maint_p99, 2),
        "inflation_factor": round(maint_p99 / base_p99, 2) if base_p99 > 0 else 0,
        "queries_during_compaction": len(maint_latencies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="Path to BenoStreamDB table")
    parser.add_argument("--workload", choices=["concurrent", "maintenance"], required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--queries-total", type=int, default=1000)
    parser.add_argument("--queries-per-thread", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="Sweep concurrency across 1 to 32 threads")
    args = parser.parse_args()

    if args.workload == "concurrent":
        res = run_concurrent_sweep(args)
    elif args.workload == "maintenance":
        res = run_maintenance(args)

    out_file = f"benchmarks/results/production_{args.workload}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
