#!/usr/bin/env python3
"""3-Way Intra-Engine Ablation Test Harness for BenoStreamDB: SQ8 vs. Vanilla TQ8 vs. BenoQuant BQ8.

Evaluates three quantization and graph routing paradigms directly inside BenoStreamDB:
1. SQ8: Standard Uniform Scalar Quantization without coordinate rotation.
2. Vanilla TQ8: Google's Fast Walsh-Hadamard Transform (FWHT) without Rademacher sign modulation.
3. BenoQuant BQ8: Deterministic Rademacher-modulated FWHT (H * D).

Outputs:
- Recall@10
- Query Throughput (QPS & Pure Index QPS)
- Query Latency (p50 and p99 in ms)
- Index Build Time (s)
- Total Index Size on Disk (MB)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

CACHE_DIR = "/home/ralbright/projects/benostreamdb/benchmarks/ann_benchmarks/data"
DATASET_METRIC = {
    "sift-128-euclidean": "l2",
    "gist-960-euclidean": "l2",
    "fashion-mnist-784-euclidean": "l2",
    "mnist-784-euclidean": "l2",
    "glove-100-angular": "cosine",
    "glove-200-angular": "cosine",
    "nytimes-256-angular": "cosine",
    "lastfm-64-dot": "inner_product",
}


def load_dataset(dataset_name: str, limit: int = 0):
    path = os.path.join(CACHE_DIR, f"{dataset_name}.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int64) if "neighbors" in f else None

    if limit > 0 and limit < len(train):
        train = train[:limit]
        neighbors = None  # Force recomputation for subset

    return train, test, neighbors


def compute_ground_truth(train: np.ndarray, test: np.ndarray, q: int, k: int, metric: str) -> np.ndarray:
    truth = []
    if metric == "cosine":
        train_n = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-12)
        for i in range(q):
            query = test[i] / (np.linalg.norm(test[i]) + 1e-12)
            d = 1.0 - train_n @ query
            truth.append(np.argsort(d)[:k])
    elif metric in ("inner_product", "ip", "dot"):
        for i in range(q):
            d = -(train @ test[i])
            truth.append(np.argsort(d)[:k])
    else:
        for i in range(q):
            d = np.linalg.norm(train - test[i], axis=1)
            truth.append(np.argsort(d)[:k])
    return np.array(truth)


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total / (1024.0 * 1024.0)


def worker_run(dataset: str, mode: str, limit: int, queries: int, k: int, m: int, ef_c: int, ef_s: int, threads: int = 32):
    """Worker function executed inside an isolated subprocess."""
    import benostreamdb as bsdb

    metric = DATASET_METRIC.get(dataset, "l2")
    train, test, neighbors = load_dataset(dataset, limit)
    n, dim = train.shape
    q = min(queries, len(test))

    if neighbors is None:
        neighbors = compute_ground_truth(train, test, q, k, metric)

    db_path = os.path.join(CACHE_DIR, f"tmp_db_{dataset}_{mode}")
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    t0 = time.time()
    table = bsdb.Table(db_path)
    index_type = "hnsw_sq8" if mode == "sq8" else "hnsw_tq8"
    table.add_index(
        "embedding",
        {
            "type": index_type,
            "complexity": m,
            "quality": ef_c,
            "metric": metric,
        },
    )

    chunk = 50_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        df = pd.DataFrame(
            {
                "id": np.arange(start, end, dtype=np.int64),
                "embedding": [row.tolist() for row in train[start:end]],
            }
        )
        table.write(df)
    table.commit()
    table.wait_for_background_tasks()
    build_time = time.time() - t0
    index_size = dir_size_mb(db_path)

    query_lists = [test[i].tolist() for i in range(q)]

    # 1. End-to-end Arrow queries (full search + ID projection across threads)
    def arrow_worker(i):
        vf = {
            "column": "embedding",
            "query": query_lists[i],
            "k": k,
            "ef_search": ef_s,
        }
        t1 = time.perf_counter()
        res = table.to_arrow(vector_filter=vf, columns=["id"])
        lat = time.perf_counter() - t1
        got = set(res.column("id").to_pylist()) if "id" in res.column_names else set()
        truth = set(neighbors[i][:k].tolist())
        hit = len(got & truth) / k
        return lat, hit

    t_wall_arrow = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        arrow_results = list(ex.map(arrow_worker, range(q)))
    wall_arrow = time.time() - t_wall_arrow

    latencies = [r[0] for r in arrow_results]
    hits = sum(r[1] for r in arrow_results)
    recall = hits / q
    qps = q / max(wall_arrow, 0.001)
    p50 = float(np.percentile(latencies, 50) * 1000)
    p99 = float(np.percentile(latencies, 99) * 1000)

    # 2. Pure index search queries (in-process index routing only across threads)
    def pure_worker(i):
        t1 = time.perf_counter()
        _ = table._inner.vector_search_scored("embedding", query_lists[i], k, metric, ef_s)
        return time.perf_counter() - t1

    t_wall_pure = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        pure_latencies = list(ex.map(pure_worker, range(q)))
    wall_pure = time.time() - t_wall_pure

    pure_qps = q / max(wall_pure, 0.001)
    pure_p50 = float(np.percentile(pure_latencies, 50) * 1000)
    pure_p99 = float(np.percentile(pure_latencies, 99) * 1000)

    # Clean up DB
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    result = {
        "dataset": dataset,
        "mode": mode,
        "n_train": n,
        "dim": dim,
        "k": k,
        "m": m,
        "ef_construction": ef_c,
        "ef_search": ef_s,
        "threads": threads,
        "recall_at_k": round(recall, 4),
        "qps": round(qps, 1),
        "pure_qps": round(pure_qps, 1),
        "p50_ms": round(p50, 2),
        "p99_ms": round(p99, 2),
        "pure_p50_ms": round(pure_p50, 2),
        "pure_p99_ms": round(pure_p99, 2),
        "build_time_s": round(build_time, 2),
        "index_size_mb": round(index_size, 2),
    }

    print(json.dumps(result))


def run_configuration(dataset: str, mode: str, limit: int, queries: int, k: int, m: int, ef_c: int, ef_s: int, threads: int = 32):
    python_bin = sys.executable
    cmd = [
        python_bin,
        __file__,
        "--worker",
        "--dataset", dataset,
        "--mode", mode,
        "--limit", str(limit),
        "--queries", str(queries),
        "--k", str(k),
        "--m", str(m),
        "--ef-construction", str(ef_c),
        "--ef-search", str(ef_s),
        "--threads", str(threads),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {mode} on {dataset}:", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)

    for line in res.stdout.strip().split("\n"):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)

    raise RuntimeError(f"Failed to parse JSON result from worker output:\n{res.stdout}")


def main():
    parser = argparse.ArgumentParser(description="3-Way Intra-Engine Ablation: SQ8 vs Vanilla TQ8 vs BenoQuant BQ8")
    parser.add_argument("--worker", action="store_true", help="Internal worker flag")
    parser.add_argument("--dataset", default="gist-960-euclidean", help="Dataset name")
    parser.add_argument("--datasets", default=None, help="Comma-separated list of datasets")
    parser.add_argument("--mode", choices=["sq8", "tq8", "bq8"], default="bq8", help="Ablation mode")
    parser.add_argument("--limit", type=int, default=10000, help="Train set limit (0 = full)")
    parser.add_argument("--queries", type=int, default=1000, help="Query set limit")
    parser.add_argument("--k", type=int, default=10, help="Top-K")
    parser.add_argument("--m", type=int, default=16, help="HNSW complexity M")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction")
    parser.add_argument("--ef-search", type=int, default=200, help="HNSW ef_search")
    parser.add_argument("--threads", type=int, default=32, help="Number of concurrent search threads")
    parser.add_argument("--out", default=None, help="Output markdown or json file")

    args = parser.parse_args()

    if args.worker:
        worker_run(
            args.dataset,
            args.mode,
            args.limit,
            args.queries,
            args.k,
            args.m,
            args.ef_construction,
            args.ef_search,
            args.threads,
        )
        return

    datasets = args.datasets.split(",") if args.datasets else [args.dataset]
    modes = ["sq8", "tq8", "bq8"]
    all_results = []

    print("==========================================================================================")
    print(f"Starting 3-Way Intra-Engine Ablation (SQ8 vs. Vanilla TQ8 vs. BenoQuant BQ8)")
    print(f"Datasets: {datasets} | Limit: {args.limit} | Queries: {args.queries} | M={args.m} | ef_c={args.ef_construction} | ef_s={args.ef_search} | Threads={args.threads}")
    print("==========================================================================================\n")

    for ds in datasets:
        print(f"\n--- Benchmarking Dataset: {ds} (Limit={args.limit}) ---")
        for m in modes:
            print(f"  Executing Mode: {m.upper()} ...", end="", flush=True)
            t_start = time.time()
            record = run_configuration(
                ds,
                m,
                args.limit,
                args.queries,
                args.k,
                args.m,
                args.ef_construction,
                args.ef_search,
                args.threads,
            )
            elapsed = time.time() - t_start
            all_results.append(record)
            print(f" Done ({elapsed:.1f}s) -> Recall@{args.k}: {record['recall_at_k']:.4f} | Pure QPS: {record['pure_qps']} | Build: {record['build_time_s']}s")

    # Generate Markdown Summary Table
    md_rows = [
        "| Dataset | Quantizer Paradigm | Recall@10 | Pure Index QPS | End-to-End QPS | p50 Latency (ms) | p99 Latency (ms) | Build Time (s) | Index Size (MB) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in all_results:
        mode_label = {
            "sq8": "Native SQ8 (hnsw_sq8)",
            "tq8": "TurboQuant TQ8 (hnsw_tq8)",
            "bq8": "TurboQuant TQ8 (hnsw_tq8)",
        }.get(r["mode"], r["mode"])
        md_rows.append(
            f"| `{r['dataset']}` | {mode_label} | **{r['recall_at_k']:.4f}** | {r['pure_qps']:,.1f} | {r['qps']:,.1f} | {r['pure_p50_ms']:.2f} | {r['pure_p99_ms']:.2f} | {r['build_time_s']:.1f}s | {r['index_size_mb']:.1f} MB |"
        )

    summary_md = "\n".join(md_rows)
    print("\n\n==========================================================================================")
    print("FINAL EMPIRICAL BENCHMARK RESULTS")
    print("==========================================================================================")
    print(summary_md)

    if args.out:
        with open(args.out, "w") as f:
            if args.out.endswith(".json"):
                json.dump(all_results, f, indent=2)
            else:
                f.write(summary_md)
        print(f"\nSaved benchmark results to {args.out}")


if __name__ == "__main__":
    main()

