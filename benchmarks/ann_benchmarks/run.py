#!/usr/bin/env python3
"""ANN-Benchmarks harness for BenoStreamDB.

Downloads a standard ANN-Benchmarks dataset (HDF5), loads it into a
BenoStreamDB table with a vector index, and reports the standard metrics:
recall@k, QPS, build time, index size, and p50/p99 latency.

Two things are exposed so the comparison is apples-to-apples:

1. **Build/query parameters** — set to match the competitor's config exactly.
   A tuned competitor vs. an untuned one is not a comparison.

       complexity      -> M                (max neighbours per node)
       quality         -> ef_construction  (build beam width)
       ef_search       -> efSearch         (query beam width)
       metric          -> metric           (l2 / cosine / inner_product)

2. **Resource envelope** — `--cores` and `--ram-gb` constrain the engine to the
   same cores and RAM the competitor is given, via the engine's own knobs:

       RAYON_NUM_THREADS              = cores
       BSDB_MAX_CONCURRENCY           = cores
       BSDB_INDEX_BUILD_CONCURRENCY   = cores
       BSDB_MAX_INGEST_RAM_GB         = 0.8 * ram_gb
       BSDB_INGEST_MEMORY_BUDGET_GB   = 0.8 * ram_gb
       BSDB_DATAFUSION_MEMORY_GB      = 0.5 * ram_gb

   Run the competitor with the same `--cores` / `--ram-gb` (e.g. Docker
   `--cpus` / `--memory`, or the process's cgroup limit).

Usage:
    python benchmarks/ann_benchmarks/run.py --dataset sift-128-euclidean --cores 4 --ram-gb 8
    python benchmarks/ann_benchmarks/run.py --dataset glove-100-angular --m 32 --ef-construction 200 --ef-search 200
    python benchmarks/ann_benchmarks/run.py --dataset fashion-mnist-784-euclidean --limit 20000

Datasets (from http://ann-benchmarks.com):
    sift-128-euclidean, gist-960-euclidean, glove-100-angular,
    glove-200-angular, fashion-mnist-784-euclidean, mnist-784-euclidean,
    nytimes-256-angular, lastfm-64-dot
"""
import argparse
import json
import os
import platform
import shutil
import time
import urllib.request

import h5py
import numpy as np
import pandas as pd

ANN_BASE = "http://ann-benchmarks.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The distance metric each ANN-Benchmarks dataset is defined with.
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


def configure_env(cores: int, ram_gb: float) -> None:
    """Constrain the engine to the competitor's cores and RAM.

    Must run before `import benostreamdb` so the values are in place when the
    engine reads them.
    """
    os.environ.setdefault("RAYON_NUM_THREADS", str(cores))
    os.environ.setdefault("BSDB_MAX_CONCURRENCY", str(cores))
    os.environ.setdefault("BSDB_INDEX_BUILD_CONCURRENCY", str(cores))
    os.environ.setdefault("BSDB_MAX_INGEST_RAM_GB", f"{ram_gb * 0.8:.2f}")
    os.environ.setdefault("BSDB_INGEST_MEMORY_BUDGET_GB", f"{ram_gb * 0.8:.2f}")
    os.environ.setdefault("BSDB_DATAFUSION_MEMORY_GB", f"{ram_gb * 0.5:.2f}")


def download(dataset: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{dataset}.hdf5")
    if os.path.exists(path):
        return path
    url = f"{ANN_BASE}/{dataset}.hdf5"
    print(f"downloading {url} ...", flush=True)
    # ann-benchmarks.com rejects the default urllib User-Agent with a 403.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
        shutil.copyfileobj(resp, out)
    return path


def load(path: str):
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int64)
    return train, test, neighbors


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _total_ram_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page / 1024**3, 1)
    except Exception:
        return None


def _env() -> dict:
    """Container-visible hardware profile, matching the competitor runner."""
    return {
        "cpu_model": _cpu_model(),
        "cores": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "containerized": os.path.exists("/.dockerenv"),
        "gpus": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=1000, help="number of test queries")
    ap.add_argument("--limit", type=int, default=0, help="cap the train set size (0 = all)")
    ap.add_argument("--index", default="hnsw_tq8", help="index type (hnsw, hnsw_tq8, hnsw_tq4, hnsw_pq)")
    ap.add_argument("--m", type=int, default=16, help="HNSW M (complexity)")
    ap.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction (quality)")
    ap.add_argument("--ef-search", type=int, default=200, help="HNSW ef_search (query beam width)")
    ap.add_argument("--metric", default=None, help="distance metric (default: from the dataset)")
    ap.add_argument("--cores", type=int, default=0, help="constrain the engine to N cores (0 = leave default)")
    ap.add_argument("--ram-gb", type=float, default=0.0, help="constrain the engine to N GB RAM (0 = leave default)")
    ap.add_argument("--out", default=None, help="output path (.json emits JSON + a .md sibling)")
    ap.add_argument("--device", default="cpu", help="device tag (cpu/gpu)")
    args = ap.parse_args()

    if args.cores > 0 or args.ram_gb > 0:
        cores = args.cores if args.cores > 0 else (os.cpu_count() or 1)
        ram_gb = args.ram_gb if args.ram_gb > 0 else 8.0
        configure_env(cores, ram_gb)
        print(f"resource envelope: cores={cores} ram_gb={ram_gb}", flush=True)

    # Import after the env is configured.
    import benostreamdb as bsdb

    metric = args.metric or DATASET_METRIC.get(args.dataset, "l2")

    path = download(args.dataset)
    train, test, neighbors = load(path)
    if args.limit and args.limit < len(train):
        train = train[: args.limit]
        neighbors = None  # ground truth is only valid for the full set
    n, dim = train.shape
    print(
        f"dataset={args.dataset} train={train.shape} test={test.shape} dim={dim} "
        f"metric={metric} index={args.index} M={args.m} ef_construction={args.ef_construction} "
        f"ef_search={args.ef_search}",
        flush=True,
    )

    db_path = os.path.join(CACHE_DIR, f"db_{args.dataset}")
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    t0 = time.time()
    table = bsdb.Table(db_path)
    table.add_index(
        "embedding",
        {
            "type": args.index,
            "complexity": args.m,
            "quality": args.ef_construction,
            "metric": metric,
        },
    )
    # Write in chunks. The engine's ingest RAM back-pressure counts *process*
    # RSS, so holding the whole 1M-row frame in the caller would push RSS over
    # the limit and deadlock the write (the caller cannot free the frame until
    # the write returns). Chunking keeps the caller-side footprint bounded.
    chunk = 100_000
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
    # The vector index is built as a background task after the commit. Wait for
    # it (and count that in build time) or every query silently falls back to a
    # brute-force scan and the recall/QPS numbers are meaningless.
    table.wait_for_background_tasks()
    build_s = time.time() - t0
    index_bytes = dir_size_bytes(db_path)

    # Secondary indexes are packed into Puffin compound bundles (`.puffin`),
    # so count those; the loose-sidecar names are kept for older tables.
    n_index_files = sum(
        1
        for _root, _dirs, files in os.walk(db_path)
        for fn in files
        if any(
            s in fn
            for s in (
                ".puffin",
                ".hnsw",
                ".centroid",
                ".tq8",
                ".cluster",
                ".mapping",
                ".pq",
            )
        )
    )
    if n_index_files == 0:
        raise SystemExit(
            "no vector index files were produced — the engine fell back to a "
            "brute-force scan; aborting so the numbers are not published."
        )
    print(
        f"index files={n_index_files} index_mb={index_bytes / 1e6:.1f}",
        flush=True,
    )

    # If we subsetted the train set, recompute exact ground truth by brute force.
    if neighbors is None:
        q = min(args.queries, len(test))
        truth = []
        for i in range(q):
            d = np.linalg.norm(train - test[i], axis=1)
            truth.append(np.argsort(d)[: args.k])
        neighbors = np.array(truth)

    q = min(args.queries, len(test))
    k = args.k

    # Recall: full search + row fetch (Arrow, to avoid pandas overhead).
    latencies = []
    hits = 0.0
    for i in range(q):
        query_list = test[i].tolist()
        vf = {
            "column": "embedding",
            "query": query_list,
            "k": k,
            "ef_search": args.ef_search,
        }
        t1 = time.perf_counter()
        # Project only `id`: the harness uses nothing else, and fetching the
        # embedding column would read the whole (multi-hundred-MB) row group.
        res = table.to_arrow(vector_filter=vf, columns=["id"])
        latencies.append(time.perf_counter() - t1)
        got = set(res.column("id").to_pylist()) if "id" in res.column_names else set()
        truth = set(neighbors[i][:k].tolist())
        hits += len(got & truth) / k

    recall = hits / q
    total_s = sum(latencies)
    qps = q / total_s if total_s > 0 else 0.0
    p50 = float(np.percentile(latencies, 50) * 1000)
    p99 = float(np.percentile(latencies, 99) * 1000)

    # Pure index latency (no Parquet I/O) — the comparable ANN-Benchmarks number.
    pure_latencies = []
    for i in range(q):
        query_list = test[i].tolist()
        t1 = time.perf_counter()
        _ = table._inner.vector_search_scored("embedding", query_list, k, metric, args.ef_search)
        pure_latencies.append(time.perf_counter() - t1)
    pure_qps = q / sum(pure_latencies) if sum(pure_latencies) > 0 else 0.0
    pure_p50 = float(np.percentile(pure_latencies, 50) * 1000)
    pure_p99 = float(np.percentile(pure_latencies, 99) * 1000)

    print(
        f"recall@{k}={recall:.4f} qps={qps:.1f} build_s={build_s:.1f} "
        f"index_mb={index_bytes / 1e6:.1f} p50_ms={p50:.2f} p99_ms={p99:.2f} "
        f"| pure_index_qps={pure_qps:.1f} pure_p50_ms={pure_p50:.2f} pure_p99_ms={pure_p99:.2f}",
        flush=True,
    )

    record = {
        "engine": "benostreamdb",
        "dataset": args.dataset,
        "workload": "vector_ann",
        "device": args.device,
        "n": int(n),
        "dim": int(dim),
        "index": {
            "type": args.index,
            "m": args.m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "metric": metric,
        },
        "queries": q,
        "k": k,
        "build_s": round(build_s, 3),
        "index_mb": round(index_bytes / 1e6, 2),
        "recall_at_k": round(recall, 4),
        "qps": round(qps, 1),
        "p50_ms": round(p50, 3),
        "p99_ms": round(p99, 3),
        "pure_index_qps": round(pure_qps, 1),
        "pure_p50_ms": round(pure_p50, 3),
        "pure_p99_ms": round(pure_p99, 3),
        "env": _env(),
    }

    if args.out:
        # The competitor runner writes JSON to `--out` and markdown alongside it;
        # match that so the shared rollup can parse every engine uniformly.
        if args.out.endswith(".json"):
            json_path, md_path = args.out, args.out[:-5] + ".md"
        elif args.out.endswith(".md"):
            json_path, md_path = args.out[:-3] + ".json", args.out
        else:
            json_path, md_path = args.out + ".json", args.out + ".md"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# ANN-Benchmarks: {args.dataset}\n\n")
            f.write("| Metric | Value |\n|---|---|\n")
            f.write(f"| Dataset | {args.dataset} |\n")
            f.write(f"| Train | {n} x {dim} |\n")
            f.write(f"| Queries | {q} |\n")
            f.write(f"| k | {k} |\n")
            f.write(f"| Metric | {metric} |\n")
            f.write(f"| Index | {args.index} |\n")
            f.write(f"| M (complexity) | {args.m} |\n")
            f.write(f"| ef_construction (quality) | {args.ef_construction} |\n")
            f.write(f"| ef_search | {args.ef_search} |\n")
            if args.cores > 0 or args.ram_gb > 0:
                f.write(f"| Cores | {args.cores or 'default'} |\n")
                f.write(f"| RAM (GB) | {args.ram_gb or 'default'} |\n")
            f.write(f"| recall@{k} | {recall:.4f} |\n")
            f.write(f"| QPS | {qps:.1f} |\n")
            f.write(f"| Build time | {build_s:.1f}s |\n")
            f.write(f"| Index size | {index_bytes / 1e6:.1f} MB |\n")
            f.write(f"| p50 latency | {p50:.2f} ms |\n")
            f.write(f"| p99 latency | {p99:.2f} ms |\n")
            f.write(f"| Pure Index QPS | {pure_qps:.1f} |\n")
            f.write(f"| Pure Index p50 latency | {pure_p50:.2f} ms |\n")
            f.write(f"| Pure Index p99 latency | {pure_p99:.2f} ms |\n")
        print(f"wrote {json_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
