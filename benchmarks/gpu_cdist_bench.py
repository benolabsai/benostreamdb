#!/usr/bin/env python3
"""Batched GPU distance (``bsdb.cdist``) benchmark.

The ANN harness exercises HNSW/IVF index build and search, which are CPU-bound
at the dataset sizes it uses. The 2D-tiled CUDA kernels are reached through
``bsdb.cdist`` (and k-means), so this benchmark measures them directly:

* CPU vs CUDA for L2 and cosine at several (queries x vectors x dim) shapes.
* One GPU vs two GPUs for a compute-heavy shape, driving each device from its
  own Python thread (the GIL is released around the kernel).

Results are written to ``benchmarks/results/gpu_cdist.json`` and folded into
``benchmarks/results/BENCHMARK_REPORT.md`` by ``generate_summary.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import threading
import time

import numpy as np

import benostreamdb as bsdb


def bench(fn, warmup: int = 2, reps: int = 5) -> float:
    """Best-of-``reps`` wall time in milliseconds, after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


def throughput(q: int, v: int, ms: float) -> float:
    return (q * v) / (ms / 1000.0) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "gpu_cdist.json"
        ),
    )
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    np.random.seed(7)
    cpu = bsdb.Device("cpu")
    cuda = bsdb.Device("cuda:0") if bsdb.Device.is_available("cuda") else None
    has_dual = bool(
        cuda
        and bsdb.Device.is_available("cuda")
        and _device_count() >= 2
    )

    cases = []
    shapes = [
        # queries, vectors, dim, metric
        (512, 50_000, 128, "l2"),
        (2048, 50_000, 128, "l2"),
        (1024, 200_000, 128, "l2"),
        (512, 50_000, 100, "cosine"),
    ]
    for q, v, d, metric in shapes:
        queries = np.random.randn(q, d).astype(np.float32)
        vectors = np.random.randn(v, d).astype(np.float32)
        row = {"queries": q, "vectors": v, "dim": d, "metric": metric}
        cpu_ms = bench(
            lambda: bsdb.cdist(queries, vectors, metric=metric, device=cpu),
            reps=args.reps,
        )
        row["cpu_ms"] = round(cpu_ms, 2)
        row["cpu_mdist_s"] = round(throughput(q, v, cpu_ms), 1)
        if cuda is not None:
            gpu_ms = bench(
                lambda: bsdb.cdist(queries, vectors, metric=metric, device=cuda),
                reps=args.reps,
            )
            row["cuda_ms"] = round(gpu_ms, 2)
            row["cuda_mdist_s"] = round(throughput(q, v, gpu_ms), 1)
            row["speedup"] = round(cpu_ms / gpu_ms, 2)
        cases.append(row)

    dual = None
    if has_dual:
        q, v, d = 8192, 20_000, 2048
        queries = np.random.randn(q, d).astype(np.float32)
        vectors = np.random.randn(v, d).astype(np.float32)
        gpu1 = bsdb.Device("cuda:1")
        single = bench(
            lambda: bsdb.cdist(queries, vectors, metric="l2", device=cuda),
            reps=args.reps,
        )
        half = q // 2

        def split() -> None:
            t0 = threading.Thread(
                target=lambda: bsdb.cdist(queries[:half], vectors, metric="l2", device=cuda)
            )
            t1 = threading.Thread(
                target=lambda: bsdb.cdist(queries[half:], vectors, metric="l2", device=gpu1)
            )
            t0.start()
            t1.start()
            t0.join()
            t1.join()

        both = bench(split, reps=args.reps)
        dual = {
            "queries": q,
            "vectors": v,
            "dim": d,
            "single_ms": round(single, 2),
            "dual_ms": round(both, 2),
            "speedup": round(single / both, 2),
        }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": _gpu_names(),
        "cases": cases,
        "dual_gpu": dual,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}")
    for row in cases:
        print(
            f"{row['queries']}x{row['vectors']}x{row['dim']} {row['metric']}: "
            f"cpu={row['cpu_ms']}ms ({row['cpu_mdist_s']} Mdist/s), "
            f"cuda={row.get('cuda_ms')}ms ({row.get('cuda_mdist_s')} Mdist/s), "
            f"speedup={row.get('speedup')}x"
        )
    if dual:
        print(
            f"dual {dual['queries']}x{dual['vectors']}x{dual['dim']}: "
            f"single={dual['single_ms']}ms dual={dual['dual_ms']}ms "
            f"({dual['speedup']}x)"
        )


def _device_count() -> int:
    try:
        out = os.popen("nvidia-smi -L").read()
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except OSError:
        return 0


def _gpu_names() -> list[str]:
    try:
        out = os.popen(
            "nvidia-smi --query-gpu=name --format=csv,noheader"
        ).read()
        return [line.strip() for line in out.splitlines() if line.strip()]
    except OSError:
        return []


if __name__ == "__main__":
    main()
