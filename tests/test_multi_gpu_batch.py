import time
import concurrent.futures
import numpy as np
import pyarrow as pa
try:
    import pytest
except ImportError:
    pytest = None

import benostreamdb as bsdb

def has_dual_cuda():
    try:
        d0 = bsdb.Device("cuda:0")
        d1 = bsdb.Device("cuda:1")
        return True
    except Exception:
        return False

def test_multi_gpu_concurrent_cdist():
    if not has_dual_cuda():
        if pytest is not None:
            pytest.skip("Multi-GPU cdist test requires at least 2 CUDA devices")
        else:
            print("Skipping test_multi_gpu_concurrent_cdist (requires at least 2 CUDA devices)")
            return

    print("=== Testing Concurrent Multi-GPU cdist (cuda:0 & cuda:1) ===")
    dim = 128
    n_queries = 2000
    n_vectors = 10000

    np.random.seed(42)
    q1 = np.random.randn(n_queries, dim).astype(np.float32)
    v1 = np.random.randn(n_vectors, dim).astype(np.float32)

    q2 = np.random.randn(n_queries, dim).astype(np.float32)
    v2 = np.random.randn(n_vectors, dim).astype(np.float32)

    def run_worker(dev, q, v):
        t0 = time.perf_counter()
        res = bsdb.cdist(q, v, metric="l2", device=dev)
        dt = (time.perf_counter() - t0) * 1000.0
        name = dev if isinstance(dev, str) else getattr(dev, "device", str(dev))
        return name, res.shape, dt

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(run_worker, bsdb.Device("cuda:0"), q1, v1)
        f1 = executor.submit(run_worker, "cuda:1", q2, v2)
        r0 = f0.result()
        r1 = f1.result()

    print(f"[{r0[0]}] shape={r0[1]} time={r0[2]:.2f} ms")
    print(f"[{r1[0]}] shape={r1[1]} time={r1[2]:.2f} ms")
    total_evals = n_queries * n_vectors * 2
    print(f"Evaluated {total_evals:,} pairwise distances across 2 GPUs in parallel!")

def test_multi_gpu_device_pool_dispatch():
    devices = "cuda:0,cuda:1" if has_dual_cuda() else "cpu,cpu"
    print(f"\n=== Testing GpuDevicePool Multi-GPU Concurrent Ingestion & Search ({devices}) ===")
    pool = bsdb.set_gpu_device_pool(devices)
    expected = devices.split(",")
    print(f"Pool installed with devices: {pool}")
    assert pool == expected
    assert bsdb.gpu_device_pool() == expected

    # Ingest 20,000 vectors (128-d)
    dim = 128
    n_rows = 20_000
    np.random.seed(99)
    vecs = np.random.randn(n_rows, dim).astype(np.float32)

    schema = pa.schema([
        ("id", pa.int64()),
        ("vector", pa.list_(pa.float32(), dim)),
    ])
    table = bsdb.Table.create("memory://pool_test", schema)

    batch_size = 5_000
    t0 = time.perf_counter()
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch = pa.record_batch([
            pa.array(range(start, end), type=pa.int64()),
            pa.FixedSizeListArray.from_arrays(vecs[start:end].ravel(), dim),
        ], schema=schema)
        table.write(pa.Table.from_batches([batch]))
    t_ingest = (time.perf_counter() - t0) * 1000.0
    print(f"Ingested {n_rows:,} vectors into Table in {t_ingest:.2f} ms ({n_rows / (t_ingest / 1000.0):,.0f} rows/s)")

    # Build index using GPU
    t0 = time.perf_counter()
    table.add_index("vector", "hnsw")
    t_index = (time.perf_counter() - t0) * 1000.0
    print(f"Built & committed index in {t_index:.2f} ms")

    # Concurrent search across multiple threads pulling from GPU pool
    n_search_threads = 4
    queries = [np.random.randn(dim).astype(np.float32) for _ in range(n_search_threads)]

    def search_worker(tid, q):
        t_start = time.perf_counter()
        res = table.vector_search("vector", q, k=10)
        dt = (time.perf_counter() - t_start) * 1000.0
        return tid, len(res), dt

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_search_threads) as executor:
        futures = [executor.submit(search_worker, i, q) for i, q in enumerate(queries)]
        results = [f.result() for f in futures]

    for r in results:
        print(f"Thread {r[0]} search returned {r[1]} results in {r[2]:.2f} ms")

if __name__ == "__main__":
    test_multi_gpu_concurrent_cdist()
    test_multi_gpu_device_pool_dispatch()
    print("\nAll multi-GPU concurrent tests passed successfully!")

