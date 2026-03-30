import hyperstreamdb as hs
import numpy as np
import time
import os
import shutil
import pandas as pd

def run_benchmark(backend, num_rows, num_centroids, dim):
    print(f"\n--- Testing Backend: {backend} | Rows: {num_rows} | Clusters: {num_centroids} ---")
    
    db_dir = os.path.abspath(f"./test_smart_db_{backend}_{num_rows}")
    db_path = f"file://{db_dir}"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    
    # Initialize Compute Context
    if backend == "gpu":
        ctx = hs.ComputeContext.auto_detect()
    else:
        ctx = hs.ComputeContext(backend="cpu")
    ctx.activate()
    
    table = hs.Table(db_path)
    table.add_index_columns(["vector"])
    
    # Generate vectors
    base_vectors = np.random.randn(num_centroids, dim).astype(np.float32)
    labels = np.random.randint(0, num_centroids, size=num_rows)
    vectors = base_vectors[labels] + np.random.randn(num_rows, dim).astype(np.float32) * 0.1
    
    df = pd.DataFrame({
        "id": np.arange(num_rows, dtype=np.int32),
        "vector": [v.tolist() for v in vectors]
    })
    
    # Benchmark
    start_time = time.time()
    table.write(df)
    table.flush()
    table.wait_for_background_tasks()
    duration = time.time() - start_time
    print(f"Ingestion: {duration:.2f}s ({num_rows/duration:.2f} rows/s)")
    
    return num_rows / duration

if __name__ == "__main__":
    print("Testing Smart Backend Selection Strategy")
    dim = 128
    
    # CASE 1: Small dataset (5M ops) -> Should use CPU even if GPU is active
    # Target: 50,000 vectors * 100 clusters = 5,000,000 ops (< 10M threshold)
    print("\n[SMART TEST: Small Dataset]")
    gpu_small = run_benchmark("gpu", 50000, 100, dim)
    cpu_small = run_benchmark("cpu", 50000, 100, dim)
    
    # CASE 2: Large dataset (40M ops) -> Should use GPU if active
    # Target: 200,000 vectors * 200 clusters = 40,000,000 ops (> 10M threshold)
    print("\n[SMART TEST: Large Dataset]")
    gpu_large = run_benchmark("gpu", 200000, 200, dim)
    cpu_large = run_benchmark("cpu", 200000, 200, dim)
    
    print("\n" + "="*50)
    print(f"{'Scale':<10} | {'GPU Context (rows/s)':<25} | {'CPU Context (rows/s)':<25}")
    print("-" * 65)
    print(f"{'Small':<10} | {gpu_small:>20.2f} | {cpu_small:>20.2f}")
    print(f"{'Large':<10} | {gpu_large:>20.2f} | {cpu_large:>20.2f}")
    print("="*50)
