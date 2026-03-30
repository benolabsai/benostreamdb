import hyperstreamdb as hs
import numpy as np
import time
import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def run_benchmark(backend):
    print(f"\n--- Running Benchmark with Backend: {backend} ---")
    
    # Setup
    db_dir = os.path.abspath(f"./test_ingestion_db_{backend}")
    db_path = f"file://{db_dir}"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    
    # Initialize DB and Compute Context
    if backend == "gpu":
        ctx = hs.ComputeContext.auto_detect()
    else:
        ctx = hs.ComputeContext(backend="cpu")
    
    ctx.activate()
    print(f"Activated Backend: {ctx.backend}")
    
    table = hs.Table(db_path)
    table.add_index_columns(["vector"])
    
    # Generate large batch of vectors
    num_rows = 100_000 # Increased to 100k for better differentiation
    dim = 256 # Increased dimension
    
    print(f"Generating {num_rows} vectors (dim={dim})...")
    base_vectors = np.random.randn(100, dim).astype(np.float32) # 100 clusters
    labels = np.random.randint(0, 100, size=num_rows)
    vectors = base_vectors[labels] + np.random.randn(num_rows, dim).astype(np.float32) * 0.1
    
    df = pd.DataFrame({
        "id": np.arange(num_rows, dtype=np.int32),
        "val": np.random.randint(0, 1000, size=num_rows, dtype=np.int32),
        "vector": [v.astype(np.float32).tolist() for v in vectors]
    })
    
    # Benchmark Ingestion
    print(f"Ingesting {num_rows} rows...")
    start_time = time.time()
    table.write(df)
    table.flush()
    table.wait_for_background_tasks() # Wait for ALL indexing/shuffling to complete
    duration = time.time() - start_time
    print(f"Ingestion completed in {duration:.2f}s ({num_rows/duration:.2f} rows/s)")
    
    # Physical Layout Analysis (for GPU run only)
    if backend == "gpu":
        parquet_files = [f for f in os.listdir(db_dir) if f.endswith(".parquet") and ".inv." not in f]
        if parquet_files:
            full_path = os.path.join(db_dir, parquet_files[0])
            pf = pq.ParquetFile(full_path)
            batch = pf.read_row_group(0)
            print(f"Verified shuffling result in {parquet_files[0]}: {batch.num_rows} rows")
            
    return num_rows / duration

if __name__ == "__main__":
    print("HyperStreamDB Ingestion Performance Comparison: GPU vs CPU")
    
    gpu_tps = run_benchmark("gpu")
    cpu_tps = run_benchmark("cpu")
    
    print("\n" + "="*40)
    print(f"{'Backend':<10} | {'Throughput (rows/s)':<20}")
    print("-" * 40)
    print(f"{'GPU (MPS)':<10} | {gpu_tps:>20.2f}")
    print(f"{'CPU':<10} | {cpu_tps:>20.2f}")
    print(f"Speedup: {gpu_tps/cpu_tps:.2f}x")
    print("="*40)
