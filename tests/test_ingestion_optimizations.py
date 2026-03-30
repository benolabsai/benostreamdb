import hyperstreamdb as hs
import numpy as np
import time
import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def test_ingestion_optimizations():
    print("--- HyperStreamDB Ingestion Optimization Benchmark ---")
    
    # Setup
    db_dir = os.path.abspath("./test_ingestion_db")
    db_path = f"file://{db_dir}"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    
    # 1. Initialize DB and Compute Context
    ctx = hs.ComputeContext.auto_detect()
    ctx.activate()
    print(f"GPU Acceleration: {ctx.backend}")
    
    table = hs.Table(db_path)
    table.add_index_columns(["vector"])
    
    # 2. Generate large batch of vectors (Unsorted)
    num_rows = 50_000
    dim = 128
    
    print(f"Generating {num_rows} vectors...")
    base_vectors = np.random.randn(10, dim).astype(np.float32)
    labels = np.random.randint(0, 10, size=num_rows)
    vectors = base_vectors[labels] + np.random.randn(num_rows, dim).astype(np.float32) * 0.1
    
    df = pd.DataFrame({
        "id": np.arange(num_rows, dtype=np.int32),
        "val": np.random.randint(0, 1000, size=num_rows, dtype=np.int32),
        # Explicitly use float32 to avoid List(Float64)
        "vector": [v.astype(np.float32).tolist() for v in vectors]
    })
    
    # 3. Benchmark Ingestion
    print(f"Ingesting {num_rows} rows...")
    start_time = time.time()
    table.write(df)
    table.flush()
    duration = time.time() - start_time
    print(f"Ingestion completed in {duration:.2f}s ({num_rows/duration:.2f} rows/s)")
    
    # 4. Verify Layout (Vector Shuffling)
    segments_dir = db_dir
    parquet_files = [f for f in os.listdir(segments_dir) if f.endswith(".parquet") and ".inv." not in f]
    
    if parquet_files:
        full_path = os.path.join(segments_dir, parquet_files[0])
        print(f"Analyzing data layout in {full_path}...")
        pf = pq.ParquetFile(full_path)
        batch = pf.read_row_group(0)
        
        first_vectors = batch.column("vector").to_numpy()
        v_matrix = np.array([np.array(v) for v in first_vectors[:1000]])
        mean_v = np.mean(v_matrix, axis=0)
        avg_dist = np.mean(np.linalg.norm(v_matrix - mean_v, axis=1))
        
        all_v = np.array([np.array(v) for v in first_vectors])
        random_indices = np.random.choice(len(all_v), 1000, replace=False)
        random_v = all_v[random_indices]
        random_avg_dist = np.mean(np.linalg.norm(random_v - np.mean(random_v, axis=0), axis=1))
        
        print(f"Local block variance (avg dist): {avg_dist:.4f}")
        print(f"Global variance (avg dist): {random_avg_dist:.4f}")
        
    # Cleanup
    # shutil.rmtree(db_dir)

if __name__ == "__main__":
    test_ingestion_optimizations()
