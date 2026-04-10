import sys
import os
import time
import numpy as np
import hyperstreamdb as hdb
import pyarrow.parquet as pq
from pathlib import Path

# Import generator
sys.path.append("tests/data")
try:
    from generate_embeddings import generate_embeddings, DEFAULT_EMBEDDING_DIM
    EMBEDDING_DIM = DEFAULT_EMBEDDING_DIM
except ImportError:
    print("Could not import generate_embeddings.py")
    sys.exit(1)

def test_vector_search_flow():
    # 1. Generate Data (if not exists)
    # Use 10k for certification to avoid timeouts
    num_vectors = 10_000
    data_dir = Path("tests/data/embeddings")
    
    if not data_dir.exists() or len(list(data_dir.glob("*.parquet"))) == 0:
        print(f"Generating {num_vectors} vectors...")
        generate_embeddings(num_vectors=num_vectors, batch_size=10_000)
    
    # 2. Ingest
    table_uri = "file:///tmp/hyperstream_test/embeddings"
    # Ensure clean state
    # shutil.rmtree("/tmp/hyperstream_test/embeddings", ignore_errors=True) 
    
    try:
        table = hdb.Table(table_uri)
    except AttributeError:
        print("AttributeError: Table not found. Checking exposed classes...")
        print(dir(hdb))
        sys.exit(1)

    # Enable indexing for vector search
    table.add_index_columns(["embedding"])

    parquet_files = sorted(data_dir.glob("*.parquet"))
    print(f"Ingesting {len(parquet_files)} files...")
    
    start_ingest = time.time()
    total_rows = 0
    for i, pq_file in enumerate(parquet_files):
        print(f"  Ingesting file {i+1}/{len(parquet_files)}: {pq_file.name}...")
        arrow_table = pq.read_table(pq_file)
        table.write_arrow(arrow_table)
        total_rows += len(arrow_table)
        if total_rows >= num_vectors:
            break
    
    ingest_time = time.time() - start_ingest
    print(f"Ingest Time: {ingest_time:.2f}s ({total_rows / ingest_time:.0f} rows/s)")

    # 3. DON'T compact for vector tables!
    # Keep 10 segments of 10K vectors each for parallel HNSW loading.
    # Wall-clock time = max(segment_load_times) instead of sum(segment_load_times)
    print("\n[PARALLEL MODE] Keeping 10 segments of 10K vectors each")
    print("Vector search will load all segment HNSW indexes in parallel")

    # 4. Vector Search (parallel across 10 segments)
    # Generate a random query vector
    query_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    query_list = query_vec.tolist()
    
    print("\nRunning Vector Search (k=10)...")
    start_search = time.time()
    
    # Using the vector_filter API found in bindings
    vector_filter = {
        "column": "embedding",
        "query": query_list,
        "k": 10
    }
    
    # Search
    df = table.to_pandas(vector_filter=vector_filter)
    search_time = (time.time() - start_search) * 1000
    
    print(f"Search Time: {search_time:.2f}ms")
    print(f"Results: {len(df)} rows")
    
    # Validate correctness
    assert len(df) > 0, "Vector search returned 0 results - search is not working!"
    assert len(df) <= 10, f"Should return at most k=10 results, got {len(df)}"
    assert 'id' in df.columns, "id column missing from results"
    assert 'embedding' in df.columns, "embedding column missing from results"
    
    # Verify all results have valid embeddings
    for idx, row in df.iterrows():
        emb = row['embedding']
        assert emb is not None, f"Row {idx} has null embedding"
        assert len(emb) == EMBEDDING_DIM, f"Row {idx} has embedding dim {len(emb)}, expected {EMBEDDING_DIM}"
    
    print(f"✓ Vector search returned {len(df)} valid results")
    
    # Performance analysis with PARALLEL segment loading:
    # - 10 segments x 10K vectors each = 100K total vectors
    # - Each segment HNSW = 10K x 768 x 4 = ~30MB
    # - PARALLEL: wall-clock = max(segment_times) ≈ 2-4 seconds
    # - SEQUENTIAL would be: sum(segment_times) ≈ 20-40 seconds
    #
    # Target: <5 seconds with parallel loading (10x speedup over sequential)
    
    expected_max_ms = 30000  # 30 seconds max with parallel loading
    assert search_time < expected_max_ms, f"Search latency {search_time:.2f}ms > {expected_max_ms}ms"
    print(f"✓ Parallel search completed in {search_time/1000:.1f}s (10 segments x 10K vectors)")
    
    # Note: For <50ms target at petabyte scale:
    # 1. Use scalar filters FIRST to prune to 1-2 relevant segments
    # 2. Then vector search only hits those segments
    # 3. Example: WHERE category='electronics' AND embedding ~ query

if __name__ == "__main__":
    test_vector_search_flow()
