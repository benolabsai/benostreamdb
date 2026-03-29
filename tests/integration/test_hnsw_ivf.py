"""
Test HNSW-IVF implementation with a simple dataset
"""
import hyperstreamdb as hdb
import pandas as pd
import numpy as np
import time

print("="*60)
print("Testing HNSW-IVF Implementation")
print("="*60)


def test_hnsw_ivf_integration():
    print("="*60)
    print("Testing HNSW-IVF Implementation")
    print("="*60)

    # Create test data
    np.random.seed(42)
    n_vectors = 1000
    dim = 128
    
    print(f"\nGenerating {n_vectors} random {dim}-dim vectors...")
    vectors = [np.random.rand(dim).tolist() for _ in range(n_vectors)]
    df = pd.DataFrame({
        'id': range(n_vectors),
        'embedding': vectors,
        'category': [f'cat_{i % 10}' for i in range(n_vectors)]
    })
    
    # Test 1: Default (HNSW-IVF)
    print("\n" + "="*60)
    print("Test 1: Default Index (HNSW-IVF)")
    print("="*60)
    table_path = "file:///tmp/test_hnsw_ivf_default_pytest"
    
    # Clean up previous run if exists
    import shutil
    import os
    if os.path.exists("/tmp/test_hnsw_ivf_default_pytest"):
        shutil.rmtree("/tmp/test_hnsw_ivf_default_pytest")

    table_default = hdb.Table(table_path)
    table_default.add_index_columns(["embedding"]) 
    start = time.time()
    table_default.write_pandas(df)
    build_time = time.time() - start
    print(f"✓ Build time: {build_time:.2f}s")
    
    # Test vector search
    print("\n" + "="*60)
    print("Testing Vector Search")
    print("="*60)
    query_vec = vectors[0] # Should match ID 0
    
    print("\nSearching with HNSW-IVF (default)...")
    start = time.time()
    results_default = table_default.to_pandas(
        vector_filter={"column": "embedding", "query": query_vec, "k": 10}
    )
    search_time = time.time() - start
    print(f"✓ Found {len(results_default)} results in {search_time*1000:.1f}ms")
    
    assert len(results_default) > 0, "Expected results but found 0!"
    
    # Verify we found the exact match
    # Depending on result structure, adjust check. Assuming it returns the dataframe rows.
    # We used ID 0's vector, so ID 0 must be in top results (likely first)
    found_id_0 = 0 in results_default['id'].values
    assert found_id_0, "Did not find the exact match (ID 0) in results"

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_hnsw_ivf_integration()
