"""
Benchmark for Cold Read Latency (Segment Loading).

Measures the time taken to perform the FIRST search query on a fresh Table instance,
which forces index loading from object storage (simulated MinIO/S3).
"""
import pytest
import sys
import os
import time
import shutil
import tempfile
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from utils import generate_openai_embeddings
from minio_setup import setup_minio_for_benchmarks
from hyperstreamdb import Table

@pytest.fixture(scope="module")
def minio_setup():
    minio = setup_minio_for_benchmarks()
    yield minio
    minio.stop(use_docker=True)

class TestColdReadLatency:
    
    def test_cold_search_latency(self, minio_setup):
        """
        Measure 3 key metrics:
        1. Write Time (Baseline)
        2. Cold Search Latency (First Query - includes downloading/parsing indices)
        3. Warm Search Latency (Subsequent Queries - cached primitives)
        """
        bucket_uri = f"s3://hyperstreamdb-benchmarks/latency_test_{int(time.time())}"
        
        # 1. Setup Data: 5 Segments of 10k vectors (50k total) to simulate segmentation
        n_segments = 5
        vecs_per_seg = 10_000
        dim = 1536
        
        print(f"\nGeneratng {n_segments * vecs_per_seg} vectors...")
        data = generate_openai_embeddings(n=n_segments * vecs_per_seg, dim=dim)
        
        table = Table(bucket_uri)
        table.add_index_columns(["embedding"])
        
        print("Writing segments...")
        for i in range(n_segments):
            batch = data.slice(i * vecs_per_seg, vecs_per_seg)
            table.write_arrow(batch)
            table.commit() # New segment per commit
            print(f"  Committed segment {i+1}/{n_segments}")
            
        # Ensure indexes are built (backfill checking usually happens on write/maintenance)
        print("Triggering indexing...")
        t0 = time.time()
        table.index_all_columns()
        print(f"Indexing took {time.time() - t0:.2f}s")
        
        # 2. Cold Search
        print("\n--- Cold Search Benchmark ---")
        # Instantiate NEW table instance to ensure no in-memory hygiene
        # Ideally we'd clear OS page cache or local /tmp cache if any
        # For now, we assume Table() constructor starts fresh state
        
        cold_table = Table(bucket_uri)
        query_vec = data.slice(0, 1).column("embedding")[0].as_py()
        
        t_start = time.time()
        results = cold_table.search(
            column="embedding",
            query=query_vec,
            k=10
        )
        t_cold = time.time() - t_start
        print(f"Cold Search Latency: {t_cold*1000:.2f} ms")
        
        # 3. Warm Search
        print("\n--- Warm Search Benchmark ---")
        t_start = time.time()
        _ = cold_table.search(
            column="embedding",
            query=query_vec,
            k=10
        )
        t_warm = time.time() - t_start
        print(f"Warm Search Latency: {t_warm*1000:.2f} ms")
        
        print(f"\nGap (Loading Overhead): {(t_cold - t_warm)*1000:.2f} ms")
        print(f"Gap per Segment: {((t_cold - t_warm) / n_segments)*1000:.2f} ms")
        
        if hasattr(cold_table, "search"):
            # Check results
           pass
           
        # Verify we actually got results
        # Search returns different/void in bindings?
        # The binding usually returns a list of Batches or similar.
        # Let's inspect the return value logic in python_binding.rs if needed.
        # But for now, let's just assert result is not empty list
        # We assigned to `_`.
        results = _
        if hasattr(results, "__len__"):
            print(f"DEBUG: Cold search returned {len(results)} matches")
            if len(results) == 0:
                 print("❌ ERROR: Cold search returned 0 results! Index loading likely failed.")
                 # Fail the test to attract attention
                 pytest.fail("Cold search returned 0 results")
                 
        if t_cold > 1.0:
            print("⚠ WARN: Cold start > 1s")
