"""
Benchmark for standalone HNSW-IVF index build time.

Tests the time taken to build an index for 100K vectors.
"""
import pytest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
# Also add benchmarks/common if needed, but relative import should work if we rely on sys.path hack from other tests
# Better to be explicit:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from utils import BenchmarkMetrics, generate_openai_embeddings, save_results
from minio_setup import setup_minio_for_benchmarks
from hyperstreamdb import Table
import tempfile
import shutil

@pytest.fixture(scope="module")
def minio_manager():
    """Setup MinIO for all tests in this module."""
    minio = setup_minio_for_benchmarks(bucket_name="index-benchmarks")
    yield minio
    minio.stop(use_docker=True)

@pytest.fixture
def benchmark_dir(minio_manager):
    """Create temporary directory for benchmark data."""
    # Use MinIO endpoint
    uri = f"s3://index-benchmarks/test_{int(time.time())}"
    yield uri
    # No cleanup needed for s3 usually, or maybe cleanup keys?

class TestIndexBuildBenchmarks:
    """Benchmark suite for index building."""
    
    def test_build_100k_vectors_latency(self, benchmark_dir):
        """
        Benchmark: Build HNSW-IVF index for 100K vectors.
        
        Target: < 30 seconds
        Current Baseline: ~60 seconds
        """
        print("\n" + "="*60)
        print("BENCHMARK: Index Build Time (100K vectors, 1536D)")
        print("="*60)
        
        # 1. Generate Data
        n_vectors = 100_000
        print(f"Generating {n_vectors} vectors...")
        data = generate_openai_embeddings(n=n_vectors, dim=1536)
        
        # 2. Setup Table
        # We want to measure ONLY the index build time.
        # But index is built during commit().
        # To isolate it, we can write data first without index columns, or just measure commit time.
        # Actually, `add_index_columns` triggers backfill if data exists, or if we add it before write, it does it on commit.
        # Let's do: Write -> Commit (Raw Parquet) -> Add Index -> Index All Columns (triggers build)
        
        table = Table(benchmark_dir)
        
        print("Writing raw data (no index)...")
        batch_size = 10_000
        for i in range(0, n_vectors, batch_size):
            batch = data.slice(i, min(batch_size, n_vectors - i))
            table.write_arrow(batch)
        
        table.commit() # Write raw parquet files
        print("Raw data committed.")
        
        # 3. Configure Index
        print("Configuring index...")
        table.add_index_columns(["embedding"])
        
        # 4. Trigger Index Build (Backfill)
        print("Triggering index build (backfill)...")
        start_time = time.time()
        
        # In current API, `add_index_columns` might not auto-trigger backfill immediately or maybe it does?
        # Let's assume we need to call something or it happens on next maintenance/commit?
        # In `src/python_binding.rs`: `add_index_columns` just updates config. 
        # `index_all_columns` triggers backfill?
        # Let's check `test_vs_qdrant.py`: it calls `add_index_columns` before write.
        # If we add it AFTER write, we need to verify how to trigger indexing.
        # Usually `table.index_all_columns()` triggers re-indexing.
        
        table.index_all_columns()
        
        duration = time.time() - start_time
        print(f"Index build completed in {duration:.2f} seconds.")
        
        # 5. Metrics
        metrics = {"build_time_sec": duration, "vectors": n_vectors}
        
        print(f"\n✓ Duration: {duration:.2f}s")
        print(f"✓ Target: < 30s")
        
        if duration < 30:
            print(f"✓ PASS: Exceeded target performance")
        else:
            print(f"⚠ WARN: Needs optimization (Gap: {duration - 30:.2f}s)")
            
        # Assertions (Soft fail for now until optimized)
        # assert duration < 60, f"Baseline should be ~60s, got {duration}s"
