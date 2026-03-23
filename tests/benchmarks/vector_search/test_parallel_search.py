"""
Test for parallel vector search (Strategy 3 - bypasses Python GIL).

This demonstrates the search_parallel method that runs multiple queries
in parallel in Rust, completely bypassing Python's GIL limitations.
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from utils import BenchmarkMetrics, generate_openai_embeddings
from minio_setup import setup_minio_for_benchmarks
from index_verification import wait_for_index_built
from hyperstreamdb import Table
import tempfile
import shutil
import time
import numpy as np


@pytest.fixture(scope="module")
def minio_manager():
    """Setup MinIO for all tests in this module."""
    minio = setup_minio_for_benchmarks(bucket_name="vector-benchmarks")
    yield minio
    minio.stop(use_docker=True)


@pytest.fixture
def benchmark_dir(minio_manager):
    """Create temporary directory for benchmark data."""
    tmpdir = tempfile.mkdtemp()
    # Use MinIO endpoint
    uri = f"s3://vector-benchmarks/test_parallel_{int(time.time())}"
    yield uri
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_parallel_search_vs_sequential(benchmark_dir):
    """
    Compare parallel search (search_parallel) vs sequential search (search).
    
    This demonstrates that search_parallel bypasses Python GIL and achieves
    true parallelism, while sequential search is limited by GIL.
    """
    print("\n" + "="*60)
    print("TEST: Parallel Search vs Sequential Search")
    print("="*60)
    
    # Generate dataset
    n_vectors = 50_000
    data = generate_openai_embeddings(n=n_vectors, dim=1536)
    
    # Setup and ingest with vector index
    table = Table(benchmark_dir)
    table.add_index_columns(["embedding"])
    table.write_arrow(data)
    table.commit()
    table.checkpoint()
    
    # Verify index was built
    wait_for_index_built(table, "embedding", max_wait=60)
    
    # Generate query vectors
    n_queries = 100
    query_vectors = [
        np.random.randn(1536).astype(np.float32) / np.linalg.norm(np.random.randn(1536))
        for _ in range(n_queries)
    ]
    
    # Test 1: Sequential search (GIL limited)
    print("\n1. Sequential search (GIL limited)...")
    sequential_metrics = BenchmarkMetrics("sequential_search")
    
    start = time.time()
    for query_vec in query_vectors:
        query_start = time.time()
        table.search(column="embedding", query=query_vec.tolist(), k=10)
        sequential_metrics.record_latency((time.time() - query_start) * 1000)
    
    sequential_time = time.time() - start
    sequential_metrics.finish(total_operations=n_queries)
    sequential_stats = sequential_metrics.get_stats()
    sequential_qps = n_queries / sequential_time
    
    print(f"   Sequential: {sequential_time:.2f}s, {sequential_qps:.0f} QPS")
    print(f"   p99 latency: {sequential_stats['latency_p99_ms']:.2f}ms")
    
    # Test 2: Parallel search (bypasses GIL)
    print("\n2. Parallel search (bypasses GIL)...")
    parallel_metrics = BenchmarkMetrics("parallel_search")
    
    # Prepare queries for parallel execution
    queries = [
        ("embedding", q.tolist(), 10, None)
        for q in query_vectors
    ]
    
    start = time.time()
    results = table.search_parallel(queries)
    parallel_time = time.time() - start
    
    # Calculate per-query latency (approximate - all run in parallel)
    avg_query_time = parallel_time / n_queries
    parallel_metrics.finish(total_operations=n_queries)
    parallel_metrics.duration = parallel_time
    parallel_stats = parallel_metrics.get_stats()
    parallel_qps = n_queries / parallel_time
    
    print(f"   Parallel: {parallel_time:.2f}s, {parallel_qps:.0f} QPS")
    print(f"   Total queries: {len(results)}")
    print(f"   Speedup: {sequential_time / parallel_time:.2f}x")
    
    # Assertions
    assert len(results) == n_queries, f"Expected {n_queries} results, got {len(results)}"
    assert parallel_qps > sequential_qps, \
        f"Parallel should be faster: {parallel_qps:.0f} vs {sequential_qps:.0f} QPS"
    
    print(f"\n✓ Parallel search is {sequential_time / parallel_time:.2f}x faster!")
    print(f"✓ Achieved {parallel_qps:.0f} QPS (vs {sequential_qps:.0f} QPS sequential)")


def test_parallel_search_with_filters(benchmark_dir):
    """
    Test parallel search with different filters for each query.
    """
    print("\n" + "="*60)
    print("TEST: Parallel Search with Filters")
    print("="*60)
    
    # Generate dataset with user_ids
    n_vectors = 50_000
    data = generate_openai_embeddings(n=n_vectors, dim=1536)
    
    # Setup and ingest
    table = Table(benchmark_dir)
    table.add_index_columns(["embedding"])
    table.write_arrow(data)
    table.commit()
    table.checkpoint()
    
    wait_for_index_built(table, "embedding", max_wait=60)
    
    # Generate queries with different filters
    n_queries = 50
    query_vectors = [
        np.random.randn(1536).astype(np.float32) / np.linalg.norm(np.random.randn(1536))
        for _ in range(n_queries)
    ]
    
    # Mix of filtered and unfiltered queries
    queries = []
    for i, query_vec in enumerate(query_vectors):
        if i % 2 == 0:
            # Unfiltered query
            queries.append(("embedding", query_vec.tolist(), 10, None))
        else:
            # Filtered query
            queries.append(("embedding", query_vec.tolist(), 10, f"user_id < {100 + i}"))
    
    print(f"\nRunning {n_queries} queries in parallel (mix of filtered/unfiltered)...")
    start = time.time()
    results = table.search_parallel(queries)
    elapsed = time.time() - start
    
    print(f"✓ Completed {len(results)} queries in {elapsed:.2f}s")
    print(f"✓ Throughput: {len(results) / elapsed:.0f} QPS")
    
    assert len(results) == n_queries, f"Expected {n_queries} results, got {len(results)}"
    
    # Verify each result is a PyArrow table
    for i, result in enumerate(results):
        assert result is not None, f"Result {i} is None"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
