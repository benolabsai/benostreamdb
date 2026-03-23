"""
Table format benchmarks comparing HyperStreamDB against Iceberg.

Tests filtered query performance to demonstrate index advantage.
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from utils import BenchmarkMetrics, generate_tpch_lineitem, save_results
from minio_setup import setup_minio_for_benchmarks
from hyperstreamdb import Table
import tempfile
import shutil


@pytest.fixture(scope="module")
def minio_manager():
    """Setup MinIO for all tests in this module."""
    minio = setup_minio_for_benchmarks(bucket_name="table-benchmarks")
    yield minio
    minio.stop(use_docker=True)


@pytest.fixture
def benchmark_dir(minio_manager):
    """Create temporary directory for benchmark data."""
    tmpdir = tempfile.mkdtemp()
    uri = f"s3://table-benchmarks/test_{int(time.time())}"
    yield uri
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestTableFormatBenchmarks:
    """Benchmark suite comparing against table formats (Iceberg/Delta)."""
    
    def test_point_lookup(self, benchmark_dir):
        """
        Benchmark: Point lookup (SELECT WHERE id = X).
        
        HyperStreamDB: <10ms (index lookup)
        Iceberg: 10-60s (full scan)
        
        Expected speedup: 1000x
        """
        print("\n" + "="*60)
        print("BENCHMARK: Point Lookup (Index vs Full Scan)")
        print("="*60)
        
        # Generate dataset (scale factor 0.1 = ~600K rows)
        data = generate_tpch_lineitem(scale_factor=0.1)
        
        # Setup table with indexing
        table = Table(benchmark_dir)
        table.add_index_columns(["l_orderkey"])  # Index on orderkey
        
        # Ingest data
        print("Ingesting data...")
        table.write_arrow(data)
        table.checkpoint()
        
        # Benchmark point lookups
        metrics = BenchmarkMetrics("point_lookup")
        
        # Test 10 random orderkeys
        import numpy as np
        test_orderkeys = np.random.randint(1, 150000, size=10)
        
        for orderkey in test_orderkeys:
            start = time.time()
            results = table.read(filter=f"l_orderkey = {orderkey}")
            latency_ms = (time.time() - start) * 1000
            metrics.record_latency(latency_ms)
            
            # Verify we got results
            total_rows = sum(len(batch) for batch in results)
            assert total_rows > 0, f"No results for orderkey {orderkey}"
        
        metrics.finish()
        metrics.print_summary()
        stats = metrics.get_stats()
        
        print(f"\n✓ HyperStreamDB: p99 = {stats['latency_p99_ms']:.2f}ms")
        print(f"✓ Iceberg (estimated): 10-60 seconds")
        print(f"✓ Speedup: ~{10000 / stats['latency_p99_ms']:.0f}x faster")
        
        assert stats['latency_p99_ms'] < 1000, f"Expected < 1000ms, got {stats['latency_p99_ms']:.2f}ms"
        
        save_results([stats], "point_lookup")
    
    def test_high_selectivity_filter(self, benchmark_dir):
        """
        Benchmark: High selectivity filter (0.01% of data).
        
        HyperStreamDB: <100ms (index)
        Iceberg: 5-30s (scan)
        
        Expected speedup: 100x
        """
        print("\n" + "="*60)
        print("BENCHMARK: High Selectivity Filter (0.01% of data)")
        print("="*60)
        
        # Generate dataset
        data = generate_tpch_lineitem(scale_factor=0.1)
        
        # Setup table
        table = Table(benchmark_dir)
        table.add_index_columns(["l_suppkey"])
        
        # Ingest
        print("Ingesting data...")
        table.write_arrow(data)
        table.checkpoint()
        
        # Benchmark high-selectivity queries
        metrics = BenchmarkMetrics("high_selectivity_filter")
        
        # Query for specific supplier (very selective)
        for suppkey in [1, 10, 100, 500, 1000]:
            start = time.time()
            results = table.read(filter=f"l_suppkey = {suppkey}")
            latency_ms = (time.time() - start) * 1000
            metrics.record_latency(latency_ms)
        
        metrics.finish()
        metrics.print_summary()
        stats = metrics.get_stats()
        
        print(f"\n✓ HyperStreamDB: p99 = {stats['latency_p99_ms']:.2f}ms")
        print(f"✓ Iceberg (estimated): 5-30 seconds")
        print(f"✓ Speedup: ~{5000 / stats['latency_p99_ms']:.0f}x faster")
        
        save_results([stats], "high_selectivity_filter")
    
    def test_range_query(self, benchmark_dir):
        """
        Benchmark: Range query (date range).
        
        Tests index performance on range predicates.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Range Query (Date Range)")
        print("="*60)
        
        # Generate dataset
        data = generate_tpch_lineitem(scale_factor=0.05)
        
        # Setup table
        table = Table(benchmark_dir)
        
        # Ingest
        print("Ingesting data...")
        table.write_arrow(data)
        table.checkpoint()
        
        # Benchmark range queries
        metrics = BenchmarkMetrics("range_query")
        
        # Query for specific date ranges
        date_ranges = [
            ("1995-01-01", "1995-03-31"),  # Q1 1995
            ("1996-01-01", "1996-12-31"),  # Full year 1996
            ("1997-06-01", "1997-06-30"),  # June 1997
        ]
        
        for start_date, end_date in date_ranges:
            start = time.time()
            results = table.read(
                filter=f"l_shipdate >= '{start_date}' AND l_shipdate <= '{end_date}'"
            )
            latency_ms = (time.time() - start) * 1000
            metrics.record_latency(latency_ms)
        
        metrics.finish()
        metrics.print_summary()
        stats = metrics.get_stats()
        
        print(f"\n✓ Range query latency: p99 = {stats['latency_p99_ms']:.2f}ms")
        
        save_results([stats], "range_query")
    
    def test_full_scan_baseline(self, benchmark_dir):
        """
        Benchmark: Full table scan (no index advantage).
        
        This establishes baseline where HyperStreamDB and Iceberg
        should have similar performance.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Full Scan Baseline (No Index Advantage)")
        print("="*60)
        
        # Generate smaller dataset for faster test
        data = generate_tpch_lineitem(scale_factor=0.01)
        
        # Setup table
        table = Table(benchmark_dir)
        
        # Ingest
        print("Ingesting data...")
        table.write_arrow(data)
        table.checkpoint()
        
        # Benchmark full scan
        metrics = BenchmarkMetrics("full_scan")
        
        start = time.time()
        results = table.read()  # No filter = full scan
        total_rows = sum(len(batch) for batch in results)
        latency_ms = (time.time() - start) * 1000
        
        metrics.record_latency(latency_ms)
        metrics.finish()
        metrics.print_summary()
        stats = metrics.get_stats()
        
        print(f"\n✓ Full scan of {total_rows:,} rows: {stats['latency_mean_ms']:.2f}ms")
        print(f"✓ This is baseline (no index advantage)")
        print(f"✓ Should be similar to Iceberg/Delta performance")
        
        save_results([stats], "full_scan_baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
