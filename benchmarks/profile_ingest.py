#!/usr/bin/env python3
"""
Profile HyperStreamDB ingest performance to identify bottlenecks

Compares:
1. Raw Parquet write (baseline)
2. HyperStreamDB write (with Iceberg overhead)
3. Step-by-step timing of each operation
"""

import time
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import cProfile
import pstats
from io import StringIO

try:
    import hyperstreamdb as hdb
except ImportError:
    print("Error: hyperstreamdb not installed")
    print("Run: pip install -e .")
    exit(1)


def generate_test_data(n_rows: int, dim: int = 768):
    """Generate test data"""
    print(f"Generating {n_rows:,} test vectors ({dim}-dim)...")
    
    vectors = np.random.randn(n_rows, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    metadata = pd.DataFrame({
        'id': range(n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'price': np.random.uniform(10, 1000, n_rows),
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min'),
        'text': [f'Document {i}' for i in range(n_rows)],
        'embedding': list(vectors),
    })
    
    return metadata


def profile_raw_parquet(df: pd.DataFrame, tmpdir: str):
    """Baseline: Raw Parquet write"""
    print("\n" + "="*80)
    print("BASELINE: Raw Parquet Write")
    print("="*80)
    
    parquet_path = f"{tmpdir}/raw.parquet"
    
    start = time.time()
    df.to_parquet(parquet_path, compression='snappy')
    elapsed = time.time() - start
    
    size_mb = Path(parquet_path).stat().st_size / 1024 / 1024
    throughput = len(df) / elapsed
    
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:,.0f} rows/sec")
    print(f"Size: {size_mb:.1f} MB")
    
    return elapsed, throughput


def profile_hyperstreamdb_detailed(df: pd.DataFrame, tmpdir: str):
    """Detailed timing of HyperStreamDB operations"""
    print("\n" + "="*80)
    print("HYPERSTREAMDB: Detailed Timing")
    print("="*80)
    
    uri = f"file://{tmpdir}/test_table"
    
    # 1. Table creation
    t0 = time.time()
    table = hdb.Table(uri)
    t_create = time.time() - t0
    print(f"1. Table creation: {t_create*1000:.1f}ms")
    
    # 2. Write data
    t0 = time.time()
    table.write_pandas(df)
    t_write = time.time() - t0
    print(f"2. write_pandas(): {t_write*1000:.1f}ms ({len(df)/t_write:,.0f} rows/sec)")
    
    # 3. Commit (manifest creation)
    t0 = time.time()
    table.commit()
    t_commit = time.time() - t0
    print(f"3. commit(): {t_commit*1000:.1f}ms")
    
    # Total time (excluding table creation overhead)
    total_time = t_write + t_commit
    throughput = len(df) / total_time
    
    # Calculate storage
    table_path = Path(tmpdir) / "test_table"
    size_mb = sum(f.stat().st_size for f in table_path.rglob('*') if f.is_file()) / 1024 / 1024
    
    print(f"\nTotal time (write+commit): {total_time:.3f}s")
    print(f"Throughput: {throughput:,.0f} rows/sec")
    print(f"Size: {size_mb:.1f} MB")
    
    return total_time, throughput


def profile_hyperstreamdb_with_profiler(df: pd.DataFrame, tmpdir: str):
    """Profile with cProfile to find hotspots"""
    print("\n" + "="*80)
    print("HYPERSTREAMDB: cProfile Analysis")
    print("="*80)
    
    uri = f"file://{tmpdir}/test_table_profile"
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the ingest
    table = hdb.Table(uri)
    table.write_pandas(df)
    table.commit()
    
    profiler.disable()
    
    # Print top 20 time-consuming functions
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())


def compare_with_without_vectors(n_rows: int):
    """Compare ingest with and without vector column"""
    print("\n" + "="*80)
    print("COMPARISON: With vs Without Vectors")
    print("="*80)
    
    # Without vectors
    df_no_vec = pd.DataFrame({
        'id': range(n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'price': np.random.uniform(10, 1000, n_rows),
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min'),
        'text': [f'Document {i}' for i in range(n_rows)],
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        uri = f"file://{tmpdir}/test_table"
        table = hdb.Table(uri)
        
        start = time.time()
        table.write_pandas(df_no_vec)
        table.commit()
        elapsed_no_vec = time.time() - start
        throughput_no_vec = n_rows / elapsed_no_vec
    
    print(f"Without vectors: {throughput_no_vec:,.0f} rows/sec")
    
    # With vectors
    vectors = np.random.randn(n_rows, 768).astype(np.float32)
    df_with_vec = df_no_vec.copy()
    df_with_vec['embedding'] = list(vectors)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        uri = f"file://{tmpdir}/test_table"
        table = hdb.Table(uri)
        
        start = time.time()
        table.write_pandas(df_with_vec)
        table.commit()
        elapsed_with_vec = time.time() - start
        throughput_with_vec = n_rows / elapsed_with_vec
    
    print(f"With vectors (768-dim): {throughput_with_vec:,.0f} rows/sec")
    print(f"Slowdown: {throughput_no_vec/throughput_with_vec:.1f}x")
    
    # Calculate vector data size
    vector_size_mb = (n_rows * 768 * 4) / 1024 / 1024
    print(f"Vector data size: {vector_size_mb:.1f} MB")


def analyze_iceberg_overhead(n_rows: int):
    """Analyze Iceberg-specific overhead"""
    print("\n" + "="*80)
    print("ICEBERG OVERHEAD ANALYSIS")
    print("="*80)
    
    df = generate_test_data(n_rows, dim=768)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Raw Parquet
        parquet_time, parquet_throughput = profile_raw_parquet(df, tmpdir)
        
        # 2. HyperStreamDB (with Iceberg)
        hsdb_time, hsdb_throughput = profile_hyperstreamdb_detailed(df, tmpdir)
        
        # 3. Calculate overhead
        overhead_time = hsdb_time - parquet_time
        overhead_pct = (overhead_time / parquet_time) * 100
        
        print("\n" + "="*80)
        print("OVERHEAD SUMMARY")
        print("="*80)
        print(f"Raw Parquet: {parquet_throughput:,.0f} rows/sec")
        print(f"HyperStreamDB: {hsdb_throughput:,.0f} rows/sec")
        print(f"Slowdown: {parquet_throughput/hsdb_throughput:.1f}x")
        print(f"Overhead: {overhead_time:.3f}s ({overhead_pct:.1f}%)")
        
        # 4. Profile to find hotspots
        profile_hyperstreamdb_with_profiler(df, tmpdir)


def main():
    """Run all profiling tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile HyperStreamDB ingest performance')
    parser.add_argument('--rows', type=int, default=10000, help='Number of rows (default: 10000)')
    parser.add_argument('--quick', action='store_true', help='Quick test with 1K rows')
    
    args = parser.parse_args()
    
    n_rows = 1000 if args.quick else args.rows
    
    print("="*80)
    print(f"HYPERSTREAMDB INGEST PROFILING ({n_rows:,} rows)")
    print("="*80)
    
    # 1. Compare with/without vectors
    compare_with_without_vectors(n_rows)
    
    # 2. Analyze Iceberg overhead
    analyze_iceberg_overhead(n_rows)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nKey Questions:")
    print("1. Is the slowdown due to vector data size?")
    print("2. Is the slowdown due to Iceberg manifest creation?")
    print("3. What are the top time-consuming functions?")
    print("\nNext Steps:")
    print("- Optimize identified bottlenecks")
    print("- Consider batching manifest updates")
    print("- Profile Parquet write performance")


if __name__ == '__main__':
    main()
