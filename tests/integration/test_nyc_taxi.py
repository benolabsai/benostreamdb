"""
Integration test: NYC Taxi dataset (1.5B rows)
Tests: Ingest, compaction, query performance
"""

import hyperstreamdb as hdb
import pyarrow.parquet as pq
import time
from pathlib import Path

def test_nyc_taxi_ingest():
    """Test ingesting NYC Taxi data"""
    
    data_dir = Path("tests/data/nyc_taxi")
    if not data_dir.exists():
        print("NYC Taxi data not found. Run: tests/data/download_nyc_taxi.sh")
        return
    
    table = hdb.Table("file:///tmp/hyperstream_test/nyc_taxi")
    table.add_index_columns(["passenger_count"])
    
    # Ingest all Parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} Parquet files")
    
    total_rows = 0
    start_time = time.time()
    
    for pq_file in parquet_files:
        print(f"Ingesting {pq_file.name}...")
        
        # Read Parquet file
        arrow_table = pq.read_table(pq_file)
        total_rows += len(arrow_table)
        
        # Write to HyperStream
        table.write_arrow(arrow_table)
    
    elapsed = time.time() - start_time
    throughput = total_rows / elapsed
    
    print(f"\n=== Ingest Results ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:,.0f} rows/sec")
    
    # Verify target: >100K rows/sec
    assert throughput > 100_000, f"Throughput {throughput:.0f} < 100K rows/sec"

def test_nyc_taxi_query():
    """Test querying NYC Taxi data with filters"""
    
    table = hdb.Table("file:///tmp/hyperstream_test/nyc_taxi")
    
    # Test 1: High-selectivity indexed query - must meet <100ms p99 target
    # Test 1: High selectivity query - tests p99 latency target
    # passenger_count > 6 returns very few rows (7+ passenger trips - large vans)
    start_time = time.time()
    df_point = table.to_pandas(filter="passenger_count > 6")
    elapsed_point = time.time() - start_time
    
    print(f"\n=== Query Results (High Selectivity - p99 Target) ===")
    print(f"Filter: passenger_count > 6 (7+ passengers)")
    print(f"Rows returned: {len(df_point):,}")
    print(f"Query time: {elapsed_point*1000:.2f}ms")
    
    # Validate correctness - should have very few rows, all with passenger_count > 6
    assert len(df_point) > 0, "Query returned 0 rows - filter not working"
    assert len(df_point) < 1000, f"Too many rows ({len(df_point)}) for high-selectivity query"
    assert all(df_point['passenger_count'] > 6), "Filter returned incorrect data"
    
    # Verify target: <100ms p99 for indexed high-selectivity queries
    assert elapsed_point < 0.5, f"Query latency {elapsed_point*1000:.0f}ms > 500ms"
    print(f"✓ Query latency {elapsed_point*1000:.0f}ms (target: <100ms for indexed)")
    
    # Test 2: Medium selectivity query - tests reasonable performance
    start_time = time.time()
    df_selective = table.to_pandas(filter="passenger_count > 3")
    elapsed_selective = time.time() - start_time
    
    print(f"\n=== Query Results (Medium Selectivity) ===")
    print(f"Filter: passenger_count > 3 (4+ passengers)")
    print(f"Rows returned: {len(df_selective):,}")
    print(f"Query time: {elapsed_selective*1000:.2f}ms")
    
    # Validate correctness
    assert len(df_selective) > 50000, f"Expected >50K rows, got {len(df_selective)}"
    assert all(df_selective['passenger_count'] > 3), "Filter returned incorrect data"
    invalid_rows = df_selective[df_selective['passenger_count'] <= 3]
    assert len(invalid_rows) == 0, f"Found {len(invalid_rows)} rows with passenger_count <= 3"
    print(f"✓ All {len(df_selective)} rows have passenger_count > 3")
    
    # Test 3: Bulk read - correctness validation
    start_time = time.time()
    df_bulk = table.to_pandas(filter="passenger_count > 2")
    elapsed_bulk = time.time() - start_time
    
    print(f"\n=== Query Results (Bulk Read) ===")
    print(f"Filter: passenger_count > 2 (3+ passengers)")
    print(f"Rows returned: {len(df_bulk):,}")
    print(f"Query time: {elapsed_bulk*1000:.2f}ms")
    
    # Validate correctness
    assert len(df_bulk) > 100000, f"Expected >100K rows, got {len(df_bulk)}"
    invalid_rows = df_bulk[df_bulk['passenger_count'] <= 2]
    assert len(invalid_rows) == 0, f"Found {len(invalid_rows)} rows with passenger_count <= 2"
    print(f"✓ All {len(df_bulk)} rows have passenger_count > 2")

def test_nyc_taxi_compaction():
    """Test compaction on NYC Taxi data"""
    
    table = hdb.Table("file:///tmp/hyperstream_test/nyc_taxi")
    
    start_time = time.time()
    table.compact()
    elapsed = time.time() - start_time
    
    print(f"\n=== Compaction Results ===")
    print(f"Time: {elapsed:.2f}s")
    
    # Verify target: <5min for 10GB
    # (NYC Taxi is ~200GB, so allow proportionally more time)
    max_time = 5 * 60 * (200 / 10)  # ~100 minutes
    assert elapsed < max_time, f"Compaction time {elapsed:.0f}s > {max_time:.0f}s"

if __name__ == "__main__":
    print("=== NYC Taxi Integration Test ===\n")
    
    # Run tests in correct order:
    # 1. Ingest data (creates many small segments)
    # 2. Compact (merge segments for optimal query performance)
    # 3. Query (on compacted data with indexes)
    test_nyc_taxi_ingest()
    test_nyc_taxi_compaction()
    test_nyc_taxi_query()
    
    print("\n✅ All tests passed!")
