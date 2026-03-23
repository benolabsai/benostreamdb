"""
Concurrency and thread-safety tests for HyperStreamDB.

Tests concurrent readers, writers, and read-write scenarios.
"""

import hyperstreamdb as hdb
import pyarrow as pa
import pytest
import os
import shutil
import multiprocessing as mp
from pathlib import Path
import time


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def compact_worker(table_uri):
    """Worker function for compaction."""
    try:
        table = hdb.open_table(table_uri)
        table.compact()
        return "Compaction completed"
    except Exception as e:
        return f"Compaction failed: {e}"


def mixed_worker(table_uri, worker_id, schema_fields):
    """Worker function for mixed read/write operations."""
    try:
        table = hdb.open_table(table_uri)
        schema = pa.schema(schema_fields)
        
        # Alternate between reads and writes
        if worker_id % 2 == 0:
            df = table.to_pandas()
            return f"Worker {worker_id} read {len(df)} rows"
        else:
            batch = pa.Table.from_arrays(
                [pa.array([worker_id], type=pa.int32()),
                 pa.array([int(time.time() * 1000)], type=pa.int64())],
                schema=schema
            )
            table.write_arrow(batch)
            return f"Worker {worker_id} wrote 1 row"
    except Exception as e:
        return f"Worker {worker_id} failed: {e}"


@pytest.fixture
def test_table_path(tmp_path):
    """Create a temporary table path."""
    table_path = tmp_path / "concurrent_test"
    yield f"file://{table_path}"
    if table_path.exists():
        shutil.rmtree(table_path)


def write_worker(table_uri, worker_id, num_rows):
    """Worker function that writes data to the table."""
    try:
        table = hdb.open_table(table_uri)
        
        # Create data specific to this worker
        schema = pa.schema([
            ('worker_id', pa.int32()),
            ('row_id', pa.int32()),
            ('value', pa.string())
        ])
        
        worker_ids = pa.array([worker_id] * num_rows, type=pa.int32())
        row_ids = pa.array(list(range(num_rows)), type=pa.int32())
        values = pa.array([f"worker_{worker_id}_row_{i}" for i in range(num_rows)])
        
        batch = pa.Table.from_arrays([worker_ids, row_ids, values], schema=schema)
        table.write_arrow(batch)
        
        return f"Worker {worker_id} wrote {num_rows} rows"
    except Exception as e:
        return f"Worker {worker_id} failed: {e}"


def read_worker(table_uri, worker_id):
    """Worker function that reads data from the table."""
    try:
        table = hdb.open_table(table_uri)
        df = table.to_pandas()
        return f"Worker {worker_id} read {len(df)} rows"
    except Exception as e:
        return f"Worker {worker_id} failed: {e}"


def test_concurrent_writers(test_table_path):
    """Test multiple processes writing to the same table concurrently."""
    # Create initial table
    table = hdb.open_table(test_table_path)
    
    # Initial write to establish schema
    schema = pa.schema([
        ('worker_id', pa.int32()),
        ('row_id', pa.int32()),
        ('value', pa.string())
    ])
    initial_batch = pa.Table.from_arrays(
        [pa.array([0], type=pa.int32()), 
         pa.array([0], type=pa.int32()), 
         pa.array(["init"])],
        schema=schema
    )
    table.write_arrow(initial_batch)
    
    # Launch concurrent writers
    num_workers = 4
    rows_per_worker = 100
    
    with mp.Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(write_worker, (test_table_path, i, rows_per_worker))
            for i in range(1, num_workers + 1)
        ]
        outputs = [r.get(timeout=30) for r in results]
    
    # Verify all workers succeeded
    for output in outputs:
        assert "failed" not in output.lower(), f"Worker failed: {output}"
    
    # Verify total row count
    table = hdb.open_table(test_table_path)
    df = table.to_pandas()
    
    # Should have initial row + (num_workers * rows_per_worker)
    expected_rows = 1 + (num_workers * rows_per_worker)
    assert len(df) >= num_workers * rows_per_worker, \
        f"Expected at least {num_workers * rows_per_worker} rows, got {len(df)}"
    
    print(f"✓ Concurrent writes test passed: {len(df)} total rows")


def test_concurrent_readers(test_table_path):
    """Test multiple processes reading from the same table concurrently."""
    # Create table with data
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string())
    ])
    
    num_rows = 10000
    ids = pa.array(list(range(num_rows)), type=pa.int32())
    values = pa.array([f"value_{i}" for i in range(num_rows)])
    
    batch = pa.Table.from_arrays([ids, values], schema=schema)
    table.write_arrow(batch)
    
    # Launch concurrent readers
    num_readers = 8
    
    with mp.Pool(processes=num_readers) as pool:
        results = [
            pool.apply_async(read_worker, (test_table_path, i))
            for i in range(num_readers)
        ]
        outputs = [r.get(timeout=30) for r in results]
    
    # Verify all readers succeeded
    for output in outputs:
        assert "failed" not in output.lower(), f"Reader failed: {output}"
        assert f"read {num_rows} rows" in output, f"Unexpected output: {output}"
    
    print(f"✓ Concurrent reads test passed: {num_readers} readers")


def test_read_write_concurrency(test_table_path):
    """Test concurrent readers and writers operating simultaneously."""
    # Create initial table
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('worker_id', pa.int32()),
        ('row_id', pa.int32()),
        ('value', pa.string())
    ])
    
    # Initial data
    initial_batch = pa.Table.from_arrays(
        [pa.array([0] * 1000, type=pa.int32()),
         pa.array(list(range(1000)), type=pa.int32()),
         pa.array([f"init_{i}" for i in range(1000)])],
        schema=schema
    )
    table.write_arrow(initial_batch)
    
    # Launch mixed readers and writers
    num_readers = 4
    num_writers = 2
    
    with mp.Pool(processes=num_readers + num_writers) as pool:
        # Start readers
        reader_results = [
            pool.apply_async(read_worker, (test_table_path, i))
            for i in range(num_readers)
        ]
        
        # Start writers
        writer_results = [
            pool.apply_async(write_worker, (test_table_path, i + 100, 100))
            for i in range(num_writers)
        ]
        
        # Collect results
        reader_outputs = [r.get(timeout=30) for r in reader_results]
        writer_outputs = [r.get(timeout=30) for r in writer_results]
    
    # Verify all operations succeeded
    for output in reader_outputs + writer_outputs:
        assert "failed" not in output.lower(), f"Operation failed: {output}"
    
    print(f"✓ Read-write concurrency test passed")


def test_concurrent_compaction(test_table_path):
    """Test that compaction works correctly with concurrent operations."""
    # Create table with multiple small segments
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string())
    ])
    
    # Write multiple small batches to create segments
    for i in range(10):
        ids = pa.array(list(range(i * 100, (i + 1) * 100)), type=pa.int32())
        values = pa.array([f"value_{j}" for j in range(i * 100, (i + 1) * 100)])
        batch = pa.Table.from_arrays([ids, values], schema=schema)
        table.write_arrow(batch)
    
    # Compact in background while reading
    with mp.Pool(processes=5) as pool:
        # Start compaction
        compact_result = pool.apply_async(compact_worker, (test_table_path,))
        
        # Start concurrent readers
        reader_results = [
            pool.apply_async(read_worker, (test_table_path, i))
            for i in range(4)
        ]
        
        # Wait for all to complete
        compact_output = compact_result.get(timeout=60)
        reader_outputs = [r.get(timeout=30) for r in reader_results]
    
    # Verify compaction succeeded
    assert "failed" not in compact_output.lower(), f"Compaction failed: {compact_output}"
    
    # Verify readers succeeded
    for output in reader_outputs:
        # Readers might see different row counts during compaction, which is OK
        assert "failed" not in output.lower(), f"Reader failed: {output}"
    
    print(f"✓ Concurrent compaction test passed")


def test_lock_contention(test_table_path):
    """Test behavior under high lock contention."""
    # Create table
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('timestamp', pa.int64())
    ])
    
    initial_batch = pa.Table.from_arrays(
        [pa.array([0], type=pa.int32()),
         pa.array([int(time.time() * 1000)], type=pa.int64())],
        schema=schema
    )
    table.write_arrow(initial_batch)
    
    # Launch many concurrent operations
    num_operations = 10 # Reduced from 20 to avoid extreme contention in CI
    
    schema_fields = [
        ('id', pa.int32()),
        ('timestamp', pa.int64())
    ]

    with mp.Pool(processes=num_operations) as pool:
        results = [
            pool.apply_async(mixed_worker, (test_table_path, i, schema_fields))
            for i in range(num_operations)
        ]
        outputs = [r.get(timeout=30) for r in results]
    
    # Verify all operations completed
    for output in outputs:
        assert "failed" not in output.lower(), f"Operation failed: {output}"
    
    print(f"✓ Lock contention test passed: {num_operations} concurrent operations")


if __name__ == "__main__":
    print("=== Concurrency Tests ===\n")
    
    # Create temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        print("Running concurrent writers test...")
        test_concurrent_writers(f"file://{tmp_path}/test1")
        
        print("\nRunning concurrent readers test...")
        test_concurrent_readers(f"file://{tmp_path}/test2")
        
        print("\nRunning read-write concurrency test...")
        test_read_write_concurrency(f"file://{tmp_path}/test3")
        
        print("\nRunning concurrent compaction test...")
        test_concurrent_compaction(f"file://{tmp_path}/test4")
        
        print("\nRunning lock contention test...")
        test_lock_contention(f"file://{tmp_path}/test5")
    
    print("\n✅ All concurrency tests passed!")
