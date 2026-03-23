"""
Performance benchmarks for HyperStreamDB.

Tests both in-memory and out-of-memory scenarios to validate performance
characteristics under different memory pressure conditions.
"""
import pytest
import time
import psutil
import os
import tempfile
import shutil
import numpy as np
import pyarrow as pa
from hyperstreamdb import Table


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestIngestThroughput:
    """Test write performance for in-memory and out-of-memory datasets."""
    
    def test_ingest_throughput_in_memory(self, temp_dir):
        """Test ingest throughput with data that fits in memory (~10MB)."""
        table = Table(f"file://{temp_dir}")
        
        # Small dataset: 100K rows × 3 columns × ~8 bytes = ~2.4MB
        num_rows = 100_000
        batch_size = 10_000
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
            pa.field("category", pa.int32()),
        ])
        
        start_time = time.time()
        total_rows = 0
        
        for i in range(0, num_rows, batch_size):
            batch = pa.RecordBatch.from_arrays([
                pa.array(range(i, i + batch_size), type=pa.int64()),
                pa.array(np.random.random(batch_size), type=pa.float64()),
                pa.array(np.random.randint(0, 10, batch_size), type=pa.int32()),
            ], schema=schema)
            
            table.write([batch])
            total_rows += batch_size
        
        table.commit()
        elapsed = time.time() - start_time
        throughput = total_rows / elapsed
        
        print(f"\n[IN-MEMORY] Ingest throughput: {throughput:,.0f} rows/sec")
        print(f"Total rows: {total_rows:,}, Time: {elapsed:.2f}s")
        
        # Relaxed for CI/test environments
        assert throughput > 30_000, f"Expected >30K rows/sec, got {throughput:,.0f}"
    
    def test_ingest_throughput_out_of_memory(self, temp_dir):
        """Test ingest throughput with large dataset requiring disk spill (~500MB)."""
        table = Table(f"file://{temp_dir}")
        
        # Large dataset: 5M rows × 3 columns × ~8 bytes = ~120MB raw data
        # With overhead and indexes, will exceed typical buffer sizes
        num_rows = 5_000_000
        batch_size = 50_000
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
            pa.field("category", pa.int32()),
        ])
        
        start_time = time.time()
        total_rows = 0
        mem_start = get_memory_usage_mb()
        
        for i in range(0, num_rows, batch_size):
            batch = pa.RecordBatch.from_arrays([
                pa.array(range(i, i + batch_size), type=pa.int64()),
                pa.array(np.random.random(batch_size), type=pa.float64()),
                pa.array(np.random.randint(0, 10, batch_size), type=pa.int32()),
            ], schema=schema)
            
            table.write([batch])
            total_rows += batch_size
            
            # Commit periodically to trigger disk writes
            if total_rows % 500_000 == 0:
                table.commit()
        
        table.commit()
        elapsed = time.time() - start_time
        throughput = total_rows / elapsed
        mem_end = get_memory_usage_mb()
        mem_delta = mem_end - mem_start
        
        print(f"\n[OUT-OF-MEMORY] Ingest throughput: {throughput:,.0f} rows/sec")
        print(f"Total rows: {total_rows:,}, Time: {elapsed:.2f}s")
        print(f"Memory delta: {mem_delta:.1f} MB")
        
        # Should still achieve >30K rows/sec even with disk spill
        assert throughput > 30_000, f"Expected >30K rows/sec, got {throughput:,.0f}"


class TestQueryLatency:
    """Test query performance for in-memory and out-of-memory datasets."""
    
    def test_query_latency_indexed_in_memory(self, temp_dir):
        """Test indexed query latency with in-memory dataset."""
        table = Table(f"file://{temp_dir}")
        table.index_columns = ["id"]
        
        # Small dataset: 100K rows
        num_rows = 100_000
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
        ])
        
        batch = pa.RecordBatch.from_arrays([
            pa.array(range(num_rows), type=pa.int64()),
            pa.array(np.random.random(num_rows), type=pa.float64()),
        ], schema=schema)
        
        table.write([batch])
        table.commit()
        table.wait_for_indexes()
        
        # Warm up
        _ = table.read(filter="id > 50000")
        
        # Benchmark indexed query
        latencies = []
        for _ in range(10):
            start = time.time()
            result = table.read(filter="id > 50000")
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            assert len(result) > 0
        
        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)
        
        print(f"\n[IN-MEMORY] Indexed query latency: p50={p50:.1f}ms, p99={p99:.1f}ms")
        
        # Should be very fast for in-memory indexed queries
        assert p99 < 100, f"Expected p99 < 100ms, got {p99:.1f}ms"
    
    def test_query_latency_full_scan_out_of_memory(self, temp_dir):
        """Test full scan latency with large out-of-memory dataset."""
        table = Table(f"file://{temp_dir}")
        
        # Large dataset: 1M rows
        num_rows = 1_000_000
        batch_size = 100_000
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
        ])
        
        for i in range(0, num_rows, batch_size):
            batch = pa.RecordBatch.from_arrays([
                pa.array(range(i, i + batch_size), type=pa.int64()),
                pa.array(np.random.random(batch_size), type=pa.float64()),
            ], schema=schema)
            table.write([batch])
        
        table.commit()
        
        # Benchmark full scan
        start = time.time()
        result = table.read()
        latency = (time.time() - start) * 1000  # ms
        
        total_rows = len(result)
        
        print(f"\n[OUT-OF-MEMORY] Full scan latency: {latency:.1f}ms for {total_rows:,} rows")
        print(f"Throughput: {total_rows / (latency / 1000):,.0f} rows/sec")
        
        assert total_rows == num_rows
        # Full scan should still be reasonably fast
        assert latency < 5000, f"Expected < 5000ms, got {latency:.1f}ms"


class TestCompactionSpeed:
    """Test compaction performance for in-memory and out-of-memory datasets."""
    
    def test_compaction_speed_in_memory(self, temp_dir):
        """Test compaction speed with small segments (in-memory)."""
        table = Table(f"file://{temp_dir}")
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
        ])
        
        # Create 20 small segments (5K rows each = 100K total)
        num_segments = 20
        rows_per_segment = 5_000
        
        for seg in range(num_segments):
            start_id = seg * rows_per_segment
            batch = pa.RecordBatch.from_arrays([
                pa.array(range(start_id, start_id + rows_per_segment), type=pa.int64()),
                pa.array(np.random.random(rows_per_segment), type=pa.float64()),
            ], schema=schema)
            table.write([batch])
            table.commit()
        
        # Benchmark compaction
        start = time.time()
        table.compact()
        elapsed = time.time() - start
        
        total_rows = num_segments * rows_per_segment
        throughput = total_rows / elapsed
        
        print(f"\n[IN-MEMORY] Compaction speed: {throughput:,.0f} rows/sec")
        print(f"Compacted {num_segments} segments ({total_rows:,} rows) in {elapsed:.2f}s")
        
        # Should be fast for in-memory compaction
        assert throughput > 50_000, f"Expected >50K rows/sec, got {throughput:,.0f}"
    
    def test_compaction_speed_out_of_memory(self, temp_dir):
        """Test compaction speed with large segments (out-of-memory)."""
        table = Table(f"file://{temp_dir}")
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
        ])
        
        # Create 10 large segments (100K rows each = 1M total)
        num_segments = 10
        rows_per_segment = 100_000
        
        for seg in range(num_segments):
            start_id = seg * rows_per_segment
            batch = pa.RecordBatch.from_arrays([
                pa.array(range(start_id, start_id + rows_per_segment), type=pa.int64()),
                pa.array(np.random.random(rows_per_segment), type=pa.float64()),
            ], schema=schema)
            table.write([batch])
            table.commit()
        
        # Benchmark compaction
        mem_start = get_memory_usage_mb()
        start = time.time()
        table.compact()
        elapsed = time.time() - start
        mem_end = get_memory_usage_mb()
        
        total_rows = num_segments * rows_per_segment
        throughput = total_rows / elapsed
        mem_delta = mem_end - mem_start
        
        print(f"\n[OUT-OF-MEMORY] Compaction speed: {throughput:,.0f} rows/sec")
        print(f"Compacted {num_segments} segments ({total_rows:,} rows) in {elapsed:.2f}s")
        print(f"Memory delta: {mem_delta:.1f} MB")
        
        # Should still be reasonable even with disk I/O
        assert throughput > 20_000, f"Expected >20K rows/sec, got {throughput:,.0f}"


class TestVectorSearchLatency:
    """Test vector search performance for in-memory and out-of-memory datasets."""
    
    def test_vector_search_latency_in_memory(self, temp_dir):
        """Test vector search latency with small in-memory dataset."""
        table = Table(f"file://{temp_dir}")
        
        # Small dataset: 10K vectors, 128D
        num_vectors = 10_000
        dim = 128
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=dim)),
        ])
        
        # Enable vector indexing
        table.index_columns = ["embedding"]
        
        embeddings = [np.random.random(dim).astype(np.float32).tolist() for _ in range(num_vectors)]
        
        batch = pa.RecordBatch.from_arrays([
            pa.array(range(num_vectors), type=pa.int64()),
            pa.array(embeddings),
        ], schema=schema)
        
        table.write([batch])
        table.commit()
        table.wait_for_indexes()
        
        # Benchmark vector search
        query_vector = np.random.random(dim).astype(np.float32).tolist()
        
        # Warm up
        _ = table.search(column="embedding", query=query_vector, k=10)
        
        latencies = []
        for _ in range(10):
            start = time.time()
            results = table.search(column="embedding", query=query_vector, k=10)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            assert len(results) > 0
        
        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)
        
        print(f"\n[IN-MEMORY] Vector search latency: p50={p50:.1f}ms, p99={p99:.1f}ms")
        print(f"Dataset: {num_vectors:,} vectors, {dim}D")
        
        # Should be very fast for small in-memory dataset
        assert p99 < 50, f"Expected p99 < 50ms, got {p99:.1f}ms"
    
    def test_vector_search_latency_out_of_memory(self, temp_dir):
        """Test vector search latency with large out-of-memory dataset."""
        table = Table(f"file://{temp_dir}")
        
        # Large dataset: 100K vectors, 384D (similar to OpenAI embeddings)
        num_vectors = 100_000
        dim = 384
        batch_size = 10_000
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=dim)),
        ])
        
        # Enable vector indexing
        table.index_columns = ["embedding"]
        
        # Write in batches to avoid memory issues
        for i in range(0, num_vectors, batch_size):
            embeddings = [
                np.random.random(dim).astype(np.float32).tolist()
                for _ in range(batch_size)
            ]
            
            batch = pa.RecordBatch.from_arrays([
                pa.array(range(i, i + batch_size), type=pa.int64()),
                pa.array(embeddings),
            ], schema=schema)
            
            table.write([batch])
        
        table.commit()
        table.wait_for_indexes()
        
        # Benchmark vector search
        query_vector = np.random.random(dim).astype(np.float32).tolist()
        
        latencies = []
        for _ in range(5):  # Fewer iterations for large dataset
            start = time.time()
            results = table.search(column="embedding", query=query_vector, k=10)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            assert len(results) > 0
        
        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)
        
        print(f"\n[OUT-OF-MEMORY] Vector search latency: p50={p50:.1f}ms, p99={p99:.1f}ms")
        print(f"Dataset: {num_vectors:,} vectors, {dim}D")
        
        # Should still be reasonable even with disk-based index
        # Increased threshold to account for first-load overhead and CI variability
        assert p99 < 1500, f"Expected p99 < 1500ms, got {p99:.1f}ms"


class TestMemoryUsage:
    """Test memory usage during write and read operations."""
    
    def test_memory_usage_write_in_memory(self, temp_dir):
        """Test memory usage during writes with in-memory dataset."""
        table = Table(f"file://{temp_dir}")
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("data", pa.string()),
        ])
        
        mem_start = get_memory_usage_mb()
        
        # Write 50K rows with small strings
        num_rows = 50_000
        batch = pa.RecordBatch.from_arrays([
            pa.array(range(num_rows), type=pa.int64()),
            pa.array([f"data_{i}" for i in range(num_rows)]),
        ], schema=schema)
        
        table.write([batch])
        table.commit()
        
        mem_end = get_memory_usage_mb()
        mem_delta = mem_end - mem_start
        mem_per_row = (mem_delta * 1024 * 1024) / num_rows  # bytes per row
        
        print(f"\n[IN-MEMORY] Write memory usage: {mem_delta:.1f} MB for {num_rows:,} rows")
        print(f"Memory per row: {mem_per_row:.1f} bytes")
        
        # Should be efficient for in-memory writes
        assert mem_per_row < 1000, f"Expected < 1KB per row, got {mem_per_row:.1f} bytes"
    
    def test_memory_usage_read_out_of_memory(self, temp_dir):
        """Test memory usage during reads with out-of-memory dataset."""
        table = Table(f"file://{temp_dir}")
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("data", pa.string()),
        ])
        
        # Write large dataset
        num_rows = 500_000
        batch_size = 50_000
        
        for i in range(0, num_rows, batch_size):
            batch = pa.RecordBatch.from_arrays([
                pa.array(range(i, i + batch_size), type=pa.int64()),
                pa.array([f"data_{j}" for j in range(i, i + batch_size)]),
            ], schema=schema)
            table.write([batch])
        
        table.commit()
        
        # Measure memory during read
        mem_start = get_memory_usage_mb()
        
        result = table.read()
        total_rows = len(result)
        
        mem_end = get_memory_usage_mb()
        mem_delta = mem_end - mem_start
        
        print(f"\n[OUT-OF-MEMORY] Read memory usage: {mem_delta:.1f} MB for {total_rows:,} rows")
        if mem_delta > 0:
            print(f"Memory efficiency: {total_rows / mem_delta:,.0f} rows/MB")
        else:
            print(f"Memory efficiency: N/A (mem_delta=0)")
        
        assert total_rows == num_rows
        # Should not load entire dataset into memory at once
        assert mem_delta < 500, f"Expected < 500MB memory usage, got {mem_delta:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
