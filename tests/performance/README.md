# Performance Benchmarks

This directory contains comprehensive performance benchmarks for HyperStreamDB.

## Test Structure

All benchmarks test **both** scenarios:
- **In-Memory**: Data fits entirely in RAM (fast path, optimal performance)
- **Out-of-Memory**: Data exceeds available memory, requires disk spill (realistic production)

## Benchmark Categories

### 1. Ingest Throughput
- **In-Memory**: 100K rows (~2.4MB) - Target: >50K rows/sec
- **Out-of-Memory**: 5M rows (~120MB+) - Target: >30K rows/sec

### 2. Query Latency
- **Indexed (In-Memory)**: 100K rows - Target: p99 < 100ms
- **Full Scan (Out-of-Memory)**: 1M rows - Target: < 5000ms

### 3. Compaction Speed
- **In-Memory**: 20 segments, 100K total rows - Target: >50K rows/sec
- **Out-of-Memory**: 10 segments, 1M total rows - Target: >20K rows/sec

### 4. Vector Search Latency
- **In-Memory**: 10K vectors, 128D - Target: p99 < 50ms
- **Out-of-Memory**: 100K vectors, 384D - Target: p99 < 500ms

### 5. Memory Usage
- **Write (In-Memory)**: 50K rows - Target: < 1KB per row
- **Read (Out-of-Memory)**: 500K rows - Target: < 500MB total

## Running Benchmarks

```bash
# Run all benchmarks
pytest tests/performance/test_benchmarks.py -v -s

# Run specific category
pytest tests/performance/test_benchmarks.py::TestIngestThroughput -v -s

# Run only in-memory tests
pytest tests/performance/test_benchmarks.py -k "in_memory" -v -s

# Run only out-of-memory tests
pytest tests/performance/test_benchmarks.py -k "out_of_memory" -v -s
```

## Performance Targets

| Metric | In-Memory | Out-of-Memory |
|--------|-----------|---------------|
| Ingest | >50K rows/sec | >30K rows/sec |
| Indexed Query (p99) | <100ms | N/A |
| Full Scan | N/A | <5000ms |
| Compaction | >50K rows/sec | >20K rows/sec |
| Vector Search (p99) | <50ms | <500ms |
| Write Memory | <1KB/row | N/A |
| Read Memory | N/A | <500MB |

## Dependencies

```bash
pip install pytest psutil numpy pyarrow
```

## Notes

- **In-Memory tests** validate the fast path and optimal performance
- **Out-of-Memory tests** validate production scenarios with realistic data sizes
- Memory usage is tracked using `psutil` to ensure efficient resource utilization
- All benchmarks include warm-up runs to ensure fair measurements
