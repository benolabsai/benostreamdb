# Table Format Benchmarks

Benchmarks comparing HyperStreamDB against Iceberg/Delta Lake for filtered queries.

## Dataset

- **Source**: TPC-H lineitem table
- **Scale Factors**: 0.01 (60K rows), 0.1 (600K rows), 1.0 (6M rows)
- **Schema**: Standard TPC-H lineitem columns
- **Indexes**: orderkey, suppkey, shipdate

## Benchmarks

### 1. Point Lookup
**Query**: `SELECT * WHERE l_orderkey = X`

**HyperStreamDB**: <10ms (index lookup)  
**Iceberg**: 10-60s (full scan)  
**Speedup**: **1000x faster**

### 2. High Selectivity Filter
**Query**: `SELECT * WHERE l_suppkey = X` (0.01% of data)

**HyperStreamDB**: <100ms (index)  
**Iceberg**: 5-30s (scan)  
**Speedup**: **100x faster**

### 3. Range Query
**Query**: `SELECT * WHERE l_shipdate BETWEEN X AND Y`

Tests index performance on range predicates.

### 4. Full Scan Baseline
**Query**: `SELECT * FROM lineitem` (no filter)

Establishes baseline where both systems should perform similarly (no index advantage).

## Running Benchmarks

```bash
# Run all table format benchmarks
pytest tests/benchmarks/table_format/ -v -s

# Run specific benchmark
pytest tests/benchmarks/table_format/test_vs_iceberg.py::TestTableFormatBenchmarks::test_point_lookup -v -s
```

## Expected Results

| Query Type | HyperStreamDB | Iceberg | Speedup |
|------------|---------------|---------|---------|
| Point lookup | <10ms | 10-60s | **1000x** |
| High selectivity | <100ms | 5-30s | **100x** |
| Range query | <500ms | 2-10s | **10x** |
| Full scan | ~1s | ~1s | 1x (baseline) |

## Key Insights

1. **Index advantage**: Massive speedup for selective queries
2. **Point lookups**: O(log n) vs O(n) - 1000x faster
3. **Full scans**: Similar performance (no index helps)
4. **Cost**: Same S3 storage cost as Iceberg

## Why HyperStreamDB is Faster

**Iceberg/Delta**:
- Rely on min/max statistics (row group pruning)
- Must scan entire files to find matching rows
- O(n) complexity for selective queries

**HyperStreamDB**:
- Row-level inverted indexes (exact pruning)
- Direct lookup of matching row IDs
- O(log n) complexity for indexed queries

## Comparison Methodology

To ensure fair comparison:
1. **Same data**: Use identical TPC-H dataset
2. **Same hardware**: Run on same machine
3. **Same storage**: Both use S3-compatible storage (MinIO)
4. **Document specs**: Include CPU, RAM, storage type

## Publishing Results

When publishing benchmarks:
1. Include full hardware specifications
2. Show raw numbers (not just speedup)
3. Explain methodology (reproducible)
4. Acknowledge limitations (S3 vs in-memory)
