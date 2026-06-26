# Vector Search Configuration Guide

This guide covers configuration parameters for tuning vector search performance in HyperStreamDB.

## Configuration Parameters

### HNSW Parameters

#### `hnsw.ef_search`

Controls the search beam width for HNSW index queries.

- **Type**: Integer
- **Default**: 64
- **Range**: 1 - 1000
- **Effect**: Higher values increase accuracy but reduce speed

**Usage**:
```sql
-- Set for session
SET hnsw.ef_search = 128;

-- Query with custom ef_search
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Python API**:
```python
import hyperstreamdb as hdb

session = hdb.Session()
session.set_config("hnsw.ef_search", 128)

results = session.sql("""
    SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 10
""")
```

**Performance Impact**:
- `ef_search = 32`: Fast, ~90% recall
- `ef_search = 64`: Balanced, ~95% recall (default)
- `ef_search = 128`: Accurate, ~98% recall
- `ef_search = 256`: Very accurate, ~99.5% recall

### IVF Parameters

#### `ivf.probes`

Controls the number of IVF clusters to search.

- **Type**: Integer
- **Default**: 10
- **Range**: 1 - 1000
- **Effect**: Higher values increase accuracy but reduce speed

**Usage**:
```sql
-- Set for session
SET ivf.probes = 20;

-- Query with custom probes
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Python API**:
```python
session.set_config("ivf.probes", 20)
```

**Performance Impact**:
- `probes = 5`: Fast, ~85% recall
- `probes = 10`: Balanced, ~92% recall (default)
- `probes = 20`: Accurate, ~97% recall
- `probes = 50`: Very accurate, ~99% recall

### Index Control

#### `vector.use_index`

Controls whether to use vector indexes or force sequential scan.

- **Type**: Boolean
- **Default**: true
- **Effect**: When false, forces sequential scan (useful for testing)

**Usage**:
```sql
-- Force sequential scan (for testing)
SET vector.use_index = false;

-- Re-enable index usage
SET vector.use_index = true;
```

**Python API**:
```python
# Disable index for testing
session.set_config("vector.use_index", False)

# Re-enable
session.set_config("vector.use_index", True)
```

## Tuning Guidelines

### Accuracy vs Speed Tradeoff

| Use Case | ef_search | probes | Expected Recall | Relative Speed |
|----------|-----------|--------|-----------------|----------------|
| Real-time search | 32 | 5 | ~88% | 4x faster |
| Balanced | 64 | 10 | ~94% | 1x (baseline) |
| High accuracy | 128 | 20 | ~98% | 0.5x slower |
| Maximum accuracy | 256 | 50 | ~99.5% | 0.2x slower |

### Dataset Size Recommendations

**Small datasets (< 100K vectors)**:
```sql
SET hnsw.ef_search = 128;
SET ivf.probes = 20;
```

**Medium datasets (100K - 10M vectors)**:
```sql
SET hnsw.ef_search = 64;  -- Default
SET ivf.probes = 10;      -- Default
```

**Large datasets (> 10M vectors)**:
```sql
SET hnsw.ef_search = 32;
SET ivf.probes = 5;
```

### Dimensionality Recommendations

**Low dimensions (< 128)**:
```sql
SET hnsw.ef_search = 64;
SET ivf.probes = 10;
```

**Medium dimensions (128 - 768)**:
```sql
SET hnsw.ef_search = 64;
SET ivf.probes = 10;
```

**High dimensions (> 768)**:
```sql
SET hnsw.ef_search = 128;
SET ivf.probes = 20;
```

## Benchmarking

### Measuring Recall

```python
import hyperstreamdb as hdb
import numpy as np

session = hdb.Session()

# Ground truth (sequential scan)
session.set_config("vector.use_index", False)
ground_truth = session.sql("""
    SELECT id FROM documents
    ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
    LIMIT 100
""")
ground_truth_ids = set(ground_truth['id'])

# Test with index
session.set_config("vector.use_index", True)
session.set_config("hnsw.ef_search", 64)

results = session.sql("""
    SELECT id FROM documents
    ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
    LIMIT 100
""")
result_ids = set(results['id'])

# Calculate recall
recall = len(ground_truth_ids & result_ids) / len(ground_truth_ids)
print(f"Recall@100: {recall:.2%}")
```

### Measuring Latency

```python
import time

session.set_config("hnsw.ef_search", 64)

# Warmup
for _ in range(10):
    session.sql("SELECT id FROM documents ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector LIMIT 10")

# Measure
latencies = []
for _ in range(100):
    start = time.time()
    session.sql("SELECT id FROM documents ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector LIMIT 10")
    latencies.append(time.time() - start)

print(f"P50: {np.percentile(latencies, 50)*1000:.2f}ms")
print(f"P95: {np.percentile(latencies, 95)*1000:.2f}ms")
print(f"P99: {np.percentile(latencies, 99)*1000:.2f}ms")
```

## Advanced Configuration

### Per-Query Configuration (Future)

Future versions will support per-query hints:

```sql
-- Planned syntax (not yet implemented)
SELECT /*+ INDEX_HINT(ef_search=128, probes=20) */ 
    id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

Index build parameters are set during index creation using the fluent `add_index` method:

```python
import hyperstreamdb as hdb

table = hdb.Table("s3://bucket/my-table")

# TurboQuant 8-bit quantization (Recommended Default)
# 4x compression, near-lossless accuracy
table.add_index("embedding", "hnsw_tq8")

# TurboQuant 4-bit quantization
# 8x compression, maximum efficiency
table.add_index("embedding", "hnsw_tq4")

# Custom HNSW parameters
table.add_index(
    column="embedding",
    index_config={
        "type": "hnsw",
        "complexity": 16, # Max connections per node (formerly 'm')
        "quality": 200,    # Construction beam width (formerly 'ef_construction')
    }
)

# Product Quantization (PQ)
table.add_index(
    column="embedding",
    index_config={
        "type": "hnsw_pq",
        "compression": 32 # PQ subspaces (formerly 'subspaces')
    }
)
```

## Troubleshooting

### Slow Queries

**Symptom**: Queries taking longer than expected

**Solutions**:
1. Reduce `ef_search` or `probes` for faster (but less accurate) results
2. Check if index exists: `SHOW INDEXES FROM table_name`
3. Verify index is being used: Check query plan
4. Consider using binary vectors for 32x speedup

### Low Recall

**Symptom**: Missing relevant results

**Solutions**:
1. Increase `ef_search` or `probes`
2. Verify index quality: Rebuild if necessary
3. Check vector normalization (for cosine distance)
4. Ensure query vector has same preprocessing as indexed vectors

### Out of Memory

**Symptom**: OOM errors during queries

**Solutions**:
1. Reduce `ef_search` to lower memory usage
2. Use binary vectors for 32x memory reduction
3. Use sparse vectors for high-dimensional sparse data
4. Increase available memory

## Best Practices

1. **Start with defaults**: Use default parameters (ef_search=64, probes=10) initially
2. **Measure first**: Benchmark recall and latency before tuning
3. **Tune incrementally**: Adjust one parameter at a time
4. **Monitor production**: Track recall and latency metrics in production
5. **Document settings**: Record configuration choices for reproducibility

## Configuration Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hnsw.ef_search` | Integer | 64 | 1-1000 | HNSW search beam width |
| `ivf.probes` | Integer | 10 | 1-1000 | IVF clusters to search |
| `vector.use_index` | Boolean | true | true/false | Enable/disable index usage |

---

**Last Updated**: 2026-02-08
