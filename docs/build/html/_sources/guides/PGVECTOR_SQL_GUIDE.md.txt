# pgvector SQL Support Guide

**HyperStreamDB pgvector-Compatible SQL Interface**

HyperStreamDB provides full pgvector-compatible SQL syntax through Apache DataFusion integration, allowing you to use familiar PostgreSQL vector operations for similarity search, aggregations, and type conversions.

## Table of Contents

1. [Distance Operators](#distance-operators)
2. [Vector Literals](#vector-literals)
3. [KNN Queries](#knn-queries)
4. [Configuration Parameters](#configuration-parameters)
5. [Sparse Vectors](#sparse-vectors)
6. [Binary Vectors](#binary-vectors)
7. [Vector Aggregations](#vector-aggregations)
8. [Type Casting](#type-casting)
9. [Query Examples](#query-examples)

---

## Distance Operators

HyperStreamDB supports all six pgvector distance operators for computing vector similarity:

| Operator | Distance Metric | Description | Use Case |
|----------|----------------|-------------|----------|
| `<->` | L2 (Euclidean) | Straight-line distance | General similarity |
| `<=>` | Cosine | Angle-based similarity | Text embeddings |
| `<#>` | Inner Product | Negative dot product | Recommendation systems |
| `<+>` | L1 (Manhattan) | Sum of absolute differences | Sparse data |
| `<~>` | Hamming | Bit difference count | Binary vectors |
| `<%>` | Jaccard | Set similarity | Categorical data |

### Basic Usage

```sql
-- L2 distance (Euclidean)
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
WHERE embedding <-> '[0.1, 0.2, 0.3]'::vector < 0.5;

-- Cosine distance
SELECT id, embedding <=> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance LIMIT 10;

-- Inner product
SELECT id, embedding <#> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents;
```

### Operator Equivalence

Distance operators are equivalent to their corresponding UDF functions:

```sql
-- These are equivalent:
SELECT embedding <-> query_vec FROM table;
SELECT dist_l2(embedding, query_vec) FROM table;

-- These are equivalent:
SELECT embedding <=> query_vec FROM table;
SELECT dist_cosine(embedding, query_vec) FROM table;
```

---

## Vector Literals

Vector literals allow you to specify vectors directly in SQL queries without parameter binding.

### Dense Vectors

```sql
-- Basic syntax
'[1, 2, 3]'::vector

-- With floating-point values
'[0.1, 0.2, 0.3]'::vector

-- With dimension specification
'[1, 2, 3]'::vector(3)

-- In queries
SELECT * FROM documents
WHERE embedding <-> '[0.1, 0.2, 0.3]'::vector < 0.5;
```

### Sparse Vectors

```sql
-- Sparse vector literal (index:value pairs)
'{1:0.5, 10:0.3, 100:0.8}'::sparsevec(1000)

-- In queries
SELECT id, sparse_embedding <-> '{1:0.5, 10:0.3}'::sparsevec(1000) AS distance
FROM documents
ORDER BY distance LIMIT 10;
```

### Binary Vectors

```sql
-- Binary literal (bit string)
B'10110101'

-- Hexadecimal format
'\\xB5'::bit(8)

-- In queries
SELECT id, binary_embedding <~> B'10110101' AS distance
FROM documents;
```

---

## KNN Queries

K-Nearest Neighbors (KNN) queries find the k most similar vectors to a query vector. HyperStreamDB automatically optimizes these queries to use vector indexes.

### Basic KNN

```sql
-- Find 10 nearest neighbors
SELECT id, content, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

### KNN with Filters

Combine vector search with scalar filters for hybrid queries:

```sql
-- Find nearest neighbors in a specific category
SELECT id, content, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
WHERE category = 'science'
ORDER BY distance
LIMIT 10;

-- Multiple filters
SELECT id, content, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
WHERE category = 'science' 
  AND published_date > '2024-01-01'
  AND author_id IN (1, 2, 3)
ORDER BY distance
LIMIT 10;
```

### KNN with Pagination

```sql
-- First page (results 1-10)
SELECT id, content, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Second page (results 11-20)
SELECT id, content, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10 OFFSET 10;
```

### Distance Threshold Queries

```sql
-- Find all vectors within distance threshold
SELECT id, content, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
WHERE embedding <-> '[0.1, 0.2, 0.3]'::vector < 0.5
ORDER BY distance;
```

---

## Configuration Parameters

Control vector search behavior with session configuration parameters.

### HNSW Parameters

```sql
-- Set ef_search (search beam width)
-- Higher values = more accurate but slower
-- Default: 64
SET hnsw.ef_search = 128;

-- Query with custom ef_search
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

### IVF Parameters

```sql
-- Set number of clusters to search
-- Higher values = more accurate but slower
-- Default: 10
SET ivf.probes = 20;

-- Query with custom probes
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

### Index Control

```sql
-- Force index usage (default: true)
SET vector.use_index = true;

-- Disable index for testing (forces sequential scan)
SET vector.use_index = false;
```

### Python API

```python
import hyperstreamdb as hdb

session = hdb.Session()

# Set configuration parameters
session.set_config("hnsw.ef_search", 128)
session.set_config("ivf.probes", 20)

# Execute query with custom config
results = session.sql("""
    SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 10
""")
```

---

## Sparse Vectors

Sparse vectors efficiently represent high-dimensional vectors with mostly zero values.

### Creating Sparse Vector Tables

```sql
CREATE TABLE documents (
    id INTEGER,
    content TEXT,
    sparse_embedding sparsevec(10000)
);
```

### Sparse Vector Operations

```sql
-- Insert sparse vectors
INSERT INTO documents (id, content, sparse_embedding)
VALUES (1, 'document text', '{1:0.5, 100:0.3, 500:0.8}'::sparsevec(10000));

-- Query sparse vectors
SELECT id, sparse_embedding <-> '{1:0.5, 10:0.3}'::sparsevec(10000) AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Get sparse vector dimensions
SELECT id, sparsevec_dims(sparse_embedding) AS dimensions
FROM documents;

-- Get non-zero count
SELECT id, sparsevec_nnz(sparse_embedding) AS non_zero_count
FROM documents;
```

### Sparse Vector Utility Functions

```sql
-- Get dimensionality
SELECT sparsevec_dims('{1:0.5, 10:0.3}'::sparsevec(1000));
-- Returns: 1000

-- Get non-zero element count
SELECT sparsevec_nnz('{1:0.5, 10:0.3}'::sparsevec(1000));
-- Returns: 2
```

---

## Binary Vectors

Binary vectors use bit-packed representation for memory-efficient storage and fast Hamming distance computation.

### Binary Quantization

```sql
-- Convert dense vector to binary
SELECT binary_quantize('[0.1, -0.2, 0.3, -0.4]'::vector) AS binary_vec;
-- Returns: B'1010' (1 if >= 0, 0 if < 0)

-- Store binary vectors
CREATE TABLE documents (
    id INTEGER,
    content TEXT,
    binary_embedding bit(1024)
);
```

### Binary Vector Operations

```sql
-- Hamming distance on binary vectors
SELECT id, binary_embedding <~> B'10110101' AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Binary vector literals
SELECT id, binary_embedding <~> '\\xB5'::bit(8) AS distance
FROM documents;
```

### Binary Vector Display

```sql
-- Format binary vector for display
SELECT id, format_binary_vector(binary_embedding, 8) AS binary_str
FROM documents;
-- Returns: "10110101" or "0xB5"
```

---

## Vector Aggregations

Compute aggregate statistics over vector columns.

### Vector Sum

```sql
-- Sum all vectors in a table
SELECT vector_sum(embedding) AS total
FROM documents;

-- Sum vectors by group
SELECT category, vector_sum(embedding) AS category_sum
FROM documents
GROUP BY category;
```

### Vector Average (Centroid)

```sql
-- Compute centroid of all vectors
SELECT vector_avg(embedding) AS centroid
FROM documents;

-- Compute centroid per category
SELECT category, vector_avg(embedding) AS centroid
FROM documents
GROUP BY category;

-- Find documents closest to category centroid
WITH centroids AS (
    SELECT category, vector_avg(embedding) AS centroid
    FROM documents
    GROUP BY category
)
SELECT d.id, d.content, d.embedding <-> c.centroid AS distance
FROM documents d
JOIN centroids c ON d.category = c.category
ORDER BY distance
LIMIT 10;
```

### Aggregation with Filters

```sql
-- Aggregate filtered results
SELECT category, vector_avg(embedding) AS centroid
FROM documents
WHERE published_date > '2024-01-01'
GROUP BY category
HAVING COUNT(*) > 10;
```

---

## Type Casting

Convert between different vector representations.

### Dense to Sparse

```sql
-- Cast dense vector to sparse (filters out zeros)
SELECT '[0.1, 0, 0.3, 0, 0.5]'::vector::sparsevec AS sparse_vec;
-- Returns: {0:0.1, 2:0.3, 4:0.5}
```

### Sparse to Dense

```sql
-- Cast sparse vector to dense (expands with zeros)
SELECT '{1:0.5, 10:0.3}'::sparsevec(100)::vector AS dense_vec;
-- Returns: [0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, ...]
```

### Vector to Binary

```sql
-- Cast vector to binary (binary quantization)
SELECT '[0.1, -0.2, 0.3, -0.4]'::vector::bit AS binary_vec;
-- Returns: B'1010'
```

### Float Precision Conversion

```sql
-- Cast Float32 to Float16 (reduces precision)
SELECT '[0.123456789, 0.987654321]'::vector::halfvec AS half_vec;

-- Cast Float16 to Float32 (expands precision)
SELECT '[0.123, 0.987]'::halfvec::vector AS full_vec;
```

### Round-Trip Conversions

```sql
-- Dense -> Sparse -> Dense (preserves values)
SELECT '[0.1, 0, 0.3]'::vector::sparsevec::vector AS round_trip;

-- Vector -> Binary -> Vector (loses precision, binary quantization)
SELECT '[0.1, -0.2, 0.3]'::vector::bit::vector AS quantized;
```

---

## Query Examples

### Example 1: Semantic Search

```sql
-- Find documents similar to a query
SELECT 
    id,
    title,
    content,
    embedding <=> '[0.1, 0.2, 0.3, ...]'::vector AS similarity
FROM documents
WHERE language = 'en'
ORDER BY similarity
LIMIT 20;
```

### Example 2: Hybrid Search with Multiple Filters

```sql
-- Find similar documents with complex filters
SELECT 
    d.id,
    d.title,
    d.embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance,
    d.published_date
FROM documents d
WHERE d.category IN ('science', 'technology')
  AND d.published_date BETWEEN '2024-01-01' AND '2024-12-31'
  AND d.author_id IN (SELECT id FROM authors WHERE verified = true)
ORDER BY distance
LIMIT 10;
```

### Example 3: Clustering Analysis

```sql
-- Find cluster centroids and assign documents
WITH cluster_centroids AS (
    SELECT 
        cluster_id,
        vector_avg(embedding) AS centroid
    FROM documents
    GROUP BY cluster_id
),
document_distances AS (
    SELECT 
        d.id,
        d.cluster_id AS assigned_cluster,
        c.cluster_id AS nearest_cluster,
        d.embedding <-> c.centroid AS distance,
        ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.embedding <-> c.centroid) AS rank
    FROM documents d
    CROSS JOIN cluster_centroids c
)
SELECT 
    id,
    assigned_cluster,
    nearest_cluster,
    distance
FROM document_distances
WHERE rank = 1 AND assigned_cluster != nearest_cluster;
```

### Example 4: Recommendation System

```sql
-- Find similar items based on user preferences
WITH user_profile AS (
    SELECT vector_avg(i.embedding) AS preference_vector
    FROM user_interactions ui
    JOIN items i ON ui.item_id = i.id
    WHERE ui.user_id = 123
      AND ui.rating >= 4
)
SELECT 
    i.id,
    i.name,
    i.embedding <#> up.preference_vector AS score
FROM items i
CROSS JOIN user_profile up
WHERE i.id NOT IN (
    SELECT item_id FROM user_interactions WHERE user_id = 123
)
ORDER BY score DESC
LIMIT 20;
```

### Example 5: Deduplication

```sql
-- Find near-duplicate documents
SELECT 
    d1.id AS doc1_id,
    d2.id AS doc2_id,
    d1.embedding <-> d2.embedding AS distance
FROM documents d1
JOIN documents d2 ON d1.id < d2.id
WHERE d1.embedding <-> d2.embedding < 0.1
ORDER BY distance;
```

### Example 6: Multi-Vector Search

```sql
-- Search across multiple vector columns
SELECT 
    id,
    title,
    LEAST(
        text_embedding <=> '[0.1, 0.2, ...]'::vector,
        image_embedding <=> '[0.3, 0.4, ...]'::vector
    ) AS min_distance
FROM multimedia_documents
ORDER BY min_distance
LIMIT 10;
```

### Example 7: Temporal Vector Search

```sql
-- Find similar documents with time decay
SELECT 
    id,
    title,
    embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance,
    EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400 AS days_old,
    (embedding <-> '[0.1, 0.2, 0.3]'::vector) * 
        (1 + 0.01 * EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400) AS time_weighted_distance
FROM documents
ORDER BY time_weighted_distance
LIMIT 10;
```

---

## Performance Tips

### Index Usage

1. **Ensure indexes exist**: Vector searches automatically use HNSW or IVF indexes when available
2. **Monitor fallback**: Check logs for "using sequential scan" messages indicating missing indexes
3. **Tune parameters**: Adjust `ef_search` and `probes` based on accuracy/speed requirements

### Query Optimization

1. **Use filters early**: Scalar filters are applied before vector search for better performance
2. **Limit results**: Always use `LIMIT` for KNN queries to enable index optimization
3. **Project only needed columns**: Avoid `SELECT *` when working with large embeddings

### Data Types

1. **Use sparse vectors**: For high-dimensional vectors with >90% zeros
2. **Use binary vectors**: For memory-constrained environments (32x compression)
3. **Use Float16**: For 2x memory savings with acceptable precision loss

---

## Error Handling

### Common Errors

**Dimension Mismatch**:
```sql
-- Error: Vector dimension mismatch: expected 768, got 512
SELECT embedding <-> '[0.1, 0.2]'::vector FROM documents;
```

**Malformed Literal**:
```sql
-- Error: Vector literal must be enclosed in brackets
SELECT '[0.1, 0.2'::vector;

-- Error: Invalid number at position 5
SELECT '[0.1, abc]'::vector;
```

**Invalid Cast**:
```sql
-- Error: Cannot cast string to vector without proper format
SELECT 'not a vector'::vector;
```

**Aggregation Dimension Mismatch**:
```sql
-- Error: Cannot aggregate vectors of different dimensions
SELECT vector_avg(embedding) FROM mixed_dimension_table;
```

### Troubleshooting

1. **Check vector dimensions**: Use `vector_dims(column)` to verify dimensions
2. **Validate data**: Ensure no NaN or infinite values in vectors
3. **Test with small data**: Verify queries on small datasets before scaling
4. **Check configuration**: Verify `ef_search` and `probes` are positive integers

---

## Compatibility Notes

### pgvector Compatibility

HyperStreamDB implements the pgvector SQL interface with the following compatibility:

✅ **Fully Compatible**:
- All six distance operators (`<->`, `<=>`, `<#>`, `<+>`, `<~>`, `<%>`)
- Vector literal syntax (`'[...]'::vector`)
- KNN query optimization (`ORDER BY distance LIMIT k`)
- Vector aggregations (`vector_sum`, `vector_avg`)
- Type casting between vector types

⚠️ **Differences**:
- Configuration uses session parameters instead of GUC variables
- Index hints use session config instead of SQL comments (future feature)
- Some advanced pgvector functions may not be implemented

### Migration from pgvector

```sql
-- PostgreSQL with pgvector
SELECT * FROM items
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'
LIMIT 10;

-- HyperStreamDB (identical syntax)
SELECT * FROM items
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;
```

---

## Additional Resources

- [HyperStreamDB Architecture](architecture.md)
- [DataFusion SQL Reference](https://arrow.apache.org/datafusion/user-guide/sql/index.html)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Vector Search Best Practices](COMPREHENSIVE_GUIDE.md)

---

**Last Updated**: 2026-02-08
