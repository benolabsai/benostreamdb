-- HyperStreamDB pgvector SQL Examples
-- This file contains practical examples of pgvector-compatible SQL queries

-- ============================================================================
-- 1. BASIC DISTANCE OPERATORS
-- ============================================================================

-- L2 distance (Euclidean) - Most common for general similarity
SELECT id, content, 
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS l2_distance
FROM documents
ORDER BY l2_distance
LIMIT 10;

-- Cosine distance - Best for text embeddings
SELECT id, content,
       embedding <=> '[0.1, 0.2, 0.3]'::vector AS cosine_distance
FROM documents
ORDER BY cosine_distance
LIMIT 10;

-- Inner product - Used in recommendation systems
SELECT id, product_name,
       embedding <#> '[0.1, 0.2, 0.3]'::vector AS ip_distance
FROM products
ORDER BY ip_distance
LIMIT 10;

-- L1 distance (Manhattan) - Good for sparse data
SELECT id, embedding <+> '[0.1, 0.2, 0.3]'::vector AS l1_distance
FROM documents
ORDER BY l1_distance
LIMIT 10;

-- Hamming distance - For binary vectors
SELECT id, binary_embedding <~> B'10110101' AS hamming_distance
FROM documents
ORDER BY hamming_distance
LIMIT 10;

-- Jaccard distance - For set similarity
SELECT id, embedding <%> '[0.1, 0.2, 0.3]'::vector AS jaccard_distance
FROM documents
ORDER BY jaccard_distance
LIMIT 10;

-- ============================================================================
-- 2. KNN QUERIES WITH FILTERS
-- ============================================================================

-- Basic hybrid search: vector similarity + scalar filter
SELECT id, title, content,
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
WHERE category = 'science'
ORDER BY distance
LIMIT 10;

-- Multiple filters
SELECT id, title, 
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
WHERE category IN ('science', 'technology')
  AND published_date > '2024-01-01'
  AND author_verified = true
ORDER BY distance
LIMIT 10;

-- Range query with distance threshold
SELECT id, title,
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
WHERE embedding <-> '[0.1, 0.2, 0.3]'::vector < 0.5
  AND category = 'science'
ORDER BY distance;

-- ============================================================================
-- 3. PAGINATION
-- ============================================================================

-- First page (results 1-10)
SELECT id, title,
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Second page (results 11-20)
SELECT id, title,
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10 OFFSET 10;

-- Third page (results 21-30)
SELECT id, title,
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10 OFFSET 20;

-- ============================================================================
-- 4. VECTOR AGGREGATIONS
-- ============================================================================

-- Compute centroid of all vectors
SELECT vector_avg(embedding) AS centroid
FROM documents;

-- Compute centroid per category
SELECT category, 
       vector_avg(embedding) AS centroid,
       COUNT(*) AS doc_count
FROM documents
GROUP BY category;

-- Sum vectors by group
SELECT category,
       vector_sum(embedding) AS total_embedding
FROM documents
GROUP BY category
HAVING COUNT(*) > 10;

-- Find documents closest to category centroid
WITH category_centroids AS (
    SELECT category, 
           vector_avg(embedding) AS centroid
    FROM documents
    GROUP BY category
)
SELECT d.id, d.title, d.category,
       d.embedding <-> c.centroid AS distance_to_centroid
FROM documents d
JOIN category_centroids c ON d.category = c.category
ORDER BY distance_to_centroid
LIMIT 10;

-- ============================================================================
-- 5. SPARSE VECTORS
-- ============================================================================

-- Query with sparse vector literal
SELECT id, content,
       sparse_embedding <-> '{1:0.5, 10:0.3, 100:0.8}'::sparsevec(10000) AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Get sparse vector statistics
SELECT id,
       sparsevec_dims(sparse_embedding) AS dimensions,
       sparsevec_nnz(sparse_embedding) AS non_zero_count,
       CAST(sparsevec_nnz(sparse_embedding) AS FLOAT) / sparsevec_dims(sparse_embedding) AS density
FROM documents
LIMIT 10;

-- Filter by sparsity
SELECT id, content
FROM documents
WHERE CAST(sparsevec_nnz(sparse_embedding) AS FLOAT) / sparsevec_dims(sparse_embedding) < 0.1
ORDER BY id
LIMIT 10;

-- ============================================================================
-- 6. BINARY VECTORS
-- ============================================================================

-- Query with binary vector
SELECT id, content,
       binary_embedding <~> B'10110101' AS hamming_distance
FROM documents
ORDER BY hamming_distance
LIMIT 10;

-- Binary quantization
SELECT id,
       binary_quantize(embedding) AS binary_vec
FROM documents
LIMIT 10;

-- Hexadecimal binary literal
SELECT id,
       binary_embedding <~> '\\xB5'::bit(8) AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- ============================================================================
-- 7. TYPE CASTING
-- ============================================================================

-- Dense to sparse conversion
SELECT id,
       embedding::sparsevec AS sparse_vec
FROM documents
WHERE id = 1;

-- Sparse to dense conversion
SELECT id,
       sparse_embedding::vector AS dense_vec
FROM documents
WHERE id = 1;

-- Vector to binary quantization
SELECT id,
       embedding::bit AS binary_vec
FROM documents
WHERE id = 1;

-- Float32 to Float16 precision reduction
SELECT id,
       embedding::halfvec AS half_precision_vec
FROM documents
WHERE id = 1;

-- ============================================================================
-- 8. ADVANCED QUERIES
-- ============================================================================

-- Semantic search with re-ranking
WITH initial_results AS (
    SELECT id, title, content, embedding,
           embedding <=> '[0.1, 0.2, 0.3]'::vector AS cosine_dist
    FROM documents
    WHERE category = 'science'
    ORDER BY cosine_dist
    LIMIT 100
)
SELECT id, title, content,
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS l2_dist
FROM initial_results
ORDER BY l2_dist
LIMIT 10;

-- Multi-vector search (find best match across multiple embeddings)
SELECT id, title,
       LEAST(
           text_embedding <=> '[0.1, 0.2, ...]'::vector,
           image_embedding <=> '[0.3, 0.4, ...]'::vector,
           audio_embedding <=> '[0.5, 0.6, ...]'::vector
       ) AS min_distance
FROM multimedia_documents
ORDER BY min_distance
LIMIT 10;

-- Temporal decay in similarity search
SELECT id, title,
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance,
       EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400 AS days_old,
       (embedding <-> '[0.1, 0.2, 0.3]'::vector) * 
           (1 + 0.01 * EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400) AS time_weighted_distance
FROM documents
ORDER BY time_weighted_distance
LIMIT 10;

-- Clustering: Find documents assigned to wrong cluster
WITH cluster_centroids AS (
    SELECT cluster_id,
           vector_avg(embedding) AS centroid
    FROM documents
    GROUP BY cluster_id
),
document_distances AS (
    SELECT d.id,
           d.cluster_id AS assigned_cluster,
           c.cluster_id AS nearest_cluster,
           d.embedding <-> c.centroid AS distance,
           ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.embedding <-> c.centroid) AS rank
    FROM documents d
    CROSS JOIN cluster_centroids c
)
SELECT id, assigned_cluster, nearest_cluster, distance
FROM document_distances
WHERE rank = 1 AND assigned_cluster != nearest_cluster
ORDER BY distance DESC
LIMIT 100;

-- Deduplication: Find near-duplicate documents
SELECT d1.id AS doc1_id,
       d2.id AS doc2_id,
       d1.title AS doc1_title,
       d2.title AS doc2_title,
       d1.embedding <-> d2.embedding AS distance
FROM documents d1
JOIN documents d2 ON d1.id < d2.id
WHERE d1.embedding <-> d2.embedding < 0.1
ORDER BY distance
LIMIT 100;

-- Recommendation: Find similar items based on user history
WITH user_profile AS (
    SELECT vector_avg(i.embedding) AS preference_vector
    FROM user_interactions ui
    JOIN items i ON ui.item_id = i.id
    WHERE ui.user_id = 123
      AND ui.rating >= 4
      AND ui.interaction_date > NOW() - INTERVAL '30 days'
)
SELECT i.id,
       i.name,
       i.category,
       i.embedding <#> up.preference_vector AS score
FROM items i
CROSS JOIN user_profile up
WHERE i.id NOT IN (
    SELECT item_id FROM user_interactions WHERE user_id = 123
)
ORDER BY score DESC
LIMIT 20;

-- ============================================================================
-- 9. CONFIGURATION EXAMPLES
-- ============================================================================

-- Set HNSW search parameters for higher accuracy
SET hnsw.ef_search = 128;

-- Set IVF probes for better recall
SET ivf.probes = 20;

-- Query with custom configuration
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Reset to defaults
SET hnsw.ef_search = 64;
SET ivf.probes = 10;

-- Disable index for testing (forces sequential scan)
SET vector.use_index = false;

-- Re-enable index
SET vector.use_index = true;

-- ============================================================================
-- 10. JOINS WITH VECTOR SEARCH
-- ============================================================================

-- Join documents with their similar documents
SELECT d1.id AS doc_id,
       d1.title AS doc_title,
       d2.id AS similar_doc_id,
       d2.title AS similar_doc_title,
       d1.embedding <-> d2.embedding AS distance
FROM documents d1
CROSS JOIN LATERAL (
    SELECT id, title, embedding
    FROM documents d2
    WHERE d2.id != d1.id
    ORDER BY d2.embedding <-> d1.embedding
    LIMIT 5
) d2
WHERE d1.id IN (1, 2, 3);

-- Join users with recommended items
SELECT u.id AS user_id,
       u.name AS user_name,
       i.id AS item_id,
       i.name AS item_name,
       u.preference_embedding <#> i.embedding AS score
FROM users u
CROSS JOIN LATERAL (
    SELECT id, name, embedding
    FROM items i
    WHERE i.category = u.preferred_category
    ORDER BY i.embedding <#> u.preference_embedding
    LIMIT 10
) i
WHERE u.id = 123;

-- ============================================================================
-- 11. ERROR HANDLING EXAMPLES
-- ============================================================================

-- These queries will produce errors (for testing error messages)

-- Dimension mismatch
-- SELECT embedding <-> '[0.1, 0.2]'::vector FROM documents;
-- Error: Vector dimension mismatch: expected 768, got 2

-- Malformed literal
-- SELECT '[0.1, 0.2'::vector;
-- Error: Vector literal must be enclosed in brackets

-- Invalid number
-- SELECT '[0.1, abc]'::vector;
-- Error: Invalid number at position 5

-- Invalid cast
-- SELECT 'not a vector'::vector;
-- Error: Cannot cast string to vector without proper format

-- Aggregation dimension mismatch (if table has mixed dimensions)
-- SELECT vector_avg(embedding) FROM mixed_dimension_table;
-- Error: Cannot aggregate vectors of different dimensions

-- ============================================================================
-- 12. PERFORMANCE TESTING QUERIES
-- ============================================================================

-- Measure query latency (run multiple times)
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Compare index vs sequential scan
SET vector.use_index = false;
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

SET vector.use_index = true;
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Test different ef_search values
SET hnsw.ef_search = 32;
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents ORDER BY distance LIMIT 10;

SET hnsw.ef_search = 64;
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents ORDER BY distance LIMIT 10;

SET hnsw.ef_search = 128;
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents ORDER BY distance LIMIT 10;
