"""
HyperStreamDB pgvector SQL Examples - Python API

This file demonstrates how to use pgvector-compatible SQL syntax
through the HyperStreamDB Python API.
"""

import hyperstreamdb as hdb
import pandas as pd
import numpy as np
from typing import List


# ============================================================================
# 1. SETUP AND BASIC QUERIES
# ============================================================================

def setup_session():
    """Create a session and register a table."""
    session = hdb.Session()
    
    # Register a table
    table = hdb.Table("s3://bucket/documents")
    session.register("documents", table)
    
    return session


def basic_distance_queries(session: hdb.Session):
    """Examples of basic distance operator queries."""
    
    # L2 distance (Euclidean)
    results = session.sql("""
        SELECT id, content, 
               embedding <-> '[0.1, 0.2, 0.3]'::vector AS l2_distance
        FROM documents
        ORDER BY l2_distance
        LIMIT 10
    """)
    print("L2 Distance Results:")
    print(results.to_pandas())
    
    # Cosine distance
    results = session.sql("""
        SELECT id, content,
               embedding <=> '[0.1, 0.2, 0.3]'::vector AS cosine_distance
        FROM documents
        ORDER BY cosine_distance
        LIMIT 10
    """)
    print("\nCosine Distance Results:")
    print(results.to_pandas())
    
    # Inner product
    results = session.sql("""
        SELECT id, content,
               embedding <#> '[0.1, 0.2, 0.3]'::vector AS ip_distance
        FROM documents
        ORDER BY ip_distance
        LIMIT 10
    """)
    print("\nInner Product Results:")
    print(results.to_pandas())


# ============================================================================
# 2. HYBRID SEARCH (VECTOR + FILTERS)
# ============================================================================

def hybrid_search(session: hdb.Session, query_vector: List[float], 
                  category: str, k: int = 10):
    """Perform hybrid search with vector similarity and scalar filters."""
    
    # Convert query vector to SQL literal
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    results = session.sql(f"""
        SELECT id, title, content, category,
               embedding <-> {vec_literal} AS distance
        FROM documents
        WHERE category = '{category}'
        ORDER BY distance
        LIMIT {k}
    """)
    
    return results.to_pandas()


def multi_filter_search(session: hdb.Session, query_vector: List[float]):
    """Search with multiple filters."""
    
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    results = session.sql(f"""
        SELECT id, title, content,
               embedding <-> {vec_literal} AS distance
        FROM documents
        WHERE category IN ('science', 'technology')
          AND published_date > '2024-01-01'
          AND author_verified = true
        ORDER BY distance
        LIMIT 10
    """)
    
    return results.to_pandas()


def distance_threshold_search(session: hdb.Session, query_vector: List[float],
                               threshold: float = 0.5):
    """Find all documents within a distance threshold."""
    
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    results = session.sql(f"""
        SELECT id, title, content,
               embedding <-> {vec_literal} AS distance
        FROM documents
        WHERE embedding <-> {vec_literal} < {threshold}
        ORDER BY distance
    """)
    
    return results.to_pandas()


# ============================================================================
# 3. PAGINATION
# ============================================================================

def paginated_search(session: hdb.Session, query_vector: List[float],
                     page: int = 1, page_size: int = 10):
    """Perform paginated vector search."""
    
    offset = (page - 1) * page_size
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    results = session.sql(f"""
        SELECT id, title, content,
               embedding <-> {vec_literal} AS distance
        FROM documents
        ORDER BY distance
        LIMIT {page_size} OFFSET {offset}
    """)
    
    return results.to_pandas()


# ============================================================================
# 4. VECTOR AGGREGATIONS
# ============================================================================

def compute_centroids(session: hdb.Session):
    """Compute centroids for each category."""
    
    results = session.sql("""
        SELECT category,
               vector_avg(embedding) AS centroid,
               COUNT(*) AS doc_count
        FROM documents
        GROUP BY category
    """)
    
    return results.to_pandas()


def find_closest_to_centroid(session: hdb.Session, category: str, k: int = 10):
    """Find documents closest to their category centroid."""
    
    results = session.sql(f"""
        WITH category_centroid AS (
            SELECT vector_avg(embedding) AS centroid
            FROM documents
            WHERE category = '{category}'
        )
        SELECT d.id, d.title, d.content,
               d.embedding <-> c.centroid AS distance_to_centroid
        FROM documents d
        CROSS JOIN category_centroid c
        WHERE d.category = '{category}'
        ORDER BY distance_to_centroid
        LIMIT {k}
    """)
    
    return results.to_pandas()


# ============================================================================
# 5. CONFIGURATION MANAGEMENT
# ============================================================================

def configure_search_parameters(session: hdb.Session, 
                                ef_search: int = 64,
                                probes: int = 10):
    """Set vector search configuration parameters."""
    
    session.set_config("hnsw.ef_search", ef_search)
    session.set_config("ivf.probes", probes)
    
    print(f"Configuration set: ef_search={ef_search}, probes={probes}")


def benchmark_configurations(session: hdb.Session, query_vector: List[float]):
    """Benchmark different configuration settings."""
    
    import time
    
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    configs = [
        (32, 5, "Fast"),
        (64, 10, "Balanced"),
        (128, 20, "Accurate"),
    ]
    
    results = []
    
    for ef_search, probes, label in configs:
        session.set_config("hnsw.ef_search", ef_search)
        session.set_config("ivf.probes", probes)
        
        # Warmup
        for _ in range(5):
            session.sql(f"""
                SELECT id FROM documents
                ORDER BY embedding <-> {vec_literal}
                LIMIT 10
            """)
        
        # Measure
        latencies = []
        for _ in range(50):
            start = time.time()
            session.sql(f"""
                SELECT id FROM documents
                ORDER BY embedding <-> {vec_literal}
                LIMIT 10
            """)
            latencies.append((time.time() - start) * 1000)
        
        results.append({
            'config': label,
            'ef_search': ef_search,
            'probes': probes,
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
        })
    
    return pd.DataFrame(results)


# ============================================================================
# 6. SPARSE VECTORS
# ============================================================================

def sparse_vector_search(session: hdb.Session, sparse_indices: List[int],
                         sparse_values: List[float], dimension: int):
    """Search using sparse vectors."""
    
    # Build sparse vector literal
    pairs = [f"{idx}:{val}" for idx, val in zip(sparse_indices, sparse_values)]
    sparse_literal = f"'{{{','.join(pairs)}}}'::sparsevec({dimension})"
    
    results = session.sql(f"""
        SELECT id, content,
               sparse_embedding <-> {sparse_literal} AS distance
        FROM documents
        ORDER BY distance
        LIMIT 10
    """)
    
    return results.to_pandas()


def analyze_sparsity(session: hdb.Session):
    """Analyze sparsity of sparse vectors."""
    
    results = session.sql("""
        SELECT id,
               sparsevec_dims(sparse_embedding) AS dimensions,
               sparsevec_nnz(sparse_embedding) AS non_zero_count,
               CAST(sparsevec_nnz(sparse_embedding) AS FLOAT) / 
                   sparsevec_dims(sparse_embedding) AS density
        FROM documents
        ORDER BY density
        LIMIT 100
    """)
    
    return results.to_pandas()


# ============================================================================
# 7. BINARY VECTORS
# ============================================================================

def binary_vector_search(session: hdb.Session, binary_string: str):
    """Search using binary vectors."""
    
    results = session.sql(f"""
        SELECT id, content,
               binary_embedding <~> B'{binary_string}' AS hamming_distance
        FROM documents
        ORDER BY hamming_distance
        LIMIT 10
    """)
    
    return results.to_pandas()


def quantize_vectors(session: hdb.Session):
    """Convert dense vectors to binary quantized vectors."""
    
    results = session.sql("""
        SELECT id,
               binary_quantize(embedding) AS binary_vec
        FROM documents
        LIMIT 10
    """)
    
    return results.to_pandas()


# ============================================================================
# 8. ADVANCED QUERIES
# ============================================================================

def semantic_search_with_reranking(session: hdb.Session, 
                                   query_vector: List[float],
                                   category: str):
    """Perform semantic search with two-stage ranking."""
    
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    # First stage: cosine similarity
    # Second stage: L2 distance on top results
    results = session.sql(f"""
        WITH initial_results AS (
            SELECT id, title, content, embedding,
                   embedding <=> {vec_literal} AS cosine_dist
            FROM documents
            WHERE category = '{category}'
            ORDER BY cosine_dist
            LIMIT 100
        )
        SELECT id, title, content,
               embedding <-> {vec_literal} AS l2_dist
        FROM initial_results
        ORDER BY l2_dist
        LIMIT 10
    """)
    
    return results.to_pandas()


def find_duplicates(session: hdb.Session, threshold: float = 0.1):
    """Find near-duplicate documents."""
    
    results = session.sql(f"""
        SELECT d1.id AS doc1_id,
               d2.id AS doc2_id,
               d1.title AS doc1_title,
               d2.title AS doc2_title,
               d1.embedding <-> d2.embedding AS distance
        FROM documents d1
        JOIN documents d2 ON d1.id < d2.id
        WHERE d1.embedding <-> d2.embedding < {threshold}
        ORDER BY distance
        LIMIT 100
    """)
    
    return results.to_pandas()


def recommend_items(session: hdb.Session, user_id: int, k: int = 20):
    """Generate recommendations based on user history."""
    
    results = session.sql(f"""
        WITH user_profile AS (
            SELECT vector_avg(i.embedding) AS preference_vector
            FROM user_interactions ui
            JOIN items i ON ui.item_id = i.id
            WHERE ui.user_id = {user_id}
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
            SELECT item_id FROM user_interactions WHERE user_id = {user_id}
        )
        ORDER BY score DESC
        LIMIT {k}
    """)
    
    return results.to_pandas()


def temporal_decay_search(session: hdb.Session, query_vector: List[float],
                          decay_rate: float = 0.01):
    """Search with temporal decay factor."""
    
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    results = session.sql(f"""
        SELECT id, title,
               embedding <-> {vec_literal} AS distance,
               EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400 AS days_old,
               (embedding <-> {vec_literal}) * 
                   (1 + {decay_rate} * EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400) 
                   AS time_weighted_distance
        FROM documents
        ORDER BY time_weighted_distance
        LIMIT 10
    """)
    
    return results.to_pandas()


# ============================================================================
# 9. PERFORMANCE TESTING
# ============================================================================

def measure_recall(session: hdb.Session, query_vector: List[float], k: int = 100):
    """Measure recall by comparing index search to ground truth."""
    
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    # Ground truth (sequential scan)
    session.set_config("vector.use_index", False)
    ground_truth = session.sql(f"""
        SELECT id FROM documents
        ORDER BY embedding <-> {vec_literal}
        LIMIT {k}
    """)
    ground_truth_ids = set(ground_truth.to_pandas()['id'])
    
    # Index search
    session.set_config("vector.use_index", True)
    results = session.sql(f"""
        SELECT id FROM documents
        ORDER BY embedding <-> {vec_literal}
        LIMIT {k}
    """)
    result_ids = set(results.to_pandas()['id'])
    
    # Calculate recall
    recall = len(ground_truth_ids & result_ids) / len(ground_truth_ids)
    
    return recall


def measure_latency(session: hdb.Session, query_vector: List[float],
                    num_queries: int = 100):
    """Measure query latency statistics."""
    
    import time
    
    vec_literal = f"'[{','.join(map(str, query_vector))}]'::vector"
    
    # Warmup
    for _ in range(10):
        session.sql(f"""
            SELECT id FROM documents
            ORDER BY embedding <-> {vec_literal}
            LIMIT 10
        """)
    
    # Measure
    latencies = []
    for _ in range(num_queries):
        start = time.time()
        session.sql(f"""
            SELECT id FROM documents
            ORDER BY embedding <-> {vec_literal}
            LIMIT 10
        """)
        latencies.append((time.time() - start) * 1000)
    
    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'max_ms': np.max(latencies),
    }


# ============================================================================
# 10. EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of pgvector SQL functions."""
    
    # Setup
    session = setup_session()
    
    # Example query vector (768 dimensions)
    query_vector = np.random.randn(768).tolist()
    
    # Basic queries
    print("=== Basic Distance Queries ===")
    basic_distance_queries(session)
    
    # Hybrid search
    print("\n=== Hybrid Search ===")
    results = hybrid_search(session, query_vector, category='science', k=10)
    print(results)
    
    # Pagination
    print("\n=== Paginated Search ===")
    page1 = paginated_search(session, query_vector, page=1, page_size=10)
    page2 = paginated_search(session, query_vector, page=2, page_size=10)
    print(f"Page 1: {len(page1)} results")
    print(f"Page 2: {len(page2)} results")
    
    # Aggregations
    print("\n=== Centroids ===")
    centroids = compute_centroids(session)
    print(centroids)
    
    # Configuration
    print("\n=== Configuration Benchmark ===")
    config_results = benchmark_configurations(session, query_vector)
    print(config_results)
    
    # Performance testing
    print("\n=== Performance Metrics ===")
    recall = measure_recall(session, query_vector, k=100)
    print(f"Recall@100: {recall:.2%}")
    
    latency = measure_latency(session, query_vector, num_queries=100)
    print(f"Latency - P50: {latency['p50_ms']:.2f}ms, "
          f"P95: {latency['p95_ms']:.2f}ms, "
          f"P99: {latency['p99_ms']:.2f}ms")


if __name__ == "__main__":
    main()
