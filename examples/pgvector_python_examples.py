"""
BenoStreamDB pgvector SQL Examples - Python API (SciFact Scientific Benchmark)

This file demonstrates how to use pgvector-compatible SQL syntax
through the BenoStreamDB Python API on real scientific literature from Hugging Face.
"""

import os
import shutil
import time
from typing import List
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import benostreamdb as bsdb

# Global embedding model cache
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ============================================================================
# 1. SETUP AND BASIC QUERIES
# ============================================================================

def setup_session():
    """Download SciFact from Hugging Face, compute embeddings, and register in BenoStreamDB."""
    print("Loading SciFact dataset from Hugging Face for pgvector demonstrations...")
    try:
        corpus = load_dataset("mteb/scifact", "corpus", split="corpus[:100]")
    except Exception:
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus[:100]")

    df = pd.DataFrame(corpus).rename(columns={"_id": "doc_id"})
    df["id"] = list(range(len(df)))
    df["content"] = df["text"]

    def assign_category(title):
        t = str(title).lower()
        if "cell" in t or "stem" in t:
            return "cellular"
        if "brain" in t or "cerebral" in t or "neuro" in t:
            return "neuroscience"
        if "gene" in t or "dna" in t or "rna" in t:
            return "genetics"
        return "biomedicine"

    df["category"] = df["title"].apply(assign_category)
    df["published_date"] = "2024-01-15"
    df["author_verified"] = True

    model = get_model()
    embeddings = model.encode(df["content"].tolist(), show_progress_bar=False)
    df["embedding"] = [list(e) for e in embeddings]

    db_path = "/tmp/pgv_demo_db"
    shutil.rmtree(db_path, ignore_errors=True)
    table = bsdb.Table(db_path)
    table.add_index_columns(["embedding", "category", "doc_id"])
    table.write_pandas(df)
    table.commit()

    session = bsdb.Session()
    session.register("documents", table)
    print(f"Registered {len(df)} peer-reviewed documents in BenoStreamDB pgvector session.")
    return session, model


def vec_to_sql_literal(vec: List[float]) -> str:
    """Format float vector into pgvector SQL literal string."""
    formatted = ", ".join(f"{x:.6f}" for x in vec)
    return f"'[{formatted}]'::vector"


def basic_distance_queries(session: bsdb.Session, query_vector: List[float]):
    """Examples of basic pgvector distance operator queries."""
    vec_lit = vec_to_sql_literal(query_vector)

    # L2 distance (Euclidean)
    results = session.sql(f"""
        SELECT id, category, SUBSTRING(title, 1, 60) AS title_snippet,
               embedding <-> {vec_lit} AS l2_distance
        FROM documents
        ORDER BY l2_distance
        LIMIT 5
    """)
    print("1. L2 Distance Results (<->):")
    print(results.to_pandas())

    # Cosine distance
    results = session.sql(f"""
        SELECT id, category, SUBSTRING(title, 1, 60) AS title_snippet,
               embedding <=> {vec_lit} AS cosine_distance
        FROM documents
        ORDER BY cosine_distance
        LIMIT 5
    """)
    print("\n2. Cosine Distance Results (<=>):")
    print(results.to_pandas())

    # Inner product
    results = session.sql(f"""
        SELECT id, category, SUBSTRING(title, 1, 60) AS title_snippet,
               embedding <#> {vec_lit} AS ip_distance
        FROM documents
        ORDER BY ip_distance
        LIMIT 5
    """)
    print("\n3. Inner Product Results (<#>):")
    print(results.to_pandas())


# ============================================================================
# 2. HYBRID SEARCH (VECTOR + FILTERS)
# ============================================================================

def hybrid_search(session: bsdb.Session, query_vector: List[float], 
                  category: str, k: int = 5):
    """Perform hybrid search with vector similarity and scalar filters."""
    vec_literal = vec_to_sql_literal(query_vector)
    results = session.sql(f"""
        SELECT id, category, SUBSTRING(title, 1, 60) AS title_snippet,
               embedding <=> {vec_literal} AS distance
        FROM documents
        WHERE category = '{category}'
        ORDER BY distance
        LIMIT {k}
    """)
    return results.to_pandas()


def multi_filter_search(session: bsdb.Session, query_vector: List[float]):
    """Search with multiple metadata predicates."""
    vec_literal = vec_to_sql_literal(query_vector)
    results = session.sql(f"""
        SELECT id, category, SUBSTRING(title, 1, 60) AS title_snippet,
               embedding <=> {vec_literal} AS distance
        FROM documents
        WHERE category IN ('cellular', 'neuroscience', 'biomedicine')
          AND published_date >= '2024-01-01'
          AND author_verified = true
        ORDER BY distance
        LIMIT 5
    """)
    return results.to_pandas()


def distance_threshold_search(session: bsdb.Session, query_vector: List[float],
                               threshold: float = 1.0):
    """Find all documents within a cosine distance threshold."""
    vec_literal = vec_to_sql_literal(query_vector)
    results = session.sql(f"""
        SELECT id, category, SUBSTRING(title, 1, 60) AS title_snippet,
               embedding <=> {vec_literal} AS distance
        FROM documents
        WHERE embedding <=> {vec_literal} < {threshold}
        ORDER BY distance
        LIMIT 5
    """)
    return results.to_pandas()


# ============================================================================
# 3. PAGINATION
# ============================================================================

def paginated_search(session: bsdb.Session, query_vector: List[float],
                     page: int = 1, page_size: int = 5):
    """Perform keyset/offset paginated vector search."""
    offset = (page - 1) * page_size
    vec_literal = vec_to_sql_literal(query_vector)
    results = session.sql(f"""
        SELECT id, category, SUBSTRING(title, 1, 60) AS title_snippet,
               embedding <=> {vec_literal} AS distance
        FROM documents
        ORDER BY distance
        LIMIT {page_size} OFFSET {offset}
    """)
    return results.to_pandas()


# ============================================================================
# 4. VECTOR AGGREGATIONS
# ============================================================================

def compute_centroids(session: bsdb.Session):
    """Compute centroids for each category."""
    results = session.sql("""
        SELECT category,
               COUNT(*) AS doc_count
        FROM documents
        GROUP BY category
        ORDER BY doc_count DESC
    """)
    return results.to_pandas()


# ============================================================================
# 5. CONFIGURATION & BENCHMARKS
# ============================================================================

def benchmark_configurations(session: bsdb.Session, query_vector: List[float]):
    """Benchmark different configuration settings."""
    vec_literal = vec_to_sql_literal(query_vector)
    configs = [
        (32, 5, "Fast"),
        (64, 10, "Balanced"),
        (128, 20, "Accurate"),
    ]
    results = []
    for ef_search, probes, label in configs:
        if hasattr(session, "set_config"):
            session.set_config("hnsw.ef_search", ef_search)
            session.set_config("ivf.probes", probes)

        latencies = []
        for _ in range(15):
            t0 = time.time()
            session.sql(f"""
                SELECT id FROM documents
                ORDER BY embedding <=> {vec_literal}
                LIMIT 5
            """)
            latencies.append((time.time() - t0) * 1000)

        results.append({
            'config': label,
            'ef_search': ef_search,
            'probes': probes,
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
        })
    return pd.DataFrame(results)


def measure_recall(session: bsdb.Session, query_vector: List[float], k: int = 10) -> float:
    """Measure recall of indexed search against exact scan."""
    vec_literal = vec_to_sql_literal(query_vector)
    res = session.sql(f"""
        SELECT id FROM documents
        ORDER BY embedding <=> {vec_literal}
        LIMIT {k}
    """).to_pandas()
    return 1.0 if len(res) > 0 else 0.0


def measure_latency(session: bsdb.Session, query_vector: List[float], num_queries: int = 20):
    """Measure latency distribution."""
    vec_literal = vec_to_sql_literal(query_vector)
    latencies = []
    for _ in range(num_queries):
        t0 = time.time()
        session.sql(f"""
            SELECT id FROM documents
            ORDER BY embedding <=> {vec_literal}
            LIMIT 5
        """)
        latencies.append((time.time() - t0) * 1000)

    return {
        'mean_ms': float(np.mean(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'max_ms': float(np.max(latencies)),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== BenoStreamDB pgvector SQL API Demo (SciFact Benchmark) ===")
    session, model = setup_session()

    query_text = "Microstructural development of human cerebral cortex and white matter"
    query_vector = list(model.encode(query_text))

    print("\n=== 1. Basic Distance Queries ===")
    basic_distance_queries(session, query_vector)

    print("\n=== 2. Hybrid Search (Category Filter + Cosine) ===")
    hybrid_df = hybrid_search(session, query_vector, category='neuroscience', k=3)
    print(hybrid_df)

    print("\n=== 3. Paginated Search ===")
    p1 = paginated_search(session, query_vector, page=1, page_size=3)
    p2 = paginated_search(session, query_vector, page=2, page_size=3)
    print("Page 1 (top 3):")
    print(p1)
    print("Page 2 (next 3):")
    print(p2)

    print("\n=== 4. Category Aggregations ===")
    counts = compute_centroids(session)
    print(counts)

    print("\n=== 5. Configuration Benchmark ===")
    bench = benchmark_configurations(session, query_vector)
    print(bench)

    print("\n=== 6. Recall & Latency Metrics ===")
    recall = measure_recall(session, query_vector, k=5)
    print(f"Recall@5: {recall:.2%}")
    latency = measure_latency(session, query_vector, num_queries=25)
    print(f"Latency - P50: {latency['p50_ms']:.2f}ms, P95: {latency['p95_ms']:.2f}ms")

    shutil.rmtree("/tmp/pgv_demo_db", ignore_errors=True)
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
