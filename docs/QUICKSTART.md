# 2-Minute Quickstart

Get up and running with **BenoStreamDB** in under two minutes.

BenoStreamDB is an embedded, serverless index-overlay engine for Apache Iceberg. You run it directly in your Python or Rust application with **zero external database servers to manage or pay for**.

---

## ⚡ 1. Install (15 seconds)

Install the official Python package from PyPI (precompiled wheels with PyArrow, DataFusion, and CPU/GPU support):

```bash
pip install benostreamdb
```

*Prefer Docker or REST APIs? Jump to [Docker Gateway Quickstart](#-docker-gateway-opensearch--qdrant-apis).*

---

## 🐍 2. Embedded Python: Indexes & Vector Search (45 seconds)

Run this Python script to create an Iceberg table, attach secondary index overlays (HNSW vector, BM25 full-text, and bitmap scalar), insert records, and query:

```python
import benostreamdb as bsdb

# 1. Open (or create) an Iceberg table on local disk or object storage
table = bsdb.Table("file:///tmp/my_quickstart_table")

# 2. Layer reconstructible secondary indexes onto the table
table.add_index("vec", "hnsw")          # HNSW index for vector similarity
table.add_index("title", "bm25")        # BM25 Okapi index for full-text search
table.add_index("category", "bitmap")   # Roaring Bitmap for fast scalar filtering

# 3. Write records (schema is automatically inferred and evolved)
table.write([
    {
        "id": 1,
        "title": "Lakehouse Vector Search",
        "category": "ai",
        "vec": [0.12, 0.54, 0.89],
    },
    {
        "id": 2,
        "title": "Graph RAG on Iceberg",
        "category": "ai",
        "vec": [0.18, 0.49, 0.82],
    },
    {
        "id": 3,
        "title": "Serverless Database Engine",
        "category": "database",
        "vec": [0.85, 0.12, 0.28],
    },
    {
        "id": 4,
        "title": "Hybrid Search with BM25",
        "category": "ai",
        "vec": [0.15, 0.62, 0.81],
    },
    {
        "id": 5,
        "title": "CSR Graph Algorithms",
        "category": "graph",
        "vec": [0.72, 0.35, 0.14],
    },
])

# 4. Commit the batch (generates Iceberg metadata & secondary index overlays)
table.commit()

# 5. Search for the nearest vectors (Top 3)
query_vector = [0.10, 0.50, 0.85]
results = table.vector_search("vec", query_vector, k=3)

# View results as a DataFrame
print(results[["id", "title", "distance"]])
```

**Expected output:**
```text
   id                    title  distance
0   1  Lakehouse Vector Search    0.0036
1   2     Graph RAG on Iceberg    0.0074
2   4  Hybrid Search with BM25    0.0185
```

---

## 🔀 3. Hybrid Search: BM25 + Vector Fusion (20 seconds)

Combine BM25 keyword matching and HNSW vector similarity using Reciprocal Rank Fusion (RRF):

```python
hybrid_results = table.hybrid_search(
    text_column="title",
    query_text="Lakehouse",
    vector_column="vec",
    query_vector=[0.10, 0.50, 0.85],
    k=3,
)

print(hybrid_results[["id", "title", "distance", "rrf_score"]])
```

**Expected output:**
```text
   id                    title  distance  rrf_score
0   1  Lakehouse Vector Search    0.0036   0.032787
1   2     Graph RAG on Iceberg    0.0074   0.016129
2   4  Hybrid Search with BM25    0.0185   0.015873
```

---

## 📊 4. Query with SQL & pgvector Syntax (30 seconds)

BenoStreamDB provides a full Apache DataFusion SQL engine with native support for standard **PostgreSQL pgvector operators** (`<->` L2 distance, `<=>` Cosine distance, `<#>` Inner product). SQL queries transparently benefit from the attached secondary index overlays:

```python
import benostreamdb as bsdb

table = bsdb.Table("file:///tmp/my_quickstart_table")

# Create a local SQL session and register the table
session = bsdb.Session()
session.register("documents", table)

# Query using pgvector cosine distance operator (<=>) filtered by category
query_sql = """
    SELECT id, title, category, vec <=> '[0.10, 0.50, 0.85]'::vector AS cosine_dist
    FROM documents
    WHERE category = 'ai'
    ORDER BY cosine_dist ASC
    LIMIT 3
"""

df = session.sql(query_sql)
print(df.to_pandas())
```

**Expected output:**
```text
   id                    title category  cosine_dist
0   1  Lakehouse Vector Search       ai     0.000186
1   2     Graph RAG on Iceberg       ai     0.003650
2   4  Hybrid Search with BM25       ai     0.008276
```

---

## 🏎️ 5. 4x Index Compression with TurboQuant™ (15 seconds)

BenoStreamDB ships with **TurboQuant™ (TQ8)** out-of-the-box. It applies Fast Walsh-Hadamard Transform (FWHT) and scalar quantization to compress vectors **4x** on disk and RAM while retaining >99% recall:

```python
# Add an accelerated TurboQuant 8-bit HNSW index to your vector column
table.add_index("vec", "hnsw_tq8")

# Searches automatically leverage hardware-accelerated Asymmetric Distance Calculation
results = table.vector_search("vec", [0.10, 0.50, 0.85], k=3)
print(results[["id", "title", "distance"]])
```

**Expected output:**
```text
   id                    title  distance
0   1  Lakehouse Vector Search    0.0036
1   2     Graph RAG on Iceberg    0.0074
2   4  Hybrid Search with BM25    0.0185
```

---

## 🐳 Docker Gateway: OpenSearch & Qdrant APIs

If your stack already uses **OpenSearch / Elasticsearch 7.10** or **Qdrant**, you can spin up the dual-API gateway with one command:

```bash
docker run -d --name benostreamdb \
  -p 9200:9200 \
  -p 6333:6333 \
  -p 50051:50051 \
  benostreamdb/quickstart:latest
```

* **OpenSearch / Elasticsearch 7.10 API (`:9200`)**: Text indexing, BM25, and hybrid search. Test with `curl http://localhost:9200/_cluster/health`.
* **Qdrant Vector API (`:6333`)**: Vector upsert, collection management, and filtered similarity search.
* **Arrow Flight SQL (`:50051`)**: Streaming columnar SQL for DuckDB, Spark, and BI tools.

Both APIs share the exact same underlying Iceberg tables with zero data duplication.

---

## 🧭 Next Steps

Now that you've run your first query:

* **[Installation Guide](installation.md)** — Hardware acceleration (CUDA, Metal, ROCm), package variants, and security guidelines.
* **[pgvector SQL Reference](pgvector_sql_guide.md)** — Complete reference for operators (`<->`, `<=>`, `<#>`, `<~>`), sparse vectors, and index acceleration.
* **[Graph RAG on Iceberg](graph_rag_edge_tables.md)** — Traverse edge tables natively with SQL UDFs (`graph_neighbors`, PageRank).
* **[Connectors](../index.rst#connectors-integrations)** — Integrate with **Apache Spark**, **Trino**, **dbt**, or **Arrow Flight SQL**.
