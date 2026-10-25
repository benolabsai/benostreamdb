# 2-Minute Quickstart

Get up and running with **BenoStreamDB** in under two minutes.

BenoStreamDB is an embedded, serverless index-overlay engine for Apache Iceberg. You run it directly in your Python or Rust application with **zero external database servers to manage or pay for**.

---

## ⚡ 1. Install (15 seconds)

Install the official Python package from PyPI (precompiled wheels with PyArrow, DataFusion, and CPU/GPU support):

```bash
pip install benostreamdb
```

*Prefer a network endpoint or an existing query engine? Jump to [Core Connectors](#-core-connectors-flight-sql-trino-spark--dbt).*

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

## 🔌 Core Connectors: Flight SQL, Trino, Spark & dbt

BenoStreamDB's primary integration surface is the lakehouse query stack: **Arrow Flight SQL**, **Trino**, **Apache Spark**, and **dbt**. Every connector reads and writes the same Apache Iceberg tables through the Rust engine — there is no export step and no second copy of the data.

### Arrow Flight SQL (`:50051`)

The network endpoint for JDBC/ODBC/ADBC clients, DuckDB, and BI tools:

```bash
# Build and run the server (binds 127.0.0.1:50051 by default)
cargo run --release -p benostreamdb-flight
```

```sql
-- DuckDB client
INSTALL flight;
LOAD flight;
SELECT * FROM flight_sql_query('grpc://127.0.0.1:50051', 'SELECT * FROM documents');
```

### Trino

Build the JNI-backed connector and drop the plugin into every Trino node:

```bash
cd trino-benostreamdb
mvn clean install -DskipTests
```

```properties
# etc/catalog/benostreamdb.properties
connector.name=benostreamdb
benostream.base-uri=s3://my-bucket/
```

```sql
SELECT * FROM benostreamdb.default.logs WHERE severity = 'ERROR';
```

### Apache Spark

A fully native Spark DataSource V2 connector — reads, writes, DDL, row-level operations, and vector UDFs run in-process through JNI:

```bash
cd spark-benostreamdb
mvn clean package -Pspark-4.0     # or -Pspark-3.5 / -Pspark-4.1 / -Pspark-4.2
```

```scala
val spark = SparkSession.builder()
  .config("spark.sql.catalog.benostream", "com.benostreamdb.spark.BenoStreamProcedureCatalog")
  .config("spark.sql.catalog.benostream.warehouse", "s3://my-lakehouse/warehouse")
  .getOrCreate()
```

### dbt

```bash
pip install dbt-benostreamdb
```

```yaml
# ~/.dbt/profiles.yml
benostreamdb_profile:
  target: dev
  outputs:
    dev:
      type: benostreamdb
      database: default
      schema: public
      threads: 4
      uri: "file:///path/to/benostreamdb/warehouse"
```

---

## 🧭 Next Steps

Now that you've run your first query:

* **[Arrow Flight SQL Connector](integrations/FLIGHT_SQL.md)** — Configuration, authentication, and client examples for JDBC/ODBC/ADBC.
* **[Trino Connector](integrations/TRINO.md)** — Distributed SQL, predicate pushdown, graph table functions, and GPU device selection.
* **[Apache Spark Connector](integrations/SPARK.md)** — DataSource V2 reads/writes, row-level operations, vector functions, and streaming.
* **[dbt Adapter](integrations/DBT.md)** — Models, incremental materializations, vector/hybrid-search macros, and lineage.
* **[Installation Guide](installation.md)** — Hardware acceleration (CUDA, Metal, ROCm), package variants, and security guidelines.
* **[pgvector SQL Reference](pgvector_sql_guide.md)** — Complete reference for operators (`<->`, `<=>`, `<#>`, `<~>`), sparse vectors, and index acceleration.
* **[Graph RAG on Iceberg](graph_rag_edge_tables.md)** — Traverse edge tables natively with SQL UDFs (`graph_neighbors`, PageRank).
