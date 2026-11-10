# BenoStreamDB Documentation

BenoStreamDB is an index-overlay engine for the lakehouse built in Rust. It layers reconstructible, persistent secondary indexes—HNSW vector search, BM25 Okapi full-text, Roaring Bitmaps, JSON path filters, and CSR graphs—directly onto Apache Iceberg tables in object storage.

> **⚠️ Deployment Security Notice:** BenoStreamDB network servers (`benostreamdb-flight`, `bsdb-search`, and `benostreamdb-mcp`) do not terminate TLS natively. They support stateless authentication (`BSDB_API_KEY` or JWT tokens), but should be deployed on a trusted internal network or bound to `127.0.0.1` behind a reverse proxy that terminates TLS. See [SECURITY.md](https://github.com/benolabsai/benostreamdb/blob/main/SECURITY.md).

---

## 🔌 Connectors & Integrations

BenoStreamDB provides native connectors and API bridges for data engineering and AI ecosystems:

* **[Python Client & Vector API](https://www.benostreamdb.org/integrations/python.html)**: Embedded PyO3 bindings with zero-copy PyArrow integration, `Table` operations, vector search, and SQL sessions.
* **[Apache Spark Connector](https://www.benostreamdb.org/integrations/spark.html)**: Native DataSource V2 connector for Spark 3.5 & 4.x with vector UDFs, secondary index pushdown, catalog stored procedures, and row-level operations.
* **[Trino Connector](https://www.benostreamdb.org/integrations/trino.html)**: Distributed SQL queries over Iceberg tables with index-accelerated predicate pushdown and graph traversal functions.
* **[dbt Adapter](https://www.benostreamdb.org/integrations/dbt.html)**: `dbt-benostreamdb` adapter for analytics engineering, incremental materializations, and vector/graph macros.
* **[Arrow Flight SQL Gateway](https://www.benostreamdb.org/integrations/flight_sql.html)**: High-performance single-process gRPC database endpoint for polyglot clients (Go, C++, JDBC/ODBC) and BI tools.
* **[REST Search Gateway](https://www.benostreamdb.org/integrations/search_gateway.html)**: Drop-in REST compatibility subset for OpenSearch / Elasticsearch 7.10 (port 9200) and Qdrant v1.x (port 6333).
* **[Model Context Protocol (MCP)](https://www.benostreamdb.org/integrations/mcp.html)**: AI agent tool server for LLMs and autonomous agents (Claude Desktop, Cursor, Antigravity).
* **[Java JNI Bridge](https://www.benostreamdb.org/integrations/java_jni.html)**: Low-level Arrow C Data Interface bridge powering JVM query engines.

---

## 📚 Topics & Architecture Guides

### 1. Getting Started & Architecture
* **[2-Minute Quickstart](QUICKSTART.md)**: Lightning-fast guide to install, write records, and run vector search and pgvector SQL.
* **[Installation Guide](INSTALLATION.md)**: Hardware acceleration (CUDA/Metal), connectors build, Docker, and security guidelines.
* **[Core Architecture](ARCHITECTURE.md)**: The indexed lakehouse philosophy, storage layout, write path, and overlay invariants.
* **[ADR-001: Tightened Index Overlay Contract](https://www.benostreamdb.org/guides/adr_001.html)**: Compound Puffin bundles, artifact lifecycle, and metadata synchronization.
* **[ADR-002: Global Vector Clustering](https://www.benostreamdb.org/guides/adr_002.html)**: Cluster-keyed index artifacts, sequential Puffin sharding, and index-only compaction.
* **[ADR-003: Graph RAG Hybrid Memory](https://www.benostreamdb.org/guides/adr_003.html)**: O(V) resident state over a mmap'd CSR, with the O(E) edge set kept out-of-core.
* **[ADR-004: Observability Pipeline](https://www.benostreamdb.org/guides/adr_004.html)**: Metrics registry, exporters, OTLP push, and health endpoints.
* **[ADR-005: Graph RAG Feature Set](https://www.benostreamdb.org/guides/adr_005.html)**: CSR/SQL subgraph parity, seeded Leiden/Louvain, bounded materialization, and LLM community summaries.

### 2. Table Storage & Catalogs
* **[Apache Iceberg Storage & Zero-Copy Interop](ICEBERG_STORAGE.md)**: Unified architecture guide to native Iceberg V2/V3 storage, PyIceberg validation, metadata specifications, partition/sort evolution, and multi-engine zero-copy reads (DuckDB, Spark, Trino).
* **[Iceberg Format Compatibility](ICEBERG_COMPATIBILITY.md)**: Granular conformance matrix for Apache Iceberg V2 and V3 specifications.
* **[Iceberg V2/V3 APIs & Evolution](ICEBERG_V2_V3_API.md)**: Sort orders, partition spec evolution, row lineage, and statistics.
* **[Catalog Integration Guide](CATALOG_USAGE.md)**: Production catalog configuration for Apache Polaris, Project Nessie, AWS Glue, Hive Metastore, Databricks Unity, and JDBC.

### 3. Query, Search & Vector Processing
* **[pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md)**: ANSI SQL vector search using standard PostgreSQL pgvector operators (`<->`, `<=>`, `<#>`).
* **[Python Vector API Reference](PYTHON_VECTOR_API.md)**: Standalone batch vector distance functions, sparse vectors, and bit-packed Hamming distances.
* **[Hardware Acceleration & GPU Setup](GPU_SETUP_GUIDE.md)**: Setting up NVIDIA CUDA (dynamic NVRTC), Apple Metal, and Vulkan/WGPU for batch operations.

### 4. Graph Analytics & Graph RAG
* **[Graph RAG & Edge Tables](GRAPH_RAG_EDGE_TABLES.md)**: Managing relationship graphs in Iceberg edge tables, automated CSR graph indexing, NetworkX interoperability, and SQL graph table functions.

### 5. Operations, Hardening & Security
* **[Configuration Reference](CONFIGURATION.md)**: Complete reference of all environment variables, memory knobs, and engine thresholds.
* **[Concurrency & Durability](CONCURRENCY.md)**: Write buffers, WAL durability modes, optimistic concurrency control, and compaction barriers.
* **[Resource Limits & Memory Budget](RESOURCE_LIMITS.md)**: Engine memory caps, block cache tuning, and Rayon thread pool scaling.
* **[Admin CLI Reference](ADMIN_CLI.md)**: Command-line utilities for table inspection, index verification, compaction, and diagnostics.
* **[Monitoring & Observability](MONITORING.md)**: Prometheus metrics endpoints, OpenTelemetry traces, and health check probes.
* **[Recovery Runbook](RECOVERY_RUNBOOK.md)**: Operational runbook for crash recovery, WAL truncation, and index reconciliation.
* **[Zero-Panic Policy](NO_PANIC_POLICY.md)**: Engineering invariants ensuring production paths fail gracefully with descriptive errors instead of crashing.
* **[Dependency Risk Analysis](https://github.com/benolabsai/benostreamdb/blob/main/docs/governance/DEPENDENCY_RISK.md)**: Supply chain posture, reviewed dependency advisories, and dated resolution tracking.

### 6. Tutorials & Walkthroughs
* **[Tutorial 1: AI Retrieval via MCP](https://www.benostreamdb.org/tutorials/01_ai_retrieval_mcp.html)**: Building an agentic retrieval pipeline with MCP tools.
* **[Tutorial 2: JSON Document Retrieval](https://www.benostreamdb.org/tutorials/02_json_document_retrieval.html)**: Inverted JSON path indexing and extraction.
* **[Tutorial 3: Graph Traversal](https://www.benostreamdb.org/tutorials/03_graph_traversal.html)**: Traversing multi-hop entity relationships with SQL table functions.

### 7. Evaluation & Project Roadmap
* **[Benchmarking Methodology](BENCHMARKING.md)**: Reproducibility contract, test harnesses, and competitive analysis guidelines.
* **[Benchmarking Plan](BENCHMARKING_PLAN.md)**: Workloads, competitor matrices, and evaluation dimensions.
* **[MuSiQue GraphRAG Results](MUSIQUE_GRAPHRAG.md)**: Multi-hop retrieval parity with Zilliz/HippoRAG (engine-native two-stage, no hosted LLM).
* **[Production Readiness Assessment](PRODUCTION_READINESS_ASSESSMENT.md)**: Verification of core invariants and stability milestones.
* **[Supported Surface & Build Matrix](SUPPORT_MATRIX.md)**: Shipped platforms/ABIs, cargo feature gates, trusted vs untrusted surfaces, and known limitations.
* **[Project Roadmap](ROADMAP.md)**: Architecture status, shipped milestones, and planned roadmap items.
