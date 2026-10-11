# BenoStreamDB Documentation

BenoStreamDB is an index-overlay engine for the lakehouse built in Rust. It layers reconstructible, persistent secondary indexes—HNSW vector search, BM25 Okapi full-text, Roaring Bitmaps, JSON path filters, and CSR graphs—directly onto Apache Iceberg tables in object storage.

> **⚠️ Deployment Security Notice:** BenoStreamDB network servers (`benostreamdb-flight`, `bsdb-search`, and `benostreamdb-mcp`) do not terminate TLS natively. They support stateless authentication (`BSDB_API_KEY` or JWT tokens), but should be deployed on a trusted internal network or bound to `127.0.0.1` behind a reverse proxy that terminates TLS. See [SECURITY.md](../SECURITY.md).

---

## 🔌 Connectors & Integrations

BenoStreamDB provides native connectors and API bridges for data engineering and AI ecosystems:

* **[Python Client & Vector API](integrations/PYTHON.md)**: Embedded PyO3 bindings with zero-copy PyArrow integration, `Table` operations, vector search, and SQL sessions.
* **[Apache Spark Connector](integrations/SPARK.md)**: Native DataSource V2 connector for Spark 3.5 & 4.x with vector UDFs, secondary index pushdown, catalog stored procedures, and row-level operations.
* **[Trino Connector](integrations/TRINO.md)**: Distributed SQL queries over Iceberg tables with index-accelerated predicate pushdown and graph traversal functions.
* **[dbt Adapter](integrations/DBT.md)**: `dbt-benostreamdb` adapter for analytics engineering, incremental materializations, and vector/graph macros.
* **[Arrow Flight SQL Gateway](integrations/FLIGHT_SQL.md)**: High-performance single-process gRPC database endpoint for polyglot clients (Go, C++, JDBC/ODBC) and BI tools.
* **[REST Search Gateway](integrations/SEARCH_GATEWAY.md)**: Drop-in REST compatibility subset for OpenSearch / Elasticsearch 7.10 (port 9200) and Qdrant v1.x (port 6333).
* **[Model Context Protocol (MCP)](integrations/MCP.md)**: AI agent tool server for LLMs and autonomous agents (Claude Desktop, Cursor, Antigravity).
* **[Java JNI Bridge](integrations/JAVA_JNI.md)**: Low-level Arrow C Data Interface bridge powering JVM query engines.

---

## 📚 Topics & Architecture Guides

### 1. Getting Started & Architecture
* **[Installation Guide](INSTALLATION.md)**: Quickstart, PyPI wheel installation, build instructions, and security deployment patterns.
* **[Core Architecture](ARCHITECTURE.md)**: The indexed lakehouse philosophy, storage layout, write path, and overlay invariants.
* **[ADR-001: Tightened Index Overlay Contract](architecture/ADR_001_TIGHTENED_INDEX_OVERLAY_CONTRACT.md)**: Compound Puffin bundles, artifact lifecycle, and metadata synchronization.

### 2. Table Storage & Catalogs
* **[Iceberg Format Compatibility](ICEBERG_COMPATIBILITY.md)**: Conformance matrix for Apache Iceberg V2 and V3 specifications.
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
* **[Dependency Risk Analysis](DEPENDENCY_RISK.md)**: Supply chain posture and reviewed dependency advisories.

### 6. Tutorials & Walkthroughs
* **[Tutorial 1: AI Retrieval via MCP](tutorials/01_AI_RETRIEVAL_MCP.md)**: Building an agentic retrieval pipeline with MCP tools.
* **[Tutorial 2: JSON Document Retrieval](tutorials/02_JSON_DOCUMENT_RETRIEVAL.md)**: Inverted JSON path indexing and extraction.
* **[Tutorial 3: Graph Traversal](tutorials/03_GRAPH_TRAVERSAL.md)**: Traversing multi-hop entity relationships with SQL table functions.

### 7. Evaluation & Project Roadmap
* **[Benchmarking Methodology](BENCHMARKING.md)**: Reproducibility contract, test harnesses, and competitive analysis guidelines.
* **[Benchmarking Plan](BENCHMARKING_PLAN.md)**: Workloads, competitor matrices, and evaluation dimensions.
* **[Production Readiness Assessment](PRODUCTION_READINESS_ASSESSMENT.md)**: Verification of core invariants and stability milestones.
* **[Project Roadmap](ROADMAP.md)**: Architecture status, shipped milestones, and planned roadmap items.
