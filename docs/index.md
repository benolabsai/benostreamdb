# BenoStreamDB

BenoStreamDB is a high-performance, indexed streaming lakehouse built in Rust. It combines authoritative open table storage (Apache Iceberg V2 & V3) with reconstructible persistent index overlays (Roaring Bitmaps, BM25 Okapi, and HNSW vector search) directly on object storage.

## Key Features

*   **Streaming-First**: Designed for continuous ingestion with WAL durability modes and non-blocking background index construction.
*   **Advisory Index Overlays**: Native support for Vector (HNSW with TurboQuant TQ4/TQ8), Full-Text (BM25 Okapi), and Scalar (Roaring Bitmap & Composite Bitmap) indexes.
*   **Hot Row Cache**: In-memory decoded `RecordBatch` cache (`BLOCK_CACHE`) eliminating disk I/O on scattered kNN row lookups for sub-2ms query responses.
*   **Dual REST Search API (`bsdb-search`)**: Drop-in OpenSearch / Elasticsearch 7.10 compatibility (port 9200) and Qdrant Vector API emulation (port 6333) from a single server.
*   **Multi-Vector Search**: Concurrent retrieval across multiple vector columns with Reciprocal Rank Fusion (RRF) ranking.
*   **Multi-Catalog & Governance**: Native support for Apache Polaris and Lakekeeper with OAuth2 client credentials, Project Nessie, AWS Glue, and Hive Metastore.
*   **Zero-Copy Multiglot**: Core in Rust, with high-performance Python bindings, DataFusion vectorized SQL with pgvector operators, and connectors for Trino and Spark.

## Getting Started

Check out the [Architecture](architecture.md) guide to understand how BenoStreamDB works, the [Comprehensive Guide](COMPREHENSIVE_GUIDE.md) for feature overviews, or the [Benchmarking Guide](BENCHMARKING.md) for competitive performance analysis.
