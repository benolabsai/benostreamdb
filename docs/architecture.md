# Architecture: The Indexed Lakehouse

HyperStreamDB implements an indexed, compute-disaggregated lakehouse storage architecture that pairs authoritative open table storage with advisory, persistent secondary indexes and a unified retrieval layer.

```text
               Iceberg Table
                     │
       ┌─────────────┴──────────────┐
       │                            │
Authoritative Storage        Advisory Index Overlay
       │                            │
  Parquet Files              Bitmap / Bloom / BM25 / HNSW / TQ
```

## Core Philosophy

**"Your Data is Standard. Your Index is Custom."**

HyperStreamDB attaches persistent, reconstructible sidecar index files *alongside* standard Parquet files:
*   **100% Format Compatibility**: Standard data engines (Spark, Trino, DuckDB, Pandas) read the underlying Parquet and Iceberg tables directly at native speed.
*   **O(log N) Accelerated Retrieval**: HyperStreamDB-aware query engines and REST search gateways leverage inverted bitmap and vector indexes for low-latency queries directly on object storage.
*   **The Overlay Invariant**: Indexes are derived state. If an index file is absent, corrupted, or stale, queries safely degrade to Parquet scanning without failing or returning incorrect results.

---

## Storage Layout: Hybrid Segments

Data is stored in immutable **Segments**, structured as:

1.  **Authoritative Raw Data**:
    *   `segment_id.parquet`: Columnar data written with ZSTD compression and dictionary encoding.
2.  **Advisory Indexes**:
    *   `segment_id.col.inv.parquet`: Inverted Indexes (RoaringBitmaps) for scalar filtering.
    *   `segment_id.col1_col2.comp.parquet`: Composite Roaring Bitmaps for multi-column predicates.
    *   `segment_id.col.bm25.parquet`: BM25 inverted index sidecar with document lengths.
    *   `segment_id.col.centroids.parquet`: Vector IV-centroids for IVF clustering.
    *   `segment_id.col.cluster_N.hnsw.graph`: HNSW graph sidecars (supporting float32 or TurboQuant TQ4/TQ8 packed vectors).
3.  **Iceberg Table Metadata**:
    *   `_manifest/v{N}.avro`: Iceberg V2/V3 snapshot manifests.
    *   `_metadata/v{N}.metadata.json`: Iceberg table schema, sort orders, and partition specs.

---

## Ingestion Pipeline: Non-Blocking Indexing & Async WAL

HyperStreamDB achieves high ingestion throughput while maintaining real-time queryability:

1.  **Memtable & Async WAL**: Incoming records write to an in-memory batch buffer with adaptive or immediate WAL fsync durability.
2.  **Parquet Flushes**: Records flush to compressed Parquet files for immediate durability.
3.  **Background Indexing**: High-dimensional vector graphs (HNSW-IVF) and full-text indexes build asynchronously across a Rayon thread pool without blocking writes.
4.  **Atomic Manifest Snapshot**: Once background index construction completes, the manifest is atomically swapped via Optimistic Concurrency Control (OCC) or distributed CAS locks (`FileBasedLock`).

---

## The Read Path & Hot Row Cache

HyperStreamDB combines scalar pre-filtering, vector search, and in-memory caching for ultra-low latencies:

1.  **Scalar Pruning**: Query planner evaluates inverted index sidecars, producing a candidate row bitmap.
2.  **Vector / Keyword Scoring**:
    *   **Vector Search**: Traverses HNSW graph sidecars, evaluating candidates with dynamic SIMD metrics.
    *   **Full-Text Search**: Scores terms using BM25 Okapi sidecars.
    *   **Hybrid RRF**: Combines ranks from vector and keyword search using Reciprocal Rank Fusion ($1 / (k + \text{rank} + 1)$).
3.  **Hot Row Cache (`BLOCK_CACHE`)**:
    *   Decoded Parquet `RecordBatch` chunks are held in an in-memory LRU block cache.
    *   Scattered kNN candidate row fetches resolve directly from memory in sub-millisecond time, completely bypassing disk I/O and eliminating tail latency spikes.

---

## Multi-Protocol Search Gateway (`hypersearch`)

HyperStreamDB exposes its columnar storage and index overlays via standard REST protocols:
*   **OpenSearch / Elasticsearch 7.10 API (Port 9200)**: Drop-in compatibility for `_search`, `_bulk`, `_mapping`, and `_cat/indices`.
*   **Qdrant Vector API (Port 6333)**: Compatibility for point upserts and vector similarity search.
*   **Arrow Flight SQL (Port 50051)**: Low-latency zero-copy gRPC queries for analytical tools.
