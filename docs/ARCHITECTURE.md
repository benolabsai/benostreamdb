# Architecture: The Indexed Lakehouse

BenoStreamDB implements an indexed, compute-disaggregated lakehouse storage architecture that pairs authoritative open table storage with advisory, persistent secondary indexes and a unified retrieval layer.

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

BenoStreamDB attaches persistent, reconstructible sidecar index files *alongside* standard Parquet files:
*   **100% Format Compatibility**: Standard data engines (Spark, Trino, DuckDB, Pandas) read the underlying Parquet and Iceberg tables directly at native speed.
*   **O(log N) Accelerated Retrieval**: BenoStreamDB-aware query engines and REST search gateways leverage inverted bitmap and vector indexes for low-latency queries directly on object storage.
*   **The Overlay Invariant**: Indexes are derived state. If an index file is absent, corrupted, or stale, queries safely degrade to Parquet scanning without failing or returning incorrect results. See [ADR-001: Tightened Overlay Index Artifact Contract & Compound Puffin Bundles](https://www.benostreamdb.org/guides/adr_001.html).

---

## Storage Layout: Hybrid Segments

Data is stored in immutable **Segments** pairing open columnar storage with advisory index sidecars:

1.  **Authoritative Raw Data**:
    *   `data/segment_id.parquet`: Standard Apache Iceberg columnar data files written with ZSTD compression and dictionary encoding. 100% readable by external engines (DuckDB, Spark, Trino, Polars).
2.  **Advisory Secondary Indexes (Apache Iceberg Puffin Compound Bundles)**:
    *   `metadata/segment_id.puffin`: As specified in [ADR-001](https://www.benostreamdb.org/guides/adr_001.html), production deployments consolidate secondary index artifacts into standard Apache Iceberg Puffin (`.puffin`) compound bundles. A single `.puffin` file encapsulates Roaring Bitmap inverted indexes, BM25 term frequencies, and Voronoi-clustered HNSW/TurboQuant vector graphs queried via HTTP byte-range offsets, preventing cloud object count explosion.
    *   *(Uncompacted / Local Debug Mode)*: Prior to Voronoi cluster compaction (`rebuild_cluster_index`), transient per-column sidecars (`.inv.parquet`, `.bm25.parquet`, `.centroids.parquet`, `.hnsw.graph`) may be emitted before `compact()` or `vacuum()` merges them into compound Puffin bundles.
3.  **Iceberg Table Metadata**:
    *   `_manifest/v{N}.avro`: Standard Iceberg V2/V3 snapshot manifests and manifest lists.
    *   `_metadata/v{N}.metadata.json`: Iceberg table schema, sort orders, partition specs, and snapshot lineage.

---

## Ingestion Pipeline: Non-Blocking Indexing & Async WAL

BenoStreamDB achieves high ingestion throughput while maintaining real-time queryability:

1.  **Memtable & Async WAL**: Incoming records write to an in-memory batch buffer with adaptive or immediate WAL fsync durability.
2.  **Parquet Flushes**: Records flush to compressed Parquet files for immediate durability.
3.  **Background Indexing**: High-dimensional vector graphs (HNSW-IVF) and full-text indexes build asynchronously across a Rayon thread pool without blocking writes.
4.  **Atomic Manifest Snapshot**: Once background index construction completes, the manifest is atomically swapped via Optimistic Concurrency Control (OCC) or distributed CAS locks (`FileBasedLock`).

---

## The Read Path & Hot Row Cache

BenoStreamDB combines scalar pre-filtering, vector search, and in-memory caching for ultra-low latencies:

1.  **Scalar Pruning**: Query planner evaluates inverted index sidecars, producing a candidate row bitmap.
2.  **Vector / Keyword Scoring**:
    *   **Vector Search**: Traverses HNSW graph sidecars, evaluating candidates with dynamic SIMD metrics.
    *   **Full-Text Search**: Scores terms using BM25 Okapi sidecars.
    *   **Hybrid RRF**: Combines ranks from vector and keyword search using Reciprocal Rank Fusion ($1 / (k + \text{rank} + 1)$).
3.  **Hot Row Cache (`BLOCK_CACHE`)**:
    *   Decoded Parquet `RecordBatch` chunks are held in an in-memory LRU block cache.
    *   Scattered kNN candidate row fetches resolve directly from memory in sub-millisecond time, completely bypassing disk I/O and eliminating tail latency spikes.

---

## Table-Level Cluster Indexing & Sequential 512 MiB Puffin Sharding

For multi-million and 100M+ vector tables, per-segment vector sidecars create query fan-out overhead across hundreds of small HNSW graphs. BenoStreamDB provides **Table-Level Cluster Indexing** (`ClusterIndex`) via `table.rebuild_cluster_index()`:

```text
                        Table-Level Cluster Index (v42)
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
Shard 0000 (≤512 MiB)          Shard 0001 (≤512 MiB)          Shard 0002 (≤512 MiB)
Clusters [0 .. K0]             Clusters [K0+1 .. K1]          Clusters [K1+1 .. K2]
  - HNSW Graphs (IPC)            - HNSW Graphs (IPC)            - HNSW Graphs (IPC)
  - (seg_id, row_id) Parquet     - (seg_id, row_id) Parquet     - (seg_id, row_id) Parquet
```

### Architectural Guarantees:
1.  **Decoupled Overlay Invariant**: Cluster nodes carry `(segment_idx, row_id)` tuples stored in Arrow Parquet metadata tables. The underlying Parquet data files are read once and **never rewritten or compacted**, completely eliminating fragile ETL pipeline failures and maintaining standard Apache Iceberg read compatibility.
2.  **Sequential 512 MiB Rolling Shards**: Shards (`cluster_{col}_{version}_shard_{i}.puffin`) roll strictly at the advisory target size ($\le 512\text{ MiB}$), complying with cloud object store single-PUT requirements (avoiding AWS S3 5 GB multi-part upload penalties and latency).
3.  **Monotonic Cluster Ordering & Disjoint Ranges**: Clusters are assigned in strictly monotonic order ($0, 1, 2, \dots, K-1$) and packed into non-overlapping cluster ranges (`min_cluster..=max_cluster`), ensuring $O(1)$ shard lookup and high spatial locality during top-$N$ Voronoi centroid routing.
4.  **Zero-Copy Out-of-Core Memory Mapping (`memmap2`)**: Sealed Puffin shards are wrapped in zero-copy memory maps via `Bytes::from_owner(MmapOwner(Arc::new(mmap)))`. Builds stream directly to disk via `BufWriter<File>`, maintaining flat memory consumption ($\sim 2.2 - 3.5\text{ GiB}$) inside strict $16\text{ GiB}$ zero-swap container cgroups without OOM events.
5.  **Multi-Threaded Rayon Compaction**: Centroid assignment (Pass 2) and intra-bucket cluster graph construction (Pass 3) execute across all available CPU cores using Rayon work-stealing parallel iterators, achieving $300\% - 600\%$ CPU saturation and a $3.5\times$ compaction speedup.

---

## Multi-Protocol Search Gateway (`bsdb-search`)

BenoStreamDB exposes its columnar storage and index overlays via standard REST protocols:
*   **OpenSearch / Elasticsearch 7.10 API (Port 9200)**: Drop-in compatibility for `_search`, `_bulk`, `_mapping`, and `_cat/indices`.
*   **Qdrant Vector API (Port 6333)**: Qdrant v1.x REST emulation — collections, point upsert/retrieve, payload & vector edits, vector search, aliases, and the universal query API. See [QDRANT_COMPATIBILITY.md](https://github.com/benolabsai/benostreamdb/blob/main/contrib/benostreamdb-search/docs/QDRANT_COMPATIBILITY.md).
*   **Arrow Flight SQL (Port 50051)**: Low-latency zero-copy gRPC queries for analytical tools.
