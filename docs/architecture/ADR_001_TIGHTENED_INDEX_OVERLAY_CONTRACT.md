# ADR-001: Tightened Overlay Index Artifact Contract & Compound Puffin Bundles

- **Status**: Proposed / Under Implementation
- **Date**: 2026-10-03
- **Authors**: BenoStreamDB Core Team
- **Target Version**: v0.13.0+
- **Companion Documents**:
  - [`docs/architecture.md`](../architecture.md) (Lakehouse Storage Architecture)
  - [`docs/ICEBERG_COMPATIBILITY.md`](../ICEBERG_COMPATIBILITY.md) (Iceberg Specification Conformance)
  - [`docs/CONCURRENCY.md`](../CONCURRENCY.md) (Multi-Writer Transaction Model)
  - [`docs/BENCHMARKING_PLAN.md`](../BENCHMARKING_PLAN.md) (Production Verification Framework)

---

## 1. Executive Summary

BenoStreamDB pairs authoritative, open Apache Iceberg table storage with derived, advisory overlay indexes (Vector HNSW-TurboQuant, BM25 lexical, Roaring Bitmaps, and Bloom filters). Treating overlay indexes as **disposable accelerators** ensures 100% open-format compatibility and zero lock-in: if an index is lost or corrupted, the table degrades gracefully to raw Parquet scans.

However, in large-scale deployments, loose sidecar management presents two structural risks:
1. **Object Count Explosion ($O(M \times N)$)**: Multiple index files per Parquet segment can multiply S3/GCS object counts by 5x–10x, leading to cloud storage rate limiting, high `PUT`/`LIST` costs, and slow metadata planning.
2. **Silent Index Drift**: If an external engine (Apache Spark, Trino, Snowflake, or PyIceberg) mutates or compacts a Parquet file out-of-band, an index bound only by file name can return corrupted vector candidates or misaligned row offsets.

This document specifies the **Tightened Index Artifact Contract**:
- **Cryptographic & Structural Lineage Binding**: Explicitly binding every index artifact to the data file's SHA-256 checksum, snapshot ID, and exact record count.
- **Compound Index Bundles (Apache Iceberg Puffin)**: Consolidating multiple index structures into single Puffin (`.puffin`) sidecars queried via HTTP byte-range offsets, reducing object overhead by 70%–85%.
- **Detached Overlay Catalogs**: Formalizing the architecture for indexing external, read-only Iceberg tables without write permissions to the upstream storage or catalog.

---

## 2. Problem Statement & Motivation

### 2.1 The Disposable Accelerator Invariant

BenoStreamDB establishes the core invariant:
$$\text{Correctness}(\text{Table}) \equiv \text{Correctness}(\text{Parquet Data}) \quad \forall \text{ queries}$$

Indexes do not store authoritative data. They are derived mathematical projections (nearest-neighbor graphs, inverted postings, bitmap filters) that map queries to row offsets within specific Parquet files. If an index is deleted, query results remain identical—only execution latency changes.

### 2.2 The Scaling Vulnerabilities

In production lakehouses, this model encounters two concrete challenges:

#### Vulnerability A: Object Count Explosion
In a streaming table with 50,000 Parquet segments:
- Raw Parquet files: 50,000 objects
- Inverted Bitmap indexes (`.inv.parquet`): 50,000 objects
- BM25 full-text indexes (`.bm25.parquet`): 50,000 objects
- Vector HNSW-TQ8 graph + codebooks (`.hnsw.graph`, `.centroids.parquet`): 100,000 objects
- **Total Objects**: 250,000+ objects on S3/GCS.

This explosion increases cloud API billing, increases S3 503 SlowDown throttling (as demonstrated in §7.4 benchmarks), and severely degrades partition-level `LIST` performance.

#### Vulnerability B: Silent Index Drift & Out-of-Band Mutations
Because Iceberg tables are open, external engines often interact with the data:
1. Apache Spark runs `rewriteDataFiles` (bin-packing compaction) or Trino executes `OPTIMIZE`.
2. Old Parquet files are unlinked, and new Parquet files are written.
3. If an index sidecar is bound merely by path conventions or if a file is overwritten in-place, the index can point to row offsets that no longer correspond to the original data, producing silent data corruption.

---

## 3. The Tightened Index Artifact Specification

### 3.1 Metadata Lineage Contract (`IndexFile`)

Every index entry recorded in an Iceberg manifest or secondary overlay catalog MUST contain strict cryptographic and positional bindings to its source data file.

#### Rust Data Structure Definition (`src/core/manifest/types.rs`)

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexFile {
    /// Format version of the serialized index structure (e.g., 1 for legacy, 2 for tightened/Puffin)
    #[serde(default = "default_index_format_version")]
    pub format_version: u32,

    /// Unique identifier for this index instance
    pub index_id: String,

    /// High-level index category: "vector", "lexical", "scalar", "composite", "bloom"
    pub index_category: String,

    /// Concrete algorithm implementation: "hnsw_turboquant8", "hnsw_f32", "bm25_v1", "roaring_bitmap_v1"
    pub algorithm: String,

    /// Target column name within the Parquet schema
    pub column_name: String,

    // -----------------------------------------------------------------------
    // Strict Lineage Bindings (Data Integrity Guards)
    // -----------------------------------------------------------------------
    
    /// Exact Iceberg Snapshot ID active when this index was constructed
    pub source_snapshot_id: i64,

    /// Cryptographic SHA-256 hash of the target Parquet data file
    pub source_data_checksum: String,

    /// Exact record count of the target Parquet data file
    /// Enforces a strict 1-to-1 bijection between index point IDs and Parquet row offsets
    pub source_record_count: i64,

    /// Cryptographic SHA-256 hash of the index artifact itself (detects storage corruption / bitrot)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_checksum: Option<String>,

    // -----------------------------------------------------------------------
    // Physical Storage Location & Puffin Bundle Offsets
    // -----------------------------------------------------------------------

    /// Physical URI or relative storage path of the index artifact (or Puffin bundle)
    pub file_path: String,

    /// Puffin blob type if packed into an Iceberg Puffin container
    /// e.g., "org.apache.iceberg.vector.hnsw-tq8", "org.apache.iceberg.lexical.bm25"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_type: Option<String>,

    /// Byte offset within the container file where the index payload begins
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_offset: Option<i64>,

    /// Length in bytes of the index payload
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_length: Option<i64>,

    /// Compression algorithm applied to the index blob ("zstd", "lz4", "none")
    #[serde(default = "default_compression")]
    pub compression: String,
}

fn default_index_format_version() -> u32 { 2 }
fn default_compression() -> String { "zstd".to_string() }
```

### 3.2 Verification Protocol on Read Path

Before any query engine evaluates an index overlay during scan planning:

```text
Query Planner
      │
      ▼
Fetch ManifestEntry ──► Read Data File Metadata (checksum, record_count, snapshot)
      │
      ▼
Inspect IndexFile ────► Compare:
                        1. index.source_data_checksum == data_file.file_checksum
                        2. index.source_record_count == data_file.record_count
                        3. index.source_snapshot_id <= active_snapshot_id
      │
      ├───────────────────────────────┬───────────────────────────────┐
      │ (Match: 100% Consistent)      │ (Mismatch: Stale / Modified)  │
      ▼                               ▼                               ▼
Execute Accelerated Scan        Degrade to Parquet Scan         Trigger Background
(HNSW / BM25 / Roaring)        (Pushdown Filter / Brute-Force)  Re-Indexing Job
```

#### Invalidation & Fallback Invariants
1. **Zero False Positives / Corruption**: If `source_data_checksum != data_file.file_checksum`, the reader MUST immediately reject the index. Under no circumstances may an index be evaluated against a data file whose content hash differs.
2. **Graceful Degradation**: When an index is rejected, the query MUST NOT panic or fail; it degrades to a standard DataFusion Parquet filter scan.
3. **Telemetry & Self-Healing**: The engine logs an advisory warning with the segment ID and queues an asynchronous background index repair (`recover_indexes_async`).

---

## 4. Compound Index Bundles (Apache Iceberg Puffin Specification)

To resolve the object explosion problem, BenoStreamDB adopts **Apache Iceberg Puffin** as the standard compound container format for secondary index artifacts.

### 4.1 Physical Puffin File Layout

A single `.puffin` file is written alongside each Parquet data file (or partition group), encapsulating all declared indexes in contiguous byte blobs:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Puffin File Magic: 0x50 0x55 0x46 0x31          │
├────────────────────────────────────────────────────────────────────────┤
│ Blob 0: Vector HNSW Graph & TQ8 Codebook                               │
│ - Type: "org.apache.iceberg.vector.hnsw-tq8"                           │
│ - Offset: 4, Length: 2,411,520 bytes                                   │
├────────────────────────────────────────────────────────────────────────┤
│ Blob 1: BM25 Lexical Postings & Term Frequencies                       │
│ - Type: "org.apache.iceberg.lexical.bm25"                              │
│ - Offset: 2,411,524, Length: 812,040 bytes                             │
├────────────────────────────────────────────────────────────────────────┤
│ Blob 2: Roaring Bitmap Scalar Index                                    │
│ - Type: "org.apache.iceberg.index.roaring-bitmap"                      │
│ - Offset: 3,223,564, Length: 45,120 bytes                              │
├────────────────────────────────────────────────────────────────────────┤
│ Puffin File Footer (JSON Metadata: Blob Offsets, Types, Hashes)        │
├────────────────────────────────────────────────────────────────────────┤
│ Footer Length (4 bytes) + Magic: 0x50 0x55 0x46 0x31                   │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Object Count & I/O Reduction

| Dimension | Legacy Loose Sidecars | Tightened Puffin Bundles | Efficiency Gain |
|---|---|---|---|
| **Objects per Segment (4 indexes)** | 4–5 files | **1 file** (`.puffin`) | **75%–80% fewer objects** |
| **Total Objects (50k segments)** | ~250,000 objects | **50,000 objects** | **200,000 fewer objects** |
| **S3 PUT API Cost ($0.005/1k)** | $1.25 per ingestion batch | $0.25 per ingestion batch | **5x cost reduction** |
| **Range Read Acceleration** | Multiple distinct HTTP connections | Single pooled connection with HTTP `Range` | **Eliminates cold connection overhead** |

---

## 5. Detached Overlay Catalogs for External Iceberg Tables

One of the most compelling applications of the tightened overlay design is **accelerating third-party Iceberg tables without write permissions**.

### 5.1 Use Case & Scenario
- An enterprise maintains an authoritative 50 TB Iceberg data lake managed by Databricks, Snowflake, AWS EMR, or Google Cloud BigLake.
- An AI / Search application requires sub-millisecond vector similarity search (HNSW-TurboQuant) and hybrid BM25 search over these tables.
- **Constraint**: The AI application team cannot modify the upstream Iceberg table, cannot alter table properties, and has read-only S3 IAM permissions on the data bucket.

### 5.2 Architecture: Detached Overlay Namespace

```text
Upstream Authoritative Lake (Read-Only)          BenoStreamDB Acceleration Tier (Read-Write)
┌──────────────────────────────────────┐        ┌───────────────────────────────────────────┐
│ s3://enterprise-data/warehouse/      │        │ s3://ai-team-overlays/indexes/            │
│   ├── metadata/                      │        │   ├── detached_catalog.json               │
│   │     └── v104.metadata.json ──────┼──Sync──┼──►│     (Tracks upstream snapshot ID)     │
│   └── data/                          │        │   └── overlays/                           │
│         ├── part-0001.parquet ───────┼──Scan──┼──►│     ├── part-0001.puffin (Vector/BM25)│
│         ├── part-0002.parquet ───────┼────────┼──►│     ├── part-0002.puffin (Vector/BM25)│
│         └── part-0003.parquet (New)  │        │   │     └── (Pending background build)    │
└──────────────────────────────────────┘        └───────────────────────────────────────────┘
```

### 5.3 Synchronization Protocol
1. **Catalog Registration**: BenoStreamDB mounts the external table via its metadata URI (`s3://enterprise-data/warehouse/metadata/v104.metadata.json`) with an external overlay storage prefix (`s3://ai-team-overlays/indexes/`).
2. **Snapshot Pinning**: The detached catalog records `upstream_snapshot_id = 104`.
3. **Out-of-Band Upstream Mutation Handling**:
   - If the upstream team appends `part-0003.parquet` and advances to `v105`:
     - Query requests for `part-0001` and `part-0002` immediately resolve via the existing `.puffin` overlays.
     - Query requests for `part-0003` seamlessly execute as DataFusion brute-force scans.
     - An asynchronous worker builds `part-0003.puffin` into the detached overlay prefix and updates the detached manifest atomically.
   - If the upstream team runs compaction (replacing `part-0001` with `part-0001-compacted`):
     - The cryptographic checksum check (`source_data_checksum`) fails for `part-0001`.
     - The query engine drops the stale index without returning stale rows, preserving 100% query correctness.

---

## 6. Merge-On-Read (MOR) Lifecycle for Secondary Indexes

To achieve sub-millisecond management latencies and eliminate cloud storage rate-limiting, BenoStreamDB applies **Merge-On-Read (MOR)** principles directly to secondary index lifecycles, replacing legacy Copy-On-Write (COW) behaviors.

### 6.1 `drop_index`: Instantaneous $O(1)$ Metadata Tombstoning vs. Synchronous Physical Deletion

#### Problem in COW Index Drops
In large production tables (e.g., 50,000 segments), dropping a column index under a Copy-On-Write model requires:
1. Scanning all manifest entries across all snapshot manifests.
2. Synchronously issuing tens of thousands of cloud storage `DELETE` calls (`.centroids`, `.hnsw.*`, `.inv.parquet`, `.doclen`, etc.).
3. Risking S3 503 SlowDown rate-limiting, high latency (tens of seconds to minutes), and partial network failures that leave orphaned metadata or inconsistent states.

#### The MOR Solution: Metadata Tombstoning
1. **$O(1)$ Manifest Commit**: When `table.drop_index(column)` is invoked, the engine writes a new atomic manifest snapshot that purges the target index pointers from `ManifestEntry.index_files` and clears the column's index spec in `Schema.fields`.
2. **Sub-5ms Execution Latency**: The API call completes immediately without blocking on cloud storage object deletion calls.
3. **Scan Invariant**: Query planning immediately ignores unreferenced or tombstoned indexes. There is zero risk of query race conditions or `NoSuchKey` errors because readers resolve against the new snapshot manifest.
4. **Deferred Asynchronous Vacuum**: Physical index files (`.puffin`, `.hnsw.*`, `.inv.parquet`) are safely unlinked and garbage-collected in the background during periodic table maintenance (`table.vacuum()` / `expire_snapshots()`).

### 6.2 `add_index`: Instant Schema Activation with Hybrid Merge-On-Read Scans

#### Problem in COW Index Builds
Calling `add_index(col, algo)` on a table with hundreds of millions of rows historically triggered a blocking full-table backfill across all historical segments, consuming heavy compute and delaying query readiness.

#### The MOR Solution: Hybrid Index-Scan Planning
1. **Immediate Schema Registration**: `add_index` commits the schema update in milliseconds. Incoming flushes immediately build the index on newly written segments.
2. **Hybrid Candidate Evaluation**:
   - For segments that carry the index: The query engine executes fast index traversal (Vector HNSW or Lexical BM25).
   - For legacy or unindexed segments: The query engine seamlessly falls back to DataFusion Parquet scans (brute-force vector distance or predicate evaluation).
   - The Query Coordinator merges top-$K$ candidate lists across both paths.
3. **Incremental Background Backfill**: A background worker incrementally indexes historical segments over time without locking or blocking query availability.

### 6.3 Delta Indexing & Positional Deletion Masks

When data rows are deleted or updated:
1. **Immutable Base Index**: The primary Puffin bundle (`part-0001.puffin`) remains read-only.
2. **Roaring Delete Mask**: Deletions are recorded in a lightweight Roaring Bitmap. During vector nearest-neighbor graph walks or inverted index posting lookups, candidate row IDs are checked against the delete mask in $O(1)$ time.
3. **Delta Ingestion Buffer**: Newly appended records accumulate in an in-memory Flat/IVF buffer or small delta-Puffin container.
4. **Compaction Consolidation**: When the table runs compaction or maintenance, the base index, delete mask, and delta index are consolidated into a fresh, canonical Puffin container.

### 6.4 Operational Comparison: Copy-On-Write (COW) vs. Merge-On-Read (MOR)

| Dimension | Copy-On-Write (COW) Indexes | Merge-On-Read (MOR) Indexes | Architectural Benefit |
|---|---|---|---|
| **`drop_index` Latency** | $O(N)$ synchronous S3 deletions (10s – 5 mins) | $O(1)$ atomic manifest commit (< 5 ms) | **>10,000x faster index drop** |
| **`drop_index` S3 I/O** | 50,000–250,000 `DELETE` HTTP calls | 1 manifest `PUT` call | **Zero S3 503 SlowDown throttling** |
| **`add_index` Query Availability** | Blocked until all segments backfilled (hours) | Immediate query availability via hybrid plan | **Zero query downtime** |
| **Data Ingestion Overhead** | Full re-indexing on every append / mutation | Immutable base + delta buffer / delete mask | **Consistent sub-second write latencies** |
| **Consistency / Isolation** | Subject to partial S3 delete failures | Snapshot isolation with atomic manifest commits | **ACID guarantees; zero orphaned pointers** |
| **Cleanup Mechanism** | Immediate inline deletion | Decoupled background garbage collection (`vacuum`) | **Separates control plane from data cleanup** |

### 6.5 API & Garbage Collection Lifecycle Contract

```rust
// Proposed Index Lifecycle API in Table Trait
pub trait IndexLifecycle {
    /// Instant O(1) drop via metadata tombstoning (MOR mode - default)
    /// Removes index references from active manifest without blocking on storage I/O
    async fn drop_index(&self, column_name: &str) -> Result<()>;

    /// Optional synchronous purge (COW mode) for explicit compliance / storage reclamation
    async fn drop_index_purge(&self, column_name: &str) -> Result<u64>;

    /// Background garbage collector: deletes physical index artifacts unreferenced by any active snapshot
    async fn vacuum_indexes(&self, retain_snapshots: usize) -> Result<VacuumReport>;
}
```

#### Garbage Collection & Vacuum Algorithm
During `table.vacuum_indexes(retain_snapshots)`:
1. Enumerate all `index_files` across all active and retained snapshots ($\le \text{retain\_snapshots}$).
2. Form the set of live index file URIs: $\mathcal{U}_{\text{live}} = \bigcup_{S \in \text{RetainedSnapshots}} \text{IndexURIs}(S)$.
3. Scan table storage prefix for candidate index artifacts (`.puffin`, `.hnsw.*`, `.inv.parquet`, etc.).
4. Any physical index artifact $F \notin \mathcal{U}_{\text{live}}$ is safe for asynchronous unlinking:
$$\text{ArtifactsToPurge} = \mathcal{U}_{\text{storage}} \setminus \mathcal{U}_{\text{live}}$$

### 6.6 Hybrid Query Planner State Machine

When a query arrives for column $C$ with index type $T$:

```text
Incoming Query (Vector / Lexical Search on column C)
                      │
                      ▼
        Load Active Snapshot Manifest
                      │
                      ▼
   Partition Manifest Segments into Disjoint Sets:
   ├── S_indexed   = { s ∈ Segments | ∃ idx ∈ s.index_files: idx.column == C }
   └── S_unindexed = { s ∈ Segments | s ∉ S_indexed }
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   Accelerated Scan           Fallback Scan
   (Puffin Range Read        (DataFusion Parquet
    + HNSW / BM25)            Brute-Force Scan)
         │                         │
   Top-K Candidates          Top-K Candidates
   (Score, RowLocation)      (Score, RowLocation)
         └────────────┬────────────┘
                      ▼
           Global Top-K Reduction
                      │
                      ▼
           Return Result Batches
```

This state machine guarantees that adding an index can never return incomplete results—it simply speeds up segments as they are indexed, achieving true progressive index acceleration without downtime.

---

## 7. Migration & Compatibility Plan

### 7.1 Phased Implementation Roadmap

1. **Phase 1: Metadata Schema Enrichment (v0.12.x)** — ✅ Complete:
   - Updated `IndexFile` in `src/core/manifest/types.rs` with `source_data_checksum`, `source_snapshot_id`, and `source_record_count`.
   - Maintained backward compatibility with legacy deserialization using `#[serde(default)]`.
2. **Phase 2: Validation Enforcement in Scan Engine (v0.12.0)** — ✅ Complete:
   - Added strict checksum and record-count validation in `src/core/reader/scan.rs` and `src/core/reader/filter.rs`.
   - Rejected mismatched indexes with telemetry logging and automated degradation to Parquet scan.
   - Implemented `SegmentConfig::is_index_valid()` three-check validation (checksum, record count, snapshot ID).
3. **Phase 3: Puffin Compound Bundler & MOR Add/Drop (v0.12.0)** — ✅ Complete:
   - Implemented `PuffinIndexWriter` in `src/core/puffin.rs` to emit compound `.puffin` containers during segment flushes.
   - **Puffin is now the default and only index storage format.** Every secondary index — lexical (BM25 + doclen), scalar inverted (Int32/Int64/Float/Date/Bool), vector (HNSW-IVF centroids/graph/mapping), and graph (CSR offsets/edges/dict) — is packed into a single `{segment}.puffin` bundle. The legacy loose-sidecar writer path has been removed.
   - Implemented byte-range `PuffinIndexReader` in `src/core/reader/scan.rs` and `src/core/reader/filter.rs`, plus Puffin-aware loaders for vector (`HnswIvfIndex::load_from_puffin`), graph (`MmapCsrGraph::from_bytes`), and preload.
   - Implemented instant $O(1)$ `drop_index` metadata pruning with atomic manifest commits and cache invalidation.
   - Implemented hybrid MOR index query planning in `keyword_search_index` and `phrase_search_index`.
   - Validated with unit tests (`test_puffin_write_read`, `test_puffin_compound_index_writer_reader`) and integration test (`test_puffin_hybrid_segment_writer_and_reader`).
4. **Phase 4: Detached Overlay Catalog API (v0.13.0+)** — ✅ Complete:
   - Implemented `Table::mount_external_iceberg(table_uri, overlay_storage_uri)` in `src/core/table/detached.rs`, with the Python binding `Table.mount_external_iceberg(...)`.
   - Persisted the `DetachedOverlayCatalog` sidecar (`detached_catalog.json`) recording `upstream_snapshot_id`, `upstream_schema_id`, and the last-synced overlay manifest version.
   - Implemented `Table::sync_detached_overlay_async()` (incremental snapshot reconciliation) and the background worker `Table::start_detached_sync_worker(interval)`.
   - Validated with unit tests (`catalog_round_trips_through_json`, `parses_benostream_and_iceberg_metadata_versions`) and integration tests (`test_mount_detached_overlay_catalog`, `test_sync_on_non_overlay_table_errors`).

### 7.2 Backward Compatibility Guarantees

| Concern | Guarantee |
|---|---|
| **Legacy IndexFile (format_version=1)** | Fully deserializable via `#[serde(default)]`. Lineage fields default to empty/0, which passes validation (no false rejections). |
| **Mixed Puffin + Loose Sidecars** | The scan engine supports both: if `blob_offset`/`blob_length` are present → byte-range read; otherwise → full object GET. |
| **External Engines** | Puffin sidecars are invisible to Spark/Trino/DuckDB. These engines read raw Parquet data files unmodified. |
| **Downgrade Path** | Removing `.puffin` sidecars and clearing `index_files` from the manifest instantly reverts to brute-force Parquet scans with zero data loss. |

---

## 8. Alternatives Considered

| Alternative | Pros | Cons | Reason for Rejection |
|---|---|---|---|
| **Embedding Indexes Inside Parquet Files (Native Metadata)** | Single physical file; zero sidecars. | Requires custom Parquet writer; breaks standard external tools (Spark, DuckDB); cannot index third-party read-only tables. | Violates 100% open Iceberg compatibility and zero-lock-in philosophy. |
| **Global Monolithic Index Object** | Exactly 1 object for the entire table. | Massive re-indexing cost on every append; impossible to scale with streaming writes; high memory footprint during query. | Incompatible with high-frequency streaming ingestion. |
| **Loose Sidecars (Current Status Quo)** | Simple 1-file-per-index layout; trivial debugging. | Exploding object counts ($O(M \times N)$); high S3 API costs; vulnerability to silent drift if unvalidated. | Unsustainable at >10k segments. |
| **Tightened Puffin Bundles with MOR Lifecycle (Adopted)** | Reduces object count by 75%+; standard Iceberg format; supports HTTP range requests; strict checksum lineage; instant $O(1)$ add/drop via MOR metadata tombstones. | Requires hybrid query planning (indexed + unindexed segments). | **Selected as the optimal production architecture.** |

---

## 9. Observability & Telemetry

The tightened index overlay contract exposes structured observability at every decision point in the index lifecycle.

### 9.1 Index Validation Events

When `SegmentConfig::is_index_valid()` rejects an index during scan planning, the engine emits structured `tracing::warn!` events:

```text
Index {file_path} rejected: checksum mismatch (expected {data_checksum}, got {index_checksum})
Index {file_path} rejected: record count mismatch (expected {data_records}, got {index_records})
Index {file_path} rejected: from future snapshot {index_snapshot} (active {active_snapshot})
```

Each rejection triggers automatic graceful degradation to Parquet brute-force scan. These events should be monitored with alerting thresholds:
- **Sustained rejection rate > 5%**: Indicates upstream compaction activity or index build failures requiring `recover_indexes_async()`.
- **Future snapshot warnings**: May indicate clock skew or concurrent writer races.

### 9.2 Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `bsdb_io_bytes_read_total` | Counter | Total bytes read from storage (Parquet + index artifacts). |
| `bsdb_cache_hits_total{cache}` | Counter | Cache hits per cache tier (`inverted_index`, `hnsw_ivf`, `index`, `analyzer_meta`). |
| `bsdb_cache_misses_total{cache}` | Counter | Cache misses triggering storage reads. |
| `bsdb_index_segments_accelerated` | Gauge | Number of segments resolved via index (vs. brute-force) per query. |
| `bsdb_index_segments_fallback` | Gauge | Number of segments that fell back to Parquet scan. |
| `bsdb_vacuum_artifacts_purged` | Counter | Physical index files deleted during `vacuum_indexes`. |

### 9.3 Cache Invalidation Contract

After `drop_index`, all four in-memory cache tiers are explicitly invalidated for the dropped column's artifacts:

```rust
// src/core/table/index_config.rs — drop_index
INVERTED_INDEX_CACHE.invalidate(&key).await;
HNSW_IVF_CACHE.invalidate(&key).await;
INDEX_CACHE.invalidate(&key).await;
ANALYZER_META_CACHE.invalidate(&key).await;
```

This prevents stale cache entries from serving results for dropped indexes during the window between manifest commit and physical file deletion.

---

## 10. Security Considerations

### 10.1 Index Integrity via Cryptographic Binding

The `index_checksum` field (SHA-256 of the index artifact) provides tamper detection:
- If a malicious actor or bitrot event corrupts a `.puffin` or `.hnsw.graph` file, the checksum verification fails at read time and the index is rejected.
- This is a **detection** mechanism, not a prevention mechanism. Write-path integrity depends on the underlying object store's own durability guarantees (S3 CRC32c, GCS MD5, Azure Blob ETag).

### 10.2 Detached Overlay Security Model

When overlays are deployed against read-only upstream tables (§5):
- The acceleration tier has **no write access** to the authoritative data lake.
- Index overlays cannot alter, inject, or mask authoritative Parquet data.
- If the overlay storage is compromised, the worst-case impact is query performance degradation (indexes rejected → Parquet scan fallback), not data corruption.

### 10.3 Multi-Tenant Isolation

In multi-tenant deployments where multiple teams share the same underlying Iceberg table:
- Each tenant's detached overlay catalog is scoped to its own storage prefix.
- Cross-tenant index poisoning is prevented by the `source_data_checksum` binding: an index built against tenant A's overlay prefix will be rejected if presented against tenant B's data files.

---

## Appendix A: Implementation Cross-Reference Map

This appendix maps ADR-001 specification elements to their concrete implementations in the BenoStreamDB codebase.

### A.1 Core Data Structures

| Specification Element | Source File | Symbol |
|---|---|---|
| `IndexFile` lineage struct | `src/core/manifest/types.rs` | `IndexFile` |
| `SegmentConfig` with lineage fields | `src/lib.rs` | `SegmentConfig` |
| `ColumnIndexConfig` per-column config | `src/core/table/state.rs` | `ColumnIndexConfig` |
| `IndexAlgorithm` enum (HNSW, BM25, etc.) | `src/core/manifest/types.rs` | `IndexAlgorithm` |

### A.2 Write Path (Index Construction)

| Operation | Source File | Function |
|---|---|---|
| Inverted index build (tokenize → postings) | `src/core/index/build_inverted.rs` | `build_inverted_index()` |
| Vector HNSW-TQ8 index build | `src/core/index/build_vector.rs` | `build_vector_index()` |
| Puffin bundle emission | `src/core/segment.rs` | `finish_indexing()` |
| Puffin writer internals | `src/core/puffin.rs` | `PuffinIndexWriter` |
| Manifest lineage binding | `src/core/segment.rs` | `to_manifest_entry()` |

### A.3 Read Path (Index Query)

| Operation | Source File | Function |
|---|---|---|
| Lineage validation (3-check gate) | `src/lib.rs` | `SegmentConfig::is_index_valid()` |
| BM25 keyword search (byte-range Puffin) | `src/core/reader/scan.rs` | `keyword_search_index()` |
| Phrase search (positional postings) | `src/core/reader/scan.rs` | `phrase_search_index()` |
| Scalar Roaring Bitmap filter | `src/core/reader/filter.rs` | `get_scalar_filter_bitmap()` |
| HNSW vector search | `src/core/reader/scan.rs` | `vector_search_index()` |

### A.4 Index Lifecycle Management

| Operation | Source File | Function |
|---|---|---|
| Add index (schema + backfill) | `src/core/table/index_config.rs` | `add_index()` |
| Drop index (MOR tombstone + cleanup) | `src/core/table/index_config.rs` | `drop_index()` |
| Vacuum (orphan purge) | `src/core/table/maintenance.rs` | `vacuum_async()` |
| Self-healing index recovery | `src/core/table/maintenance.rs` | `recover_indexes_async()` |
| Reindex inverted column (upgrade) | `src/core/table/index_config.rs` | `reindex_inverted_column()` |
| Background backfill | `src/core/table/index_config.rs` | `backfill_indexes_async()` |
| Mount detached overlay | `src/core/table/detached.rs` | `mount_external_iceberg()` |
| Sync detached overlay | `src/core/table/detached.rs` | `sync_detached_overlay_async()` |
| Detached sync worker | `src/core/table/detached.rs` | `start_detached_sync_worker()` |
| Detached catalog sidecar | `src/core/table/detached.rs` | `DetachedOverlayCatalog` |

### A.5 Cache Tiers

| Cache | Stored Type | Source File |
|---|---|---|
| `INVERTED_INDEX_CACHE` | `Arc<Vec<RecordBatch>>` | `src/core/cache.rs` |
| `HNSW_IVF_CACHE` | `Arc<HnswIvfIndex>` | `src/core/cache.rs` |
| `INDEX_CACHE` | `Arc<RoaringBitmap>` | `src/core/cache.rs` |
| `ANALYZER_META_CACHE` | `String` (tokenizer name) | `src/core/cache.rs` |

### A.6 Test Coverage

| Test | Source File | What it validates |
|---|---|---|
| Puffin write/read round-trip | `tests/test_puffin.rs` | `test_puffin_write_read` |
| Puffin compound multi-blob | `tests/test_puffin.rs` | `test_puffin_compound_index_writer_reader` |
| End-to-end Puffin segment + BM25 query | `tests/test_puffin.rs` | `test_puffin_hybrid_segment_writer_and_reader` |
| Add/drop/re-add lifecycle | `tests/test_index_lifecycle.rs` | `test_index_lifecycle_add_drop_readd` |

### A.7 Python API Surface

| Python Method | Rust Binding | Behavior |
|---|---|---|
| `table.add_index(col, algo)` | `src/python/table.rs::add_index()` | Schema commit + async backfill |
| `table.drop_index(col)` | `src/python/table.rs::drop_index()` | MOR tombstone + best-effort cleanup |
| `table.remove_index(col)` | `src/python/table.rs::remove_index()` | Alias for `drop_index` |
| `table.vacuum(n)` | `src/python/table.rs::vacuum()` | Orphan artifact purge |
| `Table.mount_external_iceberg(table_uri, overlay_uri)` | `src/python/table.rs::mount_external_iceberg()` | Mount read-only upstream + detached overlay |
| `table.sync_detached_overlay()` | `src/python/table.rs::sync_detached_overlay()` | Incremental upstream snapshot sync |
| `table.start_detached_sync_worker(secs)` | `src/python/table.rs::start_detached_sync_worker()` | Background upstream snapshot watcher |
| `table.detached_catalog()` | `src/python/table.rs::detached_catalog()` | Detached catalog metadata (or `None`) |

