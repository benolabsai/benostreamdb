# BenoStreamDB Recovery Runbook

This document outlines the formal recovery procedures for BenoStreamDB, detailing RPO/RTO expectations and step-by-step resolution paths for catastrophic failures.

BenoStreamDB is designed around a strict separation of **authoritative state** (Iceberg snapshots, Parquet data files) and **derived state** (HNSW/BM25 indexes). This architecture ensures that most "database corruption" events are merely cache-invalidation events that can be resolved without data loss.

## Recovery Objectives

### RPO (Recovery Point Objective)
BenoStreamDB’s RPO depends entirely on your configured `WalDurability` setting (`BSDB_WAL_DURABILITY` / `TableBuilder::with_durability`):
- **`WalDurability::Sync` (`"sync"`, Default)**: **RPO = 0**. Every acknowledged write waits for `fdatasync` (`File::sync_data`, falling back to `File::sync_all`) on the WAL segment before the caller receives `Ok`. Failed syncs return an error to the caller. Note: for `s3://`, `gs://`, or `az://` tables, configure a persistent local volume via `BSDB_WAL_DIR` or `TableBuilder::with_wal_dir()`; otherwise the WAL defaults to temporary local storage (`/tmp/benostream_wal/...`) and logs a startup warning.
- **`WalDurability::Async` (`"async"`)**: **RPO <= sync window**. Batches are handed to the WAL worker channel without blocking for `fdatasync`; the worker syncs every `BSDB_WAL_SYNC_BATCH_SIZE` (`10` batches) or `BSDB_WAL_SYNC_INTERVAL_MS` (`100` ms). Writes buffered inside the un-synced window are lost on hard termination.

### RTO (Recovery Time Objective)
- **Data Availability**: **Seconds**. On startup, BenoStreamDB reads the latest manifest (`_manifest/v{N}.json`), replays uncommitted records from the segmented WAL while skipping any `benostream:tx_id` already present in the manifest property `benostream.committed_wal_tx`, and immediately resumes queries (falling back to Parquet scans while secondary indexes rebuild).
- **Performance Recovery (Index Rebuild)**: **Minutes to Hours**. If an index is lost or corrupted, queries degrade deterministically to Parquet/exact scans. Rebuilding the index executes asynchronously without blocking ingestion.

---

## 1. Restore-from-Object-Store Procedure

Because Iceberg/Parquet in the object store is the authoritative source of truth, BenoStreamDB compute nodes are essentially stateless cache engines.

**Scenario:** The physical server hosting BenoStreamDB is destroyed.
**Action:**
1. Provision a new compute node.
2. Ensure the node has IAM/network access to the backing object store (S3/GCS/Azure).
3. Start the BenoStreamDB process pointing to the existing URI (with `BSDB_WAL_DIR` pointing to persistent local WAL storage if present):
   ```rust
   let table = Table::new_async("s3://my-bucket/my-table").await?;
   ```
4. **Result:** The system loads the latest atomic manifest (`_manifest/v{N}.json` via `load_latest_direct`) and Iceberg metadata pointer, replays any uncommitted WAL segments, and resumes reads and writes.

---

## 2. WAL Recovery Procedure

The Write-Ahead Log (`_wal/` by default for local tables, or `BSDB_WAL_DIR` for remote URIs) stores durable writes in rotated Arrow IPC stream segments named `log_<timestamp>_<pid>_<seq>.arrow` until the transaction commits to a manifest and watermark pruning removes the sealed file (`prune_committed_segments`).

### Automatic Startup Recovery (Standard Crashes & Torn Tails)

In most crash scenarios (`SIGKILL`, power loss mid-append, or crash between manifest commit and WAL cleanup), no manual action is required:
- **Torn Trailing Frame / Interrupted Header:** On startup, `open_wal_segment` and `SafeIpcSegmentIter` automatically recover from 0-byte segments, zeroed preallocation headers (`<= 8` bytes), truncated Arrow IPC continuation/schema headers at EOF (`Empty`), and an `UnexpectedEof` mid-way through a trailing uncommitted batch frame (logging a `warn` and preserving every complete synced prefix batch).
- **Idempotent Crash Replay:** If the process exited after committing `_manifest/v{N}.json` (`CommitPhase::ManifestCommitted`) but before watermark pruning deleted the sealed `.arrow` segment, startup filters out every batch whose `benostream:tx_id` already appears in `benostream.committed_wal_tx` and registers the segment for pruning on the next commit.

### Manual Quarantine Procedure (Mid-Stream Byte Corruption)

**Scenario:** Storage hardware corruption damages bytes inside a WAL header (non-zero wrong magic bytes, invalid metadata length) or mid-stream FlatBuffers payload, causing startup to refuse initialization with a `WAL Recovery Error`.

**Action:**
BenoStreamDB refuses silent data truncation when mid-stream corruption is detected inside a non-EOF record. To restore table availability by quarantining only the damaged segment:
1. Inspect startup logs for the failing segment path (under `[table_uri]/_wal/` or `$BSDB_WAL_DIR`).
2. Inspect the directory (`ls -la` sorted chronologically; filenames encode nanosecond timestamps and sequence numbers). Keep all undamaged segments in place.
3. Quarantine only the failing segment outside the WAL directory:
   ```bash
   mkdir -p /var/backups/bsdb-wal-quarantine
   mv /path/to/table/_wal/log_0001762780000000000_004210_00000003.arrow /var/backups/bsdb-wal-quarantine/
   ```
4. Restart the process.
5. **Result:** Startup replays all valid WAL segments, skips already-committed transactions idempotently, and resumes serving traffic. Only uncommitted batches contained after the corruption point in the quarantined segment are absent.

---

## 3. Index Rebuild Procedure

Indexes (HNSW, BM25, Bloom filters) are non-authoritative. If they become corrupted, out-of-sync, or are accidentally deleted, they can be regenerated directly from the Parquet data files.

**Scenario:** Vector search latency spikes because the HNSW index was partially deleted, or logs report `IndexChecksumMismatch`.
**Action:**
1. Drop the corrupted index. This removes the metadata pointer and deletes the corrupted `.bin` files:
   ```rust
   table.drop_index("embedding").await?;
   ```
2. Re-create the index. The database will launch a background task to rebuild it from the authoritative Iceberg data:
   ```rust
   table.add_index("embedding".to_string(), IndexAlgorithm::HNSW).await?;
   ```
3. **Result:** The system serves vector similarity queries using exact nearest-neighbor scans until the background HNSW build attaches to the manifest, then switches queries onto the rebuilt HNSW index automatically.

---

## 4. Catalog Recovery Procedure

If you are using a decoupled catalog (e.g., Nessie, REST, or Hive Metastore) to track table states instead of the built-in file-based `commit.lock`:

**Scenario:** The external catalog database (e.g., PostgreSQL backing Nessie) is irrecoverably corrupted.
**Action:**
Because BenoStreamDB writes valid Iceberg metadata files (`v1.metadata.json`, `v2.metadata.json`) into the `metadata/` directory of the object store, the table can be recovered even if the catalog is lost.
1. Inspect the `metadata/` directory in your object store.
2. Identify the highest numbered metadata file (e.g., `v42.metadata.json`).
3. Manually register this table in your new/restored catalog using the Iceberg REST API or catalog-specific CLI:
   ```http
   POST /v1/namespaces/default/register HTTP/1.1
   Content-Type: application/json

   {
     "name": "my-table",
     "metadata-location": "s3://my-bucket/my-table/metadata/v42.metadata.json"
   }
   ```
4. **Result:** The new catalog will point to the exact transaction state of `v42`, and BenoStreamDB will resume operations normally.
