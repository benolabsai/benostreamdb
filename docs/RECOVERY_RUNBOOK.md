# BenoStreamDB Recovery Runbook

This document outlines the formal recovery procedures for BenoStreamDB, detailing RPO/RTO expectations and step-by-step resolution paths for catastrophic failures.

BenoStreamDB is designed around a strict separation of **authoritative state** (Iceberg snapshots, Parquet data files) and **derived state** (HNSW/BM25 indexes). This architecture ensures that most "database corruption" events are merely cache-invalidation events that can be resolved without data loss.

## Recovery Objectives

### RPO (Recovery Point Objective)
BenoStreamDB’s RPO depends entirely on your configured `WalDurability` setting:
- **`WalDurability::Sync`**: **RPO = 0**. Every acknowledged write is synchronously fsync'd to the Write-Ahead Log before the client receives an OK. No committed data is lost in a crash.
- **`WalDurability::Async`**: **RPO = < 1s** (configurable). Writes are buffered in memory and flushed periodically. A hard crash will lose the in-memory buffer (unacknowledged or acknowledged asynchronously).

### RTO (Recovery Time Objective)
- **Data Availability**: **Seconds**. On startup, BenoStreamDB reads the latest Iceberg manifest and replays the WAL. This process typically takes under a second. The data is immediately available for querying (falling back to scalar/vector scans if indexes are missing).
- **Performance Recovery (Index Rebuild)**: **Minutes to Hours**. If an index is lost or corrupted, queries will gracefully degrade to full-scans. Rebuilding the index is an asynchronous background process whose duration depends on dataset size.

---

## 1. Restore-from-Object-Store Procedure

Because Iceberg/Parquet in the object store is the authoritative source of truth, BenoStreamDB compute nodes are essentially stateless cache engines.

**Scenario:** The physical server hosting BenoStreamDB is destroyed.
**Action:**
1. Provision a new compute node.
2. Ensure the node has IAM/network access to the backing object store (S3/GCS/Azure).
3. Start the BenoStreamDB process pointing to the existing URI:
   ```rust
   let table = Table::new_async("s3://my-bucket/my-table").await?;
   ```
4. **Result:** The system will immediately read the latest `commit.lock` or catalog pointer, load the manifest, and resume serving reads and writes.

---

## 2. WAL Recovery Procedure

The Write-Ahead Log (`_wal/`) stores durable writes that have not yet been compacted into an Iceberg manifest. 

**Scenario:** The process crashes mid-write, corrupting the active WAL file. Upon restart, the database fails to initialize with a `WalCorruptionError`.
**Action:**
BenoStreamDB prioritizes data safety and will refuse to start if it detects torn or corrupted WAL batches. To forcibly recover availability at the cost of losing un-manifested writes:
1. Locate the WAL directory (default: `[table_uri]/_wal/`).
2. Identify the corrupted `.arrow` file (usually the one with the most recent timestamp).
3. Move the file out of the directory:
   ```bash
   mv s3://my-bucket/my-table/_wal/wal_0042.arrow /backup/corrupted_wal_0042.arrow
   ```
4. Restart the database.
5. **Result:** The DB will replay all valid WAL files and resume operation. Any writes contained in `wal_0042.arrow` are permanently lost.

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
3. **Result:** The system will serve vector similarity searches using exact nearest-neighbor (flat scan) until the HNSW build completes, at which point it will seamlessly swap to using the new index.

---

## 4. Catalog Recovery Procedure

If you are using a decoupled catalog (e.g., Nessie, REST, or Hive Metastore) to track table states instead of the built-in file-based `commit.lock`:

**Scenario:** The external catalog database (e.g., PostgreSQL backing Nessie) is irrecoverably corrupted.
**Action:**
Because BenoStreamDB writes valid Iceberg metadata files (`v1.metadata.json`, `v2.metadata.json`) into the `metadata/` directory of the object store, the table can be recovered even if the catalog is lost.
1. Inspect the `metadata/` directory in your object store.
2. Identify the highest numbered metadata file (e.g., `v42.metadata.json`).
3. Manually register this table in your new/restored catalog using the Iceberg REST API or catalog-specific CLI:
   ```json
   POST /v1/namespaces/default/register
   {
     "name": "my-table",
     "metadata-location": "s3://my-bucket/my-table/metadata/v42.metadata.json"
   }
   ```
4. **Result:** The new catalog will point to the exact transaction state of `v42`, and BenoStreamDB will resume operations normally.
