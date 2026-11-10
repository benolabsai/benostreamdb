# Concurrency and Atomic Commits

BenoStreamDB is designed for high-concurrency environments where multiple clients may be reading from and writing to the same table simultaneously.

## Optimistic Concurrency Control (OCC)

BenoStreamDB employs **Optimistic Concurrency Control** to ensure ACID compliance without the need for heavyweight central locks in most cases.

### Snapshot Versioning
Every table state is represented by a specific version of the manifest file (e.g., `_manifest/v100.json`). These files are immutable once written.

### The Commit Protocol
When a client (writer) wants to commit changes:
1.  **Read Latest**: The client reads the current latest version (e.g., `v100`).
2.  **Prepare**: The client calculates the new state (`v101`) based on the changes (e.g., added or removed segments).
3.  **Atomic Swap**: The client attempts to write the new manifest file `v101.json` using an **atomic "create-if-not-exists"** primitive.

### Conflict Resolution
If another client successfully committed `v101.json` while the first client was preparing its changes:
- The first client's write operation will fail with an `AlreadyExists` or conflict error.
- BenoStreamDB automatically **retries** the commit inside the `ManifestManager` / `TableMetadata` CAS loops (and up to 64 attempts on strict V3 row-lineage OCC conflicts, cleaning up obsolete staged Parquet files before retrying).
- In each retry, the client re-reads the *new* latest version, merges its changes again, and attempts to commit the *next* version (e.g., `v102`).
- Randomized backoff is used between retries to reduce contention. For concurrent writes buffered inside a single handle, commits transition explicitly through `PendingCommit` (`Staging -> ManifestCommitted -> IcebergPublished`); see [ADR-010](https://www.benostreamdb.org/guides/adr_010.html).

## Distributed Locking (`FileBasedLock`)

In addition to catalog-level atomicity, BenoStreamDB includes a built-in cloud-agnostic distributed lock mechanism (`FileBasedLock`):
- **Object Storage CAS**: Uses atomic conditional object creation (`PutMode::Create`) on S3, GCS, Azure Blob, and local filesystems.
- **Heartbeats & Expiration**: Automatically manages lock leases with configurable TTL and proactive heartbeat renewals.
- **Maintenance Locking**: Compaction and snapshot expiration acquire dedicated maintenance locks to prevent concurrent mutations while active queries proceed cleanly.

## Write-Ahead Log (WAL) Durability Modes

BenoStreamDB writes each incoming batch as a transaction-tagged Arrow IPC record (`benostream:tx_id`, `benostream:seq`) into rotated WAL segment files (`log_<timestamp>_<pid>_<seq>.arrow` under `BSDB_WAL_DIR` or `{table_uri}/_wal`). Before committing a snapshot, the active segment is flushed and sealed via `rotate_async()` so concurrent writes open a fresh segment, and `prune_committed_segments()` removes sealed segments only once every transaction ID in that file appears in the durable manifest (`benostream.committed_wal_tx`).

Two `WalDurability` policies are supported via `BSDB_WAL_DURABILITY` or `TableBuilder::with_durability(...)`:
- **`sync` (Default, `WalDurability::Sync`)**: Runs `fdatasync` (`File::sync_data`, falling back to `File::sync_all`) on the WAL segment before acknowledging the write to the caller (`RPO = 0` for acknowledged writes). Failed syncs propagate as write errors rather than false acknowledgements.
- **`async` (`WalDurability::Async`)**: Enqueues batches to the dedicated WAL worker channel and returns without waiting for disk sync. The worker flushes and syncs on either `BSDB_WAL_SYNC_BATCH_SIZE` (`10` batches) or `BSDB_WAL_SYNC_INTERVAL_MS` (`100` ms). Individual segments compact automatically above `BSDB_WAL_COMPACT_MB` (`1024` MiB).

## Read Isolation

Readers in BenoStreamDB always see a **consistent snapshot** of the table. Once a reader loads a particular version (e.g., `v100`), it will continue to see that state even if newer versions are committed by other clients. This provides **Snapshot Isolation**, which is ideal for long-running analytical queries.

## Query Execution Concurrency (I/O vs CPU)

BenoStreamDB decouples I/O stream concurrency from CPU compute parallelism:

1. **I/O Pipeline (`BSDB_MAX_CONCURRENCY`)**: Tokio asynchronously pulls up to $N$ Parquet segments concurrently via non-blocking streams (`buffer_unordered`). Non-blocking tasks spend the majority of their time awaiting network packets or NVMe block reads without consuming CPU cores.
2. **Compute Pipeline (`RAYON_NUM_THREADS`)**: As segment buffers arrive in memory, Rayon parallelizes CPU-intensive SIMD distance metrics, HNSW graph deserialization, and scalar filters across dedicated OS worker threads.

### Sizing & Tuning Matrix

| Deployment Profile | Machine Specs | `RAYON_NUM_THREADS` (Compute) | `BSDB_MAX_CONCURRENCY` (I/O) | Focus |
|:---|:---|:---|:---|:---|
| **Benchmark / Container** | 4 Cores, 4 GB RAM | `2` – `4` | `4` – `8` | Low memory footprint, zero cache thrashing. |
| **Production Server** | 16 Cores, 32 GB RAM | `8` – `14` | `16` – `32` | High-throughput mixed analytics and hybrid search. |
| **Cloud Object Store (S3/GCS)** | 8 Cores, 32 GB RAM | `6` | `32` – `48` | High I/O concurrency to hide cloud network latency. |

