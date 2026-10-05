# Production Crash Injection & Recovery Benchmark

- **Engine**: BenoStreamDB (Apache Iceberg native)
- **Failure Model**: Process aborted at named write/WAL/manifest boundaries (modeling SIGKILL)
- **Pre-Crash Seed Rows**: 1000
- **In-Flight Batch Rows**: 500

| Injection Boundary | Operation Result | Rows Visible | Recovery Time (ms) | Atomicity | Zero Data Loss | Status |
|---|---|---|---|---|---|---|
| `wal_append` | Clean Rollback | 1000 | **1.82 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `wal_flush` | WAL Replayed | 1500 | **1.42 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `wal_truncate` | WAL Replayed | 1500 | **3.66 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `data_upload` | WAL Replayed | 1500 | **1.41 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `index_upload` | Committed | 1500 | **3.32 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `manifest_commit` | WAL Replayed | 1500 | **1.32 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `manifest_visible` | WAL Replayed | 1500 | **3.47 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `compaction_start` | Committed | 1500 | **3.30 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `compaction_manifest_swap` | Committed | 1500 | **3.23 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `vacuum_start` | Committed | 1500 | **3.54 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `vacuum_delete` | Committed | 1500 | **3.29 ms** | ✅ Yes | ✅ Yes | ✅ PASS |

### Recovery Invariants Verified
- **Atomicity**: Re-opened table always observes either the pre-crash snapshot or the post-commit snapshot, never a torn state.
- **Idempotency**: Replaying WAL segments never duplicates already-committed rows.
- **Recovery Speed**: Average time-to-first-read after crash is sub-5ms across all failure points.
