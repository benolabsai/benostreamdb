# Production Crash Injection & Recovery Benchmark

- **Engine**: BenoStreamDB (Apache Iceberg native)
- **Failure Model**: Process aborted at named write/WAL/manifest boundaries (modeling SIGKILL)
- **Pre-Crash Seed Rows**: 1000
- **In-Flight Batch Rows**: 500

| Injection Boundary | Operation Result | Rows Visible | Recovery Time (ms) | Atomicity | Zero Data Loss | Status |
|---|---|---|---|---|---|---|
| `wal_append` | Clean Rollback | 1000 | **0.70 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `wal_flush` | WAL Replayed | 1500 | **0.59 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `wal_truncate` | WAL Replayed | 1500 | **1.17 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `data_upload` | WAL Replayed | 1500 | **0.55 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `index_upload` | Committed | 1500 | **0.86 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `manifest_commit` | WAL Replayed | 1500 | **0.55 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `manifest_visible` | WAL Replayed | 1500 | **0.89 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `compaction_start` | Committed | 1500 | **0.97 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `compaction_manifest_swap` | Committed | 1500 | **0.87 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `vacuum_start` | Committed | 1500 | **0.94 ms** | ✅ Yes | ✅ Yes | ✅ PASS |
| `vacuum_delete` | Committed | 1500 | **0.88 ms** | ✅ Yes | ✅ Yes | ✅ PASS |

### Recovery Invariants Verified
- **Atomicity**: Re-opened table always observes either the pre-crash snapshot or the post-commit snapshot, never a torn state.
- **Idempotency**: Replaying WAL segments never duplicates already-committed rows.
- **Recovery Speed**: Average time-to-first-read after crash is sub-5ms across all failure points.
