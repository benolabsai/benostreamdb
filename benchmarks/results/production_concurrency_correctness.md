# Multi-Writer Concurrency Correctness Benchmark (§7.7)

- **Engine**: BenoStreamDB (OCC Distributed Locking & Multi-Writer Table Engine)
- **Workload**: Multi-writer concurrent commits, simultaneous background compaction, and position deletes on shared object storage
- **Invariants Enforced**: Zero lost updates, zero duplicate rows, zero torn snapshots, exact mathematical row counts, full key content integrity

| Scenario | Concurrent Threads | Total Ops | Throughput (Ops/sec) | Duration (ms) | Expected vs Actual Rows | Lost Rows | Duplicates | Torn Reads | Row Content Integrity | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **C1.2 (Concurrent Writers: 2)** | 2 | 20 | **806.2** | 24.81 ms | 101 / 101 | 0 | 0 | 0 | ✅ 100% Match | ✅ PASS |
| **C1.4 (Concurrent Writers: 4)** | 4 | 40 | **629.6** | 63.53 ms | 201 / 201 | 0 | 0 | 0 | ✅ 100% Match | ✅ PASS |
| **C1.8 (Concurrent Writers: 8)** | 8 | 80 | **501.3** | 159.57 ms | 401 / 401 | 0 | 0 | 0 | ✅ 100% Match | ✅ PASS |
| **C1.16 (Concurrent Writers: 16)** | 16 | 160 | **657.3** | 243.43 ms | 801 / 801 | 0 | 0 | 0 | ✅ 100% Match | ✅ PASS |
| **C2 (Inserts + Concurrent Compaction)** | 9 | 80 | **533.3** | 150.02 ms | 401 / 401 | 0 | 0 | 0 | ✅ 100% Match | ✅ PASS |
| **C3 (Inserts + Position Deletes)** | 8 | 80 | **241.6** | 331.19 ms | 337 / 337 | 0 | 0 | 0 | ✅ 100% Match | ✅ PASS |
| **C4 (Readers + Writers Isolation)** | 8 | 160 | **1194.8** | 133.91 ms | 410 / 410 | 0 | 0 | 0 | ✅ 100% Match | ✅ PASS |

### Concurrency Invariants Verified
- **Row Content & Key Integrity**: Unlike superficial row-count checks, every scenario reads the full primary key space and asserts 100% mathematical set equivalence against the expected set of active keys with zero lost keys, zero duplicate keys, and zero resurrected deleted keys.
- **Zero Lost Updates Under Contention**: Even with 16 simultaneous writers competing for the active manifest, 100% of rows are durable through OCC rebase and commit retries.
- **Compaction Concurrency Safety**: Manifest swaps during data file rewriting never delete in-flight writes or produce duplicate row versions.
- **Snapshot Isolation**: Readers observing the table during concurrent write and delete surges observe monotonic, atomic committed snapshots without seeing partial state.
