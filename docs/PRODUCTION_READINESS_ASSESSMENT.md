# BenoStreamDB — Release Readiness Assessment
Date: 2026-10-03 (Updated for v0.12.0 Developer Preview)
Scope: Architectural audit of core storage invariants, differential oracles, crash recovery, and security against the **Developer Preview Release Bar**.
Current Version: 0.12.0

Status: **Developer Preview Ready** (All core integrity gates passed; enterprise HA/clustering deferred).

> **Release Scope Notice:**  
> For release v0.12.0, BenoStreamDB is evaluated against a **Developer Preview release standard** rather than an enterprise production service standard. The evaluation verifies that developers can install, query, and evaluate the engine without risk of silent data corruption or unhandled write loss. Enterprise operational controls (multi-node automatic failover, distributed cluster consensus, high-availability SLAs) are explicitly deferred to post-v1.0.

This assessment summarizes the verification of core invariants (WS1-WS5) and the end-to-end benchmark suite.

---

## 1. Overall Verdict

| Area | Status (Post-Benchmarks) |
|---|---|
| Core storage architecture | **Proven** |
| Index architecture (overlay) | **Proven** (differential oracles pass perfectly) |
| Iceberg integration | **Proven** (100% core V2/V3 compliant via PyIceberg testing) |
| Query engine | **Proven** (matches DuckDB/DataFusion row counts; competitive performance) |
| Rust implementation | **Proven** |
| Testing infrastructure | **Hardened** (WS1-WS5 differential harnesses and soak tests active) |
| Crash/recovery confidence | **High** (survives randomized WS2 crash-injection sweep) |
| Multi-writer production deployment | **Ready** (survives WS3 concurrency harness with object-store faults) |
| Security | **Shipped** (API key + JWT auth on search gateway and Flight SQL; OAuth2 for catalogs; RBAC roles in JWT claims) |
| Operational tooling | Progressing (observability plan in motion) |
| **Overall** | **Developer Preview Ready** |

### ✅ Resolved Blocker: `lastfm-64-dot` Vector Recall Anomaly
During the execution of the end-to-end benchmark suite, an anomaly was discovered where vector recall dropped to **~6%** specifically on the `lastfm-64-dot` dataset for `hnsw_tq8`. 
- **Root Cause:** The MIPS dataset `lastfm-64-dot` contains padded training vectors with a uniform small norm (~0.7256), whereas the query vectors possessed wildly varying, unpadded natural norms (up to ~31.23). The query vectors severely overflowed the `TurboQuant` encoder's dynamic range during SDC, clamping their values to zero or max, thus destroying vector directionality.
- **Resolution:** A pre-quantization dynamic scaling step was introduced for `InnerProduct` and `Cosine` metrics in `HnswGraph`. Because query magnitude does not alter metric rankings, the query is perfectly scaled into the quantizer's `offset` and `scale` window, completely avoiding overflow and maximizing 8-bit precision.

---

## 2. Core Invariants Verified

The following core invariants have been tested and proven under adversarial conditions:
1. **Overlay Invariant** — indexes are derived, reconstructible state.
2. **Publication Invariant** — a manifest references only durable artifacts.
3. **Durability Invariant** — WAL truncation only after a committed snapshot.
4. **Maintenance Invariant** — delete only if not referenced / not in-flight.
5. **Resource Invariant** — memory-scaled build concurrency; no-panic policy.

---

## 3. Workstream Execution Summary

All five workstreams testing the storage/transaction/recovery boundaries have been successfully executed:

### WS1: Differential indexed-vs-full-scan oracle (Complete)
**Goal:** Prove `indexed(query) == full_scan(query)` on the same snapshot.
- **Findings & Fixes:**
  - `explain()` misreported the vector access path for TurboQuant indexes by failing to match the file pattern; resolved by reading from the manifest instead of globbing.
  - A BM25 index silently dropped equality matches by inappropriately applying tokenized indexes to exact-match filters; resolved by restricting exact matches to the `identity` analyzer.

### WS2: Crash-injection matrix at every boundary (Complete)
**Goal:** Kill the process at boundaries (WAL/manifest/index) and prove deterministic recovery.
- **Findings & Fixes:**
  - WAL replay was not idempotent. A crash between manifest commit and WAL truncation caused duplicate rows on recovery. Fixed by recording committed WAL transaction IDs in the manifest properties (`benostream.committed_wal_tx`), allowing recovery to skip already-committed records.

### WS3: Multi-writer / object-store concurrency (Complete)
**Goal:** Prove OCC + distributed-locking design is safe with N concurrent writers against shared object storage.
- **Findings & Fixes:**
  - Vacuum deleted all data files due to empty `Manifest.entries` for tiered manifests; fixed by resolving entries correctly via `load_all_entries`.
  - GC-vs-writer race caused new files to be deleted; fixed by re-validating the live set before deletion and adding a grace period.
  - Partition-spec updates dropped data by dropping `manifest_list_path`; fixed to preserve it.
  - Delete files were incorrectly written to the local FS for S3 stores; fixed to write to the `ObjectStore` relative to the table root.

### WS4: Iceberg interoperability / conformance (Complete)
**Goal:** Validate "100% of core V2/V3" claim via independent implementations (PyIceberg).
- **Findings & Fixes:**
  - PyIceberg validation uncovered issues with relative paths in `manifest-list`, `manifest_path`, and `file_path`. Fixed to use absolute URIs on write and relativize on read.
  - Avro schemas were missing spec-required metadata (`field-id`, map `logicalType`, list `element-id`). Added to schemas.

### WS5: Long-duration GC / compaction / delete / recovery (Complete)
**Goal:** Prove the Maintenance Invariant under sustained churn.
- **Findings & Fixes:**
  - Compaction resurrected deleted rows because it didn't pass delete files into the reader; fixed to apply the delete mask during compaction.
  - Compaction misclassified index sidecars as data files; fixed by strictly matching the data parquet file name.
  - Read-after-delete hung due to unbounded per-entry delete file lists; fixed by storing delete files once partition-scoped in the manifest.
  - Vector recall miss due to querying during a background index build; fixed test-side by awaiting background tasks.
  - Append-only commits rewrote the entire manifest list; optimized to only write new entries and reference unchanged prior manifests, falling back to consolidation after a threshold.

---

## 4. Security Posture

Authentication is implemented as a shared, stateless module (`src/core/auth.rs`)
used by both the search gateway and the Flight SQL server:

- **API key** — constant-time comparison against `BSDB_API_KEY`.
- **JWT** — HS256 (`BSDB_JWT_SECRET`) or RS256 (`BSDB_JWT_PUBLIC_KEY`); claims
  extraction for subject, roles, audience, issuer, and expiry.
- **RBAC** — roles extracted from JWT claims (`roles` array); available to
  authorization logic.
- **Search gateway** (`contrib/benostreamdb-search/src/auth.rs`) — Axum
  middleware accepting `Authorization: Bearer <jwt>`, `Authorization: ApiKey
  <key>`, and `api-key: <key>` (Qdrant convention). Public-path exemption for
  health checks; fail-closed when `required()` is set.
- **Flight SQL** (`server/flight_sql/src/main.rs`) — Tonic interceptor using the
  same `AuthConfig`; `authorization` metadata header.
- **Fail-closed on network exposure** — both servers refuse to bind to a
  non-loopback address without credentials (`process::exit(1)`).
- **Catalog authentication** — OAuth2 client-credentials grant with token
  caching and expiry (`src/core/catalog/rest.rs`); Bearer token support; SSRF
  guard on catalog URLs (`src/core/catalog/mod.rs`).
- **SQL session hardening** — `ALLOWED_SESSION_SETTINGS` prevents SQL `SET`
  from rewriting environment variables (credentials, endpoints).

Remaining gaps: no built-in user database (identity is delegated to an external
IdP or a shared secret); no per-table or per-column access control; no audit
logging of authentication events.

## 5. Final Exit Criteria

With the completion of WS1-WS5 and the resolution of the quantization boundary anomaly, the core storage engine has survived millions of operations (insert/update/delete/predicate/vector/snapshot/commit/compaction/index-rebuild) with injected failures without divergence between indexed and full-scan execution.

The system meets the exit criteria for a **Developer Preview** (pending real-world usage and operational tooling maturity).
