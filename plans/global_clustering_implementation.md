# Implementation Plan: Cluster-Keyed Index Artifact

**Status:** Implementation plan
**Date:** November 2026
**Design:** [`global_clustering_compaction.md`](plans/global_clustering_compaction.md)
**Evidence:** [`COHERE_10M_BOTTLENECK_ANALYSIS.md`](file:///home/ralbright/projects/benolabs/patent-proposal/paper/COHERE_10M_BOTTLENECK_ANALYSIS.md) § 6

---

## 0. The Shape of the Change

Today the index is **per-segment**: `ManifestEntry.index_files` holds a Puffin sidecar owned by one Parquet file, and `search_targeted_clusters` returns `(usize, f32)` — a *segment-local* row index.

The target is a **table-level** artifact: one Puffin file, not owned by any data file, keyed by global cluster, whose nodes carry `(segment_id, row_id)` so the existing fetch path resolves them.

The single most important interface fact: [`fetch_results_by_id`](src/core/table/read.rs:1523) already consumes `Vec<ScoredResult>` with `segment_id`, `row_id`, `score`. **If the new index returns `Vec<ScoredResult>`, the entire hydration path is reused unchanged.** That is the seam to build against.

---

## Phase 0 — Baseline (no code, blocking)

**Why blocking:** every later phase is judged against a recall/QPS curve, and `/data/cohere` is gone.

1. Re-download the Cohere 10M corpus and `gt_100.fbin` (`benchmarks/download_cohere.py` exists).
2. Run the existing harness to get the baseline curve: `per_segment`, `n_probe ∈ {4,8,16,32,48}`, `ef ∈ {32,64,128,256}`.
3. Record recall@10 and QPS at each point. This is the reference for every later comparison.

**Acceptance:** a table of (n_probe, ef) → (recall, QPS) for the current engine.

---

## Phase 1 — Global Centroids, Per-Segment Artifacts (validation only)

**Goal:** prove or disprove that co-location is what matters, before building the artifact.

**Change:** let the build accept precomputed centroids instead of running per-segment k-means.

| File | Change |
|---|---|
| [`hnsw_ivf.rs`](src/core/index/hnsw_ivf.rs:1410) | Add `build_with_centroids(vectors, metric, centroids: Vec<Vec<f32>>, hnsw_m, algo, offset, seed)`. The existing `build_with_seed` already computes `centroids` at line 1471 and `labels` at 1480 — factor those two lines so the caller can supply them. |
| [`hnsw_ivf.rs`](src/core/index/hnsw_ivf.rs:850) | The `n_lists` derivation stays for the default path; the new entry point bypasses it. |
| New: `src/core/index/global_centroids.rs` | `train_global(vectors_sample, k, seed) -> Vec<Vec<f32>>`, warm-started from the concatenated per-segment centroids. Reuse `simple_kmeans_with_seed` and `gpu::compute_kmeans_assignment`. |
| [`maintenance.rs`](src/core/table/maintenance.rs:48) | `rebuild_indexes_with_global_centroids()` — sample across segments, train once, pass to every segment's build. |

**Expected result:** no improvement, possibly a regression, because each global cluster is split across 25 segments (partial clusters, ~0.23 ms overhead each).

**Acceptance:** measured curve. If it *does* improve, the co-location theory is wrong and Phase 2 should be reconsidered before it is built.

**Effort:** 1–2 days.

---

## Phase 2 — Table-Level Artifact (the real work)

### 2.1 Manifest schema

| File | Change |
|---|---|
| [`types.rs`](src/core/manifest/types.rs:13) | Add `pub struct TableIndexFile` — same fields as `IndexFile` minus the per-entry lineage bindings, plus `cluster_count: usize` and `dim: usize`. |
| [`types.rs`](src/core/manifest/types.rs:1421) | Add `#[serde(default)] pub table_indexes: Vec<TableIndexFile>` to `Manifest`. Default-empty keeps old manifests readable. |

`#[serde(default)]` is what makes this a non-breaking change — existing tables load with an empty list and fall back to the per-segment path.

### 2.2 Artifact format

| File | Change |
|---|---|
| [`puffin.rs`](src/core/puffin.rs:44) | Graph blobs use a single quantization-agnostic `PUFFIN_BLOB_VECTOR_HNSW_GRAPH` label; the quantizer is carried by the centroids `quantizer_config`. |
| New: `src/core/index/cluster_index.rs` | The artifact. Layout per cluster: `centroid (f32×dim)`, `codes (u8×dim × n)`, `adjacency (u32×M × n)`, `node_meta ((u16 segment_idx, u32 row_id) × n)`, plus a `segment_table: Vec<String>` mapping `segment_idx → segment_id`. |

`node_meta` is the only genuinely new structure. Everything else is the existing HNSW + quantizer output.

### 2.3 Build path

| File | Change |
|---|---|
| `cluster_index.rs` | `build_from_table(store, manifest, column, algo, k, m, ef_c) -> ClusterIndex`: one sequential pass over all Parquet files, quantize, assign to global cluster, build per-cluster HNSW, record `(segment_idx, row_id)` per node. |
| `cluster_index.rs` | `save(store, path)` / `load(store, path)` via the existing Puffin writer/reader. |
| [`maintenance.rs`](src/core/table/maintenance.rs:48) | `rebuild_cluster_index(column, k, m, ef_c)` — orchestrates train → build → save → commit manifest. |

**Memory bound:** build one cluster at a time and stream it out. Do not hold 10M codes in RAM; the whole point of the exercise is bounded working set.

### 2.4 Search path

| File | Change |
|---|---|
| `cluster_index.rs` | `search(query, k, n_probe, ef_search) -> Vec<ScoredResult>`: coarse-scan the global centroids, take top `n_probe`, traverse those cluster graphs, map `node_meta` → `(segment_id, row_id)`, return `ScoredResult`. |
| [`read.rs`](src/core/table/read.rs:480) | In the vector-filter block, prefer the table-level index when present: if `manifest.table_indexes` has a matching column, call `cluster_index.search(...)` and skip the per-segment router entirely. |
| [`read.rs`](src/core/table/read.rs:1523) | **No change.** `fetch_results_by_id` already groups by `segment_id` and reads by `row_id`. |

This is the payoff: the query path becomes one index call plus the existing hydration, with no per-segment fan-out and no `targeted_clusters` plumbing.

### 2.5 Loading and lifecycle

| File | Change |
|---|---|
| [`preload.rs`](src/core/table/preload.rs:428) | `ensure_cluster_index(column)` — mirror `ensure_vector_router`, cache in `TableState` alongside `vector_routers`. |
| [`state.rs`](src/core/table/state.rs:55) | Add `cluster_indexes: Arc<RwLock<HashMap<String, Arc<ClusterIndex>>>>`. |
| [`table.rs`](src/python/table.rs:442) | Expose `rebuild_cluster_index(column, k=1024, m=16, ef_construction=200)`. |

**Acceptance:** recall/QPS curve at equal recall vs Phase 0, plus a cold-cache run (`echo 3 > /proc/sys/vm/drop_caches`) to quantify the scattered-fetch cost from § 6.2 of the design note.

**Effort:** 1–2 weeks.

---

## Phase 3 — Delta Artifact (incremental ingestion)

Only after Phase 2 shows a real win.

| File | Change |
|---|---|
| `cluster_index.rs` | `DeltaIndex`: rows ingested after the last rebuild, assigned to the nearest global centroid, stored as a small per-cluster append-only artifact. |
| `cluster_index.rs` | `search` merges base + delta results before returning. |
| [`maintenance.rs`](src/core/table/maintenance.rs:48) | Rebuild folds delta into base and clears it. |

**Acceptance:** ingest 100k rows after a rebuild, confirm recall does not degrade and the delta is folded on the next rebuild.

**Effort:** 3–5 days.

---

## Ordering and Risk

| Phase | Effort | Risk | Decision it informs |
|---|---|---|---|
| 0 Baseline | 0.5 day | Low | Everything |
| 1 Global centroids | 1–2 days | Low | Whether co-location is the real cause |
| 2 Table-level artifact | 1–2 weeks | **Medium-high** — new manifest field, new artifact, new search path | Whether the 3–6× projection holds |
| 3 Delta | 3–5 days | Medium | Whether it survives continuous ingestion |

**Do Phase 1 before Phase 2.** It is two days and it either confirms the theory or saves two weeks. The failure mode to avoid is building the artifact, finding no improvement, and not knowing whether the theory was wrong or the implementation was.

**The riskiest single item is the manifest change.** `IndexFile` is currently per-`ManifestEntry` and the lineage bindings (`snapshot_id`, etc.) are load-bearing for the tightened-overlay contract in [`ADR_001`](docs/architecture/ADR_001_TIGHTENED_INDEX_OVERLAY_CONTRACT.md). A table-level index has no single snapshot to bind to, so the contract needs an explicit answer: bind to the manifest version, and invalidate the artifact when any data file changes. That decision should be written down before the code, not discovered during review.

---

## What Not to Do

- **Do not rewrite Parquet.** The design note's first draft proposed it; it is 4× the I/O and breaks time-range scans for no benefit.
- **Do not add a `vector_cluster` strategy to `ClusteringOptions`.** That path rewrites data. The cluster artifact is a separate concern.
- **Do not ship `BSDB_ROUTING_MODE=global` as a default.** Measured at equal cluster count it is within 0.4 % of `per_segment`; its apparent speedup is a recall tradeoff.
- **Do not claim the win before Phase 2's equal-recall measurement.** The 3–6× figure is a projection from synthetic queries with no ground truth.
