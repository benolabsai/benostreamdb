# Global Vector Clustering via Index-Only Compaction

**Status:** Design note (revised)
**Date:** November 2026
**Context:** [`COHERE_10M_BOTTLENECK_ANALYSIS.md`](file:///home/ralbright/projects/benolabs/patent-proposal/paper/COHERE_10M_BOTTLENECK_ANALYSIS.md) § 6.4

---

## 1. The Question

Should compaction run after ingestion to get global vector clustering?

**Yes — but only the index needs compacting, not the data.** The first draft of this note proposed rewriting the Parquet files. That is unnecessary and strictly worse. The index artifact can be reorganized by global cluster while the data files stay in arrival order, because a graph node already carries `(segment_id, row_id)` and the fetch path resolves rows by segment ([`fetch_results_by_id`](src/core/table/read.rs:1523)).

| Approach | Bytes rewritten | Time-range / id-range scans | New code |
|---|---:|---|---|
| Data compaction (first draft) | 41.7 GB (30.97 Parquet + 10.75 Puffin) | **Broken** — rows sorted by cluster | Reuses `ClusteringOptions` |
| **Index-only compaction (this revision)** | **10.75 GB (Puffin only)** | **Preserved** — data untouched | New table-level artifact type |

4× cheaper, and it does not regress the other access patterns. This is the right design.

---

## 2. Why the Index Must Be Reorganized (and the Data Need Not Be)

Today each segment trains its own k-means over its own 400,000 rows ([`hnsw_ivf.rs`](src/core/index/hnsw_ivf.rs:850), `n_lists = sqrt(n)/5` → 126 clusters). The Puffin sidecar for segment $S$ contains only the members of cluster $C$ that happen to live in $S$.

If you train global centroids but leave the sidecars per-segment, every sidecar still holds partial clusters. A probe for global cluster $C$ must visit all 25 segments to collect $C$'s 3,175 members — 25 partial reads instead of one whole read. The router would map one global cluster to 25 `(segment, local_cluster)` pairs.

**Partial clusters are penalized by per-cluster overhead, not by bytes.** The measured cost is ~0.23 ms per cluster read (8 clusters → 21.6 ms, 200 → 66.3 ms). Sixteen global clusters split across 25 segments is 400 partial reads ≈ 92 ms — worse than today's 200 whole clusters at 66 ms, even though the bytes are identical. So the cluster must be *whole* in the index.

**But "whole in the index" does not require "contiguous in the data."** A graph node already carries `(segment_id, row_id)`, and [`fetch_results_by_id`](src/core/table/read.rs:1523) groups results by segment and reads rows by local id. The index can therefore hold a cluster whose members live in 25 different Parquet files, and the final payload fetch still resolves correctly.

So the sequence is: **global k-means → assign every row a cluster id → rewrite the index artifact keyed by cluster (nodes keep their `(segment_id, row_id)`) → rebuild the router from global centroids.** The Parquet files are never touched.

---

## 3. What Already Exists

| Piece | Location | State |
|---|---|---|
| `rewrite_data_files` / `compact` | [`maintenance.rs`](src/core/table/maintenance.rs:48) | Implemented |
| `CompactionOptions` | [`compaction.rs`](src/core/compaction.rs:25) | Implemented |
| `ClusteringOptions { columns, strategy }` | [`compaction.rs`](src/core/compaction.rs:19) | Implemented — `zorder` \| `hilbert` |
| `compact_after` on ingest | [`ingest.rs`](src/core/table/ingest.rs:43) | Implemented |
| GPU k-means assignment | `crate::core::index::gpu::compute_kmeans_assignment` | Implemented |
| Per-segment k-means | [`hnsw_ivf.rs`](src/core/index/hnsw_ivf.rs:872) | Implemented |

**The gap:** `ClusteringOptions.columns` is `Vec<String>` of scalar columns, and `rewrite_data_files` rewrites *data*. Neither is what we want. What is missing is a **table-level index artifact** — a Puffin file not owned by any single segment, keyed by global cluster, whose nodes carry `(segment_id, row_id)`.

That is a manifest schema addition (`IndexFile` currently hangs off `ManifestEntry`, i.e. per data file), not a new storage format. The Puffin container, the code layout, and the graph serialization are all reusable.

---

## 4. Proposed Pipeline

```
ingest (append-only, arrival order)          <- unchanged
   │
   ├─ 1. sample 1–2% of rows across all segments
   ├─ 2. global k-means  ->  K centroids (K ~ 1024)
   ├─ 3. assign every row -> global cluster id  (GPU path)
   ├─ 4. build cluster-keyed index artifact:
   │        for each cluster C:
   │          codes  = quantize(rows in C)          <- read from Parquet
   │          graph  = HNSW over those codes
   │          nodes  = (segment_id, row_id) pairs   <- already carried
   ├─ 5. write ONE table-level Puffin artifact (or a small set)
   ├─ 6. register it in the manifest as a table-level index, not per-segment
   └─ 7. rebuild TableVectorRouter from the global centroids
```

Steps 1–3 are new. Step 4 is a new build path but reuses the existing quantizer and HNSW builder. Steps 5–7 are manifest and router plumbing.

**The Parquet files are never rewritten.** The cluster artifact is derived from them, and the `(segment_id, row_id)` in each node is what lets [`fetch_results_by_id`](src/core/table/read.rs:1523) resolve the top-k rows afterwards.

**Why this is cheaper than it sounds:** the codes are recomputed from Parquet rather than copied from 25 sidecars, so the build is a single sequential pass over the data plus a 10.75 GB write. No read-modify-write of the existing sidecars, and no 30.97 GB Parquet rewrite.

---

## 5. Cost

Global k-means is the expensive part, and it is 25× the current per-segment work:

| | Centroids | Distance ops per iteration |
|---|---:|---:|
| Today (per segment) | 126 × 25 | 10M × 126 × 768 = 0.97 TFLOP |
| Global, K=1024 | 1024 | 10M × 1024 × 768 = 7.9 TFLOP |
| Global, K=3150 | 3150 | 10M × 3150 × 768 = 24 TFLOP |

At ~10 iterations this is 79–240 TFLOP. On CPU at ~50 GFLOPS that is 26–80 minutes; the GPU assignment path is what makes it tractable. Mitigations, in order of preference:

1. **Use the GPU k-means path** for both assignment and centroid update.
2. **K = 1024, not 3150.** Fewer, larger clusters cost less to train and still give global coverage; the probe count needed for a given recall is what matters, not the cluster count.
3. **Mini-batch k-means** over a 1–2 % sample for centroid fitting, full assignment once.
4. **Reuse the existing 3,150 per-segment centroids as an initialization** (concatenate, then run a few Lloyd iterations) instead of random init. This converges in far fewer passes and is nearly free.

The artifact write is **10.75 GB of Puffin only** — the 30.97 GB of Parquet is read once and never rewritten. At NVMe write speeds that is minutes; the k-means dominates.

---

## 6. The Three Traps

### 6.1 Incremental ingestion breaks the invariant

After compaction, new rows arrive in arrival order and belong to no compacted cluster. Three options:

| Option | Behavior | Cost |
|---|---|---|
| **Re-cluster everything** | Correct, simple | Full k-means + full rewrite per batch — untenable |
| **Assign to existing centroids** | New rows join the nearest global cluster, written to a *delta* segment | Cheap; clusters drift as data grows |
| **LSM-style delta segments** | Delta segments searched separately, merged on next compaction | Standard, and the right answer |

The LSM approach is the one to build: compacted "base" segments plus uncompacted "delta" segments, with the router aware of both. This is also the structure worth claiming — see § 8.

### 6.2 The payload fetch becomes scattered (the real cost of index-only)

This is the tradeoff that replaces the access-pattern regression from the first draft. Because the data files keep their arrival order, the top-k rows for a query now live in up to `k` different Parquet files at arbitrary offsets, instead of being contiguous.

Measured today, this is cheap: `k=1` and `k=100` differ by 4 % (15.06 vs 15.73 ms), so payload hydration is not the bottleneck while the data is in page cache. It will matter on a cold cache or on a dataset larger than RAM, where each scattered read pulls a full page.

Mitigations, in order: (a) accept it — `k` is small and the reads are few; (b) sort the top-k by `(segment_id, row_id)` before fetching so each file is read once; (c) if cold-cache latency becomes the constraint, *then* consider data compaction as a second-phase optimization, at which point the first draft's approach becomes the follow-up rather than the starting point.

### 6.3 Centroid drift

Global centroids trained at time $T$ become stale as the corpus grows. There is no cheap incremental k-means update that preserves cluster quality. Practical answer: re-cluster on a schedule (nightly/weekly), and accept that recall degrades slowly between runs. Track a drift metric (mean distance from rows to their assigned centroid) and trigger re-clustering on threshold.

---

## 7. Validation Plan

Do not ship this on the strength of the projection in § 6.4 of the analysis. The projection assumes recall is preserved at `n_probe=16`, which is unmeasured.

1. **Restore the corpus.** `/data/cohere` is gone; without `GT` there is no recall number and the whole exercise is latency-only.
2. **Baseline:** current table, `per_segment`, sweep `n_probe` → the recall/QPS curve.
3. **Global centroids, per-segment sidecars:** train global centroids, keep the existing per-segment artifacts, and let the router map one global cluster to many `(segment, local_cluster)` pairs. Expect *no* improvement, and possibly a regression from partial-cluster overhead. This isolates the co-location effect and confirms § 2.
4. **Global centroids + cluster-keyed artifact:** the same sweep. The delta between (3) and (4) is the value of index reorganization; the delta between (2) and (4) is the value of the whole change.
5. **Report at equal recall**, not equal `n_probe`. A 3× QPS win at 0.75 recall is not a win.
6. **Cold-cache check:** drop the page cache (`echo 3 > /proc/sys/vm/drop_caches`) and repeat (4). This is where the scattered payload fetch in § 6.2 shows up, and it decides whether data compaction is ever needed.

---

## 8. Patent Consequence

This pipeline is a stronger Family G embodiment than the current text, because it recites the *lifecycle* rather than a static layout:

> A method of maintaining a similarity index for a table stored as a plurality of immutable data files, comprising: deriving a set of global partition centroids from a sample of rows spanning the plurality of data files; assigning each row a partition identifier; generating an index artifact, referenced by version metadata of the table and not owned by any single data file, that stores for each partition a graph over compressed representations of that partition's rows together with, for each node, an identifier of the data file and a position within it, such that a partition's members are contiguous in the artifact while remaining non-contiguous in the data files; and, for rows ingested after the artifact is generated, storing a delta artifact searched in conjunction with the index artifact until a subsequent regeneration.

Two clauses here are the ones worth having, and neither is in the current draft:

- **"not owned by any single data file"** — this is the structural difference from Borycki's Puffin-backed DiskANN, which attaches the graph to a data file. It is also the clause that makes index-only reorganization possible at all.
- **"contiguous in the artifact while remaining non-contiguous in the data files"** — this is the decoupling, stated as a positive technical fact rather than a negative limitation, and it is directly verifiable by inspecting the artifact against the data layout.

The delta-artifact clause covers continuous ingestion. All three should be added to [`CLAIM_BREADTH_STRATEGY.md`](file:///home/ralbright/projects/benolabs/patent-proposal/paper/CLAIM_BREADTH_STRATEGY.md) § 4 Family G, and "data-file co-location" should be added to the alternative-embodiments list as the variant we deliberately do *not* require.

---

## 9. Recommendation

**Yes — but compact the index, not the data.** Global centroids without index reorganization are a no-op; index reorganization without global centroids is also a no-op. They only work together, and neither requires touching the Parquet files.

Order of work:

1. Restore the corpus and get a recall baseline. Nothing below is verifiable without it.
2. Add global k-means (GPU path, K=1024, warm-started from the existing 3,150 per-segment centroids).
3. Add a table-level index artifact type to the manifest — `IndexFile` currently hangs off `ManifestEntry`, so this is the one real schema change.
4. Build the cluster-keyed artifact: one sequential pass over Parquet, quantize, build per-cluster HNSW, store `(segment_id, row_id)` per node.
5. Rebuild the router from global centroids; point it at the new artifact.
6. Measure at equal recall, then repeat cold-cache. Only if cold-cache payload fetch dominates does data compaction become worth doing.
7. If the win is real, add the delta-artifact path for incremental ingestion.
