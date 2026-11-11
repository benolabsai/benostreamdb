# ADR-011: Hive-Partitioned Indexing, Pruned KNN Routing, and Query-Execution Conformance

> **Status: implemented (`v0.16.0`).**
> Implemented across [`src/core/index/vector_router.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/index/vector_router.rs), [`src/core/table/preload.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/preload.rs), [`src/core/table/primary_key.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/primary_key.rs), [`src/core/table/stats.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/stats.rs), [`src/core/table/read.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/read.rs), [`src/core/table/merge.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/merge.rs), [`src/core/merge.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/merge.rs), [`src/core/segment.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/segment.rs), [`src/core/compaction.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/compaction.rs), [`src/core/sql/mod.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/mod.rs), [`src/core/sql/physical_plan.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/physical_plan.rs), and [`src/core/sql/physical_plan/vector_scan.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/physical_plan/vector_scan.rs).
> Certified by [`tests/test_partitioned_indexing.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_partitioned_indexing.rs) (6/6 acceptance dimensions), the `--lib` suite (441/441), and [`scripts/no_panic_check.sh`](https://github.com/benolabsai/benostreamdb/blob/main/scripts/no_panic_check.sh) (zero panic sites, hard gate OK). Promoted from `plans/partitioned_indexing_and_query_fixes_plan.md`.

---

## 1. Context & Operational Motivation

BenoStreamDB stores Hive-partitioned tables as standard directory prefixes under the table root — `{col}={value}/{segment_id}.parquet`, with co-located Puffin and sidecar index files at `{col}={value}/{segment_id}.puffin` — while maintaining table-level routing artifacts (`TableVectorRouter`, `ClusterIndex`, `TableVectorRouter` centroid routing) over segment centroids. That layout is what makes a BenoStreamDB table readable by external Iceberg engines unmodified: partition directories are plain object-store prefixes, and the metadata tree never encodes an engine-specific layout. Keeping each partition's data file and its auxiliary indexes physically adjacent also preserves the I/O affinity that the local-cache path depends on.

An end-to-end audit of indexed partitioned queries found four failure classes that only appear once segments live under Hive subdirectories rather than at the flat table root:

1. **Path and segment-identifier drift.** Readers, primary-key probes, split readers, preload caches, and the `merge` planner stripped or reconstructed `segment_id` / Puffin cache keys inconsistently for nested `col=val/seg_*.parquet` paths, silently dropping `data_store`, `iceberg_schema`, or the sidecar index match. The failure mode is quiet: the query still returns rows, just through a slower path with indexes ignored.

2. **Global centroid probe theft on filtered KNN.** Routing globally through `TableVectorRouter` *before* partition and statistics pruning let decoy centroids in already-pruned partitions consume the IVF probe budget (`probes`) and skew the adaptive threshold, lowering recall inside the partitions that actually survived pruning. This is a correctness bug, not a performance bug: the answer changes depending on which partitions the optimizer happened to eliminate.

3. **Unfiltered fallthrough and premature projection.** `read_split_async` fell through to an unfiltered `stream_row_groups` scan whenever `query_index_first` validly returned zero rows, and `read_write_buffer` / `read_file_async` / `read_split_async` projected `SELECT` columns *before* evaluating `WHERE` predicates that referenced non-projected columns — so `SELECT id WHERE category = 'A'` either errored or returned unfiltered rows.

4. **SQL, merge, and cloud-prefix gaps.** `BenoStreamTableProvider::scan` formatted partition-pruning summaries for `EXPLAIN` without removing pruned segments before DataFusion partition assignment; `BenoStreamExec` and `VectorScanExec` re-parsed SQL filters per segment through the legacy `QueryFilter::parse_multi`, discarding secondary conjuncts and non-scalar types; `MergePlanner` wrote flat-root unindexed segments on partitioned tables and skipped schema alignment on evolved tables; and `flush_internal_async` called `std::fs::create_dir_all` unconditionally, creating literal `s3:/bucket/...` directories in the process working directory on every non-local write.

**Design invariant (unchanged):** preserve the existing Hive partition layout exactly, while making every write, merge, preload, SQL scan, vector search, and split read partition-aware, delete-aware, and prefix-safe. No on-disk format change was introduced; the fix is entirely in path resolution, planner ordering, and predicate handling.

---

## 2. Decision

The partition layout is correct and stays. The defect was that partition awareness was *layered on* at some call sites rather than being a property of the shared resolution helpers. The decision is therefore to funnel every path-, routing-, and predicate-sensitive operation through the same helpers, and to make pruning strictly precede routing:

1. One cache-root and `segment_id` resolution path shared by `HybridReader` and `Table` (`cache_root`, `relativize_against`, `resolve_entry_index_path`, `segment_id_from_path`).
2. Routing is *constrained*, not global: `TableVectorRouter::route_allowed` / `route_with_policy_allowed` take an explicit `allowed_segments` set, and every vector entry point prunes first, then routes.
3. Predicates are parsed once, into `FilterExpr`, at plan time — never re-parsed per segment, never evaluated after projection.
4. Schema alignment (`align_batch_to_schema`) is mandatory before `concat_batches` or write, so evolved tables cannot fail on nullability/order/missing-field drift.

---

## 3. What Shipped

### 3.1 Partition-constrained vector routing — [`vector_router.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/index/vector_router.rs)

`TableVectorRouter::route_allowed` and `route_with_policy_allowed` constrain `GlobalPruning`, `PerSegmentTargeting`, and `Adaptive` routing strictly to `allowed_segments: Option<&HashSet<String>>`. Decoy centroids in pruned partitions can no longer consume probe budget. Unit test: `test_route_with_allowed_segments_ignores_pruned_decoys`.

### 3.2 Index preloading and cache-key parity — [`preload.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/preload.rs)

`Table::cache_root()` now matches `HybridReader::cache_root()`. `preload_indexes_async` and `ensure_vector_router` resolve through `relativize_against` / `resolve_entry_index_path` / `segment_id_from_path(&rel_entry)`. `ANALYZER_META_CACHE` is populated with `analyzer` and `:source_type` even on warm inverted-cache hits (previously a warm hit left the analyzer unset, which is what made partitioned reindex fall back to `identity`). `warm_on_disk` covers Puffin vector blobs and CSR graph triples.

### 3.3 Hive-aware fallback data-file discovery — [`segment.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/segment.rs), [`table/mod.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/mod.rs)

`is_data_parquet_path` excludes `_manifest`, `_wal`, `deletes`, and `metadata` path components as well as delete and index sidecars, and is wired into `load_initial_schema` and `list_segments_from_store`. This is what makes schema recovery work on a table with no readable tiered manifest.

### 3.4 Read and explain paths — [`read.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/read.rs)

`explain()` uses `self.segment_reader(entry)` plus the active `iceberg_schema`; the `ClusterIndex` fast path is gated on `!has_deletes`; entries are pruned through `QueryPlanner` *before* `router.route_allowed(..., Some(&allowed_seg_ids))`; `VectorSearchRequest` forwards `iceberg_schema` and the full query parameter set; uncompressed write-buffer batches run `filter_expr` before `columns` projection; and `execute_phrase_search_as_scored_filtered` was added on top of `reader.phrase_search_index_filtered`.

### 3.5 Primary-key storage probes — [`primary_key.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/primary_key.rs)

`_check_pk_in_storage_async` no longer hand-copies `entry.file_path` and `manifest.schemas.last()`. It resolves the active `iceberg_schema` through `manifest.current_schema_id` (falling back to `.last()`) and builds each per-segment reader through `self.segment_reader(&entry)?.with_iceberg_schema(...)`, preserving `data_store`, `record_count`, `file_checksum`, and the normalized relative path.

### 3.6 Connector file/split reads and write-buffer projection — [`stats.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/stats.rs)

- `read_file_async` matches manifest entries through `segment_reader(...).with_iceberg_schema(...)` instead of `SegmentConfig::new("", seg_id)`.
- `get_splits`, `get_splits_async`, `read_file_async`, and `read_split_async` all attach `data_store`.
- Filters parse once via `FilterExpr::parse_sql(...).coerce_literals(...)`; `read_schema` becomes `target_columns ∪ filter_expr.get_referenced_columns()`; `planner.filter_expr` runs first and the result is re-projected to `target_schema` afterward, dropping empty batches and propagating evaluation errors rather than returning unfiltered rows.
- `read_split_async` tracks `index_used: bool` so a successful-but-empty index hit returns immediately instead of falling through to an unfiltered `stream_row_groups`.
- `read_write_buffer_expr` aligns pending/staging batches to `self.arrow_schema()`, filters before projecting, and `read_write_buffer` is a thin wrapper over it.

### 3.7 Partitioned, indexed, schema-aligned merge — [`merge.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/merge.rs), [`table/merge.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/merge.rs)

`MergePlanner` gained optional table context with a fluent `with_table_context(...)`; `MergePlanner::new()` remains zero-argument so existing runtime tests compile unchanged. `Table::merge` wires `store`, `data_store`, `partition_spec`, current `iceberg_schema`, `arrow_schema()`, `primary_key`, indexing state, and `format_version`, then clears `vector_routers` and `cluster_indexes` after a non-empty commit plus `publish_snapshot_async`. Step 2 copy-on-write propagates `index_all`, `index_columns`, `index_configs`, `default_device`, and `primary_key` into `HybridSegmentWriter` and runs index build/upload/prefixing on both the data file and every index file. Step 3 splits unmatched inserts through `partition_spec.partition_batch(...)` and writes one indexed segment per non-empty `hive_dir`, emitting one `MergeCommitAction::AddData` per partition.

### 3.8 SQL scan pruning and filter pushdown — [`sql/mod.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/mod.rs)

`combined_expr` is built from `filters` and passed to `QueryPlanner::prune_entries(...)` **before** segments are distributed across `target_partitions`; `all_segments` is retained so `EXPLAIN`'s `pruning_summary` stays exact. `expr_to_sql` was generalized to `Vec<(String, String)>`: it unwraps `Cast`/`TryCast`, flips comparison operators for `Literal op Column`, flattens nested `AND`, formats `Date32` as `DATE 'YYYY-MM-DD'`, formats `Date64`/timestamps, escapes `'` as `''` across `Utf8`/`Utf8View`/`LargeUtf8`, and supports `Between`, `IsNull`, `IsNotNull`, and multi-type `InList`. BM25 equality detection was normalized for operand order, casts, and all UTF-8 scalar variants in both `scan` and `supports_filters_pushdown`.

### 3.9 Execution-plan filter expressiveness — [`physical_plan.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/physical_plan.rs), [`vector_scan.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/sql/physical_plan/vector_scan.rs)

`BenoStreamExec` parses `self.filter` once ahead of the segment loop via `FilterExpr::parse_sql`, falling back to `FilterExpr::from_filters(QueryFilter::parse_multi(f))`, extracts `iceberg_schema` once, and streams through `read_segment_expr`. Write-buffer batches on partition `0` go through `read_write_buffer_expr` and the same cast/metadata-stripping normalization as committed batches, which removes the `"Write buffer schema mismatch"` failure on Iceberg-field-metadata drift.

`VectorScanExec` parses the filter once, forwards `data_store`, `vp.probes`, `vp.refine_factor`, and `iceberg_schema`, and computes `allowed_seg_ids` over its own partitions, calling `ensure_vector_router(&vp.column).await` followed by `route_allowed(...)` for un-routed float-vector queries while leaving unindexed segments outside the routed set. `BenoStreamExecRef` now carries both `base_schema` (unprojected, for `VectorScanExec::new`) and the projected `schema` (for `VectorMergeExec::new`), keeping `VectorSearchOptimizerRule::schema_check() == true` satisfied.

### 3.10 Cloud-prefix write safety — [`compaction.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/compaction.rs), [`table/write.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/write.rs)

`Compactor::new` sets `base_path = String::new()` (retaining URL validation) so `PrefixStore`-wrapped S3/GS/Azure URIs are not double-prefixed. Both the outer and per-partition `std::fs::create_dir_all` calls are gated on `is_local_uri` (`!uri.contains("://") || uri.starts_with("file://")`), so non-local writes no longer materialize relative directories in the process CWD.

---

## 4. Verification

[`tests/test_partitioned_indexing.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_partitioned_indexing.rs) exercises all six acceptance dimensions end to end against real partitioned tables:

1. **Hive layout and index fidelity** — multi-batch partitioned writes (`category = 'A' | 'B'`, `event_date = Date32`) across scalar (`Int32`, `Utf8`, `Date32`), BM25, JSON-path, CSR graph, HNSW-IVF, and TurboQuant TQ8 indexes; asserts `{col}={val}/{seg}.parquet` plus co-located `.puffin` sidecars and exact manifest registration.
2. **Backfill and reindex preservation** — `backfill_indexes_async` and `reindex_inverted_column` on partitioned tables leave Hive subdirectories, prior Puffin blobs, and preload/router cache hits intact.
3. **Partition-pruned KNN routing and SQL pushdown** — a decoy centroid is placed in partition `'A'` while `WHERE category = 'B' AND event_date >= DATE '2026-01-01' ORDER BY emb <=> ... LIMIT k` targets partition `'B'` with column-subset projection; `route_allowed` spends no probes on the decoy, recalls the exact `'B'` neighbors, and `explain()` reports the quantized/index path accurately.
4. **Deletes and hybrid search** — after `delete_async` on a cluster-indexed partitioned table, the `ClusterIndex` fast path steps aside and vector, keyword, phrase, and hybrid RRF queries match full scans.
5. **Write-buffer and split-reader edge cases** — `SELECT id WHERE category = 'A'` through both `read_write_buffer_expr` and SQL; `read_split_async` returns zero rows on an empty index hit without an unfiltered fallthrough, and filters fallback row-group scans correctly.
6. **Merge, compaction, vacuum, schema evolution, and non-local prefixes** — `Table::merge` in both `CopyOnWrite` and `MergeOnRead` modes across multiple partitions on an indexed, schema-evolved table; `compact` + `vacuum` preserving live Hive indexes and pruning obsolete ones; `load_initial_schema` fallback on Hive layouts; and `Compactor`/`flush_internal_async` on `memory://` URIs without creating local directories.

Exit gates all pass with `#![deny(warnings)]` and zero non-test panics:

```bash
cargo test --test test_partitioned_indexing          # 6/6
cargo test --test test_differential_index_oracle
cargo test --test test_graph_csr_path_conformance
cargo test --test test_maintenance_invariant
cargo test --test integration_test_hnsw_ivf_native
cargo test --lib                                     # 441/441
./scripts/no_panic_check.sh                          # 0 panic sites
```

---

## 5. Consequences and Trade-offs

- **No format change.** Partition directories, Puffin sidecar placement, and manifest contents are unchanged, so tables written by this version are readable by the previous one and by external Iceberg engines. The whole change set is resolution order and predicate handling.
- **Routing is now two-phase everywhere.** Pruning must run before routing on every vector path. That is a permanent ordering contract: any new vector entry point that routes first reintroduces decoy probe theft, and the `test_route_with_allowed_segments_ignores_pruned_decoys` unit test is the guard.
- **`FilterExpr` parsing is per-plan, not per-segment.** This removes a class of bug (secondary conjuncts silently dropped) but means an unparseable filter now fails the plan rather than degrading to a per-segment parse. Given `supports_filters_pushdown` already gates which predicates reach this path, failing loudly is the correct behavior.
- **Projection is late.** `read_schema` reads `target_columns ∪ referenced_columns`, so a predicate on a wide unprojected column reads that column. That is strictly better than returning wrong rows, and the extra I/O is bounded by the predicate's own column set.
- **`MergePlanner::new()` keeps its zero-argument signature.** Table context is optional, so out-of-tree callers and the runtime tests are unaffected; a context-free planner still produces the pre-ADR flat-root behavior, which is why `Table::merge` always supplies context.
