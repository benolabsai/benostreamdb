# BenoStreamDB — Partitioned Indexing & Query Execution Fixes Plan (Step 1C–1D)

**Status (2026-11-10): Completed & locally verified (Steps 1A–1D passed; `tests/test_partitioned_indexing` 6/6, `--lib` 441/441, `scripts/no_panic_check.sh` 0 panic sites & hard gate OK).**  
**Companion plans:** [production_readiness_plan.md](production_readiness_plan.md), [iceberg_v3_and_catalog_parity_plan.md](iceberg_v3_and_catalog_parity_plan.md).

---

## 0. Why This Work Matters (Architectural Context)

BenoStreamDB stores Hive-partitioned tables using standard directory prefixes under the table root (`{col}={value}/{segment_id}.parquet` alongside co-located Puffin and sidecar index files `{col}={value}/{segment_id}.puffin`, while maintaining table-level routing artifacts such as `TableVectorRouter` and `ClusterIndex` over segment centroids). This layout keeps Parquet segments portable to external Iceberg engines while preserving local I/O affinity between a partition's data file and its auxiliary indexes.

An end-to-end audit of indexed partitioned queries uncovered four failure classes when segments live under Hive subdirectories instead of the flat table root:

1. **Path & segment-identifier drift:** Readers, primary-key probes, split readers, preloading caches, and `merge` planners stripped or reconstructed `segment_id` / Puffin cache keys inconsistently for nested `col=val/seg_*.parquet` paths, dropping `data_store`, `iceberg_schema`, or sidecar index matches.
2. **Global centroid probe theft on filtered KNN queries:** Routing globally via `TableVectorRouter` before partition/statistics pruning allowed decoy centroids in already-pruned partitions to consume IVF probe budget (`probes`) and skew adaptive thresholds, lowering recall inside the surviving partitions.
3. **Unfiltered fallthrough & premature projection:** `read_split_async` fell through to an unfiltered `stream_row_groups` scan whenever `query_index_first` validly returned zero rows, and `read_write_buffer` / `read_file_async` / `read_split_async` projected `SELECT` columns before evaluating `WHERE` predicates that referenced non-projected columns.
4. **SQL, Merge & Cloud-Prefix Gaps:** `BenoStreamTableProvider::scan` formatted partition pruning summaries for `EXPLAIN` without dropping pruned segments ahead of DataFusion partitioning; `BenoStreamExec` and `VectorScanExec` parsed SQL filters per segment via legacy `QueryFilter::parse_multi`; `MergePlanner` wrote flat-root unindexed segments on partitioned tables and omitted schema alignment on evolved tables; and `flush_internal_async` ran `std::fs::create_dir_all` unconditionally on non-local (`s3://`, `memory://`) URIs.

**Design invariant:** Preserve BenoStreamDB's existing Hive partition layout (`col=val/{seg}.parquet` and `col=val/{seg}.puffin`) while making every write, merge, preload, SQL scan, vector search, and split read partition-aware, delete-aware, and prefix-safe.

---

## 1. Completed Baseline in Working Tree (Do Not Regress)

The following changes are already applied in the working tree (alongside three separate, pre-existing Track 2 catalog WIP files `Cargo.toml`, `src/core/catalog/common.rs`, and `src/core/catalog/memory.rs` that must remain unstaged until Track 2 resumes):

- **Core path, segment, Puffin, reader, planner, & write/maintenance baseline (Steps 1A–1B):** `src/core/{cache.rs,iceberg/delete.rs,planner.rs,puffin.rs,query.rs,segment.rs}`, `src/core/index/{build_graph.rs,build_inverted.rs,json_path.rs}`, `src/core/reader/{delete.rs,filter.rs,mod.rs,scan.rs}`, and `src/core/table/{index_config.rs,ingest.rs,maintenance.rs,primary_key.rs,stage.rs}`.
- **Completed Step 1C modules:**
  1. **Partition-constrained vector routing ([`src/core/index/vector_router.rs`](../src/core/index/vector_router.rs)):** Added `TableVectorRouter::route_allowed` and `route_with_policy_allowed` constraining `GlobalPruning`, `PerSegmentTargeting`, and `Adaptive` routing strictly to `allowed_segments: Option<&HashSet<String>>`, backed by unit test `test_route_with_allowed_segments_ignores_pruned_decoys`.
  2. **Index preloading & cache-key parity ([`src/core/table/preload.rs`](../src/core/table/preload.rs)):** Added `Table::cache_root()` (matching `HybridReader::cache_root()`), normalized `preload_indexes_async` and `ensure_vector_router` to resolve `relativize_against` / `resolve_entry_index_path` / `segment_id_from_path(&rel_entry)`, populated `ANALYZER_META_CACHE` (`analyzer` and `:source_type`) even on warm inverted cache hits, and updated `warm_on_disk` for Puffin vector blobs and CSR graph triples.
  3. **Hive-aware fallback data-file discovery ([`src/core/segment.rs`](../src/core/segment.rs) & [`src/core/table/mod.rs`](../src/core/table/mod.rs)):** Added `is_data_parquet_path` (excluding `_manifest`, `_wal`, `deletes`, `metadata` path components and delete/index sidecars) and wired it into `load_initial_schema` and `list_segments_from_store`.
  4. **Table read & explain paths ([`src/core/table/read.rs`](../src/core/table/read.rs)):** Updated `explain()` to use `self.segment_reader(entry)` + active `iceberg_schema`; gated the `ClusterIndex` fast path with `!has_deletes`; pruned entries via `QueryPlanner` *before* calling `router.route_allowed(..., Some(&allowed_seg_ids))`; forwarded `iceberg_schema` and query parameters on `VectorSearchRequest`; fixed uncompressed write-buffer `mem_batches` so `filter_expr` runs before `columns` projection; and added `execute_phrase_search_as_scored_filtered` calling `reader.phrase_search_index_filtered`.
  5. **Compactor remote-prefix fix ([`src/core/compaction.rs`](../src/core/compaction.rs)):** Set `base_path = String::new()` in `Compactor::new` (retaining URL validation) so `PrefixStore`-wrapped S3/GS/Azure URIs are not double-prefixed.
  6. **Flush outer directory guard ([`src/core/table/write.rs`](../src/core/table/write.rs#L1254-L1258)):** Guarded outer `std::fs::create_dir_all(base_path)?` with `is_local_uri` (`!self.uri.contains("://") || self.uri.starts_with("file://")`).

---

## 2. Remaining Step 1C Fixes

### Task 1C.1 — Inner Local-Directory Guard on Flush (`src/core/table/write.rs`)
- **Gap:** Inside `flush_internal_async` ([`src/core/table/write.rs`](../src/core/table/write.rs#L1481)), the per-partition task calls `let _ = std::fs::create_dir_all(&full_base_path);` unconditionally, creating bogus relative directories (e.g., `s3:/bucket/...`) in the process CWD during non-local writes.
- **Fix:** Capture `is_local_uri` inside the segment-build closure and guard `std::fs::create_dir_all(&full_base_path)` with `if is_local_uri`.

### Task 1C.2 — Partition-Aware Primary-Key Storage Checks (`src/core/table/primary_key.rs`)
- **Gap:** `_check_pk_in_storage_async` ([`src/core/table/primary_key.rs`](../src/core/table/primary_key.rs#L264-L318)) manually copies `entry.file_path` and `manifest.schemas.last()` onto `HybridReader::new(config, store, &uri)`, dropping `data_store`, `record_count`, `file_checksum`, `current_schema_id` lookup, and `relativize_against` / `segment_id_from_path` normalization.
- **Fix:** Resolve active `iceberg_schema` using `manifest.current_schema_id` (falling back to `.last()`), construct each per-segment reader via `self.segment_reader(&entry)?.with_iceberg_schema(...)`, and stream `reader.check_value_exists(&column, &value)` concurrently.

### Task 1C.3 — Connector File/Split Reads & Write-Buffer Projection (`src/core/table/stats.rs`)
- **Gaps:**
  1. `read_file_async` ([`src/core/table/stats.rs`](../src/core/table/stats.rs#L399-L577)) constructs `SegmentConfig::new("", seg_id)` manually when matching manifest entries, omitting `data_store`, `record_count`, `file_checksum`, and `iceberg_schema`.
  2. `read_file_async` and `read_split_async` ([`src/core/table/stats.rs`](../src/core/table/stats.rs#L580-L705)) read/project `target_schema` *before* post-filtering, breaking filters on non-projected columns, and parse `filter` via naive `QueryFilter::parse` rather than coerced `FilterExpr::parse_sql`.
  3. `read_split_async` falls through from an index hit returning `Ok(vec![])` to an **unfiltered** `stream_row_groups` scan, never post-filters `stream_row_groups`, and does not forward `data_store` or `iceberg_schema` in `get_splits` / `get_splits_async` / `read_split_async`.
  4. `read_write_buffer` ([`src/core/table/stats.rs`](../src/core/table/stats.rs#L817-L866)) projects `columns` *before* calling `planner.filter_batch`, silently failing or dropping predicates whose columns are not in `SELECT`, does not align uncommitted batches to `self.arrow_schema()`, and lacks a `FilterExpr` entrypoint for SQL scans.
- **Fix:**
  1. Match manifest entries cleanly via `self.segment_reader(entry)?.with_iceberg_schema(...)` (resolving `iceberg_schema` from `manifest.current_schema_id`), and attach `self.data_store.clone()` on split/file fallback readers (`get_splits`, `get_splits_async`, `read_file_async`, `read_split_async`).
  2. Parse `filter` once with `FilterExpr::parse_sql(filter_str, full_schema.clone()).await?.coerce_literals(&full_schema)`, build `read_schema` as `target_columns ∪ filter_expr.get_referenced_columns()` when projecting, evaluate `planner.filter_expr`, and re-project to `target_schema` afterward (dropping empty batches and returning evaluation errors instead of returning unfiltered rows).
  3. Track `index_used: bool` in `read_split_async`: when `query_index_first` succeeds (`Ok(batches)`), set `index_used = true` and return post-filtered/projected batches directly without falling through to unfiltered `stream_row_groups`. When `!index_used`, read `stream_row_groups(Some(&split.row_group_ids), read_schema)` and post-filter + project when `filter_expr` is `Some`.
  4. Add `pub fn read_write_buffer_expr(&self, expr: Option<&FilterExpr>, columns: Option<&[&str]>) -> Result<Vec<RecordBatch>>` that aligns pending/staging batches to `self.arrow_schema()`, runs `planner.filter_expr` *before* projecting `columns`, and keeps `read_write_buffer` as a thin wrapper.

### Task 1C.4 — Partitioned, Indexed & Schema-Aligned Merge (`src/core/merge.rs` & `src/core/table/merge.rs`)
- **Gaps:**
  1. `MergePlanner` ([`src/core/merge.rs`](../src/core/merge.rs#L29-L47)) holds no table context (`data_store`, `store`, `partition_spec`, `iceberg_schema`, `table_schema`, `primary_key`, `index_all`, `index_columns`, `index_configs`, `default_device`, `format_version`).
  2. Step 1/2 readers hardcode local `create_object_store(base_path)` without `data_store` or `iceberg_schema`; Step 2 CoW uses `source_batch.schema()` directly (failing `concat_batches` if evolved columns differ in nullability/order/missing fields) and never rebuilds indexes on the rewritten segment; Step 3 MoR/unmatched inserts write a single unpartitioned, unindexed flat segment at table root regardless of `partition_spec` or index configuration.
  3. `Table::merge` ([`src/core/table/merge.rs`](../src/core/table/merge.rs#L77-L129)) neither passes table context to `MergePlanner` nor invalidates cached `vector_routers` / `cluster_indexes` after committing replaced/deleted/added segments.
- **Fix:**
  1. Add optional table-context fields and a fluent `with_table_context(...)` setter on `MergePlanner`, leaving zero-argument `MergePlanner::new()` intact for existing runtime tests.
  2. Wire `store`, `data_store`, `partition_spec`, current `iceberg_schema`, `arrow_schema()`, `primary_key`, `indexing` state, and `format_version` from `Table::merge` into `MergePlanner`, and clear `self.indexing.vector_routers` and `self.indexing.cluster_indexes` after a non-empty commit + `publish_snapshot_async`.
  3. Add `align_batch_to_schema` so Step 2 CoW and Step 3 inserts normalize columns/metadata/missing evolved columns cleanly before `concat_batches` / writing.
  4. In Step 2 CoW, propagate `index_all`, `index_columns`, `index_configs`, `default_device`, and `primary_key` to `HybridSegmentWriter`; conditionally run `write_batch`, `build_indexes`, `finish_indexing`, `upload_to_store`, `to_manifest_entry()`, and `ensure_prefix(hive_dir, ...)` on both `entry.file_path` and every `idx.file_path`.
  5. In Step 3 unmatched inserts, split `inserts_batch` through `self.partition_spec.partition_batch(&aligned_inserts)?`, write each non-empty partition under its `hive_dir` (`seg_{uuid}.parquet`) with full index build + upload + Hive prefixing, and emit one `MergeCommitAction::AddData` per partition segment. (Propagate `self.format_version.max(2)` to `IcebergDeleteWriter`, and handle V3 lineage fields `_row_id` / `_last_updated_sequence_number` safely during schema alignment.)

### Task 1C.5 — SQL Scan Partition Pruning & Filter Pushdown (`src/core/sql/mod.rs`)
- **Gaps:**
  1. `BenoStreamTableProvider::scan` ([`src/core/sql/mod.rs`](../src/core/sql/mod.rs#L338-L671)) computes `pruning_summary` from `all_segments` for `EXPLAIN` output, yet **never removes pruned segments** from `segments` unless handled by the specialized PK-tuple or single-column OR-range branches. Partition filters like `WHERE region = 'B'` or `WHERE event_date >= DATE '2026-01-01'` still schedule every segment into DataFusion partitions.
  2. `expr_to_sql` only handles uncast `Column op Literal` with `Utf8`/`Int`/`Float`/`Boolean` and simple `InList`, silently dropping `Date32`, `Date64`/`Timestamp`, `Utf8View`/`LargeUtf8`, single quotes in strings, `Literal op Column` operand order, nested `AND`, `Between`, `IsNull`, and `IsNotNull`.
  3. BM25 pushdown detection in `scan` and `supports_filters_pushdown` misses flipped `literal = col`, `Cast`/`TryCast` wrappers, and `Utf8View`/`LargeUtf8`.
- **Fix:**
  1. Construct `combined_expr` from `filters` via `AND` and apply `QueryPlanner::new().prune_entries(&segments, Some(&FilterExpr::DataFusion(combined_expr)), None)` *before* distributing `segments` across `target_partitions` (preserving `all_segments` before PK/OR/general pruning so `EXPLAIN` `pruning_summary` remains exact).
  2. Generalize `expr_to_sql` -> `Vec<(String, String)>`: unwrap `Cast`/`TryCast`, flip comparison operators when `Literal op Column`, flatten nested `AND`s, format `Date32` as `DATE 'YYYY-MM-DD'`, format `Date64`/timestamps cleanly, escape `'` as `''` in `Utf8`/`Utf8View`/`LargeUtf8`, and support `Between` (`!negated`), `IsNull`, `IsNotNull`, and multi-type `InList`.
  3. Normalize BM25 equality detection in both `scan` and `supports_filters_pushdown` for operand order, `Cast`/`TryCast`, and all UTF-8 scalar variants.

### Task 1C.6 — SQL `BenoStreamExec` Filter Expressiveness & Write-Buffer Schema Alignment (`src/core/sql/physical_plan.rs`)
- **Gaps:** `BenoStreamExec::execute` ([`src/core/sql/physical_plan.rs`](../src/core/sql/physical_plan.rs#L204-L333)) re-parses `self.filter` inside the per-segment loop using `QueryFilter::parse_multi(f).into_iter().next()` (discarding secondary conjuncts, `Date32`, `BETWEEN`, `IS NULL`, etc.), calls `table.stream_segment`, and fails with `"Write buffer schema mismatch"` when Iceberg field metadata or types differ on uncommitted `read_write_buffer` batches.
- **Fix:**
  1. Parse `self.filter` **once** ahead of the segment loop via `FilterExpr::parse_sql(f, original_schema.clone()).await` with fallback to `FilterExpr::from_filters(QueryFilter::parse_multi(f))`, and extract `iceberg_schema` once from `table.manifest().await`.
  2. Stream each entry via `table.read_segment_expr(&entry, filter_expr.as_ref(), 1, col_slice, iceberg_schema.as_ref()).await`.
  3. On partition `0`, call `table.read_write_buffer_expr(filter_expr.as_ref(), col_slice)` and run the same column cast / metadata-stripping normalization against `expected_schema_inner` used for committed batches.

### Task 1C.7 — SQL KNN Partition Routing & Projection Safety (`src/core/sql/physical_plan/vector_scan.rs`, `plan_detection.rs`, `plan_rewriter.rs`)
- **Gaps:**
  1. `VectorScanExec::execute` ([`src/core/sql/physical_plan/vector_scan.rs`](../src/core/sql/physical_plan/vector_scan.rs#L152-L256)) parses `filter` via `QueryFilter::parse_multi`, drops `table.data_store.clone()`, omits `vp.probes`, `vp.refine_factor`, and `iceberg_schema`, and never consults `TableVectorRouter::route_allowed` constrained to the partition-pruned entries in `self.partitions`.
  2. `BenoStreamExecRef` ([`plan_detection.rs`](../src/core/sql/optimizer/vector_search/plan_detection.rs#L39-L61)) passes `hs.schema().clone()` (already projected) instead of unprojected `base_schema` into `VectorScanExec::new` in [`plan_rewriter.rs`](../src/core/sql/optimizer/vector_search/plan_rewriter.rs#L88-L96).
- **Fix:**
  1. In `VectorScanExec::execute`, parse `filter` once via `FilterExpr::parse_sql` (with `parse_multi` fallback), load `iceberg_schema` from `table.manifest().await` once, forward `table.data_store.clone()`, `.with_probes(vp.probes)`, `.with_refine_factor(vp.refine_factor)`, and `.with_iceberg_schema(iceberg_schema)`.
  2. Compute `allowed_seg_ids` over all entries in `self.partitions` (or current partition entries intersected with partition-wide routed candidates so each partition sees its own surviving routed entries); for float-vector queries without pre-populated routing, call `table.ensure_vector_router(&vp.column).await` -> `router.route_allowed(q_f32, probes, Some(&allowed_seg_ids))`, filter `entries` to matching candidate segments (while keeping unindexed segments outside `routed`), and attach `targeted_clusters`.
  3. Preserve both `base_schema` (for `VectorScanExec::new` projection logic) and projected `schema` (for `VectorMergeExec::new`) in `BenoStreamExecRef` / `plan_rewriter.rs`, and keep `VectorSearchOptimizerRule::schema_check() == true` satisfied.

---

## 3. Step 1D Verification Matrix & Exit Gates

Create `/home/ralbright/projects/benostreamdb/tests/test_partitioned_indexing.rs` covering all six acceptance dimensions end to end against real partitioned tables:

1. **Hive layout & index fidelity:** Write multi-batch partitioned tables (`category = 'A' | 'B'`, `event_date = Date32`) across scalar (`Int32`, `Utf8`, `Date32`), BM25, JSON-path, CSR graph, HNSW-IVF, and TurboQuant TQ8 indexes; flush and assert Hive paths `{col}={val}/{seg}.parquet` + co-located `.puffin` sidecars and exact manifest registration.
2. **Backfill & reindex preservation:** Run `backfill_indexes_async` and `reindex_inverted_column` on partitioned tables and verify Hive subdirectories, prior Puffin blobs, and `preload_indexes_async` / `ensure_vector_router` cache hits remain intact.
3. **Partition-pruned KNN routing (`route_allowed`) & SQL pushdown:** Place a closer decoy centroid inside partition `'A'` and target partition `'B'` (`WHERE category = 'B' AND event_date >= DATE '2026-01-01' ORDER BY emb <=> ... LIMIT k` with column subset projection); verify `route_allowed` does not waste probes on decoy partition `'A'`, recalls exact partition `'B'` neighbors, and `explain()` reports quantized / index paths accurately.
4. **Deletes & hybrid search:** Delete rows (`delete_async`) on a cluster-indexed partitioned table; verify `ClusterIndex` fast path steps aside when `has_deletes` is true and vector, keyword, phrase (`execute_phrase_search_as_scored_filtered`), and hybrid RRF queries match full scans.
5. **Write-buffer & split-reader edge cases:** Query uncommitted write-buffer rows projecting columns disjoint from the predicate (`SELECT id WHERE category = 'A'`) via both `read_write_buffer_expr` and SQL; verify `read_split_async` returns zero rows on an empty-index hit without falling through to an unfiltered scan, and filters fallback row-group scans correctly.
6. **Partitioned merge, compaction, vacuum, schema evolution & non-local prefixes:** Exercise `Table::merge` (`CopyOnWrite` and `MergeOnRead` inserting across multiple partitions on an indexed, schema-evolved table), `compact` + `vacuum` (preserving live Hive indexes and pruning obsolete ones), `load_initial_schema` fallback on Hive layouts, and `Compactor::new` / `flush_internal_async` behavior on `memory://` URIs without creating local CWD directories.

### Mandatory Exit Gates Before Step 1 Local Commit

All commands must run against `/home/ralbright/projects/benostreamdb` with zero compiler warnings (`#![deny(warnings)]`) and zero non-test panics:

```bash
cargo +1.98.0 check --manifest-path Cargo.toml --features python --tests
cargo +1.98.0 test --manifest-path Cargo.toml --test test_partitioned_indexing
cargo +1.98.0 test --manifest-path Cargo.toml --test test_differential_index_oracle
cargo +1.98.0 test --manifest-path Cargo.toml --test test_graph_csr_path_conformance
cargo +1.98.0 test --manifest-path Cargo.toml --test test_maintenance_invariant
cargo +1.98.0 test --manifest-path Cargo.toml --test integration_test_hnsw_ivf_native
cargo +1.98.0 test --manifest-path Cargo.toml --lib
./scripts/no_panic_check.sh
```

Upon passing all gates: create a clean **local commit** (no `git push`) for Step 1 + this plan while keeping `Cargo.toml`, `src/core/catalog/common.rs`, and `src/core/catalog/memory.rs` unstaged, then immediately resume **Track 2 (`Phase A -> Phase B -> Phase C`)**, followed by **Track 1** and **Track 3**.
