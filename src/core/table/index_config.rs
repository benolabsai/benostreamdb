// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::manifest::{IndexAlgorithm, ManifestManager};
use crate::core::reader::HybridReader;
use crate::core::segment::HybridSegmentWriter;
use crate::SegmentConfig;
/// Index configuration: setting indexed columns, adding/dropping indexes,
/// backfill logic, and physical-index inference.
///
/// Contains methods on `Table` for:
/// - `set_indexed_columns`, `set_index_columns`
/// - `add_index`, `drop_index`
/// - `add_index_columns`, `add_index_columns_async`
/// - `index_all_columns`, `index_all_columns_async`
/// - `backfill_indexes`, `backfill_indexes_async`
/// - `infer_index_metadata_from_physical_async`
use anyhow::Result;
use std::collections::HashMap;

use super::Table;

impl Table {
    // -----------------------------------------------------------------------
    // Indexed column management
    // -----------------------------------------------------------------------

    pub fn set_indexed_columns(&self, columns: Vec<String>) {
        let mut cols = self.indexing.index_columns.write();
        *cols = columns;
    }

    /// Update indexing specifications for multiple columns at once.
    /// This is an atomic operation that commits a new manifest version.
    pub async fn set_index_columns(
        &self,
        column_indexes: HashMap<String, Vec<IndexAlgorithm>>,
    ) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager
            .update_index_specs(column_indexes.clone())
            .await?;

        // Update in-memory state
        {
            let mut index_cols = self.indexing.index_columns.write();
            let mut index_configs = self.indexing.index_configs.write();

            for (col, algs) in &column_indexes {
                if algs.is_empty() {
                    // Drop index
                    index_cols.retain(|c| c != col);
                    index_configs.remove(col);
                } else {
                    if !index_cols.contains(col) {
                        index_cols.push(col.clone());
                    }

                    // Extract tokenizer from algorithms if present
                    let tokenizer = algs.iter().find_map(|alg| match alg {
                        IndexAlgorithm::Bm25 { tokenizer, .. } => {
                            if tokenizer.is_empty() || tokenizer == "default" {
                                Some("default".to_string())
                            } else {
                                Some(tokenizer.clone())
                            }
                        }
                        _ => None,
                    });

                    // Update config
                    let config = index_configs.entry(col.clone()).or_insert_with(|| {
                        crate::core::table::state::ColumnIndexConfig {
                            enabled: true,
                            algorithms: algs.clone(),
                            ..Default::default()
                        }
                    });
                    config.algorithms = algs.clone();
                    if let Some(tok) = tokenizer {
                        config.tokenizer = Some(tok);
                    }
                }
            }
            index_cols.sort();
        }

        // Trigger backfill for updated columns that have index algorithms configured.
        // Dropped columns (empty algorithms) should NOT trigger a backfill.
        let cols_to_backfill: Vec<String> = column_indexes
            .iter()
            .filter(|(_, algs)| !algs.is_empty())
            .map(|(col, _)| col.clone())
            .collect();
        if !cols_to_backfill.is_empty() {
            let table_clone = self.clone();
            let handle = tokio::spawn(async move {
                if let Err(e) = table_clone.backfill_indexes_async(cols_to_backfill).await {
                    tracing::error!("Failed to backfill indexes: {}", e);
                }
            });

            self.background_tasks.lock().await.push(handle);
        }

        Ok(())
    }

    pub async fn add_index(&self, column: String, algorithm: IndexAlgorithm) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        let manifest = self.manifest().await?;
        let latest_schema = match manifest.schemas.last() {
            Some(s) => s,
            None => {
                // Update in-memory state if no manifest/schema exists yet.
                // This allows pre-configuring indexes before the first write.
                let mut index_configs = self.indexing.index_configs.write();
                let config = index_configs.entry(column.clone()).or_insert_with(|| {
                    crate::core::table::state::ColumnIndexConfig {
                        enabled: true,
                        ..Default::default()
                    }
                });
                config.algorithms.push(algorithm);

                let mut index_cols = self.indexing.index_columns.write();
                if !index_cols.contains(&column) {
                    index_cols.push(column);
                }
                return Ok(());
            }
        };

        // If it's a composite index, we map it to the first column for manifest storage
        let mut target_col = column.clone();
        if let IndexAlgorithm::CompositeBitmap { columns } = &algorithm {
            if let Some(first) = columns.first() {
                target_col = first.clone();
            }
        } else if let IndexAlgorithm::CsrGraph { src_column, .. } = &algorithm {
            target_col = src_column.clone();
        }

        let field = latest_schema
            .fields
            .iter()
            .find(|f| f.name == target_col)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", target_col))?;

        let mut next_indexes = field.indexes.clone();

        // Deduplicate by type for unique algorithms (Vector families, BM25, Bloom)
        match &algorithm {
            IndexAlgorithm::Hnsw { .. }
            | IndexAlgorithm::HnswPq { .. }
            | IndexAlgorithm::HnswTq4 { .. }
            | IndexAlgorithm::HnswTq8 { .. }
            | IndexAlgorithm::HnswSq8 { .. } => {
                // Vector index family - replace existing ones
                next_indexes.retain(|idx| {
                    !matches!(
                        idx,
                        IndexAlgorithm::Hnsw { .. }
                            | IndexAlgorithm::HnswPq { .. }
                            | IndexAlgorithm::HnswTq4 { .. }
                            | IndexAlgorithm::HnswTq8 { .. }
                            | IndexAlgorithm::HnswSq8 { .. }
                    )
                });
            }
            IndexAlgorithm::Bm25 { .. } => {
                next_indexes.retain(|idx| !matches!(idx, IndexAlgorithm::Bm25 { .. }));
            }
            IndexAlgorithm::Bloom { .. } => {
                next_indexes.retain(|idx| !matches!(idx, IndexAlgorithm::Bloom { .. }));
            }
            IndexAlgorithm::CsrGraph { .. } => {
                next_indexes.retain(|idx| !matches!(idx, IndexAlgorithm::CsrGraph { .. }));
            }
            _ => {
                if !next_indexes.contains(&algorithm) {
                    next_indexes.push(algorithm.clone());
                    let mut updates = HashMap::new();
                    updates.insert(target_col.clone(), next_indexes);
                    self.set_index_columns(updates).await?;

                    if target_col != column {
                        let mut index_configs = self.indexing.index_configs.write();
                        let config = index_configs.entry(column.clone()).or_default();
                        config.enabled = true;
                        if !config.algorithms.contains(&algorithm) {
                            config.algorithms.push(algorithm.clone());
                        }
                        let mut index_cols = self.indexing.index_columns.write();
                        if !index_cols.contains(&column) {
                            index_cols.push(column);
                        }
                    }
                    return Ok(());
                }
            }
        }

        next_indexes.push(algorithm.clone());

        let mut updates = HashMap::new();
        updates.insert(target_col.clone(), next_indexes);

        self.set_index_columns(updates).await?;

        // A CSR graph index is identified solely by its `src_column` (the column
        // whose out-neighbours it stores), which is what `set_index_columns`
        // above already keyed it under. Registering a second entry under the
        // original `column` argument would make two graph indexes (forward and
        // reverse) collide in `index_configs`, clobbering each other and
        // mislabeling the physical CSR files — the CSR fast path would then
        // follow the wrong direction. Composite indexes still need the virtual
        // `column` entry, so only CsrGraph is excluded here.
        if target_col != column && !matches!(algorithm, IndexAlgorithm::CsrGraph { .. }) {
            let mut index_configs = self.indexing.index_configs.write();
            let config = index_configs.entry(column.clone()).or_default();
            config.enabled = true;
            if !config.algorithms.contains(&algorithm) {
                config.algorithms.push(algorithm);
            }
            let mut index_cols = self.indexing.index_columns.write();
            if !index_cols.contains(&column) {
                index_cols.push(column);
            }
        }

        Ok(())
    }

    /// Add a composite roaring bitmap index across multiple scalar columns (e.g., ["tenant_id", "status"]).
    pub fn add_composite_index(&self, columns: Vec<String>) -> Result<()> {
        self.runtime()
            .block_on(self.add_composite_index_async(columns))
    }

    /// Async implementation of add_composite_index
    pub async fn add_composite_index_async(&self, columns: Vec<String>) -> Result<()> {
        if columns.len() < 2 {
            anyhow::bail!("Composite index requires at least 2 columns");
        }
        let composite_name = columns.join(",");
        let alg = IndexAlgorithm::CompositeBitmap {
            columns: columns.clone(),
        };
        self.add_index(composite_name, alg).await
    }

    /// Remove all indexing strategies from a column.
    /// This is an atomic operation that commits a new manifest version,
    /// removes index metadata from all segment entries, invalidates query caches,
    /// and deletes physical sidecar files.
    pub async fn drop_index(&self, column: String) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        let _ = self.wait_for_background_tasks_async().await;
        // Collect all file paths associated with this index from the current manifest.
        let manifest = self.manifest().await?;
        let manager =
            crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri);
        let entries = manager.load_all_entries(&manifest).await?;
        let mut paths_to_delete = Vec::new();
        let mut updated_entries = Vec::new();
        let mut any_removed = false;

        let mut puffin_vector_keys = Vec::new();
        let mut cache_keys = Vec::new();

        for mut entry in entries {
            let rel_entry_path =
                crate::core::segment::relativize_against(&self.uri, &entry.file_path);
            let mut remaining_indexes = Vec::new();
            for idx in &entry.index_files {
                if idx.column_name == column.as_str() {
                    any_removed = true;
                    let resolved_path = crate::core::segment::resolve_entry_index_path(
                        &rel_entry_path,
                        &idx.file_path,
                    );
                    if let Some(offset) = idx.blob_offset {
                        cache_keys.push(format!("{}/{}:{}", self.uri, resolved_path, offset));
                    }
                    cache_keys.push(format!("{}/{}", self.uri, resolved_path));
                    if idx.index_category == "vector" {
                        puffin_vector_keys.push(
                            crate::core::index::hnsw_ivf::HnswIvfIndex::puffin_vector_cache_key(
                                &self.uri,
                                &resolved_path,
                                &idx.column_name,
                                &idx.algorithm,
                            ),
                        );
                    }
                    if !resolved_path.ends_with(".puffin") {
                        match idx.index_category.as_str() {
                            // The CSR is a triple (offsets, edges, dict)
                            "graph_v2" => {
                                paths_to_delete
                                    .push(format!("{}.graph_v2.csr.offsets", resolved_path));
                                paths_to_delete
                                    .push(format!("{}.graph_v2.csr.edges", resolved_path));
                                paths_to_delete
                                    .push(format!("{}.graph_v2.csr.dict", resolved_path));
                            }
                            "vector" => {
                                // Base and auxiliary paths for vector indexes
                                paths_to_delete.push(format!("{}.hnsw.graph", resolved_path));
                                paths_to_delete.push(format!("{}.hnsw.pq", resolved_path));
                                paths_to_delete
                                    .push(format!("{}.centroids.parquet", resolved_path));
                                paths_to_delete.push(format!("{}.mapping.parquet", resolved_path));
                                for c in 0..128 {
                                    paths_to_delete.push(format!(
                                        "{}.cluster_{}.hnsw.graph",
                                        resolved_path, c
                                    ));
                                    paths_to_delete
                                        .push(format!("{}.cluster_{}.hnsw.data", resolved_path, c));
                                }
                            }
                            "lexical" => {
                                paths_to_delete.push(resolved_path.clone());
                                if let Some(base) = resolved_path.strip_suffix(".inv.parquet") {
                                    paths_to_delete.push(format!("{}.doclen.parquet", base));
                                }
                            }
                            _ => {
                                paths_to_delete.push(resolved_path);
                            }
                        }
                    }
                } else {
                    remaining_indexes.push(idx.clone());
                }
            }

            // Check if any puffin files can now be deleted (if no other column uses them)
            for idx in &entry.index_files {
                if idx.column_name == column.as_str() && idx.file_path.ends_with(".puffin") {
                    let resolved_path = crate::core::segment::resolve_entry_index_path(
                        &rel_entry_path,
                        &idx.file_path,
                    );
                    let still_used = remaining_indexes.iter().any(|rem| {
                        crate::core::segment::resolve_entry_index_path(
                            &rel_entry_path,
                            &rem.file_path,
                        ) == resolved_path
                    });
                    if !still_used {
                        paths_to_delete.push(resolved_path);
                    }
                }
            }

            entry.index_files = remaining_indexes;
            updated_entries.push(entry);
        }

        // Commit updated segment entries removing the index from manifest entries
        if any_removed {
            manager.commit_imported_entries(updated_entries).await?;
        }

        // Commit the manifest update dropping the index from the schema
        let mut updates = HashMap::new();
        updates.insert(column.clone(), vec![]);
        self.set_index_columns(updates).await?;

        self.indexing.vector_routers.write().remove(&column);
        self.indexing.cluster_indexes.write().remove(&column);
        self.indexing.global_centroids.write().remove(&column);

        // Invalidate in-memory caches
        for key in cache_keys {
            crate::core::cache::INVERTED_INDEX_CACHE
                .invalidate(&key)
                .await;
            crate::core::cache::HNSW_IVF_CACHE.invalidate(&key).await;
            crate::core::cache::INDEX_CACHE.invalidate(&key).await;
            crate::core::cache::BYTE_CACHE.invalidate(&key).await;
            crate::core::cache::ANALYZER_META_CACHE
                .invalidate(&key)
                .await;
        }
        for key in puffin_vector_keys {
            crate::core::cache::HNSW_IVF_CACHE.invalidate(&key).await;
        }
        for path in &paths_to_delete {
            let key = format!("{}/{}", self.uri, path);
            crate::core::cache::INVERTED_INDEX_CACHE
                .invalidate(&key)
                .await;
            crate::core::cache::HNSW_IVF_CACHE.invalidate(&key).await;
            crate::core::cache::INDEX_CACHE.invalidate(&key).await;
            crate::core::cache::BYTE_CACHE.invalidate(&key).await;
            crate::core::cache::ANALYZER_META_CACHE
                .invalidate(&key)
                .await;
        }

        // Best-effort cleanup of index files
        for path in paths_to_delete {
            let p = object_store::path::Path::from(path.as_str());
            if let Err(e) = self.store.delete(&p).await {
                tracing::debug!("Failed to delete dropped index file {}: {}", path, e);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Add index columns (sync + async)
    // -----------------------------------------------------------------------

    pub fn add_index_columns(
        &mut self,
        columns: Vec<String>,
        device: Option<String>,
    ) -> Result<()> {
        {
            let mut index_cols = self.indexing.index_columns.write();
            let mut index_configs = self.indexing.index_configs.write();

            // Cascade: use the explicit device if provided, otherwise fall back to the table's default
            let effective_device = device.or_else(|| self.indexing.default_device.read().clone());

            for col in &columns {
                if !index_cols.contains(col) {
                    index_cols.push(col.clone());
                }
                index_configs.insert(
                    col.clone(),
                    crate::core::table::state::ColumnIndexConfig {
                        device: effective_device.clone(),
                        enabled: true,
                        tokenizer: None,
                        algorithms: Vec::new(),
                    },
                );
            }
            index_cols.sort();
            index_cols.dedup();
        }
        self.backfill_indexes(columns)
    }

    pub async fn add_index_columns_async(
        &mut self,
        columns: Vec<String>,
        device: Option<String>,
    ) -> Result<()> {
        {
            let mut index_cols = self.indexing.index_columns.write();
            let mut index_configs = self.indexing.index_configs.write();

            // Cascade: use the explicit device if provided, otherwise fall back to the table's default
            let effective_device = device.or_else(|| self.indexing.default_device.read().clone());

            for col in &columns {
                if !index_cols.contains(col) {
                    index_cols.push(col.clone());
                }
                index_configs.insert(
                    col.clone(),
                    crate::core::table::state::ColumnIndexConfig {
                        device: effective_device.clone(),
                        enabled: true,
                        tokenizer: None,
                        algorithms: Vec::new(),
                    },
                );
            }
            index_cols.sort();
            index_cols.dedup();
        }
        self.backfill_indexes_async(columns).await
    }

    pub fn index_all_columns(&mut self) -> Result<()> {
        self.indexing.index_all = true;
        self.backfill_indexes(Vec::new())
    }

    pub async fn index_all_columns_async(&mut self) -> Result<()> {
        self.indexing.index_all = true;
        self.backfill_indexes_async(Vec::new()).await?;
        // After building indexes, infer metadata so query planner knows about them
        self.infer_index_metadata_from_physical_async().await
    }

    // -----------------------------------------------------------------------
    // Backfill
    // -----------------------------------------------------------------------

    pub fn backfill_indexes(&self, target_columns: Vec<String>) -> Result<()> {
        self.runtime()
            .block_on(self.backfill_indexes_async(target_columns))
    }

    pub(crate) async fn backfill_indexes_async(&self, target_columns: Vec<String>) -> Result<()> {
        self.backfill_indexes_inner(target_columns, false).await
    }

    /// Rebuild `target_columns` on every segment even when the index already
    /// exists.
    ///
    /// `backfill_indexes_async` is idempotent and skips segments that already
    /// carry the required index type, which is right for `add_index` but wrong
    /// for a forced reindex: switching the *quantizer* of an existing
    /// `hnsw_tq8` index keeps the same (column, type) pair, so the skip check
    /// would leave the old bytes in place and silently report success.
    pub(crate) async fn backfill_indexes_forced(&self, target_columns: Vec<String>) -> Result<()> {
        self.backfill_indexes_inner(target_columns, true).await
    }

    async fn backfill_indexes_inner(&self, target_columns: Vec<String>, force: bool) -> Result<()> {
        use futures::StreamExt;
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manager.load_latest_full().await?;

        if all_entries.is_empty() {
            return Ok(());
        }

        // Idempotency: only rebuild segments that are actually missing a
        // required index. Without this, every `add_index` on a non-empty table
        // re-indexed every prior segment — O(n^2) work and memory, which is what
        // OOM-killed the demo load (per-chunk time grew 293s -> 885s -> 2071s).
        let required = self.required_index_types(&target_columns);
        let total = all_entries.len();
        let to_build: Vec<crate::core::manifest::ManifestEntry> = if force || required.is_empty() {
            // No configured algorithms to compare against (e.g. `index_all` or `add_index_columns`):
            // fall back to rebuilding everything.
            all_entries
        } else {
            all_entries
                .into_iter()
                .filter(|e| {
                    required
                        .iter()
                        .any(|(col, ty)| !entry_has_index(e, col, ty))
                })
                .collect()
        };
        let skipped = total - to_build.len();
        if skipped > 0 {
            tracing::info!(
                total,
                skipped,
                to_build = to_build.len(),
                "backfill: skipping segments that already carry the required indexes"
            );
        }
        if to_build.is_empty() {
            return Ok(());
        }

        // Bound the fan-out. Each build holds its segment's vectors plus the
        // HNSW/IVF/quantizer structures (several GB at a 1 GB flush size); an
        // unbounded `join_all` over every segment is what OOM-killed the
        // Wikipedia load. `buffer_unordered` caps in-flight builds, and each
        // build also takes the shared index-build gate so a backfill cannot
        // exceed the memory budget alongside concurrent writes.
        let concurrency = super::index_build_concurrency().max(1);
        let table_uri = self.uri.clone();
        let store = self.store.clone();
        let data_store = self
            .data_store
            .clone()
            .unwrap_or_else(|| self.store.clone());
        let index_columns = self.indexing.index_columns.read().clone();
        let index_configs = self.indexing.index_configs.read().clone();
        let index_all = self.indexing.index_all;
        let primary_key = self.primary_key.read().clone();
        let gate = self.index_build_gate.clone();
        // Table-level centroids for this pass, if `rebuild_indexes_with_global_centroids`
        // staged them. Empty for ordinary builds.
        let global_centroids = self.indexing.global_centroids.read().clone();

        let entries_results: Vec<Result<crate::core::manifest::ManifestEntry>> =
            futures::stream::iter(to_build.into_iter().map(|entry| {
                let table_uri = table_uri.clone();
                let store = store.clone();
                let data_store = data_store.clone();
                let target_cols = target_columns.clone();
                let index_columns = index_columns.clone();
                let index_configs = index_configs.clone();
                let primary_key = primary_key.clone();
                let gate = gate.clone();
                let global_centroids = global_centroids.clone();

                async move {
                    // Share the write path's build gate (see `Table::index_build_gate`).
                    let _permit = gate
                        .acquire_owned()
                        .await
                        .map_err(|e| anyhow::anyhow!("index build gate closed: {e}"))?;

                    let mut current_entry = entry.clone();
                    let rel_parquet_path = crate::core::segment::relativize_against(
                        &table_uri,
                        &current_entry.file_path,
                    );
                    let segment_id = crate::core::segment::segment_id_from_path(&rel_parquet_path);

                    let rel_parent = rel_parquet_path
                        .rfind('/')
                        .map(|pos| &rel_parquet_path[..pos])
                        .unwrap_or("");
                    let full_base_uri = if rel_parent.is_empty() {
                        table_uri.clone()
                    } else {
                        let base = table_uri.trim_end_matches('/');
                        format!("{}/{}", base, rel_parent)
                    };

                    let mut cols_to_index = index_columns.clone();
                    for col in &target_cols {
                        if !cols_to_index.contains(col) {
                            cols_to_index.push(col.clone());
                        }
                    }

                    let normalized_existing: Vec<crate::core::manifest::IndexFile> = current_entry
                        .index_files
                        .iter()
                        .map(|idx| {
                            let mut cloned = idx.clone();
                            let rel_idx = crate::core::segment::relativize_against(
                                &table_uri,
                                &idx.file_path,
                            );
                            cloned.file_path = crate::core::segment::resolve_entry_index_path(
                                &rel_parquet_path,
                                &rel_idx,
                            );
                            cloned
                        })
                        .collect();

                    let config = SegmentConfig::new(&full_base_uri, segment_id)
                        .with_parquet_path(rel_parquet_path.clone())
                        .with_data_store(data_store)
                        .with_index_all(index_all)
                        .with_file_checksum(current_entry.file_checksum.clone())
                        .with_columns_to_index(cols_to_index);

                    let reader = HybridReader::new(config.clone(), store.clone(), &table_uri);
                    let mut writer = HybridSegmentWriter::new(config)
                        .with_index_configs(index_configs)
                        .with_global_centroids(global_centroids)
                        .with_record_count(current_entry.record_count as usize)
                        .with_existing_stats(current_entry.column_stats.clone())
                        .with_preserve_existing_index_files(normalized_existing.clone());
                    writer.primary_key = primary_key;
                    writer.set_store(store.clone());

                    let stream = reader.stream_raw_row_groups(None, None).await?;
                    let mut stream = stream.boxed();
                    let mut current_offset = 0;
                    while let Some(batch) = stream.next().await {
                        let batch = batch?;
                        let batch_rows = batch.num_rows();
                        writer.build_indexes(&batch, current_offset)?;
                        current_offset += batch_rows;
                    }

                    writer.finish_indexing().await?;
                    writer.upload_to_store().await?;

                    let table_uri_norm = table_uri.trim_end_matches('/');

                    // Invalidate caches for this segment's parquet and old index blobs.
                    let pq_cache_key = format!("{}/{}", table_uri_norm, rel_parquet_path);
                    crate::core::cache::PARQUET_META_CACHE
                        .invalidate(&pq_cache_key)
                        .await;
                    let legacy_pq_key = format!("{}/{}", table_uri_norm, current_entry.file_path);
                    crate::core::cache::PARQUET_META_CACHE
                        .invalidate(&legacy_pq_key)
                        .await;

                    let mut updated_entry = writer.to_manifest_entry();
                    for idx in &mut updated_entry.index_files {
                        idx.file_path =
                            crate::core::segment::ensure_prefix(rel_parent, &idx.file_path);
                    }
                    let regenerated_families: std::collections::HashSet<(String, String)> =
                        updated_entry
                            .index_files
                            .iter()
                            .map(|idx| {
                                (
                                    idx.column_name.clone(),
                                    crate::core::segment::normalize_index_family(
                                        &idx.index_category,
                                    )
                                    .to_string(),
                                )
                            })
                            .collect();
                    let puffin_rel = crate::core::segment::ensure_prefix(
                        rel_parent,
                        &format!("{segment_id}.puffin"),
                    );

                    for old_idx in &normalized_existing {
                        let resolved_idx_path = &old_idx.file_path;
                        let base_key = format!("{}/{}", table_uri_norm, resolved_idx_path);
                        let raw_base_key = format!("{}/{}", table_uri, resolved_idx_path);
                        let key_with_offset = match old_idx.blob_offset {
                            Some(off) => format!("{}:{}", base_key, off),
                            None => base_key.clone(),
                        };
                        let mut extra_keys = Vec::new();
                        if raw_base_key != base_key {
                            if let Some(off) = old_idx.blob_offset {
                                extra_keys.push(format!("{}:{}", raw_base_key, off));
                            }
                            extra_keys.push(raw_base_key);
                        }
                        if old_idx.index_category == "vector" {
                            extra_keys.push(
                                crate::core::index::hnsw_ivf::HnswIvfIndex::puffin_vector_cache_key(
                                    table_uri_norm,
                                    resolved_idx_path,
                                    &old_idx.column_name,
                                    &old_idx.algorithm,
                                ),
                            );
                        }
                        crate::core::cache::invalidate_column_caches_async(
                            &old_idx.column_name,
                            &key_with_offset,
                            &base_key,
                            &extra_keys,
                        )
                        .await;
                    }

                    let gen_files = writer.get_generated_files();
                    tracing::debug!(
                        segment = %current_entry.file_path,
                        generated_files = ?gen_files,
                        "Backfill index files generated"
                    );

                    drop(writer);
                    drop(reader);

                    // Explicitly purge dirty jemalloc pages back to the kernel after every segment
                    // to prevent RSS ratchet across consecutive segments.
                    crate::core::memory::trim_heap();
                    tracing::debug!(
                        segment = %current_entry.file_path,
                        index_files = ?updated_entry.index_files,
                        "Backfill index manifest updated"
                    );
                    let mut merged_index_files: Vec<crate::core::manifest::IndexFile> =
                        normalized_existing
                            .into_iter()
                            .filter(|idx| {
                                if idx.file_path == puffin_rel {
                                    false
                                } else {
                                    !regenerated_families.contains(&(
                                        idx.column_name.clone(),
                                        crate::core::segment::normalize_index_family(
                                            &idx.index_category,
                                        )
                                        .to_string(),
                                    ))
                                }
                            })
                            .collect();
                    merged_index_files.extend(updated_entry.index_files);
                    current_entry.index_files = merged_index_files;

                    Ok(current_entry)
                }
            }))
            .buffer_unordered(concurrency)
            .collect()
            .await;

        let mut updated_entries = Vec::new();
        for res in entries_results {
            updated_entries.push(res?);
        }

        if !updated_entries.is_empty() {
            manager.commit_imported_entries(updated_entries).await?;
        }

        Ok(())
    }

    /// Rebuild the position-aware inverted index for `column` on every segment
    /// that already carries an inverted index for it.
    ///
    /// This is a *targeted* reindex. Unlike `backfill_indexes_async` —
    /// which rebuilds every configured index for a segment (including the HNSW
    /// vector indexes, tens of GB at Wikipedia scale) and skips segments that
    /// already carry an `inverted` index — this rebuilds only the
    /// `{segment}.{column}.inv.parquet` sidecar (and its `.doclen.parquet`
    /// companion) in place, leaving every other index file untouched.
    ///
    /// It exists to upgrade legacy 2-column (position-less) inverted indexes to
    /// the 3-column position-aware format, which lets the keyword-search path
    /// compute term frequency in O(1) via `ListArray::value_length` instead of
    /// counting duplicate row ids. The analyzer recorded in the existing index
    /// footer is preserved, so query semantics are unchanged — only the on-disk
    /// format (and therefore the query speed) changes.
    ///
    /// The sidecar paths are unchanged, so the manifest needs no update; any
    /// cached copy of the old file is invalidated. Returns the number of
    /// segments rebuilt.
    pub async fn reindex_inverted_column(&self, column: &str) -> Result<usize> {
        use futures::StreamExt;

        let _maintenance_guard = self.maintenance_lock.write().await;

        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _) = manager.load_latest_full().await?;

        let store = self.store.clone();
        let data_store = self
            .data_store
            .clone()
            .unwrap_or_else(|| self.store.clone());
        let table_uri = self.uri.clone();
        let mut index_configs = self.indexing.index_configs.read().clone();

        let mut rebuilt = 0usize;
        let mut updated_entries: Vec<crate::core::manifest::ManifestEntry> = Vec::new();
        for entry in entries {
            // Only segments that already carry a lexical (inverted) index on `column`.
            let inv = entry
                .index_files
                .iter()
                .find(|f| f.is_lexical() && f.column_name == column);
            let Some(inv) = inv else { continue };

            // Preserve the analyzer the existing index was built with so the
            // rebuilt index answers queries identically (only faster).
            let analyzer = read_inverted_analyzer(&store, &table_uri, &entry, inv)
                .await
                .unwrap_or_else(|| "identity".to_string());

            // Force the exact analyzer on the rebuild. `build_inverted_index`
            // reads `config.tokenizer` first, so this overrides the algorithm's
            // default (e.g. `analyzer:english` for BM25).
            {
                let cfg = index_configs.entry(column.to_string()).or_default();
                cfg.enabled = true;
                cfg.tokenizer = Some(analyzer);
            }

            // The manifest may store an absolute URI (e.g. after a relocate).
            // The writer derives its output path from `parquet_path`, so pass
            // the store-relative form the reader and store expect — otherwise
            // `finish_indexing` writes to a bogus nested `file:` path.
            let rel_parquet_path =
                crate::core::segment::relativize_against(&table_uri, &entry.file_path);
            let segment_id =
                crate::core::segment::segment_id_from_path(&rel_parquet_path).to_string();
            let rel_parent = if let Some(pos) = rel_parquet_path.rfind('/') {
                rel_parquet_path[..pos].to_string()
            } else {
                String::new()
            };
            let full_base_uri = if rel_parent.is_empty() {
                table_uri.clone()
            } else {
                format!("{}/{}", table_uri.trim_end_matches('/'), rel_parent)
            };

            let normalized_existing: Vec<crate::core::manifest::IndexFile> = entry
                .index_files
                .iter()
                .map(|idx| {
                    let mut cloned = idx.clone();
                    let rel_idx =
                        crate::core::segment::relativize_against(&table_uri, &idx.file_path);
                    cloned.file_path =
                        crate::core::segment::resolve_entry_index_path(&rel_parquet_path, &rel_idx);
                    cloned
                })
                .collect();

            let config = SegmentConfig::new(&full_base_uri, &segment_id)
                .with_parquet_path(rel_parquet_path.clone())
                .with_data_store(data_store.clone())
                .with_file_checksum(entry.file_checksum.clone())
                .with_columns_to_index(vec![column.to_string()]);

            let reader = HybridReader::new(config.clone(), store.clone(), &table_uri);
            let mut writer = HybridSegmentWriter::new(config)
                .with_index_configs(index_configs.clone())
                .with_record_count(entry.record_count as usize)
                .with_existing_stats(entry.column_stats.clone())
                .with_preserve_existing_index_files(normalized_existing.clone());
            writer.set_store(store.clone());

            // Re-tokenize only `column`; `index_column` (not `build_indexes`)
            // keeps the vector columns out of the rebuild entirely. Project the
            // read down to `column` so a segment's other (potentially huge)
            // columns — e.g. a 384-dim embedding — are never fetched.
            let projection: Option<arrow::datatypes::SchemaRef> = {
                let schema = self.schema.read();
                schema
                    .field_with_name(column)
                    .ok()
                    .map(|f| std::sync::Arc::new(arrow::datatypes::Schema::new(vec![f.clone()])))
            };
            let stream = reader.stream_raw_row_groups(None, projection).await?;
            let mut stream = stream.boxed();
            let mut offset = 0usize;
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let rows = batch.num_rows();
                if let Some(col) = batch.column_by_name(column) {
                    writer.index_column(column, col, offset)?;
                }
                offset += rows;
            }
            writer.finish_indexing().await?;

            // The Puffin bundle is rewritten in place (preserving untouched blobs),
            // and blob offsets change, so the manifest entry must be updated with
            // the new index metadata.
            let mut updated = writer.to_manifest_entry();
            for idx in &mut updated.index_files {
                idx.file_path = crate::core::segment::ensure_prefix(&rel_parent, &idx.file_path);
            }
            let regenerated_families: std::collections::HashSet<(String, String)> = updated
                .index_files
                .iter()
                .map(|idx| {
                    (
                        idx.column_name.clone(),
                        crate::core::segment::normalize_index_family(&idx.index_category)
                            .to_string(),
                    )
                })
                .collect();
            let puffin_rel =
                crate::core::segment::ensure_prefix(&rel_parent, &format!("{segment_id}.puffin"));
            let mut merged: Vec<crate::core::manifest::IndexFile> = normalized_existing
                .iter()
                .filter(|f| {
                    if f.file_path == puffin_rel {
                        false
                    } else {
                        !regenerated_families.contains(&(
                            f.column_name.clone(),
                            crate::core::segment::normalize_index_family(&f.index_category)
                                .to_string(),
                        ))
                    }
                })
                .cloned()
                .collect();
            merged.extend(updated.index_files);
            let mut new_entry = entry.clone();
            new_entry.index_files = merged;
            updated_entries.push(new_entry);

            // Any cached copy of the old bundle/blob must be dropped.
            let table_uri_norm = table_uri.trim_end_matches('/');
            let rel_inv = crate::core::segment::relativize_against(&table_uri, &inv.file_path);
            let inv_path =
                crate::core::segment::resolve_entry_index_path(&rel_parquet_path, &rel_inv);
            let base_key = format!("{}/{}", table_uri_norm, inv_path);
            let raw_base_key = format!("{}/{}", table_uri, inv_path);
            let offset_key = match inv.blob_offset {
                Some(off) => format!("{}:{}", base_key, off),
                None => base_key.clone(),
            };
            let mut extra_keys = Vec::new();
            if raw_base_key != base_key {
                if let Some(off) = inv.blob_offset {
                    extra_keys.push(format!("{}:{}", raw_base_key, off));
                }
                extra_keys.push(raw_base_key);
            }
            crate::core::cache::invalidate_column_caches_async(
                column,
                &offset_key,
                &base_key,
                &extra_keys,
            )
            .await;

            rebuilt += 1;
            tracing::info!(
                segment = %entry.file_path,
                column,
                "reindexed inverted index (position-aware)"
            );
        }

        if !updated_entries.is_empty() {
            manager.commit_imported_entries(updated_entries).await?;
        }

        Ok(rebuilt)
    }

    /// Physical index types required for `target_columns` (or every configured
    /// column when `target_columns` is empty), as `(column, index_category)` pairs.
    ///
    /// Used by [`Self::backfill_indexes_async`] to skip segments that already
    /// carry the required indexes.
    fn required_index_types(&self, target_columns: &[String]) -> Vec<(String, &'static str)> {
        let configs = self.indexing.index_configs.read();
        let cols: Vec<String> = if target_columns.is_empty() {
            configs.keys().cloned().collect()
        } else {
            target_columns.to_vec()
        };
        let mut out = Vec::new();
        for col in cols {
            if let Some(cfg) = configs.get(&col) {
                for alg in &cfg.algorithms {
                    out.push((col.clone(), physical_index_type(alg)));
                }
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Physical index inference
    // -----------------------------------------------------------------------

    /// Internal helper: Detect existing physical indexes and migrate them to logical Schema specifications.
    /// This provides zero-touch backward compatibility for tables created before the IndexSpec refactor.
    pub async fn infer_index_metadata_from_physical_async(&self) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest
            .schemas
            .last()
            .ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        // Build a set of columns that already have logical index metadata
        let indexed_columns: std::collections::HashSet<String> = latest_schema
            .fields
            .iter()
            .filter(|f| !f.indexes.is_empty())
            .map(|f| f.name.clone())
            .collect();

        // Scan latest manifest entries for physical index files. Entries live in
        // the tiered manifest list, so resolve them via `load_all_entries` — a
        // direct `manifest.entries` scan is always empty for tiered manifests and
        // would make inference a no-op.
        let manager =
            crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri);
        let entries = manager.load_all_entries(&manifest).await?;
        let mut inferred_specs: HashMap<String, Vec<IndexAlgorithm>> = HashMap::new();

        for entry in &entries {
            for index_file in &entry.index_files {
                let col_name = &index_file.column_name;
                {
                    // Skip composite index virtual columns (they are managed via CompositeBitmap)
                    if col_name.contains(',') {
                        continue;
                    }
                    // Skip columns that already have logical index metadata
                    if indexed_columns.contains(col_name) {
                        continue;
                    }

                    let algorithms = inferred_specs.entry(col_name.clone()).or_default();

                    let alg = match index_file.index_category.as_str() {
                        "vector" | "hnsw" => Some(IndexAlgorithm::Hnsw {
                            metric: "l2".to_string(),
                            complexity: 16,
                            quality: 128,
                            build_device: None,
                            search_device: None,
                        }),
                        "lexical" | "inverted" | "bm25" => Some(IndexAlgorithm::Bm25 {
                            k1: 1.5,
                            b: 0.75,
                            tokenizer: "default".to_string(),
                        }),
                        "scalar" => Some(IndexAlgorithm::Bitmap),
                        "bloom" => Some(IndexAlgorithm::Bloom { fpr: 0.05 }),
                        _ => None,
                    };

                    if let Some(a) = alg {
                        if !algorithms.contains(&a) {
                            algorithms.push(a);
                        }
                    }
                }
            }
        }

        if !inferred_specs.is_empty() {
            tracing::info!("Implicit Inference: Detected legacy physical indexes for columns {:?}. Updating manifest...", inferred_specs.keys());
            self.set_index_columns(inferred_specs).await?;
        }

        Ok(())
    }
}

/// Physical `index_category` string a configured algorithm produces in the manifest.
///
/// Mirrors the values written by the segment writer (`"vector"`, `"lexical"`,
/// `"scalar"`, `"graph_v2"`, `"bloom"`) so backfill can tell whether a segment
/// already carries the index a column needs.
fn physical_index_type(alg: &IndexAlgorithm) -> &'static str {
    match alg {
        IndexAlgorithm::Hnsw { .. }
        | IndexAlgorithm::HnswPq { .. }
        | IndexAlgorithm::HnswTq4 { .. }
        | IndexAlgorithm::HnswTq8 { .. }
        | IndexAlgorithm::HnswSq8 { .. } => "vector",
        IndexAlgorithm::Bm25 { .. } => "lexical",
        IndexAlgorithm::Bloom { .. } => "bloom",
        IndexAlgorithm::Bitmap | IndexAlgorithm::CompositeBitmap { .. } => "scalar",
        // The `graph_v2` suffix is the on-disk graph format version.
        IndexAlgorithm::CsrGraph { .. } => "graph_v2",
        IndexAlgorithm::JsonPath { .. } => "json_path",
    }
}

/// Whether two `index_category` strings denote the same index family.
///
/// The lexical family was renamed from `"inverted"` to `"lexical"`; legacy
/// manifests still carry `"inverted"` (and some paths used `"bm25"`).
/// Scalar `Bitmap` / `CompositeBitmap` indexes are also packed into Puffin as
/// Parquet posting maps under `"lexical"` (`algorithm = "inverted"`,
/// `analyzer = "identity"`), while `physical_index_type` classifies them as
/// `"scalar"`. Treating those as compatible keeps `backfill_indexes_async`
/// idempotent instead of rebuilding already-indexed scalar segments.
fn category_matches(actual: &str, expected: &str) -> bool {
    if actual == expected {
        return true;
    }
    let lexical = |s: &str| matches!(s, "lexical" | "inverted" | "bm25");
    if lexical(actual) && lexical(expected) {
        return true;
    }
    expected == "scalar" && matches!(actual, "lexical" | "inverted" | "bitmap")
}

/// Whether a manifest entry already carries an index of `ty` on `col`.
fn entry_has_index(entry: &crate::core::manifest::ManifestEntry, col: &str, ty: &str) -> bool {
    entry
        .index_files
        .iter()
        .any(|f| f.column_name == col && category_matches(&f.index_category, ty))
}

/// Read the `analyzer` key/value from an existing inverted-index footer.
///
/// Used by [`Table::reindex_inverted_column`] to preserve the tokenization the
/// index was originally built with. Returns `None` when the file is missing or
/// carries no analyzer metadata (legacy indexes), in which case the caller
/// falls back to `identity`.
async fn read_inverted_analyzer(
    store: &std::sync::Arc<dyn object_store::ObjectStore>,
    table_uri: &str,
    entry: &crate::core::manifest::ManifestEntry,
    inv: &crate::core::manifest::IndexFile,
) -> Option<String> {
    let rel_parquet = crate::core::segment::relativize_against(table_uri, &entry.file_path);
    let rel_inv = crate::core::segment::relativize_against(table_uri, &inv.file_path);
    let full = crate::core::segment::resolve_entry_index_path(&rel_parquet, &rel_inv);
    let path = object_store::path::Path::from(full.as_str());

    // Helper to extract `"analyzer"` from a Parquet payload's KV metadata.
    let extract_from_parquet = |raw: bytes::Bytes| -> Option<String> {
        let builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(raw).ok()?;
        let kv_list = builder.metadata().file_metadata().key_value_metadata()?;
        for kv in kv_list {
            if kv.key == "analyzer" {
                return kv.value.clone();
            }
        }
        None
    };

    if full.ends_with(".puffin") || inv.blob_offset.is_some() {
        // Read only the Puffin footer (tail + footer JSON) so large HNSW blobs aren't fetched.
        let size = store.head(&path).await.ok()?.size;
        if size >= 12 {
            if let Ok(tail) = store.get_range(&path, (size - 12)..size).await {
                if tail.len() == 12 && tail[8..12] == crate::core::puffin::PUFFIN_MAGIC {
                    let footer_size =
                        u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as u64;
                    let flags = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
                    if footer_size + 12 <= size {
                        let footer_start = size - 12 - footer_size;
                        if let Ok(footer_bytes) =
                            store.get_range(&path, footer_start..(size - 12)).await
                        {
                            let footer_payload = if flags & 0x01 != 0 {
                                zstd::decode_all(&footer_bytes[..]).ok()?
                            } else {
                                footer_bytes.to_vec()
                            };
                            if let Ok(footer) = serde_json::from_slice::<
                                crate::core::puffin::PuffinFooter,
                            >(&footer_payload)
                            {
                                let matched_blob = inv
                                    .blob_offset
                                    .zip(inv.blob_length)
                                    .and_then(|(off, len)| {
                                        footer
                                            .blobs
                                            .iter()
                                            .find(|b| b.offset == off && b.length == len)
                                    })
                                    .or_else(|| {
                                        footer.blobs.iter().find(|b| {
                                            b.r#type
                                                == crate::core::puffin::PUFFIN_BLOB_LEXICAL_BM25
                                        })
                                    });
                                if let Some(blob) = matched_blob {
                                    if let Some(a) = blob.properties.get("analyzer") {
                                        return Some(a.clone());
                                    }
                                    // Fallback to reading the single blob range's Parquet KV metadata.
                                    if blob.offset >= 0 && blob.length > 0 {
                                        let range = (blob.offset as u64)
                                            ..((blob.offset as u64)
                                                .saturating_add(blob.length as u64));
                                        if let Ok(raw) = store.get_range(&path, range).await {
                                            let decoded = if blob.compression_codec.as_deref()
                                                == Some("zstd")
                                            {
                                                zstd::decode_all(&raw[..])
                                                    .ok()
                                                    .map(bytes::Bytes::from)
                                            } else {
                                                Some(raw)
                                            };
                                            if let Some(buf) = decoded {
                                                if let Some(a) = extract_from_parquet(buf) {
                                                    return Some(a);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return None;
    }

    // Legacy standalone `.inv.parquet` file.
    let bytes = store.get(&path).await.ok()?.bytes().await.ok()?;
    extract_from_parquet(bytes)
}

#[cfg(test)]
mod backfill_tests {
    use super::*;
    use crate::core::manifest::{IndexAlgorithm, IndexFile, ManifestEntry};

    fn tq8() -> IndexAlgorithm {
        IndexAlgorithm::HnswTq8 {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 200,
        }
    }

    #[test]
    fn physical_index_type_maps_vector_family_to_vector() {
        // The manifest records the vector family as "vector", regardless of the
        // concrete quantization (hnsw / tq4 / tq8), so the skip check must too.
        assert_eq!(physical_index_type(&tq8()), "vector");
        assert_eq!(physical_index_type(&IndexAlgorithm::Bitmap), "scalar");
        assert_eq!(
            physical_index_type(&IndexAlgorithm::Bm25 {
                k1: 1.5,
                b: 0.75,
                tokenizer: "default".to_string(),
            }),
            "lexical"
        );
    }

    #[test]
    fn entry_has_index_matches_column_and_type() {
        let entry = ManifestEntry {
            index_files: vec![
                IndexFile {
                    file_path: "seg.title.inv.parquet".to_string(),
                    index_category: "inverted".to_string(),
                    column_name: "title".to_string(),
                    ..Default::default()
                },
                IndexFile {
                    file_path: "seg.embedding.tq8.centroids.parquet".to_string(),
                    index_category: "vector".to_string(),
                    column_name: "embedding".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert!(entry_has_index(&entry, "title", "inverted"));
        assert!(entry_has_index(&entry, "embedding", "vector"));
        // The lexical family is aliased: a legacy `inverted` entry satisfies a
        // `lexical` requirement (and vice versa), so the rename does not force a
        // rebuild of every legacy segment.
        assert!(entry_has_index(&entry, "title", "lexical"));
        assert!(entry_has_index(&entry, "title", "bm25"));
        // Wrong column, wrong type, or an unknown column must not match — an
        // over-eager match would silently skip a needed rebuild.
        assert!(!entry_has_index(&entry, "title", "vector"));
        assert!(!entry_has_index(&entry, "embedding", "inverted"));
        assert!(!entry_has_index(&entry, "missing", "inverted"));
    }

    #[test]
    fn an_indexed_segment_is_reported_complete() {
        // A segment carrying every required (column, type) pair is complete, so
        // `backfill_indexes_async` skips it instead of rebuilding it.
        let entry = ManifestEntry {
            index_files: vec![
                IndexFile {
                    file_path: "seg.title.inv.parquet".to_string(),
                    index_category: "inverted".to_string(),
                    column_name: "title".to_string(),
                    ..Default::default()
                },
                IndexFile {
                    file_path: "seg.embedding.tq8.centroids.parquet".to_string(),
                    index_category: "vector".to_string(),
                    column_name: "embedding".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let required = [
            ("embedding".to_string(), "vector"),
            ("title".to_string(), "inverted"),
        ];
        let needs_build = required
            .iter()
            .any(|(col, ty)| !entry_has_index(&entry, col, ty));
        assert!(!needs_build, "fully-indexed segment must be skipped");
    }
}
