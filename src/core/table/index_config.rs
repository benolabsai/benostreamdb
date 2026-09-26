// Copyright (c) 2026 Richard Albright. All rights reserved.

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

        // Trigger backfill for updated columns. The backfill bounds its own
        // per-segment concurrency through the index-build gate (see
        // `backfill_indexes_async`), so no outer permit is held here — holding
        // one would starve the inner per-segment permits when the gate is 1.
        let cols_to_backfill: Vec<String> = column_indexes.keys().cloned().collect();
        let table_clone = self.clone();
        let handle = tokio::spawn(async move {
            if let Err(e) = table_clone.backfill_indexes_async(cols_to_backfill).await {
                tracing::error!("Failed to backfill indexes: {}", e);
            }
        });

        self.background_tasks.lock().await.push(handle);

        Ok(())
    }

    pub async fn add_index(&self, column: String, algorithm: IndexAlgorithm) -> Result<()> {
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
            | IndexAlgorithm::HnswTq8 { .. } => {
                // Vector index family - replace existing ones
                next_indexes.retain(|idx| {
                    !matches!(
                        idx,
                        IndexAlgorithm::Hnsw { .. }
                            | IndexAlgorithm::HnswPq { .. }
                            | IndexAlgorithm::HnswTq4 { .. }
                            | IndexAlgorithm::HnswTq8 { .. }
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
    /// This is an atomic operation that commits a new manifest version.
    pub async fn drop_index(&self, column: String) -> Result<()> {
        // Collect all file paths associated with this index from the current manifest.
        // NOTE: entries live in the tiered manifest list, not inline in
        // `Manifest.entries` (which is empty for tiered manifests), so we must
        // resolve them through `load_all_entries` — otherwise `drop_index` would
        // silently leave every index file orphaned on disk.
        let manifest = self.manifest().await?;
        let manager = crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri);
        let entries = manager.load_all_entries(&manifest).await?;
        let mut paths_to_delete = Vec::new();

        for entry in &entries {
            for idx in &entry.index_files {
                if idx.column_name.as_deref() == Some(column.as_str()) {
                    match idx.index_type.as_str() {
                        // Both the legacy v1 and the current v2 graph formats
                        // must be cleaned up so a drop leaves no orphaned CSRs.
                        // The CSR is a triple (offsets, edges, dict) — omitting
                        // the `.dict` sidecar left it orphaned on every drop.
                        "graph" | "graph_v2" => {
                            paths_to_delete.push(format!("{}.graph.csr.offsets", idx.file_path));
                            paths_to_delete.push(format!("{}.graph.csr.edges", idx.file_path));
                            paths_to_delete.push(format!("{}.graph.csr.dict", idx.file_path));
                            paths_to_delete.push(format!("{}.graph_v2.csr.offsets", idx.file_path));
                            paths_to_delete.push(format!("{}.graph_v2.csr.edges", idx.file_path));
                            paths_to_delete.push(format!("{}.graph_v2.csr.dict", idx.file_path));
                        }
                        "vector" => {
                            // Base paths for vector indexes
                            paths_to_delete.push(format!("{}.hnsw.graph", idx.file_path));
                            paths_to_delete.push(format!("{}.hnsw.pq", idx.file_path));
                        }
                        _ => {
                            paths_to_delete.push(idx.file_path.clone());
                        }
                    }
                }
            }
        }

        // Commit the manifest update dropping the index from the schema
        let mut updates = HashMap::new();
        updates.insert(column.clone(), vec![]);
        self.set_index_columns(updates).await?;

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

    pub(crate) fn backfill_indexes(&self, target_columns: Vec<String>) -> Result<()> {
        self.runtime()
            .block_on(self.backfill_indexes_async(target_columns))
    }

    pub(crate) async fn backfill_indexes_async(&self, target_columns: Vec<String>) -> Result<()> {
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
        let to_build: Vec<crate::core::manifest::ManifestEntry> = if required.is_empty() {
            // No configured algorithms to compare against (e.g. `index_all`):
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

                async move {
                    // Share the write path's build gate (see `Table::index_build_gate`).
                    let _permit = gate
                        .acquire_owned()
                        .await
                        .map_err(|e| anyhow::anyhow!("index build gate closed: {e}"))?;

                    let mut current_entry = entry.clone();
                    let file_path_str = current_entry.file_path.clone();
                    let segment_id = file_path_str
                        .split('/')
                        .next_back()
                        .unwrap_or(&file_path_str)
                        .strip_suffix(".parquet")
                        .unwrap_or(&file_path_str);

                    let rel_parent = if let Some(pos) = file_path_str.rfind('/') {
                        &file_path_str[..pos]
                    } else {
                        ""
                    };
                    let full_base_uri = if rel_parent.is_empty() {
                        table_uri.clone()
                    } else {
                        let base = table_uri.trim_end_matches('/');
                        format!("{}/{}", base, rel_parent)
                    };

                    let mut cols_to_index = index_columns.clone();
                    for col in target_cols {
                        if !cols_to_index.contains(&col) {
                            cols_to_index.push(col);
                        }
                    }

                    let config = SegmentConfig::new(&full_base_uri, segment_id)
                        .with_parquet_path(current_entry.file_path.clone())
                        .with_data_store(data_store)
                        .with_index_all(index_all)
                        .with_columns_to_index(cols_to_index);

                    let reader = HybridReader::new(config.clone(), store.clone(), &table_uri);
                    let mut writer = HybridSegmentWriter::new(config)
                        .with_index_configs(index_configs)
                        .with_record_count(current_entry.record_count as usize)
                        .with_existing_stats(current_entry.column_stats.clone());
                    writer.primary_key = primary_key;
                    writer.set_store(store.clone());

                    let stream = reader.stream_row_groups(None, None).await?;
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

                    // Invalidate the cache for this segment
                    let cache_key = format!("{}/{}", table_uri, current_entry.file_path);
                    crate::core::cache::PARQUET_META_CACHE
                        .invalidate(&cache_key)
                        .await;

                    let gen_files = writer.get_generated_files();
                    tracing::debug!(
                        segment = %current_entry.file_path,
                        generated_files = ?gen_files,
                        "Backfill index files generated"
                    );

                    let updated_entry = writer.to_manifest_entry();
                    tracing::debug!(
                        segment = %current_entry.file_path,
                        index_files = ?updated_entry.index_files,
                        "Backfill index manifest updated"
                    );
                    current_entry.index_files = updated_entry.index_files;

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

    /// Physical index types required for `target_columns` (or every configured
    /// column when `target_columns` is empty), as `(column, index_type)` pairs.
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
        let manager = crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri);
        let entries = manager.load_all_entries(&manifest).await?;
        let mut inferred_specs: HashMap<String, Vec<IndexAlgorithm>> = HashMap::new();

        for entry in &entries {
            for index_file in &entry.index_files {
                if let Some(col_name) = &index_file.column_name {
                    // Skip composite index virtual columns (they are managed via CompositeBitmap)
                    if col_name.contains(',') {
                        continue;
                    }
                    // Skip columns that already have logical index metadata
                    if indexed_columns.contains(col_name) {
                        continue;
                    }

                    let algorithms = inferred_specs.entry(col_name.clone()).or_default();

                    let alg = match index_file.index_type.as_str() {
                        "vector" | "hnsw" => Some(IndexAlgorithm::Hnsw {
                            metric: "l2".to_string(),
                            complexity: 16,
                            quality: 128,
                            build_device: None,
                            search_device: None,
                        }),
                        "inverted" => Some(IndexAlgorithm::Bm25 {
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

/// Physical `index_type` string a configured algorithm produces in the manifest.
///
/// Mirrors the values written by the segment writer (`"vector"`, `"inverted"`,
/// `"scalar"`, `"graph_v2"`, `"bloom"`) so backfill can tell whether a segment
/// already carries the index a column needs.
fn physical_index_type(alg: &IndexAlgorithm) -> &'static str {
    match alg {
        IndexAlgorithm::Hnsw { .. }
        | IndexAlgorithm::HnswPq { .. }
        | IndexAlgorithm::HnswTq4 { .. }
        | IndexAlgorithm::HnswTq8 { .. } => "vector",
        IndexAlgorithm::Bm25 { .. } => "inverted",
        IndexAlgorithm::Bloom { .. } => "bloom",
        IndexAlgorithm::Bitmap | IndexAlgorithm::CompositeBitmap { .. } => "scalar",
        // The v2 suffix is the on-disk graph format version. Reporting it here
        // makes `backfill_indexes_async` treat a segment carrying only legacy
        // v1 graph files as incomplete, so it is rebuilt in the correct format.
        IndexAlgorithm::CsrGraph { .. } => "graph_v2",
    }
}

/// Whether a manifest entry already carries an index of `ty` on `col`.
fn entry_has_index(entry: &crate::core::manifest::ManifestEntry, col: &str, ty: &str) -> bool {
    entry
        .index_files
        .iter()
        .any(|f| f.column_name.as_deref() == Some(col) && f.index_type == ty)
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
            "inverted"
        );
    }

    #[test]
    fn entry_has_index_matches_column_and_type() {
        let entry = ManifestEntry {
            index_files: vec![
                IndexFile {
                    file_path: "seg.title.inv.parquet".to_string(),
                    index_type: "inverted".to_string(),
                    column_name: Some("title".to_string()),
                    ..Default::default()
                },
                IndexFile {
                    file_path: "seg.embedding.tq8.centroids.parquet".to_string(),
                    index_type: "vector".to_string(),
                    column_name: Some("embedding".to_string()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert!(entry_has_index(&entry, "title", "inverted"));
        assert!(entry_has_index(&entry, "embedding", "vector"));
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
                    index_type: "inverted".to_string(),
                    column_name: Some("title".to_string()),
                    ..Default::default()
                },
                IndexFile {
                    file_path: "seg.embedding.tq8.centroids.parquet".to_string(),
                    index_type: "vector".to_string(),
                    column_name: Some("embedding".to_string()),
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
