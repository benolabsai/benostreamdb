// Copyright (c) 2026 Richard Albright. All rights reserved.

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
use crate::core::manifest::{ManifestManager, IndexAlgorithm};
use crate::core::reader::HybridReader;
use crate::core::segment::HybridSegmentWriter;
use crate::SegmentConfig;

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
    pub async fn set_index_columns(&self, column_indexes: HashMap<String, Vec<IndexAlgorithm>>) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_index_specs(column_indexes.clone()).await?;

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
                    let tokenizer = algs.iter().find_map(|alg| {
                        match alg {
                            IndexAlgorithm::Bm25 { tokenizer, .. } => {
                                if tokenizer.is_empty() || tokenizer == "default" {
                                    Some("default".to_string())
                                } else {
                                    Some(tokenizer.clone())
                                }
                            },
                            _ => None,
                        }
                    });

                    // Update config
                    let config = index_configs.entry(col.clone()).or_insert_with(|| crate::core::table::state::ColumnIndexConfig {
                        enabled: true,
                        algorithms: algs.clone(),
                        ..Default::default()
                    });
                    config.algorithms = algs.clone();
                    if let Some(tok) = tokenizer {
                        config.tokenizer = Some(tok);
                    }
                }
            }
            index_cols.sort();
        }

        // Trigger backfill for updated columns
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
                let config = index_configs.entry(column.clone()).or_insert_with(|| crate::core::table::state::ColumnIndexConfig {
                    enabled: true,
                    ..Default::default()
                });
                config.algorithms.push(algorithm);

                let mut index_cols = self.indexing.index_columns.write();
                if !index_cols.contains(&column) {
                    index_cols.push(column);
                }
                return Ok(());
            }
        };

        let field = latest_schema.fields.iter()
            .find(|f| f.name == column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", column))?;

        let mut next_indexes = field.indexes.clone();

        // Deduplicate by type for unique algorithms (Vector families, BM25, Bloom)
        match &algorithm {
            IndexAlgorithm::Hnsw { .. } | IndexAlgorithm::HnswPq { .. } |
            IndexAlgorithm::HnswTq4 { .. } | IndexAlgorithm::HnswTq8 { .. } => {
                // Vector index family - replace existing ones
                next_indexes.retain(|idx| !matches!(idx,
                    IndexAlgorithm::Hnsw { .. } | IndexAlgorithm::HnswPq { .. } |
                    IndexAlgorithm::HnswTq4 { .. } | IndexAlgorithm::HnswTq8 { .. }));
            }
            IndexAlgorithm::Bm25 { .. } => {
                next_indexes.retain(|idx| !matches!(idx, IndexAlgorithm::Bm25 { .. }));
            }
            IndexAlgorithm::Bloom { .. } => {
                next_indexes.retain(|idx| !matches!(idx, IndexAlgorithm::Bloom { .. }));
            }
            _ => {
                if !next_indexes.contains(&algorithm) {
                    next_indexes.push(algorithm.clone());
                    return self.set_index_columns(HashMap::from([(column, next_indexes)])).await;
                }
            }
        }

        next_indexes.push(algorithm);

        let mut updates = HashMap::new();
        updates.insert(column, next_indexes);

        self.set_index_columns(updates).await
    }

    /// Remove all indexing strategies from a column.
    /// This is an atomic operation that commits a new manifest version.
    pub async fn drop_index(&self, column: String) -> Result<()> {
        let mut updates = HashMap::new();
        updates.insert(column, vec![]);
        self.set_index_columns(updates).await
    }

    // -----------------------------------------------------------------------
    // Add index columns (sync + async)
    // -----------------------------------------------------------------------

    pub fn add_index_columns(&mut self, columns: Vec<String>, device: Option<String>) -> Result<()> {
        {
            let mut index_cols = self.indexing.index_columns.write();
            let mut index_configs = self.indexing.index_configs.write();

            // Cascade: use the explicit device if provided, otherwise fall back to the table's default
            let effective_device = device.or_else(|| self.indexing.default_device.read().clone());

            for col in &columns {
                if !index_cols.contains(col) {
                    index_cols.push(col.clone());
                }
                index_configs.insert(col.clone(), crate::core::table::state::ColumnIndexConfig { device: effective_device.clone(), enabled: true, tokenizer: None, algorithms: Vec::new() });
            }
            index_cols.sort();
            index_cols.dedup();
        }
        self.backfill_indexes(columns)
    }

    pub async fn add_index_columns_async(&mut self, columns: Vec<String>, device: Option<String>) -> Result<()> {
        {
            let mut index_cols = self.indexing.index_columns.write();
            let mut index_configs = self.indexing.index_configs.write();

            // Cascade: use the explicit device if provided, otherwise fall back to the table's default
            let effective_device = device.or_else(|| self.indexing.default_device.read().clone());

            for col in &columns {
                if !index_cols.contains(col) {
                    index_cols.push(col.clone());
                }
                index_configs.insert(col.clone(), crate::core::table::state::ColumnIndexConfig { device: effective_device.clone(), enabled: true, tokenizer: None, algorithms: Vec::new() });
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
        self.runtime().block_on(self.backfill_indexes_async(target_columns))
    }

    async fn backfill_indexes_async(&self, target_columns: Vec<String>) -> Result<()> {
        use futures::StreamExt;
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manager.load_latest_full().await?;

        if all_entries.is_empty() {
            return Ok(());
        }

        let entries_results: Vec<Result<crate::core::manifest::ManifestEntry>> = futures::future::join_all(all_entries.iter().map(|entry| {
            let entry = entry.clone();
            let table_uri = self.uri.clone();
            let store = self.store.clone();
            let data_store = self.data_store.clone().unwrap_or(self.store.clone());
            let target_cols = target_columns.clone();

            async move {
                let mut current_entry = entry.clone();
                let file_path_str = current_entry.file_path.clone();
                let segment_id = file_path_str.split('/').next_back().unwrap_or(&file_path_str)
                    .strip_suffix(".parquet").unwrap_or(&file_path_str);

                let mut cols_to_index = self.indexing.index_columns.read().clone();
                for col in target_cols {
                    if !cols_to_index.contains(&col) {
                        cols_to_index.push(col);
                    }
                }

                let config = SegmentConfig::new(&table_uri, segment_id)
                    .with_parquet_path(current_entry.file_path.clone())
                    .with_data_store(data_store)
                    .with_index_all(self.indexing.index_all)
                    .with_columns_to_index(cols_to_index);

                let reader = HybridReader::new(config.clone(), store.clone(), &table_uri);
                let mut writer = HybridSegmentWriter::new(config)
                    .with_index_configs(self.indexing.index_configs.read().clone())
                    .with_record_count(current_entry.record_count as usize)
                    .with_existing_stats(current_entry.column_stats.clone());
                writer.primary_key = self.primary_key.read().clone();
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
                crate::core::cache::PARQUET_META_CACHE.invalidate(&cache_key).await;

                let gen_files = writer.get_generated_files();
                println!("backfill: segment={}, generated_files={:?}", current_entry.file_path, gen_files);

                let updated_entry = writer.to_manifest_entry();
                println!("backfill: segment={}, index_files={:?}", current_entry.file_path, updated_entry.index_files);
                current_entry.index_files = updated_entry.index_files;

                Ok(current_entry)
            }
        })).await;

        let mut updated_entries = Vec::new();
        for res in entries_results {
            updated_entries.push(res?);
        }

        if !updated_entries.is_empty() {
            manager.commit_imported_entries(updated_entries).await?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Physical index inference
    // -----------------------------------------------------------------------

    /// Internal helper: Detect existing physical indexes and migrate them to logical Schema specifications.
    /// This provides zero-touch backward compatibility for tables created before the IndexSpec refactor.
    pub async fn infer_index_metadata_from_physical_async(&self) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest.schemas.last().ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        // Build a set of columns that already have logical index metadata
        let indexed_columns: std::collections::HashSet<String> = latest_schema.fields.iter()
            .filter(|f| !f.indexes.is_empty())
            .map(|f| f.name.clone())
            .collect();

        // Scan latest manifest entries for physical index files
        println!("infer: checking {} entries", manifest.entries.len());
        let mut inferred_specs: HashMap<String, Vec<IndexAlgorithm>> = HashMap::new();

        for entry in &manifest.entries {
            for index_file in &entry.index_files {
                if let Some(col_name) = &index_file.column_name {
                    // Skip columns that already have logical index metadata
                    if indexed_columns.contains(col_name) {
                        continue;
                    }

                    let algorithms = inferred_specs.entry(col_name.clone()).or_insert_with(Vec::new);

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
