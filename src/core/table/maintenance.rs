// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Table maintenance, compaction, vacuuming, and file management.
///
/// Contains methods on `Table` for:
/// - `rewrite_data_files`, `compact`, `rewrite_data_files_async`
/// - `update_schema`, `rollback_to_snapshot`
/// - `vacuum`, `vacuum_async`
/// - `delete`, `delete_async`
/// - `remove_orphan_files`
/// - `shuffle_batch_by_centroids`
use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::sync::Arc;

use super::Table;
use crate::core::compaction::{CompactionOptions, Compactor};
use crate::core::maintenance::Maintenance;
use crate::core::manifest::ManifestManager;
use crate::core::planner::{FilterExpr, QueryPlanner};
use crate::core::reader::HybridReader;
use crate::SegmentConfig;

impl Table {
    /// Rewrite data files to optimize snapshots (Compaction)
    pub fn rewrite_data_files(&self, options: Option<CompactionOptions>) -> Result<()> {
        self.runtime()
            .block_on(self.rewrite_data_files_async(options))
    }

    /// Legacy alias for rewrite_data_files
    pub fn compact(&self, options: Option<CompactionOptions>) -> Result<()> {
        self.rewrite_data_files(options)
    }

    /// Rewrite data files (Asynchronous)
    pub async fn rewrite_data_files_async(&self, options: Option<CompactionOptions>) -> Result<()> {
        // Flush before compaction to include recent writes
        self.flush_async().await?;

        let opts = options.unwrap_or_default();
        let compactor = Compactor::new(&self.uri, opts)?;
        compactor.rewrite_data_files().await
    }

    /// Update the table schema (Evolution)
    pub async fn update_schema(&self, new_schema: crate::core::manifest::Schema) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, _, _) = manifest_manager.load_latest_full().await?;

        let new_schema_id = manifest
            .schemas
            .iter()
            .map(|s| s.schema_id)
            .max()
            .unwrap_or(0)
            + 1;

        let mut new_schemas = manifest.schemas.clone();
        let mut schema_to_add = new_schema.clone();
        schema_to_add.schema_id = new_schema_id;
        new_schemas.push(schema_to_add);

        let max_id = new_schema.fields.iter().map(|f| f.id).max().unwrap_or(0);

        manifest_manager
            .update_schema(new_schemas, new_schema_id, Some(max_id))
            .await?;

        // Reload local schema
        let mut local_schema = self.schema.write();
        *local_schema = Arc::new(new_schema.to_arrow());

        Ok(())
    }

    /// Rollback table to a specific snapshot ID
    pub async fn rollback_to_snapshot(&self, snapshot_id: i64) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager
            .rollback_to_snapshot(snapshot_id as u64)
            .await?;
        Ok(())
    }

    /// Physically delete unreferenced data and manifest files
    pub fn vacuum(&self, retention_versions: usize) -> Result<usize> {
        self.runtime()
            .block_on(self.vacuum_async(retention_versions))
    }

    /// Async implementation of vacuum
    pub async fn vacuum_async(&self, retention_versions: usize) -> Result<usize> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.vacuum(retention_versions).await
    }

    /// Delete rows matching a SQL-like filter string (Synchronous)
    pub fn delete(&self, filter: &str) -> Result<()> {
        self.runtime().block_on(self.delete_async(filter))
    }

    /// Async implementation of delete
    pub async fn delete_async(&self, filter: &str) -> Result<()> {
        use futures::StreamExt;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

        if all_entries.is_empty() {
            return Ok(());
        }

        let planner = QueryPlanner::new();
        let arrow_schema = self.arrow_schema();
        let expr = FilterExpr::parse_sql(filter, arrow_schema)
            .await
            .context("Failed to parse delete filter")?;

        let candidates = planner.prune_entries(&all_entries, Some(&expr), None);
        let candidate_paths: std::collections::HashSet<String> = candidates
            .iter()
            .map(|(e, _)| e.file_path.clone())
            .collect();
        tracing::debug!(
            "delete_async: filter='{}', potential candidates: {}",
            filter,
            candidates.len()
        );

        let mut all_updated_entries = Vec::new();

        for entry in all_entries {
            if !candidate_paths.contains(&entry.file_path) {
                // Preserve non-candidate segments as-is
                all_updated_entries.push(entry);
                continue;
            }

            tracing::debug!("delete_async: processing segment {}", entry.file_path);
            let file_path_str = entry.file_path.clone();

            // Fix path resolution: find correct physical subdirectory
            let path = std::path::Path::new(&file_path_str);
            let rel_parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
            let full_base_path = if rel_parent.is_empty() {
                self.uri.clone()
            } else {
                format!("{}/{}", self.uri, rel_parent)
            };

            let segment_id = file_path_str
                .split('/')
                .next_back()
                .unwrap_or(&file_path_str)
                .strip_suffix(".parquet")
                .unwrap_or(&file_path_str);

            let config = SegmentConfig::new(&full_base_path, segment_id)
                .with_index_files(entry.index_files.clone())
                .with_record_count(entry.record_count as u64);
            let reader = HybridReader::new(config, self.store.clone(), &self.uri);

            let mut new_deletes = Vec::new();

            // OPTIMIZATION: Check for Sidecar Scalar Index (Inverted Index)
            let and_filters = expr.extract_and_conditions();
            let mut bitmap_opt: Option<roaring::RoaringBitmap> = None;

            if !and_filters.is_empty() {
                for filter in and_filters {
                    if let Ok(Some(bm)) = reader.get_scalar_filter_bitmap(&filter).await {
                        if let Some(current) = bitmap_opt {
                            bitmap_opt = Some(current & bm);
                        } else {
                            bitmap_opt = Some(bm);
                        }
                    } else {
                        // If any part of the AND is MISSING an index, fallback to scan
                        bitmap_opt = None;
                        break;
                    }
                }
            }

            if let Some(bitmap) = bitmap_opt {
                // Index HIT! We found the deleted rows instantly.
                for row_id in bitmap.iter() {
                    new_deletes.push(row_id as i64);
                }
            } else {
                // Index MISS: Full file scan (fallback)
                let mut stream = reader
                    .stream_all(None as Option<Arc<arrow::datatypes::Schema>>)
                    .await?;
                let mut current_row_offset = 0;
                while let Some(batch_res) = stream.next().await {
                    let batch = batch_res?;
                    let num_rows = batch.num_rows();
                    let mask = planner.evaluate_expr(&batch, &expr)?;
                    for i in 0..num_rows {
                        if mask.value(i) {
                            new_deletes.push((current_row_offset + i) as i64);
                        }
                    }
                    current_row_offset += num_rows;
                }
            }

            if !new_deletes.is_empty() {
                // Generate NEW Position Delete File
                let mut file_paths = arrow::array::StringBuilder::new();
                let mut positions = arrow::array::Int64Builder::new();

                for &pos in &new_deletes {
                    file_paths.append_value(&entry.file_path);
                    positions.append_value(pos);
                }

                let file_path_array = file_paths.finish();
                let pos_array = positions.finish();

                let delete_writer = crate::core::iceberg::iceberg_delete::IcebergDeleteWriter::new(
                    self.uri.clone(),
                    2, // Format V2
                );

                let partition_data = if !entry.partition_values.is_empty() {
                    let path = std::path::Path::new(&entry.file_path);
                    let rel_path = path
                        .parent()
                        .and_then(|p| p.to_str())
                        .unwrap_or("")
                        .trim_start_matches('/')
                        .to_string();

                    Some((rel_path, entry.partition_values.clone()))
                } else {
                    None
                };

                let delete_file = delete_writer
                    .write_position_delete(partition_data, &file_path_array, &pos_array)
                    .await?;

                let mut new_entry = entry.clone();
                new_entry.delete_files.push(delete_file);
                all_updated_entries.push(new_entry);
            } else {
                all_updated_entries.push(entry.clone());
            }
        }

        if !all_updated_entries.is_empty() {
            // Commit the entire updated state.
            manifest_manager
                .commit(
                    &all_updated_entries,
                    &[],
                    crate::core::manifest::CommitMetadata::default(),
                )
                .await?;
        }

        Ok(())
    }

    /// Remove orphan files
    pub fn remove_orphan_files(&self, older_than_days: u64) -> Result<()> {
        self.runtime().block_on(async {
            let maintenance = Maintenance::new(&self.uri)?;
            let older_than_ms = older_than_days * 24 * 60 * 60 * 1000;
            maintenance.remove_orphan_files(older_than_ms as i64).await
        })
    }

    #[allow(dead_code)]
    fn get_vector_column_for_shuffling(&self, batch: &RecordBatch) -> Option<String> {
        let index_cols = self.indexing.index_columns.read();
        for col_name in index_cols.iter() {
            if let Ok(idx) = batch.schema().index_of(col_name) {
                let col = batch.column(idx);
                if matches!(col.data_type(), arrow::datatypes::DataType::FixedSizeList(inner, _)
                  if *inner.data_type() == arrow::datatypes::DataType::Float32)
                {
                    return Some(col_name.clone());
                }
            }
        }

        batch
            .schema()
            .fields()
            .iter()
            .find(|f| {
                matches!(f.data_type(), arrow::datatypes::DataType::FixedSizeList(inner, _)
                  if *inner.data_type() == arrow::datatypes::DataType::Float32)
            })
            .map(|f| f.name().clone())
    }

    #[allow(dead_code)]
    pub(crate) async fn shuffle_batch_by_centroids(
        &self,
        batch: &RecordBatch,
        col_name: &str,
    ) -> Result<RecordBatch> {
        use crate::core::index::gpu::get_thread_gpu_context;
        use crate::core::index::ivf::simple_kmeans;
        use arrow::array::Int32Array;

        let col_idx = batch.schema().index_of(col_name)?;
        let list_array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
            .ok_or_else(|| anyhow::anyhow!("Column '{}' must be a FixedSizeListArray", col_name))?;

        let n = list_array.len();
        if n < 1024 {
            return Ok(batch.clone());
        }

        // 1. Convert vectors to Vec<Vec<f32>> for K-Means (Training step)
        let vectors: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .step_by(n / 1000 + 1)
            .map(|i| {
                list_array
                    .value(i)
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .map(|a| a.values().to_vec())
                    .unwrap_or_default()
            })
            .collect();

        // 2. Train centroids (Sampled)
        let k = (n as f64).sqrt() as usize;
        let k = k.clamp(16, 1024);
        let (centroids, _) = simple_kmeans(&vectors, k, 3)?;

        // 3. Assign all vectors (GPU Accelerated!)
        let _ = get_thread_gpu_context()
            .unwrap_or_else(crate::core::index::gpu::ComputeContext::auto_detect);

        let dim = list_array.value_length() as usize;
        let flat_vectors: Vec<f32> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                list_array
                    .value(i)
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .map(|a| a.values().to_vec())
                    .unwrap_or_default()
            })
            .collect();

        let flat_centroids: Vec<f32> = centroids.iter().flatten().copied().collect();

        let assignments = crate::core::index::gpu::compute_kmeans_assignment(
            &flat_vectors,
            &flat_centroids,
            dim,
        )?;

        // 4. Sort batch by assignments
        let assignment_array = Int32Array::from(
            assignments
                .into_iter()
                .map(|a| a as i32)
                .collect::<Vec<i32>>(),
        );
        let sort_indices = arrow::compute::sort_to_indices(&assignment_array, None, None)?;

        let mut columns = Vec::new();
        for i in 0..batch.num_columns() {
            columns.push(arrow::compute::take(batch.column(i), &sort_indices, None)?);
        }

        RecordBatch::try_new(batch.schema(), columns)
            .context("Failed to reconstruct shuffled batch")
    }

    /// Re-indexes data files that are missing overlay index sidecars.
    ///
    /// This recovers tables when an external Iceberg engine (such as Apache Spark
    /// `rewriteDataFiles`, Trino `OPTIMIZE`, or PyIceberg) has compacted or rewritten
    /// data files, which creates new Parquet files lacking HyperStreamDB sidecars.
    pub async fn recover_indexes_async(&self) -> Result<usize> {
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manager.load_latest_full().await?;

        let unindexed_count = all_entries
            .iter()
            .filter(|e| e.index_files.is_empty())
            .count();
        if unindexed_count == 0 {
            tracing::info!("All segments have valid overlay indexes; no recovery needed.");
            return Ok(0);
        }

        tracing::info!(
            "Recovering overlay indexes for {} unindexed/compacted segments...",
            unindexed_count
        );
        let target_columns = self.indexing.index_columns.read().clone();
        self.backfill_indexes_async(target_columns).await?;
        self.infer_index_metadata_from_physical_async().await?;
        Ok(unindexed_count)
    }

    pub fn recover_indexes(&self) -> Result<usize> {
        self.runtime().block_on(self.recover_indexes_async())
    }
}
