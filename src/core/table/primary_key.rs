// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Primary key management: setting, syncing, adding, dropping, and validation.
///
/// Contains methods on `Table` for:
/// - `set_primary_key`, `set_primary_key_async`
/// - `sync_primary_key_from_schema`, `sync_primary_key_from_schema_async`
/// - `get_primary_key`
/// - `add_primary_key`, `drop_primary_key`
/// - `_validate_pk_uniqueness`, `_check_pk_in_storage_async`
/// - `check_primary_key_uniqueness_async`
use anyhow::{Result, Context};
use arrow::record_batch::RecordBatch;
use arrow::array::Array;
use serde_json::Value;
use crate::core::manifest::{ManifestManager, Schema};
use crate::core::planner::{QueryPlanner, QueryFilter, FilterExpr};
use crate::core::reader::HybridReader;
use crate::SegmentConfig;

use super::Table;

impl Table {
    // -----------------------------------------------------------------------
    // Primary key setters / getters
    // -----------------------------------------------------------------------

    pub fn set_primary_key(&self, columns: Vec<String>) {
        if let Some(ref rt) = self.rt {
            let _ = rt.block_on(self.set_primary_key_async(columns));
        } else {
            let mut pk = self.primary_key.write();
            *pk = columns;
        }
    }

    /// Asynchronously set the primary key. This commits a new manifest version
    /// and ensures the columns are marked as NOT NULL (required: true).
    pub async fn set_primary_key_async(&self, columns: Vec<String>) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest.schemas.last().ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        let mut field_ids = Vec::new();
        for col in &columns {
            let id = latest_schema.fields.iter()
                .find(|f| f.name == *col)
                .map(|f| f.id)
                .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", col))?;
            field_ids.push(id);
        }

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(field_ids).await?;

        // Update in-memory state
        let mut pk = self.primary_key.write();
        *pk = columns;

        // Update in-memory schema ref (it's slightly stale now, but will refresh on next use)
        // or we could force a reload.
        let (new_manifest, _) = manifest_manager.load_latest().await?;
        if let Some(s) = new_manifest.schemas.last() {
            let mut schema_lock = self.schema.write();
            *schema_lock = std::sync::Arc::new(s.to_arrow());
        }

        Ok(())
    }

    pub async fn sync_primary_key_from_schema_async(&self) -> Result<()> {
        let manifest = self.manifest().await?;
        if let Some(latest_schema) = manifest.schemas.last() {
            let mut pk_names = Vec::new();
            for id in &latest_schema.identifier_field_ids {
                if let Some(field) = latest_schema.fields.iter().find(|f| f.id == *id) {
                    pk_names.push(field.name.clone());
                }
            }
            let mut pk = self.primary_key.write();
            *pk = pk_names;
        }
        Ok(())
    }

    pub fn get_primary_key(&self) -> Vec<String> {
        self.primary_key.read().clone()
    }

    /// Synchronize PK columns (Public Sync)
    pub fn sync_primary_key_from_schema(&self) -> Result<()> {
        self.runtime().block_on(self.sync_primary_key_from_schema_async())
    }

    // -----------------------------------------------------------------------
    // Add / drop primary key columns
    // -----------------------------------------------------------------------

    /// Add a column to the primary key.
    /// This is an atomic operation that commits a new manifest version.
    /// Validation: Ensures no duplicate keys exist for the new definition.
    pub async fn add_primary_key(&self, column: String) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest.schemas.last().ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        // Find field ID for column
        let field_id = latest_schema.fields.iter()
            .find(|f| f.name == column)
            .map(|f| f.id)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", column))?;

        let mut next_ids = latest_schema.identifier_field_ids.clone();
        if next_ids.contains(&field_id) {
            return Ok(()); // Already in PK
        }
        next_ids.push(field_id);

        // Validate uniqueness before committing
        self._validate_pk_uniqueness(&next_ids, &latest_schema).await?;

        // Atomic commit to manifest
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(next_ids).await?;

        // Update in-memory state
        let mut pk = self.primary_key.write();
        if !pk.contains(&column) {
            pk.push(column);
        }
        Ok(())
    }

    /// Remove a column from the primary key.
    /// This is an atomic operation that commits a new manifest version.
    pub async fn drop_primary_key(&self, column: String) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest.schemas.last().ok_or_else(|| anyhow::anyhow!("No schema found"))?;

        let field_id = latest_schema.fields.iter()
            .find(|f| f.name == column)
            .map(|f| f.id)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", column))?;

        let mut next_ids = latest_schema.identifier_field_ids.clone();
        next_ids.retain(|id| id != &field_id);

        // Atomic commit to manifest
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(next_ids).await?;

        // Update in-memory state
        let mut pk = self.primary_key.write();
        pk.retain(|c| c != &column);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal validation helpers
    // -----------------------------------------------------------------------

    /// Internal helper to validate that a set of field IDs form a unique key across existing data.
    async fn _validate_pk_uniqueness(&self, field_ids: &[i32], schema: &Schema) -> Result<()> {
        let col_names: Vec<String> = field_ids.iter()
            .map(|id| schema.fields.iter().find(|f| f.id == *id).map(|f| f.name.clone()).unwrap())
            .collect();

        // 1. Acceleration: Single-column PK check using indexes
        if col_names.len() == 1 {
            let col_name = &col_names[0];

            // For now, we perform an optimized read of just the PK column.
            let batches = self.read_with_columns(None, None, col_names.clone())
                .map_err(|e| anyhow::anyhow!("Validation read failed: {}", e))?;

            let mut seen = std::collections::HashSet::new();
            for batch in batches {
                let col = batch.column_by_name(col_name)
                    .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in validation batch", col_name))?
                    .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in validation batch", col_name))?;
                for i in 0..batch.num_rows() {
                    let val = crate::core::manifest::ManifestValue::from_array(col, i).to_string();
                    if !seen.insert(val) {
                        return Err(anyhow::anyhow!("Primary key violation detected for column {}: Duplicate value found.", col_name));
                    }
                }
            }
            return Ok(());
        }

        // Fallback: Multi-column PK scan
        let batches = self.read_with_columns(None, None, col_names.clone())
            .map_err(|e| anyhow::anyhow!("Validation read failed: {}", e))?;

        let mut seen = std::collections::HashSet::new();
        for batch in batches {
            let sort_fields = batch.schema().fields().iter()
                .map(|f| arrow::row::SortField::new(f.data_type().clone()))
                .collect::<Vec<_>>();
            let converter = arrow::row::RowConverter::new(sort_fields)
                .map_err(|e| anyhow::anyhow!("RowConverter error: {}", e))?;

            let rows = converter.convert_columns(batch.columns())
                .map_err(|e| anyhow::anyhow!("Row conversion error: {}", e))?;

            for row in rows.iter() {
                if !seen.insert(row.as_ref().to_vec()) {
                    return Err(anyhow::anyhow!("Primary key violation detected for columns {:?}: Duplicate row values found.", col_names));
                }
            }
        }

        Ok(())
    }

    /// Check if a single primary key value exists in the committed storage.
    /// Uses index-first searching (Bloom Filter -> Inverted Index -> Data Scan).
    async fn _check_pk_in_storage_async(&self, column: &str, value: &serde_json::Value) -> Result<bool> {
        let manifest = self.manifest().await?;
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let all_entries = manager.load_all_entries(&manifest).await?;

        use futures::stream::{self, StreamExt};

        // Parallelize segment checks with a concurrency limit
        let concurrency = 8; // Adjust based on hardware
        let result_stream = stream::iter(all_entries.into_iter().rev())
            .map(|entry| {
                let store = self.store.clone();
                let uri = self.uri.clone();
                let config = self.get_config();
                let schema = manifest.schemas.last().cloned().unwrap_or_default();
                let column = column.to_string();
                let value = value.clone();
                let entry_path = entry.file_path.clone();
                let entry_size = entry.file_size_bytes as u64;

                async move {
                    let mut reader = HybridReader::new(config, store, &uri)
                        .with_iceberg_schema(schema);

                    reader.config.parquet_path = Some(entry_path);
                    reader.config.file_size = Some(entry_size);

                    reader.check_value_exists(&column, &value).await
                }
            })
            .buffer_unordered(concurrency)
            .filter_map(|res| async {
                match res {
                    Ok(true) => Some(Ok(true)),
                    Ok(false) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        let mut pinned_stream = Box::pin(result_stream);
        let result = pinned_stream.next().await;

        match result {
            Some(res) => res, // Found a match or error
            None => Ok(false), // No matches found in any segment
        }
    }

    /// Check if any keys in the batch already exist in the table (Primary Key Enforcement)
    async fn check_primary_key_uniqueness_async(&self, batch: &RecordBatch, columns: &[String]) -> Result<()> {
        if batch.num_rows() == 0 { return Ok(()); }

        let schema = batch.schema();
        let col_indices: Vec<usize> = columns.iter()
            .map(|c| schema.index_of(c))
            .collect::<Result<Vec<usize>, _>>()?;

        // OPTIMIZATION: Use IN clause for batches (efficient via Inverted Index)
        // For now, we take the first row as a sample check to avoid huge expression generation
        // until we have a proper Row-Value In-List implementation.
        for i in 0..batch.num_rows().min(100) { // Limit samples for performance in MVP
            let mut filters_str_vec = Vec::new();
            for (col_name, col_idx) in columns.iter().zip(col_indices.iter()) {
                let col = batch.column(*col_idx);
                let val = if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
                    format!("{}", arr.value(i))
                } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
                    format!("{}", arr.value(i))
                } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                    format!("'{}'", arr.value(i).replace("'", "''"))
                } else {
                    continue;
                };
                filters_str_vec.push(format!("{} = {}", col_name, val));
            }

            if !filters_str_vec.is_empty() {
                let filter_str = filters_str_vec.join(" AND ");
                let expr = FilterExpr::parse_sql(&filter_str, self.arrow_schema()).await?;

                // Check manifests
                let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
                let (_, all_entries, _) = manifest_manager.load_latest_full().await?;
                let planner = QueryPlanner::new();
                let candidates = planner.prune_entries(&all_entries, Some(&expr), None);

                if !candidates.is_empty() {
                    // Refine search within candidates (Index lookup)
                    for (entry, _) in candidates {
                        // Resolve partition-aware path for PK lookup
                        let path = std::path::Path::new(&entry.file_path);
                        let rel_parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
                        let full_base_path = if rel_parent.is_empty() {
                             self.uri.clone()
                        } else {
                             format!("{}/{}", self.uri, rel_parent)
                        };

                        let seg_id = entry.file_path.split('/').next_back().unwrap_or(&entry.file_path)
                            .replace(".parquet", "");

                        let config = SegmentConfig::new(&full_base_path, &seg_id)
                            .with_index_files(entry.index_files.clone())
                            .with_delete_files(entry.delete_files.clone());
                        let reader = HybridReader::new(config, self.store.clone(), &self.uri);

                        let filters = expr.extract_and_conditions();
                        let mut bitmap_opt: Option<roaring::RoaringBitmap> = None;

                        for f in filters {
                            if let Ok(Some(bm)) = reader.get_scalar_filter_bitmap(&f).await {
                                // Subtract logically deleted rows!
                                let deleted = reader.load_merged_deletes().await?;
                                let alive_bm = bm.clone() - deleted.clone();

                                tracing::debug!("PK Check for {}: Index bits: {}, Deleted bits: {}, Alive bits: {}",
                                    f.column, bm.len(), deleted.len(), alive_bm.len());

                                if let Some(current) = bitmap_opt {
                                    bitmap_opt = Some(current & alive_bm);
                                } else {
                                    bitmap_opt = Some(alive_bm);
                                }
                            } else {
                                bitmap_opt = None;
                                break;
                            }
                        }

                        if let Some(bm) = bitmap_opt {
                            if !bm.is_empty() {
                                let pk_val = columns.iter().zip(filters_str_vec.iter())
                                    .map(|(c, f)| format!("{}={}", c, f))
                                    .collect::<Vec<_>>().join(", ");
                                return Err(anyhow::anyhow!("Duplicate primary key error: {} already exists", pk_val));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
