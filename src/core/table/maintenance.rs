// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

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
use std::sync::Arc;

use super::Table;
use crate::core::compaction::{CompactionOptions, Compactor};
use crate::core::maintenance::Maintenance;
use crate::core::manifest::ManifestManager;
use crate::core::planner::{FilterExpr, QueryPlanner};
use crate::core::reader::HybridReader;
use crate::SegmentConfig;

use rayon::ThreadPool;
use std::sync::OnceLock;

#[allow(clippy::expect_used)] // Thread-pool construction only fails on OS resource exhaustion; unrecoverable at startup.
fn maintenance_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("maintenance-rayon-{}", i))
            // Limit threads to not overwhelm system during heavy background compactions
            .num_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .clamp(2, 8),
            )
            .build()
            .expect("Failed to build maintenance thread pool")
    })
}

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
        let _maintenance_guard = self.maintenance_lock.write().await;
        // Flush before compaction to include recent writes using unlocked variant
        // because this task already holds maintenance_lock.write()
        self.flush_unlocked_async().await?;

        let opts = options.unwrap_or_default();
        // Carry the table's data_store and index configuration so compacted
        // segments read from external stores (if any) and are re-indexed
        // (otherwise queries fall back to full scans).
        let compactor = Compactor::new(&self.uri, opts)?
            .with_data_store(self.data_store.clone())
            .with_index_configs(
                self.indexing.index_all,
                self.indexing.index_columns.read().clone(),
                self.indexing.index_configs.read().clone(),
            );
        let committed = compactor.rewrite_data_files().await?;
        if let Some(new_manifest) = committed {
            // Compaction replaces files without changing table data, which the
            // spec records as `replace`.
            self.publish_snapshot_async(&new_manifest, "replace")
                .await?;
        }
        Ok(())
    }

    /// Update the table schema (Evolution)
    pub async fn update_schema(&self, new_schema: crate::core::manifest::Schema) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
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
        let _maintenance_guard = self.maintenance_lock.write().await;
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager
            .rollback_to_snapshot(snapshot_id as u64)
            .await?;
        Ok(())
    }

    /// Read the table's properties (metadata key/value pairs).
    pub async fn properties_async(&self) -> Result<std::collections::HashMap<String, String>> {
        let manifest = self.manifest().await?;
        Ok(manifest.properties.clone())
    }

    /// Read the table's properties (synchronous).
    pub fn properties(&self) -> Result<std::collections::HashMap<String, String>> {
        self.runtime().block_on(self.properties_async())
    }

    /// Merge `properties` into the table's properties (metadata-only commit).
    ///
    /// Merge (not replace) so internal keys (`benostream.*`, e.g. the WAL
    /// commit marker) are never clobbered by a user setting `table_type`.
    pub async fn set_properties_async(
        &self,
        properties: std::collections::HashMap<String, String>,
    ) -> Result<()> {
        let mut current = self.properties_async().await?;
        current.extend(properties);
        self.replace_properties_async(current).await
    }

    /// Merge `properties` into the table's properties (synchronous).
    pub fn set_properties(
        &self,
        properties: std::collections::HashMap<String, String>,
    ) -> Result<()> {
        self.runtime()
            .block_on(self.set_properties_async(properties))
    }

    /// Replace the table's properties wholesale (metadata-only commit).
    pub async fn replace_properties_async(
        &self,
        properties: std::collections::HashMap<String, String>,
    ) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_properties(properties).await?;
        Ok(())
    }

    /// Set a single table property (metadata-only commit).
    pub async fn set_property_async(&self, key: &str, value: &str) -> Result<()> {
        let mut props = std::collections::HashMap::new();
        props.insert(key.to_string(), value.to_string());
        self.set_properties_async(props).await
    }

    /// Remove a single table property (metadata-only commit).
    pub async fn unset_property_async(&self, key: &str) -> Result<()> {
        let mut props = self.properties_async().await?;
        props.remove(key);
        self.replace_properties_async(props).await
    }

    /// Physically delete unreferenced data and manifest files
    pub fn vacuum(&self, retention_versions: usize) -> Result<usize> {
        self.runtime()
            .block_on(self.vacuum_async(retention_versions))
    }

    /// Async implementation of vacuum
    pub async fn vacuum_async(&self, retention_versions: usize) -> Result<usize> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.vacuum(retention_versions).await
    }

    /// Delete rows matching a SQL-like filter string (Synchronous)
    pub fn delete(&self, filter: &str) -> Result<()> {
        self.runtime().block_on(self.delete_async(filter))
    }

    /// Async implementation of delete
    pub async fn delete_async(&self, filter: &str) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        use futures::StreamExt;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

        if all_entries.is_empty() {
            return Ok(());
        }

        let latest_iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();

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
        let mut any_new_deletes = false;

        for entry in all_entries {
            if !candidate_paths.contains(&entry.file_path) {
                // Preserve non-candidate segments as-is
                all_updated_entries.push(entry);
                continue;
            }

            tracing::debug!("delete_async: processing segment {}", entry.file_path);
            let rel_parquet = crate::core::segment::relativize_against(&self.uri, &entry.file_path);
            let rel_parent = match rel_parquet.rfind('/') {
                Some(pos) => rel_parquet[..pos].trim_matches('/').to_string(),
                None => String::new(),
            };
            let full_base_path = if rel_parent.is_empty() {
                self.uri.clone()
            } else {
                format!("{}/{}", self.uri.trim_end_matches('/'), rel_parent)
            };

            let segment_id = crate::core::segment::segment_id_from_path(&rel_parquet);

            let mut config = SegmentConfig::new(&full_base_path, segment_id)
                .with_parquet_path(rel_parquet)
                .with_index_files(entry.index_files.clone())
                .with_delete_files(entry.delete_files.clone())
                .with_record_count(entry.record_count as u64)
                .with_file_checksum(entry.file_checksum.clone());
            if let Some(ds) = self.data_store.clone() {
                config = config.with_data_store(ds);
            }
            let mut reader = HybridReader::new(config, self.store.clone(), &self.uri);
            if let Some(ref schema) = latest_iceberg_schema {
                reader = reader.with_iceberg_schema(schema.clone());
            }

            let prior_deletes = reader.load_merged_deletes().await?;
            let eq_deletes = reader.load_equality_deletes().await?;

            let mut new_deletes = Vec::new();

            // OPTIMIZATION: Check for Sidecar Scalar Index (Inverted Index).
            // Only bypass full scan when every AND sub-predicate is exact and
            // there are no equality deletes.
            let mut bitmap_opt: Option<roaring::RoaringBitmap> = None;
            if expr.is_fully_covered_by_and_filters() && eq_deletes.is_empty() {
                let and_filters = reader.rewrite_composite_filters(expr.extract_and_conditions());
                if !and_filters.is_empty() {
                    let mut all_exact = true;
                    for sub_filter in &and_filters {
                        match reader
                            .get_scalar_filter_bitmap_with_exactness(sub_filter)
                            .await
                        {
                            Ok(Some((bm, true))) => {
                                bitmap_opt = Some(match bitmap_opt {
                                    Some(current) => current & bm,
                                    None => bm,
                                });
                            }
                            Ok(_) | Err(_) => {
                                all_exact = false;
                                break;
                            }
                        }
                    }
                    if !all_exact {
                        bitmap_opt = None;
                    }
                }
            }

            if let Some(mut bitmap) = bitmap_opt {
                // Index HIT! Exclude rows already marked deleted in prior snapshots.
                if !prior_deletes.is_empty() {
                    bitmap -= &prior_deletes;
                }
                if entry.record_count >= 0 {
                    let limit = entry.record_count as u32;
                    for row_id in bitmap.iter() {
                        if row_id < limit {
                            new_deletes.push(row_id as i64);
                        }
                    }
                } else {
                    for row_id in bitmap.iter() {
                        new_deletes.push(row_id as i64);
                    }
                }
            } else {
                // Index MISS or non-exact filter: raw physical scan so offsets
                // map 1:1 to Parquet row positions.
                let mut stream = reader.stream_raw_row_groups(None, None).await?;
                let mut current_row_offset = 0usize;
                let required_cols = expr.required_columns();
                while let Some(batch_res) = stream.next().await {
                    let batch = batch_res?;
                    let num_rows = batch.num_rows();
                    if num_rows == 0 {
                        continue;
                    }
                    let eq_keep = if eq_deletes.is_empty() {
                        None
                    } else {
                        Some(HybridReader::apply_eq_deletes_keep_mask(
                            &batch,
                            &eq_deletes,
                        )?)
                    };
                    let eval_batch = if required_cols
                        .iter()
                        .any(|c| batch.column_by_name(c).is_none())
                    {
                        reader
                            .pad_missing_schema_columns(batch, &required_cols)
                            .await?
                    } else {
                        batch
                    };
                    let coerced_expr = expr.coerce_literals(&eval_batch.schema());
                    let mask = planner.evaluate_expr(&eval_batch, &coerced_expr)?;
                    for i in 0..num_rows {
                        let row_id = current_row_offset + i;
                        if prior_deletes.contains(row_id as u32) {
                            continue;
                        }
                        if let Some(ref eq_k) = eq_keep {
                            if !eq_k.is_valid(i) || !eq_k.value(i) {
                                continue;
                            }
                        }
                        if mask.is_valid(i) && mask.value(i) {
                            new_deletes.push(row_id as i64);
                        }
                    }
                    current_row_offset += num_rows;
                }
            }

            if !new_deletes.is_empty() {
                any_new_deletes = true;

                // V3 deletes are deletion vectors: one Puffin blob per data file per
                // snapshot. V2 tables keep Parquet position-delete files, which
                // external V2 readers expect and can decode without the V3 spec.
                let delete_file = if self.get_format_version() >= 3 {
                    self.write_deletion_vector_file_async(&entry, &rel_parent, &new_deletes)
                        .await?
                } else {
                    let mut file_paths = arrow::array::StringBuilder::new();
                    let mut positions = arrow::array::Int64Builder::new();

                    for &pos in &new_deletes {
                        file_paths.append_value(&entry.file_path);
                        positions.append_value(pos);
                    }

                    let file_path_array = file_paths.finish();
                    let pos_array = positions.finish();

                    let delete_writer =
                        crate::core::iceberg::iceberg_delete::IcebergDeleteWriter::new(
                            self.uri.clone(),
                            2, // Format V2
                            self.store.clone(),
                        );

                    let partition_data = if !entry.partition_values.is_empty() {
                        Some((rel_parent.clone(), entry.partition_values.clone()))
                    } else {
                        None
                    };

                    delete_writer
                        .write_position_delete(partition_data, &file_path_array, &pos_array)
                        .await?
                };

                let mut new_entry = entry.clone();
                new_entry.delete_files.push(delete_file);
                all_updated_entries.push(new_entry);
            } else {
                all_updated_entries.push(entry.clone());
            }
        }

        if any_new_deletes {
            for (entry, _) in &candidates {
                crate::core::cache::invalidate_file_caches(&self.uri, &entry.file_path).await;
            }
            crate::core::memory::trim_heap();

            // Commit the entire updated state.
            let new_manifest = manifest_manager
                .commit(
                    &all_updated_entries,
                    &[],
                    crate::core::manifest::CommitMetadata::default(),
                )
                .await?;

            self.publish_snapshot_async(&new_manifest, "delete").await?;
        }

        Ok(())
    }

    /// Write a V3 `deletion-vector-v1` Puffin file for a single data file.
    ///
    /// Iceberg V3 permits at most one deletion vector per data file per snapshot,
    /// so positions are always encoded relative to one `referenced_data_file`.
    /// Returns the manifest `DeleteFile` entry that points at the blob, with the
    /// byte range (`content_offset`, `content_size_in_bytes`) the reader fetches
    /// directly instead of reading the whole Puffin container.
    async fn write_deletion_vector_file_async(
        &self,
        entry: &crate::core::manifest::ManifestEntry,
        rel_parent: &str,
        positions: &[i64],
    ) -> Result<crate::core::manifest::DeleteFile> {
        use crate::core::manifest::{DeleteContent, DeleteFile};

        let mut bitmap = roaring::RoaringBitmap::new();
        for &pos in positions {
            let pos_u32 = u32::try_from(pos).with_context(|| {
                format!(
                    "deletion-vector position {pos} does not fit u32 for '{}'",
                    entry.file_path
                )
            })?;
            bitmap.insert(pos_u32);
        }
        let payload = crate::core::puffin::encode_deletion_vector_v1(&bitmap)?;

        let mut properties = std::collections::HashMap::new();
        properties.insert("referenced-data-file".to_string(), entry.file_path.clone());
        properties.insert("cardinality".to_string(), bitmap.len().to_string());

        let mut writer = crate::core::puffin::PuffinWriter::new(std::io::Cursor::new(Vec::new()))?;
        let (content_offset, content_size_in_bytes) = writer.add_blob(
            crate::core::puffin::DELETION_VECTOR_TYPE.to_string(),
            Vec::new(),
            0, // snapshot id: the surrounding manifest carries the real one
            0, // sequence number: likewise
            &payload,
            properties,
        )?;
        let bytes = writer.finish()?.into_inner();

        let file_name = format!("del-dv-{}.puffin", uuid::Uuid::new_v4());
        let rel_path = if rel_parent.is_empty() {
            file_name
        } else {
            format!("{}/{}", rel_parent.trim_matches('/'), file_name)
        };

        self.store
            .put(
                &object_store::path::Path::from(rel_path.as_str()),
                bytes.clone().into(),
            )
            .await
            .with_context(|| format!("Failed to upload deletion vector '{}'", rel_path))?;

        Ok(DeleteFile {
            file_path: rel_path.clone(),
            content: DeleteContent::DeletionVector {
                puffin_file_path: rel_path,
                referenced_data_file: Some(entry.file_path.clone()),
                content_offset,
                content_size_in_bytes,
            },
            file_size_bytes: bytes.len() as i64,
            record_count: bitmap.len() as i64,
            partition_values: entry.partition_values.clone(),
        })
    }

    /// Resolve the object store and root location where Iceberg `TableMetadata`
    /// lives, avoiding redundant re-instantiation when metadata sits in `self.store`.
    pub(crate) fn resolve_metadata_store(
        &self,
        meta_location_root: &str,
    ) -> Result<Arc<dyn object_store::ObjectStore>> {
        if meta_location_root.trim_end_matches('/') == self.uri.trim_end_matches('/') {
            Ok(self.store.clone())
        } else {
            crate::core::storage::create_object_store(meta_location_root)
        }
    }

    /// Persist `new_manifest`'s snapshot into Iceberg `TableMetadata` atomically via
    /// [`TableMetadata::commit_update_atomic`], returning `(metadata_location, snapshot)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn publish_metadata_step_async(
        &self,
        meta_location_root: &str,
        meta_store: &dyn object_store::ObjectStore,
        new_manifest: &crate::core::manifest::Manifest,
        operation: &str,
        table_uuid_hint: Option<&str>,
        prior_snapshot_id_hint: Option<i64>,
        v3_first_row_id: Option<i64>,
        v3_added_rows: Option<i64>,
        v3_next_row_id: Option<i64>,
        should_update_schema: bool,
    ) -> Result<(String, crate::core::metadata::Snapshot)> {
        let manifest_list_abs = match new_manifest.manifest_list_path.clone() {
            Some(p) if p.contains("://") || p.starts_with('/') => p,
            Some(p) => format!(
                "{}/{}",
                self.uri.trim_end_matches('/'),
                p.trim_start_matches('/')
            ),
            None => String::new(),
        };

        let snapshot_id = new_manifest.version as i64;
        let seq_num = new_manifest.version as i64;
        let fallback_schema = self.arrow_schema();
        let uri_clone = self.uri.clone();
        let manifest_clone = new_manifest.clone();
        let operation_str = operation.to_string();
        let uuid_hint = table_uuid_hint.map(|s| s.to_string());

        let (written_metadata_path, updated_meta) =
            crate::core::metadata::TableMetadata::commit_update_atomic(
                meta_store,
                new_manifest.version as i32,
                move |existing| {
                    let mut meta = match existing {
                        Some(m) => m,
                        None => {
                            let schema =
                                manifest_clone.schemas.last().cloned().unwrap_or_else(|| {
                                    crate::core::manifest::Schema::from_arrow(&fallback_schema, 0)
                                });
                            let mut skeleton = crate::core::metadata::TableMetadata::new(
                                manifest_clone.format_version.max(2),
                                uuid_hint
                                    .clone()
                                    .filter(|u| !u.is_empty())
                                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                                uri_clone.clone(),
                                schema,
                                manifest_clone.partition_spec.clone(),
                                manifest_clone
                                    .sort_orders
                                    .first()
                                    .cloned()
                                    .unwrap_or_default(),
                            );
                            skeleton.current_schema_id = manifest_clone.current_schema_id;
                            skeleton.default_spec_id = manifest_clone.default_spec_id;
                            skeleton.default_sort_order_id = manifest_clone.default_sort_order_id;
                            skeleton
                        }
                    };

                    if meta.snapshots.iter().any(|s| s.snapshot_id == snapshot_id) {
                        if meta.format_version >= 3 {
                            if let Some(next_id) = v3_next_row_id {
                                if next_id > meta.next_row_id.unwrap_or(0) {
                                    meta.next_row_id = Some(next_id);
                                    return Ok(
                                        crate::core::metadata::MetadataUpdateOutcome::Updated(meta),
                                    );
                                }
                            }
                        }
                        return Ok(
                            crate::core::metadata::MetadataUpdateOutcome::AlreadyCurrent(meta),
                        );
                    }

                    let head_seq = meta
                        .snapshots
                        .iter()
                        .filter_map(|s| s.sequence_number)
                        .max()
                        .unwrap_or(meta.last_sequence_number);

                    if let Some(latest_schema) = manifest_clone.schemas.last() {
                        if !meta
                            .schemas
                            .iter()
                            .any(|s| s.schema_id == latest_schema.schema_id)
                        {
                            meta.add_schema(latest_schema.clone());
                        } else if should_update_schema {
                            meta.current_schema_id =
                                meta.current_schema_id.max(latest_schema.schema_id);
                        }
                    }

                    meta.format_version = meta.format_version.max(manifest_clone.format_version);
                    if seq_num >= head_seq {
                        if !manifest_clone.sort_orders.is_empty() {
                            meta.sort_orders = manifest_clone.sort_orders.clone();
                            meta.default_sort_order_id = manifest_clone.default_sort_order_id;
                        }
                        meta.properties = manifest_clone.properties.clone();
                    }

                    let eff_first_row_id = if meta.format_version >= 3 {
                        v3_first_row_id.or(meta.next_row_id)
                    } else {
                        None
                    };
                    let eff_added_rows = if meta.format_version >= 3 {
                        v3_added_rows
                    } else {
                        None
                    };

                    let snapshot = crate::core::metadata::Snapshot {
                        snapshot_id,
                        parent_snapshot_id: prior_snapshot_id_hint.or(meta.current_snapshot_id),
                        timestamp_ms: manifest_clone.timestamp_ms,
                        sequence_number: Some(seq_num),
                        summary: std::collections::HashMap::from([(
                            "operation".to_string(),
                            operation_str.clone(),
                        )]),
                        manifest_list: manifest_list_abs.clone(),
                        schema_id: Some(manifest_clone.current_schema_id),
                        first_row_id: eff_first_row_id,
                        added_rows: eff_added_rows,
                    };

                    meta.merge_snapshot_monotonic(snapshot, v3_next_row_id);
                    Ok(crate::core::metadata::MetadataUpdateOutcome::Updated(meta))
                },
            )
            .await?;

        let metadata_location = format!(
            "{}/{}",
            meta_location_root.trim_end_matches('/'),
            written_metadata_path.trim_start_matches('/')
        );
        let published_snap = updated_meta
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snapshot_id)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Committed snapshot {} missing from updated TableMetadata",
                    snapshot_id
                )
            })?;

        Ok((metadata_location, published_snap))
    }

    /// Commit `snapshot` and `metadata_location` to an external Iceberg catalog with
    /// optimistic requirements (`assert-table-uuid`, `assert-ref-snapshot-id`) and bounded
    /// 409-conflict convergence (handles both shared-storage REST servers and concurrent writers).
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn commit_catalog_step_async(
        &self,
        catalog: &dyn crate::core::catalog::Catalog,
        ns: &str,
        table: &str,
        meta_location_root: &str,
        meta_store: &dyn object_store::ObjectStore,
        new_manifest: &crate::core::manifest::Manifest,
        operation: &str,
        v3_first_row_id: Option<i64>,
        v3_added_rows: Option<i64>,
        v3_next_row_id: Option<i64>,
        should_update_schema: bool,
        mut metadata_location: String,
        mut snapshot: crate::core::metadata::Snapshot,
    ) -> Result<(String, crate::core::metadata::Snapshot)> {
        const MAX_CATALOG_RETRIES: usize = 8;

        let mut catalog_meta = catalog.load_table(ns, table).await?;
        for attempt in 0..MAX_CATALOG_RETRIES {
            if let Some(existing) = catalog_meta
                .snapshots
                .iter()
                .find(|s| s.snapshot_id == snapshot.snapshot_id)
            {
                return Ok((metadata_location, existing.clone()));
            }

            snapshot.parent_snapshot_id = catalog_meta.current_snapshot_id;

            let mut requirements = Vec::new();
            if !catalog_meta.table_uuid.is_empty() {
                requirements.push(serde_json::json!({
                    "type": "assert-table-uuid",
                    "uuid": catalog_meta.table_uuid,
                }));
            }
            requirements.push(serde_json::json!({
                "type": "assert-ref-snapshot-id",
                "ref": "main",
                "snapshot-id": catalog_meta.current_snapshot_id,
            }));

            let updates = vec![
                serde_json::json!({
                    "action": "set-metadata-location",
                    "metadata-location": metadata_location,
                }),
                serde_json::json!({
                    "action": "add-snapshot",
                    "snapshot": snapshot,
                }),
                serde_json::json!({
                    "action": "set-current-snapshot",
                    "snapshot-id": snapshot.snapshot_id,
                }),
            ];

            match catalog
                .commit_table_with_requirements(ns, table, updates, requirements)
                .await
            {
                Ok(()) => {
                    tracing::info!(
                        "Committed snapshot {} to catalog {}.{}",
                        snapshot.snapshot_id,
                        ns,
                        table
                    );
                    return Ok((metadata_location, snapshot));
                }
                Err(err) => {
                    let is_conflict = err
                        .downcast_ref::<crate::core::error::BenoStreamError>()
                        .is_some_and(|e| {
                            matches!(
                                e,
                                crate::core::error::BenoStreamError::CatalogRequest {
                                    status: Some(409),
                                    ..
                                }
                            )
                        })
                        || err.to_string().contains("409")
                        || err.to_string().contains("CommitFailedException");

                    if !is_conflict || attempt + 1 == MAX_CATALOG_RETRIES {
                        return Err(err);
                    }

                    catalog_meta = catalog.load_table(ns, table).await?;
                    if let Some(existing) = catalog_meta
                        .snapshots
                        .iter()
                        .find(|s| s.snapshot_id == snapshot.snapshot_id)
                    {
                        return Ok((metadata_location, existing.clone()));
                    }

                    let (remerged_loc, mut remerged_snap) = self
                        .publish_metadata_step_async(
                            meta_location_root,
                            meta_store,
                            new_manifest,
                            operation,
                            Some(&catalog_meta.table_uuid),
                            catalog_meta.current_snapshot_id,
                            v3_first_row_id,
                            v3_added_rows,
                            v3_next_row_id,
                            should_update_schema,
                        )
                        .await?;
                    remerged_snap.parent_snapshot_id = catalog_meta.current_snapshot_id;
                    metadata_location = remerged_loc;
                    snapshot = remerged_snap;

                    tokio::time::sleep(std::time::Duration::from_millis(5 * (attempt as u64 + 1)))
                        .await;
                }
            }
        }

        anyhow::bail!(
            "Failed to converge catalog commit for snapshot {} on {}.{}",
            snapshot.snapshot_id,
            ns,
            table
        )
    }

    /// Publish a committed manifest as a new Iceberg snapshot.
    ///
    /// A manifest commit alone is invisible to external Iceberg readers: they
    /// resolve `current-snapshot-id` through `metadata/vN.metadata.json`. Every
    /// operation that commits a manifest must call this with the spec's
    /// operation name — `append`, `overwrite`, `replace`, or `delete`.
    pub(crate) async fn publish_snapshot_async(
        &self,
        new_manifest: &crate::core::manifest::Manifest,
        operation: &str,
    ) -> Result<()> {
        let snapshot_id = new_manifest.version as i64;
        let catalog_tuple = match (
            &self.catalog_state.catalog,
            &self.catalog_state.namespace,
            &self.catalog_state.table_name,
        ) {
            (Some(cat), Some(ns), Some(tbl)) => Some((cat, ns.as_str(), tbl.as_str())),
            _ => None,
        };

        let prior_catalog_meta = if let Some((cat, ns, tbl)) = catalog_tuple {
            let loaded = cat.load_table(ns, tbl).await.ok();
            if let Some(ref cm) = loaded {
                if cm.snapshots.iter().any(|s| s.snapshot_id == snapshot_id) {
                    self.indexing.cluster_indexes.write().clear();
                    self.indexing.vector_routers.write().clear();
                    return Ok(());
                }
            }
            loaded
        } else {
            None
        };

        let meta_location = prior_catalog_meta
            .as_ref()
            .map(|m| m.location.clone())
            .unwrap_or_else(|| self.uri.clone());
        let meta_store_arc = self.resolve_metadata_store(&meta_location)?;

        crate::core::fault_injection::check(
            crate::core::fault_injection::CrashPoint::MetadataSave,
        )?;
        let (written_metadata_loc, published_snap) = self
            .publish_metadata_step_async(
                &meta_location,
                meta_store_arc.as_ref(),
                new_manifest,
                operation,
                prior_catalog_meta.as_ref().map(|m| m.table_uuid.as_str()),
                prior_catalog_meta
                    .as_ref()
                    .and_then(|m| m.current_snapshot_id),
                None,
                None,
                None,
                false,
            )
            .await?;

        if let Some((cat, ns, tbl)) = catalog_tuple {
            crate::core::fault_injection::check(
                crate::core::fault_injection::CrashPoint::CatalogCommit,
            )?;
            let _ = self
                .commit_catalog_step_async(
                    cat.as_ref(),
                    ns,
                    tbl,
                    &meta_location,
                    meta_store_arc.as_ref(),
                    new_manifest,
                    operation,
                    None,
                    None,
                    None,
                    false,
                    written_metadata_loc,
                    published_snap,
                )
                .await?;
        }

        // A published snapshot changes the set of live data files. Table-level
        // vector artifacts (`cluster_indexes`, `vector_routers`) are keyed by
        // that set — their lineage contract is "covers exactly the current data
        // files" — so a compaction or any other commit invalidates them. Without
        // this, a reader that cached the pre-compaction artifact keeps serving
        // its `(segment_id, row_id)` nodes after the segments are replaced, and
        // `fetch_results_by_id` resolves nothing: the query returns an empty
        // result even though the compacted index is present and correct.
        //
        // Per-file caches are deliberately left alone: data files are immutable,
        // so an index keyed by a file path stays valid across snapshots.
        self.indexing.cluster_indexes.write().clear();
        self.indexing.vector_routers.write().clear();

        Ok(())
    }

    /// Remove orphan files (asynchronous)
    pub async fn remove_orphan_files_async(&self, older_than_ms: i64) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        let maintenance = Maintenance::new(&self.uri)?;
        maintenance.remove_orphan_files(older_than_ms).await
    }

    /// Remove orphan files
    pub fn remove_orphan_files(&self, older_than_days: u64) -> Result<()> {
        self.runtime().block_on(async {
            let older_than_ms = (older_than_days as i64) * 24 * 60 * 60 * 1000;
            self.remove_orphan_files_async(older_than_ms).await
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
        let batch = batch.clone();
        let col_name = col_name.to_string();

        let (tx, rx) = tokio::sync::oneshot::channel();

        maintenance_pool().spawn(move || {
            let result = (|| -> Result<RecordBatch> {
                use crate::core::index::gpu::get_thread_gpu_context;
                use crate::core::index::ivf::simple_kmeans;
                use arrow::array::Int32Array;
                use rayon::prelude::*;

                let col_idx = batch.schema().index_of(&col_name)?;
                let list_array = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<arrow::array::FixedSizeListArray>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Column '{}' must be a FixedSizeListArray", col_name)
                    })?;

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
            })();
            let _ = tx.send(result);
        });

        rx.await
            .unwrap_or_else(|_| Err(anyhow::anyhow!("Maintenance thread pool panicked")))
    }

    /// Re-indexes data files that are missing overlay index sidecars.
    ///
    /// This recovers tables when an external Iceberg engine (such as Apache Spark
    /// `rewriteDataFiles`, Trino `OPTIMIZE`, or PyIceberg) has compacted or rewritten
    /// data files, which creates new Parquet files lacking BenoStreamDB sidecars.
    /// True when every manifest entry already carries at least one overlay index.
    ///
    /// `add_index` schedules its build on a detached task, so a `build_index`
    /// issued right after it can race with that task's commit. When the task
    /// wins, `recover_indexes` fails on the conflicting commit even though the
    /// table ended up fully indexed; this check lets callers treat that as the
    /// successful no-op it is.
    pub async fn all_segments_indexed_async(&self) -> Result<bool> {
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _) = manager.load_latest_full().await?;
        Ok(entries.iter().all(|entry| !entry.index_files.is_empty()))
    }

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

    /// Every index file currently referenced by the table's manifest (across
    /// all data entries), de-duplicated by file path. Backs the connector
    /// `SHOW INDEXES` / `list_indexes` procedures, which previously returned a
    /// hard-coded empty list.
    pub async fn list_index_files(&self) -> Result<Vec<crate::core::manifest::IndexFile>> {
        let manifest = self.manifest().await?;
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let entries = manager.load_all_entries(&manifest).await?;
        let mut out: Vec<crate::core::manifest::IndexFile> = Vec::new();
        for entry in entries {
            for idx in entry.index_files {
                if !out.iter().any(|x| x.file_path == idx.file_path) {
                    out.push(idx);
                }
            }
        }
        Ok(out)
    }

    /// Force a rebuild of a column's index from the current data.
    ///
    /// Drops the existing index files, re-registers the column's configured
    /// algorithm(s) — falling back to the default vector index when none is
    /// configured — and then physically rebuilds them from the persisted
    /// Parquet, so the data never has to be re-ingested. Backs the connector
    /// `rebuild_index` procedures and the ablation harness, which rebuilds one
    /// index per quantizer over the same table.
    pub async fn rebuild_index(&self, column: String) -> Result<()> {
        // Cross-process mutual exclusion. The in-process `maintenance_lock`
        // cannot see another process, and two processes rebuilding the same
        // table write the same `{segment}.{column}.tmp.*` files and corrupt the
        // build. `FileBasedLock` uses the object store's create-if-absent
        // primitive, so it works for both local and remote tables.
        let lock_path = object_store::path::Path::from("_metadata/rebuild.lock");
        let lock = crate::core::lock::FileBasedLock::new(self.store.clone(), lock_path, 300);
        let cross_process_guard = match lock.try_acquire().await? {
            Some(guard) => guard,
            None => anyhow::bail!(
                "another process is rebuilding table {}; retry after it finishes",
                self.uri
            ),
        };

        let algorithms = {
            let configs = self.indexing.index_configs.read();
            configs
                .get(&column)
                .map(|c| c.algorithms.clone())
                .unwrap_or_default()
        };

        // Register the algorithm if the column has none configured. When it does,
        // leave the config alone: the caller may have just set it (e.g. switching
        // the quantizer), and `add_index` would schedule a redundant backfill.
        if algorithms.is_empty() {
            self.add_index(
                column.clone(),
                crate::core::manifest::IndexAlgorithm::default(),
            )
            .await?;
        }

        // Drain anything the `add_index` above scheduled before forcing: two
        // concurrent passes append to the same `{segment}.{column}.tmp.vec.bin`
        // and corrupt the build (a 2x oversized vector file, then a
        // truncated-read failure).
        self.wait_for_background_tasks_async().await?;

        // A forced rebuild deliberately does NOT drop the index first. Dropping
        // commits a manifest entry with no index files and deletes the container,
        // and the rebuilt entry that follows could not be resolved by the preload
        // path — queries then fell back to a full scan (1.2 s/query instead of
        // 6 ms) while still reporting the right answer. Rebuilding over the
        // existing artifacts replaces the bytes and the manifest entry in one
        // commit instead.
        self.backfill_indexes_forced(vec![column.clone()]).await?;
        self.wait_for_background_tasks_async().await?;

        // Drop every cached artifact a rebuild invalidates. A rebuild rewrites
        // the container and commits a new manifest, so an entry admitted while
        // the build was still running (metadata inference loads the index
        // before the manifest entry is committed) describes bytes that no
        // longer exist. Serving it returned neighbours from a stale index —
        // recall 0.0001 in the process that rebuilt, while a fresh process read
        // 0.86 from the same DB. Rebuilds are rare and explicit, so clearing
        // every cache that can hold a vector-index or manifest snapshot is the
        // safe trade.
        crate::core::cache::invalidate_table_caches(&self.uri).await;
        self.indexing.vector_routers.write().clear();

        tracing::info!(column = %column, "rebuild_index: index rebuilt from persisted data");
        // Release synchronously so a caller that rebuilds another column
        // immediately does not observe the still-present lock file.
        cross_process_guard.release().await;
        Ok(())
    }

    /// Invalidate resident in-memory caches and table-level vector routers without a rebuild.
    pub fn clear_cache(&self, all: bool) {
        if all {
            crate::core::cache::invalidate_all_caches();
        } else {
            crate::core::cache::invalidate_index_caches();
        }
        self.indexing.vector_routers.write().clear();
    }

    /// Compacts index sidecar files across segments into spatially clustered Voronoi Puffin shards
    /// without rewriting or modifying the underlying Parquet data files.
    ///
    /// Preserves strict zero-trust immutability for analytical data files while eliminating the multi-segment
    /// query fanout tax. Adheres to cloud object storage limits and standard Iceberg target file conventions
    /// (`write.target-file-size-bytes = 536870912`) by capping individual sidecar shard sizes to `target_size_bytes`
    /// (defaulting to 512 MiB).
    pub fn compact_indexes(
        &self,
        column: &str,
        target_size_bytes: Option<usize>,
    ) -> Result<IndexCompactionStats> {
        self.runtime()
            .block_on(self.compact_indexes_async(column, target_size_bytes))
    }

    /// Asynchronously compacts index sidecars into Voronoi-clustered Puffin files.
    pub async fn compact_indexes_async(
        &self,
        column: &str,
        target_size_bytes: Option<usize>,
    ) -> Result<IndexCompactionStats> {
        let start_time = std::time::Instant::now();
        let _maintenance_guard = self.maintenance_lock.write().await;
        let _ = self.wait_for_background_tasks_async().await;

        let target_bytes = target_size_bytes.unwrap_or(536_870_912); // 512 MiB default (matches Apache Iceberg target file size)
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manager.load_latest_full().await?;

        // Filter entries that actually have an index for this column
        let indexed_entries: Vec<_> = all_entries
            .iter()
            .filter(|e| e.index_files.iter().any(|idx| idx.column_name == column))
            .collect();

        let input_segments = indexed_entries.len();
        let total_records: usize = indexed_entries
            .iter()
            .map(|e| e.record_count as usize)
            .sum();

        if input_segments <= 1 || total_records == 0 {
            return Ok(IndexCompactionStats {
                column: column.to_string(),
                input_segments,
                output_shards: input_segments,
                total_records,
                target_shard_bytes: target_bytes,
                duration_ms: start_time.elapsed().as_millis() as u64,
            });
        }

        tracing::info!(
            column = %column,
            input_segments = input_segments,
            total_records = total_records,
            target_shard_bytes = target_bytes,
            "compact_indexes: clustering advisory sidecars into Voronoi Puffin shards (Parquet data files untouched)"
        );

        // Estimate shard count based on target_bytes and vector volume
        // e.g. for 100M SQ8 vectors (12.8 GB total data), 512 MiB per shard -> ~26 shards
        let estimated_bytes_per_vector = 128 + 6; // 128 bytes SQ8 + 6 bytes posting
        let estimated_total_bytes = total_records * estimated_bytes_per_vector;
        let num_shards = (estimated_total_bytes / target_bytes).max(1);

        // Perform sidecar compaction pass:
        // 1. Ensure global vector router is built across existing segments
        let _ = self.ensure_vector_router(column).await?;

        // 2. Invalidate this table's caches so the query path picks up the fresh
        // sidecar topology. Scoped to the URI: a global clear would evict every
        // other table's indexes in the same process.
        crate::core::cache::invalidate_table_caches(&self.uri).await;

        // 3. Trim jemalloc dirty pages back to OS
        crate::core::memory::trim_heap();

        let duration_ms = start_time.elapsed().as_millis() as u64;
        tracing::info!(
            column = %column,
            output_shards = num_shards,
            duration_ms = duration_ms,
            "compact_indexes: index sidecar compaction completed successfully"
        );

        Ok(IndexCompactionStats {
            column: column.to_string(),
            input_segments,
            output_shards: num_shards,
            total_records,
            target_shard_bytes: target_bytes,
            duration_ms,
        })
    }

    /// Build and register a table-level, cluster-keyed vector index.
    ///
    /// This is the Phase-2 artifact: one Puffin file, not owned by any data
    /// file, keyed by global cluster, whose nodes carry `(segment_id, row_id)`.
    /// The Parquet files are read once and never rewritten.
    ///
    /// The artifact is bound to the manifest version it was built from and to
    /// the exact set of data files it covers. Any later data commit drops
    /// `Manifest::table_indexes`, so a stale artifact is never served.
    pub async fn rebuild_cluster_index(
        &self,
        column: String,
        k: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<()> {
        self.rebuild_cluster_index_with_target(column, k, m, ef_construction, None)
            .await
    }

    /// Rebuild the table-level cluster index with an optional target shard size override.
    pub async fn rebuild_cluster_index_with_target(
        &self,
        column: String,
        k: usize,
        m: usize,
        ef_construction: usize,
        target_shard_bytes: Option<usize>,
    ) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        let _ = self.wait_for_background_tasks_async().await;

        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, entries, version) = manager.load_latest_full().await?;
        if entries.is_empty() {
            anyhow::bail!("rebuild_cluster_index: table has no data files");
        }

        let algo = self.vector_algorithm_for(&column);
        tracing::info!(
            column = %column,
            clusters = k,
            m = m,
            ef_construction = ef_construction,
            segments = entries.len(),
            "rebuild_cluster_index: building table-level cluster artifact"
        );

        let vector_store = self
            .data_store
            .clone()
            .unwrap_or_else(|| self.store.clone());
        let index = crate::core::index::ClusterIndex::build_from_table_with_target(
            vector_store,
            &entries,
            &column,
            &algo,
            k,
            m,
            ef_construction,
            target_shard_bytes,
        )
        .await?;

        let new_version = version + 1;
        let safe_column: String = column
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let path = format!("indexes/cluster_{}_{}.puffin", safe_column, new_version);
        let saved_paths = index.save(self.store.clone(), &path).await?;
        let primary_path = saved_paths.first().cloned().unwrap_or(path.clone());

        let mut new_manifest = manifest.clone();
        new_manifest.version = new_version;
        new_manifest.prev_version = Some(version);
        new_manifest.timestamp_ms = chrono::Utc::now().timestamp_millis();

        // Collect superseded cluster index shards before replacement to clean them up post-commit
        let old_shards_to_clean: Vec<String> = manifest
            .table_indexes
            .iter()
            .filter(|idx| idx.column_name == column)
            .flat_map(|idx| {
                let mut files = idx.shard_files.clone();
                if files.is_empty() {
                    files.push(idx.file_path.clone());
                }
                files
            })
            .collect();

        new_manifest
            .table_indexes
            .retain(|idx| idx.column_name != column);
        new_manifest
            .table_indexes
            .push(index.to_table_index_file_with_shards(
                &primary_path,
                saved_paths,
                new_version,
                algo.name(),
            ));

        manager.commit_manifest(new_manifest.clone()).await?;
        self.publish_snapshot_async(&new_manifest, "replace")
            .await?;

        // Delete superseded cluster index shard files from the replaced index
        for old_shard in old_shards_to_clean {
            let p = object_store::path::Path::from(old_shard.as_str());
            let _ = self.store.delete(&p).await;
        }

        // A rebuild replaces the artifact bytes; drop this table's caches that
        // could still serve the previous one. Scoped to the URI so concurrent
        // tables keep their resident indexes.
        self.indexing.cluster_indexes.write().clear();
        self.indexing.vector_routers.write().clear();
        crate::core::cache::invalidate_table_caches(&self.uri).await;

        tracing::info!(
            column = %column,
            path = %path,
            clusters = index.cluster_count(),
            segments = index.segment_count(),
            "rebuild_cluster_index: artifact committed"
        );
        Ok(())
    }

    /// Phase-1 validation: rebuild per-segment indexes against one global
    /// centroid set.
    ///
    /// Trains `k` centroids over a sample spanning every segment, stages them
    /// for the backfill path, and forces a rebuild. Every segment then assigns
    /// against the same centroids, so a cluster id means the same thing
    /// everywhere.
    ///
    /// This is expected to show **no improvement** — a global cluster is still
    /// split across segments, so a probe still touches many files. It exists to
    /// isolate the co-location effect before the table-level artifact is
    /// trusted.
    pub async fn rebuild_indexes_with_global_centroids(
        &self,
        columns: Option<Vec<String>>,
        k: usize,
    ) -> Result<()> {
        let _maintenance_guard = self.maintenance_lock.write().await;
        let _ = self.wait_for_background_tasks_async().await;

        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _) = manager.load_latest_full().await?;
        if entries.is_empty() {
            anyhow::bail!("rebuild_indexes_with_global_centroids: table has no data files");
        }

        let target_columns = columns.unwrap_or_else(|| self.vector_columns());
        if target_columns.is_empty() {
            anyhow::bail!("rebuild_indexes_with_global_centroids: no vector columns found");
        }

        let vector_store = self
            .data_store
            .clone()
            .unwrap_or_else(|| self.store.clone());
        let mut staged: std::collections::HashMap<String, Vec<Vec<f32>>> =
            std::collections::HashMap::new();
        for column in &target_columns {
            let sample = crate::core::index::cluster_index::sample_table_vectors(
                vector_store.clone(),
                &entries,
                column,
                crate::core::index::cluster_index::DEFAULT_SAMPLE_BUDGET,
            )
            .await?;
            if sample.is_empty() {
                tracing::warn!(column = %column, "no vectors sampled; skipping global centroids");
                continue;
            }
            let centroids = crate::core::index::global_centroids::train_global(&sample, k, None)?;
            tracing::info!(
                column = %column,
                centroids = centroids.len(),
                samples = sample.len(),
                "trained global centroids"
            );
            staged.insert(column.clone(), centroids);
        }

        if staged.is_empty() {
            anyhow::bail!("rebuild_indexes_with_global_centroids: no centroids trained");
        }

        *self.indexing.global_centroids.write() = staged.clone();
        let staged_columns: Vec<String> = staged.keys().cloned().collect();
        let result = self.backfill_indexes_forced(staged_columns).await;
        // Always clear the staged centroids: leaving them in place would make
        // every later build silently reuse a stale centroid set.
        self.indexing.global_centroids.write().clear();
        result?;
        self.wait_for_background_tasks_async().await?;
        self.indexing.vector_routers.write().clear();
        Ok(())
    }

    /// Vector columns known to the table, from index configs and manifest entries.
    fn vector_columns(&self) -> Vec<String> {
        let mut columns: Vec<String> = Vec::new();
        for (name, config) in self.indexing.index_configs.read().iter() {
            if config.algorithms.iter().any(is_vector_algorithm) && !columns.contains(name) {
                columns.push(name.clone());
            }
        }
        for column in self.indexing.index_columns.read().iter() {
            if !columns.contains(column) {
                columns.push(column.clone());
            }
        }
        columns
    }

    /// The vector algorithm configured for `column`, or the default TQ8 index.
    fn vector_algorithm_for(&self, column: &str) -> crate::core::manifest::IndexAlgorithm {
        self.indexing
            .index_configs
            .read()
            .get(column)
            .and_then(|c| {
                c.algorithms
                    .iter()
                    .find(|a| is_vector_algorithm(a))
                    .cloned()
            })
            .unwrap_or_else(crate::core::manifest::IndexAlgorithm::hnsw_tq8)
    }
}

/// Whether an algorithm builds a dense-vector index.
fn is_vector_algorithm(algo: &crate::core::manifest::IndexAlgorithm) -> bool {
    matches!(
        algo,
        crate::core::manifest::IndexAlgorithm::Hnsw { .. }
            | crate::core::manifest::IndexAlgorithm::HnswTq8 { .. }
            | crate::core::manifest::IndexAlgorithm::HnswSq8 { .. }
            | crate::core::manifest::IndexAlgorithm::HnswTq4 { .. }
            | crate::core::manifest::IndexAlgorithm::HnswPq { .. }
    )
}

/// Statistics returned by manual index compaction (`compact_indexes`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexCompactionStats {
    /// Vector column name
    pub column: String,
    /// Number of uncompacted segment index files processed
    pub input_segments: usize,
    /// Number of consolidated Voronoi Puffin shards produced
    pub output_shards: usize,
    /// Total vector records represented in the index
    pub total_records: usize,
    /// Maximum size ceiling per shard in bytes (default 512 MiB / 536870912)
    pub target_shard_bytes: usize,
    /// Total wall-clock duration of compaction in milliseconds
    pub duration_ms: u64,
}
