// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{Result, Context};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::core::manifest::ManifestManager;
use crate::telemetry::metrics::INGEST_ROWS_TOTAL;
use crate::SegmentConfig;

use super::Table;
use crate::core::index::memory::InMemoryVectorIndex;
use crate::core::segment::HybridSegmentWriter;
use crate::core::metadata::TableMetadata;
use crate::core::storage::create_object_store;
use std::collections::HashMap;
use serde_json::Value;
use arrow::datatypes::Schema;
use futures::StreamExt;
impl Table {
    /// Write Arrow RecordBatches to the table (Buffered)
    /// 
    /// Data is written to an in-memory buffer. It is NOT persisted to disk until:
    /// 1. The buffer exceeds `HYPERSTREAM_CACHE_GB`
    /// 2. `commit()` is called explicitly
    pub fn write(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.runtime().block_on(self.write_async(batches))
    }

    /// Commit buffered writes to disk
    pub fn commit(&self) -> Result<()> {
        self.runtime().block_on(self.flush_async())
    }
    
    /// Async commit
    pub async fn commit_async(&self) -> Result<()> {
        self.flush_async().await?;
        
        // Wait for all background indexing tasks to finish (ensures manifest consistency in tests)
        let tasks = {
            let mut lock = self.background_tasks.lock().await;
            std::mem::take(&mut *lock)
        };
        
        for task in tasks {
            if let Err(e) = task.await {
                tracing::warn!("Background indexing task failed: {}", e);
            }
        }
        
        Ok(())
    }

    /// Truncate the table (metadata-only operation)
    pub fn truncate(&self) -> Result<()> {
        self.runtime().block_on(self.truncate_async())
    }

    /// Async implementation of truncate
    pub async fn truncate_async(&self) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        
        // Step 1: Get all current entry paths
        // We must load THE WHOLE current manifest to know what to remove.
        let (_, entries, _) = manifest_manager.load_latest_full().await.unwrap_or((crate::core::manifest::Manifest::default(), Vec::new(), 0));
        let remove_paths: Vec<String> = entries.iter().map(|e| e.file_path.clone()).collect();
        
        // Step 2: Commit with all paths in remove_paths and empty add_entries
        // This is a single atomic snapshot swap that points to an empty segment set.
        manifest_manager.commit(&[], &remove_paths, crate::core::manifest::CommitMetadata::default()).await?;
        
        // Step 3: Truncate WAL
        {
            let mut wal = self.wal.lock().await;
            wal.truncate().context("Failed to truncate WAL during table truncate")?;
        }
        
        // Step 4: Clear memory index
        {
            let mut idx = self.indexing.memory_index.write().unwrap();
            *idx = None;
        }
        
        // Step 5: Clear write buffer
        {
            let mut buffer = self.write_buffer.write().unwrap();
            buffer.clear();
        }
        
        tracing::info!("Table truncated: metadata reset, all {} segments removed, buffers cleared.", remove_paths.len());
        Ok(())
    }

    /// Compact the WAL (consolidate log entries)
    pub fn checkpoint(&self) -> Result<()> {
        let mut wal = self.wal.blocking_lock();
        wal.compact()
    }

    // Schema evolution logic moved to schema.rs

    /// Async implementation of write (Buffered) with Schema Validation
    pub async fn write_async(&self, batches: Vec<RecordBatch>) -> Result<()> {
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        INGEST_ROWS_TOTAL.inc_by(total_rows as u64);
        
        if batches.is_empty() {
             return Ok(());
        }

        let batches = batches;
        
        let mut is_empty_schema = false;
        if let Some(first_batch) = batches.first() {
            let mut lock = self.schema.write().unwrap();
            is_empty_schema = lock.fields().is_empty();
            if is_empty_schema {
                let incoming_schema = first_batch.schema();
                let table_pattern = self.label_pattern;
                
                let mut fields = Vec::with_capacity(incoming_schema.fields().len());
                for (i, field) in incoming_schema.fields().iter().enumerate() {
                    let mut new_name = (*field).name().clone();
                    
                    // Check if name is numeric or empty, and apply pattern
                    let is_numeric = new_name.chars().all(|c| c.is_ascii_digit());
                    if is_numeric || new_name.is_empty() {
                        new_name = match table_pattern {
                            crate::core::table::LabelPattern::ExcelAlpha => crate::core::table::excel_column_label(i),
                            crate::core::table::LabelPattern::Polars => format!("column_{}", i + 1),
                            crate::core::table::LabelPattern::Pandas => i.to_string(),
                        };
                    }
                    
                    let new_field = (**field).clone().with_name(new_name);
                    fields.push(new_field);
                }
                *lock = Arc::new(arrow::datatypes::Schema::new(fields));
            }
            drop(lock);
        }

        // 2. Primary Key Uniqueness Validation
        let pk_cols = self.primary_key.read().unwrap().clone();
        if !pk_cols.is_empty() {
            // Accelerated path for single-column Primary Keys
            if pk_cols.len() == 1 {
                let pk_col = &pk_cols[0];
                let mut seen_keys = std::collections::HashSet::new();
                
                {
                    let buffer = self.write_buffer.read().unwrap();
                    // 1. Pre-populate seen keys from the in-memory write buffer
                    for b_batch in buffer.iter() {
                        if let Some(b_col) = b_batch.column_by_name(pk_col) {
                            for j in 0..b_batch.num_rows() {
                                let b_val = crate::core::manifest::ManifestValue::from_array(b_col, j);
                                seen_keys.insert(b_val.to_string());
                            }
                        }
                    }
                }
                
                // 2. Check each record in the incoming batches
                for batch in &batches {
                    if let Some(col) = batch.column_by_name(pk_col) {
                        for i in 0..batch.num_rows() {
                            let m_val = crate::core::manifest::ManifestValue::from_array(col, i);
                            
                            // Check for nulls in PK
                            if matches!(m_val, crate::core::manifest::ManifestValue::Null) {
                                return Err(anyhow::anyhow!("Null constraint violation: Primary key column '{}' cannot contain null values", pk_col));
                            }

                            let val_str = m_val.to_string();
                            
                            // Check against buffer and current batch
                            if seen_keys.contains(&val_str) {
                                return Err(anyhow::anyhow!("Duplicate primary key error: id = {}", val_str));
                            }
                            
                            // 3. Check against storage (Index-driven)
                            let val_json = serde_json::to_value(&m_val).unwrap_or(serde_json::Value::Null);
                            if self._check_pk_in_storage_async(pk_col, &val_json).await? {
                                return Err(anyhow::anyhow!("Duplicate primary key error: id = {}", val_str));
                            }
                            
                            seen_keys.insert(val_str);
                        }
                    }
                }
            } else {
                let buffer = self.write_buffer.read().unwrap();
                // Fallback for multi-column PKs (O(N*M) check for now)
                for batch in &batches {
                    for pk_col in &pk_cols {
                        if let Some(col) = batch.column_by_name(pk_col) {
                            for i in 0..batch.num_rows() {
                                let val = crate::core::manifest::ManifestValue::from_array(col, i);
                                
                                // Check for nulls in PK
                                if matches!(val, crate::core::manifest::ManifestValue::Null) {
                                    return Err(anyhow::anyhow!("Null constraint violation: Primary key column '{}' cannot contain null values", pk_col));
                                }

                                // Check against buffer
                                for b_batch in buffer.iter() {
                                    if let Some(b_col) = b_batch.column_by_name(pk_col) {
                                        for j in 0..b_batch.num_rows() {
                                            let b_val = crate::core::manifest::ManifestValue::from_array(b_col, j);
                                            if val == b_val {
                                                return Err(anyhow::anyhow!("Duplicate primary key error: id = {}", val));
                                            }
                                        }
                                    }
                                }
                                
                                // Check against other rows in the same batch (before index i)
                                for j in 0..i {
                                    let b_val = crate::core::manifest::ManifestValue::from_array(col, j);
                                    if val == b_val {
                                        return Err(anyhow::anyhow!("Duplicate primary key error: id = {}", val));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut target_schema = self.arrow_schema();
        
        if let Some(first_batch) = batches.first() {
            let incoming_schema = first_batch.schema();
            let mut evolved_schema = (*target_schema).clone();
            let mut changed = false;

            for field in incoming_schema.fields() {
                let existing_field_info = evolved_schema.field_with_name(field.name()).ok().map(|f| (f.data_type().clone(), f.is_nullable()));
                
                if let Some((existing_dtype, existing_nullable)) = existing_field_info {
                    // Check if we need to widen the type (e.g. Int32 -> Int64)
                    if existing_dtype != *field.data_type() {
                        tracing::info!("Schema Evolution: Widening column '{}' from {:?} to {:?}", field.name(), existing_dtype, field.data_type());
                        
                        let idx = evolved_schema.index_of(field.name()).unwrap();
                        let mut fields: Vec<arrow::datatypes::Field> = evolved_schema.fields().iter().map(|f| (**f).clone()).collect();
                        fields[idx] = (**field).clone();
                        evolved_schema = Schema::new(fields);
                        changed = true;
                    }
                    
                    // Check if we need to change Nullability (Required -> Nullable)
                    if !existing_nullable && field.is_nullable() {
                        tracing::info!("Schema Evolution: Changing column '{}' to nullable", field.name());
                        let idx = evolved_schema.index_of(field.name()).unwrap();
                        let mut fields: Vec<arrow::datatypes::Field> = evolved_schema.fields().iter().map(|f| (**f).clone()).collect();
                        let mut new_field = (**field).clone();
                        new_field.set_nullable(true);
                        fields[idx] = new_field;
                        evolved_schema = Schema::new(fields);
                        changed = true;
                    }
                } else {
                    // New column added
                    tracing::info!("Schema Evolution: Adding new column '{}'", field.name());
                    let mut fields: Vec<arrow::datatypes::Field> = evolved_schema.fields().iter().map(|f| (**f).clone()).collect();
                    fields.push((**field).clone());
                    evolved_schema = Schema::new(fields);
                    changed = true;
                }
            }

            if changed {
                let mut lock = self.schema.write().unwrap();
                *lock = Arc::new(evolved_schema);
                drop(lock);
                target_schema = self.arrow_schema();
            }
        }

        // 1. Strict Nullability Validation
        // Ensure no required (NOT NULL) columns contain nulls in the incoming batches,
        // using the potential EVOLVED schema.
        for batch in &batches {
            for field in target_schema.fields() {
                if !field.is_nullable() {
                    if let Some(col) = batch.column_by_name(field.name()) {
                        if col.null_count() > 0 {
                            return Err(anyhow::anyhow!("Null constraint violation: column '{}' is NOT NULL but batch contains {} nulls", field.name(), col.null_count()));
                        }
                    }
                }
            }
        }

        let batches: Vec<Result<RecordBatch>> = batches.into_iter().map(|b| {
            if b.schema() != target_schema {
                let mut cols = Vec::with_capacity(target_schema.fields().len());
                for field in target_schema.fields() {
                    let col = if let Some(c) = b.column_by_name(field.name()) {
                        c.clone()
                    } else {
                        arrow::array::new_null_array(field.data_type(), b.num_rows())
                    };
                    cols.push(col);
                }
                RecordBatch::try_new(target_schema.clone(), cols).map_err(|e| anyhow::anyhow!(e))
            } else {
                Ok(b)
            }
        }).collect();
        
        let batches: Vec<RecordBatch> = batches.into_iter().filter_map(|r| {
            match r {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!("Batch schema coercion failed during write: {}", e);
                    None
                }
            }
        }).collect();

        if is_empty_schema {
            // Schema was empty, now we have detected column types from the data
            // This enables schema-on-write behavior
        }

        // 1. Write-Ahead Log (Durability) & 2. Indexing (In-Memory) in PARALLEL
        let wal = self.wal.clone();
        let memory_index = self.indexing.memory_index.clone();
        let target_col = self.indexing.index_columns.read().unwrap().first().cloned()
             .or_else(|| {
                 batches.first().and_then(|b| {
                     b.schema().fields().iter()
                         .find(|f| f.name() == "embedding")
                         .map(|f| f.name().clone())
                 })
             });

        // 0. Primary Key Uniqueness Check (if defined)
        let primary_keys = self.get_primary_key();
        if !primary_keys.is_empty() {
            for batch in &batches {
                self.check_primary_key_uniqueness_async(batch, &primary_keys).await?;
            }
        }

        let buffer_len_before = { 
            let buffer = self.write_buffer.read().unwrap();
            buffer.iter().map(|b| b.num_rows()).sum()
        };

        let batches_for_wal = batches.clone();
        let batches_for_idx = batches.clone();
        
        let (wal_res, idx_res) = tokio::join!(
            // WAL Task
            async move {
                let wal_lock = wal.lock().await;
                for batch in batches_for_wal {
                    wal_lock.append_async(batch).await?;
                }
                
                // Check if WAL needs compaction (sync check for now)
                wal_lock.should_compact()?;
                Ok::<(), anyhow::Error>(())
            },
            // Indexing Task
            async move {
                if let Some(col_name) = target_col {
                    let mut idx_lock = memory_index.write().unwrap();
                    
                    if idx_lock.is_none() {
                        if let Some(first) = batches_for_idx.first() {
                            if let Some(col) = first.column_by_name(&col_name) {
                                if let Some(fsl) = col.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                                     let dim = fsl.value_length() as usize;
                                     *idx_lock = Some(InMemoryVectorIndex::new(dim));
                                }
                                // Time32 Not Supported
                                /*
                                else if let Some(time32) = col.as_any().downcast_ref::<arrow::array::Time32Array>() {
                                    // This block is for indexing, not vector search, so dim is not applicable here.
                                    // It should be handled by the `insert_batch` logic if Time32/Time64 indexing is supported.
                                }
                                */
                                // Time64 Not Supported
                                /*
                                else if let Some(time64) = col.as_any().downcast_ref::<arrow::array::Time64Array>() {
                                }
                                */
                            }
                        }
                    }

                    if let Some(idx) = idx_lock.as_mut() {
                        let mut current_offset = buffer_len_before;
                        for batch in &batches_for_idx {
                            let _ = idx.insert_batch(batch, &col_name, current_offset);
                            current_offset += batch.num_rows();
                        }
                    }
                }
                Ok::<(), anyhow::Error>(())
            }
        );

        wal_res?;
        idx_res?;
        // -----------------------------
 
        // Buffer the batches
        {
            let mut buffer = self.write_buffer.write().unwrap();
            buffer.extend(batches);
        }

        // Check if we should flush (spillover)
        let should_flush = {
            let buffer = self.write_buffer.read().unwrap();
            
            // Calculate size in bytes (approximate)
            let total_bytes: usize = buffer.iter()
                .map(|b| b.get_array_memory_size())
                .sum();
            
            let cache_gb: usize = std::env::var("HYPERSTREAM_CACHE_GB")
                .unwrap_or_else(|_| "2".to_string())
                .parse()
                .unwrap_or(2);
            let limit_bytes = cache_gb * 1024 * 1024 * 1024;
            
            total_bytes > limit_bytes
        };

        if should_flush || self.get_autocommit() {
            if should_flush {
                tracing::info!("Write buffer exceeded limit. Flushing to disk (Spillover)...");
            }
            self.commit_async().await?;
        }

        Ok(())
    }

    /// Flush buffer to disk
    /// Flush buffer to disk
    pub async fn flush_async(&self) -> Result<()> {
        // Type alias for stream results to avoid complex type annotation
        type PartitionSegment = (crate::core::manifest::ManifestEntry, Vec<String>, String, RecordBatch, HashMap<String, Value>);
        // Extract batches from buffer
        let batches_to_write: Vec<RecordBatch> = {
            let mut buffer = self.write_buffer.write().unwrap();
            if buffer.is_empty() {
                return Ok(());
            }
            std::mem::take(&mut *buffer)
        };

        // Reset memory index
        {
            let mut idx = self.indexing.memory_index.write().unwrap();
            *idx = None;
        }

        if batches_to_write.is_empty() {
            return Ok(());
        }

        // --- NEW: Coalesce Batches for larger segments ---
        let schema = batches_to_write[0].schema();
        let mut coalesced_batch = arrow::compute::concat_batches(&schema, &batches_to_write)?;
        
        // --- NEW: Vector Shuffling (Optimization inspired by LanceDB)
        // Attribution: Adapted from Lance partition/shuffle logic (Apache 2.0)
        // Copyright The Lance Authors. Modified for HyperStreamDB Parquet layout.
        // MODIFIED by Richard Albright / HyperStreamDB on 2026-03-29 to integrate with Iceberg V2/V3 manifests and sidecar indexing.
        if let Some(vector_col) = self.get_vector_column_for_shuffling(&coalesced_batch) {
            tracing::info!("Optimizing data layout: Shuffling rows by vector similarity (LanceDB-style)...");
            coalesced_batch = self.shuffle_batch_by_centroids(&coalesced_batch, &vector_col).await?;
        }
        
        // Apply sort order if configured (Iceberg V2 spec compliance)
        let sorted_batch = self.apply_sort_order(&coalesced_batch)?;
        
        let spec = self.partition_spec.clone();
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        
        // Add V3 metadata columns if format_version >= 3 (Iceberg V3 Row Lineage)
        let (manifest, _, _) = manifest_manager.load_latest_full().await.unwrap_or_default();
        let batch_with_metadata = if manifest.format_version >= 3 {
            let sequence_number = manifest.version as i64;
            self.add_v3_metadata_columns(&sorted_batch, sequence_number)?
        } else {
            sorted_batch.clone()
        };

        // Split batches by partition
        let partitioned_batches = spec.partition_batch(&batch_with_metadata)?;
        
        // Extract local path from URI for writer
        let base_path = self.uri.strip_prefix("file://").unwrap_or(&self.uri);
        std::fs::create_dir_all(base_path)?;
        
        let mut all_new_entries = Vec::new();
        let mut all_generated_files: Vec<String> = Vec::new();
        let index_cols = self.indexing.index_columns.read().unwrap().clone();
        let index_all_flag = self.indexing.index_all;

        let index_configs_map: HashMap<String, crate::core::table::ColumnIndexConfig> = {
            self.indexing.index_configs.read().unwrap().clone()
        };
        let default_device = self.indexing.default_device.read().unwrap().clone();
        let default_device_for_stream = default_device.clone();

        // Parallelize partition writing using futures stream
        let stream = futures::stream::iter(partitioned_batches.into_iter().map(|(partition_values, batch)| {
            let base_path = base_path.to_string();
            let spec = spec.clone();
            let default_device_inner = default_device_for_stream.clone();
            async move {
                let segment_id = format!("seg_{}", uuid::Uuid::new_v4());
                let hive_path = spec.partition_to_path(&partition_values);
                let full_base_path = if hive_path.is_empty() { 
                    base_path 
                } else { 
                    format!("{}/{}", base_path, hive_path)
                };
                let _ = std::fs::create_dir_all(&full_base_path);

                // 1. Create writer for data write (no index building yet)
                let config_write = SegmentConfig::new(&full_base_path, &segment_id)
                    .with_index_all(false)
                    .with_columns_to_index(Vec::new())
                    .with_partition_values(partition_values.clone())
                    .with_default_device(default_device_inner);
                let mut writer_write = HybridSegmentWriter::new(config_write);
                writer_write.primary_key = self.primary_key.read().unwrap().clone();
                writer_write.set_store(self.store.clone());

                let batch_inner = batch.clone();
                let (entry, generated_files) = tokio::task::spawn_blocking(move || {
                    writer_write.write_batch(&batch_inner)?;
                    let entry = writer_write.to_manifest_entry();
                    let files = writer_write.get_generated_files(); 
                    Ok::<(crate::core::manifest::ManifestEntry, Vec<String>), anyhow::Error>((entry, files))
                }).await.context("Flush task panicked")??;
                
                Ok::<PartitionSegment, anyhow::Error>((entry, generated_files, segment_id, batch, partition_values))
            }
        })).buffer_unordered(16); // High concurrency for IO

        let results: Vec<Result<PartitionSegment>> = stream.collect().await;
        
        for res in results {
            let spec = spec.clone();
            let (mut entry, generated_files, segment_id, batch, partition_values): PartitionSegment = res?;
            
            // Adjust paths relative to Table Root (Hive style)
            let hive_path = spec.partition_to_path(&partition_values);
            if !hive_path.is_empty() {
                entry.file_path = format!("{}/{}", hive_path, entry.file_path);
                for idx in &mut entry.index_files {
                    idx.file_path = format!("{}/{}", hive_path, idx.file_path);
                }
            }

            all_generated_files.extend(generated_files);
            all_new_entries.push(entry.clone());

            // 2. Queue index building asynchronously (if needed)
            let has_pks = !self.primary_key.read().unwrap().is_empty();
            if index_all_flag || !index_cols.is_empty() || has_pks {
                let index_cols_clone = index_cols.clone();
                let base_path_clone = base_path.to_string();
                let segment_id_clone = segment_id.clone();
                let batch_for_indexing = batch.clone();
                let partition_values_clone = partition_values.clone();
                let index_configs_clone = index_configs_map.clone();
                let default_device_clone = default_device.clone();
                
                let entry_clone = entry.clone();
                let manifest_manager_clone = manifest_manager.clone();

                let pk_clone = self.primary_key.read().unwrap().clone();
                let table_store = self.store.clone();

                let spec_bg = spec.clone();
                let handle = tokio::spawn(async move {
                    let _start = std::time::Instant::now();
                    
                    let hive_path = spec_bg.partition_to_path(&partition_values_clone);
                    let full_base_path = if hive_path.is_empty() { 
                        base_path_clone 
                    } else { 
                        format!("{}/{}", base_path_clone, hive_path)
                    };

                    let parquet_path_rel = if hive_path.is_empty() {
                        format!("{}.parquet", segment_id_clone)
                    } else {
                        format!("{}/{}.parquet", hive_path, segment_id_clone)
                    };

                    let config_index = SegmentConfig::new(&full_base_path, &segment_id_clone)
                        .with_index_all(index_all_flag)
                        .with_columns_to_index(index_cols_clone)
                        .with_partition_values(partition_values_clone.clone())
                        .with_column_devices(index_configs_clone.iter().filter_map(|(c, cfg)| cfg.device.as_ref().map(|d| (c.clone(), d.clone()))).collect())
                        .with_default_device(default_device_clone)
                        .with_parquet_path(parquet_path_rel);
                    
                    let index_res = tokio::spawn(async move {
                        let mut index_writer = HybridSegmentWriter::new(config_index)
                            .with_index_configs(index_configs_clone);
                        index_writer.primary_key = pk_clone;
                        index_writer.set_store(table_store);
                        // In commit path, we typically have ONE batch per segment write
                        index_writer.build_indexes(&batch_for_indexing, 0)?; 
                        index_writer.finish_indexing().await?;
                        let files = index_writer.get_generated_files();
                        let updated_entry_info = index_writer.to_manifest_entry();
                        Ok::<(crate::core::manifest::ManifestEntry, Vec<String>), anyhow::Error>((updated_entry_info, files))
                    }).await;

                    match index_res {
                        Ok(Ok((updated_entry, _files))) => {
                            // Atomic metadata update: Add index files to the existing entry
                            let mut merged_entry = entry_clone;
                            let mut updated_index_files = updated_entry.index_files;

                            // Prefix index files if in a partitioned subdirectory
                            let hive_path = spec_bg.partition_to_path(&partition_values_clone);
                            if !hive_path.is_empty() {
                                for idx in &mut updated_index_files {
                                    idx.file_path = format!("{}/{}", hive_path, idx.file_path);
                                }
                            }

                            tracing::debug!("Background indexing task: segment={}, found index_files={:?}", merged_entry.file_path, updated_index_files);
                            merged_entry.index_files = updated_index_files;
                            // NOTE: We MUST preserve the record_count and column_stats from entry_clone,
                            // as updated_entry (from index_writer) only contains the index metadata.
                            
                            let commit_metadata = crate::core::manifest::CommitMetadata::default();

                            let file_path = merged_entry.file_path.clone();
                            let index_count = merged_entry.index_files.len();

                            let remove_paths = vec![merged_entry.file_path.clone()];
                            match manifest_manager_clone.commit(&[merged_entry], &remove_paths, commit_metadata).await {
                                Ok(_) => tracing::info!("Successfully attached {} indexes to manifest for segment {}", 
                                                 index_count, file_path),
                                Err(e) => tracing::error!("Failed to attach indexes for segment {}: {}", file_path, e),
                            }
                        }
                        _ => {
                            tracing::error!("Index building failed for segment {}", segment_id_clone);
                        }
                    }
                });
                self.background_tasks.lock().await.push(handle);
            }
        }
        
        // Detect if schema has evolved since last manifest load
        let (manifest, _, _) = manifest_manager.load_latest_full().await.unwrap_or_default();
        let current_schema = self.arrow_schema();
        
        let should_update_schema = if manifest.schemas.is_empty() {
            true
        } else {
            let latest_schema = manifest.schemas.last().unwrap().to_arrow();
            // Compare schemas, but ignore metadata if necessary. 
            // Simple != check works for basic evolution.
            latest_schema != *current_schema
        };

        let (final_schemas, final_schema_id) = if should_update_schema {
            let mut new_schemas = manifest.schemas.clone();
            let new_id = if manifest.schemas.is_empty() { 0 } else { manifest.current_schema_id + 1 };
            new_schemas.push(crate::core::manifest::Schema::from_arrow(&current_schema, new_id));
            (Some(new_schemas), Some(new_id))
        } else {
            (None, None)
        };

        // Final commit for all data segments (with possible schema update)
        let commit_metadata = crate::core::manifest::CommitMetadata {
            updated_schemas: final_schemas,
            updated_schema_id: final_schema_id,
            updated_partition_specs: None,
            updated_default_spec_id: None,
            updated_properties: None,
            removed_properties: None,
            updated_sort_orders: None,
            updated_default_sort_order_id: None,
            updated_last_column_id: None,
            is_fast_append: false,
        };
        
        let new_manifest = manifest_manager.commit(&all_new_entries, &[], commit_metadata).await?;

        // 4. Update Table Metadata (Iceberg v2 Spec)
        // Determine the root for metadata. If we have a catalog, use its reported location.
        let meta_location = if let Some(catalog) = &self.catalog_state.catalog {
            if let (Some(ns), Some(t)) = (&self.catalog_state.namespace, &self.catalog_state.table_name) {
                catalog.load_table(ns, t).await.map(|m| m.location).unwrap_or_else(|_| self.uri.clone())
            } else {
                self.uri.clone()
            }
        } else {
            self.uri.clone()
        };

        let meta_store_arc = create_object_store(&meta_location)?;
        let meta_store = meta_store_arc.as_ref();
        
        let mut table_meta = match TableMetadata::load_latest(meta_store).await {
            Ok(meta) => meta,
            Err(_) => {
                // Initialize skeleton if not found
                TableMetadata::new(
                    2, 
                    uuid::Uuid::new_v4().to_string(), 
                    self.uri.clone(), 
                    new_manifest.schemas.last().cloned().unwrap_or_else(|| {
                        crate::core::manifest::Schema::from_arrow(&current_schema, 0)
                    }), 
                    new_manifest.partition_spec.clone(),
                    new_manifest.sort_orders.first().cloned().unwrap_or_default()
                )
            }
        };

        // Add a new snapshot pointing to the latest manifest
        let snapshot = crate::core::metadata::Snapshot {
            snapshot_id: new_manifest.version as i64,
            parent_snapshot_id: table_meta.current_snapshot_id,
            timestamp_ms: new_manifest.timestamp_ms,
            sequence_number: Some(new_manifest.version as i64),
            summary: HashMap::from([("operation".to_string(), "append".to_string())]),
            manifest_list: new_manifest.manifest_list_path.clone().unwrap_or_default(),
            schema_id: Some(new_manifest.current_schema_id),
            first_row_id: table_meta.next_row_id,
            added_rows: None, // TODO: Calculate from manifest
        };
        table_meta.add_snapshot(snapshot);
        
        // Save metadata file (vX.metadata.json)
        let new_meta_version = (new_manifest.version) as i32; // Sync with manifest version for simplicity
        table_meta.save_to_store(meta_store, new_meta_version).await?;
        
        // 5. Commit to Catalog if configured (Iceberg Atomic Swap)
        if let Some(catalog) = &self.catalog_state.catalog {
            if let (Some(ns), Some(table)) = (&self.catalog_state.namespace, &self.catalog_state.table_name) {
                let updates = vec![
                    serde_json::json!({
                        "action": "add-snapshot",
                        "snapshot": table_meta.snapshots.last().unwrap()
                    }),
                    serde_json::json!({
                        "action": "set-current-snapshot",
                        "snapshot-id": table_meta.current_snapshot_id
                    })
                ];
                catalog.commit_table(ns, table, updates).await?;
                tracing::info!("Committed snapshot {} to catalog {}.{}", new_manifest.version, ns, table);
            }
        }

        // 3. Upload data files asynchronously if remote
        if self.uri.contains("://") && !self.uri.starts_with("file://") {
             let store_clone = self.store.clone();
             let files_to_upload = all_generated_files;
             let handle = tokio::spawn(async move {
                 for file_path in files_to_upload {
                      let local_path = std::path::Path::new(file_path.as_str());
                      if let Ok(data) = tokio::fs::read(&local_path).await {
                           let remote_path = object_store::path::Path::from(file_path.as_str());
                           let _ = store_clone.put(&remote_path, data.into()).await;
                      }
                 }
             });
             self.background_tasks.lock().await.push(handle);
        }
        
         // 5. Truncate WAL (Durability Checkpoint)
         {
              let mut wal = self.wal.lock().await;
              wal.truncate().context("Failed to truncate WAL")?;
              
              // 5b. Cleanup recovered files
              let mut recovered = self.recovered_wal_paths.lock().unwrap();
              if !recovered.is_empty() {
                  let paths: Vec<std::path::PathBuf> = recovered.iter().map(std::path::PathBuf::from).collect();
                  wal.cleanup_files(&paths).unwrap_or_default();
                  recovered.clear();
              }
         }

        Ok(())
    }
}
