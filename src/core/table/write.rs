// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::core::error::BenoStreamError;
use crate::core::manifest::ManifestManager;
use crate::telemetry::metrics::INGEST_ROWS_TOTAL;
use crate::SegmentConfig;

use super::Table;
use crate::core::index::memory::InMemoryVectorIndex;
use crate::core::metadata::TableMetadata;
use crate::core::segment::HybridSegmentWriter;
use crate::core::storage::create_object_store;
use arrow::datatypes::Schema;
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;

/// Resident set size in bytes, or `0` when the platform cannot report it.
///
/// Delegates to [`crate::core::memory::rss_bytes`] so the back-pressure
/// high-water mark and the heap-trim policy read the same number. A `0` result
/// (unknown) makes the back-pressure check fail-open rather than block forever.
fn current_rss_bytes() -> usize {
    crate::core::memory::rss_bytes().unwrap_or(0) as usize
}

impl Table {
    /// Write Arrow RecordBatches to the table (Buffered)
    ///
    /// Data is written to an in-memory buffer. It is NOT persisted to disk until:
    /// 1. The buffer exceeds `BENOSTREAM_CACHE_GB`
    /// 2. `commit()` is called explicitly
    pub fn write(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.runtime().block_on(self.write_async(batches))
    }

    /// Commit buffered writes to disk
    pub fn commit(&self) -> Result<()> {
        self.runtime().block_on(self.flush_async())
    }

    /// Async commit
    #[tracing::instrument(skip(self))]
    pub async fn commit_async(&self) -> Result<()> {
        self.flush_async().await?;
        Ok(())
    }

    /// Truncate the table (metadata-only operation)
    pub fn truncate(&self) -> Result<()> {
        self.runtime().block_on(self.truncate_async())
    }

    /// Async implementation of truncate
    #[tracing::instrument(skip(self))]
    pub async fn truncate_async(&self) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);

        // Step 1: Get all current entry paths
        // We must load THE WHOLE current manifest to know what to remove.
        let (_, entries, _) = manifest_manager.load_latest_full().await.unwrap_or((
            crate::core::manifest::Manifest::default(),
            Vec::new(),
            0,
        ));
        let remove_paths: Vec<String> = entries.iter().map(|e| e.file_path.clone()).collect();

        // Step 2: Commit with all paths in remove_paths and empty add_entries
        // This is a single atomic snapshot swap that points to an empty segment set.
        manifest_manager
            .commit(
                &[],
                &remove_paths,
                crate::core::manifest::CommitMetadata::default(),
            )
            .await?;

        // Step 3: Truncate WAL
        {
            let mut wal = self.wal.lock().await;
            wal.truncate()
                .context("Failed to truncate WAL during table truncate")?;
        }

        // Step 4: Clear memory index
        {
            let mut idx = self.indexing.memory_index.write();
            *idx = None;
        }

        // Step 5: Clear write buffer
        {
            let mut buffer = self.write_buffer.write();
            buffer.clear();
        }

        tracing::info!(
            "Table truncated: metadata reset, all {} segments removed, buffers cleared.",
            remove_paths.len()
        );
        Ok(())
    }

    /// Compact the WAL (consolidate log entries)
    pub fn checkpoint(&self) -> Result<()> {
        let mut wal = self.wal.blocking_lock();
        wal.compact()
    }

    // Schema evolution logic moved to schema.rs

    /// Async implementation of write using the table's configured durability mode.
    #[tracing::instrument(skip(self, batches))]
    pub async fn write_async(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.write_with_durability_async(batches, self.durability)
            .await
    }

    /// Write batches with asynchronous durability (buffered in WAL worker without waiting for fsync).
    /// Best for maximum streaming ingestion throughput.
    #[tracing::instrument(skip(self, batches))]
    pub async fn write_buffered_async(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.write_with_durability_async(batches, crate::core::table::WalDurability::Async)
            .await
    }

    /// Explicitly flush and sync pending WAL writes to durable storage.
    pub async fn flush_wal_async(&self) -> Result<()> {
        let wal = self.wal.lock().await;
        wal.flush_async().await
    }

    /// Internal write implementation with explicit durability specification.
    #[tracing::instrument(skip(self, batches))]
    pub async fn write_with_durability_async(
        &self,
        batches: Vec<RecordBatch>,
        durability: crate::core::table::WalDurability,
    ) -> Result<()> {
        if let Some(max_gb) = self.max_ingest_ram_gb {
            let max_bytes = (max_gb * 1_000_000_000.0) as usize;
            let mut logged = false;
            let mut paused: Option<std::time::Instant> = None;
            // Back-pressure on the ingest RAM high-water mark. Wait on a
            // notification from background tasks that release memory (index
            // builds, heap trims) with a bounded fallback poll so we still
            // re-check RSS if nothing fires. The previous implementation slept a
            // flat 500 ms per iteration, which both wasted time when memory was
            // freed promptly and delayed resumption when it was not.
            while max_bytes > 0 && current_rss_bytes() >= max_bytes {
                if !logged {
                    tracing::warn!(
                        "RSS ({:.2} GB) exceeds max ingest RAM limit ({:.2} GB). Pausing ingestion until background tasks reclaim memory...",
                        current_rss_bytes() as f64 / 1_000_000_000.0,
                        max_gb
                    );
                    logged = true;
                }
                if paused.is_none() {
                    paused = Some(std::time::Instant::now());
                }
                tokio::select! {
                    _ = self.memory_reclaimed.notified() => {}
                    _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {}
                }
            }
            crate::telemetry::metrics::INGEST_RSS_BYTES_GAUGE.set(current_rss_bytes() as i64);
            if let Some(start) = paused {
                let paused_secs = start.elapsed().as_secs_f64();
                crate::telemetry::metrics::INGEST_BACKPRESSURE_PAUSES_TOTAL.inc();
                crate::telemetry::metrics::INGEST_BACKPRESSURE_PAUSE_SECONDS.observe(paused_secs);
                tracing::info!(
                    paused_seconds = paused_secs,
                    "RSS back under the ingest RAM limit; resuming ingestion"
                );
            }
        }

        // Disk admission: refuse a flush when a local table's filesystem is below
        // the free-space threshold (default `BSDB_MIN_FREE_DISK_GB`), so the
        // failure is a clear error instead of a partial segment written mid-way.
        // Fail-open when the free space is unknown.
        if let Some(local) = self.uri.strip_prefix("file://") {
            let path = std::path::Path::new(local);
            if let Some(free) = crate::core::resources::free_disk_bytes_for_new_file(path) {
                crate::telemetry::metrics::FREE_DISK_BYTES_GAUGE.set(free as i64);
            }
            crate::core::resources::check_min_free_disk(path).map_err(|e| anyhow::anyhow!(e))?;
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        INGEST_ROWS_TOTAL.inc_by(total_rows as u64);

        if batches.is_empty() {
            return Ok(());
        }

        let batches = batches;

        let mut is_empty_schema = false;
        if let Some(first_batch) = batches.first() {
            let mut lock = self.schema.write();
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
                            crate::core::table::LabelPattern::ExcelAlpha => {
                                crate::core::table::excel_column_label(i)
                            }
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
        let t_pk = std::time::Instant::now();
        let pk_cols = self.primary_key.read().clone();
        if !pk_cols.is_empty() {
            // Accelerated path for single-column Primary Keys
            if pk_cols.len() == 1 {
                let pk_col = &pk_cols[0];
                let mut seen_keys = std::collections::HashSet::new();

                {
                    let buffer = self.write_buffer.read();
                    // 1. Pre-populate seen keys from the in-memory write buffer
                    for b_batch in buffer.iter() {
                        if let Some(b_col) = b_batch.column_by_name(pk_col) {
                            for j in 0..b_batch.num_rows() {
                                let b_val =
                                    crate::core::manifest::ManifestValue::from_array(b_col, j);
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
                                return Err(anyhow::Error::from(
                                    BenoStreamError::NullConstraintViolation {
                                        column: pk_col.clone(),
                                    },
                                ));
                            }

                            let val_str = m_val.to_string();

                            // Check against buffer and current batch
                            if seen_keys.contains(&val_str) {
                                return Err(anyhow::Error::from(
                                    BenoStreamError::PrimaryKeyViolation {
                                        key: val_str.clone(),
                                    },
                                ));
                            }

                            // 3. Check against storage (Index-driven)
                            let val_json =
                                serde_json::to_value(&m_val).unwrap_or(serde_json::Value::Null);
                            if self._check_pk_in_storage_async(pk_col, &val_json).await? {
                                return Err(anyhow::Error::from(
                                    BenoStreamError::PrimaryKeyViolation {
                                        key: val_str.clone(),
                                    },
                                ));
                            }

                            seen_keys.insert(val_str);
                        }
                    }
                }
            } else {
                let buffer = self.write_buffer.read();
                // Fallback for multi-column PKs (O(N*M) check for now)
                for batch in &batches {
                    for pk_col in &pk_cols {
                        if let Some(col) = batch.column_by_name(pk_col) {
                            for i in 0..batch.num_rows() {
                                let val = crate::core::manifest::ManifestValue::from_array(col, i);

                                // Check for nulls in PK
                                if matches!(val, crate::core::manifest::ManifestValue::Null) {
                                    return Err(anyhow::Error::from(
                                        BenoStreamError::NullConstraintViolation {
                                            column: pk_col.clone(),
                                        },
                                    ));
                                }

                                // Check against buffer
                                for b_batch in buffer.iter() {
                                    if let Some(b_col) = b_batch.column_by_name(pk_col) {
                                        for j in 0..b_batch.num_rows() {
                                            let b_val =
                                                crate::core::manifest::ManifestValue::from_array(
                                                    b_col, j,
                                                );
                                            if val == b_val {
                                                return Err(anyhow::Error::from(
                                                    BenoStreamError::PrimaryKeyViolation {
                                                        key: val.to_string(),
                                                    },
                                                ));
                                            }
                                        }
                                    }
                                }

                                // Check against other rows in the same batch (before index i)
                                for j in 0..i {
                                    let b_val =
                                        crate::core::manifest::ManifestValue::from_array(col, j);
                                    if val == b_val {
                                        return Err(anyhow::Error::from(
                                            BenoStreamError::PrimaryKeyViolation {
                                                key: val.to_string(),
                                            },
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let pk_ms = t_pk.elapsed().as_millis();

        let t_coerce = std::time::Instant::now();
        let mut target_schema = self.arrow_schema();

        if let Some(first_batch) = batches.first() {
            let incoming_schema = first_batch.schema();
            let mut evolved_schema = (*target_schema).clone();
            let mut changed = false;

            for field in incoming_schema.fields() {
                let existing_field_info = evolved_schema
                    .field_with_name(field.name())
                    .ok()
                    .map(|f| (f.data_type().clone(), f.is_nullable()));

                if let Some((existing_dtype, existing_nullable)) = existing_field_info {
                    // Check if we need to widen the type (e.g. Int32 -> Int64)
                    if existing_dtype != *field.data_type() {
                        tracing::info!(
                            "Schema Evolution: Widening column '{}' from {:?} to {:?}",
                            field.name(),
                            existing_dtype,
                            field.data_type()
                        );

                        // `field` came from `evolved_schema`, so `index_of` cannot
                        // fail; skip defensively rather than panicking the writer.
                        if let Ok(idx) = evolved_schema.index_of(field.name()) {
                            let mut fields: Vec<arrow::datatypes::Field> = evolved_schema
                                .fields()
                                .iter()
                                .map(|f| (**f).clone())
                                .collect();
                            fields[idx] = (**field).clone();
                            evolved_schema = Schema::new(fields);
                            changed = true;
                        }
                    }

                    // Check if we need to change Nullability (Required -> Nullable)
                    if !existing_nullable && field.is_nullable() {
                        tracing::info!(
                            "Schema Evolution: Changing column '{}' to nullable",
                            field.name()
                        );
                        if let Ok(idx) = evolved_schema.index_of(field.name()) {
                            let mut fields: Vec<arrow::datatypes::Field> = evolved_schema
                                .fields()
                                .iter()
                                .map(|f| (**f).clone())
                                .collect();
                            let mut new_field = (**field).clone();
                            new_field.set_nullable(true);
                            fields[idx] = new_field;
                            evolved_schema = Schema::new(fields);
                            changed = true;
                        }
                    }
                } else {
                    // New column added
                    tracing::info!("Schema Evolution: Adding new column '{}'", field.name());
                    let mut fields: Vec<arrow::datatypes::Field> = evolved_schema
                        .fields()
                        .iter()
                        .map(|f| (**f).clone())
                        .collect();
                    fields.push((**field).clone());
                    evolved_schema = Schema::new(fields);
                    changed = true;
                }
            }

            if changed {
                let mut lock = self.schema.write();
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

        let batches: Vec<Result<RecordBatch>> = batches
            .into_iter()
            .map(|b| {
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
                    RecordBatch::try_new(target_schema.clone(), cols)
                        .map_err(|e| anyhow::anyhow!(e))
                } else {
                    Ok(b)
                }
            })
            .collect();

        let batches: Vec<RecordBatch> = batches
            .into_iter()
            .filter_map(|r| match r {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!("Batch schema coercion failed during write: {}", e);
                    None
                }
            })
            .collect();

        if is_empty_schema {
            // Schema was empty, now we have detected column types from the data
            // This enables schema-on-write behavior
        }

        let coerce_ms = t_coerce.elapsed().as_millis();

        // 1. Write-Ahead Log (Durability) & 2. Indexing (In-Memory) in PARALLEL
        let t_wal = std::time::Instant::now();
        let wal = self.wal.clone();
        let memory_index = self.indexing.memory_index.clone();
        // Only use explicitly configured index columns — do NOT auto-detect by column name.
        // Auto-detecting "embedding" caused silent 15s HNSW builds on every write.
        let mut target_col = self.indexing.index_columns.read().first().cloned();
        if target_col.is_none() && self.indexing.index_all {
            if let Some(first) = batches.first() {
                target_col = first
                    .schema()
                    .fields()
                    .iter()
                    .find(|f| {
                        matches!(
                            f.data_type(),
                            arrow::datatypes::DataType::List(_)
                                | arrow::datatypes::DataType::FixedSizeList(_, _)
                        )
                    })
                    .map(|f| f.name().clone());
            }
        }

        // 0. Primary Key Uniqueness Check (if defined)
        let primary_keys = self.get_primary_key();
        if !primary_keys.is_empty() {
            for batch in &batches {
                for pk in &primary_keys {
                    if let Some(col) = batch.column_by_name(pk) {
                        let mut seen = std::collections::HashSet::with_capacity(batch.num_rows());
                        for i in 0..batch.num_rows() {
                            let val_str = crate::core::manifest::ManifestValue::from_array(col, i)
                                .to_string();
                            if !seen.insert(val_str.clone()) {
                                return Err(anyhow::Error::from(
                                    BenoStreamError::PrimaryKeyViolation { key: val_str },
                                ));
                            }
                        }
                    }
                }
                // Bypassed for ingestion performance optimization
                // self.check_primary_key_uniqueness_async(batch, &primary_keys).await?;
            }
        }

        let buffer_len_before = {
            let buffer = self.write_buffer.read();
            buffer.iter().map(|b| b.num_rows()).sum()
        };

        let tx_id = uuid::Uuid::new_v4();
        let batches_for_wal: Vec<RecordBatch> = batches
            .iter()
            .enumerate()
            .map(|(i, b)| {
                crate::core::wal::tag_batch_with_wal_tx(b, tx_id, i as u64)
                    .unwrap_or_else(|_| b.clone())
            })
            .collect();
        let batches_for_idx = batches.clone();

        let (wal_res, idx_res) = tokio::join!(
            // WAL Task — appends according to durability configuration (Sync waits for disk sync, Async batches)
            async move {
                let t_wal_task = std::time::Instant::now();
                let wal_lock = wal.lock().await;
                let t_wal_lock = t_wal_task.elapsed().as_millis();
                // WS2 crash boundary: process dies after the batch is handed to
                // the WAL but before it is fsynced.
                crate::core::fault_injection::check(
                    crate::core::fault_injection::CrashPoint::WalAppend,
                )?;
                for batch in batches_for_wal {
                    match durability {
                        crate::core::table::WalDurability::Sync => {
                            wal_lock.append_sync(batch).await?;
                        }
                        crate::core::table::WalDurability::Async => {
                            wal_lock.append_fire_and_forget(batch).await?;
                        }
                    }
                }
                wal_lock.should_compact()?;
                let t_wal_total = t_wal_task.elapsed().as_millis();
                tracing::debug!(
                    wal_lock_ms = t_wal_lock,
                    wal_total_ms = t_wal_total,
                    "wal_task timing"
                );
                Ok::<(), anyhow::Error>(())
            },
            // Indexing Task
            async move {
                let _t_idx = std::time::Instant::now();
                let t_idx_task = std::time::Instant::now();
                if let Some(col_name) = target_col {
                    let mut idx_lock = memory_index.write();

                    if idx_lock.is_none() {
                        if let Some(first) = batches_for_idx.first() {
                            if let Some(col) = first.column_by_name(&col_name) {
                                if let Some(fsl) = col
                                    .as_any()
                                    .downcast_ref::<arrow::array::FixedSizeListArray>()
                                {
                                    let dim = fsl.value_length() as usize;
                                    *idx_lock = Some(InMemoryVectorIndex::new(dim));
                                } else if let Some(list) =
                                    col.as_any().downcast_ref::<arrow::array::ListArray>()
                                {
                                    let mut dim = 0;
                                    for i in 0..list.len() {
                                        if !list.is_null(i) {
                                            dim = list.value(i).len();
                                            break;
                                        }
                                    }
                                    if dim > 0 {
                                        *idx_lock = Some(InMemoryVectorIndex::new(dim));
                                    }
                                } else if matches!(
                                    col.data_type(),
                                    arrow::datatypes::DataType::Time32(_)
                                        | arrow::datatypes::DataType::Time64(_)
                                ) {
                                    // Time datatypes are not vectorizable; safely
                                    // skip in-memory vector indexing for them.
                                    tracing::debug!(
                                        "Skipping in-memory vector index for Time datatype column '{}'",
                                        col_name
                                    );
                                }
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
                tracing::debug!(
                    idx_task_ms = t_idx_task.elapsed().as_millis(),
                    "idx_task timing"
                );
                Ok::<(), anyhow::Error>(())
            }
        );

        wal_res?;
        // WS2 crash boundary: WAL is durable, but the manifest has not been
        // committed yet. On reopen the WAL replay must recover these rows.
        crate::core::fault_injection::check(
            crate::core::fault_injection::CrashPoint::WalFlush,
        )?;
        idx_res?;
        let wal_idx_ms = t_wal.elapsed().as_millis();
        // -----------------------------
        tracing::debug!(
            rows = total_rows,
            pk_ms,
            coerce_ms,
            wal_idx_ms,
            "write_with_durability_async phase timing"
        );

        let _write_buffer_len = {
            let mut buffer = self.write_buffer.write();
            buffer.extend(batches);
            // Record the WAL tx id alongside the buffered rows, under the same
            // lock, so the tx id and its rows are always taken together at flush
            // time (see `flush_async`). This is what makes WAL replay idempotent.
            self.pending_wal_tx_ids.lock().push(tx_id);
            buffer.len()
        };

        // Check if we should flush (spillover)
        let should_flush = {
            let buffer = self.write_buffer.read();

            // Calculate size in bytes (approximate)
            let total_bytes: usize = buffer.iter().map(|b| b.get_array_memory_size()).sum();

            let cache_gb: usize = std::env::var("BENOSTREAM_CACHE_GB")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .unwrap_or(1);
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
    #[tracing::instrument(skip(self))]
    pub async fn flush_async(&self) -> Result<()> {
        // Extract batches from buffer. The WAL tx ids are taken under the same
        // lock so they stay paired with the rows they tag.
        let (batches_to_write, tx_ids): (Vec<RecordBatch>, Vec<uuid::Uuid>) = {
            let mut buffer = self.write_buffer.write();
            if buffer.is_empty() {
                return Ok(());
            }
            let batches = std::mem::take(&mut *buffer);
            let tx_ids = std::mem::take(&mut *self.pending_wal_tx_ids.lock());
            (batches, tx_ids)
        };

        // If flush fails at any point (upload, network, catalog lock), restore batches back into write_buffer
        // so in-memory visibility is preserved and subsequent calls can retry!
        let flush_res = self.flush_internal_async(&batches_to_write, &tx_ids).await;
        if let Err(e) = flush_res {
            let mut buffer = self.write_buffer.write();
            let mut restored = batches_to_write;
            restored.append(&mut *buffer);
            *buffer = restored;
            // Restore the tx ids too, so a retry records them at commit time.
            let mut pending = self.pending_wal_tx_ids.lock();
            let mut restored_tx = tx_ids;
            restored_tx.append(&mut *pending);
            *pending = restored_tx;
            return Err(e);
        }

        Ok(())
    }

    async fn flush_internal_async(
        &self,
        batches_to_write: &[RecordBatch],
        wal_tx_ids: &[uuid::Uuid],
    ) -> Result<()> {
        // Type alias for stream results to avoid complex type annotation
        type PartitionSegment = (
            crate::core::manifest::ManifestEntry,
            Vec<String>,
            String,
            RecordBatch,
            HashMap<String, Value>,
        );

        // Reset memory index
        {
            let mut idx = self.indexing.memory_index.write();
            *idx = None;
        }

        if batches_to_write.is_empty() {
            return Ok(());
        }

        let spec = self.partition_spec.clone();
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);

        // Add V3 metadata columns if format_version >= 3 (Iceberg V3 Row Lineage).
        // `load_latest_full` (not `load_latest`) so sharded manifest entries are
        // included — `row_id_base` must see every existing data file.
        let (manifest, existing_entries, _) = manifest_manager
            .load_latest_full()
            .await
            .unwrap_or_default();
        let sequence_number = manifest.version as i64;
        let format_version = self.get_format_version();

        // Iceberg V3 row lineage: the next `_row_id` to assign is one past the
        // highest row ID already present in the table. Derive it from the
        // manifest entries' `first_row_id` + `record_count` so it is monotonic
        // and collision-free across snapshots.
        let row_id_base: i64 = existing_entries
            .iter()
            .filter_map(|e| e.first_row_id.map(|f| f + e.record_count))
            .max()
            .unwrap_or(0);

        // Consolidate batches of compatible schemas before partitioning/writing
        // so single-row inserts from streaming/REST ingest don't create thousands of 1-row files!
        let consolidated_batches: Vec<RecordBatch> = {
            let mut groups: Vec<(arrow::datatypes::SchemaRef, Vec<RecordBatch>)> = Vec::new();
            for b in batches_to_write {
                if b.num_rows() == 0 {
                    continue;
                }
                let mut placed = false;
                for (schema, list) in groups.iter_mut() {
                    if schema == &b.schema() {
                        list.push(b.clone());
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    groups.push((b.schema(), vec![b.clone()]));
                }
            }
            let mut res = Vec::new();
            for (schema, list) in groups {
                if list.len() == 1 {
                    if let Some(first) = list.into_iter().next() {
                        res.push(first);
                    }
                } else if let Ok(merged) = arrow::compute::concat_batches(&schema, &list) {
                    res.push(merged);
                } else {
                    res.extend(list);
                }
            }
            res
        };

        // Align every batch to the table's full evolved schema so that
        // schema-evolved columns (e.g. a column added by a later doc) are
        // present in EVERY segment — with NULLs where a doc didn't have the
        // field. Without this, docs written before the column existed produce
        // segments missing the column, and reads concatenate the segments
        // dropping the column's values (data loss on schema evolution).
        let table_schema = self.arrow_schema();
        let aligned_batches: Vec<RecordBatch> = consolidated_batches
            .into_iter()
            .map(|b| {
                if b.schema() == table_schema {
                    b
                } else {
                    let mut cols = Vec::with_capacity(table_schema.fields().len());
                    for field in table_schema.fields() {
                        let col = if let Some(c) = b.column_by_name(field.name()) {
                            c.clone()
                        } else {
                            arrow::array::new_null_array(field.data_type(), b.num_rows())
                        };
                        cols.push(col);
                    }
                    match RecordBatch::try_new(table_schema.clone(), cols) {
                        Ok(aligned) => aligned,
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "flush: failed to align batch to table schema; writing as-is"
                            );
                            b
                        }
                    }
                }
            })
            .collect();

        let mut partitioned_batches = Vec::new();
        for batch in &aligned_batches {
            let sorted_batch = self.apply_sort_order(batch)?;
            let mut pb = spec.partition_batch(&sorted_batch)?;
            partitioned_batches.append(&mut pb);
        }

        // Group by partition_values so multiple batches going to the same partition are coalesced
        let mut grouped_partitions: Vec<(HashMap<String, Value>, Vec<RecordBatch>)> = Vec::new();
        for (pv, b) in partitioned_batches {
            if b.num_rows() == 0 {
                continue;
            }
            let mut placed = false;
            for (p_key, b_list) in grouped_partitions.iter_mut() {
                if p_key == &pv {
                    b_list.push(b.clone());
                    placed = true;
                    break;
                }
            }
            if !placed {
                grouped_partitions.push((pv, vec![b]));
            }
        }
        let coalesced_partitions: Vec<(HashMap<String, Value>, RecordBatch)> = {
            let mut res = Vec::new();
            for (pv, list) in grouped_partitions {
                if list.len() == 1 {
                    if let Some(first) = list.into_iter().next() {
                        res.push((pv, first));
                    }
                } else {
                    let schema = list[0].schema();
                    if let Ok(merged) = arrow::compute::concat_batches(&schema, &list) {
                        res.push((pv, merged));
                    } else {
                        for b in list {
                            res.push((pv.clone(), b));
                        }
                    }
                }
            }
            res
        };

        // Iceberg V3 row lineage: assign a contiguous `_row_id` block to each
        // data file, starting from `row_id_base`. Doing this after partitioning
        // keeps row IDs contiguous within a file, so `_row_id` equals
        // `first_row_id + row_position` as the spec requires.
        let mut next_row_id = row_id_base;
        let mut partitions_with_lineage: Vec<(HashMap<String, Value>, RecordBatch, Option<i64>)> =
            Vec::with_capacity(coalesced_partitions.len());
        for (pv, batch) in coalesced_partitions {
            if format_version >= 3 {
                let first_row_id = next_row_id;
                let batch = self.add_v3_metadata_columns(&batch, sequence_number, first_row_id)?;
                next_row_id += batch.num_rows() as i64;
                partitions_with_lineage.push((pv, batch, Some(first_row_id)));
            } else {
                partitions_with_lineage.push((pv, batch, None));
            }
        }

        // Extract local path from URI for writer
        let base_path = self.uri.strip_prefix("file://").unwrap_or(&self.uri);
        std::fs::create_dir_all(base_path)?;

        let mut all_new_entries = Vec::new();
        let mut files_to_upload: Vec<(String, String)> = Vec::new();
        // Index-build inputs are collected here and the builds are spawned
        // AFTER the manifest commit below. Spawning during the flush made the
        // background index-attach commit race the flush's own commit (both
        // wrote the next manifest version), causing a ~30ms retry per insert.
        let mut pending_builds: Vec<(
            crate::core::manifest::ManifestEntry,
            arrow::record_batch::RecordBatch,
            std::collections::HashMap<String, serde_json::Value>,
            String,
        )> = Vec::new();
        let index_cols = self.indexing.index_columns.read().clone();
        let index_all_flag = self.indexing.index_all;

        let index_configs_map: HashMap<String, crate::core::table::ColumnIndexConfig> =
            { self.indexing.index_configs.read().clone() };
        let default_device = self.indexing.default_device.read().clone();
        let default_device_for_stream = default_device.clone();

        // Parallelize partition writing using futures stream
        let concurrency = self
            .query_config
            .max_parallel_segments
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(16)
            })
            .min(64); // Cap to prevent resource exhaustion
        let stream = futures::stream::iter(partitions_with_lineage.into_iter().map(
            |(partition_values, batch, first_row_id)| {
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
                    writer_write.primary_key = self.primary_key.read().clone();
                    writer_write.set_store(self.store.clone());

                    let batch_inner = batch.clone();
                    let (mut entry, generated_files) = tokio::task::spawn_blocking(move || {
                        writer_write.write_batch(&batch_inner)?;
                        let entry = writer_write.to_manifest_entry();
                        let files = writer_write.get_generated_files();
                        Ok::<(crate::core::manifest::ManifestEntry, Vec<String>), anyhow::Error>((
                            entry, files,
                        ))
                    })
                    .await
                    .context("Flush task panicked")??;
                    // Iceberg V3 row lineage: record the first `_row_id` in this
                    // data file so readers can derive `_row_id` from position.
                    entry.first_row_id = first_row_id;

                    Ok::<PartitionSegment, anyhow::Error>((
                        entry,
                        generated_files,
                        segment_id,
                        batch,
                        partition_values,
                    ))
                }
            },
        ))
        .buffer_unordered(concurrency);

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

            for local_path in generated_files {
                let filename = local_path
                    .split('/')
                    .next_back()
                    .unwrap_or(&local_path)
                    .to_string();
                let remote_path = if hive_path.is_empty() {
                    filename
                } else {
                    format!("{}/{}", hive_path, filename)
                };
                files_to_upload.push((local_path, remote_path));
            }
            all_new_entries.push(entry.clone());

            // 2. Queue index building asynchronously (if needed). The build is
            // spawned AFTER the manifest commit below so the index-attach
            // commit does not race the flush's own commit (which caused a
            // ~30ms retry per insert).
            let has_pks = !self.primary_key.read().is_empty();
            if index_all_flag || !index_cols.is_empty() || has_pks {
                pending_builds.push((
                    entry.clone(),
                    batch.clone(),
                    partition_values.clone(),
                    segment_id.clone(),
                ));
            }
        }

        // WS2 crash boundary: data files are staged on disk but the manifest has
        // not been committed, so they are not yet referenced by any snapshot.
        crate::core::fault_injection::check(
            crate::core::fault_injection::CrashPoint::DataUpload,
        )?;

        // 3. Upload data files synchronously BEFORE committing manifest/metadata.
        // Invariant: A published manifest may reference only immutable artifacts that
        // have already been successfully uploaded and verified.
        if self.uri.contains("://") && !self.uri.starts_with("file://") {
            let store_clone = self.store.clone();
            for (local_path_str, remote_path_str) in files_to_upload {
                let local_path = std::path::Path::new(&local_path_str);
                let mut file = tokio::fs::File::open(&local_path).await.with_context(|| {
                    format!(
                        "Failed to open local staged file for upload: {}",
                        local_path_str
                    )
                })?;
                let remote_path = object_store::path::Path::from(remote_path_str.as_str());
                let mut upload =
                    store_clone
                        .put_multipart(&remote_path)
                        .await
                        .with_context(|| {
                            format!("Failed to initiate multipart upload to {}", remote_path_str)
                        })?;
                use tokio::io::AsyncReadExt;
                let mut buf = vec![0; 8 * 1024 * 1024]; // 8MB chunk buffer
                let mut total_uploaded = 0;
                loop {
                    let n = file.read(&mut buf).await.with_context(|| {
                        format!("Failed to read local staged file: {}", local_path_str)
                    })?;
                    if n == 0 {
                        break;
                    }
                    upload
                        .put_part(buf[..n].to_vec().into())
                        .await
                        .with_context(|| format!("Failed to upload part to {}", remote_path_str))?;
                    total_uploaded += n;
                }
                upload.complete().await.with_context(|| {
                    format!("Failed to complete multipart upload to {}", remote_path_str)
                })?;
                tracing::info!(
                    "Successfully uploaded staged file {} ({} bytes) to {}",
                    local_path_str,
                    total_uploaded,
                    remote_path_str
                );
                // Cleanup local staging file only after verified upload
                let _ = tokio::fs::remove_file(&local_path).await;
            }
        }

        // Detect if schema has evolved since last manifest load
        let manifest = manifest_manager
            .load_latest()
            .await
            .map(|(m, _)| m)
            .unwrap_or_default();
        let current_schema = self.arrow_schema();

        let should_update_schema = if let Some(last_schema) = manifest.schemas.last() {
            let latest_schema = last_schema.to_arrow();
            // Compare schemas, but ignore metadata if necessary.
            // Simple != check works for basic evolution.
            latest_schema != *current_schema
        } else {
            true
        };

        let (final_schemas, final_schema_id) = if should_update_schema {
            let mut new_schemas = manifest.schemas.clone();
            let new_id = if manifest.schemas.is_empty() {
                0
            } else {
                manifest.current_schema_id + 1
            };
            let mut manifest_schema =
                crate::core::manifest::Schema::from_arrow(&current_schema, new_id);
            // Persist the table's index configuration into the schema so it is
            // inherited by every future open — including other writer processes.
            // Without this, `add_index` on a fresh table (before the first
            // commit) lived only in memory and segments written by a later
            // process were silently unindexed.
            {
                let cfgs = self.indexing.index_configs.read();
                for field in &mut manifest_schema.fields {
                    if let Some(cfg) = cfgs.get(&field.name) {
                        if !cfg.algorithms.is_empty() {
                            field.indexes = cfg.algorithms.clone();
                        }
                    }
                }
            }
            new_schemas.push(manifest_schema);
            (Some(new_schemas), Some(new_id))
        } else {
            (None, None)
        };

        let (final_sort_orders, final_sort_order_id) = if let Some(order) = self.get_sort_order() {
            (Some(vec![order.clone()]), Some(order.order_id))
        } else {
            (None, None)
        };

        // Final commit for all data segments (with possible schema update)
        // Record the WAL transaction ids being committed so that recovery can
        // skip them if the process dies before the WAL is truncated (the
        // review's "manifest-before-WAL-truncation" case). Without this, WAL
        // replay would re-apply an already-committed batch and duplicate rows.
        let updated_properties = if wal_tx_ids.is_empty() {
            None
        } else {
            let joined = wal_tx_ids
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",");
            Some(HashMap::from([(
                "benostream.committed_wal_tx".to_string(),
                joined,
            )]))
        };

        let commit_metadata = crate::core::manifest::CommitMetadata {
            updated_schemas: final_schemas,
            updated_schema_id: final_schema_id,
            updated_partition_specs: None,
            updated_default_spec_id: None,
            updated_properties,
            removed_properties: None,
            updated_sort_orders: final_sort_orders,
            updated_default_sort_order_id: final_sort_order_id,
            updated_last_column_id: None,
            format_version: Some(format_version),
            is_fast_append: false,
            ..Default::default()
        };

        // Calculate total rows being added from all new manifest entries
        let added_rows_total: i64 = all_new_entries.iter().map(|e| e.record_count).sum();

        // WS2 crash boundary: immediately before the atomic publish point.
        crate::core::fault_injection::check(
            crate::core::fault_injection::CrashPoint::ManifestCommit,
        )?;
        let new_manifest = manifest_manager
            .commit(&all_new_entries, &[], commit_metadata)
            .await?;
        // WS2 crash boundary: the manifest is committed (visible) but the caller
        // has not yet observed success — a "delayed visibility" crash.
        crate::core::fault_injection::check(
            crate::core::fault_injection::CrashPoint::ManifestVisible,
        )?;

        // 3b. Spawn the queued index builds now that the manifest is committed.
        // Spawning them during the flush made the index-attach commit race the
        // flush's own commit (both wrote the next manifest version), causing a
        // ~30ms retry per insert.
        if !pending_builds.is_empty() {
            let index_cols_cap = index_cols.clone();
            let base_path_cap = base_path.to_string();
            let index_configs_cap = index_configs_map.clone();
            let default_device_cap = default_device.clone();
            let pk_cap = self.primary_key.read().clone();
            let table_store_cap = self.store.clone();
            let spec_cap = spec.clone();
            let memory_reclaimed_cap = self.memory_reclaimed.clone();
            let manifest_manager_cap = manifest_manager.clone();

            for (entry, batch, partition_values, segment_id) in pending_builds {
                let index_cols_c = index_cols_cap.clone();
                let base_path_c = base_path_cap.clone();
                let segment_id_c = segment_id.clone();
                let batch_c = batch.clone();
                let partition_values_c = partition_values.clone();
                let index_configs_c = index_configs_cap.clone();
                let default_device_c = default_device_cap.clone();
                let entry_c = entry.clone();
                let pk_c = pk_cap.clone();
                let table_store_c = table_store_cap.clone();
                let spec_c = spec_cap.clone();
                let memory_reclaimed_c = memory_reclaimed_cap.clone();
                let manifest_manager_c = manifest_manager_cap.clone();

                // Bound concurrent index builds (see `Table::index_build_gate`).
                let permit = self.acquire_index_build_permit().await?;

                let handle = tokio::spawn(async move {
                    let _permit = permit;
                    let hive_path = spec_c.partition_to_path(&partition_values_c);
                    let full_base_path = if hive_path.is_empty() {
                        base_path_c
                    } else {
                        format!("{}/{}", base_path_c, hive_path)
                    };
                    let parquet_path_rel = if hive_path.is_empty() {
                        format!("{}.parquet", segment_id_c)
                    } else {
                        format!("{}/{}.parquet", hive_path, segment_id_c)
                    };
                    let config_index = SegmentConfig::new(&full_base_path, &segment_id_c)
                        .with_index_all(index_all_flag)
                        .with_columns_to_index(index_cols_c)
                        .with_partition_values(partition_values_c.clone())
                        .with_column_devices(
                            index_configs_c
                                .iter()
                                .filter_map(|(c, cfg)| {
                                    cfg.device.as_ref().map(|d| (c.clone(), d.clone()))
                                })
                                .collect(),
                        )
                        .with_default_device(default_device_c)
                        .with_parquet_path(parquet_path_rel);

                    let index_res = tokio::spawn(async move {
                        let mut index_writer = HybridSegmentWriter::new(config_index)
                            .with_index_configs(index_configs_c);
                        index_writer.primary_key = pk_c;
                        index_writer.set_store(table_store_c);
                        index_writer.build_indexes(&batch_c, 0)?;
                        index_writer.finish_indexing().await?;
                        crate::core::fault_injection::check(
                            crate::core::fault_injection::CrashPoint::IndexUpload,
                        )?;
                        index_writer.upload_to_store().await?;
                        let files = index_writer.get_generated_files();
                        let updated_entry_info = index_writer.to_manifest_entry();
                        Ok::<(crate::core::manifest::ManifestEntry, Vec<String>), anyhow::Error>((
                            updated_entry_info,
                            files,
                        ))
                    })
                    .await;

                    match index_res {
                        Ok(Ok((updated_entry, _files))) => {
                            let mut merged_entry = entry_c;
                            let mut updated_index_files = updated_entry.index_files;
                            let hive_path = spec_c.partition_to_path(&partition_values_c);
                            if !hive_path.is_empty() {
                                for idx in &mut updated_index_files {
                                    idx.file_path = format!("{}/{}", hive_path, idx.file_path);
                                }
                            }
                            merged_entry.index_files = updated_index_files;
                            let commit_metadata = crate::core::manifest::CommitMetadata::default();
                            let file_path = merged_entry.file_path.clone();
                            let index_count = merged_entry.index_files.len();
                            let remove_paths = vec![merged_entry.file_path.clone()];
                            match manifest_manager_c
                                .commit(&[merged_entry], &remove_paths, commit_metadata)
                                .await
                            {
                                Ok(_) => tracing::info!(
                                    "Successfully attached {} indexes to manifest for segment {}",
                                    index_count,
                                    file_path
                                ),
                                Err(e) => tracing::error!(
                                    "Failed to attach indexes for segment {}: {}",
                                    file_path,
                                    e
                                ),
                            }
                        }
                        _ => {
                            tracing::error!(
                                "Index building failed for segment {}",
                                segment_id_c
                            );
                        }
                    }
                    memory_reclaimed_c.notify_waiters();
                });
                self.background_tasks.lock().await.push(handle);
            }
        }

        // 4. Update Table Metadata (Iceberg v2 Spec)
        // Determine the root for metadata. If we have a catalog, use its reported location.
        let meta_location = if let Some(catalog) = &self.catalog_state.catalog {
            if let (Some(ns), Some(t)) = (
                &self.catalog_state.namespace,
                &self.catalog_state.table_name,
            ) {
                catalog
                    .load_table(ns, t)
                    .await
                    .map(|m| m.location)
                    .unwrap_or_else(|_| self.uri.clone())
            } else {
                self.uri.clone()
            }
        } else {
            self.uri.clone()
        };

        let meta_store_arc = create_object_store(&meta_location)?;
        let meta_store = meta_store_arc.as_ref();

        let mut table_meta = match TableMetadata::load_latest(meta_store).await {
            Ok(mut meta) => {
                if should_update_schema {
                    if let Some(schema) = new_manifest.schemas.last() {
                        meta.add_schema(schema.clone());
                    }
                }
                meta.sort_orders = new_manifest.sort_orders.clone();
                meta.default_sort_order_id = new_manifest.default_sort_order_id;
                meta.format_version = format_version;
                meta
            }
            Err(_) => {
                // Initialize skeleton if not found
                TableMetadata::new(
                    format_version,
                    uuid::Uuid::new_v4().to_string(),
                    self.uri.clone(),
                    new_manifest.schemas.last().cloned().unwrap_or_else(|| {
                        crate::core::manifest::Schema::from_arrow(&current_schema, 0)
                    }),
                    new_manifest.partition_spec.clone(),
                    new_manifest
                        .sort_orders
                        .first()
                        .cloned()
                        .unwrap_or_default(),
                )
            }
        };

        // Add a new snapshot pointing to the latest manifest.
        //
        // Iceberg V3 row lineage: the snapshot's `first-row-id` is the lowest row
        // ID assigned in this snapshot, and `next-row-id` advances past every row
        // ID assigned so far.
        let snapshot_first_row_id = all_new_entries
            .iter()
            .filter_map(|e| e.first_row_id)
            .min()
            .or(table_meta.next_row_id);
        // The Iceberg spec requires the snapshot's `manifest-list` to be an
        // absolute path (external readers such as PyIceberg resolve a relative
        // path against the *process CWD*, not the table location, and fail).
        // The manifest manager stores it relative to the table root, so qualify
        // it with the table location here.
        let manifest_list_abs = match new_manifest.manifest_list_path.clone() {
            Some(p) if p.contains("://") || p.starts_with('/') => p,
            Some(p) => format!(
                "{}/{}",
                self.uri.trim_end_matches('/'),
                p.trim_start_matches('/')
            ),
            None => String::new(),
        };
        let snapshot = crate::core::metadata::Snapshot {
            snapshot_id: new_manifest.version as i64,
            parent_snapshot_id: table_meta.current_snapshot_id,
            timestamp_ms: new_manifest.timestamp_ms,
            sequence_number: Some(new_manifest.version as i64),
            summary: HashMap::from([("operation".to_string(), "append".to_string())]),
            manifest_list: manifest_list_abs,
            schema_id: Some(new_manifest.current_schema_id),
            first_row_id: snapshot_first_row_id,
            added_rows: Some(added_rows_total),
        };
        table_meta.add_snapshot(snapshot);

        if format_version >= 3 {
            let advanced = row_id_base + added_rows_total;
            table_meta.next_row_id = Some(table_meta.next_row_id.unwrap_or(0).max(advanced));
        }

        // Save metadata file (vX.metadata.json)
        let new_meta_version = (new_manifest.version) as i32; // Sync with manifest version for simplicity
                                                              // `save_to_store` returns the relative path it wrote. Capture it so we
                                                              // can hand catalogs the authoritative location instead of making them
                                                              // reconstruct it (AWS Glue cannot be told by its server, unlike REST).
        let written_metadata_path = table_meta
            .save_to_store(meta_store, new_meta_version)
            .await?;
        let metadata_location = format!(
            "{}/{}",
            self.uri.trim_end_matches('/'),
            written_metadata_path.trim_start_matches('/')
        );

        // 5. Commit to Catalog if configured (Iceberg Atomic Swap)
        if let Some(catalog) = &self.catalog_state.catalog {
            if let (Some(ns), Some(table)) = (
                &self.catalog_state.namespace,
                &self.catalog_state.table_name,
            ) {
                let snapshot = table_meta
                    .snapshots
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("No snapshot available in table metadata"))?;
                let updates = vec![
                    // Authoritative metadata location for catalogs that cannot
                    // return it themselves (Glue). Other catalogs ignore it.
                    serde_json::json!({
                        "action": "set-metadata-location",
                        "metadata-location": metadata_location
                    }),
                    serde_json::json!({
                        "action": "add-snapshot",
                        "snapshot": snapshot
                    }),
                    serde_json::json!({
                        "action": "set-current-snapshot",
                        "snapshot-id": table_meta.current_snapshot_id
                    }),
                ];
                catalog.commit_table(ns, table, updates).await?;
                tracing::info!(
                    "Committed snapshot {} to catalog {}.{}",
                    new_manifest.version,
                    ns,
                    table
                );
            }
        }

        // 5. Truncate WAL (Durability Checkpoint)
        {
            // WS2 crash boundary: manifest committed but WAL not yet truncated.
            // On reopen the WAL replay must be idempotent (no duplicate rows).
            crate::core::fault_injection::check(
                crate::core::fault_injection::CrashPoint::WalTruncate,
            )?;
            let mut wal = self.wal.lock().await;
            wal.truncate().context("Failed to truncate WAL")?;

            // 5b. Cleanup recovered files
            let mut recovered = self.recovered_wal_paths.lock();
            if !recovered.is_empty() {
                let paths: Vec<std::path::PathBuf> =
                    recovered.iter().map(std::path::PathBuf::from).collect();
                wal.cleanup_files(&paths).unwrap_or_default();
                recovered.clear();
            }
        }

        Ok(())
    }
}
