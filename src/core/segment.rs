// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::fs::File;

use crate::SegmentConfig;
use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*; // used in join
                       // allow wide imports to find standard items
use crate::core::index::gpu::{set_thread_gpu_context, ComputeContext};
use crate::core::manifest::{ColumnStats, ManifestEntry, ManifestValue, VectorStats};
use object_store::ObjectStore;
use parquet::file::statistics::Statistics as ParquetStats;
use std::collections::HashMap;
use std::sync::Arc;

pub struct HybridSegmentWriter {
    pub(crate) config: SegmentConfig,
    // Store paths of created files for upload tracking
    pub(crate) generated_files: parking_lot::Mutex<Vec<String>>,
    // Accumulated Stats
    pub(crate) stats: parking_lot::Mutex<HashMap<String, crate::core::manifest::ColumnStats>>,
    pub record_count: std::sync::atomic::AtomicUsize,
    pub store: Option<Arc<dyn ObjectStore>>,
    pub primary_key: Vec<String>,
    pub index_configs: HashMap<String, crate::core::table::ColumnIndexConfig>,
    // Add additive buffers for multi-batch indexing
    pub(crate) inverted_data:
        parking_lot::Mutex<HashMap<String, std::collections::BTreeMap<String, Vec<u32>>>>,
    pub index_metadata: parking_lot::Mutex<HashMap<String, String>>,
    pub(crate) file_checksum: parking_lot::Mutex<Option<String>>,
}

impl HybridSegmentWriter {
    pub fn new(config: SegmentConfig) -> Self {
        Self {
            config,
            generated_files: parking_lot::Mutex::new(Vec::new()),
            stats: parking_lot::Mutex::new(HashMap::new()),
            record_count: std::sync::atomic::AtomicUsize::new(0),
            store: None,
            primary_key: Vec::new(),
            index_configs: HashMap::new(),
            inverted_data: parking_lot::Mutex::new(HashMap::new()),
            index_metadata: parking_lot::Mutex::new(HashMap::new()),
            file_checksum: parking_lot::Mutex::new(None),
        }
    }

    pub fn with_index_configs(
        mut self,
        configs: HashMap<String, crate::core::table::ColumnIndexConfig>,
    ) -> Self {
        self.index_configs = configs;
        self
    }

    pub fn with_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_existing_stats(self, stats: HashMap<String, ColumnStats>) -> Self {
        *self.stats.lock() = stats;
        self
    }

    pub fn with_record_count(self, count: usize) -> Self {
        self.record_count
            .store(count, std::sync::atomic::Ordering::SeqCst);
        self
    }

    pub fn set_store(&mut self, store: Arc<dyn ObjectStore>) {
        self.store = Some(store);
    }

    pub fn get_generated_files(&self) -> Vec<String> {
        self.generated_files.lock().clone()
    }

    pub fn get_stats(&self) -> HashMap<String, ColumnStats> {
        self.stats.lock().clone()
    }

    pub fn get_record_count(&self) -> usize {
        self.record_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn to_manifest_entry(&self) -> ManifestEntry {
        let stats = self.get_stats();
        let record_count = self.get_record_count() as i64;
        let files = self.get_generated_files();

        let mut index_files = Vec::new();
        let mut parquet_file = String::new();
        let mut total_size = 0;

        for f in &files {
            let filename = f.split('/').next_back().unwrap_or(f).to_string();

            if filename.ends_with(".inv.parquet") {
                // Inverted Index
                let col = filename.split('.').nth(1).map(|c| c.to_string());
                index_files.push(crate::core::manifest::IndexFile {
                    file_path: filename.clone(),
                    index_type: "inverted".to_string(),
                    column_name: col,
                    blob_type: None,
                    offset: None,
                    length: None,
                });
            } else if filename.ends_with(".centroids.parquet")
                || filename.ends_with(".mapping.parquet")
            {
                // HNSW-IVF Auxiliary Parquet Files
            } else if filename.ends_with(".parquet") {
                // Main Data File
                parquet_file = filename.clone();
                if let Ok(meta) = std::fs::metadata(f) {
                    total_size = meta.len() as i64;
                }
            } else if filename.contains(".idx") {
                // Scalar Bitmap Index
                let col = filename.split('.').nth(1).and_then(|c| {
                    if c == "idx" {
                        None
                    } else {
                        Some(c.to_string())
                    }
                });
                index_files.push(crate::core::manifest::IndexFile {
                    file_path: filename.clone(),
                    index_type: "scalar".to_string(),
                    column_name: col,
                    blob_type: None,
                    offset: None,
                    length: None,
                });
            } else if filename.ends_with(".hnsw.graph") {
                // Vector Index
                let parts: Vec<&str> = filename.split('.').collect();
                if parts.len() >= 3 {
                    let col = parts[1].to_string();
                    let base_parts = &parts[..parts.len() - 2];

                    // Strip .cluster_X part to find the algo metadata
                    let mut base_path = base_parts.join(".");
                    if let Some(c_idx) = base_path.find(".cluster_") {
                        base_path = base_path[..c_idx].to_string();
                    }

                    let algo_name = self.index_metadata.lock().get(&base_path).cloned();

                    // The manifest file_path should be the unique base for THIS variant
                    let mut manifest_path_raw = base_parts.join(".");
                    if let Some(c_idx) = manifest_path_raw.find(".cluster_") {
                        manifest_path_raw = manifest_path_raw[..c_idx].to_string();
                    }

                    index_files.push(crate::core::manifest::IndexFile {
                        file_path: manifest_path_raw,
                        index_type: "vector".to_string(),
                        column_name: Some(col),
                        blob_type: algo_name,
                        offset: None,
                        length: None,
                    });
                }
            }
        }

        ManifestEntry {
            file_path: self.config.parquet_path.clone().unwrap_or(parquet_file),
            file_size_bytes: total_size,
            record_count,
            index_files,
            delete_files: self.config.delete_files.clone(),
            column_stats: stats,
            partition_values: self.config.partition_values.clone().into_iter().collect(),
            clustering_strategy: None,
            clustering_columns: None,
            min_clustering_score: None,
            max_clustering_score: None,
            normalization_mins: None,
            normalization_maxs: None,
            file_checksum: self.file_checksum.lock().clone(),
        }
    }

    /// Compute vector statistics (HyperStream exclusive) while delegating
    /// scalar statistics to the Parquet writer metadata (Zero-Copy).
    fn compute_vector_stats(&self, _batch: &RecordBatch) -> Result<HashMap<String, VectorStats>> {
        // Bypassing vector stats computation (dim_min, dim_max) to maximize bulk ingestion throughput.
        // HNSW builds its own bounding boxes, and Parquet provides scalar stats.
        Ok(HashMap::new())
    }

    fn merge_parquet_stats(
        &self,
        metadata: &parquet::file::metadata::ParquetMetaData,
        vector_stats_map: HashMap<String, VectorStats>,
    ) -> Result<()> {
        let mut final_stats = self.stats.lock();

        if let Some(rg) = metadata.row_groups().first() {
            for col in rg.columns() {
                let col_name = col.column_path().string();
                let mut col_stats = ColumnStats::default();

                if let Some(stats) = col.statistics() {
                    // Extract common statistics regardless of type
                    // In parquet 57.x, these methods have _opt suffix on the enum
                    col_stats.null_count = stats.null_count_opt().unwrap_or(0) as i64;
                    col_stats.distinct_count = stats.distinct_count_opt().map(|v| v as i64);

                    match stats {
                        ParquetStats::Int32(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Int32(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Int32(v));
                        }
                        ParquetStats::Int64(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Int64(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Int64(v));
                        }
                        ParquetStats::Float(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Float32(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Float32(v));
                        }
                        ParquetStats::Double(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Float64(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Float64(v));
                        }
                        ParquetStats::ByteArray(s) => {
                            if let (Some(min_val), Some(max_val)) = (s.min_opt(), s.max_opt()) {
                                col_stats.min = std::str::from_utf8(min_val.as_ref())
                                    .ok()
                                    .map(|s| ManifestValue::String(s.to_string()));
                                col_stats.max = std::str::from_utf8(max_val.as_ref())
                                    .ok()
                                    .map(|s| ManifestValue::String(s.to_string()));
                            }
                        }
                        ParquetStats::Boolean(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Boolean(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Boolean(v));
                        }
                        _ => {}
                    }
                }

                // Merge in HyperStream-specific vector stats if applicable
                if let Some(v_stats) = vector_stats_map.get(&col_name) {
                    col_stats.vector_stats = Some(v_stats.clone());
                }

                final_stats.insert(col_name, col_stats);
            }
        }
        Ok(())
    }

    /// Write a batch of data to a Parquet file (fast path, no index building).
    /// Index building should be done asynchronously via build_indexes_async().
    pub fn write_batch(&self, batch: &RecordBatch) -> Result<()> {
        let is_remote =
            self.config.base_path.contains("://") && !self.config.base_path.starts_with("file://");
        let (path, _local_staging_dir) = if is_remote {
            let temp_dir = std::env::temp_dir()
                .join("hyperstream_staging")
                .join(uuid::Uuid::new_v4().to_string());
            std::fs::create_dir_all(&temp_dir)?;
            let filename = format!("{}.parquet", self.config.segment_id);
            (temp_dir.join(&filename), Some(temp_dir))
        } else {
            let base = self
                .config
                .base_path
                .strip_prefix("file://")
                .unwrap_or(&self.config.base_path);
            let base_path = std::path::Path::new(base);
            if !base.is_empty() {
                std::fs::create_dir_all(base_path)
                    .context("Failed to create local segment directory")?;
            }
            let p = if base.is_empty() {
                format!("{}.parquet", self.config.segment_id)
            } else {
                format!("{}/{}.parquet", base, self.config.segment_id)
            };
            (std::path::PathBuf::from(p), None)
        };

        let tmp_path = format!("{}.tmp", path.to_str().context("Invalid UTF-8 in path")?);

        // Zero-Copy Stats: Calculate vector stats using Rayon
        let vec_stats = self.compute_vector_stats(batch)?;

        // Write Data (Parquet) to temporary file
        let file = File::create(&tmp_path).context("Failed to create temporary segment file")?;
        let mut props_builder = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
            .set_data_page_size_limit(8192); // 8KB pages for highly granular random access

        // Enable Bloom Filters for Primary Keys if defined
        for pk in &self.primary_key {
            props_builder = props_builder.set_column_bloom_filter_enabled(
                parquet::schema::types::ColumnPath::from(pk.clone()),
                true,
            );
        }

        let props = props_builder.build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;

        writer.write(batch)?;
        let metadata = writer.close()?; // Capture Zero-Copy metadata

        // Extract and Merge Parquet Stats
        self.merge_parquet_stats(&metadata, vec_stats)?;

        // Atomic rename
        std::fs::rename(&tmp_path, &path).context("Failed to atomically rename segment file")?;

        // Compute File Checksum
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;
        use std::io::Read;
        let mut f = File::open(&path)?;
        let mut buffer = [0; 65536];
        while let Ok(n) = f.read(&mut buffer) {
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        *self.file_checksum.lock() = Some(format!("{:x}", hasher.finalize()));

        {
            let mut files = self.generated_files.lock();
            files.push(path.to_str().context("Invalid UTF-8 in path")?.to_string());
        }

        tracing::info!(
            "Written data to {} ({} rows)",
            path.display(),
            batch.num_rows()
        );
        self.record_count
            .fetch_add(batch.num_rows(), std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Upload all generated files to ObjectStore if configured.
    /// Returns the final paths in the store.
    pub async fn upload_to_store(&self) -> Result<Vec<String>> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(self.get_generated_files()), // Local filesystem, already there
        };

        let files = self.get_generated_files();
        let mut final_paths = Vec::new();

        for local_path in files {
            if !local_path.contains("hyperstream_staging") {
                // Not a staged file, assume it's already in the right place (or local)
                final_paths.push(local_path);
                continue;
            }

            let filename = local_path
                .split('/')
                .next_back()
                .context("Missing filename")?;
            let remote_path = if self.config.base_path.contains("://") {
                let mut base = self.config.base_path.clone();
                if !base.ends_with('/') {
                    base.push('/');
                }
                format!("{}{}", base, filename)
            } else {
                format!("{}/{}", self.config.base_path, filename)
            };

            // Parse remote_path to object_store::path::Path
            // e.g. s3://bucket/data/seg.parquet -> data/seg.parquet
            let store_path = if remote_path.contains("://") {
                let url = url::Url::parse(&remote_path)?;
                object_store::path::Path::from(url.path().trim_start_matches('/'))
            } else {
                object_store::path::Path::from(remote_path.clone())
            };

            let data = std::fs::read(&local_path)?;
            store.put(&store_path, data.into()).await?;

            // Cleanup local staging file
            let _ = std::fs::remove_file(&local_path);

            final_paths.push(remote_path);
        }

        // Update generated_files with final paths
        {
            let mut g_files = self.generated_files.lock();
            *g_files = final_paths.clone();
        }

        Ok(final_paths)
    }

    /// Flush all buffered indexes (Inverted, HNSW, etc) to storage and track them.
    /// This should be called ONCE after all build_indexes() calls for a segment are complete.
    pub async fn finish_indexing(&self) -> Result<()> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No store configured for finishing indexes"))?;

        // Determine the directory prefix from the existing parquet path
        let parent_prefix = if let Some(p) = &self.config.parquet_path {
            if let Some(pos) = p.rfind('/') {
                &p[..pos + 1]
            } else {
                ""
            }
        } else {
            ""
        };

        // 1. Process Inverted Index Buffers - Drain the buffer into a local variable
        let inverted_data = {
            let mut inverted_lock = self.inverted_data.lock();
            std::mem::take(&mut *inverted_lock)
        }; // Guard is dropped here

        for (col_name, inverted_map) in inverted_data {
            tracing::info!(
                "Finishing Inverted Index for column '{}' ({} unique tokens)",
                col_name,
                inverted_map.len()
            );

            // Build Arrow Arrays for Parquet
            let mut key_builder = arrow::array::StringBuilder::new();
            let value_builder = arrow::array::UInt32Builder::new();
            let mut list_builder = arrow::array::ListBuilder::new(value_builder);

            for (key, mut row_ids) in inverted_map {
                key_builder.append_value(&key);
                row_ids.sort_unstable();
                let mut last_id = 0;
                for row_id in row_ids {
                    list_builder.values().append_value(row_id - last_id);
                    last_id = row_id;
                }
                list_builder.append(true);
            }

            let key_array = std::sync::Arc::new(key_builder.finish());
            let list_array = std::sync::Arc::new(list_builder.finish());

            let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new(
                    "row_ids",
                    arrow::datatypes::DataType::List(std::sync::Arc::new(
                        arrow::datatypes::Field::new(
                            "item",
                            arrow::datatypes::DataType::UInt32,
                            true,
                        ),
                    )),
                    false,
                ),
            ]));

            let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
            let filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name);
            let full_path_str = format!("{}{}", parent_prefix, filename);
            let target_path = object_store::path::Path::from(full_path_str.clone());

            // Write to memory buffer then to store
            let mut buffer = Vec::new();
            {
                let props = parquet::file::properties::WriterProperties::builder().build();
                let mut writer = ArrowWriter::try_new(&mut buffer, inv_schema, Some(props))?;
                writer.write(&inv_batch)?;
                writer.close()?;
            }

            store.put(&target_path, buffer.into()).await?;

            {
                let mut files = self.generated_files.lock();
                files.push(full_path_str.clone());
            }

            tracing::info!("  Inverted Index written to storage: {}", full_path_str);
        }

        Ok(())
    }

    /// Build indexes for a batch (can be called asynchronously after write_batch).
    /// This is the expensive operation that should run in background.
    pub fn build_indexes(&self, batch: &RecordBatch, row_offset: usize) -> Result<()> {
        tracing::info!(
            "Building indexes for batch of {} rows at offset {}",
            batch.num_rows(),
            row_offset
        );
        let schema = batch.schema();
        let _fields = schema.fields();

        // Build Indexes

        batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .try_for_each(|(i, field)| {
                let col_name = field.name();
                let col = batch.column(i);

                let is_pk = self.primary_key.contains(&col_name.to_string());
                let is_vector = matches!(
                    col.data_type(),
                    arrow::datatypes::DataType::FixedSizeList(_, _)
                        | arrow::datatypes::DataType::List(_)
                );
                let in_config_list = self
                    .config
                    .columns_to_index
                    .as_ref()
                    .map(|cols| cols.contains(&col_name.to_string()))
                    .unwrap_or(false);

                if self.config.index_all || is_pk || is_vector || in_config_list {
                    self.index_column(col_name, col, row_offset)
                } else {
                    Ok(())
                }
            })?;

        Ok(())
    }

    /// Build index for a single column.
    /// Can be called during ingestion OR for post-hoc backfilling.
    pub fn index_column(
        &self,
        col_name: &str,
        col_array: &std::sync::Arc<dyn Array>,
        row_offset: usize,
    ) -> Result<()> {
        // Apply per-column device override if specified
        if let Some(device_str) = self.config.column_devices.get(col_name) {
            tracing::info!(
                "Applying device override for column {}: {}",
                col_name,
                device_str
            );
            if let Ok(ctx) = ComputeContext::from_device_str(device_str) {
                tracing::info!(
                    "Successfully set global GPU context to {:?} for column {}",
                    ctx.backend,
                    col_name
                );
                set_thread_gpu_context(Some(ctx));
            } else {
                tracing::warn!("Failed to parse device string: {}", device_str);
            }
        } else if let Some(ref device_str) = self.config.default_device {
            tracing::info!(
                "Applying default device for column {}: {}",
                col_name,
                device_str
            );
            if let Ok(ctx) = ComputeContext::from_device_str(device_str) {
                tracing::info!(
                    "Successfully set global GPU context to {:?} for column {}",
                    ctx.backend,
                    col_name
                );
                set_thread_gpu_context(Some(ctx));
            } else {
                tracing::warn!("Failed to parse default device string: {}", device_str);
            }
        }

        // OPT-IN CHECK: Only index if configured, or if it's a Vector or Primary Key column
        let config = self.index_configs.get(col_name);
        let is_pk = self.primary_key.contains(&col_name.to_string());
        let is_vector = matches!(
            col_array.data_type(),
            arrow::datatypes::DataType::FixedSizeList(_, _) | arrow::datatypes::DataType::List(_)
        );
        let in_config_list = self
            .config
            .columns_to_index
            .as_ref()
            .map(|cols| cols.contains(&col_name.to_string()))
            .unwrap_or(false);

        if !is_pk
            && !is_vector
            && !self.config.index_all
            && !in_config_list
            && !config.map(|c| c.enabled).unwrap_or(false)
        {
            // Skip indexing for this column! (Massive speed gain for multi-column tables)
            return Ok(());
        }

        // Create a local staging directory if base_path is a URI
        let is_remote =
            self.config.base_path.contains("://") && !self.config.base_path.starts_with("file://");
        let local_staging_dir = if is_remote {
            let temp_dir = std::env::temp_dir()
                .join("hyperstream_staging")
                .join(uuid::Uuid::new_v4().to_string());
            std::fs::create_dir_all(&temp_dir)?;
            temp_dir
        } else {
            let path = self
                .config
                .base_path
                .strip_prefix("file://")
                .unwrap_or(&self.config.base_path);
            let p = std::path::PathBuf::from(path);
            if !path.is_empty() {
                std::fs::create_dir_all(&p)?;
            }
            p
        };

        match col_array.data_type() {
            arrow::datatypes::DataType::List(_)
            | arrow::datatypes::DataType::FixedSizeList(_, _) => {
                self.build_vector_index(col_name, col_array, row_offset, &local_staging_dir)?;
            }
            _ => {
                self.build_inverted_index(col_name, col_array, row_offset, &local_staging_dir)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListArray, Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_write_hybrid_segment() -> Result<()> {
        // 1. Setup Data: Int32 Column + Vector Column
        let dim = 4;
        let num_rows = 10;

        let id_array = Int32Array::from((0..num_rows).collect::<Vec<i32>>());

        // 10 vectors of dim 4
        let mut values = Vec::new();
        for i in 0..num_rows {
            for j in 0..dim {
                values.push((i + j) as f32);
            }
        }
        let values_array = Float32Array::from(values);
        let vectors_array = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(values_array),
            None,
        )?;

        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_array), Arc::new(vectors_array)],
        )?;

        // 2. Write Segment
        let tmp_dir = std::env::temp_dir();
        let config = SegmentConfig::new(
            tmp_dir.to_str().context("Invalid UTF-8 in path")?,
            "test_segment_001",
        )
        .with_index_all(true);
        let writer = HybridSegmentWriter::new(config.clone());

        writer.write_batch(&batch)?;

        // Build indexes (required for index files to be created)
        writer.build_indexes(&batch, 0)?;

        // 3. Verify Files
        let base = format!("{}/{}", config.base_path, config.segment_id);

        // Parquet
        assert!(
            std::path::Path::new(&format!("{}.parquet", base)).exists(),
            "Parquet file should exist"
        );

        // Inverted Index for id column (replaces old .idx format)
        assert!(
            std::path::Path::new(&format!("{}.id.inv.parquet", base)).exists(),
            "Inverted index for id should exist"
        );

        // Vector Index (embedding) - HNSW-IVF saves centroids and cluster graphs
        assert!(
            std::path::Path::new(&format!("{}.embedding.tq8.centroids.parquet", base)).exists(),
            "Vector index centroids should exist"
        );
        assert!(
            std::path::Path::new(&format!("{}.embedding.tq8.cluster_0.hnsw.graph", base)).exists(),
            "Vector index graph should exist"
        );

        Ok(())
    }
}
