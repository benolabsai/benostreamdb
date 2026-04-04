// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::fs::File;

use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use anyhow::{Context, Result};
use crate::SegmentConfig;
use arrow::array::Array;
use rayon::prelude::*; // used in join
// allow wide imports to find standard items
use std::collections::HashMap;
use std::sync::Arc;
use crate::core::manifest::{ColumnStats, ManifestEntry, VectorStats, ManifestValue};
use object_store::ObjectStore;
use crate::core::index::hnsw_ivf::HnswIvfIndex;
use crate::core::index::gpu::{ComputeContext, set_global_gpu_context};
use parquet::file::statistics::Statistics as ParquetStats;

pub struct HybridSegmentWriter {
    config: SegmentConfig,
    // Store paths of created files for upload tracking
    generated_files: std::sync::Mutex<Vec<String>>,
    // Accumulated Stats
    stats: std::sync::Mutex<HashMap<String, crate::core::manifest::ColumnStats>>,
    pub record_count: std::sync::atomic::AtomicUsize,
    pub store: Option<Arc<dyn ObjectStore>>,
    pub primary_key: Vec<String>,
    pub index_configs: HashMap<String, crate::core::table::ColumnIndexConfig>,
}

impl HybridSegmentWriter {
    pub fn new(config: SegmentConfig) -> Self {
        Self { 
            config,
            generated_files: std::sync::Mutex::new(Vec::new()), 
            stats: std::sync::Mutex::new(HashMap::new()),
            record_count: std::sync::atomic::AtomicUsize::new(0),
            store: None,
            primary_key: Vec::new(),
            index_configs: HashMap::new(),
        }
    }

    pub fn with_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn set_store(&mut self, store: Arc<dyn ObjectStore>) {
        self.store = Some(store);
    }

    pub fn get_generated_files(&self) -> Vec<String> {
        self.generated_files.lock().unwrap().clone()
    }

    pub fn get_stats(&self) -> HashMap<String, ColumnStats> {
        self.stats.lock().unwrap().clone()
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
            let filename = f.split('/').next_back().unwrap_or(&f).to_string();
            
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
            } else if filename.ends_with(".centroids.parquet") || filename.ends_with(".mapping.parquet") {
                // HNSW-IVF Auxiliary Parquet Files
            } else if filename.ends_with(".parquet") {
                // Main Data File
                parquet_file = filename.clone();
                if let Ok(meta) = std::fs::metadata(f) {
                    total_size = meta.len() as i64;
                }
            } else if filename.contains(".idx") {
                // Scalar Bitmap Index
                let col = filename.split('.').nth(1).and_then(|c| if c == "idx" { None } else { Some(c.to_string()) });
                index_files.push(crate::core::manifest::IndexFile {
                    file_path: filename.clone(),
                    index_type: "scalar".to_string(),
                    column_name: col,
                    blob_type: None,
                    offset: None,
                    length: None,
                });
            } else if filename.contains(".hnsw.") {
                  // Vector Index
                  let parts: Vec<&str> = filename.split('.').collect();
                  let col = parts.get(1).map(|s| s.to_string());
                  
                  if (filename.ends_with(".hnsw.graph") || filename.ends_with(".centroids.parquet"))
                      && !index_files.iter().any(|idx| idx.column_name == col && idx.index_type == "vector") {
                          index_files.push(crate::core::manifest::IndexFile {
                            file_path: filename.clone(),
                            index_type: "vector".to_string(),
                            column_name: col,
                            blob_type: None,
                            offset: None,
                            length: None,
                          });
                      }
            }
        }
        
        if !index_files.is_empty() {
            eprintln!("DEBUG: to_manifest_entry: segment_id={}, generated index_files={:?}", self.config.segment_id, index_files);
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
        }
    }

    /// Compute vector statistics (HyperStream exclusive) while delegating 
    /// scalar statistics to the Parquet writer metadata (Zero-Copy).
    fn compute_vector_stats(&self, batch: &RecordBatch) -> Result<HashMap<String, VectorStats>> {
        let schema = batch.schema();
        let mut vector_stats_map = HashMap::new();

        for (i, field) in schema.fields().iter().enumerate() {
            let col_name = field.name();
            let array = batch.column(i);
            
            // Vector Stats (HyperStream Extension)
            match field.data_type() {
                arrow::datatypes::DataType::FixedSizeList(_, _) | arrow::datatypes::DataType::List(_) => {
                    let mut min_norm = f32::MAX;
                    let mut max_norm = f32::MIN;
                    let mut sum_norm = 0.0;
                    let count = array.len();
                    
                    let mut dim_min: Option<Vec<f32>> = None;
                    let mut dim_max: Option<Vec<f32>> = None;

                    // Helper to get float vectors regardless of list type
                    let get_vector = |i: usize| -> Option<Vec<f32>> {
                        if array.is_null(i) { return None; }
                        let val = if let Some(arr) = array.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                            arr.value(i)
                        } else if let Some(arr) = array.as_any().downcast_ref::<arrow::array::ListArray>() {
                            arr.value(i)
                        } else {
                            return None;
                        };
                        let floats = val.as_any().downcast_ref::<arrow::array::Float32Array>()?;
                        Some(floats.values().to_vec())
                    };

                    // Parallelize vector stats calculation using Rayon
                    let zero = vec![0.0; if array.len() > 0 { get_vector(0).map(|v| v.len()).unwrap_or(0) } else { 0 }];

                    let vector_results: Vec<(f32, Vec<f32>)> = (0..count)
                        .into_par_iter()
                        .filter_map(|i| {
                            get_vector(i).map(|v| {
                                // Norm is distance from zero vector
                                let norm = crate::core::index::distance::l2_distance(&v, &zero);
                                (norm, v)
                            })
                        })
                        .collect();

                    for (norm, v) in vector_results {
                        min_norm = min_norm.min(norm);
                        max_norm = max_norm.max(norm);
                        sum_norm += norm;

                        if dim_min.is_none() {
                            dim_min = Some(v.clone());
                            dim_max = Some(v.clone());
                        } else {
                            let d_min = dim_min.as_mut().unwrap();
                            let d_max = dim_max.as_mut().unwrap();
                            for (j, &val) in v.iter().enumerate() {
                                if j < d_min.len() {
                                    d_min[j] = d_min[j].min(val);
                                    d_max[j] = d_max[j].max(val);
                                }
                            }
                        }
                    }

                    if count > 0 && min_norm != f32::MAX {
                         if let Some(dim_m) = dim_min {
                             vector_stats_map.insert(col_name.to_string(), VectorStats {
                                 min_norm,
                                 max_norm,
                                 mean_norm: sum_norm / count as f32,
                                 dim_min: Some(dim_m),
                                 dim_max,
                             });
                        }
                    }
                },
                _ => {}
            };
        }
        Ok(vector_stats_map)
    }

    fn merge_parquet_stats(&self, metadata: &parquet::file::metadata::ParquetMetaData, vector_stats_map: HashMap<String, VectorStats>) -> Result<()> {
        let mut final_stats = self.stats.lock().unwrap();
        
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
                        },
                        ParquetStats::Int64(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Int64(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Int64(v));
                        },
                        ParquetStats::Float(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Float32(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Float32(v));
                        },
                        ParquetStats::Double(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Float64(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Float64(v));
                        },
                        ParquetStats::ByteArray(s) => {
                             if let (Some(min_val), Some(max_val)) = (s.min_opt(), s.max_opt()) {
                                 col_stats.min = std::str::from_utf8(min_val.as_ref()).ok().map(|s| ManifestValue::String(s.to_string()));
                                 col_stats.max = std::str::from_utf8(max_val.as_ref()).ok().map(|s| ManifestValue::String(s.to_string()));
                             }
                        },
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
        let is_remote = self.config.base_path.contains("://") && !self.config.base_path.starts_with("file://");
        let (path, _local_staging_dir) = if is_remote {
            let temp_dir = std::env::temp_dir()
                .join("hyperstream_staging")
                .join(uuid::Uuid::new_v4().to_string());
            std::fs::create_dir_all(&temp_dir)?;
            let filename = format!("{}.parquet", self.config.segment_id);
            (temp_dir.join(&filename), Some(temp_dir))
        } else {
            let base = self.config.base_path.strip_prefix("file://").unwrap_or(&self.config.base_path);
            let base_path = std::path::Path::new(base);
            if !base.is_empty() {
                std::fs::create_dir_all(base_path).context("Failed to create local segment directory")?;
            }
            let p = if base.is_empty() {
                 format!("{}.parquet", self.config.segment_id)
            } else {
                 format!("{}/{}.parquet", base, self.config.segment_id)
            };
            (std::path::PathBuf::from(p), None)
        };
        
        let tmp_path = format!("{}.tmp", path.to_str().unwrap());

        // Zero-Copy Stats: Calculate vector stats using Rayon
        let vec_stats = self.compute_vector_stats(batch)?;
        
        // Write Data (Parquet) to temporary file
        let file = File::create(&tmp_path).context("Failed to create temporary segment file")?;
        let mut props_builder = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::try_new(3)?))
            .set_dictionary_enabled(true)
            .set_data_page_size_limit(1024 * 1024); // 1MB pages for better random access
        
        // Enable Bloom Filters for Primary Keys if defined
        for pk in &self.primary_key {
            props_builder = props_builder.set_column_bloom_filter_enabled(parquet::schema::types::ColumnPath::from(pk.clone()), true);
        }

        let props = props_builder.build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(batch)?;
        let metadata = writer.close()?; // Capture Zero-Copy metadata
        
        // Extract and Merge Parquet Stats
        self.merge_parquet_stats(&metadata, vec_stats)?;

        // Atomic rename
        std::fs::rename(&tmp_path, &path).context("Failed to atomically rename segment file")?;

        {
            let mut files = self.generated_files.lock().unwrap();
            files.push(path.to_str().unwrap().to_string());
        }

        println!("Written data to {} ({} rows)", path.display(), batch.num_rows());
        self.record_count.fetch_add(batch.num_rows(), std::sync::atomic::Ordering::Relaxed);
        
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

            let filename = local_path.split('/').next_back().unwrap();
            let remote_path = if self.config.base_path.contains("://") {
                 let mut base = self.config.base_path.clone();
                 if !base.ends_with('/') { base.push('/'); }
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
            let mut g_files = self.generated_files.lock().unwrap();
            *g_files = final_paths.clone();
        }

        Ok(final_paths)
    }

    /// Build indexes for a batch (can be called asynchronously after write_batch).
    /// This is the expensive operation that should run in background.
    pub fn build_indexes(&self, batch: &RecordBatch) -> Result<()> {
        println!("Building indexes for batch of {} rows", batch.num_rows());
        let schema = batch.schema();
        let _fields = schema.fields();

        // Build Indexes

        batch.schema().fields().iter().enumerate().collect::<Vec<_>>().into_par_iter()
            .try_for_each(|(i, field)| {
                let col_name = field.name();
                let col = batch.column(i);
                
                let is_pk = self.primary_key.contains(&col_name.to_string());
                let is_vector = matches!(col.data_type(), arrow::datatypes::DataType::FixedSizeList(_, _) | arrow::datatypes::DataType::List(_));
                let in_config_list = self.config.columns_to_index.as_ref().map(|cols| cols.contains(&col_name.to_string())).unwrap_or(false);
                
                if self.config.index_all || is_pk || is_vector || in_config_list {
                    self.index_column(col_name, col)
                } else {
                    Ok(())
                }
            })?;
        
        Ok(())
    }

    /// Build index for a single column. 
    /// Can be called during ingestion OR for post-hoc backfilling.
    pub fn index_column(&self, col_name: &str, col_array: &std::sync::Arc<dyn Array>) -> Result<()> {
        // Apply per-column device override if specified
        if let Some(device_str) = self.config.column_devices.get(col_name) {
            println!("Applying device override for column {}: {}", col_name, device_str);
            if let Ok(ctx) = ComputeContext::from_device_str(device_str) {
                println!("Successfully set global GPU context to {:?} for column {}", ctx.backend, col_name);
                set_global_gpu_context(Some(ctx));
            } else {
                println!("Failed to parse device string: {}", device_str);
            }
        } else if let Some(ref device_str) = self.config.default_device {
            println!("Applying default device for column {}: {}", col_name, device_str);
            if let Ok(ctx) = ComputeContext::from_device_str(device_str) {
                println!("Successfully set global GPU context to {:?} for column {}", ctx.backend, col_name);
                set_global_gpu_context(Some(ctx));
            } else {
                println!("Failed to parse default device string: {}", device_str);
            }
        }

        // OPT-IN CHECK: Only index if configured, or if it's a Vector or Primary Key column
        let config = self.index_configs.get(col_name);
        let is_pk = self.primary_key.contains(&col_name.to_string());
        let is_vector = matches!(col_array.data_type(), arrow::datatypes::DataType::FixedSizeList(_, _) | arrow::datatypes::DataType::List(_));
        let in_config_list = self.config.columns_to_index.as_ref().map(|cols| cols.contains(&col_name.to_string())).unwrap_or(false);
        
        if !is_pk && !is_vector && !self.config.index_all && !in_config_list && (config.is_none() || !config.unwrap().enabled) {
            // Skip indexing for this column! (Massive speed gain for multi-column tables)
            return Ok(());
        }

        // Create a local staging directory if base_path is a URI
        let is_remote = self.config.base_path.contains("://") && !self.config.base_path.starts_with("file://");
        let local_staging_dir = if is_remote {
            let temp_dir = std::env::temp_dir()
                .join("hyperstream_staging")
                .join(uuid::Uuid::new_v4().to_string());
            std::fs::create_dir_all(&temp_dir)?;
            temp_dir
        } else {
             let path = self.config.base_path.strip_prefix("file://").unwrap_or(&self.config.base_path);
             let p = std::path::PathBuf::from(path);
             if !path.is_empty() {
                 std::fs::create_dir_all(&p)?;
             }
             p
        };

        match col_array.data_type() {
                // Scalar Indexing (RoaringBitmap for Int32)
                arrow::datatypes::DataType::Int32 => {
                    println!("Indexing Int32 column: {}", col_name);
                    let _array = col_array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                    
                    // Scalar index (.idx) is removed for Int32 as we use more precise Inverted Index

                    // Optimized Inverted Index (Sort-based instead of HashMap)
                    let array = col_array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                    
                    // 1. Get sort indices to group same values together
                    let sort_indices = arrow::compute::sort_to_indices(array, None, None)?;
                    
                    let mut key_builder = arrow::array::Int32Builder::new();
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                    let mut current_val: Option<i32> = None;
                    let mut current_rows = Vec::new();

                    for i in 0..sort_indices.len() {
                        let row_idx = sort_indices.value(i) as u32;
                        if array.is_null(row_idx as usize) { continue; }
                        let val = array.value(row_idx as usize);

                        if Some(val) != current_val {
                            if let Some(v) = current_val {
                                key_builder.append_value(v);
                                current_rows.sort_unstable();
                                let mut last = 0;
                                for &rid in &current_rows {
                                    list_builder.values().append_value(rid - last);
                                    last = rid;
                                }
                                list_builder.append(true);
                                current_rows.clear();
                            }
                            current_val = Some(val);
                        }
                        current_rows.push(row_idx);
                    }
                    if let Some(v) = current_val {
                        key_builder.append_value(v);
                        current_rows.sort_unstable();
                        let mut last = 0;
                        for &rid in &current_rows {
                            list_builder.values().append_value(rid - last);
                            last = rid;
                        }
                        list_builder.append(true);
                    }

                    let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Int32, false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(
                        inv_schema.clone(),
                        vec![std::sync::Arc::new(key_builder.finish()), std::sync::Arc::new(list_builder.finish())]
                    )?;

                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name);
                    let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                     {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                },

                arrow::datatypes::DataType::List(inner) | arrow::datatypes::DataType::FixedSizeList(inner, _) => {
                     if *inner.data_type() == arrow::datatypes::DataType::Float32 {
                        println!("Indexing Vector column: {} (type={:?})", col_name, col_array.data_type());
                        
                        let vectors: Vec<Vec<f32>> = match col_array.data_type() {
                            arrow::datatypes::DataType::FixedSizeList(_, _) => {
                                let list_array = col_array.as_any().downcast_ref::<arrow::array::FixedSizeListArray>().unwrap();
                                (0..list_array.len())
                                    .into_par_iter()
                                    .map(|i| {
                                        let item = list_array.value(i);
                                        let float_array = item.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
                                        float_array.values().to_vec()
                                    })
                                    .collect()
                            },
                            arrow::datatypes::DataType::List(_) => {
                                let list_array = col_array.as_any().downcast_ref::<arrow::array::ListArray>().unwrap();
                                (0..list_array.len())
                                    .into_par_iter()
                                    .map(|i| {
                                        let item = list_array.value(i);
                                        let float_array = item.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
                                        float_array.values().to_vec()
                                    })
                                    .collect()
                            },
                            _ => unreachable!(),
                        };

                        if vectors.is_empty() { return Ok(()); }
                        let _dim = vectors[0].len();

                        // Build vector index ONLY if configured for immediate indexing
                        let in_config = self.config.columns_to_index.as_ref().map(|cols| cols.iter().any(|c| c == col_name)).unwrap_or(false);
                        if self.config.index_all || in_config {
                            let use_pq = vectors.len() > 5_000;
                            if use_pq {
                                println!("Auto-enabling PQ for segment ({} vectors)", vectors.len());
                            }
                            println!("Building HNSW-IVF index (blocking): {} vectors, {} dims, use_pq={}", vectors.len(), _dim, use_pq);
                            let hnsw_ivf_index = HnswIvfIndex::build(vectors, crate::core::index::VectorMetric::L2, None, None, use_pq)
                                .map_err(|e| anyhow::anyhow!("HNSW-IVF build failed: {}", e))?;
                            
                            let local_base_path = local_staging_dir.join(format!("{}.{}", self.config.segment_id, col_name));
                            let saved_files = hnsw_ivf_index.save(local_base_path.to_str().unwrap())
                                .map_err(|e| anyhow::anyhow!("HNSW-IVF save failed: {}", e))?;
                            
                            {
                                let mut files = self.generated_files.lock().unwrap();
                                files.extend(saved_files);
                            }
                        } else {
                            println!("Skipping vector indexing for column {} (delayed/background mode)", col_name);
                        }
                     }
                },
                arrow::datatypes::DataType::Int64 => {
                    println!("Indexing Int64 column: {}", col_name);
                    let _array = col_array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                    
                    // No mock .idx for Int64

                    // Optimized Inverted Index (Sort-based)
                    let array = col_array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                    let sort_indices = arrow::compute::sort_to_indices(array, None, None)?;

                    let mut key_builder = arrow::array::Int64Builder::new();
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                    let mut current_val: Option<i64> = None;
                    let mut current_rows = Vec::new();

                    for i in 0..sort_indices.len() {
                        let row_idx = sort_indices.value(i) as u32;
                        if array.is_null(row_idx as usize) { continue; }
                        let val = array.value(row_idx as usize);

                        if Some(val) != current_val {
                            if let Some(v) = current_val {
                                key_builder.append_value(v);
                                current_rows.sort_unstable();
                                let mut last = 0;
                                for &rid in &current_rows {
                                    list_builder.values().append_value(rid - last);
                                    last = rid;
                                }
                                list_builder.append(true);
                                current_rows.clear();
                            }
                            current_val = Some(val);
                        }
                        current_rows.push(row_idx);
                    }
                    if let Some(v) = current_val {
                        key_builder.append_value(v);
                        current_rows.sort_unstable();
                        let mut last = 0;
                        for &rid in &current_rows {
                            list_builder.values().append_value(rid - last);
                            last = rid;
                        }
                        list_builder.append(true);
                    }

                    let key_array = std::sync::Arc::new(key_builder.finish());
                    let list_array = std::sync::Arc::new(list_builder.finish());

                    let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Int64, false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name);
                    let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                },

                arrow::datatypes::DataType::Float64 => {
                    println!("Indexing Float64 column: {}", col_name);
                    let _array = col_array.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                    
                    // No mock .idx for Float64

                    // Optimized Inverted Index (Sort-based)
                    let array = col_array.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                    let sort_indices = arrow::compute::sort_to_indices(array, None, None)?;

                    let mut key_builder = arrow::array::Float64Builder::new();
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                    let mut current_val: Option<u64> = None; // Store as bits for comparison
                    let mut current_rows = Vec::new();

                    for i in 0..sort_indices.len() {
                        let row_idx = sort_indices.value(i) as u32;
                        if array.is_null(row_idx as usize) { continue; }
                        let val = array.value(row_idx as usize);
                        let val_bits = val.to_bits();

                        if Some(val_bits) != current_val {
                            if let Some(v_bits) = current_val {
                                key_builder.append_value(f64::from_bits(v_bits));
                                current_rows.sort_unstable();
                                let mut last = 0;
                                for &rid in &current_rows {
                                    list_builder.values().append_value(rid - last);
                                    last = rid;
                                }
                                list_builder.append(true);
                                current_rows.clear();
                            }
                            current_val = Some(val_bits);
                        }
                        current_rows.push(row_idx);
                    }
                    if let Some(v_bits) = current_val {
                        key_builder.append_value(f64::from_bits(v_bits));
                        current_rows.sort_unstable();
                        let mut last = 0;
                        for &rid in &current_rows {
                            list_builder.values().append_value(rid - last);
                            last = rid;
                        }
                        list_builder.append(true);
                    }

                    let key_array = std::sync::Arc::new(key_builder.finish());
                    let list_array = std::sync::Arc::new(list_builder.finish());

                    let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Float64, false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name);
                    let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                },

                arrow::datatypes::DataType::Float32 => {
                    println!("Indexing Float32 column: {}", col_name);
                    let array = col_array.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
                    
                    // No mock .idx for Float32

                    // Inverted Index
                    let mut inverted_map: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
                    for (row_i, val) in array.iter().enumerate() {
                        if let Some(v) = val {
                            inverted_map.entry(v.to_bits()).or_default().push(row_i as u32);
                        }
                    }

                    let mut key_builder = arrow::array::Float32Builder::new();
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                    for (key_bits,mut row_ids) in inverted_map {
                        key_builder.append_value(f32::from_bits(key_bits));
                        row_ids.sort_unstable();
                        let mut last_id = 0;
                        for row_id in row_ids {
                            list_builder.values().append_value(row_id - last_id);
                            last_id = row_id;
                        }
                        list_builder.append(true);
                    }

                    let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Float32, false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![
                        std::sync::Arc::new(key_builder.finish()),
                        std::sync::Arc::new(list_builder.finish())
                    ])?;
                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name);
                    let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                },
                
                // String/Utf8 Inverted Index - for category/tag filtering
                arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8 => {
                    println!("Indexing String column: {}", col_name);
                    
                    // Unified handling: cast to Utf8 to reuse existing logic
                    let casted_array = arrow::compute::cast(col_array, &arrow::datatypes::DataType::Utf8)
                        .context("Failed to cast column to Utf8 for indexing")?;
                    let array = casted_array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                    
                    // Fetch tokenizer if configured
                    let tokenizer_name = config.and_then(|c| c.tokenizer.clone()).unwrap_or_else(|| "identity".to_string());
                    let tokenizer = crate::core::index::tokenizer::GLOBAL_TOKENIZER_REGISTRY.get(&tokenizer_name)
                        .unwrap_or_else(|| crate::core::index::tokenizer::GLOBAL_TOKENIZER_REGISTRY.get("identity").unwrap());

                    // Build inverted index: Token -> RowIDs
                    let mut inverted_map: std::collections::HashMap<String, Vec<u32>> = std::collections::HashMap::new();
                    for (row_i, val) in array.iter().enumerate() {
                        if let Some(v) = val {
                            let tokens = tokenizer.tokenize(v);
                            for token in tokens {
                                inverted_map.entry(token).or_default().push(row_i as u32);
                            }
                        }
                    }
                    
                    println!("  Found {} unique values", inverted_map.len());
                    
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
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name);
                    let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                    
                    println!("String Inverted Index written to {}", inv_path.to_str().unwrap());
                },
                
                // Date32 Inverted Index - for date equality/range filtering
                // Date32 = days since Unix epoch (1970-01-01)
                arrow::datatypes::DataType::Date32 => {
                    println!("Indexing Date32 column: {}", col_name);
                    let array = col_array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap();
                    
                    // Build inverted index: Date -> RowIDs
                    let mut inverted_map: std::collections::HashMap<i32, Vec<u32>> = std::collections::HashMap::new();
                    for (row_i, val) in array.iter().enumerate() {
                        if let Some(v) = val {
                            inverted_map.entry(v).or_default().push(row_i as u32);
                        }
                    }
                    
                    println!("  Found {} unique dates", inverted_map.len());
                    
                    // Build Arrow Arrays for Parquet
                    let mut key_builder = arrow::array::Date32Builder::new();
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                    for (key, row_ids) in inverted_map {
                        key_builder.append_value(key);
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
                        arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Date32, false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name); let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                    
                    println!("Date32 Inverted Index written to {}", inv_path.to_str().unwrap());
                },
                
                // Timestamp Inverted Index - truncate to day for practical indexing
                // High-cardinality timestamps are truncated to day granularity
                arrow::datatypes::DataType::Timestamp(_, _) => {
                    println!("Indexing Timestamp column: {} (truncated to day)", col_name);
                    
                    // Truncate timestamps to day granularity for indexing
                    // This makes the inverted index practical (365 keys/year vs millions)
                    let mut inverted_map: std::collections::HashMap<i32, Vec<u32>> = std::collections::HashMap::new();
                    
                    // Handle different timestamp units
                    let array = col_array.as_any();
                    for row_i in 0..col_array.len() {
                        if col_array.is_null(row_i) {
                            continue;
                        }
                        
                        // Convert timestamp to days since epoch
                        let day = if let Some(arr) = array.downcast_ref::<arrow::array::TimestampSecondArray>() {
                            (arr.value(row_i) / 86_400) as i32
                        } else if let Some(arr) = array.downcast_ref::<arrow::array::TimestampMillisecondArray>() {
                            (arr.value(row_i) / 86_400_000) as i32
                        } else if let Some(arr) = array.downcast_ref::<arrow::array::TimestampMicrosecondArray>() {
                            (arr.value(row_i) / 86_400_000_000) as i32
                        } else if let Some(arr) = array.downcast_ref::<arrow::array::TimestampNanosecondArray>() {
                            (arr.value(row_i) / 86_400_000_000_000) as i32
                        } else {
                            continue;
                        };
                        
                        inverted_map.entry(day).or_default().push(row_i as u32);
                    }
                    
                    println!("  Found {} unique days", inverted_map.len());
                    
                    // Build Arrow Arrays (store as Date32 for the index key)
                    let mut key_builder = arrow::array::Date32Builder::new();
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                    for (key, row_ids) in inverted_map {
                        key_builder.append_value(key);
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
                        arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Date32, false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name); let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                    
                    println!("Timestamp Inverted Index (day granularity) written to {}", inv_path.to_str().unwrap());
                },
                
                // Keep default
                arrow::datatypes::DataType::Boolean => {
                    println!("Indexing Boolean column: {} (native boolean index)", col_name);
                     // Build inverted index: Boolean -> RowIDs (true/false as native booleans)
                    let mut inverted_map: std::collections::HashMap<bool, Vec<u32>> = std::collections::HashMap::new();

                    let array = col_array.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                    for row_i in 0..array.len() {
                        if array.is_null(row_i) {
                            continue;
                        }
                        let val = array.value(row_i);
                        inverted_map.entry(val).or_default().push(row_i as u32);
                    }
                    
                    // Build Arrow Arrays (store as Boolean for index key)
                    let mut key_builder = arrow::array::BooleanBuilder::new();
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                    for (key, row_ids) in inverted_map {
                        key_builder.append_value(key);
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
                        arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Boolean, false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    let inv_filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name); let inv_path = local_staging_dir.join(&inv_filename);
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                     {
                        let mut files = self.generated_files.lock().unwrap();
                        files.push(inv_path.to_str().unwrap().to_string());
                    }
                    println!("Boolean Inverted Index written to {}", inv_path.to_str().unwrap());
                },

                // Time32 (s/ms) -> Int32 keys
                arrow::datatypes::DataType::Time32(unit) => {
                    println!("Indexing Time32 column: {}", col_name);
                    let mut inverted_map: std::collections::HashMap<i32, Vec<u32>> = std::collections::HashMap::new();
                    
                    match unit {
                        arrow::datatypes::TimeUnit::Second => {
                            if let Some(array) = col_array.as_any().downcast_ref::<arrow::array::Time32SecondArray>() {
                                 for (row_i, val) in array.iter().enumerate() {
                                    if let Some(v) = val {
                                        inverted_map.entry(v).or_default().push(row_i as u32);
                                    }
                                }
                            }
                        },
                        arrow::datatypes::TimeUnit::Millisecond => {
                             if let Some(array) = col_array.as_any().downcast_ref::<arrow::array::Time32MillisecondArray>() {
                                 for (row_i, val) in array.iter().enumerate() {
                                    if let Some(v) = val {
                                        inverted_map.entry(v).or_default().push(row_i as u32);
                                    }
                                }
                            }
                        },
                        _ => {}
                    }
                    
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);
                    
                    // Sort keys
                    let mut keys: Vec<i32> = inverted_map.keys().cloned().collect();
                    keys.sort();
                    
                    // Build row_ids list
                    for key in &keys {
                        if let Some(row_ids) = inverted_map.get(key) {
                            let mut last_id = 0;
                            for row_id in row_ids {
                                list_builder.values().append_value(*row_id - last_id);
                                last_id = *row_id;
                            }
                            list_builder.append(true);
                        }
                    }
                    let list_array = std::sync::Arc::new(list_builder.finish());

                    // Build Key Array
                    let key_array: arrow::array::ArrayRef = match unit {
                        arrow::datatypes::TimeUnit::Second => {
                             let mut builder = arrow::array::Time32SecondBuilder::with_capacity(inverted_map.len());
                             for key in &keys { builder.append_value(*key); }
                             std::sync::Arc::new(builder.finish())
                        },
                        arrow::datatypes::TimeUnit::Millisecond => {
                             let mut builder = arrow::array::Time32MillisecondBuilder::with_capacity(inverted_map.len());
                             for key in &keys { builder.append_value(*key); }
                             std::sync::Arc::new(builder.finish())
                        },
                        _ => unreachable!("Invalid Time32 unit"),
                    };

                    // Use original data type for key field to preserve logical type
                    let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new("key", col_array.data_type().clone(), false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    self.write_inverted_index(col_name, inv_schema, inv_batch)?;
                },

                // Time64 (us/ns) -> Int64 keys
                arrow::datatypes::DataType::Time64(unit) => {
                    println!("Indexing Time64 column: {}", col_name);
                    let mut inverted_map: std::collections::HashMap<i64, Vec<u32>> = std::collections::HashMap::new();
                    
                    match unit {
                        arrow::datatypes::TimeUnit::Microsecond => {
                            if let Some(array) = col_array.as_any().downcast_ref::<arrow::array::Time64MicrosecondArray>() {
                                 for (row_i, val) in array.iter().enumerate() {
                                    if let Some(v) = val {
                                        inverted_map.entry(v).or_default().push(row_i as u32);
                                    }
                                }
                            }
                        },
                        arrow::datatypes::TimeUnit::Nanosecond => {
                             if let Some(array) = col_array.as_any().downcast_ref::<arrow::array::Time64NanosecondArray>() {
                                 for (row_i, val) in array.iter().enumerate() {
                                    if let Some(v) = val {
                                        inverted_map.entry(v).or_default().push(row_i as u32);
                                    }
                                }
                            }
                        },
                        _ => {}
                    }
                    
                    let value_builder = arrow::array::UInt32Builder::new();
                    let mut list_builder = arrow::array::ListBuilder::new(value_builder);
                    
                    // Sort keys
                    let mut keys: Vec<i64> = inverted_map.keys().cloned().collect();
                    keys.sort();
                    
                    // Build row_ids list
                    for key in &keys {
                        if let Some(row_ids) = inverted_map.get(key) {
                            let mut last_id = 0;
                            for row_id in row_ids {
                                list_builder.values().append_value(*row_id - last_id);
                                last_id = *row_id;
                            }
                            list_builder.append(true);
                        }
                    }
                    let list_array = std::sync::Arc::new(list_builder.finish());

                    // Build Key Array
                    let key_array: arrow::array::ArrayRef = match unit {
                        arrow::datatypes::TimeUnit::Microsecond => {
                             let mut builder = arrow::array::Time64MicrosecondBuilder::with_capacity(inverted_map.len());
                             for key in &keys { builder.append_value(*key); }
                             std::sync::Arc::new(builder.finish())
                        },
                        arrow::datatypes::TimeUnit::Nanosecond => {
                             let mut builder = arrow::array::Time64NanosecondBuilder::with_capacity(inverted_map.len());
                             for key in &keys { builder.append_value(*key); }
                             std::sync::Arc::new(builder.finish())
                        },
                        _ => unreachable!("Invalid Time64 unit"),
                    };
                    
                    // Use original data type for key field to preserve logical type
                    let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new("key", col_array.data_type().clone(), false),
                        arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                            std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                        ), false),
                    ]));

                    let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                    self.write_inverted_index(col_name, inv_schema, inv_batch)?;
                },

                // Binary / LargeBinary / FixedSizeBinary -> Key is Vec<u8>
                arrow::datatypes::DataType::Binary | arrow::datatypes::DataType::LargeBinary | arrow::datatypes::DataType::FixedSizeBinary(_) => {
                     println!("Indexing Binary column: {}", col_name);
                     // Cast to BinaryArray for uniform handling (if possible, else matching works)
                     // Simple handling: Iterate as BinaryArray (works for large and regular if we cast, or just use generics. 
                     // arrow::compute::cast to Binary is easiest)
                     let casted = arrow::compute::cast(col_array, &arrow::datatypes::DataType::Binary)?;
                     let array = casted.as_any().downcast_ref::<arrow::array::BinaryArray>().unwrap();
                     
                     let mut inverted_map: std::collections::HashMap<Vec<u8>, Vec<u32>> = std::collections::HashMap::new();
                     for (row_i, val) in array.iter().enumerate() {
                         if let Some(v) = val {
                             inverted_map.entry(v.to_vec()).or_default().push(row_i as u32);
                         }
                     }

                     let mut key_builder = arrow::array::BinaryBuilder::new();
                     let value_builder = arrow::array::UInt32Builder::new();
                     let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                     for (key, row_ids) in inverted_map {
                         key_builder.append_value(&key);
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
                         arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Binary, false),
                         arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                             std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                         ), false),
                     ]));

                     let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                     self.write_inverted_index(col_name, inv_schema, inv_batch)?;
                },

                // Decimal128
                arrow::datatypes::DataType::Decimal128(precision, scale) => {
                     println!("Indexing Decimal128 column: {}", col_name);
                     let array = col_array.as_any().downcast_ref::<arrow::array::Decimal128Array>().unwrap();
                     
                     let mut inverted_map: std::collections::HashMap<i128, Vec<u32>> = std::collections::HashMap::new();
                     for (row_i, val) in array.iter().enumerate() {
                         if let Some(v) = val {
                             inverted_map.entry(v).or_default().push(row_i as u32);
                         }
                     }

                     let mut key_builder = arrow::array::Decimal128Builder::with_capacity(inverted_map.len()).with_precision_and_scale(*precision, *scale)?;
                     let value_builder = arrow::array::UInt32Builder::new();
                     let mut list_builder = arrow::array::ListBuilder::new(value_builder);

                     for (key, row_ids) in inverted_map {
                         key_builder.append_value(key);
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
                         arrow::datatypes::Field::new("key", col_array.data_type().clone(), false),
                         arrow::datatypes::Field::new("row_ids", arrow::datatypes::DataType::List(
                             std::sync::Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::UInt32, true))
                         ), false),
                     ]));

                     let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;
                     self.write_inverted_index(col_name, inv_schema, inv_batch)?;
                },

                // Dictionary Types (Recursive)
                arrow::datatypes::DataType::Dictionary(_, value_type) => {
                    println!("Indexing Dictionary column: {} (unpacking to {:?})", col_name, value_type.as_ref());
                    // Cast to value type to unpack
                    let casted = arrow::compute::cast(col_array, value_type)
                        .map_err(|e| anyhow::anyhow!("Failed to unpack dictionary: {}", e))?;
                    self.index_column(col_name, &casted)?;
                },


                _ => {
                    // Skip unsupported
                    eprintln!("Warning: Skipping indexing for unsupported type: {:?}", col_array.data_type());
                }
            }

        Ok(())
    }

    /// Helper to write inverted index parquet file
    fn write_inverted_index(&self, col_name: &str, schema: std::sync::Arc<arrow::datatypes::Schema>, batch: RecordBatch) -> Result<()> {
        let is_remote = self.config.base_path.contains("://") && !self.config.base_path.starts_with("file://");
        let (inv_path, _staging_dir) = if is_remote {
             let temp_dir = std::env::temp_dir()
                .join("hyperstream_staging")
                .join(uuid::Uuid::new_v4().to_string());
             std::fs::create_dir_all(&temp_dir)?;
             let filename = format!("{}.{}.inv.parquet", self.config.segment_id, col_name);
             (temp_dir.join(&filename), Some(temp_dir))
        } else {
             let base = self.config.base_path.strip_prefix("file://").unwrap_or(&self.config.base_path);
             if !base.is_empty() {
                 std::fs::create_dir_all(base).context("Failed to create directory for inverted index")?;
             }
             let p = if base.is_empty() {
                  format!("{}.{}.inv.parquet", self.config.segment_id, col_name)
             } else {
                  format!("{}/{}.{}.inv.parquet", base, self.config.segment_id, col_name)
             };
             (std::path::PathBuf::from(p), None)
        };

        let inv_tmp = format!("{}.tmp", inv_path.to_str().unwrap());
        
        let inv_file = File::create(&inv_tmp)?;
        let props = parquet::file::properties::WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(inv_file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        std::fs::rename(&inv_tmp, &inv_path)?;

        {
             let mut files = self.generated_files.lock().unwrap();
             files.push(inv_path.to_str().unwrap().to_string());
        }
        println!("Inverted Index written to {}", inv_path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::array::{Int32Array, FixedSizeListArray, Float32Array};
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
            None
        )?;

        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("embedding", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim
            ), false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(id_array),
                Arc::new(vectors_array),
            ]
        )?;

        // 2. Write Segment
        let tmp_dir = std::env::temp_dir();
        let config = SegmentConfig::new(tmp_dir.to_str().unwrap(), "test_segment_001")
            .with_index_all(true);
        let writer = HybridSegmentWriter::new(config.clone());
        
        writer.write_batch(&batch)?;
        
        // Build indexes (required for index files to be created)
        writer.build_indexes(&batch)?;

        // 3. Verify Files
        let base = format!("{}/{}", config.base_path, config.segment_id);
        
        // Parquet
        assert!(std::path::Path::new(&format!("{}.parquet", base)).exists(), "Parquet file should exist");
        
        // Inverted Index for id column (replaces old .idx format)
        assert!(std::path::Path::new(&format!("{}.id.inv.parquet", base)).exists(), "Inverted index for id should exist");
        
        // Vector Index (embedding) - HNSW-IVF saves centroids and cluster graphs
        assert!(std::path::Path::new(&format!("{}.embedding.centroids.parquet", base)).exists(), "Vector index centroids should exist");
        assert!(std::path::Path::new(&format!("{}.embedding.cluster_0.hnsw.graph", base)).exists(), "Vector index graph should exist");

        Ok(())
    }
}
