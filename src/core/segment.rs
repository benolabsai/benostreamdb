// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

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
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

#[allow(clippy::expect_used)] // Thread-pool construction only fails on OS resource exhaustion; unrecoverable at startup.
fn indexing_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("indexing-rayon-{}", i))
            .num_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .clamp(2, 8),
            )
            .build()
            .expect("Failed to build indexing thread pool")
    })
}

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
    // Additive buffers for multi-batch indexing. The writer is constructed
    // per-flush and driven from a single task, but `build_indexes` fans columns
    // out through a rayon `into_par_iter`, so these locks can be touched
    // concurrently. `inverted_data` is therefore only held for a short merge
    // after each column tokenizes into a task-local map (see
    // `build_inverted_index`); the remaining buffers are written briefly, once
    // per column, and are effectively uncontended.
    //
    // Each posting entry is `(row_id, position)` — the ordinal token position
    // within the document. Storing positions enables phrase queries, span
    // queries, and highlighting, and also speeds up tf computation (count
    // positions per doc instead of counting duplicate row_ids).
    pub(crate) inverted_data:
        parking_lot::Mutex<HashMap<String, std::collections::BTreeMap<String, Vec<(u32, u32)>>>>,
    pub index_metadata: parking_lot::Mutex<HashMap<String, String>>,
    pub(crate) vector_data: parking_lot::Mutex<HashMap<String, String>>,
    pub(crate) graph_data: parking_lot::Mutex<HashMap<String, String>>,
    pub(crate) file_checksum: parking_lot::Mutex<Option<String>>,
    pub(crate) puffin_index_entries: parking_lot::Mutex<Vec<crate::core::manifest::IndexFile>>,
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
            vector_data: parking_lot::Mutex::new(HashMap::new()),
            graph_data: parking_lot::Mutex::new(HashMap::new()),
            file_checksum: parking_lot::Mutex::new(None),
            puffin_index_entries: parking_lot::Mutex::new(Vec::new()),
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

        let mut parquet_file = String::new();
        let mut total_size = 0;

        // All secondary indexes are packed into the Puffin compound bundle;
        // the PuffinIndexWriter produces the IndexFile entries at bundle-finalize
        // time and stores them in puffin_index_entries.
        let index_files = self.puffin_index_entries.lock().clone();

        // Walk the generated files list to find the main Parquet data file and
        // its on-disk size; everything else is a Puffin artifact or temp file
        // and does not need a manifest entry.
        for f in &files {
            let filename = f.split('/').next_back().unwrap_or(f).to_string();
            if filename.ends_with(".parquet") {
                parquet_file = filename.clone();
                if let Ok(meta) = std::fs::metadata(f) {
                    total_size = meta.len() as i64;
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
            first_row_id: None,
        }
    }

    /// Compute vector statistics (BenoStream exclusive) while delegating
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

                // Merge in BenoStream-specific vector stats if applicable
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
                .join("benostream_staging")
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
            // Chunk-level statistics (per column chunk min/max, null counts).
            // These were disabled, which meant `merge_parquet_stats` had nothing
            // to read and every manifest entry ended up with empty `ColumnStats`
            // — making the planner's whole stats-pruning branch inert (only
            // partition pruning did any work). Chunk granularity is what
            // row-group pruning needs, at far less overhead than per-page stats.
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
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
            if !local_path.contains("benostream_staging") {
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

        // Determine the directory prefix from the existing parquet path or base_path
        let parent_prefix_buf = if let Some(p) = &self.config.parquet_path {
            if let Some(pos) = p.rfind('/') {
                p[..pos + 1].to_string()
            } else {
                "".to_string()
            }
        } else if !self.config.base_path.is_empty() {
            let base = self
                .config
                .base_path
                .strip_prefix("file://")
                .unwrap_or(&self.config.base_path)
                .trim_end_matches('/');
            format!("{}/", base)
        } else {
            "".to_string()
        };
        let parent_prefix = parent_prefix_buf.as_str();

        let puffin_filename = format!("{}.puffin", self.config.segment_id);
        let puffin_path_str = format!("{}{}", parent_prefix, puffin_filename);
        let mut puffin_writer = crate::core::puffin::PuffinIndexWriter::new_in_memory(
            puffin_path_str.clone(),
            self.config.active_snapshot_id.unwrap_or(0),
            self.file_checksum
                .lock()
                .clone()
                .or_else(|| self.config.file_checksum.clone())
                .unwrap_or_default(),
            self.get_record_count() as i64,
        )?;

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

            // BM25 bookkeeping: only string columns have an analyzer recorded
            // (build_inverted inserts into index_metadata for Utf8/LargeUtf8; the
            // Int32/Date32/Float inverted arms do not).
            let analyzer_name = self.index_metadata.lock().get(&col_name).cloned();

            if let Some(analyzer) = analyzer_name.as_ref() {
                // Per-row token counts from the inverted map, taken by reference
                // before the consuming loop below drains it. Rows with no tokens
                // are absent and read as 0 at query time.
                let mut doc_counts: HashMap<u32, u32> = HashMap::new();
                for postings in inverted_map.values() {
                    for &(row_id, _pos) in postings {
                        *doc_counts.entry(row_id).or_insert(0) += 1;
                    }
                }
                let mut doc_rows: Vec<(u32, u32)> = doc_counts.into_iter().collect();
                doc_rows.sort_unstable_by_key(|(row_id, _)| *row_id);

                let doclen_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                    arrow::datatypes::Field::new(
                        "row_id",
                        arrow::datatypes::DataType::UInt32,
                        false,
                    ),
                    arrow::datatypes::Field::new(
                        "token_count",
                        arrow::datatypes::DataType::UInt32,
                        false,
                    ),
                ]));
                let doc_rows_vec: Vec<u32> = doc_rows.iter().map(|(row_id, _)| *row_id).collect();
                let token_counts_vec: Vec<u32> = doc_rows.iter().map(|(_, count)| *count).collect();
                let doclen_batch = RecordBatch::try_new(
                    doclen_schema.clone(),
                    vec![
                        std::sync::Arc::new(arrow::array::UInt32Array::from(doc_rows_vec)),
                        std::sync::Arc::new(arrow::array::UInt32Array::from(token_counts_vec)),
                    ],
                )?;

                let mut doclen_buffer = Vec::new();
                {
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer =
                        ArrowWriter::try_new(&mut doclen_buffer, doclen_schema, Some(props))?;
                    writer.write(&doclen_batch)?;
                    writer.close()?;
                }
                puffin_writer.add_index_blob(
                    "lexical",
                    "doclen",
                    &col_name,
                    crate::core::puffin::PUFFIN_BLOB_LEXICAL_DOCLEN,
                    &doclen_buffer,
                    vec![],
                    HashMap::new(),
                )?;
                tracing::info!(
                    "  BM25 doc-length packed into Puffin bundle for column '{}' (analyzer: {})",
                    col_name,
                    analyzer
                );
            }

            // Build Arrow Arrays for Parquet:
            // Column 0: key (Utf8)
            // Column 1: row_ids (List<UInt32>) - unique doc IDs per term (delta-encoded)
            // Column 2: positions (List<List<UInt32>>) - ordinal token positions per doc
            let mut key_builder = arrow::array::StringBuilder::new();
            let value_builder = arrow::array::UInt32Builder::new();
            let mut list_builder = arrow::array::ListBuilder::new(value_builder);

            let pos_inner_builder = arrow::array::UInt32Builder::new();
            let pos_doc_builder = arrow::array::ListBuilder::new(pos_inner_builder);
            let mut positions_builder = arrow::array::ListBuilder::new(pos_doc_builder);

            for (key, mut postings) in inverted_map {
                key_builder.append_value(&key);
                postings.sort_unstable_by_key(|&(row_id, pos)| (row_id, pos));

                // Group postings by unique row_id
                let mut doc_postings: Vec<(u32, Vec<u32>)> = Vec::new();
                for (row_id, pos) in postings {
                    if let Some(last) = doc_postings.last_mut() {
                        if last.0 == row_id {
                            last.1.push(pos);
                            continue;
                        }
                    }
                    doc_postings.push((row_id, vec![pos]));
                }

                let mut last_id = 0;
                for (row_id, positions) in doc_postings {
                    list_builder.values().append_value(row_id - last_id);
                    last_id = row_id;

                    let doc_pos_builder = positions_builder.values();
                    for pos in positions {
                        doc_pos_builder.values().append_value(pos);
                    }
                    doc_pos_builder.append(true);
                }
                list_builder.append(true);
                positions_builder.append(true);
            }

            let key_array = std::sync::Arc::new(key_builder.finish());
            let list_array = std::sync::Arc::new(list_builder.finish());
            let pos_array = std::sync::Arc::new(positions_builder.finish());

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
                arrow::datatypes::Field::new(
                    "positions",
                    arrow::datatypes::DataType::List(std::sync::Arc::new(
                        arrow::datatypes::Field::new(
                            "item",
                            arrow::datatypes::DataType::List(std::sync::Arc::new(
                                arrow::datatypes::Field::new(
                                    "item",
                                    arrow::datatypes::DataType::UInt32,
                                    true,
                                ),
                            )),
                            true,
                        ),
                    )),
                    false,
                ),
            ]));

            let inv_batch =
                RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array, pos_array])?;
            // Write to memory buffer then to store. String-indexed columns embed
            // the analyzer name in key/value metadata so readers re-tokenize
            // queries identically.
            let mut buffer = Vec::new();
            {
                let kv_meta = analyzer_name.as_ref().map(|name| {
                    vec![parquet::file::metadata::KeyValue {
                        key: "analyzer".to_string(),
                        value: Some(name.clone()),
                    }]
                });
                let props = parquet::file::properties::WriterProperties::builder()
                    .set_key_value_metadata(kv_meta)
                    .build();
                let mut writer = ArrowWriter::try_new(&mut buffer, inv_schema, Some(props))?;
                writer.write(&inv_batch)?;
                writer.close()?;
            }

            let mut props = HashMap::new();
            if let Some(ref a) = analyzer_name {
                props.insert("analyzer".to_string(), a.clone());
            }
            puffin_writer.add_index_blob(
                "lexical",
                "inverted",
                &col_name,
                crate::core::puffin::PUFFIN_BLOB_LEXICAL_BM25,
                &buffer,
                vec![],
                props,
            )?;
            tracing::info!(
                "  Inverted Index packed into Puffin bundle for column '{}'",
                col_name
            );
        }

        // 2. Process Vector Index Buffers (Out of Core)
        let vector_data = {
            let mut vector_lock = self.vector_data.lock();
            std::mem::take(&mut *vector_lock)
        };

        for (col_name, tmp_path) in vector_data {
            tracing::info!(
                "Finishing Vector Index for column '{}' from out-of-core file",
                col_name
            );
            let mut algos = self
                .index_configs
                .get(&col_name)
                .map(|c| c.algorithms.clone())
                .unwrap_or_else(|| {
                    self.config
                        .column_algorithms
                        .get(&col_name)
                        .cloned()
                        .unwrap_or_default()
                });
            if algos.is_empty() {
                algos.push(crate::core::manifest::IndexAlgorithm::default());
            }

            for (idx, algo) in algos.iter().enumerate() {
                let metric = match algo {
                    crate::core::manifest::IndexAlgorithm::Hnsw { metric, .. }
                    | crate::core::manifest::IndexAlgorithm::HnswPq { metric, .. }
                    | crate::core::manifest::IndexAlgorithm::HnswTq4 { metric, .. }
                    | crate::core::manifest::IndexAlgorithm::HnswTq8 { metric, .. } => metric
                        .parse::<crate::core::index::VectorMetric>()
                        .with_context(|| {
                            format!("Invalid vector metric for {}: {}", col_name, metric)
                        })?,
                    _ => crate::core::index::VectorMetric::L2,
                };

                let algo_id = match algo {
                    crate::core::manifest::IndexAlgorithm::Hnsw { .. } => "hnsw",
                    crate::core::manifest::IndexAlgorithm::HnswPq { .. } => "pq",
                    crate::core::manifest::IndexAlgorithm::HnswTq4 { .. } => "tq4",
                    crate::core::manifest::IndexAlgorithm::HnswTq8 { .. } => "tq8",
                    _ => "idx",
                };

                let hnsw_ivf_index = crate::core::index::hnsw_ivf::HnswIvfIndex::build_from_file(
                    &tmp_path, metric, None, None, algo,
                )?;

                let suffix = if algos.len() > 1 {
                    format!("{}.{}.{}", col_name, algo_id, idx)
                } else {
                    format!("{}.{}", col_name, algo_id)
                };

                // Get the staging directory from the tmp_path
                let tmp_path_buf = std::path::PathBuf::from(&tmp_path);
                let local_staging_dir = tmp_path_buf
                    .parent()
                    .context("temporary vector file has no parent directory")?;

                let local_base_path =
                    local_staging_dir.join(format!("{}.{}", self.config.segment_id, suffix));

                let saved_files = hnsw_ivf_index
                    .save(local_base_path.to_str().context("Invalid UTF-8 in path")?)
                    .map_err(|e| anyhow::anyhow!("HNSW-IVF save failed: {}", e))?;

                {
                    let mut meta = self.index_metadata.lock();
                    meta.insert(
                        format!("{}.{}", self.config.segment_id, suffix),
                        algo.to_string(),
                    );
                }

                // Pack vector index artifacts into the Puffin bundle.
                for local_file in &saved_files {
                    let file_name = std::path::Path::new(local_file)
                        .file_name()
                        .context("saved index file has no file name")?
                        .to_string_lossy()
                        .to_string();
                    let data = std::fs::read(local_file)?;

                    let blob_type = if file_name.ends_with(".hnsw.graph") {
                        match algo_id {
                            "tq8" => crate::core::puffin::PUFFIN_BLOB_VECTOR_HNSW_TQ8,
                            "tq4" => crate::core::puffin::PUFFIN_BLOB_VECTOR_HNSW_TQ4,
                            "pq" => crate::core::puffin::PUFFIN_BLOB_VECTOR_HNSW_PQ,
                            _ => crate::core::puffin::PUFFIN_BLOB_VECTOR_HNSW,
                        }
                    } else if file_name.contains(".centroids.parquet") {
                        crate::core::puffin::PUFFIN_BLOB_HNSW_IVF_CENTROIDS
                    } else if file_name.contains(".mapping.parquet") {
                        crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_MAPPING
                    } else if file_name.contains(".cluster_") && file_name.ends_with(".hnsw.graph")
                    {
                        crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_GRAPH
                    } else {
                        continue;
                    };

                    let mut props = HashMap::new();
                    props.insert("filename".to_string(), file_name.clone());
                    puffin_writer.add_index_blob(
                        "vector",
                        algo_id,
                        &col_name,
                        blob_type,
                        &data,
                        vec![],
                        props,
                    )?;
                    // Clean up the local temp file after packing.
                    let _ = std::fs::remove_file(local_file);
                }
            }

            // Cleanup the temporary raw vector file
            let _ = std::fs::remove_file(&tmp_path);
        }

        // 3. Process Graph Index Buffers
        let graph_data = {
            let mut graph_lock = self.graph_data.lock();
            std::mem::take(&mut *graph_lock)
        };

        for (col_name, tmp_path) in graph_data {
            tracing::info!(
                "Finishing Graph Index for virtual column '{}' from tmp file",
                col_name
            );

            // Get the staging directory from the tmp_path
            let tmp_path_buf = std::path::PathBuf::from(&tmp_path);
            let local_staging_dir = tmp_path_buf
                .parent()
                .context("temporary graph file has no parent directory")?;

            let local_base_path =
                local_staging_dir.join(format!("{}.{}", self.config.segment_id, col_name));

            let saved_files = crate::core::index::csr_graph::MmapCsrGraph::build_from_file(
                &tmp_path_buf,
                &local_base_path,
            )
            .map_err(|e| anyhow::anyhow!("Graph build failed: {}", e))?;

            // Pack graph index artifacts into the Puffin bundle.
            for local_file in &saved_files {
                let file_name = std::path::Path::new(local_file)
                    .file_name()
                    .context("saved index file has no file name")?
                    .to_string_lossy()
                    .to_string();
                let data = std::fs::read(local_file)?;

                let blob_type = if file_name.ends_with(".csr.offsets") {
                    crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_OFFSETS
                } else if file_name.ends_with(".csr.edges") {
                    crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_EDGES
                } else if file_name.ends_with(".csr.dict") {
                    crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_DICT
                } else {
                    continue;
                };

                let mut props = HashMap::new();
                props.insert("filename".to_string(), file_name.clone());
                puffin_writer.add_index_blob(
                    "graph_v2",
                    "csr_graph",
                    &col_name,
                    blob_type,
                    &data,
                    vec![],
                    props,
                )?;
                // Clean up the local temp file after packing.
                let _ = std::fs::remove_file(local_file);
            }

            {
                let mut meta = self.index_metadata.lock();
                meta.insert(
                    format!("{}.{}", self.config.segment_id, col_name),
                    "csr_graph".to_string(),
                );
            }

            // Cleanup
            let _ = std::fs::remove_file(&tmp_path);
        }

        // 4. Pack loose scalar inverted-index files (Int32/Int64/Float/Date/
        // Bool) into the Puffin bundle. These arms of `build_inverted_index`
        // serialize directly to a staging file rather than the in-memory
        // `inverted_data` map, so they are collected here and removed from the
        // upload list.
        let loose_inv: Vec<String> = {
            let files = self.generated_files.lock();
            files
                .iter()
                .filter(|f| f.ends_with(".inv.parquet"))
                .cloned()
                .collect()
        };
        for path in loose_inv {
            let data = std::fs::read(&path)?;
            let filename = path.split('/').next_back().unwrap_or(&path);
            let col_name = filename.split('.').nth(1).unwrap_or("unknown").to_string();
            puffin_writer.add_index_blob(
                "lexical",
                "inverted",
                &col_name,
                crate::core::puffin::PUFFIN_BLOB_LEXICAL_BM25,
                &data,
                vec![],
                HashMap::new(),
            )?;
            let _ = std::fs::remove_file(&path);
            let mut files = self.generated_files.lock();
            files.retain(|f| f != &path);
        }

        // 4b. Pack JSON-path overlay files (`<segment>.<col>.jsonpath.parquet`)
        // into the Puffin bundle so they get a manifest `IndexFile` entry with
        // category `json_path`. Without this the file is uploaded but never
        // registered, and the reader cannot find it.
        let loose_jsonpath: Vec<String> = {
            let files = self.generated_files.lock();
            files
                .iter()
                .filter(|f| f.ends_with(".jsonpath.parquet"))
                .cloned()
                .collect()
        };
        for path in loose_jsonpath {
            let data = std::fs::read(&path)?;
            let filename = path.split('/').next_back().unwrap_or(&path);
            let col_name = filename.split('.').nth(1).unwrap_or("unknown").to_string();
            puffin_writer.add_index_blob(
                "json_path",
                "json_path",
                &col_name,
                crate::core::puffin::PUFFIN_BLOB_JSON_PATH,
                &data,
                vec![],
                HashMap::new(),
            )?;
            let _ = std::fs::remove_file(&path);
            let mut files = self.generated_files.lock();
            files.retain(|f| f != &path);
        }

        let (puffin_bytes, puffin_entries) = puffin_writer.finish_to_bytes()?;
        store
            .put(
                &object_store::path::Path::from(puffin_path_str.clone()),
                puffin_bytes.into(),
            )
            .await?;
        {
            let mut files = self.generated_files.lock();
            files.push(puffin_path_str.clone());
        }
        {
            let mut entries = self.puffin_index_entries.lock();
            *entries = puffin_entries;
        }
        tracing::info!(
            "  Puffin compound index bundle written to storage: {}",
            puffin_path_str
        );

        Ok(())
    }

    /// Build indexes for a batch (can be called asynchronously after write_batch).
    /// This is the expensive operation that should run in background.
    pub fn build_indexes(&self, batch: &RecordBatch, row_offset: usize) -> Result<()> {
        crate::core::run_blocking(|| {
            indexing_pool().install(|| {
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

                // 2. Build Composite Indexes (Virtual Columns)
                let mut composite_tasks = Vec::new();
                let mut graph_tasks = Vec::new();
                for (col_name, config) in &self.index_configs {
                    if !config.enabled {
                        continue;
                    }
                    for alg in &config.algorithms {
                        if let crate::core::manifest::IndexAlgorithm::CompositeBitmap { columns } =
                            alg
                        {
                            composite_tasks.push((col_name.clone(), columns.clone()));
                        } else if let crate::core::manifest::IndexAlgorithm::CsrGraph {
                            src_column,
                            dst_column,
                        } = alg
                        {
                            graph_tasks.push((
                                col_name.clone(),
                                src_column.clone(),
                                dst_column.clone(),
                            ));
                        }
                    }
                }

                composite_tasks
                    .into_par_iter()
                    .try_for_each(|(col_name, columns)| {
                        let mut builder = arrow::array::StringBuilder::new();
                        let num_rows = batch.num_rows();

                        let mut arrays = Vec::new();
                        for col in &columns {
                            if let Some(arr) = batch.column_by_name(col) {
                                arrays.push(arr);
                            } else {
                                return Err(anyhow::anyhow!(
                                    "Column {} not found for composite index",
                                    col
                                ));
                            }
                        }

                        for row in 0..num_rows {
                            let mut joined = String::new();
                            for (i, arr) in arrays.iter().enumerate() {
                                if i > 0 {
                                    joined.push('\0');
                                }
                                let val =
                                    crate::core::manifest::ManifestValue::from_array(arr, row)
                                        .to_string();
                                joined.push_str(&val);
                            }
                            builder.append_value(&joined);
                        }
                        let composite_array =
                            std::sync::Arc::new(builder.finish()) as std::sync::Arc<dyn Array>;

                        self.index_column(&col_name, &composite_array, row_offset)
                    })?;

                // 3. Build Graph Indexes
                graph_tasks
                    .into_par_iter()
                    .try_for_each(|(col_name, src_col, dst_col)| {
                        let is_remote = self.config.base_path.contains("://")
                            && !self.config.base_path.starts_with("file://");
                        let local_staging_dir = if is_remote {
                            let temp_dir = std::env::temp_dir()
                                .join("benostream_staging")
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

                        std::fs::create_dir_all(&local_staging_dir)?;

                        self.build_graph_index(
                            &col_name,
                            batch,
                            &src_col,
                            &dst_col,
                            row_offset,
                            &local_staging_dir,
                        )
                    })?;

                Ok(())
            }) // end indexing_pool
        }) // end block_in_place
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
        let desired_device = self
            .config
            .column_devices
            .get(col_name)
            .or(self.config.default_device.as_ref());

        if let Some(device_str) = desired_device {
            tracing::info!(
                "Applying device override for column {}: {}",
                col_name,
                device_str
            );

            // Avoid creating a new context if the existing one already matches the desired device backend.
            // CudaDevice::new() allocates memory, so we want to reuse it.
            let mut needs_new_context = true;
            if let Some(existing_ctx) = crate::core::index::gpu::get_thread_gpu_context() {
                if existing_ctx.backend_name() == device_str
                    || (device_str == "auto" && existing_ctx.is_available())
                {
                    needs_new_context = false;
                }
            }

            if needs_new_context {
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
                .join("benostream_staging")
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

    /// Does `ArrowWriter::close()` hand back metadata that still carries the
    /// chunk-level statistics we just asked it to write?
    ///
    /// `merge_parquet_stats` populates `column_stats` from exactly that
    /// metadata, so if the stats are absent here, every segment ends up with
    /// empty `column_stats` and the planner's stats pruning can never fire.
    #[test]
    fn parquet_close_metadata_carries_chunk_statistics() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let props = parquet::file::properties::WriterProperties::builder()
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
            .build();

        let mut buf: Vec<u8> = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        let metadata = writer.close().unwrap();

        let rg = metadata.row_groups().first().expect("one row group");
        let mut seen = Vec::new();
        for col in rg.columns() {
            seen.push((
                col.column_path().string(),
                col.statistics().is_some(),
                col.statistics()
                    .and_then(|s| s.null_count_opt())
                    .map(|_| "nulls"),
            ));
        }
        println!("close() metadata statistics: {seen:?}");
        assert!(
            seen.iter().any(|(_, has, _)| *has),
            "ArrowWriter::close() metadata carried no column statistics: {seen:?}"
        );
    }

    /// Does the real write path actually populate `column_stats`?
    ///
    /// The parquet metadata carries statistics (previous test), so if this
    /// comes back empty the loss is between `write_batch` and
    /// `to_manifest_entry`.
    #[tokio::test]
    async fn write_batch_populates_column_stats() -> Result<()> {
        use arrow::array::{Int64Array, StringArray};

        let dir = tempfile::tempdir()?;
        let base = dir.path().to_str().unwrap().to_string();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![40, 41, 42])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )?;

        let w = HybridSegmentWriter::new(SegmentConfig::new(&base, "seg_stats_test"));
        w.write_batch(&batch)?;
        let stats = w.get_stats();
        println!("column_stats after write_batch: {stats:?}");
        assert!(
            stats.get("id").and_then(|s| s.min.as_ref()).is_some(),
            "id min missing from column_stats: {stats:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_write_hybrid_segment() -> Result<()> {
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
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());
        let writer = HybridSegmentWriter::new(config.clone()).with_store(store);

        writer.write_batch(&batch)?;

        // Build indexes (required for index files to be created)
        writer.build_indexes(&batch, 0)?;
        writer.finish_indexing().await?;

        // 3. Verify Files
        let base = format!("{}/{}", config.base_path, config.segment_id);

        // Parquet
        assert!(
            std::path::Path::new(&format!("{}.parquet", base)).exists(),
            "Parquet file should exist"
        );

        // All secondary indexes are packed into a single Puffin compound bundle.
        let puffin_path = format!("{}.puffin", base);
        assert!(
            std::path::Path::new(&puffin_path).exists(),
            "Puffin compound index bundle should exist"
        );
        let bytes = std::fs::read(&puffin_path)?;
        let reader = crate::core::puffin::PuffinReader::new(std::io::Cursor::new(bytes))?;
        let types: Vec<&str> = reader
            .footer()
            .blobs
            .iter()
            .map(|b| b.r#type.as_str())
            .collect();
        assert!(
            types.contains(&crate::core::puffin::PUFFIN_BLOB_LEXICAL_BM25),
            "Puffin bundle should contain the lexical (inverted) index for `id` (has {types:?})"
        );
        assert!(
            types.contains(&crate::core::puffin::PUFFIN_BLOB_HNSW_IVF_CENTROIDS),
            "Puffin bundle should contain the vector centroids (has {types:?})"
        );
        assert!(
            types.contains(&crate::core::puffin::PUFFIN_BLOB_VECTOR_HNSW_TQ8),
            "Puffin bundle should contain the vector HNSW graph (has {types:?})"
        );

        Ok(())
    }
}
