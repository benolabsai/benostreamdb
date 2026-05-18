// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Statistics and connector APIs: data file enumeration, split generation,
/// table-level statistics, and index coverage reporting.
///
/// Contains:
/// - `DataFileInfo`, `Split`, `TableStatistics`, `IndexCoverage` structs
/// - `list_data_files`, `get_splits` (sync)
/// - `list_data_files_async`, `get_splits_async` (async)
/// - `read_file_async`, `read_split_async`
/// - `get_table_statistics`, `get_table_statistics_async`
/// - `get_snapshot_segments`, `get_snapshot_segments_with_version`
/// - `read_write_buffer`
use anyhow::{Result, Context};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::Schema;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use crate::core::storage::create_object_store;
use crate::core::manifest::{Manifest, ManifestEntry, ManifestManager};
use crate::core::planner::{QueryPlanner, QueryFilter, FilterExpr};
use crate::core::reader::HybridReader;
use crate::SegmentConfig;

use super::Table;

// ---------------------------------------------------------------------------
// Connector data structures
// ---------------------------------------------------------------------------

/// Information about a data file (for Spark/Trino file-level parallelism)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileInfo {
    pub file_path: String,
    pub row_count: u64,
    pub file_size_bytes: u64,
    pub min_values: std::collections::HashMap<String, String>,
    pub max_values: std::collections::HashMap<String, String>,

    // Index metadata
    pub has_scalar_indexes: bool,
    pub has_vector_indexes: bool,
    pub indexed_columns: Vec<String>,
}

/// Split information (for Trino split-level parallelism)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    pub file_path: String,
    pub start_offset: u64,
    pub length: u64,
    pub row_group_ids: Vec<usize>,

    // Index metadata
    pub index_file_path: Option<String>,
    pub can_use_indexes: bool,
}

/// Table-level statistics (for query planning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStatistics {
    pub row_count: u64,
    pub file_count: usize,
    pub total_size_bytes: u64,

    // Index coverage
    pub index_coverage: IndexCoverage,
}

/// Index coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexCoverage {
    pub scalar_indexed_columns: Vec<String>,
    pub vector_indexed_columns: Vec<String>,
    pub inverted_indexed_columns: Vec<String>,
    pub total_index_size_bytes: u64,
}

// ---------------------------------------------------------------------------
// impl Table — statistics / connector methods
// ---------------------------------------------------------------------------

impl Table {
    // -----------------------------------------------------------------------
    // Data file listing
    // -----------------------------------------------------------------------

    /// List all data files in the table (with index metadata)
    pub fn list_data_files(&self) -> Result<Vec<DataFileInfo>> {
        self.runtime().block_on(async {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

            let mut results = Vec::new();

            for entry in all_entries {
                let mut min_values = std::collections::HashMap::new();
                let mut max_values = std::collections::HashMap::new();

                for (col, stats) in &entry.column_stats {
                    if let Some(min) = &stats.min {
                        min_values.insert(col.clone(), min.to_string());
                    }
                    if let Some(max) = &stats.max {
                        max_values.insert(col.clone(), max.to_string());
                    }
                }

                let has_scalar_indexes = entry.index_files.iter().any(|f| f.index_type == "scalar" || f.index_type == "inverted");
                let has_vector_indexes = entry.index_files.iter().any(|f| f.index_type == "vector" || f.index_type == "hnsw");

                let indexed_columns = entry.index_files.iter()
                    .filter_map(|f| f.column_name.clone())
                    .collect();

                // Ensure absolute path if possible.
                let file_path = if entry.file_path.contains("://") {
                    entry.file_path.clone()
                } else {
                    // Try to join with uri
                    let base = self.uri.trim_end_matches('/');
                    let relative = entry.file_path.trim_start_matches('/');
                    format!("{}/{}", base, relative)
                };

                results.push(DataFileInfo {
                    file_path,
                    row_count: entry.record_count as u64,
                    file_size_bytes: entry.file_size_bytes as u64,
                    min_values,
                    max_values,
                    has_scalar_indexes,
                    has_vector_indexes,
                    indexed_columns,
                });
            }
            Ok(results)
        })
    }

    /// List all data files in the table (Async)
    pub async fn list_data_files_async(&self) -> Result<Vec<DataFileInfo>> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

        let mut results = Vec::new();
        for entry in all_entries {
            let mut min_values = std::collections::HashMap::new();
            let mut max_values = std::collections::HashMap::new();

            for (col, stats) in &entry.column_stats {
                if let Some(min) = &stats.min {
                    min_values.insert(col.clone(), min.to_string());
                }
                if let Some(max) = &stats.max {
                    max_values.insert(col.clone(), max.to_string());
                }
            }

            let has_scalar_indexes = entry.index_files.iter().any(|f| f.index_type == "scalar" || f.index_type == "inverted");
            let has_vector_indexes = entry.index_files.iter().any(|f| f.index_type == "vector" || f.index_type == "hnsw");

            let indexed_columns = entry.index_files.iter()
                .filter_map(|f| f.column_name.clone())
                .collect();

            let file_path = if entry.file_path.contains("://") {
                entry.file_path.clone()
            } else {
                let base = self.uri.trim_end_matches('/');
                let relative = entry.file_path.trim_start_matches('/');
                format!("{}/{}", base, relative)
            };

            results.push(DataFileInfo {
                file_path,
                row_count: entry.record_count as u64,
                file_size_bytes: entry.file_size_bytes as u64,
                min_values,
                max_values,
                has_scalar_indexes,
                has_vector_indexes,
                indexed_columns,
            });
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Split enumeration
    // -----------------------------------------------------------------------

    /// Get splits for parallel reading (index-aware)
    pub fn get_splits(&self, max_split_size: usize) -> Result<Vec<Split>> {
        let files = self.list_data_files()?;
        let mut splits = Vec::new();

        for file in files {
            if file.file_size_bytes > max_split_size as u64 {
                let num_splits = (file.file_size_bytes / max_split_size as u64) + 1;
                for i in 0..num_splits {
                    splits.push(Split {
                        file_path: file.file_path.clone(),
                        start_offset: i * max_split_size as u64,
                        length: max_split_size as u64,
                        row_group_ids: vec![i as usize],
                        index_file_path: file.indexed_columns.first().map(|_| {
                            file.file_path.replace(".parquet", "")
                        }),
                        can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                    });
                }
            } else {
                splits.push(Split {
                    file_path: file.file_path.clone(),
                    start_offset: 0,
                    length: file.file_size_bytes,
                    row_group_ids: vec![0],
                    index_file_path: file.indexed_columns.first().map(|_| {
                        file.file_path.replace(".parquet", "")
                    }),
                    can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                });
            }
        }
        Ok(splits)
    }

    /// Get splits for parallel reading (Index-aware, Async)
    pub async fn get_splits_async(&self, max_split_size: usize) -> Result<Vec<Split>> {
        let files = self.list_data_files_async().await?;
        let mut splits = Vec::new();
        for file in files {
            if file.file_size_bytes > max_split_size as u64 {
                let num_splits = (file.file_size_bytes / max_split_size as u64) + 1;
                for i in 0..num_splits {
                    splits.push(Split {
                        file_path: file.file_path.clone(),
                        start_offset: i * max_split_size as u64,
                        length: max_split_size as u64,
                        row_group_ids: vec![i as usize],
                        index_file_path: file.indexed_columns.first().map(|_| {
                            file.file_path.replace(".parquet", "")
                        }),
                        can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                    });
                }
            } else {
                splits.push(Split {
                    file_path: file.file_path.clone(),
                    start_offset: 0,
                    length: file.file_size_bytes,
                    row_group_ids: vec![0],
                    index_file_path: file.indexed_columns.first().map(|_| {
                        file.file_path.replace(".parquet", "")
                    }),
                    can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                });
            }
        }
        Ok(splits)
    }

    // -----------------------------------------------------------------------
    // File / split reading
    // -----------------------------------------------------------------------

    /// Read a specific data file (with index acceleration)
    pub async fn read_file_async(&self, file_path: &str, columns: Option<Vec<String>>, filter: Option<&str>) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        // Use HybridReader
        let parts: Vec<&str> = file_path.split('/').collect();
        let filename = parts.last().unwrap_or(&"wrapper");
        let segment_id = filename.replace(".parquet", "");

        let (store, mut config) = if file_path.contains("://") {
            let parse_res = url::Url::parse(file_path);
            match parse_res {
                Ok(url) => {
                    let scheme = url.scheme();
                    let store = if scheme == "file" {
                        let path_str = url.path();
                        let path = std::path::Path::new(path_str);
                        let parent = path.parent().unwrap_or(std::path::Path::new("/"));
                        let parent_uri = format!("file://{}", parent.to_string_lossy());
                        create_object_store(&parent_uri)?
                    } else {
                        create_object_store(file_path)?
                    };

                    let relative_path = if scheme == "file" {
                        let path = std::path::Path::new(url.path());
                        path.file_name().and_then(|s| s.to_str()).unwrap_or("wrapper").to_string()
                    } else {
                        let p = url.path();
                        p.trim_start_matches('/').to_string()
                    };

                    let segment_id_full = relative_path;
                    let segment_id = segment_id_full.strip_suffix(".parquet").unwrap_or(&segment_id_full).to_string();

                    let config = SegmentConfig::new("", &segment_id);
                    (store, config)
                },
                Err(_) => {
                    let s = create_object_store(file_path)?;
                    let config = SegmentConfig::new("", &segment_id);
                    (s, config)
                }
            }
        } else {
            let config = SegmentConfig::new("", &segment_id);
            (self.store.clone(), config)
        };

        // Try to enrich config from manifest
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manager.load_latest_full().await.unwrap_or_default();
        if let Some(entry) = all_entries.iter().find(|e| e.file_path == file_path || e.file_path.ends_with(file_path)) {
            config = config.with_parquet_path(entry.file_path.clone())
                .with_delete_files(entry.delete_files.clone())
                .with_index_files(entry.index_files.clone())
                .with_file_size(entry.file_size_bytes as u64);
        }

        let reader = HybridReader::new(config, store, &self.uri);
        use futures::StreamExt;

        // Resolve Target Schema (Projection)
        let target_schema = if let Some(cols) = columns {
            let current_schema = self.arrow_schema();
            let fields: Vec<arrow::datatypes::Field> = cols.iter()
                .filter_map(|name| current_schema.field_with_name(name).ok().cloned())
                .collect();
            if fields.is_empty() {
                Some(Arc::new(Schema::new(Vec::<arrow::datatypes::Field>::new())))
            } else {
                Some(Arc::new(Schema::new(fields)))
            }
        } else {
            None
        };

        let mut batches = Vec::new();

        // 1. Try Index Read if filter is present
        let mut index_used = false;
        if let Some(filter_str) = filter {
            if let Some(qf) = QueryFilter::parse(filter_str) {
                if let Ok(indexed_batches) = reader.query_index_first(&qf, target_schema.clone()).await {
                    batches = indexed_batches;
                    index_used = true;
                }
            }
        }

        // 2. Fallback to Full Scan
        if !index_used {
            let stream = reader.stream_all(target_schema).await?;

            // Apply post-filter on full scan if filter is present
            if let Some(filter_str) = filter {
                let filter_expr_owned = Arc::new(FilterExpr::parse_sql(filter_str, self.arrow_schema()).await?);
                let filtered_stream = stream.filter_map(move |batch_res| {
                    let filter_expr_cloned = filter_expr_owned.clone();
                    async move {
                        let planner = QueryPlanner::new();
                        match batch_res {
                            Ok(b) => {
                                match planner.filter_expr(&b, &filter_expr_cloned) {
                                    Ok(filtered) => if filtered.num_rows() > 0 { Some(Ok::<arrow::record_batch::RecordBatch, anyhow::Error>(filtered)) } else { None },
                                    Err(e) => {
                                        tracing::error!("Error evaluating filter: {}", e);
                                        Some(Ok::<arrow::record_batch::RecordBatch, anyhow::Error>(b))
                                    }
                                }
                            }
                            Err(e) => Some(Err(e))
                        }
                    }
                });
                return Ok(filtered_stream.boxed());
            }
            return Ok(stream.boxed());
        }

        // 3. Apply post-filtering if filter is present
        if let Some(filter_str) = filter {
            let filter_expr_owned = Arc::new(FilterExpr::parse_sql(filter_str, self.arrow_schema()).await?);

            let stream = futures::stream::iter(batches.into_iter().map(Ok)).filter_map(move |batch_res: Result<RecordBatch>| {
                let filter_expr_cloned = filter_expr_owned.clone();
                async move {
                    let planner = QueryPlanner::new();
                    match batch_res {
                        Ok(b) => {
                            match planner.filter_expr(&b, &filter_expr_cloned) {
                                Ok(filtered) => if filtered.num_rows() > 0 { Some(Ok::<arrow::record_batch::RecordBatch, anyhow::Error>(filtered)) } else { None },
                                Err(e) => {
                                    tracing::error!("Error evaluating filter: {}", e);
                                    Some(Ok::<arrow::record_batch::RecordBatch, anyhow::Error>(b))
                                }
                            }
                        }
                        Err(e) => Some(Err(e))
                    }
                }
            });
            Ok(stream.boxed())
        } else {
            let stream = futures::stream::iter(batches.into_iter().map(Ok::<_, anyhow::Error>));
            Ok(stream.boxed())
        }
    }

    /// Read a specific split (with index acceleration)
    pub async fn read_split_async(&self, split: &Split, columns: Vec<String>, filter: Option<&str>) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        // New Implementation: Use stream_row_groups with column pushdown
        let file_path = &split.file_path;

        let (store, mut config) = if file_path.contains("://") {
            match url::Url::parse(file_path) {
                Ok(url) => {
                    let scheme = url.scheme();
                    let store = if scheme == "file" {
                        let path_str = url.path();
                        let path = std::path::Path::new(path_str);
                        let parent = path.parent().unwrap_or(std::path::Path::new("/"));
                        let parent_uri = format!("file://{}", parent.to_string_lossy());
                        create_object_store(&parent_uri)?
                    } else {
                        create_object_store(file_path)?
                    };

                    let relative_path = if scheme == "file" {
                        let path = std::path::Path::new(url.path());
                        path.file_name().and_then(|s| s.to_str()).unwrap_or("wrapper").to_string()
                    } else {
                        let p = url.path();
                        p.trim_start_matches('/').to_string()
                    };
                    let segment_id_full = relative_path;
                    let segment_id = segment_id_full.strip_suffix(".parquet").unwrap_or(&segment_id_full).to_string();
                    let config = SegmentConfig::new("", &segment_id);
                    (store, config)
                },
                Err(_) => {
                    let s = create_object_store(file_path)?;
                    let parts: Vec<&str> = file_path.split('/').collect();
                    let filename = parts.last().unwrap_or(&"wrapper");
                    let segment_id = filename.replace(".parquet", "");
                    let config = SegmentConfig::new("", &segment_id);
                    (s, config)
                }
            }
        } else {
            let parts: Vec<&str> = file_path.split('/').collect();
            let filename = parts.last().unwrap_or(&"wrapper");
            let segment_id = filename.replace(".parquet", "");
            let config = SegmentConfig::new("", &segment_id);
            (self.store.clone(), config)
        };

        // Enrich config from manifest (index files, delete files, file size)
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        if let Ok((_, all_entries, _)) = manager.load_latest_full().await {
            if let Some(entry) = all_entries.iter().find(|e| e.file_path == *file_path || e.file_path.ends_with(file_path)) {
                config = config.with_parquet_path(entry.file_path.clone())
                    .with_delete_files(entry.delete_files.clone())
                    .with_index_files(entry.index_files.clone())
                    .with_file_size(entry.file_size_bytes as u64);
            }
        }

        let reader = HybridReader::new(config, store, &self.uri);
        use futures::StreamExt;

        // Resolve Target Schema (Projection)
        let target_schema = if columns.is_empty() {
            None
        } else {
            let current_schema = self.arrow_schema();
            let mut fields: Vec<arrow::datatypes::Field> = columns.iter()
                .filter_map(|name| current_schema.field_with_name(name).ok().cloned())
                .collect();
            if fields.is_empty() {
                // Fallback: if the table schema is empty (no committed data),
                // resolve columns against the Parquet file's own schema
                if let Ok(file_schema) = reader.get_arrow_schema().await {
                    fields = columns.iter()
                        .filter_map(|name| file_schema.field_with_name(name).ok().cloned())
                        .collect();
                }
                if fields.is_empty() {
                    Some(Arc::new(Schema::new(Vec::<arrow::datatypes::Field>::new())))
                } else {
                    Some(Arc::new(Schema::new(fields)))
                }
            } else {
                Some(Arc::new(Schema::new(fields)))
            }
        };

        // Index acceleration path:
        // When a filter is provided and indexes exist, query the index for a bitmap
        // of matching rows, then stream the filtered row groups.
        if let Some(filter_str) = filter {
            if let Some(qf) = QueryFilter::parse(filter_str) {
                if let Ok(indexed_batches) = reader.query_index_first(&qf, target_schema.clone()).await {
                    if !indexed_batches.is_empty() {
                        let owned_batches: Vec<Result<RecordBatch>> = indexed_batches.into_iter().map(Ok).collect();
                        return Ok(futures::stream::iter(owned_batches).boxed());
                    }
                }
            }
        }

        let stream = reader.stream_row_groups(Some(&split.row_group_ids), target_schema).await?;
        Ok(stream.boxed())
    }

    // -----------------------------------------------------------------------
    // Table statistics
    // -----------------------------------------------------------------------

    /// Get table-level statistics (with index info)
    pub fn get_table_statistics(&self) -> Result<TableStatistics> {
        self.runtime().block_on(async {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

            let row_count = all_entries.iter().map(|e| e.record_count).sum::<i64>() as u64;
            let total_size = all_entries.iter().map(|e| e.file_size_bytes).sum::<i64>() as u64;

            // Calculate basic index coverage
            let mut scalar_idx = std::collections::HashSet::new();
            let mut vector_idx = std::collections::HashSet::new();

            let total_index_size = 0;

            for entry in &all_entries {
                for idx in &entry.index_files {
                    if let Some(col) = &idx.column_name {
                        if idx.index_type == "scalar" || idx.index_type == "inverted" {
                            scalar_idx.insert(col.clone());
                        } else if idx.index_type == "vector" || idx.index_type == "hnsw" {
                            vector_idx.insert(col.clone());
                        }
                    }
                }
            }

            let index_coverage = IndexCoverage {
                scalar_indexed_columns: scalar_idx.clone().into_iter().collect(),
                vector_indexed_columns: vector_idx.into_iter().collect(),
                inverted_indexed_columns: scalar_idx.into_iter().collect(),
                total_index_size_bytes: total_index_size,
            };

            Ok(TableStatistics {
                row_count,
                file_count: all_entries.len(),
                total_size_bytes: total_size,
                index_coverage,
            })
        })
    }

    /// Get table-level statistics (Asynchronous)
    pub async fn get_table_statistics_async(&self) -> Result<TableStatistics> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

        let row_count = all_entries.iter().map(|e| e.record_count).sum::<i64>() as u64;
        let total_size = all_entries.iter().map(|e| e.file_size_bytes).sum::<i64>() as u64;

        let mut scalar_idx = std::collections::HashSet::new();
        let mut vector_idx = std::collections::HashSet::new();

        let total_index_size = 0;

        for entry in &all_entries {
            for idx in &entry.index_files {
                if let Some(col) = &idx.column_name {
                    if idx.index_type == "scalar" || idx.index_type == "inverted" {
                        scalar_idx.insert(col.clone());
                    } else if idx.index_type == "vector" || idx.index_type == "hnsw" {
                        vector_idx.insert(col.clone());
                    }
                }
            }
        }

        let index_coverage = IndexCoverage {
            scalar_indexed_columns: scalar_idx.clone().into_iter().collect(),
            vector_indexed_columns: vector_idx.into_iter().collect(),
            inverted_indexed_columns: scalar_idx.into_iter().collect(),
            total_index_size_bytes: total_index_size,
        };

        Ok(TableStatistics {
            row_count,
            file_count: all_entries.len(),
            total_size_bytes: total_size,
            index_coverage,
        })
    }

    // -----------------------------------------------------------------------
    // Snapshot segments
    // -----------------------------------------------------------------------

    pub async fn get_snapshot_segments(&self) -> Result<Vec<ManifestEntry>> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_, all_entries, _) = manifest_manager.load_latest_full().await?;
        Ok(all_entries)
    }

    pub async fn get_snapshot_segments_with_version(&self) -> Result<(Manifest, u64)> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.load_latest().await
    }

    // -----------------------------------------------------------------------
    // Write buffer read
    // -----------------------------------------------------------------------

    /// Read from the in-memory write buffer with optional filter and projection
    /// Used by HyperStreamExec to include uncommitted data in SQL queries
    pub fn read_write_buffer(
        &self,
        filter: Option<&QueryFilter>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let mut result = Vec::new();
        let buffer = self.write_buffer.read();

        if buffer.is_empty() {
            return Ok(result);
        }

        let planner = QueryPlanner::new();

        for batch in buffer.iter() {
            // Apply projection first
            let batch_to_filter = if let Some(cols) = columns {
                let indices: Vec<usize> = cols.iter()
                    .filter_map(|name| batch.schema().index_of(name).ok())
                    .collect();
                batch.project(&indices).unwrap_or(batch.clone())
            } else {
                batch.clone()
            };

            // Apply filter if present
            if let Some(f) = filter {
                if let Ok(filtered) = planner.filter_batch(&batch_to_filter, f) {
                    if filtered.num_rows() > 0 {
                        result.push(filtered);
                    }
                }
            } else {
                result.push(batch_to_filter);
            }
        }
        Ok(result)
    }
}
