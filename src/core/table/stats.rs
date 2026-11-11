// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::manifest::{Manifest, ManifestEntry, ManifestManager};
use crate::core::planner::{FilterExpr, QueryFilter, QueryPlanner};
use crate::core::reader::HybridReader;
use crate::core::storage::create_object_store;
use crate::SegmentConfig;
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
use anyhow::Result;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    pub index_files: Vec<crate::core::manifest::IndexFile>,
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
    /// The file has merge-on-read delete files. A raw per-file read would
    /// resurrect deleted rows, so connectors must not fan out such tables.
    #[serde(default)]
    pub has_deletes: bool,
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
        self.runtime().block_on(self.list_data_files_async())
    }

    /// Physical scan splits for connector-side parallelism.
    ///
    /// One split per data file. A file is the smallest unit the connectors can
    /// read independently today (`openFileSession` / `read_segment`), so
    /// sub-dividing a large file would need the footer-only metadata path;
    /// `row_group_ids` is left empty (meaning "all row groups") and reserved for
    /// that. `has_deletes` is set per file so a connector never fans out a table
    /// whose deletes would be bypassed by a raw per-file read.
    pub async fn list_scan_splits(&self) -> Result<Vec<Split>> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _) = manifest_manager.load_latest_full().await?;

        // A file can be reachable through more than one manifest entry
        // (carried-over `EXISTING` entries). Emitting it twice would duplicate
        // its rows for a connector that reads each split raw, so collapse by
        // path and keep the strongest signals (any index metadata, any delete
        // file).
        let mut by_path: std::collections::BTreeMap<String, Split> =
            std::collections::BTreeMap::new();
        for entry in entries {
            // Same absolute-path rule as `list_data_files`.
            let file_path = if entry.file_path.contains("://") {
                entry.file_path.clone()
            } else {
                let base = self.uri.trim_end_matches('/');
                let relative = entry.file_path.trim_start_matches('/');
                format!("{}/{}", base, relative)
            };
            let index_file_path = entry
                .index_files
                .first()
                .map(|index| index.file_path.clone());
            let has_index_files = !entry.index_files.is_empty();
            let has_deletes = !entry.delete_files.is_empty();

            match by_path.get_mut(&file_path) {
                Some(existing) => {
                    existing.has_deletes |= has_deletes;
                    existing.can_use_indexes |= has_index_files;
                    if existing.index_file_path.is_none() {
                        existing.index_file_path = index_file_path;
                    }
                }
                None => {
                    by_path.insert(
                        file_path.clone(),
                        Split {
                            file_path,
                            start_offset: 0,
                            length: entry.file_size_bytes.max(0) as u64,
                            row_group_ids: Vec::new(),
                            index_file_path,
                            can_use_indexes: has_index_files,
                            has_deletes,
                        },
                    );
                }
            }
        }
        Ok(by_path.into_values().collect())
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

            let has_scalar_indexes = entry
                .index_files
                .iter()
                .any(|f| f.index_category == "scalar" || f.is_lexical());
            let has_vector_indexes = entry
                .index_files
                .iter()
                .any(|f| f.index_category == "vector" || f.index_category == "hnsw");

            let indexed_columns = entry
                .index_files
                .iter()
                .map(|f| f.column_name.clone())
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
                index_files: entry.index_files.clone(),
            });
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Split enumeration
    // -----------------------------------------------------------------------

    /// Get splits for parallel reading (index-aware)
    pub fn get_splits(&self, max_split_size: usize, filter: Option<&str>) -> Result<Vec<Split>> {
        let files = self.list_data_files()?;

        let qf = filter.and_then(QueryFilter::parse);
        let qf_ref = qf.as_ref();

        let mut splits = Vec::new();

        let filtered_files = self.runtime().block_on(async {
            let mut valid_files = Vec::new();
            for file in files {
                let mut skip_file = false;

                if let Some(query_filter) = qf_ref {
                    // Sidecar Index Pushdown (Bitmap/Inverted)
                    if file.indexed_columns.contains(&query_filter.column)
                        && file.has_scalar_indexes
                    {
                        let rel_parquet =
                            crate::core::segment::relativize_against(&self.uri, &file.file_path);
                        let segment_id = crate::core::segment::segment_id_from_path(&rel_parquet);

                        let config = SegmentConfig::new("", segment_id)
                            .with_parquet_path(rel_parquet)
                            .with_data_store(
                                self.data_store
                                    .clone()
                                    .unwrap_or_else(|| self.store.clone()),
                            )
                            .with_index_files(file.index_files.clone())
                            .with_file_size(file.file_size_bytes);

                        let reader = HybridReader::new(config, self.store.clone(), &self.uri);

                        // Check bitmap
                        if let Ok(Some(bitmap)) =
                            reader.get_scalar_filter_bitmap(query_filter).await
                        {
                            if bitmap.is_empty() {
                                skip_file = true;
                            }
                        }
                    }
                }

                if !skip_file {
                    valid_files.push(file);
                }
            }
            valid_files
        });

        for file in filtered_files {
            if file.file_size_bytes > max_split_size as u64 {
                let num_splits = (file.file_size_bytes / max_split_size as u64) + 1;
                for i in 0..num_splits {
                    splits.push(Split {
                        file_path: file.file_path.clone(),
                        start_offset: i * max_split_size as u64,
                        length: max_split_size as u64,
                        row_group_ids: vec![i as usize],
                        index_file_path: file
                            .indexed_columns
                            .first()
                            .map(|_| file.file_path.replace(".parquet", "")),
                        can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                        // Legacy byte-range splitter: it cannot see delete files,
                        // so callers must not fan out from here.
                        has_deletes: true,
                    });
                }
            } else {
                splits.push(Split {
                    file_path: file.file_path.clone(),
                    start_offset: 0,
                    length: file.file_size_bytes,
                    row_group_ids: vec![0],
                    index_file_path: file
                        .indexed_columns
                        .first()
                        .map(|_| file.file_path.replace(".parquet", "")),
                    can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                    // Legacy byte-range splitter: it cannot see delete files,
                    // so callers must not fan out from here.
                    has_deletes: true,
                });
            }
        }
        Ok(splits)
    }

    /// Get splits for parallel reading (Index-aware, Async)
    pub async fn get_splits_async(
        &self,
        max_split_size: usize,
        filter: Option<&str>,
    ) -> Result<Vec<Split>> {
        let files = self.list_data_files_async().await?;

        let qf = filter.and_then(QueryFilter::parse);
        let qf_ref = qf.as_ref();

        let mut splits = Vec::new();

        let mut valid_files = Vec::new();
        for file in files {
            let mut skip_file = false;

            if let Some(query_filter) = qf_ref {
                // Sidecar Index Pushdown (Bitmap/Inverted)
                if file.indexed_columns.contains(&query_filter.column) && file.has_scalar_indexes {
                    let rel_parquet =
                        crate::core::segment::relativize_against(&self.uri, &file.file_path);
                    let segment_id = crate::core::segment::segment_id_from_path(&rel_parquet);

                    let config = SegmentConfig::new("", segment_id)
                        .with_parquet_path(rel_parquet)
                        .with_data_store(
                            self.data_store
                                .clone()
                                .unwrap_or_else(|| self.store.clone()),
                        )
                        .with_index_files(file.index_files.clone())
                        .with_file_size(file.file_size_bytes);

                    let reader = HybridReader::new(config, self.store.clone(), &self.uri);

                    // Check bitmap
                    if let Ok(Some(bitmap)) = reader.get_scalar_filter_bitmap(query_filter).await {
                        if bitmap.is_empty() {
                            skip_file = true;
                        }
                    }
                }
            }

            if !skip_file {
                valid_files.push(file);
            }
        }

        for file in valid_files {
            if file.file_size_bytes > max_split_size as u64 {
                let num_splits = (file.file_size_bytes / max_split_size as u64) + 1;
                for i in 0..num_splits {
                    splits.push(Split {
                        file_path: file.file_path.clone(),
                        start_offset: i * max_split_size as u64,
                        length: max_split_size as u64,
                        row_group_ids: vec![i as usize],
                        index_file_path: file
                            .indexed_columns
                            .first()
                            .map(|_| file.file_path.replace(".parquet", "")),
                        can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                        // Legacy byte-range splitter: it cannot see delete files,
                        // so callers must not fan out from here.
                        has_deletes: true,
                    });
                }
            } else {
                splits.push(Split {
                    file_path: file.file_path.clone(),
                    start_offset: 0,
                    length: file.file_size_bytes,
                    row_group_ids: vec![0],
                    index_file_path: file
                        .indexed_columns
                        .first()
                        .map(|_| file.file_path.replace(".parquet", "")),
                    can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                    // Legacy byte-range splitter: it cannot see delete files,
                    // so callers must not fan out from here.
                    has_deletes: true,
                });
            }
        }
        Ok(splits)
    }

    // -----------------------------------------------------------------------
    // File / split reading
    // -----------------------------------------------------------------------

    /// Read a specific data file (with index acceleration)
    pub async fn read_file_async(
        &self,
        file_path: &str,
        columns: Option<Vec<String>>,
        filter: Option<&str>,
    ) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, _) = manager.load_latest_full().await.unwrap_or_default();
        let iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();
        let target_rel = crate::core::segment::relativize_against(&self.uri, file_path);
        let target_id = crate::core::segment::segment_id_from_path(&target_rel);

        let reader = if let Some(entry) = all_entries.iter().find(|e| {
            let e_rel = crate::core::segment::relativize_against(&self.uri, &e.file_path);
            e.file_path == file_path
                || e_rel == target_rel
                || crate::core::segment::segment_id_from_path(&e_rel) == target_id
        }) {
            let mut r = self.segment_reader(entry)?;
            if let Some(ref s) = iceberg_schema {
                r = r.with_iceberg_schema(s.clone());
            }
            r
        } else {
            let (store, config) = if file_path.contains("://") {
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
                            path.file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("wrapper")
                                .to_string()
                        } else {
                            let p = url.path();
                            p.trim_start_matches('/').to_string()
                        };

                        let seg_id = crate::core::segment::segment_id_from_path(&relative_path);
                        let config =
                            SegmentConfig::new("", seg_id).with_parquet_path(relative_path);
                        (store, config)
                    }
                    Err(_) => {
                        let s = create_object_store(file_path)?;
                        let seg_id = crate::core::segment::segment_id_from_path(file_path);
                        let config = SegmentConfig::new("", seg_id);
                        (s, config)
                    }
                }
            } else {
                let seg_id = crate::core::segment::segment_id_from_path(file_path);
                let config = SegmentConfig::new("", seg_id)
                    .with_parquet_path(file_path.to_string())
                    .with_data_store(
                        self.data_store
                            .clone()
                            .unwrap_or_else(|| self.store.clone()),
                    );
                (self.store.clone(), config)
            };

            let mut r = HybridReader::new(config, store, &self.uri);
            if let Some(ref s) = iceberg_schema {
                r = r.with_iceberg_schema(s.clone());
            }
            r
        };

        use futures::StreamExt;

        let full_schema = if let Some(ref s) = iceberg_schema {
            Arc::new(s.to_arrow())
        } else {
            let s = self.arrow_schema();
            if s.fields().is_empty() {
                reader.get_arrow_schema().await.unwrap_or(s)
            } else {
                s
            }
        };

        // Resolve Target Schema (Projection)
        let target_schema = columns.as_ref().map(|cols| {
            let fields: Vec<arrow::datatypes::Field> = cols
                .iter()
                .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                .collect();
            Arc::new(Schema::new(fields))
        });

        let filter_expr_opt = if let Some(filter_str) = filter {
            let parsed = FilterExpr::parse_sql(filter_str, full_schema.clone()).await?;
            Some(Arc::new(parsed.coerce_literals(&full_schema)))
        } else {
            None
        };

        // Read all columns needed for filtering AND final projection so we
        // never project away predicate columns prior to evaluation.
        let read_schema = if let (Some(cols), Some(fe)) =
            (columns.as_ref(), filter_expr_opt.as_ref())
        {
            let mut read_cols: std::collections::HashSet<String> = cols.iter().cloned().collect();
            for ref_col in fe.get_referenced_columns() {
                read_cols.insert(ref_col);
            }
            let fields: Vec<arrow::datatypes::Field> = read_cols
                .iter()
                .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                .collect();
            Some(Arc::new(Schema::new(fields)))
        } else {
            target_schema.clone()
        };

        // 1. Try Index Read if filter is present
        let mut batches = Vec::new();
        let mut index_used = false;
        if let Some(ref fe) = filter_expr_opt {
            let and_filters = reader.rewrite_composite_filters(fe.extract_and_conditions());
            for qf in &and_filters {
                if let Ok(indexed_batches) = reader.query_index_first(qf, read_schema.clone()).await
                {
                    batches = indexed_batches;
                    index_used = true;
                    break;
                }
            }
        }

        // 2. Fallback to Full Scan when no index applied
        if !index_used {
            let stream = reader.stream_all(read_schema).await?;

            if let Some(filter_expr_owned) = filter_expr_opt {
                let target_schema_cloned = target_schema.clone();
                let filtered_stream = stream.filter_map(move |batch_res| {
                    let filter_expr_cloned = filter_expr_owned.clone();
                    let target_schema_inner = target_schema_cloned.clone();
                    async move {
                        let planner = QueryPlanner::new();
                        match batch_res {
                            Ok(b) => match planner.filter_expr(&b, &filter_expr_cloned) {
                                Ok(filtered) => {
                                    if filtered.num_rows() > 0 {
                                        Some(project_to_target_schema(
                                            &filtered,
                                            target_schema_inner.as_ref(),
                                        ))
                                    } else {
                                        None
                                    }
                                }
                                Err(e) => Some(Err(e)),
                            },
                            Err(e) => Some(Err(e)),
                        }
                    }
                });
                return Ok(filtered_stream.boxed());
            }
            return Ok(stream.boxed());
        }

        // 3. Apply post-filtering and final projection on indexed batches
        if let Some(filter_expr_owned) = filter_expr_opt {
            let target_schema_cloned = target_schema.clone();
            let stream = futures::stream::iter(batches.into_iter().map(Ok)).filter_map(
                move |batch_res: Result<RecordBatch>| {
                    let filter_expr_cloned = filter_expr_owned.clone();
                    let target_schema_inner = target_schema_cloned.clone();
                    async move {
                        let planner = QueryPlanner::new();
                        match batch_res {
                            Ok(b) => match planner.filter_expr(&b, &filter_expr_cloned) {
                                Ok(filtered) => {
                                    if filtered.num_rows() > 0 {
                                        Some(project_to_target_schema(
                                            &filtered,
                                            target_schema_inner.as_ref(),
                                        ))
                                    } else {
                                        None
                                    }
                                }
                                Err(e) => Some(Err(e)),
                            },
                            Err(e) => Some(Err(e)),
                        }
                    }
                },
            );
            Ok(stream.boxed())
        } else {
            let projected_batches: Vec<Result<RecordBatch>> = batches
                .into_iter()
                .map(|b| project_to_target_schema(&b, target_schema.as_ref()))
                .collect();
            Ok(futures::stream::iter(projected_batches).boxed())
        }
    }

    /// Read a specific split (with index acceleration)
    pub async fn read_split_async(
        &self,
        split: &Split,
        columns: Vec<String>,
        filter: Option<&str>,
    ) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        let file_path = &split.file_path;
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, _) = manager.load_latest_full().await.unwrap_or_default();
        let iceberg_schema = manifest
            .schemas
            .iter()
            .find(|s| s.schema_id == manifest.current_schema_id)
            .or_else(|| manifest.schemas.last())
            .cloned();
        let target_rel = crate::core::segment::relativize_against(&self.uri, file_path);
        let target_id = crate::core::segment::segment_id_from_path(&target_rel);

        let reader = if let Some(entry) = all_entries.iter().find(|e| {
            let e_rel = crate::core::segment::relativize_against(&self.uri, &e.file_path);
            e.file_path == *file_path
                || e_rel == target_rel
                || crate::core::segment::segment_id_from_path(&e_rel) == target_id
        }) {
            let mut r = self.segment_reader(entry)?;
            if let Some(ref s) = iceberg_schema {
                r = r.with_iceberg_schema(s.clone());
            }
            r
        } else {
            let (store, config) = if file_path.contains("://") {
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
                            path.file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("wrapper")
                                .to_string()
                        } else {
                            let p = url.path();
                            p.trim_start_matches('/').to_string()
                        };
                        let seg_id = crate::core::segment::segment_id_from_path(&relative_path);
                        let config =
                            SegmentConfig::new("", seg_id).with_parquet_path(relative_path);
                        (store, config)
                    }
                    Err(_) => {
                        let s = create_object_store(file_path)?;
                        let seg_id = crate::core::segment::segment_id_from_path(file_path);
                        let config = SegmentConfig::new("", seg_id);
                        (s, config)
                    }
                }
            } else {
                let seg_id = crate::core::segment::segment_id_from_path(file_path);
                let config = SegmentConfig::new("", seg_id)
                    .with_parquet_path(file_path.clone())
                    .with_data_store(
                        self.data_store
                            .clone()
                            .unwrap_or_else(|| self.store.clone()),
                    );
                (self.store.clone(), config)
            };

            let mut r = HybridReader::new(config, store, &self.uri);
            if let Some(ref s) = iceberg_schema {
                r = r.with_iceberg_schema(s.clone());
            }
            r
        };

        use futures::StreamExt;

        let full_schema = if let Some(ref s) = iceberg_schema {
            Arc::new(s.to_arrow())
        } else {
            let s = self.arrow_schema();
            if s.fields().is_empty() {
                reader.get_arrow_schema().await.unwrap_or(s)
            } else {
                s
            }
        };

        // Resolve Target Schema (Projection)
        let target_schema = if columns.is_empty() {
            None
        } else {
            let fields: Vec<arrow::datatypes::Field> = columns
                .iter()
                .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                .collect();
            Some(Arc::new(Schema::new(fields)))
        };

        let filter_expr_opt = if let Some(filter_str) = filter {
            let parsed = FilterExpr::parse_sql(filter_str, full_schema.clone()).await?;
            Some(Arc::new(parsed.coerce_literals(&full_schema)))
        } else {
            None
        };

        let read_schema =
            if let (Some(_), Some(fe)) = (target_schema.as_ref(), filter_expr_opt.as_ref()) {
                let mut read_cols: std::collections::HashSet<String> =
                    columns.iter().cloned().collect();
                for ref_col in fe.get_referenced_columns() {
                    read_cols.insert(ref_col);
                }
                let fields: Vec<arrow::datatypes::Field> = read_cols
                    .iter()
                    .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                    .collect();
                Some(Arc::new(Schema::new(fields)))
            } else {
                target_schema.clone()
            };

        // Index acceleration path:
        // When a filter resolves cleanly through an index (even to 0 rows),
        // return the post-filtered indexed result directly rather than falling
        // through to an unfiltered row-group scan.
        let mut index_used = false;
        let mut indexed_batches = Vec::new();
        if let Some(ref fe) = filter_expr_opt {
            let and_filters = reader.rewrite_composite_filters(fe.extract_and_conditions());
            for qf in &and_filters {
                if let Ok(batches) = reader.query_index_first(qf, read_schema.clone()).await {
                    indexed_batches = batches;
                    index_used = true;
                    break;
                }
            }
        }

        if index_used {
            if let Some(fe) = filter_expr_opt {
                let planner = QueryPlanner::new();
                let mut owned_batches = Vec::new();
                for b in indexed_batches {
                    match planner.filter_expr(&b, &fe) {
                        Ok(filtered) if filtered.num_rows() > 0 => {
                            owned_batches
                                .push(project_to_target_schema(&filtered, target_schema.as_ref()));
                        }
                        Ok(_) => {}
                        Err(e) => owned_batches.push(Err(e)),
                    }
                }
                return Ok(futures::stream::iter(owned_batches).boxed());
            }
            let owned_batches: Vec<Result<RecordBatch>> = indexed_batches
                .into_iter()
                .map(|b| project_to_target_schema(&b, target_schema.as_ref()))
                .collect();
            return Ok(futures::stream::iter(owned_batches).boxed());
        }

        let stream = reader
            .stream_row_groups(Some(&split.row_group_ids), read_schema)
            .await?;

        if let Some(filter_expr_owned) = filter_expr_opt {
            let target_schema_cloned = target_schema.clone();
            let filtered_stream = stream.filter_map(move |batch_res| {
                let filter_expr_cloned = filter_expr_owned.clone();
                let target_schema_inner = target_schema_cloned.clone();
                async move {
                    let planner = QueryPlanner::new();
                    match batch_res {
                        Ok(b) => match planner.filter_expr(&b, &filter_expr_cloned) {
                            Ok(filtered) => {
                                if filtered.num_rows() > 0 {
                                    Some(project_to_target_schema(
                                        &filtered,
                                        target_schema_inner.as_ref(),
                                    ))
                                } else {
                                    None
                                }
                            }
                            Err(e) => Some(Err(e)),
                        },
                        Err(e) => Some(Err(e)),
                    }
                }
            });
            return Ok(filtered_stream.boxed());
        }

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
                    let col = &idx.column_name;
                    {
                        if idx.index_category == "scalar" || idx.is_lexical() {
                            scalar_idx.insert(col.clone());
                        } else if idx.index_category == "vector" || idx.index_category == "hnsw" {
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
                let col = &idx.column_name;
                {
                    if idx.index_category == "scalar" || idx.is_lexical() {
                        scalar_idx.insert(col.clone());
                    } else if idx.index_category == "vector" || idx.index_category == "hnsw" {
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

    /// Read from the in-memory write buffer with optional filter and projection.
    /// Used by `BenoStreamExec` to include uncommitted data in SQL queries.
    pub fn read_write_buffer(
        &self,
        filter: Option<&QueryFilter>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let expr = filter.map(|f| FilterExpr::DataFusion(f.to_expr()));
        self.read_write_buffer_expr(expr.as_ref(), columns)
    }

    /// Read from the in-memory write buffer using a full [`FilterExpr`],
    /// evaluating predicates before projecting requested `columns`.
    pub fn read_write_buffer_expr(
        &self,
        expr: Option<&FilterExpr>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let mut result = Vec::new();
        let batches: Vec<RecordBatch> = {
            let pc_guard = self.pending_commit.lock();
            let pending = self.pending_writes.read();
            let mut combined = Vec::new();
            if let Some(pc) = pc_guard.as_ref() {
                if pc.phase == super::CommitPhase::Staging {
                    combined.extend(pc.batches.iter().cloned());
                }
            }
            combined.extend(pending.iter().map(|p| p.batch.clone()));
            combined
        };

        if batches.is_empty() {
            return Ok(result);
        }

        let planner = QueryPlanner::new();
        let table_schema = self.arrow_schema();

        for batch in &batches {
            let target_schema = if table_schema.fields().is_empty() {
                batch.schema()
            } else {
                table_schema.clone()
            };
            let aligned = if batch.schema() == target_schema {
                batch.clone()
            } else {
                let mut cols = Vec::with_capacity(target_schema.fields().len());
                for field in target_schema.fields() {
                    let col = if let Some(c) = batch.column_by_name(field.name()) {
                        c.clone()
                    } else {
                        arrow::array::new_null_array(field.data_type(), batch.num_rows())
                    };
                    cols.push(col);
                }
                RecordBatch::try_new(target_schema.clone(), cols)?
            };

            let coerced_expr = expr.map(|e| e.coerce_literals(&target_schema));

            // Evaluate filter BEFORE projecting columns so predicates on
            // non-projected columns work accurately.
            let filtered = if let Some(ref e) = coerced_expr {
                let out = planner.filter_expr(&aligned, e)?;
                if out.num_rows() == 0 {
                    continue;
                }
                out
            } else {
                aligned
            };

            let projected = if let Some(cols) = columns {
                let mut proj_fields = Vec::new();
                let mut proj_cols = Vec::new();
                for &name in cols {
                    if let Ok(idx) = filtered.schema().index_of(name) {
                        proj_fields.push(filtered.schema().field(idx).clone());
                        proj_cols.push(filtered.column(idx).clone());
                    }
                }
                let proj_schema = Arc::new(Schema::new(proj_fields));
                let mut opts = arrow::record_batch::RecordBatchOptions::default();
                opts.row_count = Some(filtered.num_rows());
                RecordBatch::try_new_with_options(proj_schema, proj_cols, &opts)?
            } else {
                filtered
            };

            if projected.num_rows() > 0 {
                result.push(projected);
            }
        }
        Ok(result)
    }
}

fn project_to_target_schema(
    batch: &RecordBatch,
    target_schema: Option<&Arc<Schema>>,
) -> Result<RecordBatch> {
    let Some(ts) = target_schema else {
        return Ok(batch.clone());
    };
    let mut fields = Vec::new();
    let mut cols = Vec::new();
    for field in ts.fields() {
        if let Ok(idx) = batch.schema().index_of(field.name()) {
            fields.push(batch.schema().field(idx).clone());
            cols.push(batch.column(idx).clone());
        }
    }
    let proj_schema = Arc::new(Schema::new(fields));
    let mut opts = arrow::record_batch::RecordBatchOptions::default();
    opts.row_count = Some(batch.num_rows());
    Ok(RecordBatch::try_new_with_options(proj_schema, cols, &opts)?)
}
