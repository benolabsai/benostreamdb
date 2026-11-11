// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::cache::CacheExt;
use arrow::array::Array;
use bytes::Bytes;
use chrono::Utc;
use futures::StreamExt;
use object_store::{path::Path, ObjectMeta, ObjectStore};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, RowSelection, RowSelector,
};
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::ProjectionMask;
use std::sync::Arc;

use anyhow::Result;
use roaring::RoaringBitmap;

use super::*;

impl HybridReader {
    pub async fn check_bloom_filter(
        &self,
        column: &str,
        value: &serde_json::Value,
    ) -> Result<bool> {
        let metadata: std::sync::Arc<parquet::file::metadata::ParquetMetaData> =
            self.get_parquet_metadata().await?;
        let schema = metadata.file_metadata().schema_descr();

        let col_idx = schema
            .columns()
            .iter()
            .position(|c| c.name() == column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in Parquet schema", column))?;

        use parquet::bloom_filter::Sbbf;

        let mut possible = false;
        let mut checked_any_bloom_filter = false;
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();
        if pq_path_str.is_empty() {
            return Ok(true);
        }
        let data_store = self.data_store();

        // Get file size to handle Range correctly
        let file_size = if let Some(s) = self.config.file_size {
            s
        } else {
            data_store.head(&pq_path).await?.size
        };

        for i in 0..metadata.num_row_groups() {
            let rg_meta = metadata.row_group(i);
            let col_meta = rg_meta.column(col_idx);

            if let Some(offset) = col_meta.bloom_filter_offset() {
                checked_any_bloom_filter = true;
                let cache_key = format!("{}/{}:{}", self.cache_root(), pq_path_str, offset);

                let sbbf = if let Some(cached) = crate::core::cache::BLOOM_FILTER_CACHE
                    .get_with_metrics(&cache_key, "bloom_filter")
                    .await
                {
                    tracing::debug!("Bloom Cache HIT for key: {}", cache_key);
                    cached
                } else {
                    tracing::debug!("Bloom Cache MISS for key: {}", cache_key);
                    let start = offset as u64;
                    let end = (start + 2 * 1024 * 1024).min(file_size);

                    // Fetch the Bloom Filter blob from storage
                    let data_res = data_store
                        .get_range(&pq_path, start..end)
                        .await
                        .inspect(|b| {
                            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                        });
                    let data = match data_res {
                        Ok(d) => d,
                        Err(_) => {
                            let b = data_store.get_range(&pq_path, start..file_size).await?;
                            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                            b
                        }
                    };

                    let filter_res = Sbbf::from_bytes(&data);
                    let filter = match filter_res {
                        Ok(f) => f,
                        Err(e) => {
                            let msg = e.to_string();
                            if msg.contains("extra bytes") {
                                // Extract expected length from error: "expected 1048594 total bytes, got ..."
                                if let Some(expected_pos) = msg.find("expected ") {
                                    let rest = &msg[expected_pos + 9..];
                                    if let Some(space_pos) = rest.find(' ') {
                                        if let Ok(expected_len) = rest[..space_pos].parse::<usize>()
                                        {
                                            if expected_len <= data.len() {
                                                tracing::debug!("Retrying Bloom Filter parse with truncated length: {}", expected_len);
                                                Sbbf::from_bytes(&data[..expected_len])?
                                            } else {
                                                return Err(e.into());
                                            }
                                        } else {
                                            return Err(e.into());
                                        }
                                    } else {
                                        return Err(e.into());
                                    }
                                } else {
                                    return Err(e.into());
                                }
                            } else {
                                return Err(e.into());
                            }
                        }
                    };

                    let arc_filter = Arc::new(filter);
                    crate::core::cache::BLOOM_FILTER_CACHE
                        .insert(cache_key.clone(), arc_filter.clone())
                        .await;
                    tracing::debug!("Bloom Cache INSERTED for key: {}", cache_key);
                    arc_filter
                };

                let desc = schema.column(col_idx);
                let physical_type = desc.physical_type();

                let matches = match physical_type {
                    parquet::basic::Type::INT64 => {
                        if let Some(i) = value.as_i64() {
                            sbbf.check(&i)
                        } else {
                            true
                        }
                    }
                    parquet::basic::Type::INT32 => {
                        if let Some(i32_val) = value
                            .as_i64()
                            .and_then(|i| i32::try_from(i).ok())
                            .or_else(|| {
                                value.as_str().and_then(|s| {
                                    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                                        .ok()
                                        .and_then(|d| {
                                            chrono::NaiveDate::from_ymd_opt(1970, 1, 1).and_then(
                                                |epoch| i32::try_from((d - epoch).num_days()).ok(),
                                            )
                                        })
                                })
                            })
                        {
                            sbbf.check(&i32_val)
                        } else {
                            true
                        }
                    }
                    parquet::basic::Type::BYTE_ARRAY
                    | parquet::basic::Type::FIXED_LEN_BYTE_ARRAY => {
                        if let Some(s) = value.as_str() {
                            // s is &str. &s is &&str. &str implements AsBytes.
                            sbbf.check(&s)
                        } else if let Some(vals) = value.as_array() {
                            let bytes: Option<Vec<u8>> = vals
                                .iter()
                                .map(|v| v.as_u64().and_then(|n| u8::try_from(n).ok()))
                                .collect();
                            if let Some(bytes) = bytes {
                                sbbf.check(&bytes)
                            } else {
                                true
                            }
                        } else {
                            true
                        }
                    }
                    parquet::basic::Type::FLOAT => {
                        if let Some(f) = value.as_f64() {
                            let f32_val = f as f32;
                            sbbf.check(&f32_val)
                        } else {
                            true
                        }
                    }
                    parquet::basic::Type::DOUBLE => {
                        if let Some(f) = value.as_f64() {
                            sbbf.check(&f)
                        } else {
                            true
                        }
                    }
                    _ => true,
                };

                if matches {
                    possible = true;
                    break;
                }
            }
        }

        if !checked_any_bloom_filter {
            return Ok(true);
        }

        Ok(possible)
    }

    pub async fn check_value_exists(
        &self,
        column: &str,
        value: &serde_json::Value,
    ) -> Result<bool> {
        // 1. Bloom Filter (Fastest rejection)
        if !self.check_bloom_filter(column, value).await? {
            return Ok(false);
        }

        // 2. Inverted Index / Bitmap (Precise lookup)
        let filter = crate::core::planner::QueryFilter {
            column: column.to_string(),
            min: Some(value.clone()),
            min_inclusive: true,
            max: Some(value.clone()),
            max_inclusive: true,
            values: Some(vec![value.clone()]),
            negated: false,
        };

        if let Some((bitmap, is_exact)) = self
            .get_scalar_filter_bitmap_with_exactness(&filter)
            .await?
        {
            if bitmap.is_empty() {
                return Ok(false);
            }
            if is_exact {
                return Ok(true);
            }
        }

        // 3. Fallback: Full Scan (projecting target column; stream_all already applies position & equality deletes)
        let proj_schema = self
            .get_arrow_schema()
            .await
            .ok()
            .and_then(|s| s.field_with_name(column).ok().cloned())
            .map(|f| Arc::new(arrow::datatypes::Schema::new(vec![f])));
        let mut stream = self.stream_all(proj_schema).await?;
        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in scan", column))?;

            for i in 0..batch.num_rows() {
                if col.is_null(i) {
                    continue;
                }
                if arrow_cell_matches_json(col, i, value) {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    #[tracing::instrument(skip(self, filter))]
    pub async fn get_scalar_filter_bitmap(
        &self,
        filter: &crate::core::planner::QueryFilter,
    ) -> Result<Option<RoaringBitmap>> {
        Ok(self
            .get_scalar_filter_bitmap_with_exactness(filter)
            .await?
            .map(|(bm, _)| bm))
    }

    #[tracing::instrument(skip(self, filter))]
    pub(crate) async fn get_scalar_filter_bitmap_with_exactness(
        &self,
        filter: &crate::core::planner::QueryFilter,
    ) -> Result<Option<(RoaringBitmap, bool)>> {
        let filter_column = &filter.column;

        // Lexical (parquet) indexes: canonical `lexical`, legacy `inverted`, or
        // tokenized `bm25`. A `scalar` index is a roaring bitmap (`.idx`),
        // handled separately below.
        let inv_idx_info = self.config.index_files.iter().find(|f| {
            f.is_lexical()
                && f.algorithm != "doclen"
                && f.column_name == *filter_column
                && self.config.is_index_valid(f)
        });
        let bitmap_idx_info = self.config.index_files.iter().find(|f| {
            f.index_category == "scalar"
                && f.column_name == *filter_column
                && self.config.is_index_valid(f)
        });

        let (matching_bitmap, mut key_check_exact) = if let Some(idx_info) = inv_idx_info {
            let full_inv_path_str = self.resolve_index_path(&idx_info.file_path);

            let cache_key = if let Some(offset) = idx_info.blob_offset {
                format!("{}/{}:{}", self.cache_root(), full_inv_path_str, offset)
            } else {
                format!("{}/{}", self.cache_root(), full_inv_path_str)
            };
            let source_type_key = format!("{}:source_type", cache_key);

            let inv_path = Path::from(full_inv_path_str.as_str());
            let load_inv_bytes = || async {
                match crate::core::cache::BYTE_CACHE
                    .get_with_metrics(&cache_key, "byte")
                    .await
                {
                    Some(cached) => Ok::<Vec<u8>, anyhow::Error>(cached.as_ref().clone()),
                    None => {
                        let bytes = if let (Some(offset), Some(length)) =
                            (idx_info.blob_offset, idx_info.blob_length)
                        {
                            let b = self
                                .store
                                .get_range(
                                    &inv_path,
                                    (offset as u64)..(offset as u64 + length as u64),
                                )
                                .await?;
                            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                            b.to_vec()
                        } else {
                            let res = self.store.get(&inv_path).await?;
                            let b = res.bytes().await?;
                            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                            b.to_vec()
                        };

                        crate::core::cache::BYTE_CACHE
                            .insert(cache_key.clone(), Arc::new(bytes.clone()))
                            .await;
                        Ok(bytes)
                    }
                }
            };

            let extract_inv_metadata =
                |metadata: &parquet::file::metadata::ParquetMetaData| -> (String, String) {
                    let mut analyzer_val: Option<String> = None;
                    let mut source_type_val: Option<String> = None;
                    if let Some(kv_list) = metadata.file_metadata().key_value_metadata() {
                        for kv in kv_list {
                            if kv.key == "analyzer" {
                                analyzer_val = kv.value.clone();
                            } else if kv.key == "source_type" {
                                source_type_val = kv.value.clone();
                            }
                        }
                    }
                    let analyzer = analyzer_val.unwrap_or_else(|| {
                        if idx_info.algorithm == "bm25" {
                            "analyzer:english".to_string()
                        } else {
                            "identity".to_string()
                        }
                    });
                    let source_type = source_type_val.unwrap_or_default();
                    (analyzer, source_type)
                };

            let batches = if let Some(cached) = crate::core::cache::INVERTED_INDEX_CACHE
                .get_with_metrics(&cache_key, "inverted_index")
                .await
            {
                // Repopulate analyzer/source_type metadata if ANALYZER_META_CACHE evicted independently.
                if crate::core::cache::ANALYZER_META_CACHE
                    .get(&cache_key)
                    .await
                    .is_none()
                    || crate::core::cache::ANALYZER_META_CACHE
                        .get(&source_type_key)
                        .await
                        .is_none()
                {
                    if let Ok(inv_bytes) = load_inv_bytes().await {
                        if let Ok(builder) =
                            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                                Bytes::from(inv_bytes),
                            )
                        {
                            let (analyzer, source_type) = extract_inv_metadata(builder.metadata());
                            crate::core::cache::ANALYZER_META_CACHE
                                .insert(cache_key.clone(), analyzer)
                                .await;
                            crate::core::cache::ANALYZER_META_CACHE
                                .insert(source_type_key.clone(), source_type)
                                .await;
                        }
                    }
                }
                cached.as_ref().clone()
            } else {
                let inv_bytes = match load_inv_bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        if e.to_string().contains("not found") || e.to_string().contains("404") {
                            return Ok(None);
                        }
                        return Err(e);
                    }
                };

                let builder =
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                        Bytes::from(inv_bytes),
                    )?;

                let (analyzer, source_type) = extract_inv_metadata(builder.metadata());
                crate::core::cache::ANALYZER_META_CACHE
                    .insert(cache_key.clone(), analyzer)
                    .await;
                crate::core::cache::ANALYZER_META_CACHE
                    .insert(source_type_key.clone(), source_type)
                    .await;

                let reader = builder.build()?;
                let mut decoded = Vec::new();
                for batch_result in reader {
                    decoded.push(batch_result?);
                }

                crate::core::cache::INVERTED_INDEX_CACHE
                    .insert(cache_key.clone(), Arc::new(decoded.clone()))
                    .await;
                decoded
            };

            let analyzer = crate::core::cache::ANALYZER_META_CACHE
                .get(&cache_key)
                .await
                .unwrap_or_else(|| {
                    if idx_info.algorithm == "bm25" {
                        "analyzer:english".to_string()
                    } else {
                        "identity".to_string()
                    }
                });
            let is_exact_filter =
                filter.values.is_some() || filter.min.is_some() || filter.max.is_some();
            if is_exact_filter && analyzer != "identity" {
                return Ok(None);
            }

            let source_type = crate::core::cache::ANALYZER_META_CACHE
                .get(&source_type_key)
                .await
                .unwrap_or_default();
            let is_timestamp_day = source_type == "timestamp";
            let mut key_check_exact = analyzer == "identity" && !is_timestamp_day;

            let mut bitmap = RoaringBitmap::new();

            // The inverted index schema is [key, row_ids (List<UInt32>)]
            for batch in batches {
                let key_array = batch.column(0);
                let row_ids_list = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::ListArray>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Expected ListArray in inverted index column 1")
                    })?;

                for i in 0..batch.num_rows() {
                    if key_array.is_null(i) {
                        continue;
                    }
                    let key_ok = evaluate_inverted_key(
                        key_array,
                        i,
                        filter,
                        is_timestamp_day,
                        &mut key_check_exact,
                    )?;

                    if key_ok {
                        let row_ids = row_ids_list.value(i);
                        let row_ids_array = row_ids
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("Expected UInt32Array in inverted index row_ids")
                            })?;

                        let mut current_id = 0;
                        for ri in 0..row_ids_array.len() {
                            current_id += row_ids_array.value(ri);
                            bitmap.insert(current_id);
                        }
                    }
                }
            }
            (bitmap, key_check_exact)
        } else {
            // Legacy/fallback scalar bitmap (`.idx`) has no per-key predicate filtering.

            let idx_path = match bitmap_idx_info {
                Some(info) => {
                    let p = self.resolve_index_path(&info.file_path);
                    Path::from(p.as_str())
                }
                None => self.resolve_object_path(&format!("{}.idx", filter_column)),
            };
            let idx_path_str = idx_path.to_string();

            let cache_key = if let Some(info) = bitmap_idx_info {
                if let Some(offset) = info.blob_offset {
                    format!("{}/{}:{}", self.cache_root(), idx_path_str, offset)
                } else {
                    format!("{}/{}", self.cache_root(), idx_path_str)
                }
            } else {
                format!("{}/{}", self.cache_root(), idx_path_str)
            };

            if let Some(cached) = crate::core::cache::INDEX_CACHE
                .get_with_metrics(&cache_key, "index")
                .await
            {
                (cached.as_ref().clone(), false)
            } else {
                let bytes_res = if let Some(info) = bitmap_idx_info {
                    if let (Some(offset), Some(length)) = (info.blob_offset, info.blob_length) {
                        self.store
                            .get_range(&idx_path, (offset as u64)..(offset as u64 + length as u64))
                            .await
                    } else {
                        match self.store.get(&idx_path).await {
                            Ok(resp) => resp.bytes().await,
                            Err(e) => Err(e),
                        }
                    }
                } else {
                    match self.store.get(&idx_path).await {
                        Ok(resp) => resp.bytes().await,
                        Err(e) => Err(e),
                    }
                };

                match bytes_res {
                    Ok(index_bytes) => {
                        crate::telemetry::metrics::IO_BYTES_READ_TOTAL
                            .inc_by(index_bytes.len() as u64);
                        let bitmap = RoaringBitmap::deserialize_from(&index_bytes[..])?;
                        crate::core::cache::INDEX_CACHE
                            .insert(cache_key, Arc::new(bitmap.clone()))
                            .await;
                        (bitmap, false)
                    }
                    Err(_) => {
                        return Ok(None);
                    }
                }
            }
        };

        // Step 1b: Handle Negation
        let mut final_bitmap = matching_bitmap;
        if filter.negated {
            if !key_check_exact {
                // Complementing an approximate/superset bitmap would omit matching rows.
                return Ok(None);
            }
            if let Some(total) = self.config.record_count {
                let mut all = roaring::RoaringBitmap::new();
                all.insert_range(0..total as u32);
                final_bitmap = all - final_bitmap;
                // NULL rows are absent from inverted indexes, so `all - final_bitmap`
                // includes them; caller must evaluate the residual FilterExpr.
                key_check_exact = false;
            } else {
                return Ok(None);
            }
        }

        // Step 1c: Apply Deletes (Difference)
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
            final_bitmap -= deleted;
        }
        let eq_deletes = self.load_equality_deletes().await?;
        let is_exact = key_check_exact && eq_deletes.is_empty();

        Ok(Some((final_bitmap, is_exact)))
    }

    #[tracing::instrument(skip(self, filter, target_schema))]
    pub async fn query_index_first(
        &self,
        filter: &crate::core::planner::QueryFilter,
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        let matching_bitmap = match self.get_scalar_filter_bitmap(filter).await? {
            Some(bm) => bm,
            None => return Err(anyhow::anyhow!("No index for column {}", filter.column)),
        };
        self.read_rows_by_bitmap(&matching_bitmap, target_schema)
            .await
    }

    /// Load the `json_path` overlay for `column` and return the row bitmap for a
    /// `(path, value)` lookup, or every row where `path` resolves when `value`
    /// is `None`. Returns `None` when the segment has no `json_path` index for
    /// the column, or when the index does not cover `path` — so the caller
    /// falls back to a full scan rather than trusting a subset bitmap.
    pub async fn get_json_path_bitmap(
        &self,
        column: &str,
        path: &str,
        value: Option<&str>,
    ) -> Result<Option<RoaringBitmap>> {
        let idx_info = self.config.index_files.iter().find(|f| {
            f.index_category == "json_path"
                && f.column_name == column
                && self.config.is_index_valid(f)
        });
        let Some(idx_info) = idx_info else {
            return Ok(None);
        };

        let jp_path_resolved = self.resolve_index_path(&idx_info.file_path);
        let jp_path = Path::from(jp_path_resolved.as_str());
        let jp_path_str = jp_path.to_string();
        let cache_key = if let Some(offset) = idx_info.blob_offset {
            format!("{}/{}:{}", self.cache_root(), jp_path_str, offset)
        } else {
            format!("{}/{}", self.cache_root(), jp_path_str)
        };

        let bytes = if let Some(cached) = crate::core::cache::BYTE_CACHE
            .get_with_metrics(&cache_key, "byte")
            .await
        {
            cached.as_ref().clone()
        } else {
            let b = if let (Some(offset), Some(length)) =
                (idx_info.blob_offset, idx_info.blob_length)
            {
                self.store
                    .get_range(&jp_path, (offset as u64)..(offset as u64 + length as u64))
                    .await?
                    .to_vec()
            } else {
                match self.store.get(&jp_path).await {
                    Ok(res) => res.bytes().await?.to_vec(),
                    Err(e) => {
                        if e.to_string().contains("not found") || e.to_string().contains("404") {
                            return Ok(None);
                        }
                        return Err(e.into());
                    }
                }
            };
            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
            crate::core::cache::BYTE_CACHE
                .insert(cache_key, Arc::new(b.clone()))
                .await;
            b
        };

        // The index only covers the paths it was built for; a query on any
        // other path must not use it (an empty bitmap would be a subset).
        let covered = crate::core::index::json_path::indexed_paths(&bytes)?;
        if !covered.iter().any(|p| p == path) {
            return Ok(None);
        }

        let mut bitmap = match value {
            Some(v) => crate::core::index::json_path::load_json_path_bitmap(&bytes, path, v)?,
            None => crate::core::index::json_path::load_json_path_exists_bitmap(&bytes, path)?,
        };

        // Apply deletes (difference), matching the scalar/inverted path.
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
            bitmap -= deleted;
        }
        Ok(Some(bitmap))
    }

    /// Answer a set of JSON-path predicates from the `json_path` overlay.
    ///
    /// Returns `Ok(None)` when any predicate has no usable index, so the caller
    /// falls back to a full scan. The returned rows are a superset of the true
    /// matches; the caller re-applies the filter.
    pub async fn query_json_path_first(
        &self,
        predicates: &[crate::core::planner::JsonPathPredicate],
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<Option<Vec<arrow::record_batch::RecordBatch>>> {
        use crate::core::planner::JsonPathOp;
        let mut combined: Option<RoaringBitmap> = None;
        for p in predicates {
            let bm = match p.op {
                // `json_exists` matches a top-level object key *or* an array
                // element, so union the `$.key` presence lookup with the root
                // array-element lookup.
                JsonPathOp::Exists => {
                    let key = p.path.strip_prefix("$.").unwrap_or(&p.path);
                    let a = self.get_json_path_bitmap(&p.column, &p.path, None).await?;
                    let b = self.get_json_path_bitmap(&p.column, "$", Some(key)).await?;
                    match (a, b) {
                        (Some(a), Some(b)) => Some(a | b),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    }
                }
                _ => {
                    self.get_json_path_bitmap(&p.column, &p.path, p.value.as_deref())
                        .await?
                }
            };
            match bm {
                Some(bm) => {
                    combined = Some(match combined {
                        Some(acc) => acc & bm,
                        None => bm,
                    });
                }
                None => return Ok(None),
            }
        }
        match combined {
            Some(bm) => Ok(Some(self.read_rows_by_bitmap(&bm, target_schema).await?)),
            None => Ok(None),
        }
    }

    /// Materialize the rows selected by `matching_bitmap` from the segment's
    /// Parquet file, applying equality deletes and schema evolution to `target_schema`.
    pub(crate) async fn read_rows_by_bitmap(
        &self,
        matching_bitmap: &RoaringBitmap,
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        if matching_bitmap.is_empty() {
            return Ok(vec![]);
        }

        let equality_deletes = self.load_equality_deletes().await?;
        let data_store = self.data_store().clone();

        // Step 2: Configure Parquet Range Request with Row Selection
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();
        let meta_cache_key = format!("{}/{}", self.cache_root(), pq_path_str);

        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE
            .get_with_metrics(&meta_cache_key, "parquet_meta")
            .await
        {
            // Cache Hit
            let object_meta = ObjectMeta {
                location: pq_path.clone(),
                last_modified: Utc::now(),
                size: size as u64,
                e_tag: None,
                version: None,
            };
            let reader = ParquetObjectReader::new(data_store.clone(), object_meta.location);

            let arrow_meta = (*meta).clone();
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta)
        } else {
            let head_res = if let Some(s) = self.config.file_size {
                Ok((None, s as usize))
            } else {
                data_store.head(&pq_path).await.map(|m| {
                    let s = m.size as usize;
                    (Some(m), s)
                })
            };

            let (m_opt, size) = match head_res {
                Ok(pair) => pair,
                Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
                    return Ok(vec![]); // Segment missing
                }
                Err(e) => return Err(e.into()),
            };

            let location = if let Some(m) = &m_opt {
                m.location.clone()
            } else {
                pq_path.clone()
            };
            let mut reader = ParquetObjectReader::new(data_store.clone(), location.clone());
            let options = ArrowReaderOptions::new().with_page_index(true);
            let arrow_meta = match ArrowReaderMetadata::load_async(&mut reader, options).await {
                Ok(m) => m,
                Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
                    return Ok(vec![]);
                }
                Err(e) => return Err(e.into()),
            };
            let meta = Arc::new(arrow_meta);
            crate::core::cache::PARQUET_META_CACHE
                .insert(meta_cache_key, (meta.clone(), size))
                .await;
            let reader = ParquetObjectReader::new(data_store, location);
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader, (*meta).clone())
        };

        builder = builder.with_batch_size(65536);

        // Apply column projection/evolution (include equality-delete key columns so they
        // can be evaluated even when omitted from the caller's target projection).
        let target_schema_ref = target_schema.clone();
        if let Some(schema) = &target_schema_ref {
            let parquet_schema = builder.metadata().file_metadata().schema_descr();
            let file_arrow_schema = builder.schema();
            let mut column_indices: Vec<usize> = schema
                .fields()
                .iter()
                .filter_map(|field| file_arrow_schema.index_of(field.name()).ok())
                .collect();
            for del in &equality_deletes {
                if let Ok(idx) = file_arrow_schema.index_of(&del.column_name) {
                    if !column_indices.contains(&idx) {
                        column_indices.push(idx);
                    }
                }
            }

            let projection = ProjectionMask::roots(parquet_schema, column_indices);
            builder = builder.with_projection(projection);
        }

        // Construct RowSelection from Bitmap, skipping untouched row groups.
        let groups = Self::row_groups_covering(builder.metadata(), matching_bitmap);
        let num_groups = builder.metadata().num_row_groups();
        if !groups.is_empty() && groups.len() < num_groups {
            let mut shifted = RoaringBitmap::new();
            let mut base: u32 = 0;
            for (rg, group_start) in &groups {
                let rows = builder.metadata().row_group(*rg).num_rows() as u32;
                for id in matching_bitmap.iter() {
                    let id = id as usize;
                    if id >= *group_start && id < *group_start + rows as usize {
                        shifted.insert(base + (id as u32 - *group_start as u32));
                    }
                }
                base += rows;
            }
            builder = builder.with_row_groups(groups.iter().map(|(rg, _)| *rg).collect());
            let selection = self.bitmap_to_row_selection(&shifted, base as usize);
            builder = builder.with_row_selection(selection);
        } else {
            let selection = self.bitmap_to_row_selection(
                matching_bitmap,
                builder.metadata().file_metadata().num_rows() as usize,
            );
            builder = builder.with_row_selection(selection);
        }

        let mut stream = builder.build()?;

        let mut batches = Vec::new();
        while let Some(batch_result) = stream.next().await {
            let mut batch = batch_result?;

            // Apply Equality Deletes BEFORE schema projection/evolution so delete key columns exist
            if !equality_deletes.is_empty() && batch.num_rows() > 0 {
                let mut keep_mask = arrow::array::BooleanArray::from(vec![true; batch.num_rows()]);
                for delete in &equality_deletes {
                    if let Some(col) = batch.column_by_name(&delete.column_name) {
                        let delete_mask = check_is_in(col, &delete.values)?;
                        let not_delete = arrow::compute::not(&delete_mask)?;
                        keep_mask = arrow::compute::and(&keep_mask, &not_delete)?;
                    }
                }
                batch = arrow::compute::filter_record_batch(&batch, &keep_mask)?;
                if batch.num_rows() == 0 {
                    continue;
                }
            }

            // Schema Evolution Mapping
            if let Some(target) = &target_schema_ref {
                if target.fields().is_empty() {
                    batches.push(arrow::record_batch::RecordBatch::try_new_with_options(
                        target.clone(),
                        vec![],
                        &arrow::record_batch::RecordBatchOptions::new()
                            .with_row_count(Some(batch.num_rows())),
                    )?);
                } else {
                    let mut new_columns = Vec::with_capacity(target.fields().len());
                    for field in target.fields() {
                        if let Some(col) = batch.column_by_name(field.name()) {
                            if col.data_type() != field.data_type() {
                                let casted = arrow::compute::cast(col, field.data_type())?;
                                new_columns.push(casted);
                            } else {
                                new_columns.push(col.clone());
                            }
                        } else {
                            if !field.is_nullable() && batch.num_rows() > 0 {
                                anyhow::bail!(
                                    "read: required column '{}' is missing from the source segment",
                                    field.name()
                                );
                            }
                            let null_arr =
                                arrow::array::new_null_array(field.data_type(), batch.num_rows());
                            new_columns.push(null_arr);
                        }
                    }
                    batches.push(arrow::record_batch::RecordBatch::try_new(
                        target.clone(),
                        new_columns,
                    )?);
                }
            } else {
                batches.push(batch);
            }
        }

        Ok(batches)
    }

    pub(crate) fn bitmap_to_row_selection(
        &self,
        bitmap: &RoaringBitmap,
        total_rows: usize,
    ) -> RowSelection {
        if bitmap.is_empty() {
            return RowSelection::from(vec![RowSelector::skip(total_rows)]);
        }

        let mut selectors = Vec::new();
        let mut last_idx = 0;

        // Strategy: Coalesce small gaps to avoid fragmented I/O
        // If a gap is smaller than this, we include the rows anyway (trusting the filter to drop them later)
        // BUG FIX: For correctness, we must be strict unless we filter post-read.
        // Set to 1 to ensure we only merge immediately adjacent rows.
        let gap_threshold = 1;

        let mut current_start: Option<usize> = None;
        let mut current_end: Option<usize> = None;

        for idx in bitmap.iter() {
            let idx = idx as usize;

            match current_start {
                None => {
                    current_start = Some(idx);
                    current_end = Some(idx);
                }
                Some(start) => {
                    let end = current_end.unwrap_or(start);
                    if idx <= end + gap_threshold {
                        // Coalesce
                        current_end = Some(idx);
                    } else {
                        // Flush previous range
                        if start > last_idx {
                            selectors.push(RowSelector::skip(start - last_idx));
                        }
                        selectors.push(RowSelector::select(end - start + 1));
                        last_idx = end + 1;

                        current_start = Some(idx);
                        current_end = Some(idx);
                    }
                }
            }
        }

        // Flush last range
        if let Some(start) = current_start {
            let end = current_end.unwrap_or(start);
            if start > last_idx {
                selectors.push(RowSelector::skip(start - last_idx));
            }
            selectors.push(RowSelector::select(end - start + 1));
            last_idx = end + 1;
        }

        if last_idx < total_rows {
            selectors.push(RowSelector::skip(total_rows - last_idx));
        }

        RowSelection::from(selectors)
    }

    /// Rewrites a list of QueryFilters, combining exact match filters on multiple columns
    /// into a single composite filter if a corresponding composite index exists.
    pub(crate) fn rewrite_composite_filters(
        &self,
        mut filters: Vec<crate::core::planner::QueryFilter>,
    ) -> Vec<crate::core::planner::QueryFilter> {
        let mut composite_indices = Vec::new();
        for idx in &self.config.index_files {
            let col_name = &idx.column_name;
            {
                if col_name.contains(',') {
                    let parts: Vec<String> = col_name.split(',').map(|s| s.to_string()).collect();
                    composite_indices.push((col_name.clone(), parts));
                }
            }
        }

        for (comp_name, parts) in composite_indices {
            let mut matched_indices = Vec::new();
            let mut all_matched = true;
            let mut values = Vec::new();

            for part in &parts {
                let mut found = false;
                for (i, f) in filters.iter().enumerate() {
                    if f.column == *part && f.min == f.max && f.min_inclusive && f.max_inclusive {
                        if let Some(ref min_val) = f.min {
                            matched_indices.push(i);
                            values.push(min_val.clone());
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    all_matched = false;
                    break;
                }
            }

            if all_matched && !parts.is_empty() {
                matched_indices.sort_unstable();
                for &i in matched_indices.iter().rev() {
                    filters.remove(i);
                }

                let mut composite_val_str = String::new();
                for (i, val) in values.iter().enumerate() {
                    if i > 0 {
                        composite_val_str.push('\0');
                    }
                    if let Some(s) = val.as_str() {
                        composite_val_str.push_str(s);
                    } else {
                        composite_val_str.push_str(&val.to_string());
                    }
                }

                filters.push(crate::core::planner::QueryFilter {
                    column: comp_name.clone(),
                    min: Some(serde_json::Value::String(composite_val_str.clone())),
                    min_inclusive: true,
                    max: Some(serde_json::Value::String(composite_val_str.clone())),
                    max_inclusive: true,
                    values: None,
                    negated: false,
                });
            }
        }
        filters
    }
}

fn json_to_exact_i128(value: &serde_json::Value) -> Option<i128> {
    value
        .as_i64()
        .map(|i| i as i128)
        .or_else(|| value.as_u64().map(|u| u as i128))
}

fn json_to_date32_day(value: &serde_json::Value) -> Option<i32> {
    if let Some(i) = value.as_i64() {
        return i32::try_from(i).ok();
    }
    if let Some(s) = value.as_str() {
        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            if let Some(epoch) = chrono::NaiveDate::from_ymd_opt(1970, 1, 1) {
                return i32::try_from((d - epoch).num_days()).ok();
            }
        }
    }
    None
}

fn json_to_ts_day(value: &serde_json::Value) -> Option<i32> {
    if let Some(us) = value.as_i64() {
        return i32::try_from(us.div_euclid(86_400_000_000)).ok();
    }
    if let Some(s) = value.as_str() {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
            return i32::try_from(dt.timestamp_micros().div_euclid(86_400_000_000)).ok();
        }
        for fmt in [
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S",
        ] {
            if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
                return i32::try_from(ndt.and_utc().timestamp_micros().div_euclid(86_400_000_000))
                    .ok();
            }
        }
        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            if let Some(epoch) = chrono::NaiveDate::from_ymd_opt(1970, 1, 1) {
                return i32::try_from((d - epoch).num_days()).ok();
            }
        }
    }
    None
}

fn parse_decimal_unscaled(s: &str, scale: i8) -> Option<i128> {
    if scale < 0 {
        return None;
    }
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (neg, rest) = if let Some(r) = s.strip_prefix('-') {
        (true, r)
    } else if let Some(r) = s.strip_prefix('+') {
        (false, r)
    } else {
        (false, s)
    };
    let mut parts = rest.split('.');
    let int_part = parts.next()?;
    let frac_part = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return None;
    }
    let target_scale = scale as usize;
    if frac_part.len() > target_scale {
        let extra = &frac_part[target_scale..];
        if extra.bytes().any(|b| b != b'0') {
            return None;
        }
    }
    let int_digits = if int_part.is_empty() { "0" } else { int_part };
    if !int_digits.bytes().all(|b| b.is_ascii_digit())
        || !frac_part.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let base: i128 = int_digits.parse().ok()?;
    let pow10 = 10_i128.checked_pow(scale as u32)?;
    let mut scaled = base.checked_mul(pow10)?;
    let frac_used = &frac_part[..frac_part.len().min(target_scale)];
    if !frac_used.is_empty() {
        let frac_val: i128 = frac_used.parse().ok()?;
        let pad = 10_i128.checked_pow((target_scale - frac_used.len()) as u32)?;
        scaled = scaled.checked_add(frac_val.checked_mul(pad)?)?;
    }
    if neg {
        scaled.checked_neg()
    } else {
        Some(scaled)
    }
}

fn json_to_decimal_i128(value: &serde_json::Value, scale: i8) -> Option<i128> {
    if scale < 0 {
        return None;
    }
    if let Some(i) = json_to_exact_i128(value) {
        let pow10 = 10_i128.checked_pow(scale as u32)?;
        return i.checked_mul(pow10);
    }
    if let Some(s) = value.as_str() {
        return parse_decimal_unscaled(s, scale);
    }
    None
}

fn arrow_cell_matches_json(col: &dyn Array, i: usize, value: &serde_json::Value) -> bool {
    use arrow::array::*;
    use arrow::datatypes::TimeUnit;

    if col.is_null(i) {
        return value.is_null();
    }

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return value.as_str() == Some(arr.value(i));
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        return value.as_str() == Some(arr.value(i));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
        return value.as_str() == Some(arr.value(i));
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int8Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int16Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt8Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt16Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return value.as_f64() == Some(arr.value(i) as f64);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return value.as_f64() == Some(arr.value(i));
    }
    if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
        return value.as_bool() == Some(arr.value(i));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Date32Array>() {
        return json_to_date32_day(value) == Some(arr.value(i));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Date64Array>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Decimal128Array>() {
        let scale = match col.data_type() {
            arrow::datatypes::DataType::Decimal128(_, s) => *s,
            _ => 0,
        };
        return json_to_decimal_i128(value, scale) == Some(arr.value(i));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Time32SecondArray>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Time32MillisecondArray>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Time64MicrosecondArray>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Time64NanosecondArray>() {
        return json_to_exact_i128(value) == Some(arr.value(i) as i128);
    }
    match col.data_type() {
        arrow::datatypes::DataType::Timestamp(unit, _) => {
            let cell_us = match unit {
                TimeUnit::Second => col
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .map(|a| a.value(i).saturating_mul(1_000_000)),
                TimeUnit::Millisecond => col
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .map(|a| a.value(i).saturating_mul(1_000)),
                TimeUnit::Microsecond => col
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .map(|a| a.value(i)),
                TimeUnit::Nanosecond => col
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .map(|a| a.value(i).div_euclid(1_000)),
            };
            if let Some(us) = cell_us {
                if let Some(target_us) = value.as_i64() {
                    return us == target_us;
                }
                if let Some(s) = value.as_str() {
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
                        return us == dt.timestamp_micros();
                    }
                    for fmt in [
                        "%Y-%m-%d %H:%M:%S%.f",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S%.f",
                        "%Y-%m-%dT%H:%M:%S",
                    ] {
                        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
                            return us == ndt.and_utc().timestamp_micros();
                        }
                    }
                }
            }
            false
        }
        arrow::datatypes::DataType::Binary => col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .is_some_and(|a| value.as_str().map(str::as_bytes) == Some(a.value(i))),
        arrow::datatypes::DataType::LargeBinary => col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .is_some_and(|a| value.as_str().map(str::as_bytes) == Some(a.value(i))),
        arrow::datatypes::DataType::FixedSizeBinary(_) => col
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .is_some_and(|a| value.as_str().map(str::as_bytes) == Some(a.value(i))),
        _ => false,
    }
}

fn evaluate_ord_filter<'a, T: Ord>(
    key: T,
    filter: &'a crate::core::planner::QueryFilter,
    key_check_exact: &mut bool,
    parse: impl Fn(&'a serde_json::Value) -> Option<T>,
) -> bool {
    if let Some(ref min_val) = filter.min {
        if let Some(bound) = parse(min_val) {
            if if filter.min_inclusive {
                key < bound
            } else {
                key <= bound
            } {
                return false;
            }
        } else {
            *key_check_exact = false;
        }
    }
    if let Some(ref max_val) = filter.max {
        if let Some(bound) = parse(max_val) {
            if if filter.max_inclusive {
                key > bound
            } else {
                key >= bound
            } {
                return false;
            }
        } else {
            *key_check_exact = false;
        }
    }
    if let Some(ref values) = filter.values {
        let mut matched = false;
        let mut any_unparsed = false;
        for v in values {
            match parse(v) {
                Some(ref val) => {
                    if key == *val {
                        matched = true;
                        break;
                    }
                }
                None => {
                    any_unparsed = true;
                }
            }
        }
        if !matched {
            if any_unparsed {
                *key_check_exact = false;
                return true;
            }
            return false;
        }
    }
    true
}

fn evaluate_inverted_key(
    key_array: &dyn Array,
    i: usize,
    filter: &crate::core::planner::QueryFilter,
    is_timestamp_day: bool,
    key_check_exact: &mut bool,
) -> Result<bool> {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    if let Some(arr) = key_array.as_any().downcast_ref::<StringArray>() {
        return Ok(evaluate_ord_filter(
            arr.value(i),
            filter,
            key_check_exact,
            |v| v.as_str(),
        ));
    }
    if let Some(arr) = key_array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(evaluate_ord_filter(
            arr.value(i),
            filter,
            key_check_exact,
            |v| v.as_str(),
        ));
    }
    if let Some(arr) = key_array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(evaluate_ord_filter(
            arr.value(i),
            filter,
            key_check_exact,
            |v| v.as_str(),
        ));
    }

    let int_key: Option<i128> = if let Some(arr) = key_array.as_any().downcast_ref::<Int8Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<Int16Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<Int32Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<Int64Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<UInt8Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<UInt16Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<UInt32Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<UInt64Array>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<Time32SecondArray>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<Time32MillisecondArray>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<Time64MicrosecondArray>() {
        Some(arr.value(i) as i128)
    } else if let Some(arr) = key_array.as_any().downcast_ref::<Time64NanosecondArray>() {
        Some(arr.value(i) as i128)
    } else {
        key_array
            .as_any()
            .downcast_ref::<Date64Array>()
            .map(|arr| arr.value(i) as i128)
    };
    if let Some(k) = int_key {
        return Ok(evaluate_ord_filter(
            k,
            filter,
            key_check_exact,
            json_to_exact_i128,
        ));
    }

    if let Some(arr) = key_array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(evaluate_ord_filter(
            arr.value(i),
            filter,
            key_check_exact,
            |v| v.as_bool(),
        ));
    }

    if let Some(arr) = key_array.as_any().downcast_ref::<Date32Array>() {
        let key_day = arr.value(i);
        if is_timestamp_day {
            *key_check_exact = false;
            if let Some(ref min_val) = filter.min {
                if let Some(min_day) = json_to_ts_day(min_val) {
                    if key_day < min_day {
                        return Ok(false);
                    }
                }
            }
            if let Some(ref max_val) = filter.max {
                if let Some(max_day) = json_to_ts_day(max_val) {
                    if key_day > max_day {
                        return Ok(false);
                    }
                }
            }
            if let Some(ref values) = filter.values {
                let mut matched = false;
                let mut any_unparsed = false;
                for v in values {
                    match json_to_ts_day(v) {
                        Some(d) => {
                            if key_day == d {
                                matched = true;
                                break;
                            }
                        }
                        None => any_unparsed = true,
                    }
                }
                if !matched && !any_unparsed {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        return Ok(evaluate_ord_filter(
            key_day,
            filter,
            key_check_exact,
            json_to_date32_day,
        ));
    }

    let float_key: Option<(f64, bool)> =
        if let Some(arr) = key_array.as_any().downcast_ref::<Float32Array>() {
            Some((arr.value(i) as f64, true))
        } else {
            key_array
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(|arr| (arr.value(i), false))
        };
    if let Some((kf, is_f32)) = float_key {
        if kf.is_nan() {
            *key_check_exact = false;
            return Ok(true);
        }
        let check_f32_exact = |vf: f64, exact: &mut bool| {
            if is_f32 && ((vf as f32) as f64 != vf) {
                *exact = false;
            }
        };
        if let Some(ref min_val) = filter.min {
            if let Some(bound) = min_val.as_f64() {
                check_f32_exact(bound, key_check_exact);
                if if filter.min_inclusive {
                    kf < bound
                } else {
                    kf <= bound
                } {
                    return Ok(false);
                }
            } else {
                *key_check_exact = false;
            }
        }
        if let Some(ref max_val) = filter.max {
            if let Some(bound) = max_val.as_f64() {
                check_f32_exact(bound, key_check_exact);
                if if filter.max_inclusive {
                    kf > bound
                } else {
                    kf >= bound
                } {
                    return Ok(false);
                }
            } else {
                *key_check_exact = false;
            }
        }
        if let Some(ref values) = filter.values {
            let mut matched = false;
            let mut any_unparsed = false;
            for v in values {
                match v.as_f64() {
                    Some(bound) => {
                        check_f32_exact(bound, key_check_exact);
                        if kf == bound {
                            matched = true;
                            break;
                        }
                    }
                    None => any_unparsed = true,
                }
            }
            if !matched {
                if any_unparsed {
                    *key_check_exact = false;
                    return Ok(true);
                }
                return Ok(false);
            }
        }
        return Ok(true);
    }

    if let Some(arr) = key_array.as_any().downcast_ref::<Decimal128Array>() {
        let scale = match key_array.data_type() {
            DataType::Decimal128(_, s) => *s,
            _ => -1,
        };
        if scale < 0 {
            *key_check_exact = false;
            return Ok(true);
        }
        return Ok(evaluate_ord_filter(
            arr.value(i),
            filter,
            key_check_exact,
            |v| json_to_decimal_i128(v, scale),
        ));
    }

    let binary_key: Option<&[u8]> =
        if let Some(arr) = key_array.as_any().downcast_ref::<BinaryArray>() {
            Some(arr.value(i))
        } else if let Some(arr) = key_array.as_any().downcast_ref::<LargeBinaryArray>() {
            Some(arr.value(i))
        } else {
            None
        };
    if let Some(kbytes) = binary_key {
        return Ok(evaluate_ord_filter(kbytes, filter, key_check_exact, |v| {
            v.as_str().map(str::as_bytes)
        }));
    }

    *key_check_exact = false;
    Ok(true)
}
