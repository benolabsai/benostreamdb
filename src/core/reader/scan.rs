// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::cache::CacheExt;
use crate::core::index::bm25::Bm25Params;
use std::collections::HashMap;
use std::sync::Arc;
// use std::collections::HashSet;
use crate::core::index::hnsw_ivf::HnswIvfIndex;
use crate::core::index::VectorMetric;
use crate::core::planner::FilterExpr;
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::StreamExt;
use object_store::{path::Path, ObjectMeta, ObjectStore};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::ProjectionMask;
use parquet::file::statistics::Statistics as ParquetStats;

use anyhow::{Context, Result};
use futures::stream::BoxStream;
use roaring::RoaringBitmap;

use super::*;

/// Maximum parquet file size (bytes) cached whole in memory.
///
/// Small segments are read repeatedly (a scan touches every segment, and a
/// workload issues several scans per step). Reading them through the object
/// store each time pays a per-range `spawn_blocking` + open. Caching the whole
/// file and serving ranges from memory removes that. Larger files are left to
/// `ParquetObjectReader`, which fetches only the needed column chunks.
const PARQUET_BYTES_CACHE_MAX_FILE: u64 = 4 * 1024 * 1024;

/// An `AsyncFileReader` that serves byte ranges from an in-memory buffer.
///
/// Used with `ParquetRecordBatchStreamBuilder::new_with_metadata`, so only
/// `get_bytes`/`get_byte_ranges` are exercised; `get_metadata` is never called
/// (the metadata is supplied by the caller from `PARQUET_META_CACHE`).
struct BytesReader {
    bytes: Bytes,
}

impl parquet::arrow::async_reader::AsyncFileReader for BytesReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let start = (range.start as usize).min(self.bytes.len());
        let end = (range.end as usize).min(self.bytes.len());
        let slice = if start <= end {
            self.bytes.slice(start..end)
        } else {
            Bytes::new()
        };
        Box::pin(async move { Ok(slice) })
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> futures::future::BoxFuture<
        'a,
        parquet::errors::Result<Arc<parquet::file::metadata::ParquetMetaData>>,
    > {
        Box::pin(async move {
            Err(parquet::errors::ParquetError::General(
                "BytesReader does not load metadata; it is supplied by the caller".to_string(),
            ))
        })
    }
}

impl HybridReader {
    /// Build a parquet reader for `path`. Small files are served from an
    /// in-memory cache (see `PARQUET_BYTES_CACHE`); larger files use the object
    /// store directly so only the needed column chunks are fetched.
    async fn parquet_reader(
        &self,
        store: &Arc<dyn ObjectStore>,
        path: &Path,
        size: u64,
    ) -> Box<dyn parquet::arrow::async_reader::AsyncFileReader> {
        if size <= PARQUET_BYTES_CACHE_MAX_FILE {
            let key = format!("{}/{}", self.cache_root(), path);
            if let Some(bytes) = crate::core::cache::PARQUET_BYTES_CACHE.get(&key).await {
                return Box::new(BytesReader { bytes });
            }
            // Retry briefly on EMFILE ("Too many open files"). Under heavy
            // concurrency (background compaction rewriting segments while reads
            // are saturated) the process can transiently exceed its
            // file-descriptor limit; a short backoff lets other readers release
            // their handles instead of failing the query with `os error 24`.
            let mut attempt = 0u32;
            let mut delay = std::time::Duration::from_millis(5);
            loop {
                match store.get(path).await {
                    Ok(res) => {
                        if let Ok(bytes) = res.bytes().await {
                            crate::core::cache::PARQUET_BYTES_CACHE
                                .insert(key, bytes.clone())
                                .await;
                            return Box::new(BytesReader { bytes });
                        }
                        break;
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        let emfile =
                            msg.contains("Too many open files") || msg.contains("os error 24");
                        if emfile && attempt < 6 {
                            attempt += 1;
                            tokio::time::sleep(delay).await;
                            delay = (delay * 2).min(std::time::Duration::from_millis(200));
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        Box::new(ParquetObjectReader::new(store.clone(), path.clone()))
    }

    pub(crate) fn apply_eq_deletes_keep_mask(
        batch: &RecordBatch,
        equality_deletes: &[EqualityDelete],
    ) -> Result<arrow::array::BooleanArray> {
        if equality_deletes.is_empty() || batch.num_rows() == 0 {
            return Ok(arrow::array::BooleanArray::from(vec![
                true;
                batch.num_rows()
            ]));
        }
        let mut keep_mask = arrow::array::BooleanArray::from(vec![true; batch.num_rows()]);
        for delete in equality_deletes {
            if let Some(col) = batch.column_by_name(&delete.column_name) {
                let delete_mask = check_is_in(col, &delete.values)?;
                let not_delete = arrow::compute::not(&delete_mask)?;
                keep_mask = arrow::compute::and(&keep_mask, &not_delete)?;
            }
        }
        Ok(keep_mask)
    }

    pub(crate) async fn pad_missing_schema_columns(
        &self,
        batch: RecordBatch,
        required_cols: &[String],
    ) -> Result<RecordBatch> {
        let missing: Vec<&String> = required_cols
            .iter()
            .filter(|c| batch.column_by_name(c).is_none())
            .collect();
        if missing.is_empty() {
            return Ok(batch);
        }
        let target_schema = self.get_arrow_schema().await?;
        let num_rows = batch.num_rows();
        let mut fields = batch.schema().fields().to_vec();
        let mut columns = batch.columns().to_vec();
        for col_name in missing {
            if fields.iter().any(|f| f.name() == col_name) {
                continue;
            }
            if let Ok(field) = target_schema.field_with_name(col_name) {
                if !field.is_nullable() && num_rows > 0 {
                    anyhow::bail!(
                        "read: required column '{}' is missing from segment '{}'",
                        field.name(),
                        self.config.segment_id
                    );
                }
                fields.push(Arc::new(field.clone()));
                columns.push(arrow::array::new_null_array(field.data_type(), num_rows));
            }
        }
        let padded = RecordBatch::try_new_with_options(
            Arc::new(arrow::datatypes::Schema::new(fields)),
            columns,
            &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )?;
        Ok(padded)
    }

    fn extract_vector_at(col: &dyn Array, i: usize, expected_dim: usize) -> Option<Vec<f32>> {
        if col.is_null(i) {
            return None;
        }
        let item: arrow::array::ArrayRef = match col.data_type() {
            arrow::datatypes::DataType::FixedSizeList(_, _) => {
                let list = col
                    .as_any()
                    .downcast_ref::<arrow::array::FixedSizeListArray>()?;
                if list.is_null(i) {
                    return None;
                }
                list.value(i)
            }
            arrow::datatypes::DataType::List(_) => {
                let list = col.as_any().downcast_ref::<arrow::array::ListArray>()?;
                if list.is_null(i) {
                    return None;
                }
                list.value(i)
            }
            arrow::datatypes::DataType::LargeList(_) => {
                let list = col
                    .as_any()
                    .downcast_ref::<arrow::array::LargeListArray>()?;
                if list.is_null(i) {
                    return None;
                }
                list.value(i)
            }
            _ => return None,
        };
        if item.len() != expected_dim {
            return None;
        }
        if let Some(floats) = item.as_any().downcast_ref::<arrow::array::Float32Array>() {
            Some(floats.values().to_vec())
        } else {
            item.as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|doubles| doubles.values().iter().map(|&d| d as f32).collect())
        }
    }

    fn compute_vector_distance_f32(v: &[f32], q: &[f32], metric: VectorMetric) -> f32 {
        match metric {
            VectorMetric::L2 => crate::core::index::distance::l2_distance(v, q),
            VectorMetric::Cosine => crate::core::index::distance::cosine_distance(v, q),
            VectorMetric::InnerProduct => -crate::core::index::distance::dot_product(v, q),
            VectorMetric::L1 => crate::core::index::distance::l1_distance(v, q),
            VectorMetric::Hamming => crate::core::index::distance::hamming_distance(v, q),
            VectorMetric::Jaccard => crate::core::index::distance::jaccard_distance(v, q),
        }
    }

    async fn load_cached_parquet_meta(
        &self,
        pq_path: &Path,
    ) -> Result<(Arc<ArrowReaderMetadata>, usize)> {
        let pq_path_str = pq_path.to_string();
        let cache_key = format!("{}/{}", self.cache_root(), pq_path_str);
        if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE
            .get_with_metrics(&cache_key, "parquet_meta")
            .await
        {
            Ok((meta, size))
        } else {
            let store = self.data_store();
            let object_meta = store
                .head(pq_path)
                .await
                .context("Failed to get segment metadata")?;
            let size = object_meta.size as usize;
            let mut reader = ParquetObjectReader::new(store.clone(), object_meta.location);
            let options = ArrowReaderOptions::new().with_page_index(true);
            let arrow_meta = ArrowReaderMetadata::load_async(&mut reader, options).await?;
            let meta = Arc::new(arrow_meta);
            crate::core::cache::PARQUET_META_CACHE
                .insert(cache_key, (meta.clone(), size))
                .await;
            Ok((meta, size))
        }
    }

    async fn validate_row_ids_against_filter_and_eq_deletes(
        &self,
        candidate_bitmap: &RoaringBitmap,
        filter: Option<&FilterExpr>,
        equality_deletes: &[EqualityDelete],
    ) -> Result<RoaringBitmap> {
        if candidate_bitmap.is_empty() || (filter.is_none() && equality_deletes.is_empty()) {
            return Ok(candidate_bitmap.clone());
        }
        let pq_path = self.resolve_object_path("parquet");
        let (meta, size) = self.load_cached_parquet_meta(&pq_path).await?;
        let total_rows = meta.metadata().file_metadata().num_rows() as usize;
        if total_rows == 0 {
            return Ok(RoaringBitmap::new());
        }
        let mut target_bitmap = candidate_bitmap.clone();
        target_bitmap &= RoaringBitmap::from_iter(0..(total_rows as u32));
        if target_bitmap.is_empty() {
            return Ok(RoaringBitmap::new());
        }

        let reader = self
            .parquet_reader(self.data_store(), &pq_path, size as u64)
            .await;
        let arrow_meta = (*meta).clone();
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta);

        let groups = Self::row_groups_covering(builder.metadata(), &target_bitmap);
        let num_groups = builder.metadata().num_row_groups();
        if !groups.is_empty() && groups.len() < num_groups {
            let mut shifted = RoaringBitmap::new();
            let mut base: u32 = 0;
            for (rg, group_start) in &groups {
                let rows = builder.metadata().row_group(*rg).num_rows() as u32;
                for id in target_bitmap.iter() {
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
            let selection = self.bitmap_to_row_selection(&target_bitmap, total_rows);
            builder = builder.with_row_selection(selection);
        }

        let mut required_cols: Vec<String> = Vec::new();
        for del in equality_deletes {
            if !required_cols.contains(&del.column_name) {
                required_cols.push(del.column_name.clone());
            }
        }
        let filter_cols = filter.map(|f| f.required_columns()).unwrap_or_default();
        for col in &filter_cols {
            if !required_cols.contains(col) {
                required_cols.push(col.clone());
            }
        }

        let parquet_schema = builder.metadata().file_metadata().schema_descr();
        let file_arrow_schema = builder.schema();
        let column_indices: Vec<usize> = required_cols
            .iter()
            .filter_map(|name| file_arrow_schema.index_of(name).ok())
            .collect();
        let projection = ProjectionMask::roots(parquet_schema, column_indices);
        builder = builder.with_projection(projection);

        let mut stream = builder.build()?;
        let row_ids: Vec<u32> = target_bitmap.iter().collect();
        let mut surviving = RoaringBitmap::new();
        let mut current_offset = 0usize;
        let planner = crate::core::planner::QueryPlanner::new();

        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let num_rows = batch.num_rows();
            let avail = row_ids.len().saturating_sub(current_offset);
            let rows_to_process = std::cmp::min(num_rows, avail);
            if rows_to_process == 0 {
                break;
            }
            let batch = if rows_to_process < num_rows {
                batch.slice(0, rows_to_process)
            } else {
                batch
            };

            let eq_keep = if equality_deletes.is_empty() {
                None
            } else {
                Some(Self::apply_eq_deletes_keep_mask(&batch, equality_deletes)?)
            };

            let filter_mask = if let Some(expr) = filter {
                let eval_batch = if filter_cols
                    .iter()
                    .any(|c| batch.column_by_name(c).is_none())
                {
                    self.pad_missing_schema_columns(batch, &filter_cols).await?
                } else {
                    batch
                };
                let coerced = expr.coerce_literals(&eval_batch.schema());
                Some(planner.evaluate_expr(&eval_batch, &coerced)?)
            } else {
                None
            };

            for i in 0..rows_to_process {
                if let Some(ref eq_k) = eq_keep {
                    if !eq_k.is_valid(i) || !eq_k.value(i) {
                        continue;
                    }
                }
                if let Some(ref m) = filter_mask {
                    if !m.is_valid(i) || !m.value(i) {
                        continue;
                    }
                }
                surviving.insert(row_ids[current_offset + i]);
            }
            current_offset += rows_to_process;
        }

        Ok(surviving)
    }

    async fn load_inverted_bytes(
        &self,
        full_inv_path_str: &str,
        idx_info: &crate::core::manifest::IndexFile,
    ) -> Result<Vec<u8>> {
        let inv_path = Path::from(full_inv_path_str);
        let bytes =
            if let (Some(offset), Some(length)) = (idx_info.blob_offset, idx_info.blob_length) {
                let b = self
                    .store
                    .get_range(&inv_path, (offset as u64)..(offset as u64 + length as u64))
                    .await?;
                crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                b.to_vec()
            } else {
                let res = self.store.get(&inv_path).await?;
                let b = res.bytes().await?;
                crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                b.to_vec()
            };
        Ok(bytes)
    }

    async fn resolve_scalar_filter_bitmap_with_exactness(
        &self,
        filter: Option<&FilterExpr>,
    ) -> (Option<RoaringBitmap>, bool) {
        let Some(expr) = filter else {
            return (None, true);
        };
        let sub_filters = self.rewrite_composite_filters(expr.extract_and_conditions());
        let mut combined_bitmap: Option<RoaringBitmap> = None;
        let mut all_exact = expr.is_fully_covered_by_and_filters() && !sub_filters.is_empty();

        for sub_f in &sub_filters {
            match self.get_scalar_filter_bitmap_with_exactness(sub_f).await {
                Ok(Some((bm, sub_exact))) => {
                    if !sub_exact {
                        all_exact = false;
                    }
                    match combined_bitmap {
                        Some(ref mut existing) => *existing &= bm,
                        None => combined_bitmap = Some(bm),
                    }
                    if combined_bitmap.as_ref().is_some_and(|bm| bm.is_empty()) {
                        return (Some(RoaringBitmap::new()), all_exact);
                    }
                }
                _ => {
                    all_exact = false;
                }
            }
        }
        (combined_bitmap, all_exact)
    }

    #[tracing::instrument(skip(self, target_schema))]
    pub async fn stream_all(
        &self,
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<BoxStream<'static, Result<arrow::record_batch::RecordBatch>>> {
        self.stream_row_groups(None, target_schema).await
    }

    /// Prune row groups using Parquet column statistics (min/max/null_count).
    ///
    /// This is the "Parquet Internal Pruning" stage of the Iceberg metadata pipeline:
    /// after manifest-level pruning, we use per-row-group stats to skip entire row groups
    /// before any vector computation. This can eliminate 90-99% of I/O when data is
    /// clustered by the filter columns.
    ///
    /// Returns a list of row group indices that might match the filter. If the filter
    /// references no columns with available stats, all row groups are returned.
    fn prune_row_groups(
        &self,
        metadata: &parquet::file::metadata::ParquetMetaData,
        filter: &FilterExpr,
    ) -> Vec<usize> {
        let schema = metadata.file_metadata().schema_descr();
        let num_rgs = metadata.num_row_groups();

        let conditions = filter.extract_and_conditions();
        if conditions.is_empty() {
            return (0..num_rgs).collect();
        }

        let mut surviving = Vec::with_capacity(num_rgs);

        for rg_idx in 0..num_rgs {
            let rg = metadata.row_group(rg_idx);
            let mut might_match = true;

            for cond in &conditions {
                if cond.negated {
                    continue;
                }
                let col_name = &cond.column;
                let col_pos = schema
                    .columns()
                    .iter()
                    .position(|c| c.name() == col_name.as_str());

                if let Some(pos) = col_pos {
                    let col_meta = rg.column(pos);
                    let stats = col_meta.statistics();

                    if let Some(s) = stats {
                        // If all rows are null, skip this row group
                        let null_count = s.null_count_opt().unwrap_or(0);
                        if null_count >= rg.num_rows() as u64 {
                            might_match = false;
                            break;
                        }

                        // Only prune if stats are available
                        if Self::has_stats(s) && !Self::row_group_might_match_condition(s, cond) {
                            might_match = false;
                            break;
                        }
                    }
                }
            }

            if might_match {
                surviving.push(rg_idx);
            }
        }

        surviving
    }

    /// Check whether a Statistics enum has usable min/max data.
    fn has_stats(stats: &ParquetStats) -> bool {
        match stats {
            ParquetStats::Int32(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Int64(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Float(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Double(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::ByteArray(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Boolean(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            _ => false,
        }
    }

    /// Extract min/max as f64 from numeric statistics variants.
    fn extract_min_max_as_f64(stats: &ParquetStats) -> Option<(f64, f64)> {
        match stats {
            ParquetStats::Int32(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min as f64, max as f64))
                } else {
                    None
                }
            }
            ParquetStats::Int64(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min as f64, max as f64))
                } else {
                    None
                }
            }
            ParquetStats::Float(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min as f64, max as f64))
                } else {
                    None
                }
            }
            ParquetStats::Double(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min, max))
                } else {
                    None
                }
            }
            ParquetStats::ByteArray(s) => {
                // Try to parse byte array as numeric string
                if let (Some(min_bytes), Some(max_bytes)) = (s.min_opt(), s.max_opt()) {
                    if let (Ok(min_f), Ok(max_f)) = (
                        std::str::from_utf8(min_bytes.as_ref()).ok()?.parse::<f64>(),
                        std::str::from_utf8(max_bytes.as_ref()).ok()?.parse::<f64>(),
                    ) {
                        return Some((min_f, max_f));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Extract min/max as strings from ByteArray statistics.
    fn extract_min_max_as_str(stats: &ParquetStats) -> Option<(String, String)> {
        if let ParquetStats::ByteArray(s) = stats {
            if let (Some(min_bytes), Some(max_bytes)) = (s.min_opt(), s.max_opt()) {
                let min_str = std::str::from_utf8(min_bytes.as_ref()).ok()?.to_string();
                let max_str = std::str::from_utf8(max_bytes.as_ref()).ok()?.to_string();
                return Some((min_str, max_str));
            }
        }
        None
    }

    /// Check if a row group's column statistics might overlap with a filter condition.
    fn row_group_might_match_condition(
        stats: &ParquetStats,
        filter: &crate::core::planner::QueryFilter,
    ) -> bool {
        // Try numeric comparison first
        if let Some((rg_min, rg_max)) = Self::extract_min_max_as_f64(stats) {
            // Check min bound: if row group's max < filter's min, prune
            if let Some(filter_min) = &filter.min {
                if let Some(fmin) = filter_min.as_f64() {
                    let too_small = if filter.min_inclusive {
                        rg_max < fmin
                    } else {
                        rg_max <= fmin
                    };
                    if too_small {
                        return false;
                    }
                }
            }

            // Check max bound: if row group's min > filter's max, prune
            if let Some(filter_max) = &filter.max {
                if let Some(fmax) = filter_max.as_f64() {
                    let too_large = if filter.max_inclusive {
                        rg_min > fmax
                    } else {
                        rg_min >= fmax
                    };
                    if too_large {
                        return false;
                    }
                }
            }

            // Check IN list
            if let Some(values) = &filter.values {
                let mut possible = false;
                for v in values {
                    if let Some(vf) = v.as_f64() {
                        let in_range = !(vf < rg_min || vf > rg_max);
                        if in_range {
                            possible = true;
                            break;
                        }
                    }
                }
                if !possible {
                    return false;
                }
            }

            return true;
        }

        // Fall back to string comparison
        if let Some((rg_min_str, rg_max_str)) = Self::extract_min_max_as_str(stats) {
            // Check min bound
            if let Some(filter_min) = &filter.min {
                if let Some(fmin) = filter_min.as_str() {
                    let cmp = rg_max_str.as_str().cmp(fmin);
                    let too_small = if filter.min_inclusive {
                        cmp == std::cmp::Ordering::Less
                    } else {
                        cmp != std::cmp::Ordering::Greater
                    };
                    if too_small {
                        return false;
                    }
                }
            }

            // Check max bound
            if let Some(filter_max) = &filter.max {
                if let Some(fmax) = filter_max.as_str() {
                    let cmp = rg_min_str.as_str().cmp(fmax);
                    let too_large = if filter.max_inclusive {
                        cmp == std::cmp::Ordering::Greater
                    } else {
                        cmp != std::cmp::Ordering::Less
                    };
                    if too_large {
                        return false;
                    }
                }
            }

            // Check IN list
            if let Some(values) = &filter.values {
                let mut possible = false;
                for v in values {
                    if let Some(vs) = v.as_str() {
                        let in_range = !(vs < rg_min_str.as_str() || vs > rg_max_str.as_str());
                        if in_range {
                            possible = true;
                            break;
                        }
                    }
                }
                if !possible {
                    return false;
                }
            }

            return true;
        }

        // No usable stats — cannot prune
        true
    }

    /// Vector search using flat scan with row-group pruning.
    async fn vector_search_flat_with_row_group_pruning(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        filter: &FilterExpr,
        metric: VectorMetric,
    ) -> Result<Vec<(usize, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let metadata = self.get_parquet_metadata().await?;
        let num_rgs = metadata.num_row_groups();

        let surviving_rgs = self.prune_row_groups(&metadata, filter);

        let pruned = num_rgs.saturating_sub(surviving_rgs.len());
        if pruned > 0 {
            tracing::info!(
                "Row-group pruning: skipped {}/{} row groups ({:.0}% reduction)",
                pruned,
                num_rgs,
                (pruned as f64 / num_rgs.max(1) as f64 * 100.0)
            );
        }

        if surviving_rgs.is_empty() {
            return Ok(Vec::new());
        }

        let deleted = self.load_merged_deletes().await?;
        let eq_deletes = self.load_equality_deletes().await?;
        let required_cols = filter.required_columns();

        let mut rg_matches: Vec<(usize, f32)> = Vec::new();

        let q_vec = match query {
            crate::core::index::VectorValue::Float32(v) => v,
            _ => anyhow::bail!("Flat search with row-group pruning only supports Float32 vectors"),
        };

        let planner = crate::core::planner::QueryPlanner::new();

        for rg_idx in &surviving_rgs {
            let mut offset = 0usize;
            for i in 0..*rg_idx {
                offset += metadata.row_group(i).num_rows() as usize;
            }

            let single_rg_stream = self.stream_raw_row_groups(Some(&[*rg_idx]), None).await?;
            let mut local_offset = offset;
            let mut rg_stream = Box::pin(single_rg_stream);

            while let Some(batch_res) = rg_stream.next().await {
                let batch = batch_res?;
                let rows = batch.num_rows();
                if rows == 0 {
                    continue;
                }

                let eq_keep = if eq_deletes.is_empty() {
                    None
                } else {
                    Some(Self::apply_eq_deletes_keep_mask(&batch, &eq_deletes)?)
                };

                let eval_batch = if required_cols
                    .iter()
                    .any(|c| batch.column_by_name(c).is_none())
                {
                    self.pad_missing_schema_columns(batch.clone(), &required_cols)
                        .await?
                } else {
                    batch.clone()
                };
                let coerced_filter = filter.coerce_literals(&eval_batch.schema());
                let mask = planner.evaluate_expr(&eval_batch, &coerced_filter)?;

                if let Some(col) = batch.column_by_name(column) {
                    for i in 0..rows {
                        let row_id = local_offset + i;
                        if deleted.contains(row_id as u32) {
                            continue;
                        }
                        if let Some(ref eq_k) = eq_keep {
                            if !eq_k.is_valid(i) || !eq_k.value(i) {
                                continue;
                            }
                        }
                        if !mask.is_valid(i) || !mask.value(i) {
                            continue;
                        }
                        let Some(v) = Self::extract_vector_at(col.as_ref(), i, q_vec.len()) else {
                            continue;
                        };
                        let dist = Self::compute_vector_distance_f32(&v, q_vec, metric);
                        rg_matches.push((row_id, dist));
                    }
                }
                local_offset += rows;
            }
        }

        rg_matches.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if rg_matches.len() > k {
            rg_matches.truncate(k);
        }

        Ok(rg_matches)
    }

    pub async fn stream_row_groups(
        &self,
        row_groups: Option<&[usize]>,
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<BoxStream<'static, Result<arrow::record_batch::RecordBatch>>> {
        self.stream_row_groups_inner(row_groups, target_schema, true)
            .await
    }

    /// Stream physical row groups without applying position or equality deletes
    /// so batch row offsets map 1:1 to physical Parquet row IDs (for index
    /// building, backfill, and delete scanning).
    pub async fn stream_raw_row_groups(
        &self,
        row_groups: Option<&[usize]>,
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<BoxStream<'static, Result<arrow::record_batch::RecordBatch>>> {
        self.stream_row_groups_inner(row_groups, target_schema, false)
            .await
    }

    async fn stream_row_groups_inner(
        &self,
        row_groups: Option<&[usize]>,
        target_schema: Option<arrow::datatypes::SchemaRef>,
        apply_deletes: bool,
    ) -> Result<BoxStream<'static, Result<arrow::record_batch::RecordBatch>>> {
        let t_total = std::time::Instant::now();
        let t_meta = std::time::Instant::now();
        let store = self
            .config
            .data_store
            .clone()
            .unwrap_or_else(|| self.store.clone());
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();

        let meta_cache_key = format!("{}/{}", self.cache_root(), pq_path_str);
        let cached_meta = crate::core::cache::PARQUET_META_CACHE
            .get_with_metrics(&meta_cache_key, "parquet_meta")
            .await;

        let (arrow_meta, size) = if let Some((meta, size)) = cached_meta {
            // Cache Hit
            crate::telemetry::metrics::PARQUET_META_CACHE_TOTAL
                .with_label_values(&["hit"])
                .inc();
            ((*meta).clone(), size as u64)
        } else {
            // Miss
            crate::telemetry::metrics::PARQUET_META_CACHE_TOTAL
                .with_label_values(&["miss"])
                .inc();
            // Ensure file exists/get meta (HEAD)
            let meta_res: Result<ObjectMeta, object_store::Error> = store.head(&pq_path).await;
            let object_meta = match meta_res {
                Ok(m) => m,
                Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
                    return Ok(futures::stream::empty().boxed());
                }
                Err(e) => return Err(e.into()),
            };

            let size = object_meta.size;
            let mut reader = ParquetObjectReader::new(store.clone(), object_meta.location);
            let options = ArrowReaderOptions::new().with_page_index(true);
            let arrow_meta = match ArrowReaderMetadata::load_async(&mut reader, options).await {
                Ok(m) => m,
                Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
                    return Ok(futures::stream::empty().boxed());
                }
                Err(e) => return Err(e.into()),
            };

            crate::core::cache::PARQUET_META_CACHE
                .insert(
                    meta_cache_key,
                    (Arc::new(arrow_meta.clone()), size as usize),
                )
                .await;
            (arrow_meta, size)
        };

        // Serve small files from the in-memory byte cache; larger files stream
        // only the needed column chunks from the object store.
        let reader = self.parquet_reader(&store, &pq_path, size).await;
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta);

        crate::telemetry::metrics::READ_PHASE_SECONDS
            .with_label_values(&["meta"])
            .observe(t_meta.elapsed().as_secs_f64());

        let t_del = std::time::Instant::now();

        // Apply Deletes when reading logical rows; skip for physical-offset raw scans.
        // Load equality deletes before building ProjectionMask so any delete-key
        // columns not present in `target_schema` are still projected and read.
        let equality_deletes = if apply_deletes {
            let deleted = self.load_merged_deletes().await?;
            if !deleted.is_empty() {
                let num_rows = builder.metadata().file_metadata().num_rows() as usize;
                let full_range = RoaringBitmap::from_iter(0..num_rows as u32);
                let valid = full_range - deleted;

                let selection = self.bitmap_to_row_selection(&valid, num_rows);
                builder = builder.with_row_selection(selection);
            }

            self.load_equality_deletes().await?
        } else {
            Vec::new()
        };

        crate::telemetry::metrics::READ_PHASE_SECONDS
            .with_label_values(&["deletes"])
            .observe(t_del.elapsed().as_secs_f64());
        let t_proj = std::time::Instant::now();

        // Apply Row Group Selection
        if let Some(rgs) = row_groups {
            builder = builder.with_row_groups(rgs.to_vec());
        }

        // Schema Evolution / Projection Logic
        // Determine which PHYSICAL columns to read based on target_schema + equality_deletes
        let target_schema_ref = target_schema.clone();

        if let Some(schema) = &target_schema_ref {
            let parquet_schema = builder.metadata().file_metadata().schema_descr();
            let file_arrow_schema = builder.schema();

            // Map logical columns to physical columns
            // Only read columns that exist in the physical file
            let mut column_indices: Vec<usize> = schema
                .fields()
                .iter()
                .filter_map(|field| file_arrow_schema.index_of(field.name()).ok())
                .collect();

            for delete in &equality_deletes {
                if let Ok(idx) = file_arrow_schema.index_of(&delete.column_name) {
                    if !column_indices.contains(&idx) {
                        column_indices.push(idx);
                    }
                }
            }

            // NOTE: If column_indices is empty, ProjectionMask::roots will result in a 0-column read
            // which is exactly what we want for count queries.
            let projection = ProjectionMask::roots(parquet_schema, column_indices);
            builder = builder.with_projection(projection);
        }

        crate::telemetry::metrics::READ_PHASE_SECONDS
            .with_label_values(&["projection"])
            .observe(t_proj.elapsed().as_secs_f64());
        let t_build = std::time::Instant::now();

        let stream = builder.build()?;

        crate::telemetry::metrics::READ_PHASE_SECONDS
            .with_label_values(&["build"])
            .observe(t_build.elapsed().as_secs_f64());

        // Wrap stream to apply Equality Deletes (on physical columns) and Schema Mapping
        let mapped_stream = stream.map(move |res| {
            let mut batch = res.map_err(anyhow::Error::from)?;

            // 1. Apply Equality Deletes (Anti-Join) before projection/evolution drops delete keys
            if !equality_deletes.is_empty() {
                let keep_mask =
                    HybridReader::apply_eq_deletes_keep_mask(&batch, &equality_deletes)?;
                batch = arrow::compute::filter_record_batch(&batch, &keep_mask)?;
            }

            // 2. Schema Evolution Mapping
            if let Some(target) = &target_schema_ref {
                let mut new_columns = Vec::new();
                for field in target.fields() {
                    if let Ok(col) = batch.column_by_name(field.name()).ok_or(()) {
                        if col.data_type() != field.data_type() {
                            let casted = arrow::compute::cast(col, field.data_type())?;
                            new_columns.push(casted);
                        } else {
                            new_columns.push(col.clone());
                        }
                    } else {
                        let null_arr =
                            arrow::array::new_null_array(field.data_type(), batch.num_rows());
                        // A missing column must not manufacture NULLs into a
                        // REQUIRED (non-nullable) field: `try_new` would fail
                        // with "declared as non-nullable but contains null
                        // values". This happens when the projection matched no
                        // physical columns (e.g. a 0-column read). Surface it as
                        // a clear error rather than a confusing construction
                        // failure downstream.
                        if !field.is_nullable() && batch.num_rows() > 0 {
                            anyhow::bail!(
                                "read: required column '{}' is missing from the source segment \
                                 (projection matched no physical columns)",
                                field.name()
                            );
                        }
                        new_columns.push(null_arr);
                    }
                }

                batch = if target.fields().is_empty() {
                    arrow::record_batch::RecordBatch::try_new_with_options(
                        target.clone(),
                        vec![],
                        &arrow::record_batch::RecordBatchOptions::new()
                            .with_row_count(Some(batch.num_rows())),
                    )?
                } else {
                    arrow::record_batch::RecordBatch::try_new(target.clone(), new_columns)?
                };
            }

            Ok(batch)
        });

        crate::telemetry::metrics::READ_PHASE_SECONDS
            .with_label_values(&["total"])
            .observe(t_total.elapsed().as_secs_f64());
        Ok(mapped_stream.boxed())
    }

    pub async fn vector_search_flat(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        allowed_bitmap: &Option<RoaringBitmap>,
        metric: VectorMetric,
    ) -> Result<Vec<(usize, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let deleted = Arc::new(self.load_merged_deletes().await?);
        let eq_deletes = self.load_equality_deletes().await?;
        let mut stream = self.stream_raw_row_groups(None, None).await?;
        // Streaming top-k, kept sorted ascending by distance. The current
        // worst (k-th best) distance doubles as the early-exit threshold for
        // L2 scans, so candidates that cannot enter the top-k are abandoned
        // mid-accumulation instead of being fully computed and discarded.
        let mut topk: Vec<(usize, f32)> = Vec::with_capacity(k);
        let mut current_row_offset = 0usize;

        let q_vec = match query {
            crate::core::index::VectorValue::Float32(v) => v.clone(),
            _ => anyhow::bail!("Flat search only supports Float32 vectors currently"),
        };

        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let rows = batch.num_rows();
            if rows == 0 {
                continue;
            }

            let eq_keep = if eq_deletes.is_empty() {
                None
            } else {
                Some(Self::apply_eq_deletes_keep_mask(&batch, &eq_deletes)?)
            };

            if let Some(col) = batch.column_by_name(column) {
                let col = col.clone();
                let q_vec_clone = q_vec.clone();
                let allowed_bm = allowed_bitmap.clone();
                let deleted_bm = deleted.clone();
                let offset = current_row_offset;

                let batch_matches = tokio::task::spawn_blocking(move || {
                    for i in 0..rows {
                        let row_id = offset + i;

                        if deleted_bm.contains(row_id as u32) {
                            continue;
                        }
                        if let Some(ref eq_k) = eq_keep {
                            if !eq_k.is_valid(i) || !eq_k.value(i) {
                                continue;
                            }
                        }

                        // Check filter (allowed_bitmap)
                        if let Some(ref bm) = allowed_bm {
                            if !bm.contains(row_id as u32) {
                                continue;
                            }
                        }

                        let Some(v) = Self::extract_vector_at(col.as_ref(), i, q_vec_clone.len())
                        else {
                            continue;
                        };

                        // The k-th best distance so far bounds how good a
                        // candidate must be to enter the top-k.
                        let threshold = if topk.len() >= k {
                            topk.last().map(|(_, d)| *d).unwrap_or(f32::INFINITY)
                        } else {
                            f32::INFINITY
                        };

                        let dist = match metric {
                            VectorMetric::L2 => {
                                match crate::core::index::distance::l2_distance_squared_early_exit(
                                    &v,
                                    &q_vec_clone,
                                    threshold,
                                ) {
                                    Some(d) => d,
                                    // Cannot beat the current k-th best; skip.
                                    None => continue,
                                }
                            }
                            other => Self::compute_vector_distance_f32(&v, &q_vec_clone, other),
                        };

                        // Insert into the bounded top-k (sorted ascending).
                        // `partition_point` keeps ties in insertion order,
                        // matching the previous stable-sort-then-truncate.
                        if topk.len() < k {
                            let pos = topk.partition_point(|(_, d)| *d <= dist);
                            topk.insert(pos, (row_id, dist));
                        } else if dist < topk.last().map(|(_, d)| *d).unwrap_or(f32::INFINITY) {
                            let pos = topk.partition_point(|(_, d)| *d <= dist);
                            topk.insert(pos, (row_id, dist));
                            topk.pop();
                        }
                    }
                    Ok::<Vec<(usize, f32)>, anyhow::Error>(topk)
                })
                .await
                .context("Blocking vector distance computation panicked")??;

                topk = batch_matches;
            }
            current_row_offset += rows;
        }

        if metric == VectorMetric::L2 {
            for (_, dist) in &mut topk {
                *dist = dist.max(0.0).sqrt();
            }
        }

        Ok(topk)
    }

    #[allow(clippy::too_many_arguments)]
    async fn search_hnsw_ivf_with_deletes(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        search_k: usize,
        metric: VectorMetric,
        ef_search: Option<usize>,
        use_mmap: bool,
        target_clusters: Option<&[usize]>,
        probes: Option<usize>,
        prefer_full_precision: bool,
    ) -> Result<(Vec<(usize, f32)>, bool)> {
        if k == 0 || search_k == 0 {
            return Ok((Vec::new(), false));
        }
        let deleted = self.load_merged_deletes().await?;
        let eq_deletes = self.load_equality_deletes().await?;

        let live_bitmap: Option<RoaringBitmap> = if !deleted.is_empty() {
            let total_rows = if let Some(rc) = self.config.record_count {
                rc as usize
            } else {
                self.get_parquet_metadata()
                    .await?
                    .file_metadata()
                    .num_rows() as usize
            };
            if total_rows == 0 {
                return Ok((Vec::new(), false));
            }
            let mut live = RoaringBitmap::new();
            live.insert_range(0..(total_rows as u32));
            live -= &deleted;
            if live.is_empty() {
                return Ok((Vec::new(), false));
            }
            Some(live)
        } else {
            None
        };

        // A Puffin compound bundle registers one `IndexFile` per blob
        // (centroids, graph, mapping), so the same logical vector index
        // appears as several entries. Deduplicate by (file_path, algorithm)
        // so each index is searched exactly once.
        let mut seen = std::collections::HashSet::new();
        let vector_indices: Vec<_> = self
            .config
            .index_files
            .iter()
            .filter(|f| {
                f.index_category == "vector"
                    && f.column_name == column
                    && self.config.is_index_valid(f)
                    && !matches!(
                        f.blob_type.as_deref(),
                        Some(crate::core::puffin::PUFFIN_BLOB_HNSW_IVF_CENTROIDS)
                            | Some(crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_MAPPING)
                            | Some(crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_GRAPH)
                            | Some("centroids")
                            | Some("mapping")
                            | Some("cluster_graph")
                    )
            })
            .filter(|f| seen.insert((f.file_path.clone(), f.algorithm.clone())))
            .cloned()
            .collect();

        let algo_rank = |f: &crate::core::manifest::IndexFile| -> u8 {
            if prefer_full_precision {
                match f.algorithm.as_str() {
                    "hnsw" | "hnsw_f32" => 0,
                    "hnsw_ivf" => 1,
                    "pq" | "hnsw_pq" => 2,
                    "tq8" | "hnsw_tq8" | "tq4" | "hnsw_tq4" | "sq8" | "hnsw_sq8" => 3,
                    _ => 4,
                }
            } else {
                match f.algorithm.as_str() {
                    "tq8" | "hnsw_tq8" | "tq4" | "hnsw_tq4" | "sq8" | "hnsw_sq8" => 0,
                    "pq" | "hnsw_pq" => 1,
                    "hnsw_ivf" => 2,
                    "hnsw" | "hnsw_f32" => 3,
                    _ => 4,
                }
            }
        };
        let best_rank = vector_indices.iter().map(algo_rank).min().unwrap_or(4);
        let chunk_indices: Vec<_> = vector_indices
            .iter()
            .filter(|f| algo_rank(f) == best_rank)
            .cloned()
            .collect();

        let hnsw_k = if !eq_deletes.is_empty() {
            search_k.saturating_mul(4).max(search_k.saturating_add(64))
        } else {
            search_k
        };

        let mut used_flat = false;
        let mut matches = if chunk_indices.is_empty() {
            let idx_path = self.resolve_object_path(column).to_string();
            let idx_info = crate::core::manifest::IndexFile {
                file_path: idx_path,
                index_category: "vector".to_string(),
                column_name: column.to_string(),
                ..Default::default()
            };
            match self
                .search_hnsw_ivf(
                    &idx_info,
                    query,
                    hnsw_k,
                    &live_bitmap,
                    metric,
                    ef_search,
                    use_mmap,
                    target_clusters,
                    probes,
                )
                .await
            {
                Ok(m) => m,
                Err(_) => {
                    used_flat = true;
                    self.vector_search_flat(column, query, k, &None, metric)
                        .await?
                }
            }
        } else {
            // Search ALL chunks for the preferred algorithm in parallel,
            // then merge the top-k results by row_id (chunks have
            // non-overlapping row_id ranges due to the build offset).
            let query_c = query.clone();
            let tc_owned = target_clusters.map(|tc| tc.to_vec());
            let live_ref = &live_bitmap;
            let futures: Vec<_> = chunk_indices
                .iter()
                .map(|idx_info| {
                    let idx = idx_info.clone();
                    let q = query_c.clone();
                    let tc = tc_owned.clone();
                    async move {
                        self.search_hnsw_ivf(
                            &idx,
                            &q,
                            hnsw_k,
                            live_ref,
                            metric,
                            ef_search,
                            use_mmap,
                            tc.as_deref(),
                            probes,
                        )
                        .await
                    }
                })
                .collect();
            let results = futures::future::join_all(futures).await;
            let mut merged: Vec<(usize, f32)> = Vec::new();
            let mut any_ok = false;
            for r in results {
                match r {
                    Ok(m) => {
                        any_ok = true;
                        merged.extend(m);
                    }
                    Err(e) => {
                        tracing::warn!("HNSW chunk search failed: {}", e);
                    }
                }
            }
            if !any_ok {
                used_flat = true;
                self.vector_search_flat(column, query, k, &None, metric)
                    .await?
            } else {
                merged
            }
        };

        if !used_flat && !eq_deletes.is_empty() && !matches.is_empty() {
            let mut cand_bm = RoaringBitmap::new();
            for (row_id, _) in &matches {
                cand_bm.insert(*row_id as u32);
            }
            let surviving = self
                .validate_row_ids_against_filter_and_eq_deletes(&cand_bm, None, &eq_deletes)
                .await?;
            matches.retain(|(id, _)| surviving.contains(*id as u32));
            if matches.len() < k && k <= 256 {
                used_flat = true;
                matches = self
                    .vector_search_flat(column, query, k, &None, metric)
                    .await?;
            }
        }

        matches.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok((matches, used_flat))
    }

    #[tracing::instrument(skip(self, query, filter, target_schema))]
    #[allow(clippy::too_many_arguments)]
    pub async fn vector_search_index(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        filter: Option<&FilterExpr>,
        metric: VectorMetric,
        ef_search: Option<usize>,
        target_schema: Option<arrow::datatypes::SchemaRef>,
        use_mmap: bool,
        target_clusters: Option<&[usize]>,
        refine_factor: Option<usize>,
        probes: Option<usize>,
    ) -> Result<Vec<(arrow::record_batch::RecordBatch, Vec<f32>)>> {
        if k == 0 {
            return Ok(vec![]);
        }
        // Resolve scalar filter to combined bitmap if present
        tracing::debug!("vector_search_index called with filter: {:?}", filter);
        let (allowed_bitmap, all_subfilters_exact) = self
            .resolve_scalar_filter_bitmap_with_exactness(filter)
            .await;
        if allowed_bitmap.as_ref().is_some_and(|bm| bm.is_empty()) {
            return Ok(vec![]);
        }

        // Handle keyword search regardless of filter path
        if let crate::core::index::VectorValue::Keyword(ref q) = query {
            let results = self
                .keyword_search_index(column, q, k, &Bm25Params::default(), None, filter)
                .await?;
            if results.is_empty() {
                return Ok(vec![]);
            }

            let matches: Vec<(u32, f32)> = results
                .iter()
                .map(|(id, s)| (*id as u32, 1.0 / (1.0 + s)))
                .collect();
            let mut matches = matches;
            matches.sort_by_key(|(id, _)| *id);
            let scores: Vec<f32> = matches.iter().map(|(_, d)| *d).collect();
            let batch = self
                .read_rows_by_id_with_schema(matches, target_schema)
                .await?;
            return Ok(vec![(batch, scores)]);
        }

        // --- TWO-STEP APPROACH when filter is present and resolvable ---
        // Step 1: Apply scalar filters as hard filters (Parquet row selection)
        // Step 2: Run vector search on the filtered resultset
        if let Some(ref bitmap) = allowed_bitmap {
            let residual_filter = if all_subfilters_exact { None } else { filter };
            return self
                .vector_search_on_filtered(
                    column,
                    query,
                    k,
                    bitmap,
                    metric,
                    target_schema,
                    residual_filter,
                )
                .await;
        }

        // --- FILTER PRESENT but no scalar index bitmap ---
        // Use row-group pruning to skip entire row groups based on Parquet column stats
        // before computing any vector distances.
        if let Some(filter_expr) = filter {
            let rg_matches = self
                .vector_search_flat_with_row_group_pruning(column, query, k, filter_expr, metric)
                .await?;
            return self
                .fetch_rows_with_distances(rg_matches, target_schema, None)
                .await;
        }

        // --- FAST PATH: no filter, use HNSW-IVF index ---
        let effective_refine = refine_factor.unwrap_or(1);
        let search_k = if effective_refine > 1 {
            k.saturating_mul(effective_refine)
        } else {
            k
        };
        let (mut matches, used_flat) = self
            .search_hnsw_ivf_with_deletes(
                column,
                query,
                k,
                search_k,
                metric,
                ef_search,
                use_mmap,
                target_clusters,
                probes,
                false,
            )
            .await?;

        let idx_matches = if !used_flat && effective_refine > 1 && !matches.is_empty() {
            matches.truncate(search_k);
            match self
                .rescore_candidates(column, matches.clone(), query, metric)
                .await
            {
                Ok(mut rescored) => {
                    rescored.truncate(k);
                    rescored
                }
                Err(e) => {
                    tracing::warn!(
                        "Candidate rescore failed, falling back to approximate distances: {}",
                        e
                    );
                    matches.truncate(k);
                    matches
                }
            }
        } else {
            matches.truncate(k);
            matches
        };

        self.fetch_rows_with_distances(idx_matches, target_schema, None)
            .await
    }

    /// Vector search optimized for RRF / scored-only lookup.
    /// Returns `(row_id, score)` without fetching Parquet rows.
    #[allow(clippy::too_many_arguments)]
    pub async fn vector_search_index_raw(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        metric: VectorMetric,
        ef_search: Option<usize>,
        use_mmap: bool,
        target_clusters: Option<&[usize]>,
        probes: Option<usize>,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(usize, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let (allowed_bitmap, all_subfilters_exact) = self
            .resolve_scalar_filter_bitmap_with_exactness(filter)
            .await;
        if allowed_bitmap.as_ref().is_some_and(|bm| bm.is_empty()) {
            return Ok(Vec::new());
        }

        if let Some(ref bitmap) = allowed_bitmap {
            let residual_filter = if all_subfilters_exact { None } else { filter };
            return self
                .vector_search_on_filtered_ids(column, query, k, bitmap, metric, residual_filter)
                .await;
        }

        if let Some(filter_expr) = filter {
            return self
                .vector_search_flat_with_row_group_pruning(column, query, k, filter_expr, metric)
                .await;
        }

        let (mut matches, _) = self
            .search_hnsw_ivf_with_deletes(
                column,
                query,
                k,
                k,
                metric,
                ef_search,
                use_mmap,
                target_clusters,
                probes,
                true,
            )
            .await?;
        matches.truncate(k);
        Ok(matches)
    }

    /// Search the inverted index with Okapi BM25 scoring.
    ///
    /// Returns `(row_id, raw_bm25_score)` pairs sorted by score descending
    /// (best first). Callers needing a distance-like metric convert with
    /// `1/(1+score)` at the boundary so row order stays aligned with scores.
    pub async fn keyword_search_index(
        &self,
        column: &str,
        query: &str,
        k: usize,
        params: &Bm25Params,
        analyzer: Option<&str>,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(usize, f32)>> {
        if k == 0 {
            return Ok(vec![]);
        }
        // Scalar pre-filter: resolve the predicate to a row-id bitmap before
        // scoring so filtered-out documents never enter the BM25 heap. Without
        // this the caller would have to post-filter the top-k list, which
        // silently returns fewer than k rows.
        let (allowed_bitmap, all_subfilters_exact) = self
            .resolve_scalar_filter_bitmap_with_exactness(filter)
            .await;
        if allowed_bitmap
            .as_ref()
            .is_some_and(|bitmap| bitmap.is_empty())
        {
            return Ok(vec![]);
        }

        // 1. Find Inverted Index
        tracing::info!(
            "keyword_search_index: Segment {} searching for inverted index on column '{}'",
            self.config.segment_id,
            column
        );
        tracing::info!("  Available index files: {:?}", self.config.index_files);

        let idx_info = self.config.index_files.iter()
            .find(|f| {
                let match_type = (f.algorithm == "inverted" || f.algorithm == "bm25")
                    || ((f.index_category == "inverted" || f.index_category == "bm25" || f.index_category == "lexical") && f.algorithm != "doclen");
                let match_col = f.column_name == column && self.config.is_index_valid(f);
                match_type && match_col
            })
            .ok_or_else(|| anyhow::anyhow!("No keyword/inverted index found for column '{}' in segment {} (Available: {:?})", column, self.config.segment_id, self.config.index_files))?;

        // 2. Load Inverted Index Batches
        let full_inv_path_str = self.resolve_index_path(&idx_info.file_path);

        let cache_key = if let Some(offset) = idx_info.blob_offset {
            format!("{}/{}:{}", self.cache_root(), full_inv_path_str, offset)
        } else {
            format!("{}/{}", self.cache_root(), full_inv_path_str)
        };

        let batches = if let Some(cached) = crate::core::cache::INVERTED_INDEX_CACHE
            .get_with_metrics(&cache_key, "inverted_index")
            .await
        {
            if analyzer.is_none()
                && crate::core::cache::ANALYZER_META_CACHE
                    .get(&cache_key)
                    .await
                    .is_none()
            {
                if let Ok(inv_bytes) = self.load_inverted_bytes(&full_inv_path_str, idx_info).await
                {
                    if let Ok(builder) =
                        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                            Bytes::from(inv_bytes),
                        )
                    {
                        if let Some(kv_list) =
                            builder.metadata().file_metadata().key_value_metadata()
                        {
                            for kv in kv_list {
                                if kv.key == "analyzer" {
                                    if let Some(value) = &kv.value {
                                        crate::core::cache::ANALYZER_META_CACHE
                                            .insert(cache_key.clone(), value.clone())
                                            .await;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            cached.as_ref().clone()
        } else {
            // Cache Miss - Load from Disk
            let inv_bytes = self
                .load_inverted_bytes(&full_inv_path_str, idx_info)
                .await?;

            let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                Bytes::from(inv_bytes),
            )?;
            // Record the analyzer name from the footer's key/value metadata so
            // repeated queries re-tokenize identically without re-reading.
            if let Some(kv_list) = builder.metadata().file_metadata().key_value_metadata() {
                for kv in kv_list {
                    if kv.key == "analyzer" {
                        if let Some(value) = &kv.value {
                            crate::core::cache::ANALYZER_META_CACHE
                                .insert(cache_key.clone(), value.clone())
                                .await;
                        }
                        break;
                    }
                }
            }
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

        // 3. Tokenize Query
        // Resolution order: caller override > footer metadata (cached) > the
        // writer's default analyzer (kept in sync with build_inverted.rs).
        let analyzer_name = if let Some(override_name) = analyzer {
            override_name.to_string()
        } else if let Some(cached) = crate::core::cache::ANALYZER_META_CACHE
            .get(&cache_key)
            .await
        {
            cached
        } else {
            "analyzer:english".to_string()
        };

        let tokenizer = crate::core::index::tokenizer::GLOBAL_TOKENIZER_REGISTRY
            .read()
            .get(&analyzer_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown analyzer '{}'", analyzer_name))?;

        let mut query_tokens = tokenizer.tokenize(query);
        // Dedupe: repeated query terms must not inflate BM25 term-frequency scoring
        query_tokens.sort_unstable();
        query_tokens.dedup();
        if query_tokens.is_empty() {
            return Ok(vec![]);
        }

        // 4. Doc-length sidecar (per-row token counts) written by finish_indexing.
        // Check if recorded as companion blob (Puffin) or standalone loose sidecar.
        let doclen_idx_info = self.config.index_files.iter().find(|f| {
            f.column_name == column && f.algorithm == "doclen" && self.config.is_index_valid(f)
        });

        let mut doc_lengths: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

        let doclen_bytes_opt: Option<bytes::Bytes> = if let Some(d_info) = doclen_idx_info {
            let full_d_path = self.resolve_index_path(&d_info.file_path);
            if let (Some(offset), Some(length)) = (d_info.blob_offset, d_info.blob_length) {
                self.store
                    .get_range(
                        &Path::from(full_d_path.as_str()),
                        (offset as u64)..(offset as u64 + length as u64),
                    )
                    .await
                    .ok()
            } else {
                match self.store.get(&Path::from(full_d_path.as_str())).await {
                    Ok(res) => res.bytes().await.ok(),
                    Err(_) => None,
                }
            }
        } else {
            let doclen_path_str = full_inv_path_str
                .strip_suffix(".inv.parquet")
                .map(|p| format!("{}.doclen.parquet", p))
                .unwrap_or_default();
            if !doclen_path_str.is_empty() {
                if let Ok(res) = self.store.get(&Path::from(doclen_path_str.as_str())).await {
                    res.bytes().await.ok()
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(doclen_bytes) = doclen_bytes_opt {
            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(doclen_bytes.len() as u64);
            if let Ok(dl_builder) =
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(doclen_bytes)
            {
                if let Ok(dl_reader) = dl_builder.build() {
                    for batch in dl_reader.filter_map(Result::ok) {
                        if let (Some(row_ids), Some(counts)) = (
                            batch
                                .column(0)
                                .as_any()
                                .downcast_ref::<arrow::array::UInt32Array>(),
                            batch
                                .column(1)
                                .as_any()
                                .downcast_ref::<arrow::array::UInt32Array>(),
                        ) {
                            for i in 0..batch.num_rows() {
                                doc_lengths.insert(row_ids.value(i), counts.value(i));
                            }
                        }
                    }
                }
            }
        }
        let mut n_docs = self.config.record_count.unwrap_or(0) as usize;
        if n_docs == 0 {
            n_docs = doc_lengths.len();
        }
        let avg_doc_len = if n_docs == 0 {
            0.0
        } else {
            doc_lengths.values().sum::<u32>() as f32 / n_docs as f32
        };

        // 5. Scoring Map: RowID -> BM25 Score
        let mut scores: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();

        for token in &query_tokens {
            // Aggregate per-document term frequencies for this term across all batches
            let mut current_doc_counts: std::collections::HashMap<u32, u32> =
                std::collections::HashMap::new();
            for batch in batches.iter() {
                let key_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Expected StringArray in inverted index keys")
                    })?;
                let row_ids_list = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::ListArray>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Expected ListArray in inverted index row_ids")
                    })?;

                let positions_list = if batch.num_columns() >= 3 {
                    batch
                        .column(2)
                        .as_any()
                        .downcast_ref::<arrow::array::ListArray>()
                } else {
                    None
                };

                for i in 0..batch.num_rows() {
                    let key = key_array.value(i);
                    if key == token.as_str() {
                        // Delta-decode row ids
                        let list = row_ids_list.value(i);
                        let row_ids = list
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>()
                            .context("Invalid cast")?;

                        if let Some(pos_list_col) = positions_list {
                            let docs_pos_list = pos_list_col.value(i);
                            let docs_pos = docs_pos_list
                                .as_any()
                                .downcast_ref::<arrow::array::ListArray>()
                                .context("Invalid cast")?;

                            let mut last_id = 0;
                            for j in 0..row_ids.len() {
                                let rid = last_id + row_ids.value(j);
                                last_id = rid;
                                // In the 3-column position-aware index, row_ids has unique doc IDs
                                // and docs_pos.value_length(j) gives the exact term frequency in O(1)!
                                let tf = docs_pos.value_length(j) as u32;
                                *current_doc_counts.entry(rid).or_default() += tf;
                            }
                        } else {
                            // Backward compatibility: 2-column format where duplicate row_ids encode tf
                            let mut last_id = 0;
                            for j in 0..row_ids.len() {
                                let rid = last_id + row_ids.value(j);
                                *current_doc_counts.entry(rid).or_default() += 1;
                                last_id = rid;
                            }
                        }
                    }
                }
            }
            if current_doc_counts.is_empty() {
                continue;
            }
            let df = current_doc_counts.len();
            for (rid, tf) in current_doc_counts {
                let doc_len = doc_lengths.get(&rid).copied().unwrap_or(0);
                let score_inc = crate::core::index::bm25::term_score(
                    tf as f32,
                    df,
                    n_docs,
                    doc_len,
                    avg_doc_len,
                    params,
                );
                *scores.entry(rid).or_default() += score_inc;
            }
        }

        // 6. Drop deleted documents and rows excluded by scalar/residual filters
        // or equality deletes BEFORE truncating to top-k so live rows are never
        // displaced by deleted/filtered candidates.
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
            scores.retain(|row_id, _| !deleted.contains(*row_id));
        }
        if let Some(bitmap) = &allowed_bitmap {
            scores.retain(|row_id, _| bitmap.contains(*row_id));
        }
        if !scores.is_empty() {
            let eq_deletes = self.load_equality_deletes().await?;
            let residual_filter = if filter.is_some() && !all_subfilters_exact {
                filter
            } else {
                None
            };
            if !eq_deletes.is_empty() || residual_filter.is_some() {
                let cand_bm: RoaringBitmap = scores.keys().copied().collect();
                let surviving = self
                    .validate_row_ids_against_filter_and_eq_deletes(
                        &cand_bm,
                        residual_filter,
                        &eq_deletes,
                    )
                    .await?;
                scores.retain(|row_id, _| surviving.contains(*row_id));
            }
        }

        let mut results: Vec<(usize, f32)> = scores
            .into_iter()
            .map(|(rid, score)| (rid as usize, score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if results.len() > k {
            results.truncate(k);
        }

        Ok(results)
    }

    /// Position-aware phrase query: searches for exact or slop-bounded phrase matches.
    /// Uses the 3-column inverted index (`positions` column).
    pub async fn phrase_search_index(
        &self,
        column: &str,
        phrase: &str,
        k: usize,
        slop: usize,
        analyzer: Option<&str>,
    ) -> Result<Vec<(usize, f32)>> {
        self.phrase_search_index_filtered(column, phrase, k, slop, analyzer, None)
            .await
    }

    /// Filtered position-aware phrase query applying scalar bitmaps, deletes,
    /// equality deletes, and non-exact residual filters before top-k truncation.
    pub async fn phrase_search_index_filtered(
        &self,
        column: &str,
        phrase: &str,
        k: usize,
        slop: usize,
        analyzer: Option<&str>,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(usize, f32)>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let (allowed_bitmap, all_subfilters_exact) = self
            .resolve_scalar_filter_bitmap_with_exactness(filter)
            .await;
        if allowed_bitmap
            .as_ref()
            .is_some_and(|bitmap| bitmap.is_empty())
        {
            return Ok(vec![]);
        }

        // 1. Find Inverted Index
        let idx_info = self
            .config
            .index_files
            .iter()
            .find(|f| {
                let match_type = (f.algorithm == "inverted" || f.algorithm == "bm25")
                    || ((f.index_category == "inverted"
                        || f.index_category == "bm25"
                        || f.index_category == "lexical")
                        && f.algorithm != "doclen");
                let match_col = f.column_name == column && self.config.is_index_valid(f);
                match_type && match_col
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No keyword/inverted index found for column '{}' in segment {}",
                    column,
                    self.config.segment_id
                )
            })?;

        // 2. Load Inverted Index Batches
        let full_inv_path_str = self.resolve_index_path(&idx_info.file_path);

        let cache_key = if let Some(offset) = idx_info.blob_offset {
            format!("{}/{}:{}", self.cache_root(), full_inv_path_str, offset)
        } else {
            format!("{}/{}", self.cache_root(), full_inv_path_str)
        };

        let batches = if let Some(cached) = crate::core::cache::INVERTED_INDEX_CACHE
            .get_with_metrics(&cache_key, "inverted_index")
            .await
        {
            if analyzer.is_none()
                && crate::core::cache::ANALYZER_META_CACHE
                    .get(&cache_key)
                    .await
                    .is_none()
            {
                if let Ok(inv_bytes) = self.load_inverted_bytes(&full_inv_path_str, idx_info).await
                {
                    if let Ok(builder) =
                        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                            Bytes::from(inv_bytes),
                        )
                    {
                        if let Some(kv_list) =
                            builder.metadata().file_metadata().key_value_metadata()
                        {
                            for kv in kv_list {
                                if kv.key == "analyzer" {
                                    if let Some(value) = &kv.value {
                                        crate::core::cache::ANALYZER_META_CACHE
                                            .insert(cache_key.clone(), value.clone())
                                            .await;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            cached.as_ref().clone()
        } else {
            let inv_bytes = self
                .load_inverted_bytes(&full_inv_path_str, idx_info)
                .await?;

            let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                Bytes::from(inv_bytes),
            )?;
            if let Some(kv_list) = builder.metadata().file_metadata().key_value_metadata() {
                for kv in kv_list {
                    if kv.key == "analyzer" {
                        if let Some(value) = &kv.value {
                            crate::core::cache::ANALYZER_META_CACHE
                                .insert(cache_key.clone(), value.clone())
                                .await;
                        }
                        break;
                    }
                }
            }
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

        // 3. Tokenize Query
        let analyzer_name = if let Some(override_name) = analyzer {
            override_name.to_string()
        } else if let Some(cached) = crate::core::cache::ANALYZER_META_CACHE
            .get(&cache_key)
            .await
        {
            cached
        } else {
            "analyzer:english".to_string()
        };

        let tokenizer = crate::core::index::tokenizer::GLOBAL_TOKENIZER_REGISTRY
            .read()
            .get(&analyzer_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown analyzer '{}'", analyzer_name))?;

        let phrase_tokens = tokenizer.tokenize(phrase);
        if phrase_tokens.is_empty() {
            return Ok(vec![]);
        }
        if phrase_tokens.len() == 1 {
            return self
                .keyword_search_index(column, phrase, k, &Bm25Params::default(), analyzer, filter)
                .await;
        }

        // 4. Retrieve positions per token: token -> doc_id -> Vec<u32> (positions)
        let mut token_doc_positions: Vec<HashMap<u32, Vec<u32>>> =
            Vec::with_capacity(phrase_tokens.len());

        for token in &phrase_tokens {
            let mut doc_map: HashMap<u32, Vec<u32>> = HashMap::new();
            for batch in &batches {
                let key_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| anyhow::anyhow!("Expected StringArray"))?;
                let row_ids_list = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::ListArray>()
                    .ok_or_else(|| anyhow::anyhow!("Expected ListArray"))?;
                let positions_list = if batch.num_columns() >= 3 {
                    batch
                        .column(2)
                        .as_any()
                        .downcast_ref::<arrow::array::ListArray>()
                } else {
                    None
                };

                let positions_list = match positions_list {
                    Some(pl) => pl,
                    None => {
                        return Err(anyhow::anyhow!(
                            "Phrase search requires position-aware inverted index"
                        ))
                    }
                };

                for i in 0..batch.num_rows() {
                    if key_array.value(i) == token.as_str() {
                        let list = row_ids_list.value(i);
                        let row_ids = list
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>()
                            .context("Invalid cast")?;
                        let docs_pos_list = positions_list.value(i);
                        let docs_pos = docs_pos_list
                            .as_any()
                            .downcast_ref::<arrow::array::ListArray>()
                            .context("Invalid cast")?;

                        let mut last_id = 0;
                        for j in 0..row_ids.len() {
                            let rid = last_id + row_ids.value(j);
                            last_id = rid;
                            let pos_arr = docs_pos.value(j);
                            let pos_u32 = pos_arr
                                .as_any()
                                .downcast_ref::<arrow::array::UInt32Array>()
                                .context("Invalid cast")?;
                            let positions_vec: Vec<u32> =
                                (0..pos_u32.len()).map(|p| pos_u32.value(p)).collect();
                            doc_map.entry(rid).or_default().extend(positions_vec);
                        }
                    }
                }
            }
            token_doc_positions.push(doc_map);
        }

        let deleted = self.load_merged_deletes().await?;
        let eq_deletes = self.load_equality_deletes().await?;

        // 5. Match phrases across candidate documents (docs present in all tokens)
        let first_docs = &token_doc_positions[0];
        let mut phrase_scores: Vec<(usize, f32)> = Vec::new();

        for (&doc_id, first_positions) in first_docs {
            if deleted.contains(doc_id) {
                continue;
            }
            if let Some(ref bm) = allowed_bitmap {
                if !bm.contains(doc_id) {
                    continue;
                }
            }
            let in_all = token_doc_positions[1..]
                .iter()
                .all(|m| m.contains_key(&doc_id));
            if !in_all {
                continue;
            }

            let mut matches_count = 0;
            for &start_pos in first_positions {
                let mut matched = true;
                let mut prev_pos = start_pos;
                for term_map in token_doc_positions[1..].iter() {
                    let next_positions = &term_map[&doc_id];
                    let expected_pos = prev_pos + 1;
                    let found = next_positions
                        .iter()
                        .any(|&p| p >= expected_pos && p <= expected_pos + slop as u32);
                    if found {
                        prev_pos = expected_pos;
                    } else {
                        matched = false;
                        break;
                    }
                }
                if matched {
                    matches_count += 1;
                }
            }

            if matches_count > 0 {
                phrase_scores.push((doc_id as usize, matches_count as f32));
            }
        }

        let residual_filter = if filter.is_some() && !all_subfilters_exact {
            filter
        } else {
            None
        };
        if !phrase_scores.is_empty() && (!eq_deletes.is_empty() || residual_filter.is_some()) {
            let mut cand_bm = RoaringBitmap::new();
            for &(doc_id, _) in &phrase_scores {
                cand_bm.insert(doc_id as u32);
            }
            let surviving = self
                .validate_row_ids_against_filter_and_eq_deletes(
                    &cand_bm,
                    residual_filter,
                    &eq_deletes,
                )
                .await?;
            phrase_scores.retain(|(doc_id, _)| surviving.contains(*doc_id as u32));
        }

        phrase_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if phrase_scores.len() > k {
            phrase_scores.truncate(k);
        }

        Ok(phrase_scores)
    }

    pub async fn read_rows_by_id(
        &self,
        matches: Vec<(u32, f32)>,
        columns: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        // Honour the projection. This parameter used to be ignored (`_columns`),
        // so a vector search projecting two columns still read *every* column
        // of the row from Parquet. Build a projected schema and let the scan
        // read only what was asked for; fall back to the full schema if any
        // requested name is unknown.
        let target_schema = match columns.filter(|c| !c.is_empty()) {
            Some(cols) => match self.get_arrow_schema().await {
                Ok(full) => {
                    let fields: Vec<_> = cols
                        .iter()
                        .filter_map(|c| full.field_with_name(c).ok().cloned())
                        .collect();
                    if fields.len() == cols.len() {
                        Some(Arc::new(arrow::datatypes::Schema::new(fields)))
                    } else {
                        None
                    }
                }
                Err(_) => None,
            },
            None => None,
        };

        self.read_rows_by_id_with_schema(matches, target_schema)
            .await
    }

    pub async fn read_rows_by_id_with_schema(
        &self,
        matches: Vec<(u32, f32)>,
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<RecordBatch> {
        let matches_usize: Vec<(usize, f32)> = matches
            .into_iter()
            .map(|(id, s)| (id as usize, s))
            .collect();
        let batches: Vec<(RecordBatch, Vec<f32>)> = self
            .fetch_rows_with_distances(matches_usize, target_schema, None)
            .await?;

        if batches.is_empty() {
            return Err(anyhow::anyhow!("No rows found for specified IDs"));
        }

        // Merge batches if necessary
        if batches.len() == 1 {
            Ok(batches[0].0.clone())
        } else {
            let record_batches: Vec<RecordBatch> = batches.into_iter().map(|(b, _)| b).collect();
            let schema = record_batches[0].schema();
            Ok(arrow::compute::concat_batches(&schema, &record_batches)?)
        }
    }

    /// Row groups that contain at least one id in `bitmap`, with each group's
    /// first global row id: `(row_group_index, first_row_id)`.
    ///
    /// Parquet cannot address a single row — a read must start at a row group.
    /// Restricting a kNN hydration read to the groups that actually hold a
    /// requested row keeps it O(1) groups instead of O(file): a 10M-row segment
    /// has ~500 row groups, and walking all of them per query is what made
    /// hydration cost 44 ms at 10M versus 6 ms at 1M, with *negative* thread
    /// scaling because every thread paged its way through the whole footer.
    pub(crate) fn row_groups_covering(
        metadata: &parquet::file::metadata::ParquetMetaData,
        bitmap: &RoaringBitmap,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if bitmap.is_empty() {
            return out;
        }
        let mut start = 0usize;
        let mut cursor = bitmap.iter().peekable();
        for rg in 0..metadata.num_row_groups() {
            let rows = metadata.row_group(rg).num_rows() as usize;
            let end = start + rows;
            while let Some(id) = cursor.peek() {
                if *id as usize >= start {
                    break;
                }
                cursor.next();
            }
            let first = match cursor.peek() {
                Some(id) => *id as usize,
                None => break,
            };
            if first < end {
                out.push((rg, start));
                while let Some(id) = cursor.peek() {
                    if *id as usize >= end {
                        break;
                    }
                    cursor.next();
                }
            }
            start = end;
        }
        out
    }

    pub(crate) async fn fetch_rows_with_distances(
        &self,
        matches: Vec<(usize, f32)>,
        target_schema: Option<arrow::datatypes::SchemaRef>,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(RecordBatch, Vec<f32>)>> {
        if matches.is_empty() {
            return Ok(vec![]);
        }

        // Build bitmap and track distances
        let mut bitmap = RoaringBitmap::new();
        let mut row_distances: std::collections::HashMap<u32, f32> =
            std::collections::HashMap::new();
        for (row_id, distance) in matches {
            bitmap.insert(row_id as u32);
            row_distances.insert(row_id as u32, distance);
        }

        // Apply Deletes
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
            bitmap -= &deleted;
            for row_id in deleted.iter() {
                row_distances.remove(&row_id);
            }
        }

        if bitmap.is_empty() {
            return Ok(vec![]);
        }

        let eq_deletes = self.load_equality_deletes().await?;

        // Score-only short-circuit: only safe when neither a residual filter
        // nor equality deletes need physical column values.
        if let Some(schema) = target_schema.as_ref() {
            if schema.fields().is_empty() && filter.is_none() && eq_deletes.is_empty() {
                let row_ids: Vec<u32> = bitmap.iter().collect();
                let batch_distances: Vec<f32> = row_ids
                    .iter()
                    .map(|id| row_distances.get(id).copied().unwrap_or(f32::MAX))
                    .collect();

                let distance_array =
                    std::sync::Arc::new(arrow::array::Float32Array::from(batch_distances.clone()));
                let final_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                    arrow::datatypes::Field::new(
                        "distance",
                        arrow::datatypes::DataType::Float32,
                        false,
                    ),
                ]));
                let final_batch = RecordBatch::try_new_with_options(
                    final_schema,
                    vec![distance_array],
                    &arrow::record_batch::RecordBatchOptions::new()
                        .with_row_count(Some(batch_distances.len())),
                )?;
                return Ok(vec![(final_batch, batch_distances)]);
            }
        }

        // Fetch Rows - use resolved path and cache_root + data_store
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();
        let cache_key = format!("{}/{}", self.cache_root(), pq_path_str);

        let (meta, size) = self.load_cached_parquet_meta(&pq_path).await?;
        let data_store = self.data_store();

        // 2. Hot Row Cache (Fast Path)
        const BLOCK_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;
        let total_rows = meta.metadata().file_metadata().num_rows() as usize;
        if total_rows == 0 {
            return Ok(vec![]);
        }
        bitmap &= RoaringBitmap::from_iter(0..(total_rows as u32));
        if bitmap.is_empty() {
            return Ok(vec![]);
        }

        let sparse_selection =
            total_rows > 0 && (bitmap.len() as usize).saturating_mul(8) < total_rows;
        if size < BLOCK_CACHE_MAX_BYTES || (size < 500 * 1024 * 1024 && !sparse_selection) {
            let mut cached_batch_opt = crate::core::cache::BLOCK_CACHE
                .get_with_metrics(&cache_key, "block_cache")
                .await;

            // Miss: Read the entire file from data_store and cache it
            if cached_batch_opt.is_none() {
                let reader = ParquetObjectReader::new(data_store.clone(), pq_path.clone());
                let arrow_meta = (*meta).clone();
                let builder =
                    ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta);

                let mut stream = builder.build()?;
                let mut all_batches = Vec::new();
                while let Some(batch) = stream.next().await {
                    all_batches.push(batch?);
                }

                if !all_batches.is_empty() {
                    let full_batch = match all_batches.len() {
                        1 => all_batches.remove(0),
                        _ => {
                            arrow::compute::concat_batches(&all_batches[0].schema(), &all_batches)?
                        }
                    };
                    let arc_batch = std::sync::Arc::new(full_batch);
                    crate::core::cache::BLOCK_CACHE
                        .insert(cache_key.clone(), arc_batch.clone())
                        .await;
                    cached_batch_opt = Some(arc_batch);
                }
            }

            // Hit: Use take_record_batch to instantly grab rows
            if let Some(arc_batch) = cached_batch_opt {
                let batch_rows = arc_batch.num_rows();
                let row_ids: Vec<u32> = bitmap
                    .iter()
                    .filter(|&id| (id as usize) < batch_rows)
                    .collect();
                if row_ids.is_empty() {
                    return Ok(vec![]);
                }
                let indices = arrow::array::UInt32Array::from(row_ids.clone());
                let taken_batch = arrow::compute::take_record_batch(&arc_batch, &indices)?;

                let mut final_batch = taken_batch;

                // Extract distances
                let mut batch_distances = Vec::with_capacity(row_ids.len());
                for row_id in &row_ids {
                    let distance = row_distances.get(row_id).copied().unwrap_or(f32::MAX);
                    batch_distances.push(distance);
                }

                // Apply equality deletes FIRST (on physical columns)
                if !eq_deletes.is_empty() {
                    let eq_keep = Self::apply_eq_deletes_keep_mask(&final_batch, &eq_deletes)?;
                    let prev_rows = final_batch.num_rows();
                    final_batch = arrow::compute::filter_record_batch(&final_batch, &eq_keep)?;
                    let mut filtered_distances = Vec::with_capacity(final_batch.num_rows());
                    for i in 0..prev_rows {
                        if eq_keep.is_valid(i) && eq_keep.value(i) {
                            filtered_distances.push(batch_distances[i]);
                        }
                    }
                    batch_distances = filtered_distances;
                    if final_batch.num_rows() == 0 {
                        return Ok(vec![]);
                    }
                }

                // Apply post-filtering if a filter was provided (BEFORE schema evolution / projection)
                if let Some(expr) = filter {
                    let planner = crate::core::planner::QueryPlanner::new();
                    let required_cols = expr.required_columns();
                    let eval_batch = if required_cols
                        .iter()
                        .any(|c| final_batch.column_by_name(c).is_none())
                    {
                        self.pad_missing_schema_columns(final_batch.clone(), &required_cols)
                            .await?
                    } else {
                        final_batch.clone()
                    };
                    let coerced = expr.coerce_literals(&eval_batch.schema());
                    let mask = planner.evaluate_expr(&eval_batch, &coerced)?;
                    let prev_rows = final_batch.num_rows();
                    final_batch = arrow::compute::filter_record_batch(&final_batch, &mask)?;

                    let mut filtered_distances = Vec::with_capacity(final_batch.num_rows());
                    for i in 0..prev_rows {
                        if mask.is_valid(i) && mask.value(i) {
                            filtered_distances.push(batch_distances[i]);
                        }
                    }
                    batch_distances = filtered_distances;
                }

                if final_batch.num_rows() == 0 {
                    return Ok(vec![]);
                }

                // Project / evolve columns if needed
                if let Some(schema) = &target_schema {
                    if schema.fields().is_empty() {
                        final_batch = arrow::record_batch::RecordBatch::try_new_with_options(
                            schema.clone(),
                            vec![],
                            &arrow::record_batch::RecordBatchOptions::new()
                                .with_row_count(Some(final_batch.num_rows())),
                        )?;
                    } else {
                        let mut new_columns = Vec::new();
                        for field in schema.fields() {
                            if let Ok(col) = final_batch.column_by_name(field.name()).ok_or(()) {
                                if col.data_type() != field.data_type() {
                                    let casted = arrow::compute::cast(col, field.data_type())?;
                                    new_columns.push(casted);
                                } else {
                                    new_columns.push(col.clone());
                                }
                            } else {
                                if !field.is_nullable() && final_batch.num_rows() > 0 {
                                    anyhow::bail!(
                                        "read: required column '{}' is missing from the source segment",
                                        field.name()
                                    );
                                }
                                let null_arr = arrow::array::new_null_array(
                                    field.data_type(),
                                    final_batch.num_rows(),
                                );
                                new_columns.push(null_arr);
                            }
                        }
                        final_batch =
                            arrow::record_batch::RecordBatch::try_new(schema.clone(), new_columns)?;
                    }
                }

                // Add distance column
                let distance_array =
                    std::sync::Arc::new(arrow::array::Float32Array::from(batch_distances.clone()));
                let mut fields = final_batch.schema().fields().to_vec();
                fields.push(std::sync::Arc::new(arrow::datatypes::Field::new(
                    "distance",
                    arrow::datatypes::DataType::Float32,
                    false,
                )));
                let final_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(fields));

                let mut columns = final_batch.columns().to_vec();
                columns.push(distance_array);
                final_batch = RecordBatch::try_new_with_options(
                    final_schema,
                    columns,
                    &arrow::record_batch::RecordBatchOptions::new()
                        .with_row_count(Some(batch_distances.len())),
                )?;

                return Ok(vec![(final_batch, batch_distances)]);
            }
        }

        // 3. --- STANDARD ROW SELECTION FETCH (For large files > 500MB) ---
        let reader = ParquetObjectReader::new(data_store.clone(), pq_path.clone());
        let arrow_meta = (*meta).clone();
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta);

        let groups = Self::row_groups_covering(builder.metadata(), &bitmap);
        let num_groups = builder.metadata().num_row_groups();
        if !groups.is_empty() && groups.len() < num_groups {
            let mut shifted = RoaringBitmap::new();
            let mut base: u32 = 0;
            for (rg, group_start) in &groups {
                let rows = builder.metadata().row_group(*rg).num_rows() as u32;
                for id in bitmap.iter() {
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
                &bitmap,
                builder.metadata().file_metadata().num_rows() as usize,
            );
            builder = builder.with_row_selection(selection);
        }

        // Apply column projection including target, filter, and equality delete columns
        let target_schema_ref = target_schema.clone();
        if let Some(schema) = &target_schema_ref {
            let parquet_schema = builder.metadata().file_metadata().schema_descr();
            let file_arrow_schema = builder.schema();

            let mut required_cols: Vec<String> =
                schema.fields().iter().map(|f| f.name().clone()).collect();
            if let Some(expr) = filter {
                for col in expr.required_columns() {
                    if !required_cols.contains(&col) {
                        required_cols.push(col);
                    }
                }
            }
            for del in &eq_deletes {
                if !required_cols.contains(&del.column_name) {
                    required_cols.push(del.column_name.clone());
                }
            }

            let column_indices: Vec<usize> = required_cols
                .iter()
                .filter_map(|name| file_arrow_schema.index_of(name).ok())
                .collect();

            let projection = ProjectionMask::roots(parquet_schema, column_indices);
            builder = builder.with_projection(projection);
        }

        let mut stream = builder.build()?;

        // Collect bitmap row IDs into a Vec for O(1) indexing
        let row_ids: Vec<u32> = bitmap.iter().collect();

        let mut results = Vec::new();
        let mut current_offset = 0usize;

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let num_rows = batch.num_rows();
            let avail = row_ids.len().saturating_sub(current_offset);
            let rows_to_process = std::cmp::min(num_rows, avail);

            if rows_to_process == 0 {
                break;
            }

            // Extract distances for this batch
            let mut batch_distances = Vec::with_capacity(rows_to_process);
            for i in 0..rows_to_process {
                let row_id = row_ids[current_offset + i];
                let distance = row_distances.get(&row_id).copied().unwrap_or(f32::MAX);
                batch_distances.push(distance);
            }

            let mut final_batch = if rows_to_process < num_rows {
                batch.slice(0, rows_to_process)
            } else {
                batch
            };

            current_offset += rows_to_process;

            // Apply equality deletes FIRST (on physical columns)
            if !eq_deletes.is_empty() {
                let eq_keep = Self::apply_eq_deletes_keep_mask(&final_batch, &eq_deletes)?;
                let prev_rows = final_batch.num_rows();
                final_batch = arrow::compute::filter_record_batch(&final_batch, &eq_keep)?;
                let mut filtered_distances = Vec::with_capacity(final_batch.num_rows());
                for i in 0..prev_rows {
                    if eq_keep.is_valid(i) && eq_keep.value(i) {
                        filtered_distances.push(batch_distances[i]);
                    }
                }
                batch_distances = filtered_distances;
                if final_batch.num_rows() == 0 {
                    continue;
                }
            }

            // Apply post-filtering if a filter was provided (BEFORE schema evolution / projection)
            if let Some(expr) = filter {
                let planner = crate::core::planner::QueryPlanner::new();
                let required_cols = expr.required_columns();
                let eval_batch = if required_cols
                    .iter()
                    .any(|c| final_batch.column_by_name(c).is_none())
                {
                    self.pad_missing_schema_columns(final_batch.clone(), &required_cols)
                        .await?
                } else {
                    final_batch.clone()
                };
                let coerced = expr.coerce_literals(&eval_batch.schema());
                let mask = planner.evaluate_expr(&eval_batch, &coerced)?;
                let prev_rows = final_batch.num_rows();
                final_batch = arrow::compute::filter_record_batch(&final_batch, &mask)?;

                let mut filtered_distances = Vec::with_capacity(final_batch.num_rows());
                for i in 0..prev_rows {
                    if mask.is_valid(i) && mask.value(i) {
                        filtered_distances.push(batch_distances[i]);
                    }
                }
                batch_distances = filtered_distances;
            }

            if final_batch.num_rows() == 0 {
                continue;
            }

            // Schema Evolution Mapping (AFTER post-filtering so filter columns are still present)
            if let Some(target) = &target_schema_ref {
                if target.fields().is_empty() {
                    final_batch = arrow::record_batch::RecordBatch::try_new_with_options(
                        target.clone(),
                        vec![],
                        &arrow::record_batch::RecordBatchOptions::new()
                            .with_row_count(Some(final_batch.num_rows())),
                    )?;
                } else {
                    let mut new_columns = Vec::new();
                    for field in target.fields() {
                        if let Ok(col) = final_batch.column_by_name(field.name()).ok_or(()) {
                            if col.data_type() != field.data_type() {
                                let casted = arrow::compute::cast(col, field.data_type())?;
                                new_columns.push(casted);
                            } else {
                                new_columns.push(col.clone());
                            }
                        } else {
                            if !field.is_nullable() && final_batch.num_rows() > 0 {
                                anyhow::bail!(
                                    "read: required column '{}' is missing from the source segment",
                                    field.name()
                                );
                            }
                            let null_arr = arrow::array::new_null_array(
                                field.data_type(),
                                final_batch.num_rows(),
                            );
                            new_columns.push(null_arr);
                        }
                    }
                    final_batch =
                        arrow::record_batch::RecordBatch::try_new(target.clone(), new_columns)?;
                }
            }

            // Add distance column to batch (nullable = false for consistent schema across segments)
            let distance_array =
                std::sync::Arc::new(arrow::array::Float32Array::from(batch_distances.clone()));
            let mut fields = final_batch.schema().fields().to_vec();
            fields.push(std::sync::Arc::new(arrow::datatypes::Field::new(
                "distance",
                arrow::datatypes::DataType::Float32,
                false,
            )));
            let mut columns = final_batch.columns().to_vec();
            columns.push(distance_array);
            final_batch = arrow::record_batch::RecordBatch::try_new_with_options(
                std::sync::Arc::new(arrow::datatypes::Schema::new(fields)),
                columns,
                &arrow::record_batch::RecordBatchOptions::new()
                    .with_row_count(Some(batch_distances.len())),
            )?;

            results.push((final_batch, batch_distances));
        }

        Ok(results)
    }

    async fn vector_search_on_filtered_ids(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        bitmap: &RoaringBitmap,
        metric: VectorMetric,
        residual_filter: Option<&FilterExpr>,
    ) -> Result<Vec<(usize, f32)>> {
        if k == 0 || bitmap.is_empty() {
            return Ok(vec![]);
        }

        let q_vec = match query {
            crate::core::index::VectorValue::Float32(v) => v,
            _ => anyhow::bail!("Filtered vector search only supports Float32 vectors currently"),
        };

        let pq_path = self.resolve_object_path("parquet");
        let (meta, size) = self.load_cached_parquet_meta(&pq_path).await?;
        let num_rows = meta.metadata().file_metadata().num_rows() as usize;
        if num_rows == 0 {
            return Ok(vec![]);
        }

        let deleted = self.load_merged_deletes().await?;
        let eq_deletes = self.load_equality_deletes().await?;
        let mut merged = bitmap.clone();
        merged &= RoaringBitmap::from_iter(0..(num_rows as u32));
        if !deleted.is_empty() {
            merged -= &deleted;
        }
        if merged.is_empty() {
            return Ok(vec![]);
        }

        let reader = self
            .parquet_reader(self.data_store(), &pq_path, size as u64)
            .await;
        let arrow_meta = (*meta).clone();
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta);

        let groups = Self::row_groups_covering(builder.metadata(), &merged);
        let num_groups = builder.metadata().num_row_groups();
        if !groups.is_empty() && groups.len() < num_groups {
            let mut shifted = RoaringBitmap::new();
            let mut base: u32 = 0;
            for (rg, group_start) in &groups {
                let rows = builder.metadata().row_group(*rg).num_rows() as u32;
                for id in merged.iter() {
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
            let selection = self.bitmap_to_row_selection(&merged, num_rows);
            builder = builder.with_row_selection(selection);
        }

        let mut required_cols: Vec<String> = vec![column.to_string()];
        let filter_cols = residual_filter
            .map(|f| f.required_columns())
            .unwrap_or_default();
        for c in &filter_cols {
            if !required_cols.contains(c) {
                required_cols.push(c.clone());
            }
        }
        for del in &eq_deletes {
            if !required_cols.contains(&del.column_name) {
                required_cols.push(del.column_name.clone());
            }
        }

        let mask = {
            let parquet_schema = builder.metadata().file_metadata().schema_descr();
            let file_arrow_schema = builder.schema();
            let column_indices: Vec<usize> = required_cols
                .iter()
                .filter_map(|name| file_arrow_schema.index_of(name).ok())
                .collect();
            ProjectionMask::roots(parquet_schema, column_indices)
        };
        builder = builder.with_projection(mask);

        let mut stream = builder.build()?;
        let mut all_scored: Vec<(usize, f32)> = Vec::new();
        let mut bitmap_iter = merged.iter();
        let planner = crate::core::planner::QueryPlanner::new();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let batch_rows = batch.num_rows();
            if batch_rows == 0 {
                continue;
            }

            let mut batch_row_ids = Vec::with_capacity(batch_rows);
            for _ in 0..batch_rows {
                let rid = bitmap_iter.next().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Bitmap iterator exhausted before reader batches (schema mismatch?)"
                    )
                })?;
                batch_row_ids.push(rid);
            }

            let eq_keep = if eq_deletes.is_empty() {
                None
            } else {
                Some(Self::apply_eq_deletes_keep_mask(&batch, &eq_deletes)?)
            };

            let filter_mask = if let Some(expr) = residual_filter {
                let eval_batch = if filter_cols
                    .iter()
                    .any(|c| batch.column_by_name(c).is_none())
                {
                    self.pad_missing_schema_columns(batch.clone(), &filter_cols)
                        .await?
                } else {
                    batch.clone()
                };
                let coerced = expr.coerce_literals(&eval_batch.schema());
                Some(planner.evaluate_expr(&eval_batch, &coerced)?)
            } else {
                None
            };

            if let Some(col) = batch.column_by_name(column) {
                for (i, &row_id) in batch_row_ids.iter().enumerate() {
                    if let Some(ref eq_k) = eq_keep {
                        if !eq_k.is_valid(i) || !eq_k.value(i) {
                            continue;
                        }
                    }
                    if let Some(ref m) = filter_mask {
                        if !m.is_valid(i) || !m.value(i) {
                            continue;
                        }
                    }
                    let Some(v) = Self::extract_vector_at(col.as_ref(), i, q_vec.len()) else {
                        continue;
                    };
                    let dist = Self::compute_vector_distance_f32(&v, q_vec, metric);
                    all_scored.push((row_id as usize, dist));
                }
            }
        }

        all_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_scored.truncate(k);
        Ok(all_scored)
    }

    /// Two-step filtered vector search: apply scalar filters first (hard),
    /// then compute vector distances on the surviving rows and return top-k.
    #[allow(clippy::too_many_arguments)]
    async fn vector_search_on_filtered(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        bitmap: &RoaringBitmap,
        metric: VectorMetric,
        target_schema: Option<arrow::datatypes::SchemaRef>,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(arrow::record_batch::RecordBatch, Vec<f32>)>> {
        let all_scored = self
            .vector_search_on_filtered_ids(column, query, k, bitmap, metric, filter)
            .await?;

        if all_scored.is_empty() {
            return Ok(vec![]);
        }

        // Eq deletes and any residual filter were already applied before top-k truncation.
        self.fetch_rows_with_distances(all_scored, target_schema, None)
            .await
    }

    /// Read only the vector column for a set of row IDs directly from Parquet using the
    /// Page/Offset Index, bypassing RecordBatch projection and schema evolution overhead.
    pub(crate) async fn read_vector_column_for_rows(
        &self,
        column: &str,
        bitmap: &RoaringBitmap,
    ) -> Result<Option<arrow::array::ArrayRef>> {
        if bitmap.is_empty() {
            return Ok(None);
        }

        let pq_path = self.resolve_object_path("parquet");
        let (meta, _) = self.load_cached_parquet_meta(&pq_path).await?;
        let total_rows = meta.metadata().file_metadata().num_rows() as usize;
        if total_rows == 0 {
            return Ok(None);
        }
        let mut clamped_bitmap = bitmap.clone();
        clamped_bitmap &= RoaringBitmap::from_iter(0..(total_rows as u32));
        if clamped_bitmap.is_empty() {
            return Ok(None);
        }

        let reader = ParquetObjectReader::new(self.data_store().clone(), pq_path);
        let arrow_meta = (*meta).clone();
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta);

        // Prune to the row groups covering the requested rows
        let groups = Self::row_groups_covering(builder.metadata(), &clamped_bitmap);
        let num_groups = builder.metadata().num_row_groups();
        if !groups.is_empty() && groups.len() < num_groups {
            let mut shifted = RoaringBitmap::new();
            let mut base: u32 = 0;
            for (rg, group_start) in &groups {
                let rows = builder.metadata().row_group(*rg).num_rows() as u32;
                for id in clamped_bitmap.iter() {
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
            let selection = self.bitmap_to_row_selection(&clamped_bitmap, total_rows);
            builder = builder.with_row_selection(selection);
        }

        // Apply column projection: project ONLY the vector column
        let parquet_schema = builder.metadata().file_metadata().schema_descr();
        let file_arrow_schema = builder.schema();
        let col_idx = match file_arrow_schema.index_of(column) {
            Ok(idx) => idx,
            Err(_) => return Ok(None),
        };
        let projection = ProjectionMask::roots(parquet_schema, vec![col_idx]);
        builder = builder.with_projection(projection);

        let mut stream = builder.build()?;
        let mut arrays = Vec::new();
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            if let Some(c) = batch.column_by_name(column) {
                arrays.push(c.clone());
            }
        }

        if arrays.is_empty() {
            return Ok(None);
        }

        if arrays.len() == 1 {
            Ok(Some(arrays.remove(0)))
        } else {
            let array_refs: Vec<&dyn arrow::array::Array> =
                arrays.iter().map(|a| a.as_ref()).collect();
            Ok(Some(arrow::compute::concat(&array_refs)?))
        }
    }

    /// The distance metric the column's vector index was built with.
    ///
    /// [`Self::search_hnsw_ivf`] takes a `_metric` argument and ignores it: a
    /// graph stores no vectors, only the distances computed at build time, so it
    /// can only rank in that space. Any stage that re-ranks its output has to use
    /// the same metric or it is sorting by a different notion of "close" than the
    /// one that produced the candidates. Resolving it here — from the index
    /// itself, through the same cached loader the search used — keeps the two
    /// stages consistent even when the caller leaves `metric` at the request
    /// default (`L2`), which is what the ANN-Benchmarks harness does.
    ///
    /// Falls back to the caller's metric when the column has no vector index
    /// (brute-force path) or the index cannot be loaded; the brute-force search
    /// that produced those candidates did use the caller's metric.
    async fn index_metric_or(&self, column: &str, fallback: VectorMetric) -> VectorMetric {
        let mut seen = std::collections::HashSet::new();
        let idx_info = self
            .config
            .index_files
            .iter()
            .filter(|f| {
                f.index_category == "vector"
                    && f.column_name == column
                    && self.config.is_index_valid(f)
                    && !matches!(
                        f.blob_type.as_deref(),
                        Some(crate::core::puffin::PUFFIN_BLOB_HNSW_IVF_CENTROIDS)
                            | Some(crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_MAPPING)
                            | Some(crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_GRAPH)
                            | Some("centroids")
                            | Some("mapping")
                            | Some("cluster_graph")
                    )
            })
            .find(|f| seen.insert((f.file_path.clone(), f.algorithm.clone())));

        let Some(idx_info) = idx_info else {
            return fallback;
        };

        let idx_path_str = self.resolve_index_path(&idx_info.file_path);
        let cache_key = if idx_info.blob_offset.is_some() {
            HnswIvfIndex::puffin_vector_cache_key(
                self.cache_root(),
                &idx_path_str,
                &idx_info.column_name,
                &idx_info.algorithm,
            )
        } else {
            format!("{}/{}", self.cache_root(), idx_path_str)
        };

        let loaded = if idx_info.blob_offset.is_some() {
            HnswIvfIndex::load_from_puffin(
                self.store.clone(),
                &idx_path_str,
                &cache_key,
                true,
                &idx_info.column_name,
            )
            .await
        } else {
            HnswIvfIndex::load_async_with_cache_key(
                self.store.clone(),
                &idx_path_str,
                &cache_key,
                true,
            )
            .await
        };

        match loaded {
            Ok(index) => index.metric(),
            Err(_) => fallback,
        }
    }

    /// Rescore approximate vector search candidates by point-looking up their
    /// uncompressed Float32 representations from Parquet and computing exact SIMD distances.
    pub(crate) async fn rescore_candidates(
        &self,
        column: &str,
        candidates: Vec<(usize, f32)>,
        query: &crate::core::index::VectorValue,
        metric: VectorMetric,
    ) -> Result<Vec<(usize, f32)>> {
        if candidates.is_empty() {
            return Ok(candidates);
        }

        let q_vec = match query {
            crate::core::index::VectorValue::Float32(v) => v,
            _ => return Ok(candidates),
        };

        // The candidates were produced by the HNSW-IVF index, which ranks with the
        // metric it was built with — `search_hnsw_ivf` deliberately ignores the
        // `_metric` its caller passes, because a graph stores no vectors, only the
        // distances computed at build time. Rescoring with the caller's metric
        // therefore re-ranked the candidates in a *different* space: the request
        // default is L2, so on a cosine index the refine path returned rows worse
        // than the graph already had. Measured on `glove-100-angular` (20k rows,
        // ef=200), recall@10 fell 0.988 -> 0.637 at refine×2 and 0.511 at refine×4,
        // degrading further as the candidate list grew. Use the index's metric so
        // the exact re-rank orders the neighbours the graph intended to order.
        let metric = self.index_metric_or(column, metric).await;

        // Reconstruct row IDs in order (sorted bitmap row IDs minus deletes)
        let deleted = self.load_merged_deletes().await?;
        let mut bitmap = RoaringBitmap::new();
        for (row_id, _) in &candidates {
            bitmap.insert(*row_id as u32);
        }
        if !deleted.is_empty() {
            bitmap -= &deleted;
        }
        let sorted_row_ids: Vec<u32> = bitmap.iter().collect();
        if sorted_row_ids.is_empty() {
            return Ok(vec![]);
        }

        // Direct scatter-gather read of ONLY the vector column, bypassing RecordBatch synthesis
        let col = match self.read_vector_column_for_rows(column, &bitmap).await? {
            Some(c) => c,
            None => return Ok(candidates),
        };

        let mut rescored = Vec::with_capacity(sorted_row_ids.len());

        // Fast zero-copy SIMD path for FixedSizeListArray (standard BenoStreamDB vector column)
        if let Some(list) = col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
        {
            let dim = list.value_length() as usize;
            if dim == q_vec.len() && list.len() == sorted_row_ids.len() {
                if let Some(floats) = list
                    .values()
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                {
                    let raw_floats = floats.values();
                    for (i, &row_id) in sorted_row_ids.iter().enumerate() {
                        if list.is_null(i) {
                            continue;
                        }
                        let start = i * dim;
                        let end = start + dim;
                        if end <= raw_floats.len() {
                            let v_slice = &raw_floats[start..end];
                            let exact_dist =
                                Self::compute_vector_distance_f32(v_slice, q_vec, metric);
                            rescored.push((row_id as usize, exact_dist));
                        }
                    }
                    rescored
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    return Ok(rescored);
                }
            }
        }

        if col.len() != sorted_row_ids.len() {
            tracing::warn!(
                "rescore_candidates vector count mismatch: got {} vectors, expected {} row IDs",
                col.len(),
                sorted_row_ids.len()
            );
            return Ok(candidates);
        }

        for (i, &row_id_u32) in sorted_row_ids.iter().enumerate() {
            let Some(v) = Self::extract_vector_at(col.as_ref(), i, q_vec.len()) else {
                continue;
            };
            let exact_dist = Self::compute_vector_distance_f32(&v, q_vec, metric);
            rescored.push((row_id_u32 as usize, exact_dist));
        }

        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(rescored)
    }

    async fn search_hnsw_ivf(
        &self,
        idx_info: &crate::core::manifest::IndexFile,
        query: &crate::core::index::VectorValue,
        k: usize,
        allowed_bitmap: &Option<RoaringBitmap>,
        _metric: VectorMetric,
        ef_search: Option<usize>,
        use_mmap: bool,
        target_clusters: Option<&[usize]>,
        probes: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let idx_path_str = self.resolve_index_path(&idx_info.file_path);

        // A Puffin container is loaded as one index per (column, algorithm);
        // keying by blob offset would give each cluster blob its own entry and
        // re-materialize the entire bundle on every miss.
        let cache_key = if idx_info.blob_offset.is_some() {
            HnswIvfIndex::puffin_vector_cache_key(
                self.cache_root(),
                &idx_path_str,
                &idx_info.column_name,
                &idx_info.algorithm,
            )
        } else {
            format!("{}/{}", self.cache_root(), idx_path_str)
        };

        // Load HNSW-IVF index. When the manifest entry carries a Puffin blob
        // offset, the index is packed inside a `.puffin` compound bundle and
        // must be materialized before loading; otherwise it is a loose
        // multi-file layout (.centroids.parquet, .cluster_N.hnsw.*).
        let hnsw_ivf = if idx_info.blob_offset.is_some() {
            HnswIvfIndex::load_from_puffin(
                self.store.clone(),
                &idx_path_str,
                &cache_key,
                use_mmap,
                &idx_info.column_name,
            )
            .await?
        } else {
            HnswIvfIndex::load_async_with_cache_key(
                self.store.clone(),
                &idx_path_str,
                &cache_key,
                use_mmap,
            )
            .await?
        };

        // Search with HNSW-IVF
        let query_clone = query.clone();
        let allowed_bm_clone = allowed_bitmap.clone();
        let hnsw_ivf_clone = hnsw_ivf.clone();

        let n_probe = probes.unwrap_or_else(|| {
            if allowed_bitmap.is_some() {
                20
            } else if let Some(ef) = ef_search {
                if ef >= 100 {
                    32
                } else {
                    16
                }
            } else {
                10
            }
        });
        tracing::debug!(
            "search_hnsw_ivf allowed_bitmap is_some={}, n_probe={}",
            allowed_bitmap.is_some(),
            n_probe
        );

        let target_clusters_owned = target_clusters.map(|tc| tc.to_vec());
        let matches = tokio::task::spawn_blocking(move || {
            if let Some(ref tc) = target_clusters_owned {
                hnsw_ivf_clone.search_targeted_clusters(
                    &query_clone,
                    k,
                    tc,
                    allowed_bm_clone.as_ref(),
                    ef_search,
                )
            } else {
                hnsw_ivf_clone.search(
                    &query_clone,
                    k,
                    n_probe,
                    allowed_bm_clone.as_ref(),
                    ef_search,
                )
            }
        })
        .await??;

        #[cfg(feature = "trace-search")]
        {
            eprintln!(
                "[trace] search_hnsw_ivf path={} blob_offset={:?} algo={} cache_key={} top={:?}",
                idx_info.file_path,
                idx_info.blob_offset,
                idx_info.algorithm,
                cache_key,
                matches
                    .iter()
                    .take(5)
                    .map(|(id, d)| (*id, *d))
                    .collect::<Vec<_>>(),
            );
        }

        Ok(matches)
    }
}
