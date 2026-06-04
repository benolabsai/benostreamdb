// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::cache::CacheExt;
use std::sync::Arc;
// use std::collections::HashSet;
use crate::core::index::hnsw_ivf::HnswIvfIndex;
use crate::core::index::VectorMetric;
use crate::core::planner::FilterExpr;
use crate::SegmentConfig;
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use chrono::Utc;
use futures::StreamExt;
use object_store::{path::Path, ObjectMeta, ObjectStore};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, RowSelection, RowSelector,
};
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::statistics::Statistics as ParquetStats;

use anyhow::{Context, Result};
use futures::stream::BoxStream;
use roaring::RoaringBitmap;

use super::*;

impl HybridReader {
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
        let metadata = self.get_parquet_metadata().await?;
        let num_rgs = metadata.num_row_groups();

        let surviving_rgs = self.prune_row_groups(&metadata, filter);

        let pruned = num_rgs - surviving_rgs.len();
        if pruned > 0 {
            tracing::info!(
                "Row-group pruning: skipped {}/{} row groups ({:.0}% reduction)",
                pruned,
                num_rgs,
                (pruned as f64 / num_rgs as f64 * 100.0)
            );
        }

        if surviving_rgs.is_empty() {
            return Ok(Vec::new());
        }

        let mut rg_matches: Vec<(usize, f32)> = Vec::new();

        let q_vec = match query {
            crate::core::index::VectorValue::Float32(v) => v,
            _ => anyhow::bail!("Flat search with row-group pruning only supports Float32 vectors"),
        };

        for rg_idx in &surviving_rgs {
            let mut offset = 0usize;
            for i in 0..*rg_idx {
                offset += metadata.row_group(i).num_rows() as usize;
            }

            let single_rg_stream = self.stream_row_groups(Some(&[*rg_idx]), None).await?;
            use futures::StreamExt;

            let mut local_offset = offset;
            let mut batches: Vec<RecordBatch> = Vec::new();
            let mut rg_stream = Box::pin(single_rg_stream);
            while let Some(batch_res) = rg_stream.next().await {
                match batch_res {
                    Ok(batch) => batches.push(batch),
                    Err(e) => return Err(e),
                }
            }

            for batch in batches {
                let rows = batch.num_rows();

                if let Some(col) = batch.column_by_name(column) {
                    let vectors: Vec<Vec<f32>> = match col.data_type() {
                        arrow::datatypes::DataType::FixedSizeList(_, _) => {
                            let list = col
                                .as_any()
                                .downcast_ref::<arrow::array::FixedSizeListArray>()
                                .context("Invalid cast to FixedSizeListArray")?;
                            (0..list.len())
                                .map(|i| {
                                    let item = list.value(i);
                                    if let Some(floats) =
                                        item.as_any().downcast_ref::<arrow::array::Float32Array>()
                                    {
                                        floats.values().to_vec()
                                    } else if let Some(doubles) =
                                        item.as_any().downcast_ref::<arrow::array::Float64Array>()
                                    {
                                        doubles.values().iter().map(|&d| d as f32).collect()
                                    } else {
                                        vec![0.0; item.len()]
                                    }
                                })
                                .collect()
                        }
                        arrow::datatypes::DataType::List(_) => {
                            let list = col
                                .as_any()
                                .downcast_ref::<arrow::array::ListArray>()
                                .context("Invalid cast to ListArray")?;
                            (0..list.len())
                                .map(|i| {
                                    let item = list.value(i);
                                    if let Some(floats) =
                                        item.as_any().downcast_ref::<arrow::array::Float32Array>()
                                    {
                                        floats.values().to_vec()
                                    } else if let Some(doubles) =
                                        item.as_any().downcast_ref::<arrow::array::Float64Array>()
                                    {
                                        doubles.values().iter().map(|&d| d as f32).collect()
                                    } else {
                                        vec![0.0; item.len()]
                                    }
                                })
                                .collect()
                        }
                        _ => vec![],
                    };

                    for (i, v) in vectors.iter().enumerate() {
                        let row_id = local_offset + i;
                        let dist = match metric {
                            VectorMetric::L2 => v
                                .iter()
                                .zip(q_vec.iter())
                                .map(|(a, b)| (a - b) * (a - b))
                                .sum::<f32>(),
                            VectorMetric::Cosine => {
                                let dot: f32 = v.iter().zip(q_vec.iter()).map(|(a, b)| a * b).sum();
                                let mag_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                                let mag_q: f32 = q_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                                1.0 - (dot / (mag_v * mag_q + 1e-10))
                            }
                            VectorMetric::InnerProduct => {
                                -v.iter().zip(q_vec.iter()).map(|(a, b)| a * b).sum::<f32>()
                            }
                            VectorMetric::L1 => v
                                .iter()
                                .zip(q_vec.iter())
                                .map(|(a, b)| (a - b).abs())
                                .sum::<f32>(),
                            VectorMetric::Hamming => {
                                v.iter().zip(q_vec.iter()).filter(|(a, b)| a != b).count() as f32
                            }
                            VectorMetric::Jaccard => {
                                let mut intersection = 0.0;
                                let mut union = 0.0;
                                for (x, y) in v.iter().zip(q_vec.iter()) {
                                    if *x > 0.0 || *y > 0.0 {
                                        if *x == *y && *x > 0.0 {
                                            intersection += 1.0;
                                        }
                                        union += 1.0;
                                    }
                                }
                                if union == 0.0 {
                                    0.0
                                } else {
                                    1.0 - intersection / union
                                }
                            }
                        };
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
        let store = self
            .config
            .data_store
            .clone()
            .unwrap_or_else(|| self.store.clone());
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();

        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE
            .get_with_metrics(
                &format!("{}/{}", self.root_uri, pq_path_str),
                "parquet_meta",
            )
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
            let reader = ParquetObjectReader::new(store.clone(), object_meta.location);

            let options = ArrowReaderOptions::default();
            let arrow_meta = ArrowReaderMetadata::try_new(meta, options)?;
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta)
        } else {
            // Miss
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
            let reader = ParquetObjectReader::new(store.clone(), object_meta.location);

            let b_res = ParquetRecordBatchStreamBuilder::new(reader).await;
            let b = match b_res {
                Ok(b) => b,
                Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
                    return Ok(futures::stream::empty().boxed());
                }
                Err(e) => return Err(e.into()),
            };

            crate::core::cache::PARQUET_META_CACHE
                .insert(
                    format!("{}/{}", self.root_uri, pq_path_str),
                    (b.metadata().clone(), size as usize),
                )
                .await;
            b
        };

        // Apply Row Group Selection
        if let Some(rgs) = row_groups {
            builder = builder.with_row_groups(rgs.to_vec());
        }

        // Schema Evolution / Projection Logic
        // Determine which PHYSICAL columns to read based on target_schema
        let target_schema_ref = target_schema.clone();

        if let Some(schema) = &target_schema_ref {
            let parquet_schema = builder.metadata().file_metadata().schema_descr();
            let file_arrow_schema = builder.schema();

            // Map logical columns to physical columns
            // Only read columns that exist in the physical file
            let column_indices: Vec<usize> = schema
                .fields()
                .iter()
                .filter_map(|field| file_arrow_schema.index_of(field.name()).ok())
                .collect();

            // NOTE: If column_indices is empty, ProjectionMask::roots will result in a 0-column read
            // which is exactly what we want for count queries.
            let projection = ProjectionMask::roots(parquet_schema, column_indices);
            builder = builder.with_projection(projection);
        }

        // Apply Deletes
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
            let num_rows = builder.metadata().file_metadata().num_rows() as usize;
            let full_range = RoaringBitmap::from_iter(0..num_rows as u32);
            let valid = full_range - deleted;

            let selection = self.bitmap_to_row_selection(&valid, num_rows);
            builder = builder.with_row_selection(selection);
        }

        // Load Equality Deletes
        let equality_deletes = self.load_equality_deletes().await?;

        let stream = builder.build()?;

        // Wrap stream to apply Schema Mapping (Evolution) and Equality Deletes
        let mapped_stream = stream.map(move |res| {
            let mut batch = res.map_err(anyhow::Error::from)?;

            // 1. Schema Evolution Mapping
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

            // 2. Apply Equality Deletes (Anti-Join)
            if !equality_deletes.is_empty() {
                // Initialize mask as all true (keep all rows)
                let mut keep_mask = arrow::array::BooleanArray::from(vec![true; batch.num_rows()]);

                for delete in &equality_deletes {
                    if let Some(col) = batch.column_by_name(&delete.column_name) {
                        // Check which rows match the delete values
                        // is_in(left, right) returns true if left[i] is in right
                        match check_is_in(col, &delete.values) {
                            Ok(delete_mask) => {
                                // We want to KEEP rows that are NOT in the delete set
                                if let Ok(not_delete) = arrow::compute::not(&delete_mask) {
                                    if let Ok(new_mask) =
                                        arrow::compute::and(&keep_mask, &not_delete)
                                    {
                                        keep_mask = new_mask;
                                    }
                                }
                            }
                            Err(e) => tracing::warn!(
                                "Failed to apply equality delete filter on {}: {}",
                                delete.column_name,
                                e
                            ),
                        }
                    }
                }

                batch = arrow::compute::filter_record_batch(&batch, &keep_mask)?;
            }

            Ok(batch)
        });

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
        let mut stream = self
            .stream_all(None as Option<arrow::datatypes::SchemaRef>)
            .await?;
        let mut matches = Vec::new();
        let mut current_row_offset = 0usize;

        let q_vec = match query {
            crate::core::index::VectorValue::Float32(v) => v,
            _ => anyhow::bail!("Flat search only supports Float32 vectors currently"),
        };

        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let rows = batch.num_rows();

            if let Some(col) = batch.column_by_name(column) {
                let vectors: Vec<Vec<f32>> = match col.data_type() {
                    arrow::datatypes::DataType::FixedSizeList(_, _) => {
                        let list = col
                            .as_any()
                            .downcast_ref::<arrow::array::FixedSizeListArray>()
                            .context("Invalid cast")?;
                        (0..list.len())
                            .map(|i| {
                                let item = list.value(i);
                                if let Some(floats) =
                                    item.as_any().downcast_ref::<arrow::array::Float32Array>()
                                {
                                    floats.values().to_vec()
                                } else if let Some(doubles) =
                                    item.as_any().downcast_ref::<arrow::array::Float64Array>()
                                {
                                    doubles.values().iter().map(|&d| d as f32).collect()
                                } else {
                                    vec![0.0; item.len()]
                                }
                            })
                            .collect()
                    }
                    arrow::datatypes::DataType::List(_) => {
                        let list = col
                            .as_any()
                            .downcast_ref::<arrow::array::ListArray>()
                            .context("Invalid cast")?;
                        (0..list.len())
                            .map(|i| {
                                let item = list.value(i);
                                if let Some(floats) =
                                    item.as_any().downcast_ref::<arrow::array::Float32Array>()
                                {
                                    floats.values().to_vec()
                                } else if let Some(doubles) =
                                    item.as_any().downcast_ref::<arrow::array::Float64Array>()
                                {
                                    doubles.values().iter().map(|&d| d as f32).collect()
                                } else {
                                    vec![0.0; item.len()]
                                }
                            })
                            .collect()
                    }
                    _ => vec![],
                };

                for (i, v) in vectors.iter().enumerate() {
                    let row_id = current_row_offset + i;

                    // Check filter (allowed_bitmap)
                    if let Some(bm) = allowed_bitmap {
                        if !bm.contains(row_id as u32) {
                            continue;
                        }
                    }

                    let dist = match metric {
                        VectorMetric::L2 => v
                            .iter()
                            .zip(q_vec.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>(),
                        VectorMetric::Cosine => {
                            let dot: f32 = v.iter().zip(q_vec.iter()).map(|(a, b)| a * b).sum();
                            let mag_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let mag_q: f32 = q_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                            1.0 - (dot / (mag_v * mag_q + 1e-10))
                        }
                        VectorMetric::InnerProduct => {
                            -v.iter().zip(q_vec.iter()).map(|(a, b)| a * b).sum::<f32>()
                        }
                        VectorMetric::L1 => v
                            .iter()
                            .zip(q_vec.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f32>(),
                        VectorMetric::Hamming => {
                            v.iter().zip(q_vec.iter()).filter(|(a, b)| a != b).count() as f32
                        }
                        VectorMetric::Jaccard => {
                            let mut intersection = 0.0;
                            let mut union = 0.0;
                            for (x, y) in v.iter().zip(q_vec.iter()) {
                                if *x > 0.0 || *y > 0.0 {
                                    if *x == *y && *x > 0.0 {
                                        intersection += 1.0;
                                    }
                                    union += 1.0;
                                }
                            }
                            if union == 0.0 {
                                0.0
                            } else {
                                1.0 - intersection / union
                            }
                        }
                    };
                    matches.push((row_id, dist));
                }
            }
            current_row_offset += rows;
        }

        // Sort by distance and take k
        matches.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if matches.len() > k {
            matches.truncate(k);
        }

        Ok(matches)
    }

    #[tracing::instrument(skip(self, query, filter, target_schema))]
    pub async fn vector_search_index(
        &self,
        column: &str,
        query: &crate::core::index::VectorValue,
        k: usize,
        filter: Option<&FilterExpr>,
        metric: VectorMetric,
        ef_search: Option<usize>,
        target_schema: Option<arrow::datatypes::SchemaRef>,
    ) -> Result<Vec<(arrow::record_batch::RecordBatch, Vec<f32>)>> {
        // Resolve scalar filter to combined bitmap if present
        tracing::debug!("vector_search_index called with filter: {:?}", filter);
        let allowed_bitmap = if let Some(expr) = filter {
            let sub_filters = expr.extract_and_conditions();
            let mut combined_bitmap: Option<RoaringBitmap> = None;

            for sub_f in sub_filters {
                let res = self.get_scalar_filter_bitmap(&sub_f).await;
                if let Ok(Some(bm)) = res {
                    match combined_bitmap {
                        Some(ref mut existing) => {
                            *existing &= bm;
                        }
                        None => {
                            combined_bitmap = Some(bm);
                        }
                    }

                    // Optimization: short-circuit if bitmap is empty
                    if let Some(ref bm) = combined_bitmap {
                        if bm.is_empty() {
                            return Ok(vec![]);
                        }
                    }
                }
            }
            combined_bitmap
        } else {
            None
        };

        // Handle keyword search regardless of filter path
        if let crate::core::index::VectorValue::Keyword(ref q) = query {
            let results = self.keyword_search_index(column, q, k, filter).await?;
            if results.is_empty() {
                return Ok(vec![]);
            }

            let matches: Vec<(u32, f32)> = results.iter().map(|(id, s)| (*id as u32, *s)).collect();
            let batch = self
                .read_rows_by_id_with_schema(matches, target_schema)
                .await?;

            let scores: Vec<f32> = results.into_iter().map(|(_, s)| s).collect();
            return Ok(vec![(batch, scores)]);
        }

        // --- TWO-STEP APPROACH when filter is present and resolvable ---
        // Step 1: Apply scalar filters as hard filters (Parquet row selection)
        // Step 2: Run vector search on the filtered resultset
        // If filter exists but bitmap is None (no scalar index), fall through
        // to index search + post-filtering for correctness.
        if let Some(ref bitmap) = allowed_bitmap {
            return self
                .vector_search_on_filtered(column, query, k, bitmap, metric, target_schema, filter)
                .await;
        }

        // --- FILTER PRESENT but no scalar index bitmap ---
        // Use row-group pruning to skip entire row groups based on Parquet column stats
        // before computing any vector distances. This avoids the O(N) vector computation
        // on pruned row groups, which is the key optimization when no scalar index exists.
        if let Some(filter_expr) = filter {
            let rg_matches = self
                .vector_search_flat_with_row_group_pruning(column, query, k, filter_expr, metric)
                .await?;
            return self
                .fetch_rows_with_distances(rg_matches, target_schema, filter)
                .await;
        }

        // --- FAST PATH: no filter, use HNSW-IVF index ---
        {
            let vector_indices: Vec<_> = self
                .config
                .index_files
                .iter()
                .filter(|f| f.index_type == "vector" && f.column_name.as_deref() == Some(column))
                .collect();

            let vector_idx_info = if vector_indices.is_empty() {
                None
            } else {
                let mut sorted = vector_indices.clone();
                sorted.sort_by_key(|f| match f.blob_type.as_deref() {
                    Some("hnsw_tq8") | Some("hnsw_tq4") => 0,
                    Some("hnsw_pq") => 1,
                    Some("hnsw_ivf") => 2,
                    _ => 3,
                });
                Some(sorted[0])
            };

            let idx_matches = if let Some(idx_info) = vector_idx_info {
                match self
                    .search_hnsw_ivf(idx_info, query, k, &None, metric, ef_search)
                    .await
                {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::error!(
                            "Vector index listed in manifest failed, falling back to flat scan: {}",
                            e
                        );
                        self.vector_search_flat(column, query, k, &None, metric)
                            .await?
                    }
                }
            } else {
                let idx_path = self.resolve_object_path(column).to_string();
                let idx_info = crate::core::manifest::IndexFile {
                    file_path: idx_path,
                    index_type: "vector".to_string(),
                    column_name: Some(column.to_string()),
                    ..Default::default()
                };
                match self
                    .search_hnsw_ivf(&idx_info, query, k, &None, metric, ef_search)
                    .await
                {
                    Ok(m) => m,
                    Err(_) => {
                        self.vector_search_flat(column, query, k, &None, metric)
                            .await?
                    }
                }
            };

            self.fetch_rows_with_distances(idx_matches, target_schema, filter)
                .await
        }
    }

    pub async fn keyword_search_index(
        &self,
        column: &str,
        query: &str,
        k: usize,
        _filter: Option<&FilterExpr>,
    ) -> Result<Vec<(usize, f32)>> {
        // 1. Find Inverted Index
        tracing::info!(
            "keyword_search_index: Segment {} searching for inverted index on column '{}'",
            self.config.segment_id,
            column
        );
        tracing::info!("  Available index files: {:?}", self.config.index_files);

        let idx_info = self.config.index_files.iter()
            .find(|f| {
                let match_type = f.index_type == "inverted" || f.index_type == "bm25";
                let match_col = f.column_name.as_deref() == Some(column);
                match_type && match_col
            })
            .ok_or_else(|| anyhow::anyhow!("No keyword/inverted index found for column '{}' in segment {} (Available: {:?})", column, self.config.segment_id, self.config.index_files))?;

        // 2. Load Inverted Index Batches
        // (Similar to get_scalar_filter_bitmap logic)
        let inv_path_str = &idx_info.file_path;
        let mut dir_path = self.config.parquet_path.clone().unwrap_or_default();
        if let Some(pos) = dir_path.rfind('/') {
            dir_path.truncate(pos);
        } else {
            dir_path = "".to_string();
        }

        let full_inv_path_str = if dir_path.is_empty() || inv_path_str.contains('/') {
            inv_path_str.clone()
        } else {
            format!("{}/{}", dir_path, inv_path_str)
        };

        let cache_key = if let Some(offset) = idx_info.offset {
            format!("{}/{}:{}", self.root_uri, full_inv_path_str, offset)
        } else {
            format!("{}/{}", self.root_uri, full_inv_path_str)
        };

        let batches = if let Some(cached) = crate::core::cache::INVERTED_INDEX_CACHE
            .get_with_metrics(&cache_key, "inverted_index")
            .await
        {
            cached.as_ref().clone()
        } else {
            // Cache Miss - Load from Disk
            let inv_path = Path::from(full_inv_path_str.as_str());
            let inv_bytes = match self.store.get(&inv_path).await {
                Ok(res) => {
                    let b = res.bytes().await?;
                    crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                    b.to_vec()
                }
                Err(e) => return Err(e.into()),
            };

            let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                Bytes::from(inv_bytes),
            )?;
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
        // Use the tokenizer defined in the index metadata, fallback to standard
        let tokenizer_name = "default";

        let tokenizer = crate::core::index::tokenizer::GLOBAL_TOKENIZER_REGISTRY
            .read()
            .get(tokenizer_name)
            .ok_or_else(|| anyhow::anyhow!("Missing standard tokenizer"))?;

        let query_tokens = tokenizer.tokenize(query);
        if query_tokens.is_empty() {
            return Ok(vec![]);
        }

        // 4. Scoring Map: RowID -> BM25 Score
        let mut scores: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        let n_total = (self.config.record_count.unwrap_or(0) as f32).max(1.0); // Ensure at least 1.0 for IDF
        let k1 = 1.2;

        for token in &query_tokens {
            // Find this token in the inverted index batches
            // Build document term frequencies for scoring
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

                for i in 0..batch.num_rows() {
                    let key = key_array.value(i);
                    if key == token {
                        // Found the term!
                        let list = row_ids_list.value(i);
                        let row_ids = list
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>()
                            .context("Invalid cast")?;

                        // Count frequencies by document
                        let mut current_doc_counts: std::collections::HashMap<u32, u32> =
                            std::collections::HashMap::new();
                        let mut last_id = 0;
                        for j in 0..row_ids.len() {
                            let rid = last_id + row_ids.value(j);
                            *current_doc_counts.entry(rid).or_default() += 1;
                            last_id = rid;
                        }

                        // IDF for this token
                        let n_token = current_doc_counts.len() as f32;
                        let idf = ((n_total - n_token + 0.5) / (n_token + 0.5) + 1.0).ln();

                        // Add to global scores
                        for (rid, tf) in current_doc_counts {
                            let tf = tf as f32;
                            let score_inc = idf * (tf * (k1 + 1.0)) / (tf + k1);
                            *scores.entry(rid).or_default() += score_inc;
                        }
                    }
                }
            }
        }

        // 5. Sort by score and convert to (RowID, Distance)
        let mut results: Vec<(usize, f32)> = scores
            .into_iter()
            .map(|(rid, score)| (rid as usize, 1.0 / (1.0 + score))) // Convert score to distance-like metric
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if results.len() > k {
            results.truncate(k);
        }

        Ok(results)
    }

    pub async fn read_rows_by_id(
        &self,
        matches: Vec<(u32, f32)>,
        _columns: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        self.read_rows_by_id_with_schema(matches, None).await
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

    async fn fetch_rows_with_distances(
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
            bitmap -= &deleted; // Borrow instead of move
                                // Remove distances for deleted rows
            for row_id in deleted.iter() {
                row_distances.remove(&row_id);
            }
        }

        if bitmap.is_empty() {
            return Ok(vec![]);
        }

        // Fetch Rows - use resolved path
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();

        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE
            .get_with_metrics(
                &format!("{}/{}", self.root_uri, pq_path_str),
                "parquet_meta",
            )
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
            let reader = ParquetObjectReader::new(self.store.clone(), object_meta.location);

            let options = ArrowReaderOptions::default();
            let arrow_meta = ArrowReaderMetadata::try_new(meta, options)?;
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta)
        } else {
            // Miss
            // Ensure file exists/get meta (HEAD)
            let object_meta = self
                .store
                .head(&pq_path)
                .await
                .context("Failed to get segment metadata")?;
            let size = object_meta.size;
            let reader = ParquetObjectReader::new(self.store.clone(), object_meta.location);

            let b = ParquetRecordBatchStreamBuilder::new(reader).await?;
            crate::core::cache::PARQUET_META_CACHE
                .insert(
                    format!("{}/{}", self.root_uri, pq_path_str),
                    (b.metadata().clone(), size as usize),
                )
                .await;
            b
        };

        let selection = self.bitmap_to_row_selection(
            &bitmap,
            builder.metadata().file_metadata().num_rows() as usize,
        );
        builder = builder.with_row_selection(selection);

        // Apply column projection if specified (skip reading unused columns like embeddings)
        let target_schema_ref = target_schema.clone();
        if let Some(schema) = &target_schema_ref {
            let parquet_schema = builder.metadata().file_metadata().schema_descr();
            let file_arrow_schema = builder.schema();

            // --- BUG FIX: Ensure columns required by filter are included in projection ---
            let mut required_cols: Vec<String> =
                schema.fields().iter().map(|f| f.name().clone()).collect();
            if let Some(expr) = filter {
                for col in expr.required_columns() {
                    if !required_cols.contains(&col) {
                        required_cols.push(col);
                    }
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

            // Handle case where RowSelection returns more rows than bitmap
            // (due to gap coalescing). Only process rows we have distances for.
            let rows_to_process = std::cmp::min(num_rows, row_ids.len() - current_offset);

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

            // Add distance column to batch
            let distance_array =
                std::sync::Arc::new(arrow::array::Float32Array::from(batch_distances.clone()));
            let mut fields = final_batch.schema().fields().to_vec();
            fields.push(std::sync::Arc::new(arrow::datatypes::Field::new(
                "distance",
                arrow::datatypes::DataType::Float32,
                true,
            )));
            let mut columns = final_batch.columns().to_vec();
            columns.push(distance_array);
            final_batch = arrow::record_batch::RecordBatch::try_new(
                std::sync::Arc::new(arrow::datatypes::Schema::new(fields)),
                columns,
            )?;

            // Schema Evolution Mapping
            if let Some(target) = &target_schema_ref {
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
                        let null_arr =
                            arrow::array::new_null_array(field.data_type(), final_batch.num_rows());
                        new_columns.push(null_arr);
                    }
                }
                final_batch =
                    arrow::record_batch::RecordBatch::try_new(target.clone(), new_columns)?;
            }

            current_offset += rows_to_process;

            // --- BUG FIX: Apply post-filtering if a filter was provided ---
            // This ensures correctness even if the index search was overly permissive (e.g. missing inverted indexes for some terms)
            if let Some(expr) = filter {
                let planner = crate::core::planner::QueryPlanner::new();
                match planner.evaluate_expr(&final_batch, expr) {
                    Ok(mask) => {
                        // Filter the batch
                        let prev_rows = final_batch.num_rows();
                        final_batch = arrow::compute::filter_record_batch(&final_batch, &mask)?;

                        // Filter the distances to match the new batch
                        let mut filtered_distances = Vec::with_capacity(final_batch.num_rows());
                        for i in 0..prev_rows {
                            if mask.value(i) {
                                filtered_distances.push(batch_distances[i]);
                            }
                        }
                        batch_distances = filtered_distances;
                    }
                    Err(_) => {
                        // Silently skip if evaluation fails (to maintain backward compatibility with unknown exprs)
                    }
                }
            }

            if final_batch.num_rows() > 0 {
                results.push((final_batch, batch_distances));
            }
        }

        Ok(results)
    }

    /// Two-step filtered vector search: apply scalar filters first (hard),
    /// then compute vector distances on the surviving rows and return top-k.
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
        use futures::StreamExt;

        let q_vec = match query {
            crate::core::index::VectorValue::Float32(v) => v.clone(),
            _ => anyhow::bail!("Filtered vector search only supports Float32 vectors currently"),
        };

        // Step 1: Fetch only the rows that match the scalar filter bitmap
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();

        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE
            .get_with_metrics(
                &format!("{}/{}", self.root_uri, pq_path_str),
                "parquet_meta",
            )
            .await
        {
            let object_meta = ObjectMeta {
                location: pq_path.clone(),
                last_modified: Utc::now(),
                size: size as u64,
                e_tag: None,
                version: None,
            };
            let reader = ParquetObjectReader::new(self.store.clone(), object_meta.location);
            let options = ArrowReaderOptions::default();
            let arrow_meta = ArrowReaderMetadata::try_new(meta, options)?;
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta)
        } else {
            let object_meta = self
                .store
                .head(&pq_path)
                .await
                .context("Failed to get segment metadata")?;
            let size = object_meta.size;
            let reader = ParquetObjectReader::new(self.store.clone(), object_meta.location);
            let b = ParquetRecordBatchStreamBuilder::new(reader).await?;
            crate::core::cache::PARQUET_META_CACHE
                .insert(
                    format!("{}/{}", self.root_uri, pq_path_str),
                    (b.metadata().clone(), size as usize),
                )
                .await;
            b
        };

        // Apply row selection to read only filtered rows
        let num_rows = builder.metadata().file_metadata().num_rows() as usize;

        // Handle deletes as well
        let deleted = self.load_merged_deletes().await?;
        let merged = if !deleted.is_empty() {
            let full_range = RoaringBitmap::from_iter(0..num_rows as u32);
            let valid = full_range - deleted;
            &valid & bitmap
        } else {
            bitmap.clone()
        };

        let selection = self.bitmap_to_row_selection(&merged, num_rows);
        builder = builder.with_row_selection(selection);

        let mut stream = builder.build()?;
        let mut all_scored: Vec<(usize, f32)> = Vec::new();

        // We use an iterator over the merged bitmap to get the CORRECT global row IDs for each row returned by the reader.
        let mut bitmap_iter = merged.iter();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;

            if let Some(col) = batch.column_by_name(column) {
                for i in 0..batch.num_rows() {
                    let row_id = bitmap_iter.next().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Bitmap iterator exhausted before reader batches (schema mismatch?)"
                        )
                    })?;

                    let vec_row = if let Some(fs) = col
                        .as_any()
                        .downcast_ref::<arrow::array::FixedSizeListArray>()
                    {
                        fs.value(i)
                    } else if let Some(l) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
                        l.value(i)
                    } else {
                        return Err(anyhow::anyhow!("Invalid vector column type (expected FixedSizeListArray or ListArray) for column '{}'", column));
                    };

                    let floats = vec_row
                        .as_any()
                        .downcast_ref::<arrow::array::Float32Array>()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Expected Float32Array in vector value for column '{}'",
                                column
                            )
                        })?;

                    let dist = match metric {
                        VectorMetric::L2 => {
                            crate::core::index::distance::l2_distance(&q_vec, floats.values())
                        }
                        VectorMetric::Cosine => {
                            crate::core::index::distance::cosine_similarity(&q_vec, floats.values())
                        }
                        VectorMetric::InnerProduct => {
                            crate::core::index::distance::dot_product(&q_vec, floats.values())
                        }
                        VectorMetric::L1 => {
                            crate::core::index::distance::l1_distance(&q_vec, floats.values())
                        }
                        _ => f32::MAX,
                    };
                    all_scored.push((row_id as usize, dist));
                }
            }
        }

        // Step 2: Sort by distance and take top-k
        all_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_scored.truncate(k);

        // Step 3: Fetch the top-k rows as RecordBatches with distances
        if all_scored.is_empty() {
            return Ok(vec![]);
        }

        // Convert to (usize, f32) matches and use existing fetch_rows_with_distances
        self.fetch_rows_with_distances(all_scored, target_schema, filter)
            .await
    }

    async fn search_hnsw_ivf(
        &self,
        idx_info: &crate::core::manifest::IndexFile,
        query: &crate::core::index::VectorValue,
        k: usize,
        allowed_bitmap: &Option<RoaringBitmap>,
        _metric: VectorMetric,
        _ef_search: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let idx_path_str = idx_info.file_path.clone();

        let cache_key = if let Some(offset) = idx_info.offset {
            format!("{}/{}:{}", self.root_uri, idx_path_str, offset)
        } else {
            format!("{}/{}", self.root_uri, idx_path_str)
        };

        // Load HNSW-IVF index
        // NOTE: blob_type records the *algorithm* (e.g. "hnsw_tq8"), not the storage format.
        // The writer always uses the multi-file layout (.centroids.parquet, .cluster_N.hnsw.*),
        // so we always load via load_async_with_cache_key regardless of blob_type.
        let hnsw_ivf =
            HnswIvfIndex::load_async_with_cache_key(self.store.clone(), &idx_path_str, &cache_key)
                .await?;

        // Search with HNSW-IVF
        let query_clone = query.clone();
        let allowed_bm_clone = allowed_bitmap.clone();
        let hnsw_ivf_clone = hnsw_ivf.clone();

        // Determine n_probe based on filtering or session config
        let n_probe = if allowed_bitmap.is_some() { 20 } else { 10 };
        tracing::debug!(
            "search_hnsw_ivf allowed_bitmap is_some={}",
            allowed_bitmap.is_some()
        );

        let matches = tokio::task::spawn_blocking(move || {
            hnsw_ivf_clone.search(&query_clone, k, n_probe, allowed_bm_clone.as_ref())
        })
        .await??;

        Ok(matches)
    }
}
