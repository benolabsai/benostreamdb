// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::segment::ScalarPostingMap;
use anyhow::{Context, Result};
use arrow::array::Array;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

impl crate::core::segment::HybridSegmentWriter {
    pub(crate) fn build_inverted_index(
        &self,
        col_name: &str,
        col_array: &Arc<dyn Array>,
        row_offset: usize,
        _local_staging_dir: &std::path::Path,
    ) -> Result<()> {
        use arrow::datatypes::{DataType, TimeUnit};

        let config = self.index_configs.get(col_name);
        match col_array.data_type() {
            DataType::Int8 => {
                tracing::debug!("Indexing Int8 column: {} (offset={})", col_name, row_offset);
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Int8Array>()
                    .context("Invalid cast to Int8Array")?;
                let mut local: BTreeMap<i8, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Int8(BTreeMap::new()))
                {
                    ScalarPostingMap::Int8(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::Int16 => {
                tracing::debug!(
                    "Indexing Int16 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Int16Array>()
                    .context("Invalid cast to Int16Array")?;
                let mut local: BTreeMap<i16, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Int16(BTreeMap::new()))
                {
                    ScalarPostingMap::Int16(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::Int32 => {
                tracing::info!(
                    "Indexing Int32 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .context("Invalid cast to Int32Array")?;
                let mut local: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Int32(BTreeMap::new()))
                {
                    ScalarPostingMap::Int32(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::Int64 => {
                tracing::info!(
                    "Indexing Int64 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .context("Invalid cast to Int64Array")?;
                let mut local: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Int64(BTreeMap::new()))
                {
                    ScalarPostingMap::Int64(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::UInt8 => {
                tracing::debug!(
                    "Indexing UInt8 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt8Array>()
                    .context("Invalid cast to UInt8Array")?;
                let mut local: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::UInt8(BTreeMap::new()))
                {
                    ScalarPostingMap::UInt8(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::UInt16 => {
                tracing::debug!(
                    "Indexing UInt16 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt16Array>()
                    .context("Invalid cast to UInt16Array")?;
                let mut local: BTreeMap<u16, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::UInt16(BTreeMap::new()))
                {
                    ScalarPostingMap::UInt16(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::UInt32 => {
                tracing::debug!(
                    "Indexing UInt32 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .context("Invalid cast to UInt32Array")?;
                let mut local: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::UInt32(BTreeMap::new()))
                {
                    ScalarPostingMap::UInt32(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::UInt64 => {
                tracing::debug!(
                    "Indexing UInt64 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .context("Invalid cast to UInt64Array")?;
                let mut local: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::UInt64(BTreeMap::new()))
                {
                    ScalarPostingMap::UInt64(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::Float64 => {
                tracing::info!(
                    "Indexing Float64 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .context("Invalid cast to Float64Array")?;
                let mut local: HashMap<u64, Vec<u32>> = HashMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        let canonical = if v == 0.0 { 0.0f64 } else { v };
                        local
                            .entry(canonical.to_bits())
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Float64(HashMap::new()))
                {
                    ScalarPostingMap::Float64(map) => {
                        for (bits, mut rows) in local {
                            map.entry(bits).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::Float32 => {
                tracing::info!(
                    "Indexing Float32 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .context("Invalid cast to Float32Array")?;
                let mut local: HashMap<u32, Vec<u32>> = HashMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        let canonical = if v == 0.0 { 0.0f32 } else { v };
                        local
                            .entry(canonical.to_bits())
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Float32(HashMap::new()))
                {
                    ScalarPostingMap::Float32(map) => {
                        for (bits, mut rows) in local {
                            map.entry(bits).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            // String/Utf8/LargeUtf8/Utf8View Inverted Index - for category/tag/BM25 filtering
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                tracing::info!("Indexing String column: {}", col_name);

                let casted_array = arrow::compute::cast(col_array, &DataType::Utf8)
                    .context("Failed to cast column to Utf8 for indexing")?;
                let array = casted_array
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .context("Invalid cast to StringArray")?;

                // Fetch tokenizer if configured:
                // - If BM25 is configured on the column, use its tokenizer or default to English analyzer
                // - Otherwise (scalar inverted index, composite, or categorical), default to identity
                //   so stop-words (e.g. 'A') are not discarded for exact/scalar matching.
                let bm25_tokenizer = config.and_then(|c| {
                    c.algorithms.iter().find_map(|a| {
                        if let crate::core::manifest::IndexAlgorithm::Bm25 { tokenizer, .. } = a {
                            if !tokenizer.is_empty() {
                                Some(tokenizer.clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                });

                let is_bm25 = config
                    .map(|c| {
                        c.algorithms.iter().any(|a| {
                            matches!(a, crate::core::manifest::IndexAlgorithm::Bm25 { .. })
                        })
                    })
                    .unwrap_or(false);

                let tokenizer_name = config
                    .and_then(|c| c.tokenizer.clone())
                    .or(bm25_tokenizer)
                    .unwrap_or_else(|| {
                        if is_bm25 {
                            "analyzer:english".to_string()
                        } else {
                            "identity".to_string()
                        }
                    });
                tracing::info!(
                    "  Using tokenizer: '{}' for column '{}'",
                    tokenizer_name,
                    col_name
                );
                let tokenizer = crate::core::index::tokenizer::GLOBAL_TOKENIZER_REGISTRY
                    .read()
                    .get(&tokenizer_name)
                    .ok_or_else(|| anyhow::anyhow!("Missing tokenizer '{}'", tokenizer_name))?;

                // Remember the analyzer for this column so finish_indexing can
                // stamp it into the inverted file's parquet metadata and Puffin
                // blob properties (query-time tokenization must match index-time).
                {
                    let mut meta = self.index_metadata.lock();
                    meta.insert(col_name.to_string(), tokenizer_name.clone());
                }

                // Accumulate postings into task-local map before merging under lock.
                let mut col_inverted_map: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();

                for (batch_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        let tokens = tokenizer.tokenize(v);
                        let global_row_id = (row_offset + batch_i) as u32;
                        for (pos, token) in tokens.into_iter().enumerate() {
                            col_inverted_map
                                .entry(token)
                                .or_default()
                                .push((global_row_id, pos as u32));
                        }
                    }
                }

                {
                    let mut inverted_lock = self.inverted_data.lock();
                    let entry = inverted_lock.entry(col_name.to_string()).or_default();
                    for (token, mut postings) in col_inverted_map {
                        entry.entry(token).or_default().append(&mut postings);
                    }
                }

                // JSON-path index: accumulate across batches into json_path_data and
                // pack into the Puffin bundle in finish_indexing().
                let json_paths = config.and_then(|c| {
                    c.algorithms.iter().find_map(|a| {
                        if let crate::core::manifest::IndexAlgorithm::JsonPath { paths } = a {
                            Some(paths.clone())
                        } else {
                            None
                        }
                    })
                });
                if let Some(paths) = json_paths {
                    if !paths.is_empty() {
                        let mut batch_postings = BTreeMap::new();
                        let mut covered = Vec::new();
                        crate::core::index::json_path::accumulate_json_path_postings(
                            col_array,
                            &paths,
                            row_offset,
                            &mut batch_postings,
                            &mut covered,
                        )?;
                        let mut jp_lock = self.json_path_data.lock();
                        let entry = jp_lock.entry(col_name.to_string()).or_default();
                        for (key, mut rows) in batch_postings {
                            entry.0.entry(key).or_default().append(&mut rows);
                        }
                        for p in covered {
                            if !entry.1.iter().any(|existing| existing == &p) {
                                entry.1.push(p);
                            }
                        }
                        tracing::info!(
                            "  Buffered JSON-path postings for column '{}' ({} paths, offset={})",
                            col_name,
                            paths.len(),
                            row_offset
                        );
                    }
                }

                tracing::info!(
                    "  Buffered tokens for {} rows in column '{}' (offset={})",
                    array.len(),
                    col_name,
                    row_offset
                );
            }

            // Date32 Inverted Index - days since Unix epoch (1970-01-01)
            DataType::Date32 => {
                tracing::info!(
                    "Indexing Date32 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Date32Array>()
                    .context("Invalid cast to Date32Array")?;
                let mut local: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Date32(BTreeMap::new()))
                {
                    ScalarPostingMap::Date32(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            // Timestamp Inverted Index - truncate to UTC day for practical indexing
            DataType::Timestamp(_, _) => {
                tracing::info!(
                    "Indexing Timestamp column: {} (truncated to day, offset={})",
                    col_name,
                    row_offset
                );
                let mut local: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
                let array = col_array.as_any();
                for row_i in 0..col_array.len() {
                    if col_array.is_null(row_i) {
                        continue;
                    }
                    let day = if let Some(arr) =
                        array.downcast_ref::<arrow::array::TimestampSecondArray>()
                    {
                        arr.value(row_i).div_euclid(86_400) as i32
                    } else if let Some(arr) =
                        array.downcast_ref::<arrow::array::TimestampMillisecondArray>()
                    {
                        arr.value(row_i).div_euclid(86_400_000) as i32
                    } else if let Some(arr) =
                        array.downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    {
                        arr.value(row_i).div_euclid(86_400_000_000) as i32
                    } else if let Some(arr) =
                        array.downcast_ref::<arrow::array::TimestampNanosecondArray>()
                    {
                        arr.value(row_i).div_euclid(86_400_000_000_000) as i32
                    } else {
                        continue;
                    };
                    local
                        .entry(day)
                        .or_default()
                        .push((row_offset + row_i) as u32);
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::TimestampDay(BTreeMap::new()))
                {
                    ScalarPostingMap::TimestampDay(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            DataType::Boolean => {
                tracing::info!(
                    "Indexing Boolean column: {} (native boolean index, offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .context("Invalid cast to BooleanArray")?;
                let mut local: BTreeMap<bool, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Boolean(BTreeMap::new()))
                {
                    ScalarPostingMap::Boolean(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            // Time32 (s/ms) -> Int32 keys
            DataType::Time32(unit) => {
                tracing::info!(
                    "Indexing Time32 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let mut local: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
                match unit {
                    TimeUnit::Second => {
                        if let Some(array) = col_array
                            .as_any()
                            .downcast_ref::<arrow::array::Time32SecondArray>()
                        {
                            for (row_i, val) in array.iter().enumerate() {
                                if let Some(v) = val {
                                    local
                                        .entry(v)
                                        .or_default()
                                        .push((row_offset + row_i) as u32);
                                }
                            }
                        }
                    }
                    TimeUnit::Millisecond => {
                        if let Some(array) = col_array
                            .as_any()
                            .downcast_ref::<arrow::array::Time32MillisecondArray>()
                        {
                            for (row_i, val) in array.iter().enumerate() {
                                if let Some(v) = val {
                                    local
                                        .entry(v)
                                        .or_default()
                                        .push((row_offset + row_i) as u32);
                                }
                            }
                        }
                    }
                    other => anyhow::bail!("Unsupported Time32 unit: {:?}", other),
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard.entry(col_name.to_string()).or_insert_with(|| {
                    ScalarPostingMap::Time32 {
                        unit: *unit,
                        postings: BTreeMap::new(),
                    }
                }) {
                    ScalarPostingMap::Time32 { postings, .. } => {
                        for (k, mut rows) in local {
                            postings.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            // Time64 (us/ns) -> Int64 keys
            DataType::Time64(unit) => {
                tracing::info!(
                    "Indexing Time64 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let mut local: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
                match unit {
                    TimeUnit::Microsecond => {
                        if let Some(array) = col_array
                            .as_any()
                            .downcast_ref::<arrow::array::Time64MicrosecondArray>()
                        {
                            for (row_i, val) in array.iter().enumerate() {
                                if let Some(v) = val {
                                    local
                                        .entry(v)
                                        .or_default()
                                        .push((row_offset + row_i) as u32);
                                }
                            }
                        }
                    }
                    TimeUnit::Nanosecond => {
                        if let Some(array) = col_array
                            .as_any()
                            .downcast_ref::<arrow::array::Time64NanosecondArray>()
                        {
                            for (row_i, val) in array.iter().enumerate() {
                                if let Some(v) = val {
                                    local
                                        .entry(v)
                                        .or_default()
                                        .push((row_offset + row_i) as u32);
                                }
                            }
                        }
                    }
                    other => anyhow::bail!("Unsupported Time64 unit: {:?}", other),
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard.entry(col_name.to_string()).or_insert_with(|| {
                    ScalarPostingMap::Time64 {
                        unit: *unit,
                        postings: BTreeMap::new(),
                    }
                }) {
                    ScalarPostingMap::Time64 { postings, .. } => {
                        for (k, mut rows) in local {
                            postings.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            // Binary / LargeBinary / FixedSizeBinary -> Key is Vec<u8>
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
                tracing::info!(
                    "Indexing Binary column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let casted = arrow::compute::cast(col_array, &DataType::Binary)?;
                let array = casted
                    .as_any()
                    .downcast_ref::<arrow::array::BinaryArray>()
                    .context("Invalid cast to BinaryArray")?;
                let mut local: BTreeMap<Vec<u8>, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v.to_vec())
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard
                    .entry(col_name.to_string())
                    .or_insert_with(|| ScalarPostingMap::Binary(BTreeMap::new()))
                {
                    ScalarPostingMap::Binary(map) => {
                        for (k, mut rows) in local {
                            map.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            // Decimal128
            DataType::Decimal128(precision, scale) => {
                tracing::info!(
                    "Indexing Decimal128 column: {} (offset={})",
                    col_name,
                    row_offset
                );
                let array = col_array
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .context("Invalid cast to Decimal128Array")?;
                let mut local: BTreeMap<i128, Vec<u32>> = BTreeMap::new();
                for (row_i, val) in array.iter().enumerate() {
                    if let Some(v) = val {
                        local
                            .entry(v)
                            .or_default()
                            .push((row_offset + row_i) as u32);
                    }
                }
                let mut guard = self.scalar_inverted_data.lock();
                match guard.entry(col_name.to_string()).or_insert_with(|| {
                    ScalarPostingMap::Decimal128 {
                        precision: *precision,
                        scale: *scale,
                        postings: BTreeMap::new(),
                    }
                }) {
                    ScalarPostingMap::Decimal128 { postings, .. } => {
                        for (k, mut rows) in local {
                            postings.entry(k).or_default().append(&mut rows);
                        }
                    }
                    _ => anyhow::bail!("Mismatched scalar index type for column '{}'", col_name),
                }
            }

            // Dictionary Types (Recursive)
            DataType::Dictionary(_, value_type) => {
                tracing::info!(
                    "Indexing Dictionary column: {} (unpacking to {:?})",
                    col_name,
                    value_type.as_ref()
                );
                let casted = arrow::compute::cast(col_array, value_type)
                    .map_err(|e| anyhow::anyhow!("Failed to unpack dictionary: {}", e))?;
                self.index_column(col_name, &casted, row_offset)?;
            }

            _ => {
                tracing::warn!(
                    "Skipping indexing for unsupported type: {:?}",
                    col_array.data_type()
                );
            }
        }
        Ok(())
    }
}
