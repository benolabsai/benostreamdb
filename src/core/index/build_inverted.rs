use anyhow::{Result, Context};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use std::fs::File;
use parquet::arrow::ArrowWriter;

impl crate::core::segment::HybridSegmentWriter {
    pub(crate) fn build_inverted_index(&self, col_name: &str, col_array: &Arc<dyn Array>, row_offset: usize, local_staging_dir: &std::path::Path) -> Result<()> {
        let config = self.index_configs.get(col_name);
        match col_array.data_type() {
                 arrow::datatypes::DataType::Int32 => {
                    tracing::info!("Indexing Int32 column: {}", col_name);
                    let _array = col_array.as_any().downcast_ref::<arrow::array::Int32Array>().context("Invalid cast")?;
                    
                    // Scalar index (.idx) is removed for Int32 as we use more precise Inverted Index

                    // Optimized Inverted Index (Sort-based instead of HashMap)
                    let array = col_array.as_any().downcast_ref::<arrow::array::Int32Array>().context("Invalid cast")?;
                    
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
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                     {
                        let mut files = self.generated_files.lock();
                        files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
                    }
                },


                arrow::datatypes::DataType::Int64 => {
                    tracing::info!("Indexing Int64 column: {}", col_name);
                    let _array = col_array.as_any().downcast_ref::<arrow::array::Int64Array>().context("Invalid cast")?;
                    
                    // No mock .idx for Int64

                    // Optimized Inverted Index (Sort-based)
                    let array = col_array.as_any().downcast_ref::<arrow::array::Int64Array>().context("Invalid cast")?;
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
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock();
                        files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
                    }
                },

                arrow::datatypes::DataType::Float64 => {
                    tracing::info!("Indexing Float64 column: {}", col_name);
                    let _array = col_array.as_any().downcast_ref::<arrow::array::Float64Array>().context("Invalid cast")?;
                    
                    // No mock .idx for Float64

                    // Optimized Inverted Index (Sort-based)
                    let array = col_array.as_any().downcast_ref::<arrow::array::Float64Array>().context("Invalid cast")?;
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
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock();
                        files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
                    }
                },

                arrow::datatypes::DataType::Float32 => {
                    tracing::info!("Indexing Float32 column: {}", col_name);
                    let array = col_array.as_any().downcast_ref::<arrow::array::Float32Array>().context("Invalid cast")?;
                    
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
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock();
                        files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
                    }
                },
                
                // String/Utf8 Inverted Index - for category/tag filtering
                arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8 => {
                    tracing::info!("Indexing String column: {}", col_name);
                    
                    // Unified handling: cast to Utf8 to reuse existing logic
                    let casted_array = arrow::compute::cast(col_array, &arrow::datatypes::DataType::Utf8)
                        .context("Failed to cast column to Utf8 for indexing")?;
                    let array = casted_array.as_any().downcast_ref::<arrow::array::StringArray>().context("Invalid cast")?;
                    
                    // Fetch tokenizer if configured
                    let tokenizer_name = config.and_then(|c| c.tokenizer.clone()).unwrap_or_else(|| "identity".to_string());
                    tracing::info!("  Using tokenizer: '{}' for column '{}'", tokenizer_name, col_name);
                    let tokenizer = crate::core::index::tokenizer::GLOBAL_TOKENIZER_REGISTRY.get(&tokenizer_name)
                        .ok_or_else(|| anyhow::anyhow!("Missing identity tokenizer"))?;

                    // Build inverted index: Token -> RowIDs (buffered in memory per segment)
                    let mut inverted_lock = self.inverted_data.lock();
                    let col_inverted_map = inverted_lock.entry(col_name.to_string()).or_default();
                    
                    for (batch_i, val) in array.iter().enumerate() {
                        if let Some(v) = val {
                            let tokens = tokenizer.tokenize(v);
                            let global_row_id = (row_offset + batch_i) as u32;
                            for token in tokens {
                                col_inverted_map.entry(token).or_default().push(global_row_id);
                            }
                        }
                    }
                    
                    tracing::info!("  Buffered tokens for {} rows in column '{}'", array.len(), col_name);
                },
                
                // Date32 Inverted Index - for date equality/range filtering
                // Date32 = days since Unix epoch (1970-01-01)
                arrow::datatypes::DataType::Date32 => {
                    tracing::info!("Indexing Date32 column: {}", col_name);
                    let array = col_array.as_any().downcast_ref::<arrow::array::Date32Array>().context("Invalid cast")?;
                    
                    // Build inverted index: Date -> RowIDs
                    let mut inverted_map: std::collections::HashMap<i32, Vec<u32>> = std::collections::HashMap::new();
                    for (row_i, val) in array.iter().enumerate() {
                        if let Some(v) = val {
                            inverted_map.entry(v).or_default().push(row_i as u32);
                        }
                    }
                    
                    tracing::info!("  Found {} unique dates", inverted_map.len());
                    
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
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock();
                        files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
                    }
                    
                    tracing::info!("Date32 Inverted Index written to {}", inv_path.to_str().context("Invalid UTF-8 in path")?);
                },
                
                // Timestamp Inverted Index - truncate to day for practical indexing
                // High-cardinality timestamps are truncated to day granularity
                arrow::datatypes::DataType::Timestamp(_, _) => {
                    tracing::info!("Indexing Timestamp column: {} (truncated to day)", col_name);
                    
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
                    
                    tracing::info!("  Found {} unique days", inverted_map.len());
                    
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
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                    {
                        let mut files = self.generated_files.lock();
                        files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
                    }
                    
                    tracing::info!("Timestamp Inverted Index (day granularity) written to {}", inv_path.to_str().context("Invalid UTF-8 in path")?);
                },
                
                // Keep default
                arrow::datatypes::DataType::Boolean => {
                    tracing::info!("Indexing Boolean column: {} (native boolean index)", col_name);
                     // Build inverted index: Boolean -> RowIDs (true/false as native booleans)
                    let mut inverted_map: std::collections::HashMap<bool, Vec<u32>> = std::collections::HashMap::new();

                    let array = col_array.as_any().downcast_ref::<arrow::array::BooleanArray>().context("Invalid cast")?;
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
                    let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
                    let inv_file = File::create(&inv_tmp)?;
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer = ArrowWriter::try_new(inv_file, inv_schema, Some(props))?;
                    writer.write(&inv_batch)?;
                    writer.close()?;
                    std::fs::rename(&inv_tmp, &inv_path)?;

                     {
                        let mut files = self.generated_files.lock();
                        files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
                    }
                    tracing::info!("Boolean Inverted Index written to {}", inv_path.to_str().context("Invalid UTF-8 in path")?);
                },

                // Time32 (s/ms) -> Int32 keys
                arrow::datatypes::DataType::Time32(unit) => {
                    tracing::info!("Indexing Time32 column: {}", col_name);
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
                    tracing::info!("Indexing Time64 column: {}", col_name);
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
                     tracing::info!("Indexing Binary column: {}", col_name);
                     // Cast to BinaryArray for uniform handling (if possible, else matching works)
                     // Simple handling: Iterate as BinaryArray (works for large and regular if we cast, or just use generics. 
                     // arrow::compute::cast to Binary is easiest)
                     let casted = arrow::compute::cast(col_array, &arrow::datatypes::DataType::Binary)?;
                     let array = casted.as_any().downcast_ref::<arrow::array::BinaryArray>().context("Invalid cast")?;
                     
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
                     tracing::info!("Indexing Decimal128 column: {}", col_name);
                     let array = col_array.as_any().downcast_ref::<arrow::array::Decimal128Array>().context("Invalid cast")?;
                     
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
                    tracing::info!("Indexing Dictionary column: {} (unpacking to {:?})", col_name, value_type.as_ref());
                    // Cast to value type to unpack
                    let casted = arrow::compute::cast(col_array, value_type)
                        .map_err(|e| anyhow::anyhow!("Failed to unpack dictionary: {}", e))?;
                    self.index_column(col_name, &casted, row_offset)?;
                },


                _ => {
                    // Skip unsupported
                    tracing::warn!("Skipping indexing for unsupported type: {:?}", col_array.data_type());
                }
        }
        Ok(())
    }

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

        let inv_tmp = format!("{}.tmp", inv_path.to_str().context("Invalid UTF-8 in path")?);
        
        let inv_file = File::create(&inv_tmp)?;
        let props = parquet::file::properties::WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(inv_file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        std::fs::rename(&inv_tmp, &inv_path)?;

        {
             let mut files = self.generated_files.lock();
             files.push(inv_path.to_str().context("Invalid UTF-8 in path")?.to_string());
        }
        tracing::info!("Inverted Index written to {}", inv_path.display());
        Ok(())
    }

}
