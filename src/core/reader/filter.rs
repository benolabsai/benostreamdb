// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::cache::CacheExt;
use std::sync::Arc;
// use std::collections::HashSet;
use chrono::Utc;
use futures::StreamExt;
use bytes::Bytes;
use object_store::{path::Path, ObjectStore, ObjectMeta};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector, ArrowReaderMetadata, ArrowReaderOptions};
use crate::core::index::hnsw_ivf::HnswIvfIndex;
use arrow::record_batch::RecordBatch;
use parquet::arrow::async_reader::{ParquetRecordBatchStreamBuilder, ParquetObjectReader};
use parquet::arrow::ProjectionMask;
use crate::SegmentConfig;
use crate::core::planner::FilterExpr;
use crate::core::index::VectorMetric;
use arrow::array::Array;
use parquet::file::metadata::ParquetMetaData;


use anyhow::{Context, Result};
use roaring::RoaringBitmap;
use futures::stream::BoxStream;


use super::*;

impl HybridReader {
    pub async fn check_bloom_filter(&self, column: &str, value: &serde_json::Value) -> Result<bool> {
        let metadata: std::sync::Arc<parquet::file::metadata::ParquetMetaData> = self.get_parquet_metadata().await?;
        let schema = metadata.file_metadata().schema_descr();
        
        let col_idx = schema.columns().iter().position(|c| c.name() == column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in Parquet schema", column))?;
        
        // Split Block Bloom Filters (SBBF) are used in Parquet.
        use parquet::bloom_filter::Sbbf;
        
        let mut possible = false;
        let pq_path_str = self.config.parquet_path.clone().unwrap_or_default();
        if pq_path_str.is_empty() { return Ok(true); }
        let pq_path = object_store::path::Path::from(pq_path_str.clone());

        // Get file size to handle Range correctly
        let file_size = if let Some(s) = self.config.file_size {
            s
        } else {
            self.store.head(&pq_path).await?.size as u64
        };

        for i in 0..metadata.num_row_groups() {
            let rg_meta = metadata.row_group(i);
            let col_meta = rg_meta.column(col_idx);
            
            if let Some(offset) = col_meta.bloom_filter_offset() {
                let cache_key = format!("{}/{}:{}", self.root_uri, pq_path_str, offset);
                
                let sbbf = if let Some(cached) = crate::core::cache::BLOOM_FILTER_CACHE.get_with_metrics(&cache_key, "bloom_filter").await {
                    tracing::debug!("Bloom Cache HIT for key: {}", cache_key);
                    cached
                } else {
                    tracing::debug!("Bloom Cache MISS for key: {}", cache_key);
                    let start = offset as u64;
                    let end = (start + 2 * 1024 * 1024).min(file_size);
                    
                    // Fetch the Bloom Filter blob from storage
                    let data_res = self.store.get_range(&pq_path, start..end).await.map(|b| {
                        crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                        b
                    });
                    let data = match data_res {
                        Ok(d) => d,
                        Err(_) => {
                            let b = self.store.get_range(&pq_path, start..file_size).await?;
                            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                            b
                        },
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
                                        if let Ok(expected_len) = rest[..space_pos].parse::<usize>() {
                                            if expected_len <= data.len() {
                                                tracing::debug!("Retrying Bloom Filter parse with truncated length: {}", expected_len);
                                                Sbbf::from_bytes(&data[..expected_len])?
                                            } else { return Err(e.into()); }
                                        } else { return Err(e.into()); }
                                    } else { return Err(e.into()); }
                                } else { return Err(e.into()); }
                            } else { return Err(e.into()); }
                        }
                    };
                    
                    let arc_filter = Arc::new(filter);
                    crate::core::cache::BLOOM_FILTER_CACHE.insert(cache_key.clone(), arc_filter.clone()).await;
                    tracing::debug!("Bloom Cache INSERTED for key: {}", cache_key);
                    arc_filter
                };

                let desc = schema.column(col_idx);
                    let physical_type = desc.physical_type();
                    
                    let matches = match physical_type {
                        parquet::basic::Type::INT64 => {
                            if let Some(i) = value.as_i64() { sbbf.check(&i) } else { true }
                        }
                        parquet::basic::Type::INT32 => {
                            if let Some(i) = value.as_i64() { 
                                let i32_val = i as i32;
                                sbbf.check(&i32_val) 
                            } else { true }
                        }
                        parquet::basic::Type::BYTE_ARRAY | parquet::basic::Type::FIXED_LEN_BYTE_ARRAY => {
                             if let Some(s) = value.as_str() { 
                                 // s is &str. &s is &&str. &str implements AsBytes.
                                 sbbf.check(&s) 
                             } else if let Some(vals) = value.as_array() {
                                 let bytes: Vec<u8> = vals.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect();
                                 if bytes.len() == vals.len() {
                                     sbbf.check(&bytes)
                                 } else { true }
                             } else { true }
                        }
                        parquet::basic::Type::FLOAT => {
                            if let Some(f) = value.as_f64() { 
                                let f32_val = f as f32;
                                sbbf.check(&f32_val) 
                            } else { true }
                        }
                        parquet::basic::Type::DOUBLE => {
                            if let Some(f) = value.as_f64() { sbbf.check(&f) } else { true }
                        }
                        _ => true,
                    };

                    if matches {
                        possible = true;
                        break;
                    }
                }
            }
        Ok(possible)
    }

    pub async fn check_value_exists(&self, column: &str, value: &serde_json::Value) -> Result<bool> {
        // 1. Bloom Filter (Fastest rejection)
        if !self.check_bloom_filter(column, value).await? {
            return Ok(false);
        }
        
        // 2. Inverted Index / Bitmap (Precise lookup)
        // Convert serde_json::Value to QueryFilter for the index search
        let filter = crate::core::planner::QueryFilter {
             column: column.to_string(),
             min: Some(value.clone()),
             min_inclusive: true,
             max: Some(value.clone()),
             max_inclusive: true,
             values: None,
             negated: false,
        };
        
        if let Some(bitmap) = self.get_scalar_filter_bitmap(&filter).await? {
            return Ok(!bitmap.is_empty());
        }
        
        // 3. Fallback: Full Scan (Slowest)
        // This is only called if Bloom Filter said "Possible" and no Inverted Index exists.
        let batches = self.stream_all(None).await?;
        let mut stream = batches;
        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let col = batch.column_by_name(column)
                .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in scan", column))?;
            
            // Check rows in batch
            for i in 0..batch.num_rows() {
                let val = crate::core::manifest::ManifestValue::from_array(col, i);
                // Simple comparison (might need type handling optimization)
                if format!("{}", val) == format!("{}", value).trim_matches('"') {
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }

    #[tracing::instrument(skip(self, filter))]
    pub async fn get_scalar_filter_bitmap(&self, filter: &crate::core::planner::QueryFilter) -> Result<Option<RoaringBitmap>> {
        let filter_column = &filter.column;
        
        let inv_idx_info = self.config.index_files.iter()
            .find(|f| (f.index_type == "inverted" || f.index_type == "bitmap" || f.index_type == "bm25") && f.column_name.as_deref() == Some(filter_column));
            
        let matching_bitmap = if let Some(idx_info) = inv_idx_info {
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

            // Use Inverted Index (Value -> RowIDs)
            // 1. Check Object Cache (Decoded RecordBatches)
            let cache_key = if let Some(offset) = idx_info.offset {
                 format!("{}/{}:{}", self.root_uri, full_inv_path_str, offset)
            } else {
                 format!("{}/{}", self.root_uri, full_inv_path_str)
            };

            let batches = if let Some(cached) = crate::core::cache::INVERTED_INDEX_CACHE.get_with_metrics(&cache_key, "inverted_index").await {
                cached.as_ref().clone()
            } else {
                // Cache Miss - Load from Disk/Byte Cache
                let inv_path = Path::from(full_inv_path_str.as_str());
                
                let inv_bytes = match crate::core::cache::BYTE_CACHE.get_with_metrics(&cache_key, "byte").await {
                    Some(cached) => cached.as_ref().clone(),
                    None => {
                        let bytes = if let (Some(offset), Some(length)) = (idx_info.offset, idx_info.length) {
                             // Puffin Blob: Byte Range Read
                             {
                                 let b = self.store.get_range(&inv_path, (offset as u64)..(offset as u64 + length as u64)).await?;
                                 crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                                 b
                             }
                                 .to_vec()
                        } else {
                             // Full File read
                             match self.store.get(&inv_path).await {
                                 Ok(res) => {
                                     let b = res.bytes().await?;
                                     crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);
                                     b.to_vec()
                                 },
                                 Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
                                     // Missing index file - fallback to full scan
                                     return Ok(None);
                                 }
                                 Err(e) => return Err(e.into()),
                             }
                        };
                        
                        crate::core::cache::BYTE_CACHE.insert(cache_key.clone(), Arc::new(bytes.clone())).await;
                        bytes
                    }
                };
    
                let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(Bytes::from(inv_bytes))?;
                let reader = builder.build()?;
                
                let mut decoded = Vec::new();
                for batch_result in reader {
                    decoded.push(batch_result?);
                }
                
                crate::core::cache::INVERTED_INDEX_CACHE.insert(format!("{}/{}", self.root_uri, full_inv_path_str), Arc::new(decoded.clone())).await;
                decoded
            };
            
            let mut bitmap = RoaringBitmap::new();
            
            // The inverted index schema is [key, row_ids (List<UInt32>)]
            for batch in batches {
                let key_array = batch.column(0);
                let row_ids_list = batch.column(1).as_any().downcast_ref::<arrow::array::ListArray>()
                    .ok_or_else(|| anyhow::anyhow!("Expected ListArray in inverted index column 1"))?;
                
                // Perform range/value filtering on inverted index keys
                for i in 0..batch.num_rows() {
                    let key_ok = match (key_array.data_type(), &filter.min, &filter.max) {
                        (arrow::datatypes::DataType::Utf8, Some(min_val), _) => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::StringArray>().context("Invalid cast")?.value(i);
                            let mut ok = true;
                            if let Some(min_s) = min_val.as_str() {
                                if filter.min_inclusive { ok &= val >= min_s; } else { ok &= val > min_s; }
                            }
                            if let Some(max_val) = &filter.max {
                                if let Some(max_s) = max_val.as_str() {
                                    if filter.max_inclusive { ok &= val <= max_s; } else { ok &= val < max_s; }
                                }
                            }
                            ok
                        },
                        (arrow::datatypes::DataType::Int32, Some(min_v), _) => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::Int32Array>().context("Invalid cast")?.value(i);
                            let mut ok = true;
                            if let Some(min_i) = min_v.as_i64() {
                                let min_i = min_i as i32;
                                if filter.min_inclusive { ok &= val >= min_i; } else { ok &= val > min_i; }
                                if ok {
                                    tracing::debug!("inverted_index_match: col={}, val={}, row_ids={:?}", filter.column, val, row_ids_list.value(i).as_any().downcast_ref::<arrow::array::UInt32Array>().unwrap().values());
                                }
                            }
                            if let Some(max_v) = &filter.max {
                                if let Some(max_i) = max_v.as_i64() {
                                    let max_i = max_i as i32;
                                    if filter.max_inclusive { ok &= val <= max_i; } else { ok &= val < max_i; }
                                }
                            }
                            ok
                        },
                        (arrow::datatypes::DataType::Int64, Some(min_v), _) => {
                             let val = key_array.as_any().downcast_ref::<arrow::array::Int64Array>().context("Invalid cast")?.value(i);
                             let mut ok = true;
                             if let Some(min_i) = min_v.as_i64() {
                                 if filter.min_inclusive { ok &= val >= min_i; } else { ok &= val > min_i; }
                             }
                             if let Some(max_v) = &filter.max {
                                 if let Some(max_i) = max_v.as_i64() {
                                     if filter.max_inclusive { ok &= val <= max_i; } else { ok &= val < max_i; }
                                 }
                             }
                             ok
                        },
                        (arrow::datatypes::DataType::Float64, Some(min_v), _) => {
                             let val = key_array.as_any().downcast_ref::<arrow::array::Float64Array>().context("Invalid cast")?.value(i);
                             let mut ok = true;
                             if let Some(min_f) = min_v.as_f64() {
                                 if filter.min_inclusive { ok &= val >= min_f; } else { ok &= val > min_f; }
                             }
                             if let Some(max_v) = &filter.max {
                                 if let Some(max_f) = max_v.as_f64() {
                                     if filter.max_inclusive { ok &= val <= max_f; } else { ok &= val < max_f; }
                                 }
                             }
                             ok
                        },
                        (arrow::datatypes::DataType::Date32, Some(min_v), _) => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::Date32Array>().context("Invalid cast")?.value(i);
                            let mut ok = true;
                            if let Some(min_i) = min_v.as_i64() {
                                let min_i = min_i as i32;
                                if filter.min_inclusive { ok &= val >= min_i; } else { ok &= val > min_i; }
                            }
                            if let Some(max_v) = &filter.max {
                                if let Some(max_i) = max_v.as_i64() {
                                    let max_i = max_i as i32;
                                    if filter.max_inclusive { ok &= val <= max_i; } else { ok &= val < max_i; }
                                }
                            }
                            ok
                        },
                         // Time32 range
                        (arrow::datatypes::DataType::Time32(unit), Some(min), _) => {
                            let val = match unit {
                                arrow::datatypes::TimeUnit::Second => 
                                    key_array.as_any().downcast_ref::<arrow::array::Time32SecondArray>().map(|a| a.value(i)),
                                arrow::datatypes::TimeUnit::Millisecond => 
                                    key_array.as_any().downcast_ref::<arrow::array::Time32MillisecondArray>().map(|a| a.value(i)),
                                _ => None,
                            };

                            if let Some(v) = val {
                                let min_i = min.as_i64().unwrap_or(i64::MIN) as i32;
                                if filter.min_inclusive { v >= min_i } else { v > min_i }
                            } else {
                                true // Default to true if type check fails to avoid false negatives? Or false? usually false.
                            }
                        },
                         // Time64 range
                        (arrow::datatypes::DataType::Time64(unit), Some(min), _) => {
                            let val = match unit {
                                arrow::datatypes::TimeUnit::Microsecond => 
                                    key_array.as_any().downcast_ref::<arrow::array::Time64MicrosecondArray>().map(|a| a.value(i)),
                                arrow::datatypes::TimeUnit::Nanosecond => 
                                    key_array.as_any().downcast_ref::<arrow::array::Time64NanosecondArray>().map(|a| a.value(i)),
                                _ => None,
                            };

                            if let Some(v) = val {
                                let min_i = min.as_i64().unwrap_or(i64::MIN);
                                if filter.min_inclusive { v >= min_i } else { v > min_i }
                            } else {
                                true
                            }
                        },
                        // Boolean equality
                        (arrow::datatypes::DataType::Boolean, Some(min), _) => {
                             let val = key_array.as_any().downcast_ref::<arrow::array::BooleanArray>().context("Invalid cast")?.value(i);
                             let target = min.as_bool().unwrap_or(false);

                             val == target
                        },
                        // Binary equality
                        (arrow::datatypes::DataType::Binary, Some(min), Some(max)) 
                            if min == max && filter.min_inclusive && filter.max_inclusive => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::BinaryArray>().context("Invalid cast")?.value(i);
                            // Assume filter value is string or bytes? JSON usually string.
                            if let Some(s) = min.as_str() {
                                val == s.as_bytes()
                            } else {
                                false
                            }
                        },
                        // Decimal128 range (Best effort f64 comparison for now)
                        (arrow::datatypes::DataType::Decimal128(_p, s), Some(min), _) => {
                             let val_i128 = key_array.as_any().downcast_ref::<arrow::array::Decimal128Array>().context("Invalid cast")?.value(i);
                             // Convert i128 to f64 for comparison against JSON number
                             // Value = i128 / 10^scale
                             let divisor = 10_f64.powi(*s as i32);
                             let val_f64 = val_i128 as f64 / divisor;
                             
                             let min_f = min.as_f64().unwrap_or(f64::MIN);
                             if filter.min_inclusive { val_f64 >= min_f } else { val_f64 > min_f }
                        },
                        _ => true 
                    };

                    if key_ok {
                        let row_ids = row_ids_list.value(i);
                         let row_ids_array = row_ids.as_any().downcast_ref::<arrow::array::UInt32Array>()
                             .ok_or_else(|| anyhow::anyhow!("Expected UInt32Array in inverted index row_ids"))?;
                        
                        let mut current_id = 0;
                        for ri in 0..row_ids_array.len() {
                            current_id += row_ids_array.value(ri);
                            bitmap.insert(current_id);
                        }
                    }
                }
            }
            bitmap
        } else {
            // Step 1 (fallback): Read scalar Index (.idx)
            let idx_path = self.resolve_object_path(&format!("{}.idx", filter_column));
            let idx_path_str = idx_path.to_string();
            
            // Check Cache
            if let Some(cached) = crate::core::cache::INDEX_CACHE.get_with_metrics(&format!("{}/{}", self.root_uri, idx_path_str), "index").await {
                cached.as_ref().clone()
            } else {
                 match self.store.get(&idx_path).await {
                     Ok(resp) => {
                         let index_bytes = resp.bytes().await?;
                         crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(index_bytes.len() as u64);
                         let bitmap = RoaringBitmap::deserialize_from(&index_bytes[..])?;
                         crate::core::cache::INDEX_CACHE.insert(format!("{}/{}", self.root_uri, idx_path_str), Arc::new(bitmap.clone())).await;
                         bitmap
                     },
                     Err(_) => {
                         // No index found
                         return Ok(None);
                     }
                 }
            }
        };
        
        // Step 1b: Handle Negation
        let mut final_bitmap = matching_bitmap;
        if filter.negated {
            if let Some(total) = self.config.record_count {
                let mut all = roaring::RoaringBitmap::new();
                all.insert_range(0..total as u32);
                final_bitmap = all - final_bitmap;
            } else {
                // If we don't know total rows, we can't safely negate via bitmap
                // Fallback to full scan by returning None
                return Ok(None);
            }
        }

        // Step 1c: Apply Deletes (Difference)
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
            final_bitmap -= deleted;
        }
        
        Ok(Some(final_bitmap))
    }

    #[tracing::instrument(skip(self, filter, target_schema))]
    pub async fn query_index_first(&self, filter: &crate::core::planner::QueryFilter, target_schema: Option<arrow::datatypes::SchemaRef>) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        let matching_bitmap = match self.get_scalar_filter_bitmap(filter).await? {
            Some(bm) => bm,
            None => return Err(anyhow::anyhow!("No index for column {}", filter.column)),
        };



        if matching_bitmap.is_empty() {
             return Ok(vec![]);
        }

        // Step 2: Configure Parquet Range Request with Row Selection
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();
        
        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE.get_with_metrics(&format!("{}/{}", self.root_uri, pq_path_str), "parquet_meta").await {
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
              let head_res = if let Some(s) = self.config.file_size {
                  Ok((None, s as usize))
              } else {
                  self.store.head(&pq_path).await.map(|m| {
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

              let reader = if let Some(m) = m_opt {
                  ParquetObjectReader::new(self.store.clone(), m.location)
              } else {
                  ParquetObjectReader::new(self.store.clone(), pq_path.clone())
              };

              let b_res = ParquetRecordBatchStreamBuilder::new(reader).await;
              let b = match b_res {
                  Ok(b) => b,
                  Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
              
                      return Ok(vec![]);
                  }
                  Err(e) => return Err(e.into()),
              };

              crate::core::cache::PARQUET_META_CACHE.insert(format!("{}/{}", self.root_uri, pq_path_str), (b.metadata().clone(), size)).await;
              b
        };
        
        builder = builder.with_batch_size(65536);
        
        // Apply column projection if specified (skip reading unused columns like embeddings)
        // Apply column projection/evolution
        let target_schema_ref = target_schema.clone();
        if let Some(schema) = &target_schema_ref {
            let parquet_schema = builder.metadata().file_metadata().schema_descr();
            let file_arrow_schema = builder.schema();
            let column_indices: Vec<usize> = schema.fields().iter()
                .filter_map(|field| file_arrow_schema.index_of(field.name()).ok())
                .collect();
            
            let projection = ProjectionMask::roots(parquet_schema, column_indices);
            builder = builder.with_projection(projection);
        }
        
        // Construct RowSelection from Bitmap
        let selection = self.bitmap_to_row_selection(&matching_bitmap, builder.metadata().file_metadata().num_rows() as usize);
        builder = builder.with_row_selection(selection);

        let mut stream = builder.build()?;
        
        
        let mut batches = Vec::new();
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            // Schema Evolution Mapping
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
                         let null_arr = arrow::array::new_null_array(field.data_type(), batch.num_rows());
                         new_columns.push(null_arr);
                     }
                 }
                batches.push(arrow::record_batch::RecordBatch::try_new(target.clone(), new_columns)?);
            } else {
                batches.push(batch);
            }
        }
        

        Ok(batches)
    }

    pub(crate) fn bitmap_to_row_selection(&self, bitmap: &RoaringBitmap, total_rows: usize) -> RowSelection {
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

}
