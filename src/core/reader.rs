use std::sync::Arc;
// use std::collections::HashSet;
use chrono::Utc;
use futures::StreamExt;
use bytes::Bytes;
use object_store::{path::Path, ObjectStore, ObjectMeta};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector, ArrowReaderMetadata, ArrowReaderOptions};
use crate::core::index::hnsw_ivf::HnswIvfIndex;
// use arrow::record_batch::RecordBatch;
use parquet::arrow::async_reader::{ParquetRecordBatchStreamBuilder, ParquetObjectReader};
use parquet::arrow::ProjectionMask;
use crate::SegmentConfig;
use crate::core::planner::FilterExpr;
use crate::core::index::VectorMetric;

/// Helper function to check if values in `col` distinct are in `values` set.
/// Returns a BooleanArray where true means the value is in the set.
fn check_is_in(col: &arrow::array::ArrayRef, values: &arrow::array::ArrayRef) -> Result<arrow::array::BooleanArray> {
    use arrow::datatypes::*;
    use arrow::array::Array;
    use std::collections::HashSet;
    
    match col.data_type() {
        DataType::Utf8 => {
             let col_arr = col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
             let val_arr = arrow::compute::cast(values, &DataType::Utf8)?; // Ensure type match
             let val_arr = val_arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
             
             let mut set = HashSet::with_capacity(val_arr.len());
             for i in 0..val_arr.len() {
                 if !val_arr.is_null(i) {
                     set.insert(val_arr.value(i));
                 }
             }
             
             let mut result = arrow::array::BooleanBuilder::with_capacity(col_arr.len());
             for i in 0..col_arr.len() {
                 if col_arr.is_null(i) {
                     result.append_value(false); // Nulls don't match (usually)
                 } else {
                     result.append_value(set.contains(col_arr.value(i)));
                 }
             }
             Ok(result.finish())
        },
        DataType::Int64 | DataType::Date64 | DataType::Timestamp(_, _) => {
             // Treat all as Int64 for comparison if possible, or cast
             let col_arr = arrow::compute::cast(col, &DataType::Int64)?;
             let col_arr = col_arr.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
             
             let val_arr = arrow::compute::cast(values, &DataType::Int64)?;
             let val_arr = val_arr.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
             
             let mut set = HashSet::with_capacity(val_arr.len());
             for i in 0..val_arr.len() {
                 if !val_arr.is_null(i) {
                     set.insert(val_arr.value(i));
                 }
             }
             
             let mut result = arrow::array::BooleanBuilder::with_capacity(col_arr.len());
             for i in 0..col_arr.len() {
                  if col_arr.is_null(i) {
                      result.append_value(false);
                  } else {
                      result.append_value(set.contains(&col_arr.value(i)));
                  }
             }
             Ok(result.finish())
        },
        DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
             let col_arr = arrow::compute::cast(col, &DataType::Int32)?;
             let col_arr = col_arr.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
             
             let val_arr = arrow::compute::cast(values, &DataType::Int32)?;
             let val_arr = val_arr.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
             
             let mut set = HashSet::with_capacity(val_arr.len());
             for i in 0..val_arr.len() {
                 if !val_arr.is_null(i) {
                     set.insert(val_arr.value(i));
                 }
             }
             
             let mut result = arrow::array::BooleanBuilder::with_capacity(col_arr.len());
             for i in 0..col_arr.len() {
                  if col_arr.is_null(i) {
                      result.append_value(false);
                  } else {
                      result.append_value(set.contains(&col_arr.value(i)));
                  }
             }
             Ok(result.finish())
        },
        _ => {
             // Fallback or warning
             println!("Warning: Unsupported generic equality check for type: {:?}", col.data_type());
             // Return false (no match) to be safe (don't delete anything)
             let result = arrow::array::BooleanArray::from(vec![false; col.len()]);
             Ok(result)
        }
    }
}

use anyhow::{Context, Result};
use roaring::RoaringBitmap;
use futures::stream::BoxStream;

pub struct EqualityDelete {
    pub column_name: String,
    pub values: arrow::array::ArrayRef,
}
// use url::Url; // Unused


pub struct HybridReader {
    pub config: SegmentConfig,
    pub store: Arc<dyn ObjectStore>,
    pub root_uri: String,
    pub iceberg_schema: Option<crate::core::manifest::Schema>,
}

impl HybridReader {
    pub fn new(config: SegmentConfig, store: Arc<dyn ObjectStore>, root_uri: &str) -> Self {
                Self { config, store, root_uri: root_uri.to_string(), iceberg_schema: None }
        }

    pub fn with_iceberg_schema(mut self, schema: crate::core::manifest::Schema) -> Self {
        self.iceberg_schema = Some(schema);
        self
    }

    pub fn get_segment_id(&self) -> &str {
        &self.config.segment_id
    }

    /// Resolve a path for ObjectStore operations (relative to store root)
    fn resolve_object_path(&self, extension: &str) -> Path {
        // 1. Get the base string and determine if it includes the filename
        let (base, has_filename) = if extension == "parquet" && self.config.parquet_path.is_some() {
            (self.config.parquet_path.as_ref().unwrap().as_str(), true)
        } else {
            (self.config.base_path.as_str(), false)
        };

        let filename = if has_filename {
            String::new()
        } else {
            format!("{}.{}", self.config.segment_id, extension)
        };

        // 2. Helper to get local path from URI or absolute path
        fn to_local(s: &str) -> &str {
            s.strip_prefix("file://").unwrap_or(s)
        }

        let root_local = to_local(&self.root_uri).trim_end_matches('/');
        let base_local = to_local(base).trim_end_matches('/');

        // 3. Relativize
        let mut rel = if !root_local.is_empty() && base_local.starts_with(root_local) {
            let r = &base_local[root_local.len()..];
            r.trim_start_matches('/').to_string()
        } else if base.contains("://") {
            // If it's a URI but not matching root, try to parse it
            if let Ok(url) = url::Url::parse(base) {
                url.path().trim_start_matches('/').to_string()
            } else {
                base_local.trim_start_matches('/').to_string()
            }
        } else {
            // Fallback: just use the local part
            base_local.trim_start_matches('/').to_string()
        };

        // Append filename if needed
        if !filename.is_empty() {
            if !rel.is_empty() {
                rel.push('/');
            }
            rel.push_str(&filename);
        }

        Path::from(rel)
    }

    /// Load and merge all .del files into a single RoaringBitmap
    async fn load_merged_deletes(&self) -> Result<RoaringBitmap> {
        let mut deleted_bitmap = RoaringBitmap::new();
        
        // Determine target path for Iceberg delete matching
        let target_path = if let Some(p) = &self.config.parquet_path {
            p.clone()
        } else {
             format!("{}/{}.parquet", self.config.base_path, self.config.segment_id)
        };

        for delete_file in &self.config.delete_files {
             // Handle Position deletes (RoaringBitmap files or Iceberg Parquet)
             if let crate::core::manifest::DeleteContent::Position = &delete_file.content {
                 let path_str = delete_file.file_path.as_str();
                 
                 // Relativize path
                 let resolved_path = if path_str.starts_with("file://") {
                     let root_local = self.root_uri.strip_prefix("file://").unwrap_or(&self.root_uri).trim_end_matches('/');
                     let path_clean = if path_str.starts_with("file:///") { &path_str[7..] } else { &path_str[7..] };
                     
                     if !root_local.is_empty() && path_clean.starts_with(root_local) {
                         path_clean[root_local.len()..].trim_start_matches('/').to_string()
                     } else {
                         path_clean.trim_start_matches('/').to_string()
                     }
                 } else {
                     path_str.to_string()
                 };
                 
                 // Check if it is an Iceberg Parquet Delete File
                 if resolved_path.ends_with(".parquet") || resolved_path.ends_with(".avro") { 
                     let reader = crate::core::iceberg::PositionDeleteReader::new(self.store.clone());
                     match reader.read_deletes(&resolved_path, &target_path).await {
                         Ok(positions) => {
                             for pos in positions {
                                 // Safely cast to u32, ignoring if out of bounds (current Roaring limitation)
                                 if pos >= 0 && pos <= u32::MAX as i64 {
                                     deleted_bitmap.insert(pos as u32);
                                 }
                             }
                         },
                         Err(e) => {
                             println!("Warning: Failed to read Iceberg delete file {}: {}", path_str, e);
                         }
                     }
                 } else {
                     // Native .del (RoaringBitmap)
                     let path = Path::from(path_str);
                     if let Ok(ret) = self.store.get(&path).await {
                         if let Ok(bytes) = ret.bytes().await {
                             if let Ok(bm) = RoaringBitmap::deserialize_from(&bytes[..]) {
                                 deleted_bitmap |= bm;
                             }
                         }
                     }
                 }
             }
             // Handle V3 Deletion Vectors (Puffin files)
             else if let crate::core::manifest::DeleteContent::DeletionVector { 
                 puffin_file_path, 
                 content_offset, 
                 content_size_in_bytes 
             } = &delete_file.content {
                 // Read the deletion vector blob from the Puffin file
                 let path = Path::from(puffin_file_path.as_str());
                 match self.store.get_range(&path, (*content_offset as u64)..((*content_offset + *content_size_in_bytes) as u64)).await {
                     Ok(bytes) => {
                         match crate::core::puffin::read_deletion_vector_from_bytes(&bytes) {
                             Ok(dv_bitmap) => {
                                 deleted_bitmap |= dv_bitmap;
                             },
                             Err(e) => {
                                 println!("Warning: Failed to deserialize deletion vector from {}: {}", puffin_file_path, e);
                             }
                         }
                     },
                     Err(e) => {
                         println!("Warning: Failed to read deletion vector from Puffin file {}: {}", puffin_file_path, e);
                     }
                 }
             }
        }
        
        Ok(deleted_bitmap)
    }

    /// Load all Equality Delete files and return a bitset-like filter or HashSet
    /// For now, we return a list of (equality_ids, RecordBatch) pairs.
    /// Load all Equality Delete files and return a list of optimized EqualityFilters.
    async fn load_equality_deletes(&self) -> Result<Vec<EqualityDelete>> {
        let mut results = Vec::new();
        
        for delete_file in &self.config.delete_files {
             if let crate::core::manifest::DeleteContent::Equality { equality_ids } = &delete_file.content {
                 let path_str = delete_file.file_path.as_str();

                 // Relativize path matches load_merged_deletes logic
                 let resolved_path = if path_str.starts_with("file://") {
                     let root_local = self.root_uri.strip_prefix("file://").unwrap_or(&self.root_uri).trim_end_matches('/');
                     let path_clean = if path_str.starts_with("file:///") { &path_str[7..] } else { &path_str[7..] };
                     
                     if !root_local.is_empty() && path_clean.starts_with(root_local) {
                         path_clean[root_local.len()..].trim_start_matches('/').to_string()
                     } else {
                         path_clean.trim_start_matches('/').to_string()
                     }
                 } else {
                     path_str.to_string()
                 };

                 // Use provided schema or return error
                 let schema = if let Some(s) = &self.iceberg_schema {
                     s.clone()
                 } else {
                     return Err(anyhow::anyhow!("Cannot apply equality deletes (ID based) without table schema in HybridReader"));
                 };

                 let iceberg_reader = crate::core::iceberg::EqualityDeleteReader::new(self.store.clone());
                 match iceberg_reader.read_equality_deletes(&resolved_path, equality_ids, &schema).await {
                      Ok(batches) => {
                          if equality_ids.len() == 1 {
                              let field_id = equality_ids[0];
                              if let Some(field) = schema.fields.iter().find(|f| f.id == field_id) {
                                  let col_name = field.name.clone();
                                  
                                  // Collect all values for this column from all batches
                                  let mut arrays = Vec::new();
                                  for batch in batches {
                                      arrays.push(batch.column(0).clone());
                                  }

                                  if !arrays.is_empty() {
                                      let array_refs: Vec<&dyn arrow::array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
                                      match arrow::compute::concat(&array_refs) {
                                          Ok(combined_values) => {
                                              results.push(EqualityDelete {
                                                  column_name: col_name,
                                                  values: combined_values,
                                              });
                                          },
                                          Err(e) => println!("Warning: Failed to concat equality delete values: {}", e),
                                      }
                                  }
                              }
                          } else {
                              println!("Warning: Multi-column equality deletes not yet optimized");
                          }
                      }
                      Err(e) => println!("Warning: Failed to read equality delete file {}: {}", resolved_path, e),
                 }
             }
        }
        Ok(results)
    }

    /// Helper: Get RoaringBitmap of rows matching a scalar filter using indexes
    pub async fn get_scalar_filter_bitmap(&self, filter: &crate::core::planner::QueryFilter) -> Result<Option<RoaringBitmap>> {
        let filter_column = &filter.column;
        
        // Step 1: Check if we have inverted index files for this column
        let inv_idx_info = self.config.index_files.iter()
            .find(|f| f.index_type == "inverted" && f.column_name.as_deref() == Some(filter_column));
        
        let mut matching_bitmap = if let Some(idx_info) = inv_idx_info {
            let inv_path_str = &idx_info.file_path;
            // Use Inverted Index (Value -> RowIDs)
            // 1. Check Object Cache (Decoded RecordBatches)
            let cache_key = if let Some(offset) = idx_info.offset {
                 format!("{}/{}:{}", self.root_uri, inv_path_str, offset)
            } else {
                 format!("{}/{}", self.root_uri, inv_path_str)
            };

            let batches = if let Some(cached) = crate::core::cache::INVERTED_INDEX_CACHE.get(&cache_key).await {
                // println!("Cache HIT for {}", inv_path_str);
                cached.as_ref().clone()
            } else {
                // println!("Cache MISS for {}", inv_path_str);
                // Cache Miss - Load from Disk/Byte Cache
                let inv_path = Path::from(inv_path_str.as_str());
                
                let inv_bytes = match crate::core::cache::BYTE_CACHE.get(&cache_key).await {
                    Some(cached) => cached.as_ref().clone(),
                    None => {
                        let bytes = if let (Some(offset), Some(length)) = (idx_info.offset, idx_info.length) {
                             // Puffin Blob: Byte Range Read
                             self.store.get_range(&inv_path, (offset as u64)..(offset as u64 + length as u64)).await?
                                 .to_vec()
                        } else {
                             // Full File read
                             match self.store.get(&inv_path).await {
                                 Ok(res) => res.bytes().await?.to_vec(),
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
                let mut reader = builder.build()?;
                
                let mut decoded = Vec::new();
                while let Some(batch_result) = reader.next() {
                    decoded.push(batch_result?);
                }
                
                crate::core::cache::INVERTED_INDEX_CACHE.insert(format!("{}/{}", self.root_uri, inv_path_str), Arc::new(decoded.clone())).await;
                decoded
            };
            
            let mut bitmap = RoaringBitmap::new();
            
            // The inverted index schema is [key, row_ids (List<UInt32>)]
            for batch in batches {
                let key_array = batch.column(0);
                let row_ids_list = batch.column(1).as_any().downcast_ref::<arrow::array::ListArray>().unwrap();
                
                // Perform range/value filtering on inverted index keys
                for i in 0..batch.num_rows() {
                    let key_ok = match (key_array.data_type(), &filter.min, &filter.max) {
                        // String equality
                        (arrow::datatypes::DataType::Utf8, Some(min_val), Some(max_val)) 
                            if min_val == max_val && filter.min_inclusive && filter.max_inclusive => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap().value(i);
                            let target = min_val.as_str().unwrap_or("");
                            val == target
                        },
                        // Date32 equality
                        (arrow::datatypes::DataType::Date32, Some(min_val), Some(max_val))
                            if min_val == max_val && filter.min_inclusive && filter.max_inclusive => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap().value(i);
                            let target = min_val.as_i64().unwrap_or(0) as i32;
                            val == target
                        },
                        // Date32 range
                        (arrow::datatypes::DataType::Date32, Some(min), _) => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap().value(i);
                            let min_i = min.as_i64().unwrap_or(i64::MIN) as i32;
                            if filter.min_inclusive { val >= min_i } else { val > min_i }
                        },
                        // Float64 range
                        (arrow::datatypes::DataType::Float64, Some(min), _) => {
                             let val = key_array.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap().value(i);
                             let min_f = min.as_f64().unwrap_or(f64::MIN);
                             val > min_f
                        },
                        // Int64 range
                        (arrow::datatypes::DataType::Int64, Some(min), _) => {
                             let val = key_array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(i);
                             let min_i = min.as_i64().unwrap_or(i64::MIN);
                             val > min_i
                        },
                        // Int32 range
                        (arrow::datatypes::DataType::Int32, Some(min), _) => {
                             let val = key_array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap().value(i);
                             let min_i = min.as_i64().unwrap_or(i64::MIN) as i32;
                             val > min_i
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
                             let val = key_array.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap().value(i);
                             let target = min.as_bool().unwrap_or(false);

                             val == target
                        },
                        // Binary equality
                        (arrow::datatypes::DataType::Binary, Some(min), Some(max)) 
                            if min == max && filter.min_inclusive && filter.max_inclusive => {
                            let val = key_array.as_any().downcast_ref::<arrow::array::BinaryArray>().unwrap().value(i);
                            // Assume filter value is string or bytes? JSON usually string.
                            if let Some(s) = min.as_str() {
                                val == s.as_bytes()
                            } else {
                                false
                            }
                        },
                        // Decimal128 range (Best effort f64 comparison for now)
                        (arrow::datatypes::DataType::Decimal128(_p, s), Some(min), _) => {
                             let val_i128 = key_array.as_any().downcast_ref::<arrow::array::Decimal128Array>().unwrap().value(i);
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
                        let row_ids_array = row_ids.as_any().downcast_ref::<arrow::array::UInt32Array>().unwrap();
                        for ri in 0..row_ids_array.len() {
                            bitmap.insert(row_ids_array.value(ri));
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
            if let Some(cached) = crate::core::cache::INDEX_CACHE.get(&format!("{}/{}", self.root_uri, idx_path_str)).await {
                cached.as_ref().clone()
            } else {
                 match self.store.get(&idx_path).await {
                     Ok(resp) => {
                         let index_bytes = resp.bytes().await?;
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
        
        // Step 1b: Apply Deletes (Difference)
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
            matching_bitmap -= deleted;
        }
        
        Ok(Some(matching_bitmap))
    }

    /// The "Serverless Selectivity" Filter Query
    /// 1. Reads ONLY the tiny index file (Range Request or full GET if small).
    /// 2. Determines matching Row IDs.
    /// 3. Reads ONLY those rows from Parquet (Range Requests via RowSelection).
    /// 4. Applies column projection to read only specified columns (skips embeddings etc).
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
        
        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE.get(&format!("{}/{}", self.root_uri, pq_path_str)).await {
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

    /// Stream ALL rows from the segment (Scanning)
    /// Used for Compaction or Full Table Scans
    /// Supports column projection/evolution by mapping physical schema to logical schema
    pub async fn stream_all(&self, target_schema: Option<arrow::datatypes::SchemaRef>) -> Result<BoxStream<'static, Result<arrow::record_batch::RecordBatch>>> {
        self.stream_row_groups(None, target_schema).await
    }

    /// Stream specific Row Groups from the segment
    /// Used for Distributed Reading (Split-level)
    pub async fn stream_row_groups(&self, row_groups: Option<&[usize]>, target_schema: Option<arrow::datatypes::SchemaRef>) -> Result<BoxStream<'static, Result<arrow::record_batch::RecordBatch>>> {
        println!("DEBUG: stream_row_groups called");
        let store = self.config.data_store.clone().unwrap_or_else(|| self.store.clone());
        let pq_path = self.resolve_object_path("parquet");
        let pq_path_str = pq_path.to_string();
        
        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE.get(&format!("{}/{}", self.root_uri, pq_path_str)).await {
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

             crate::core::cache::PARQUET_META_CACHE.insert(format!("{}/{}", self.root_uri, pq_path_str), (b.metadata().clone(), size as usize)).await;
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
            let column_indices: Vec<usize> = schema.fields().iter()
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
             let mut batch = res.map_err(|e| anyhow::Error::from(e))?;
             
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
                          let null_arr = arrow::array::new_null_array(field.data_type(), batch.num_rows());
                          new_columns.push(null_arr);
                      }
                  }
                  
                  batch = if target.fields().is_empty() { 
                      arrow::record_batch::RecordBatch::try_new_with_options(target.clone(), vec![], &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())))? 
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
                                        if let Ok(new_mask) = arrow::compute::and(&keep_mask, &not_delete) {
                                            keep_mask = new_mask;
                                        }
                                   }
                               },
                               Err(e) => println!("Warning: Failed to apply equality delete filter on {}: {}", delete.column_name, e),
                          }
                      }
                  }
                 
                 batch = arrow::compute::filter_record_batch(&batch, &keep_mask)?;
             }

             Ok(batch)
        });
        
        Ok(mapped_stream.boxed())
    }

    /// Vector search that returns results with distances for global ranking
    /// Returns: Vec<(RecordBatch, Vec<f32>)> where Vec<f32> are distances for each row
    /// 
    /// Supports two index types:
    /// 1. HNSW-IVF (hybrid): Checks for `.centroids.parquet` - more memory efficient
    /// 2. Plain HNSW: Falls back to `.hnsw.graph` files
    pub async fn vector_search_index(&self, column: &str, query: &crate::core::index::VectorValue, k: usize, filter: Option<&FilterExpr>, metric: VectorMetric, ef_search: Option<usize>) -> Result<Vec<(arrow::record_batch::RecordBatch, Vec<f32>)>> {
        // Resolve scalar filter to combined bitmap if present
        let allowed_bitmap = if let Some(expr) = filter {
             let sub_filters = expr.extract_and_conditions();
             let mut combined_bitmap: Option<RoaringBitmap> = None;
             
             for sub_f in sub_filters {
                 if let Ok(Some(bm)) = self.get_scalar_filter_bitmap(&sub_f).await {
                     match combined_bitmap {
                         Some(ref mut existing) => { *existing &= bm; },
                         None => { combined_bitmap = Some(bm); }
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
        
        // Determine Index Path
        // If base_path is remote (s3://), we first check if we have the index locally in CWD (common for writers/benchmarks)
        // This bypasses the need for full download logic in this iteration.
        
        // Determine Index Path
        let vector_idx_info = self.config.index_files.iter()
            .find(|f| f.index_type == "vector" && f.column_name.as_deref() == Some(column));

        let matches = if let Some(idx_info) = vector_idx_info {
             self.search_hnsw_ivf(idx_info, query, k, &allowed_bitmap, metric, ef_search).await?
        } else {
             // Fallback to convention-based path if no manifest entry (unlikely now)
             let idx_path = self.resolve_object_path(&column).to_string();
             let idx_info = crate::core::manifest::IndexFile {
                 file_path: idx_path,
                 index_type: "vector".to_string(),
                 column_name: Some(column.to_string()),
                 blob_type: None,
                 offset: None,
                 length: None,
             };
             self.search_hnsw_ivf(&idx_info, query, k, &allowed_bitmap, metric, ef_search).await?
        };
        
        if matches.is_empty() {
            return Ok(vec![]);
        }
        
        // Build bitmap and track distances
        let mut bitmap = RoaringBitmap::new();
        let mut row_distances: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for (row_id, distance) in matches {
            bitmap.insert(row_id as u32);
            row_distances.insert(row_id as u32, distance);
        }
        
        // Apply Deletes
        let deleted = self.load_merged_deletes().await?;
        if !deleted.is_empty() {
             bitmap -= &deleted;  // Borrow instead of move
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

        let mut builder = if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE.get(&format!("{}/{}", self.root_uri, pq_path_str)).await {
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
             let object_meta = self.store.head(&pq_path).await.context("Failed to get segment metadata")?;
             let size = object_meta.size;
             let reader = ParquetObjectReader::new(self.store.clone(), object_meta.location);
             
             let b = ParquetRecordBatchStreamBuilder::new(reader).await?;
             // FIX: Remove Arc::new
             crate::core::cache::PARQUET_META_CACHE.insert(format!("{}/{}", self.root_uri, pq_path_str), (b.metadata().clone(), size as usize)).await;
             b
        };
        
        let selection = self.bitmap_to_row_selection(&bitmap, builder.metadata().file_metadata().num_rows() as usize);
        builder = builder.with_row_selection(selection);
        
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
            
            // Extract distances for this batch

            
            let final_batch = if rows_to_process < num_rows {
                // If we coalesced segments in RowSelection, we must slice the batch
                // to match the specific rows in the bitmap.
                // NOTE: This assumes RowSelection and Bitmap iteration are perfectly aligned.
                batch.slice(0, rows_to_process)
            } else {
                batch
            };
            
            current_offset += rows_to_process;
            results.push((final_batch, batch_distances));
        }
        
        Ok(results)
    }

    /// Search using HNSW-IVF hybrid index (more memory efficient)
    async fn search_hnsw_ivf(
        &self,
        idx_info: &crate::core::manifest::IndexFile,
        query: &crate::core::index::VectorValue,
        k: usize,
        allowed_bitmap: &Option<RoaringBitmap>,
        _metric: VectorMetric,
        ef_search: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let idx_path_str = idx_info.file_path.clone();
        
        let cache_key = if let Some(offset) = idx_info.offset {
             format!("{}/{}:{}", self.root_uri, idx_path_str, offset)
        } else {
             format!("{}/{}", self.root_uri, idx_path_str)
        };
        
        // Load HNSW-IVF index
        let hnsw_ivf = if idx_info.blob_type.is_some() {
             HnswIvfIndex::load_puffin_async(self.store.clone(), &idx_path_str).await?
        } else {
             HnswIvfIndex::load_async_with_cache_key(self.store.clone(), &idx_path_str, &cache_key).await?
        };
        
        // Search with HNSW-IVF
        let query_clone = query.clone();
        let allowed_bm_clone = allowed_bitmap.clone();
        let hnsw_ivf_clone = hnsw_ivf.clone();
        
        // Determine n_probe based on filtering or session config
        let n_probe = if allowed_bitmap.is_some() { 20 } else { 10 };
        
        let matches = tokio::task::spawn_blocking(move || {
            hnsw_ivf_clone.search(&query_clone, k, n_probe, allowed_bm_clone.as_ref())
        }).await?;
        
        Ok(matches)
    }

    /// Convert RoaringBitmap to Parquet RowSelection
    fn bitmap_to_row_selection(&self, bitmap: &RoaringBitmap, total_rows: usize) -> RowSelection {
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
                    let end = current_end.unwrap();
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
            let end = current_end.unwrap();
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



#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_reader_with_deletes() -> Result<()> {
        let store = Arc::new(InMemory::new());
        
        // 1. Write Parquet File (ids: 0..100)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Utf8, true),
        ]));
        
        let ids = Int32Array::from_iter_values(0..100);
        let vals = StringArray::from_iter_values((0..100).map(|i| format!("val_{}", i)));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(ids),
            Arc::new(vals),
        ])?;
        
        // Write to store
        let path = Path::from("seg_1.parquet");
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, schema.clone(), None)?;
        writer.write(&batch)?;
        writer.close()?;
        store.put(&path, buf.into()).await?;
        
        // 2. Write Delete File (delete ids 10, 20, 30..90)
        // Row indices match ids since we wrote sequentially from 0.
        let mut deleted_bitmap = RoaringBitmap::new();
        for i in (10..100).step_by(10) {
            deleted_bitmap.insert(i as u32);
        }
        
        // Serialize
        let mut del_buf = Vec::new();
        deleted_bitmap.serialize_into(&mut del_buf)?;
        let del_len = del_buf.len();
        let del_path = Path::from("seg_1.del");
        store.put(&del_path, del_buf.into()).await?;
        
        // 3. Configure Reader
        let config = SegmentConfig::new("", "seg_1")
            .with_delete_files(vec![crate::core::manifest::DeleteFile {
                file_path: del_path.to_string(),
                content: crate::core::manifest::DeleteContent::Position,
                file_size_bytes: del_len as i64,
                record_count: deleted_bitmap.len() as i64,
                partition_values: std::collections::HashMap::new(),
            }]);
            
        let reader = HybridReader::new(config, store.clone(), "memory://test");
        
        // 4. Stream All and Verify (None = all columns)
        let mut stream = reader.stream_all(None as Option<Arc<Schema>>).await?;
        let mut count = 0;
        while let Some(batch_res) = stream.next().await {
            let b = batch_res?;
            count += b.num_rows();
            
            // Verify rows 10,20... are gone.
            let ids_col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..ids_col.len() {
                let id = ids_col.value(i);
                assert!(id % 10 != 0 || id == 0, "Row {} should have been deleted", id); 
            }
        }
        
        // Total rows should be 100 - 9 = 91. (10, 20, 30, 40, 50, 60, 70, 80, 90)
        assert_eq!(count, 91);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_row_group_reading() -> Result<()> {
        let store = Arc::new(InMemory::new());
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Utf8, true),
        ]));
        
        // Write Parquet with 2 Row Groups
        // RG1: 0..5
        // RG2: 5..10
        let path = Path::from("seg_rg.parquet");
        let mut buf = Vec::new();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_size(5)
            .build();
            
        let mut writer = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props))?;
        
        let ids = Int32Array::from_iter_values(0..10);
        let vals = StringArray::from_iter_values((0..10).map(|i| format!("val_{}", i)));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(ids),
            Arc::new(vals),
        ])?;
        
        writer.write(&batch)?;
        writer.close()?;
        store.put(&path, buf.into()).await?;
        
        // Setup Reader
        let config = SegmentConfig::new("", "seg_rg");
        let reader = HybridReader::new(config, store, "memory://test");
        
        // 1. Read Only RG 1 (rows 5..10)
        let rgs = vec![1];
        let mut stream = reader.stream_row_groups(Some(&rgs), None).await?;
        let mut count = 0;
        let mut all_ids = Vec::new();
        while let Some(res) = stream.next().await {
            let b = res?;
            count += b.num_rows();
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            all_ids.extend(ids.iter().map(|v| v.unwrap()));
        }
        
        assert_eq!(count, 5);
        assert_eq!(all_ids, vec![5, 6, 7, 8, 9]);
        
        // 2. Read RG 0 with Column Projection (only "val")
        let rgs = vec![0];
        let projection_schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Int32, true),
        ]));
        let mut stream = reader.stream_row_groups(Some(&rgs), Some(projection_schema)).await?;
        let mut count = 0;
        while let Some(res) = stream.next().await {
            let b = res?;
            count += b.num_rows();
            assert_eq!(b.num_columns(), 1);
            assert_eq!(b.schema().field(0).name(), "val");
        }
        assert_eq!(count, 5); // 0..5
        
        Ok(())
    }
}
