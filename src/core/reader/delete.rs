#![allow(unused)]
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
    pub async fn load_merged_deletes(&self) -> Result<RoaringBitmap> {
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
                     let path_clean = path_str.strip_prefix("file://").unwrap_or(path_str);
                     
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
                             tracing::warn!("Failed to read Iceberg delete file {}: {}", path_str, e);
                         }
                     }
                 } else {
                     // Native .del (RoaringBitmap)
                     let path = Path::from(path_str);
                     if let Ok(ret) = self.store.get(&path).await {
                         if let Ok(bytes) = ret.bytes().await {
                             crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(bytes.len() as u64);
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
                         crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(bytes.len() as u64);
                         match crate::core::puffin::read_deletion_vector_from_bytes(&bytes) {
                             Ok(dv_bitmap) => {
                                 deleted_bitmap |= dv_bitmap;
                             },
                             Err(e) => {
                                 tracing::warn!("Failed to deserialize deletion vector from {}: {}", puffin_file_path, e);
                             }
                         }
                     },
                     Err(e) => {
                         tracing::warn!("Failed to read deletion vector from Puffin file {}: {}", puffin_file_path, e);
                     }
                 }
             }
        }
        
        Ok(deleted_bitmap)
    }

    pub(crate) async fn load_equality_deletes(&self) -> Result<Vec<EqualityDelete>> {
        let mut results = Vec::new();
        
        for delete_file in &self.config.delete_files {
             if let crate::core::manifest::DeleteContent::Equality { equality_ids } = &delete_file.content {
                 let path_str = delete_file.file_path.as_str();

                 // Relativize path matches load_merged_deletes logic
                 let resolved_path = if path_str.starts_with("file://") {
                     let root_local = self.root_uri.strip_prefix("file://").unwrap_or(&self.root_uri).trim_end_matches('/');
                     let path_clean = path_str.strip_prefix("file://").unwrap_or(path_str);
                     
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
                                          Err(e) => tracing::warn!("Failed to concat equality delete values: {}", e),
                                      }
                                  }
                              }
                          } else {
                              tracing::warn!("Multi-column equality deletes not yet optimized");
                          }
                      }
                      Err(e) => tracing::warn!("Failed to read equality delete file {}: {}", resolved_path, e),
                 }
             }
        }
        Ok(results)
    }

}
