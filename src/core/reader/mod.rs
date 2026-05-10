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

/// Helper function to check if values in `col` distinct are in `values` set.
/// Returns a BooleanArray where true means the value is in the set.
pub(crate) fn check_is_in(col: &arrow::array::ArrayRef, values: &arrow::array::ArrayRef) -> Result<arrow::array::BooleanArray> {
    use arrow::datatypes::*;
    use arrow::array::Array;
    use std::collections::HashSet;
    
    match col.data_type() {
        DataType::Utf8 => {
             let col_arr = col.as_any().downcast_ref::<arrow::array::StringArray>()
                 .ok_or_else(|| anyhow::anyhow!("Expected StringArray in column for equality check"))?;
             let val_arr = arrow::compute::cast(values, &DataType::Utf8)?; // Ensure type match
             let val_arr = val_arr.as_any().downcast_ref::<arrow::array::StringArray>()
                 .ok_or_else(|| anyhow::anyhow!("Expected StringArray in values for equality check"))?;
             
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
             let col_arr = col_arr.as_any().downcast_ref::<arrow::array::Int64Array>()
                 .ok_or_else(|| anyhow::anyhow!("Expected Int64Array in column for equality check"))?;
             
             let val_arr = arrow::compute::cast(values, &DataType::Int64)?;
             let val_arr = val_arr.as_any().downcast_ref::<arrow::array::Int64Array>()
                 .ok_or_else(|| anyhow::anyhow!("Expected Int64Array in values for equality check"))?;
             
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
             let col_arr = col_arr.as_any().downcast_ref::<arrow::array::Int32Array>()
                 .ok_or_else(|| anyhow::anyhow!("Expected Int32Array in column for equality check"))?;
             
             let val_arr = arrow::compute::cast(values, &DataType::Int32)?;
             let val_arr = val_arr.as_any().downcast_ref::<arrow::array::Int32Array>()
                 .ok_or_else(|| anyhow::anyhow!("Expected Int32Array in values for equality check"))?;
             
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
             tracing::warn!("Unsupported generic equality check for type: {:?}", col.data_type());
             // Return false (no match) to be safe (don't delete anything)
             let result = arrow::array::BooleanArray::from(vec![false; col.len()]);
             Ok(result)
        }
    }
}

use anyhow::{Context, Result};
use roaring::RoaringBitmap;
use futures::stream::BoxStream;


pub mod scan;
pub mod filter;
pub mod delete;

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

    pub async fn get_parquet_metadata(&self) -> Result<Arc<ParquetMetaData>> {
        let path = self.resolve_object_path("parquet");
        let options = ArrowReaderOptions::new().with_page_index(true);
        let mut reader = ParquetObjectReader::new(self.store.clone(), path);
        let metadata = ArrowReaderMetadata::load_async(&mut reader, options).await?;
        Ok(metadata.metadata().clone())
    }

    pub async fn get_arrow_schema(&self) -> Result<arrow::datatypes::SchemaRef> {
        if let Some(s) = &self.iceberg_schema {
             return Ok(Arc::new(s.to_arrow()));
        }
        let meta = self.get_parquet_metadata().await?;
        let options = ArrowReaderOptions::new();
        let arrow_meta = ArrowReaderMetadata::try_new(meta, options)?;
        Ok(arrow_meta.schema().clone())
    }

    pub fn with_iceberg_schema(mut self, schema: crate::core::manifest::Schema) -> Self {
        self.iceberg_schema = Some(schema);
        self
    }

    pub fn get_segment_id(&self) -> &str {
        &self.config.segment_id
    }

    fn resolve_object_path(&self, extension: &str) -> Path {
        // 1. Get the base string and determine if it includes the filename
        let (base, has_filename) = if extension == "parquet" && self.config.parquet_path.is_some() {
            (self.config.parquet_path.as_ref().map(|p| p.as_str()).unwrap_or(""), true)
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
            let ids_col = b.column(0).as_any().downcast_ref::<Int32Array>().context("Invalid cast")?;
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
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().context("Invalid cast")?;
            all_ids.extend(ids.iter().map(|v| v.unwrap_or_default()));
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
