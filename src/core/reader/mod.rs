// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::SegmentConfig;
use anyhow::Result;
use object_store::{path::Path, ObjectStore};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::file::metadata::ParquetMetaData;
use std::sync::Arc;

/// Helper function to check if values in `col` distinct are in `values` set.
/// Returns a BooleanArray where true means the value is in the set.
pub(crate) fn check_is_in(
    col: &arrow::array::ArrayRef,
    values: &arrow::array::ArrayRef,
) -> Result<arrow::array::BooleanArray> {
    use arrow::array::Array;
    use arrow::datatypes::*;
    use std::collections::HashSet;

    match col.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            let col_casted = arrow::compute::cast(col, &DataType::Utf8)?;
            let col_arr = col_casted
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected StringArray in column for equality check")
                })?;
            let val_arr = arrow::compute::cast(values, &DataType::Utf8)?;
            let val_arr = val_arr
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected StringArray in values for equality check")
                })?;

            let mut set = HashSet::with_capacity(val_arr.len());
            for i in 0..val_arr.len() {
                if !val_arr.is_null(i) {
                    set.insert(val_arr.value(i));
                }
            }

            let mut result = arrow::array::BooleanBuilder::with_capacity(col_arr.len());
            for i in 0..col_arr.len() {
                if col_arr.is_null(i) {
                    result.append_value(false); // Nulls don't match
                } else {
                    result.append_value(set.contains(col_arr.value(i)));
                }
            }
            Ok(result.finish())
        }
        DataType::Int64
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Time64(_)
        | DataType::UInt32 => {
            let col_arr = arrow::compute::cast(col, &DataType::Int64)?;
            let col_arr = col_arr
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected Int64Array in column for equality check")
                })?;

            let val_arr = arrow::compute::cast(values, &DataType::Int64)?;
            let val_arr = val_arr
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected Int64Array in values for equality check")
                })?;

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
        }
        DataType::Int32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Int8
        | DataType::Int16
        | DataType::UInt8
        | DataType::UInt16 => {
            let col_arr = arrow::compute::cast(col, &DataType::Int32)?;
            let col_arr = col_arr
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected Int32Array in column for equality check")
                })?;

            let val_arr = arrow::compute::cast(values, &DataType::Int32)?;
            let val_arr = val_arr
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected Int32Array in values for equality check")
                })?;

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
        }
        DataType::UInt64 => {
            let col_arr = arrow::compute::cast(col, &DataType::UInt64)?;
            let col_arr = col_arr
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected UInt64Array in column for equality check")
                })?;

            let val_arr = arrow::compute::cast(values, &DataType::UInt64)?;
            let val_arr = val_arr
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected UInt64Array in values for equality check")
                })?;

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
        }
        DataType::Boolean => {
            let col_arr = arrow::compute::cast(col, &DataType::Boolean)?;
            let col_arr = col_arr
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected BooleanArray in column for equality check")
                })?;
            let val_arr = arrow::compute::cast(values, &DataType::Boolean)?;
            let val_arr = val_arr
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Expected BooleanArray in values for equality check")
                })?;

            let mut has_true = false;
            let mut has_false = false;
            for i in 0..val_arr.len() {
                if !val_arr.is_null(i) {
                    if val_arr.value(i) {
                        has_true = true;
                    } else {
                        has_false = true;
                    }
                }
            }

            let mut result = arrow::array::BooleanBuilder::with_capacity(col_arr.len());
            for i in 0..col_arr.len() {
                if col_arr.is_null(i) {
                    result.append_value(false);
                } else {
                    result.append_value(if col_arr.value(i) {
                        has_true
                    } else {
                        has_false
                    });
                }
            }
            Ok(result.finish())
        }
        _ => {
            // Fallback or warning
            tracing::warn!(
                "Unsupported generic equality check for type: {:?}",
                col.data_type()
            );
            // Return false (no match) to be safe (don't delete anything)
            let result = arrow::array::BooleanArray::from(vec![false; col.len()]);
            Ok(result)
        }
    }
}
pub mod delete;
pub mod filter;
pub mod scan;

#[derive(Clone)]
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
    cached_deletes: tokio::sync::OnceCell<roaring::RoaringBitmap>,
}

impl HybridReader {
    pub fn new(config: SegmentConfig, store: Arc<dyn ObjectStore>, root_uri: &str) -> Self {
        Self {
            config,
            store,
            root_uri: root_uri.to_string(),
            iceberg_schema: None,
            cached_deletes: tokio::sync::OnceCell::new(),
        }
    }

    pub(crate) fn data_store(&self) -> &Arc<dyn ObjectStore> {
        self.config.data_store.as_ref().unwrap_or(&self.store)
    }

    pub(crate) fn cache_root(&self) -> &str {
        self.root_uri.trim_end_matches('/')
    }

    pub async fn get_parquet_metadata(&self) -> Result<Arc<ParquetMetaData>> {
        let path = self.resolve_object_path("parquet");
        let options = ArrowReaderOptions::new().with_page_index(true);
        let mut reader = ParquetObjectReader::new(self.data_store().clone(), path);
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

    pub(crate) fn segment_dir_prefix(&self) -> String {
        let dir = if let Some(pq) = self
            .config
            .parquet_path
            .as_deref()
            .filter(|p| !p.is_empty())
        {
            let rel_pq = crate::core::segment::relativize_against(&self.root_uri, pq);
            rel_pq
                .rsplit_once('/')
                .map(|(parent, _)| parent.to_string())
                .unwrap_or_default()
        } else {
            crate::core::segment::relativize_against(&self.root_uri, &self.config.base_path)
        };
        dir.trim_matches('/').to_string()
    }

    pub(crate) fn resolve_index_path(&self, raw_path: &str) -> String {
        let rel = crate::core::segment::relativize_against(&self.root_uri, raw_path);
        if rel.contains('/') {
            rel
        } else {
            let dir = self.segment_dir_prefix();
            crate::core::segment::ensure_prefix(&dir, &rel)
        }
    }

    fn resolve_object_path(&self, extension: &str) -> Path {
        let (base, has_filename) = if extension == "parquet" {
            if let Some(pq) = self
                .config
                .parquet_path
                .as_deref()
                .filter(|p| !p.is_empty())
            {
                (pq, true)
            } else {
                (self.config.base_path.as_str(), false)
            }
        } else {
            (self.config.base_path.as_str(), false)
        };

        let rel = crate::core::segment::relativize_against(&self.root_uri, base);
        let rel = if has_filename {
            rel
        } else {
            let filename = format!("{}.{}", self.config.segment_id, extension);
            crate::core::segment::ensure_prefix(&rel, &filename)
        };

        Path::parse(&rel).unwrap_or_else(|_| Path::from(rel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use futures::StreamExt;
    use object_store::memory::InMemory;
    use parquet::arrow::ArrowWriter;
    use roaring::RoaringBitmap;
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
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vals)])?;

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
        let config = SegmentConfig::new("", "seg_1").with_delete_files(vec![
            crate::core::manifest::DeleteFile {
                file_path: del_path.to_string(),
                content: crate::core::manifest::DeleteContent::Position,
                file_size_bytes: del_len as i64,
                record_count: deleted_bitmap.len() as i64,
                partition_values: std::collections::HashMap::new(),
            },
        ]);

        let reader = HybridReader::new(config, store.clone(), "memory://test");

        // 4. Stream All and Verify (None = all columns)
        let mut stream = reader.stream_all(None as Option<Arc<Schema>>).await?;
        let mut count = 0;
        while let Some(batch_res) = stream.next().await {
            let b = batch_res?;
            count += b.num_rows();

            // Verify rows 10,20... are gone.
            let ids_col = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("Invalid cast")?;
            for i in 0..ids_col.len() {
                let id = ids_col.value(i);
                assert!(
                    id % 10 != 0 || id == 0,
                    "Row {} should have been deleted",
                    id
                );
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
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vals)])?;

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
            let ids = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("Invalid cast")?;
            all_ids.extend(ids.iter().map(|v| v.unwrap_or_default()));
        }

        assert_eq!(count, 5);
        assert_eq!(all_ids, vec![5, 6, 7, 8, 9]);

        // 2. Read RG 0 with Column Projection (only "val")
        let rgs = vec![0];
        let projection_schema =
            Arc::new(Schema::new(vec![Field::new("val", DataType::Int32, true)]));
        let mut stream = reader
            .stream_row_groups(Some(&rgs), Some(projection_schema))
            .await?;
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

    #[test]
    fn test_resolve_partitioned_paths_respects_root_boundary() {
        let store = Arc::new(InMemory::new());
        // Relative `parquet_path` is preserved as store-relative even when `base_path`
        // shares a non-boundary prefix (`tbl_v2` vs `tbl`), and sidecar resolution
        // without `parquet_path` never slices `/tmp/tbl` off `/tmp/tbl_v2` into `_v2/...`.
        let config = SegmentConfig::new("file:///tmp/tbl_v2/region=us", "seg_1")
            .with_parquet_path("region=us/seg_1.parquet".to_string());
        let reader = HybridReader::new(config, store.clone(), "file:///tmp/tbl");
        assert_eq!(
            reader.resolve_object_path("parquet").to_string(),
            "region=us/seg_1.parquet"
        );
        assert_eq!(
            reader.resolve_object_path("del").to_string(),
            "tmp/tbl_v2/region=us/seg_1.del"
        );
        assert_eq!(
            reader.resolve_index_path("seg_1.category.bitmap"),
            "region=us/seg_1.category.bitmap"
        );
        assert_eq!(
            reader.resolve_index_path("file:///tmp/tbl/region=us/seg_1.puffin"),
            "region=us/seg_1.puffin"
        );

        let config_abs_outside = SegmentConfig::new("file:///tmp/tbl_v2/region=us", "seg_out")
            .with_parquet_path("file:///tmp/tbl_v2/region=us/seg_out.parquet".to_string());
        let reader_abs_outside =
            HybridReader::new(config_abs_outside, store.clone(), "file:///tmp/tbl");
        assert_eq!(
            reader_abs_outside
                .resolve_object_path("parquet")
                .to_string(),
            "tmp/tbl_v2/region=us/seg_out.parquet"
        );

        let config_exact = SegmentConfig::new("file:///tmp/tbl/region=us", "seg_2")
            .with_parquet_path("file:///tmp/tbl/region=us/seg_2.parquet".to_string());
        let reader_exact = HybridReader::new(config_exact, store, "file:///tmp/tbl");
        assert_eq!(
            reader_exact.resolve_object_path("parquet").to_string(),
            "region=us/seg_2.parquet"
        );
        assert_eq!(
            reader_exact.resolve_index_path("seg_2.puffin"),
            "region=us/seg_2.puffin"
        );
    }
}
