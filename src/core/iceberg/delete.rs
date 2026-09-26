// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Context;
use anyhow::Result;
use apache_avro::types::Record;
use arrow::array::Array;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Reader for position delete files (Avro and Parquet)
pub struct PositionDeleteReader {
    store: Arc<dyn object_store::ObjectStore>,
}

impl PositionDeleteReader {
    pub fn new(store: Arc<dyn object_store::ObjectStore>) -> Self {
        Self { store }
    }

    /// Fetch (and cache) the parsed content of a position-delete file without
    /// filtering it to a target data file. The returned map is keyed by the
    /// data-file path recorded in the delete file.
    ///
    /// This is the async half of [`Self::read_deletes`]; the CPU-bound filtering and
    /// bitmap construction is deliberately left to the caller so it can be
    /// parallelized across cores (see `load_merged_deletes_inner`).
    pub async fn fetch_deletes_map(
        &self,
        path: &str,
    ) -> Result<Arc<HashMap<String, HashSet<i64>>>> {
        let cache_key = path.to_string();

        // Probe the cache first so hit/miss is observable, then fall back to
        // `try_get_with` (which coalesces concurrent misses) on a miss.
        let full_map = match crate::core::cache::FULL_DELETE_FILE_CACHE.get(&cache_key).await {
            Some(v) => {
                crate::telemetry::metrics::DELETE_FILE_CACHE_TOTAL
                    .with_label_values(&["hit"])
                    .inc();
                v
            }
            None => {
                crate::telemetry::metrics::DELETE_FILE_CACHE_TOTAL
                    .with_label_values(&["miss"])
                    .inc();
                crate::core::cache::FULL_DELETE_FILE_CACHE
                    .try_get_with(cache_key.clone(), async {
                        let is_avro = path.ends_with(".avro");
                        let path_obj = object_store::path::Path::from(path);

                        // Add jitter/retry? No, moka will handle coalescing.
                        let res = self.store.get(&path_obj).await.map_err(|e| Arc::new(anyhow::anyhow!(e)))?;
                        let bytes = res.bytes().await.map_err(|e| Arc::new(anyhow::anyhow!(e)))?;

                        let result_map_res = tokio::task::spawn_blocking(move || {
                            if is_avro {
                                Self::read_deletes_avro_static(&bytes)
                            } else {
                                Self::read_deletes_parquet_static(&bytes)
                            }
                        }).await.map_err(|e| Arc::new(anyhow::anyhow!("JoinError: {}", e)))?;

                        // Parsed delete file successfully
                        let result_map = result_map_res.map_err(|e| Arc::new(anyhow::anyhow!(e)))?;
                        Ok::<_, Arc<anyhow::Error>>(Arc::new(result_map))
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to read delete file: {}", e))?
            }
        };

        Ok(full_map)
    }

    /// CPU-bound: filter a parsed delete-file map to `target_data_file_path`
    /// and build a `RoaringBitmap` of the deleted positions.
    ///
    /// Pure function (no I/O, no async) so it can be called from a rayon
    /// worker to parallelize the merge across cores.
    pub fn filter_map_to_bitmap(
        full_map: &HashMap<String, HashSet<i64>>,
        target_data_file_path: &str,
    ) -> roaring::RoaringBitmap {
        let mut bm = roaring::RoaringBitmap::new();
        let target_clean = target_data_file_path
            .strip_prefix("file://")
            .unwrap_or(target_data_file_path);

        for (fp, positions) in full_map.iter() {
            let fp_clean = fp.strip_prefix("file://").unwrap_or(fp);
            if fp_clean == target_clean
                || target_clean.ends_with(fp_clean)
                || fp_clean.ends_with(target_clean)
            {
                for &pos in positions {
                    if pos >= 0 && pos <= u32::MAX as i64 {
                        bm.insert(pos as u32);
                    }
                }
            }
        }
        bm
    }

    pub async fn read_deletes(
        &self,
        path: &str,
        target_data_file_path: &str,
    ) -> Result<HashSet<i64>> {
        let full_map = self.fetch_deletes_map(path).await?;

        let mut deleted_positions = HashSet::new();
        let target_clean = target_data_file_path.strip_prefix("file://").unwrap_or(target_data_file_path);

        for (fp, positions) in full_map.iter() {
            let fp_clean = fp.strip_prefix("file://").unwrap_or(fp);
            if fp_clean == target_clean
                || target_clean.ends_with(fp_clean)
                || fp_clean.ends_with(target_clean)
            {
                for pos in positions {
                    deleted_positions.insert(*pos);
                }
            }
        }

        // We intentionally do NOT populate the POSITION_DELETE_CACHE here.
        // Doing so would evict the merged deletes cache key since there are thousands of delete files.
        // The parsed file content is already cached in FULL_DELETE_FILE_CACHE.

        Ok(deleted_positions)
    }



    fn read_deletes_avro_static(
        bytes: &[u8],
    ) -> Result<HashMap<String, HashSet<i64>>> {
        let reader = apache_avro::Reader::new(bytes)?;
        let mut result_map: HashMap<String, HashSet<i64>> = HashMap::new();

        for record in reader {
            let value = record?;
            if let apache_avro::types::Value::Record(fields) = value {
                let mut file_path = None;
                let mut pos = None;

                for (name, val) in fields {
                    match name.as_str() {
                        "file_path" => {
                            if let apache_avro::types::Value::String(s) = val {
                                file_path = Some(s);
                            }
                        }
                        "pos" => {
                            if let apache_avro::types::Value::Long(p) = val {
                                pos = Some(p);
                            }
                        }
                        _ => {}
                    }
                }

                if let (Some(fp), Some(p)) = (file_path, pos) {
                    result_map.entry(fp).or_default().insert(p);
                }
            }
        }
        Ok(result_map)
    }



    fn read_deletes_parquet_static(
        bytes: &[u8],
    ) -> Result<HashMap<String, HashSet<i64>>> {
        use arrow::array::{Int64Array, StringArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let cursor = bytes::Bytes::copy_from_slice(bytes);
        let builder = ParquetRecordBatchReaderBuilder::try_new(cursor)?;
        let reader = builder.build()?;

        let mut result_map: HashMap<String, HashSet<i64>> = HashMap::new();

        for batch_res in reader {
            let batch = batch_res?;

            if let (Ok(file_path_col), Ok(pos_col)) = (
                batch.column_by_name("file_path").ok_or(()),
                batch.column_by_name("pos").ok_or(()),
            ) {
                let file_paths = file_path_col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow::anyhow!("file_path column is not string"))?;
                let positions = pos_col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow::anyhow!("pos column is not int64"))?;

                for i in 0..batch.num_rows() {
                    let fp = file_paths.value(i).to_string();
                    let pos = positions.value(i);
                    result_map.entry(fp).or_default().insert(pos);
                }
            }
        }
        Ok(result_map)
    }
}

/// Reader for equality delete files (Avro and Parquet)
pub struct EqualityDeleteReader {
    store: Arc<dyn object_store::ObjectStore>,
}

impl EqualityDeleteReader {
    pub fn new(store: Arc<dyn object_store::ObjectStore>) -> Self {
        Self { store }
    }

    /// Read equality delete records from a Parquet file.
    /// Returns batches containing ONLY the columns specified by `equality_ids`.
    pub async fn read_equality_deletes(
        &self,
        path: &str,
        equality_ids: &[i32],
        schema: &crate::core::manifest::Schema,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        let is_avro = path.ends_with(".avro");
        let path_obj = object_store::path::Path::from(path);
        let res = self.store.get(&path_obj).await?;
        let bytes = res.bytes().await?;

        if is_avro {
            self.read_equality_deletes_avro(bytes, equality_ids, schema)
        } else {
            self.read_equality_deletes_parquet(bytes, equality_ids, schema)
        }
    }

    fn read_equality_deletes_avro(
        &self,
        bytes: bytes::Bytes,
        equality_ids: &[i32],
        schema: &crate::core::manifest::Schema,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        use arrow::datatypes::{Field, Schema as ArrowSchema};
        use std::collections::HashMap;
        use std::sync::Arc;

        let reader = apache_avro::Reader::new(&bytes[..])?;

        let mut column_names = Vec::new();
        let field_map: HashMap<i32, &crate::core::manifest::SchemaField> =
            schema.fields.iter().map(|f| (f.id, f)).collect();

        for &id in equality_ids {
            if let Some(field) = field_map.get(&id) {
                column_names.push(field.name.clone());
            } else {
                return Err(anyhow::anyhow!(
                    "Equality Delete ID {} not found in schema",
                    id
                ));
            }
        }

        // Build columns
        let mut columns_data: Vec<Vec<apache_avro::types::Value>> =
            vec![Vec::new(); equality_ids.len()];
        let mut row_count = 0;

        for record in reader {
            let value = record?;
            if let apache_avro::types::Value::Record(fields) = value {
                let field_map_record: HashMap<String, apache_avro::types::Value> =
                    fields.into_iter().collect();

                for (idx, name) in column_names.iter().enumerate() {
                    if let Some(val) = field_map_record.get(name) {
                        columns_data[idx].push(val.clone());
                    } else {
                        columns_data[idx].push(apache_avro::types::Value::Null);
                    }
                }
                row_count += 1;
            }
        }

        if row_count == 0 {
            return Ok(vec![]);
        }

        // Convert to Arrow RecordBatch
        let mut arrow_columns = Vec::new();
        let mut arrow_fields = Vec::new();

        for (idx, name) in column_names.iter().enumerate() {
            let field = field_map
                .get(&equality_ids[idx])
                .context("Missing equality id")?;
            let arrow_field = Field::new(name, map_type_to_arrow(&field.type_str), true);
            arrow_fields.push(arrow_field);

            let array = avro_to_arrow_array(&columns_data[idx], &field.type_str)?;
            arrow_columns.push(array);
        }

        let arrow_schema = Arc::new(ArrowSchema::new(arrow_fields));
        let batch = arrow::record_batch::RecordBatch::try_new(arrow_schema, arrow_columns)?;
        Ok(vec![batch])
    }

    fn read_equality_deletes_parquet(
        &self,
        bytes: bytes::Bytes,
        equality_ids: &[i32],
        schema: &crate::core::manifest::Schema,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::collections::HashMap;

        let cursor = bytes;

        // 1. Resolve column names from IDs
        let mut column_names = Vec::new();
        let field_map: HashMap<i32, &crate::core::manifest::SchemaField> =
            schema.fields.iter().map(|f| (f.id, f)).collect();

        for &id in equality_ids {
            if let Some(field) = field_map.get(&id) {
                column_names.push(field.name.clone());
            } else {
                // If ID not found in schema, we can't read it. Ideally shouldn't happen for valid manifests.
                return Err(anyhow::anyhow!(
                    "Equality Delete ID {} not found in schema",
                    id
                ));
            }
        }

        // 2. Open Parquet Reader
        let builder = ParquetRecordBatchReaderBuilder::try_new(cursor)?;
        let arrow_schema = builder.schema();

        // 3. Find projection mask indices
        let mut indices = Vec::new();
        for col_name in &column_names {
            if let Ok(idx) = arrow_schema.index_of(col_name) {
                indices.push(idx);
            } else {
                // If column missing in delete file, it implies no deletes for that ID?
                // Or maybe partial schema match. For now, strict requirement.
                return Err(anyhow::anyhow!(
                    "Column {} not found in equality delete file",
                    col_name
                ));
            }
        }

        // 4. Project and Read
        let projection = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), indices);
        let reader = builder.with_projection(projection).build()?;

        let mut batches = Vec::new();
        for batch_res in reader {
            batches.push(batch_res?);
        }

        Ok(batches)
    }
}

fn map_type_to_arrow(type_str: &str) -> arrow::datatypes::DataType {
    match type_str {
        "Int32" | "int" => arrow::datatypes::DataType::Int32,
        "Int64" | "long" => arrow::datatypes::DataType::Int64,
        "Float32" | "float" => arrow::datatypes::DataType::Float32,
        "Float64" | "double" => arrow::datatypes::DataType::Float64,
        "Utf8" | "string" => arrow::datatypes::DataType::Utf8,
        "Boolean" | "bool" => arrow::datatypes::DataType::Boolean,
        _ => arrow::datatypes::DataType::Utf8,
    }
}

fn avro_to_arrow_array(
    values: &[apache_avro::types::Value],
    type_str: &str,
) -> Result<arrow::array::ArrayRef> {
    use apache_avro::types::Value;
    use arrow::array::*;

    match type_str {
        "Int32" | "int" => {
            let mut builder = Int32Builder::new();
            for v in values {
                match v {
                    Value::Int(i) => builder.append_value(*i),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        "Int64" | "long" => {
            let mut builder = Int64Builder::new();
            for v in values {
                match v {
                    Value::Long(i) => builder.append_value(*i),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        "Float32" | "float" => {
            let mut builder = Float32Builder::new();
            for v in values {
                match v {
                    Value::Float(i) => builder.append_value(*i),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        "Float64" | "double" => {
            let mut builder = Float64Builder::new();
            for v in values {
                match v {
                    Value::Double(i) => builder.append_value(*i),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        "Utf8" | "string" => {
            let mut builder = StringBuilder::new();
            for v in values {
                match v {
                    Value::String(s) => builder.append_value(s),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        "Boolean" | "bool" => {
            let mut builder = BooleanBuilder::new();
            for v in values {
                match v {
                    Value::Boolean(b) => builder.append_value(*b),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            let mut builder = StringBuilder::new();
            for v in values {
                builder.append_value(format!("{:?}", v));
            }
            Ok(Arc::new(builder.finish()))
        }
    }
}

/// Writer for Iceberg Delete Files (Position and Equality Deletes).
///
/// Delete files are written through the table's `ObjectStore`, so they land in
/// the same store as the data (local FS, S3/MinIO, …). The previous
/// implementation used `std::fs::File::create` on a path derived from the table
/// URI, which silently wrote to a bogus local path for any non-`file://` store
/// (e.g. `./s3:/bucket/…`) and made deletes invisible to readers.
pub struct IcebergDeleteWriter {
    base_path: String,
    format_version: i32,
    store: Arc<dyn object_store::ObjectStore>,
}

impl IcebergDeleteWriter {
    pub fn new(
        base_path: String,
        format_version: i32,
        store: Arc<dyn object_store::ObjectStore>,
    ) -> Self {
        Self {
            base_path: base_path.replace("file://", ""),
            format_version,
            store,
        }
    }

    /// Build the full URI recorded in the manifest for a delete file written at
    /// `object_path` (relative to the table root / store prefix).
    fn metadata_uri(&self, object_path: &str) -> String {
        let full = format!("{}/{}", self.base_path, object_path);
        if full.starts_with("file://") || full.starts_with("s3://") {
            full
        } else {
            format!("file://{}", full)
        }
    }

    /// Write a Position Delete File
    ///
    /// Schema:
    /// - file_path: string (path of the data file where row is deleted)
    /// - pos: long (ordinal position of the deleted row)
    /// - row: optional (the deleted row itself, strictly optional)
    pub async fn write_position_delete(
        &self,
        partition_data: Option<(String, std::collections::HashMap<String, serde_json::Value>)>,
        file_path_column: &arrow::array::StringArray,
        pos_column: &arrow::array::Int64Array,
    ) -> Result<crate::core::manifest::DeleteFile> {
        if file_path_column.len() != pos_column.len() {
            return Err(anyhow::anyhow!(
                "Mismatch in file_path and pos column lengths"
            ));
        }

        let file_name = format!(
            "del-pos-{}-{}.avro",
            uuid::Uuid::new_v4(),
            self.format_version
        );
        // Path relative to the table root / store prefix.
        let object_path = if let Some((ref part_path, _)) = partition_data {
            format!("{}/{}", part_path, file_name)
        } else {
            file_name.clone()
        };

        // Construct Avro Schema for Position Deletes
        let schema_json = r#"
        {
            "type": "record",
            "name": "position_delete",
            "fields": [
                {"name": "file_path", "type": "string"},
                {"name": "pos", "type": "long"}
            ]
        }
        "#;
        let schema = apache_avro::Schema::parse_str(schema_json)?;

        // Build the Avro bytes in memory, then write to the object store.
        let mut writer = apache_avro::Writer::new(&schema, Vec::new());

        for i in 0..file_path_column.len() {
            let mut record =
                Record::new(&schema).ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;
            record.put(
                "file_path",
                apache_avro::types::Value::String(file_path_column.value(i).to_string()),
            );
            record.put("pos", apache_avro::types::Value::Long(pos_column.value(i)));
            writer.append(record)?;
        }

        let len = writer.flush()?;
        let bytes = writer.into_inner()?;
        self.store
            .put(
                &object_store::path::Path::from(object_path.as_str()),
                bytes.into(),
            )
            .await?;

        // The manifest records the full URI; the reader relativizes it.
        let metadata_path = self.metadata_uri(&object_path);

        let partition_values = partition_data.map(|(_, map)| map).unwrap_or_default();

        Ok(crate::core::manifest::DeleteFile {
            file_path: metadata_path,
            content: crate::core::manifest::DeleteContent::Position,
            file_size_bytes: len as i64,
            record_count: file_path_column.len() as i64,
            partition_values,
        })
    }

    /// Write an Equality Delete File
    pub async fn write_equality_delete(
        &self,
        partition_value: Option<&str>,
        batch: &arrow::record_batch::RecordBatch,
        equality_ids: &[i32],
        table_schema: &crate::core::manifest::Schema,
    ) -> Result<crate::core::manifest::DeleteFile> {
        let file_name = format!(
            "del-eq-{}-{}.avro",
            uuid::Uuid::new_v4(),
            self.format_version
        );
        // Path relative to the table root / store prefix.
        let object_path = if let Some(part) = partition_value {
            format!("{}/{}", part, file_name)
        } else {
            file_name.clone()
        };

        // 1. Construct Avro Schema based on equality IDs
        let mut fields_json = Vec::new();
        let mut column_names = Vec::new();

        for &id in equality_ids {
            let field = table_schema
                .fields
                .iter()
                .find(|f| f.id == id)
                .ok_or_else(|| anyhow::anyhow!("Field ID {} not found in schema", id))?;

            column_names.push(field.name.clone());

            let avro_type = match field.type_str.as_str() {
                "Int32" | "int" => "int",
                "Int64" | "long" => "long",
                "Float32" | "float" => "float",
                "Float64" | "double" => "double",
                "Utf8" | "string" => "string",
                "Boolean" | "bool" => "boolean",
                _ => "string", // Fallback
            };

            fields_json.push(format!(
                r#"{{"name": "{}", "type": "{}"}}"#,
                field.name, avro_type
            ));
        }

        let schema_json = format!(
            r#"
        {{
            "type": "record",
            "name": "equality_delete",
            "fields": [{}]
        }}
        "#,
            fields_json.join(",")
        );

        let schema = apache_avro::Schema::parse_str(&schema_json)?;

        // 2. Build the Avro bytes in memory, then write to the object store.
        let mut writer = apache_avro::Writer::new(&schema, Vec::new());

        for i in 0..batch.num_rows() {
            let mut record =
                Record::new(&schema).ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;
            for &id in equality_ids.iter() {
                let field = table_schema
                    .fields
                    .iter()
                    .find(|f| f.id == id)
                    .context("Missing field")?;
                let col = batch
                    .column_by_name(&field.name)
                    .ok_or_else(|| anyhow::anyhow!("Column {} not found in batch", field.name))?;

                let value = self.arrow_to_avro_value(col, i)?;
                record.put(&field.name, value);
            }
            writer.append(record)?;
        }

        let len = writer.flush()?;
        let bytes = writer.into_inner()?;
        self.store
            .put(
                &object_store::path::Path::from(object_path.as_str()),
                bytes.into(),
            )
            .await?;

        // The manifest records the full URI; the reader relativizes it.
        let metadata_path = self.metadata_uri(&object_path);

        Ok(crate::core::manifest::DeleteFile {
            file_path: metadata_path,
            content: crate::core::manifest::DeleteContent::Equality {
                equality_ids: equality_ids.to_vec(),
            },
            file_size_bytes: len as i64,
            record_count: batch.num_rows() as i64,
            partition_values: std::collections::HashMap::new(),
        })
    }

    fn arrow_to_avro_value(
        &self,
        array: &arrow::array::ArrayRef,
        i: usize,
    ) -> Result<apache_avro::types::Value> {
        use apache_avro::types::Value;
        use arrow::array::*;

        if array.is_null(i) {
            return Ok(Value::Null);
        }

        let val = match array.data_type() {
            arrow::datatypes::DataType::Int32 => {
                let a = array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .context("Invalid cast")?;
                Value::Int(a.value(i))
            }
            arrow::datatypes::DataType::Int64 => {
                let a = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .context("Invalid cast")?;
                Value::Long(a.value(i))
            }
            arrow::datatypes::DataType::Float32 => {
                let a = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid cast")?;
                Value::Float(a.value(i))
            }
            arrow::datatypes::DataType::Float64 => {
                let a = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .context("Invalid cast")?;
                Value::Double(a.value(i))
            }
            arrow::datatypes::DataType::Utf8 => {
                let a = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid cast")?;
                Value::String(a.value(i).to_string())
            }
            arrow::datatypes::DataType::Boolean => {
                let a = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .context("Invalid cast")?;
                Value::Boolean(a.value(i))
            }
            _ => Value::String(format!("{:?}", array)), // Fallback
        };
        Ok(val)
    }
}
