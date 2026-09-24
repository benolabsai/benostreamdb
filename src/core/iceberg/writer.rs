// Copyright (c) 2026 Richard Albright. All rights reserved.

use super::value::json_to_avro_value;
use anyhow::{Context, Result};

pub const MANIFEST_LIST_SCHEMA_V2: &str = r#"
{
    "type": "record",
    "name": "manifest_list",
    "fields": [
        {"name": "manifest_path", "type": "string"},
        {"name": "manifest_length", "type": "long"},
        {"name": "partition_spec_id", "type": "int"},
        {"name": "content", "type": "int", "doc": "0=data, 1=deletes"},
        {"name": "sequence_number", "type": "long", "default": 0},
        {"name": "min_sequence_number", "type": "long", "default": 0},
        {"name": "added_snapshot_id", "type": "long"},
        {"name": "added_data_files_count", "type": "int"},
        {"name": "existing_data_files_count", "type": "int"},
        {"name": "deleted_data_files_count", "type": "int"},
        {"name": "added_rows_count", "type": "long"},
        {"name": "existing_rows_count", "type": "long"},
        {"name": "deleted_rows_count", "type": "long"},
        {"name": "partitions", "type": ["null", {
            "type": "array",
            "items": {
                "type": "record",
                "name": "field_summary",
                "fields": [
                    {"name": "contains_null", "type": "boolean"},
                    {"name": "contains_nan", "type": ["null", "boolean"]},
                    {"name": "lower_bound", "type": ["null", "bytes"]},
                    {"name": "upper_bound", "type": ["null", "bytes"]}
                ]
            }
        }]}
    ]
}
"#;

pub struct IcebergWriter;

impl Default for IcebergWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl IcebergWriter {
    pub fn new() -> Self {
        Self {}
    }

    /// Write a Manifest List (snap-*.avro)
    pub fn write_manifest_list(
        &self,
        entries: &[crate::core::manifest::ManifestListEntry],
    ) -> Result<Vec<u8>> {
        let schema = apache_avro::Schema::parse_str(MANIFEST_LIST_SCHEMA_V2)?;
        let mut writer = apache_avro::Writer::new(&schema, Vec::new());

        for entry in entries {
            let mut record = apache_avro::types::Record::new(&schema)
                .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;
            record.put("manifest_path", entry.manifest_path.clone());
            record.put("manifest_length", entry.manifest_length);
            record.put("partition_spec_id", entry.partition_spec_id);
            record.put("content", entry.content);
            record.put("sequence_number", entry.sequence_number);
            record.put("min_sequence_number", entry.min_sequence_number);
            record.put("added_snapshot_id", entry.added_snapshot_id);
            record.put("added_data_files_count", entry.added_files_count);
            record.put("existing_data_files_count", entry.existing_files_count);
            record.put("deleted_data_files_count", entry.deleted_files_count);
            record.put("added_rows_count", entry.added_rows_count);
            record.put("existing_rows_count", entry.existing_rows_count);
            record.put("deleted_rows_count", entry.deleted_rows_count);

            // Build partition field summaries from partition_stats
            let partition_summaries: Vec<apache_avro::types::Value> = entry
                .partition_stats
                .values()
                .map(|stats| {
                    use crate::core::manifest::ManifestValue;
                    let lower_bound = match &stats.min {
                        None | Some(ManifestValue::Null) => apache_avro::types::Value::Null,
                        Some(val) => {
                            apache_avro::types::Value::Bytes(format!("{}", val).into_bytes())
                        }
                    };
                    let upper_bound = match &stats.max {
                        None | Some(ManifestValue::Null) => apache_avro::types::Value::Null,
                        Some(val) => {
                            apache_avro::types::Value::Bytes(format!("{}", val).into_bytes())
                        }
                    };
                    apache_avro::types::Value::Record(vec![
                        (
                            "contains_null".to_string(),
                            apache_avro::types::Value::Boolean(stats.null_count > 0),
                        ),
                        (
                            "contains_nan".to_string(),
                            apache_avro::types::Value::Union(
                                0,
                                Box::new(apache_avro::types::Value::Null),
                            ),
                        ),
                        (
                            "lower_bound".to_string(),
                            apache_avro::types::Value::Union(1, Box::new(lower_bound)),
                        ),
                        (
                            "upper_bound".to_string(),
                            apache_avro::types::Value::Union(1, Box::new(upper_bound)),
                        ),
                    ])
                })
                .collect();

            if partition_summaries.is_empty() {
                record.put("partitions", apache_avro::types::Value::Null);
            } else {
                record.put(
                    "partitions",
                    apache_avro::types::Value::Array(partition_summaries),
                );
            }

            writer.append(record)?;
        }

        Ok(writer.into_inner()?)
    }

    /// Write a Manifest File (*.avro)
    pub fn write_manifest_file(
        &self,
        entries: &[crate::core::manifest::ManifestEntry],
        partition_spec: &crate::core::manifest::PartitionSpec,
        schema: &crate::core::manifest::Schema,
        snapshot_id: i64,
        seq_num: i64,
    ) -> Result<Vec<u8>> {
        // 1. Generate Schema based on Partition Spec
        let schema_json = self.generate_manifest_schema(partition_spec, schema);
        let schema = apache_avro::Schema::parse_str(&schema_json)?;
        let mut writer = apache_avro::Writer::new(&schema, Vec::new());

        for entry in entries {
            let mut record = apache_avro::types::Record::new(&schema)
                .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;
            record.put("status", apache_avro::types::Value::Int(1)); // 1=ADDED
            record.put(
                "snapshot_id",
                apache_avro::types::Value::Union(
                    1,
                    Box::new(apache_avro::types::Value::Long(snapshot_id)),
                ),
            );
            record.put(
                "sequence_number",
                apache_avro::types::Value::Union(
                    1,
                    Box::new(apache_avro::types::Value::Long(seq_num)),
                ),
            );
            record.put(
                "file_sequence_number",
                apache_avro::types::Value::Union(
                    1,
                    Box::new(apache_avro::types::Value::Long(seq_num)),
                ),
            );

            let data_file_schema = match &schema {
                apache_avro::Schema::Record(r) => {
                    &r.fields
                        .iter()
                        .find(|f| f.name == "data_file")
                        .context("Missing data_file")?
                        .schema
                }
                _ => unreachable!(),
            };
            let mut data_file = apache_avro::types::Record::new(data_file_schema)
                .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;

            data_file.put("content", apache_avro::types::Value::Int(0)); // 0=Data
            data_file.put(
                "file_path",
                apache_avro::types::Value::String(entry.file_path.clone()),
            );
            data_file.put(
                "file_format",
                apache_avro::types::Value::String("PARQUET".to_string()),
            );
            data_file.put(
                "record_count",
                apache_avro::types::Value::Long(entry.record_count),
            );
            data_file.put(
                "file_size_in_bytes",
                apache_avro::types::Value::Long(entry.file_size_bytes),
            );

            // Statistics (Unions)
            data_file.put(
                "column_sizes",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            data_file.put(
                "value_counts",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            data_file.put(
                "null_value_counts",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            data_file.put(
                "nan_value_counts",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            data_file.put(
                "lower_bounds",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            data_file.put(
                "upper_bounds",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );

            // Partition Data
            let mut partition_record_values = Vec::new();
            for field in &partition_spec.fields {
                let val = entry
                    .partition_values
                    .get(&field.name)
                    .unwrap_or(&serde_json::Value::Null);
                let avro_val = json_to_avro_value(val);
                let union_val = match avro_val {
                    apache_avro::types::Value::Null => apache_avro::types::Value::Union(
                        0,
                        Box::new(apache_avro::types::Value::Null),
                    ),
                    _ => apache_avro::types::Value::Union(1, Box::new(avro_val)),
                };
                partition_record_values.push((field.name.clone(), union_val));
            }
            data_file.put(
                "partition",
                apache_avro::types::Value::Record(partition_record_values),
            );
            data_file.put(
                "equality_ids",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );

            // Populate index_files if present
            if !entry.index_files.is_empty() {
                let index_json =
                    serde_json::to_string(&entry.index_files).unwrap_or_else(|_| "[]".to_string());
                data_file.put(
                    "index_files",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::String(index_json)),
                    ),
                );
            } else {
                data_file.put(
                    "index_files",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
            }

            if let Some(chk) = &entry.file_checksum {
                data_file.put(
                    "file_checksum",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::String(chk.clone())),
                    ),
                );
            } else {
                data_file.put(
                    "file_checksum",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
            }

            record.put("data_file", data_file);
            writer.append(record)?;

            // Write associated Delete Files
            for del_file in &entry.delete_files {
                let mut record = apache_avro::types::Record::new(&schema)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;
                record.put("status", apache_avro::types::Value::Int(1)); // 1=ADDED
                record.put(
                    "snapshot_id",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::Long(snapshot_id)),
                    ),
                );
                record.put(
                    "sequence_number",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::Long(seq_num)),
                    ),
                );
                record.put(
                    "file_sequence_number",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::Long(seq_num)),
                    ),
                );

                let data_file_schema = match &schema {
                    apache_avro::Schema::Record(r) => {
                        &r.fields
                            .iter()
                            .find(|f| f.name == "data_file")
                            .context("Missing data_file")?
                            .schema
                    }
                    _ => unreachable!(),
                };
                let mut data_file = apache_avro::types::Record::new(data_file_schema)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;

                let content_id = match del_file.content {
                    crate::core::manifest::DeleteContent::Position => 1,
                    crate::core::manifest::DeleteContent::Equality { .. } => 2,
                    crate::core::manifest::DeleteContent::DeletionVector { .. } => 3,
                };

                data_file.put("content", apache_avro::types::Value::Int(content_id));
                data_file.put(
                    "file_path",
                    apache_avro::types::Value::String(del_file.file_path.clone()),
                );
                data_file.put(
                    "file_format",
                    apache_avro::types::Value::String("AVRO".to_string()),
                );
                data_file.put(
                    "record_count",
                    apache_avro::types::Value::Long(del_file.record_count),
                );
                data_file.put(
                    "file_size_in_bytes",
                    apache_avro::types::Value::Long(del_file.file_size_bytes),
                );

                // Statistics (Unions)
                data_file.put(
                    "column_sizes",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "value_counts",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "null_value_counts",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "nan_value_counts",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "lower_bounds",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "upper_bounds",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );

                // Use Delete File's partition values (or inherit from parent if empty?)
                // DeleteFile has partition_values.
                let mut partition_record_values = Vec::new();
                for field in &partition_spec.fields {
                    let val = del_file
                        .partition_values
                        .get(&field.name)
                        .unwrap_or(&serde_json::Value::Null);
                    let avro_val = json_to_avro_value(val);
                    let union_val = match avro_val {
                        apache_avro::types::Value::Null => apache_avro::types::Value::Union(
                            0,
                            Box::new(apache_avro::types::Value::Null),
                        ),
                        _ => apache_avro::types::Value::Union(1, Box::new(avro_val)),
                    };
                    partition_record_values.push((field.name.clone(), union_val));
                }
                data_file.put(
                    "partition",
                    apache_avro::types::Value::Record(partition_record_values),
                );

                // Equality IDs
                if let crate::core::manifest::DeleteContent::Equality { equality_ids } =
                    &del_file.content
                {
                    let avro_ids: Vec<apache_avro::types::Value> = equality_ids
                        .iter()
                        .map(|&i| apache_avro::types::Value::Int(i))
                        .collect();
                    data_file.put(
                        "equality_ids",
                        apache_avro::types::Value::Union(
                            1,
                            Box::new(apache_avro::types::Value::Array(avro_ids)),
                        ),
                    );
                } else {
                    data_file.put(
                        "equality_ids",
                        apache_avro::types::Value::Union(
                            0,
                            Box::new(apache_avro::types::Value::Null),
                        ),
                    );
                }

                record.put("data_file", data_file);
                writer.append(record)?;
            }
        }

        Ok(writer.into_inner()?)
    }

    /// Write Manifest Files dynamically chunked by byte size (e.g. 8MB)
    pub fn write_manifest_chunks(
        &self,
        entries: &[crate::core::manifest::ManifestEntry],
        partition_spec: &crate::core::manifest::PartitionSpec,
        schema: &crate::core::manifest::Schema,
        snapshot_id: i64,
        seq_num: i64,
        target_size_bytes: usize,
    ) -> Result<Vec<(Vec<u8>, usize, i64)>> {
        let schema_json = self.generate_manifest_schema(partition_spec, schema);
        let avro_schema = apache_avro::Schema::parse_str(&schema_json)?;

        let mut chunks = Vec::new();
        let mut current_file_count = 0;
        let mut current_row_count = 0;

        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        #[derive(Clone)]
        struct ByteTracker<W: std::io::Write> {
            inner: W,
            written: Arc<AtomicUsize>,
        }
        impl<W: std::io::Write> std::io::Write for ByteTracker<W> {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let n = self.inner.write(buf)?;
                self.written.fetch_add(n, Ordering::Relaxed);
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.inner.flush()
            }
        }

        let written = Arc::new(AtomicUsize::new(0));
        let mut writer = apache_avro::Writer::new(
            &avro_schema,
            ByteTracker {
                inner: Vec::new(),
                written: written.clone(),
            },
        );

        for entry in entries {
            let mut record = apache_avro::types::Record::new(&avro_schema)
                .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;
            record.put("status", apache_avro::types::Value::Int(1));
            record.put(
                "snapshot_id",
                apache_avro::types::Value::Union(
                    1,
                    Box::new(apache_avro::types::Value::Long(snapshot_id)),
                ),
            );
            record.put(
                "sequence_number",
                apache_avro::types::Value::Union(
                    1,
                    Box::new(apache_avro::types::Value::Long(seq_num)),
                ),
            );
            record.put(
                "file_sequence_number",
                apache_avro::types::Value::Union(
                    1,
                    Box::new(apache_avro::types::Value::Long(seq_num)),
                ),
            );

            let data_file_schema = match &avro_schema {
                apache_avro::Schema::Record(r) => {
                    &r.fields
                        .iter()
                        .find(|f| f.name == "data_file")
                        .context("Missing data_file")?
                        .schema
                }
                _ => unreachable!(),
            };
            let mut data_file = apache_avro::types::Record::new(data_file_schema)
                .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;

            data_file.put("content", apache_avro::types::Value::Int(0));
            data_file.put(
                "file_path",
                apache_avro::types::Value::String(entry.file_path.clone()),
            );
            data_file.put(
                "file_format",
                apache_avro::types::Value::String("PARQUET".to_string()),
            );
            data_file.put(
                "record_count",
                apache_avro::types::Value::Long(entry.record_count),
            );
            data_file.put(
                "file_size_in_bytes",
                apache_avro::types::Value::Long(entry.file_size_bytes),
            );

            data_file.put(
                "column_sizes",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            data_file.put(
                "value_counts",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            // Per-column bounds and null counts. The reader reconstructs
            // `column_stats` from these three, so writing Null here (as this
            // code used to) silently discarded every min/max and left the
            // planner's stats-pruning branch with nothing to work on.
            let (null_counts, lower_bounds, upper_bounds) =
                bounds_avro_values(&entry.column_stats, schema);
            data_file.put("null_value_counts", null_counts);
            data_file.put(
                "nan_value_counts",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );
            data_file.put("lower_bounds", lower_bounds);
            data_file.put("upper_bounds", upper_bounds);

            let mut partition_record_values = Vec::new();
            for field in &partition_spec.fields {
                let val = entry
                    .partition_values
                    .get(&field.name)
                    .unwrap_or(&serde_json::Value::Null);
                let avro_val = json_to_avro_value(val);
                let union_val = match avro_val {
                    apache_avro::types::Value::Null => apache_avro::types::Value::Union(
                        0,
                        Box::new(apache_avro::types::Value::Null),
                    ),
                    _ => apache_avro::types::Value::Union(1, Box::new(avro_val)),
                };
                partition_record_values.push((field.name.clone(), union_val));
            }
            data_file.put(
                "partition",
                apache_avro::types::Value::Record(partition_record_values),
            );
            data_file.put(
                "equality_ids",
                apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
            );

            if !entry.index_files.is_empty() {
                let index_json =
                    serde_json::to_string(&entry.index_files).unwrap_or_else(|_| "[]".to_string());
                data_file.put(
                    "index_files",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::String(index_json)),
                    ),
                );
            } else {
                data_file.put(
                    "index_files",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
            }

            if let Some(chk) = &entry.file_checksum {
                data_file.put(
                    "file_checksum",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::String(chk.clone())),
                    ),
                );
            } else {
                data_file.put(
                    "file_checksum",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
            }

            record.put("data_file", data_file);
            writer.append(record)?;
            current_file_count += 1;
            current_row_count += entry.record_count;

            for del_file in &entry.delete_files {
                let mut record = apache_avro::types::Record::new(&avro_schema)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;
                record.put("status", apache_avro::types::Value::Int(1)); // 1=ADDED
                record.put(
                    "snapshot_id",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::Long(snapshot_id)),
                    ),
                );
                record.put(
                    "sequence_number",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::Long(seq_num)),
                    ),
                );
                record.put(
                    "file_sequence_number",
                    apache_avro::types::Value::Union(
                        1,
                        Box::new(apache_avro::types::Value::Long(seq_num)),
                    ),
                );

                let data_file_schema = match &avro_schema {
                    apache_avro::Schema::Record(r) => {
                        &r.fields
                            .iter()
                            .find(|f| f.name == "data_file")
                            .context("Missing data_file")?
                            .schema
                    }
                    _ => unreachable!(),
                };
                let mut data_file = apache_avro::types::Record::new(data_file_schema)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create Record"))?;

                let content_id = match del_file.content {
                    crate::core::manifest::DeleteContent::Position => 1,
                    crate::core::manifest::DeleteContent::Equality { .. } => 2,
                    crate::core::manifest::DeleteContent::DeletionVector { .. } => 3,
                };

                data_file.put("content", apache_avro::types::Value::Int(content_id));
                data_file.put(
                    "file_path",
                    apache_avro::types::Value::String(del_file.file_path.clone()),
                );
                data_file.put(
                    "file_format",
                    apache_avro::types::Value::String("AVRO".to_string()),
                );
                data_file.put(
                    "record_count",
                    apache_avro::types::Value::Long(del_file.record_count),
                );
                data_file.put(
                    "file_size_in_bytes",
                    apache_avro::types::Value::Long(del_file.file_size_bytes),
                );

                data_file.put(
                    "column_sizes",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "value_counts",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "null_value_counts",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "nan_value_counts",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "lower_bounds",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );
                data_file.put(
                    "upper_bounds",
                    apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                );

                let mut partition_record_values = Vec::new();
                for field in &partition_spec.fields {
                    let val = del_file
                        .partition_values
                        .get(&field.name)
                        .unwrap_or(&serde_json::Value::Null);
                    let avro_val = json_to_avro_value(val);
                    let union_val = match avro_val {
                        apache_avro::types::Value::Null => apache_avro::types::Value::Union(
                            0,
                            Box::new(apache_avro::types::Value::Null),
                        ),
                        _ => apache_avro::types::Value::Union(1, Box::new(avro_val)),
                    };
                    partition_record_values.push((field.name.clone(), union_val));
                }
                data_file.put(
                    "partition",
                    apache_avro::types::Value::Record(partition_record_values),
                );

                if let crate::core::manifest::DeleteContent::Equality { equality_ids } =
                    &del_file.content
                {
                    let avro_ids: Vec<apache_avro::types::Value> = equality_ids
                        .iter()
                        .map(|&i| apache_avro::types::Value::Int(i))
                        .collect();
                    data_file.put(
                        "equality_ids",
                        apache_avro::types::Value::Union(
                            1,
                            Box::new(apache_avro::types::Value::Array(avro_ids)),
                        ),
                    );
                } else {
                    data_file.put(
                        "equality_ids",
                        apache_avro::types::Value::Union(
                            0,
                            Box::new(apache_avro::types::Value::Null),
                        ),
                    );
                }

                record.put("data_file", data_file);
                writer.append(record)?;
            }

            writer.flush()?;
            if written.load(Ordering::Relaxed) >= target_size_bytes {
                let tracker = writer.into_inner()?;
                chunks.push((tracker.inner, current_file_count, current_row_count));

                written.store(0, Ordering::Relaxed);
                writer = apache_avro::Writer::new(
                    &avro_schema,
                    ByteTracker {
                        inner: Vec::new(),
                        written: written.clone(),
                    },
                );
                current_file_count = 0;
                current_row_count = 0;
            }
        }

        if current_file_count > 0 {
            writer.flush()?;
            let tracker = writer.into_inner()?;
            chunks.push((tracker.inner, current_file_count, current_row_count));
        }

        Ok(chunks)
    }

    fn generate_manifest_schema(
        &self,
        spec: &crate::core::manifest::PartitionSpec,
        schema: &crate::core::manifest::Schema,
    ) -> String {
        let mut partition_fields = Vec::new();
        for field in &spec.fields {
            let type_str = partition_field_avro_type(field, schema);
            partition_fields.push(format!(
                r#"{{"name": "{}", "type": {}, "default": null}}"#,
                field.name, type_str
            ));
        }
        let partition_fields_json = partition_fields.join(",");

        format!(
            r#"
{{
    "type": "record",
    "name": "manifest",
    "fields": [
        {{"name": "status", "type": "int", "doc": "0=EXISTING, 1=ADDED, 2=DELETED"}},
        {{"name": "snapshot_id", "type": ["null", "long"]}},
        {{"name": "sequence_number", "type": ["null", "long"]}},
        {{"name": "file_sequence_number", "type": ["null", "long"]}},
        {{"name": "data_file", "type": {{
            "type": "record",
            "name": "r2",
            "fields": [
                {{"name": "content", "type": "int", "doc": "0=DATA, 1=POSITION DELETES, 2=EQUALITY DELETES"}},
                {{"name": "file_path", "type": "string"}},
                {{"name": "file_format", "type": "string"}},
                {{"name": "partition", "type": {{
                    "type": "record",
                    "name": "r102",
                    "fields": [{}]
                }}}},

                {{"name": "record_count", "type": "long"}},
                {{"name": "file_size_in_bytes", "type": "long"}},
                {{"name": "column_sizes", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k1", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "value_counts", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k2", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "null_value_counts", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k3", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "nan_value_counts", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k4", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "lower_bounds", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k5", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"bytes"}}]}}}}], "default": null}},
                {{"name": "upper_bounds", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k6", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"bytes"}}]}}}}], "default": null}},
                {{"name": "equality_ids", "type": ["null", {{"type": "array", "items": "int"}}], "default": null}},
                {{"name": "index_files", "type": ["null", "string"], "default": null}},
                {{"name": "file_checksum", "type": ["null", "string"], "default": null}}
            ]
        }}
    }}]
}}
"#,
            partition_fields_json
        )
    }
}

/// Build the manifest's `null_value_counts`, `lower_bounds` and `upper_bounds`
/// Avro values from a segment's `column_stats`, keyed by Iceberg field id.
///
/// This is the write side of the pair that
/// [`crate::core::iceberg::manifest`] reads: it reconstructs `column_stats`
/// from exactly these three maps, and only when all three are present.
fn bounds_avro_values(
    stats: &std::collections::HashMap<String, crate::core::manifest::ColumnStats>,
    schema: &crate::core::manifest::Schema,
) -> (
    apache_avro::types::Value,
    apache_avro::types::Value,
    apache_avro::types::Value,
) {
    use crate::core::iceberg::value::encode_iceberg_value;
    use apache_avro::types::Value as AvroValue;

    let mut nulls: Vec<AvroValue> = Vec::new();
    let mut lowers: Vec<AvroValue> = Vec::new();
    let mut uppers: Vec<AvroValue> = Vec::new();

    for (col_name, col_stats) in stats {
        let Some(field) = schema.fields.iter().find(|f| &f.name == col_name) else {
            continue; // not addressable by field id; skip rather than guess
        };

        nulls.push(AvroValue::Record(vec![
            ("key".to_string(), AvroValue::Int(field.id)),
            ("value".to_string(), AvroValue::Long(col_stats.null_count)),
        ]));
        if let Some(bytes) = col_stats.min.as_ref().and_then(encode_iceberg_value) {
            lowers.push(AvroValue::Record(vec![
                ("key".to_string(), AvroValue::Int(field.id)),
                ("value".to_string(), AvroValue::Bytes(bytes)),
            ]));
        }
        if let Some(bytes) = col_stats.max.as_ref().and_then(encode_iceberg_value) {
            uppers.push(AvroValue::Record(vec![
                ("key".to_string(), AvroValue::Int(field.id)),
                ("value".to_string(), AvroValue::Bytes(bytes)),
            ]));
        }
    }

    let wrap = |v: Vec<AvroValue>| {
        if v.is_empty() {
            AvroValue::Union(0, Box::new(AvroValue::Null))
        } else {
            AvroValue::Union(1, Box::new(AvroValue::Array(v)))
        }
    };
    (wrap(nulls), wrap(lowers), wrap(uppers))
}

/// Avro type for a partition field, derived from its transform.
///
/// Iceberg's rule: `bucket` and the time transforms (`year`/`month`/`day`/
/// `hour`) always produce an `int`; `identity`, `void` and `truncate` preserve
/// the source column's type. Getting this wrong makes the manifest writer
/// reject the partition value (e.g. an `int` bucket value against a `string`
/// field), which is exactly what blocked cross-partition compaction.
fn partition_field_avro_type(
    field: &crate::core::manifest::PartitionField,
    schema: &crate::core::manifest::Schema,
) -> &'static str {
    let t = field.transform.trim().to_ascii_lowercase();

    if t == "year" || t == "month" || t == "day" || t == "hour" {
        return r#"["null", "int"]"#;
    }
    if t.starts_with("bucket(") || t.starts_with("bucket[") {
        return r#"["null", "int"]"#;
    }

    // identity / void / truncate(...) preserve the source type.
    let source_id = field.source_ids.first().copied().or(field.source_id);
    let resolved = schema
        .fields
        .iter()
        .find(|sf| sf.name == field.name)
        .or_else(|| source_id.and_then(|id| schema.fields.iter().find(|f| f.id == id)));

    match resolved.map(|f| f.type_str.as_str()) {
        Some("Int32") | Some("int") => r#"["null", "int"]"#,
        Some("Int64") | Some("long") => r#"["null", "long"]"#,
        Some("Float32") | Some("float") => r#"["null", "float"]"#,
        Some("Float64") | Some("double") => r#"["null", "double"]"#,
        Some("Boolean") | Some("bool") | Some("boolean") => r#"["null", "boolean"]"#,
        _ => r#"["null", "string"]"#,
    }
}

/// GPU Accelerated Puffin Index Writer for Iceberg
pub struct GpuPuffinWriter {
    // Orchestrates GPU-based index builds (HNSW, Bloom)
}

impl Default for GpuPuffinWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuPuffinWriter {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn build_index(&self, _column: &str) -> Result<String> {
        // GPU build...
        Ok("sidecar_path".to_string())
    }
}

#[cfg(test)]
mod bounds_roundtrip_tests {
    use super::*;
    use crate::core::manifest::{
        ColumnStats, ManifestEntry, ManifestValue, PartitionSpec, Schema, SchemaField,
    };

    fn one_field_schema() -> Schema {
        Schema {
            schema_id: 0,
            fields: vec![SchemaField {
                id: 1,
                name: "id".to_string(),
                type_str: "long".to_string(),
                required: false,
                indexes: Vec::new(),
                fields: Vec::new(),
                initial_default: None,
                write_default: None,
            }],
            identifier_field_ids: Vec::new(),
        }
    }

    /// Does a column with a known min/max actually end up with bounds in the
    /// written manifest? `bounds_avro_values` building them (observed via a
    /// temporary eprintln) is not the same as them surviving serialization.
    #[test]
    fn chunks_write_lower_and_upper_bounds() {
        let schema = one_field_schema();

        let mut stats = std::collections::HashMap::new();
        stats.insert(
            "id".to_string(),
            ColumnStats {
                min: Some(ManifestValue::Int64(40)),
                max: Some(ManifestValue::Int64(42)),
                null_count: 0,
                distinct_count: None,
                vector_stats: None,
            },
        );

        let entry = ManifestEntry {
            file_path: "seg_x.parquet".to_string(),
            file_size_bytes: 10,
            record_count: 3,
            column_stats: stats,
            ..Default::default()
        };

        let chunks = IcebergWriter::new()
            .write_manifest_chunks(
                &[entry],
                &PartitionSpec::default(),
                &schema,
                1,
                1,
                1_000_000,
            )
            .expect("write_manifest_chunks");

        assert!(!chunks.is_empty(), "expected at least one manifest chunk");

        let mut saw_lower = false;
        let mut saw_upper = false;
        for (bytes, _, _) in &chunks {
            let reader = apache_avro::Reader::new(&bytes[..]).expect("avro reader");
            for rec in reader {
                let apache_avro::types::Value::Record(top) = rec.expect("avro record") else {
                    continue;
                };
                // Top level is {status, snapshot_id, ..., data_file}.
                let Some(apache_avro::types::Value::Record(fields)) =
                    top.iter().find(|(k, _)| k == "data_file").map(|(_, v)| v)
                else {
                    continue;
                };
                for (key, value) in fields.iter() {
                    let non_empty_array = matches!(
                        value,
                        apache_avro::types::Value::Union(_, inner)
                            if matches!(inner.as_ref(), apache_avro::types::Value::Array(a) if !a.is_empty())
                    );
                    if key == "lower_bounds" && non_empty_array {
                        saw_lower = true;
                    }
                    if key == "upper_bounds" && non_empty_array {
                        saw_upper = true;
                    }
                }
            }
        }

        assert!(
            saw_lower && saw_upper,
            "manifest did not carry non-empty bounds for a column with min/max \
             (lower={saw_lower}, upper={saw_upper})"
        );
    }
}
