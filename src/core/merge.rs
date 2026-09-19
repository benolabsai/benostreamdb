use anyhow::Result;
use arrow::array::{Array, BooleanBuilder, UInt32Builder};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::core::manifest::ManifestEntry;
use crate::core::table::merge::MergeMode;
use crate::SegmentConfig;

pub enum MergeCommitAction {
    /// Overwrite an entire data file (CoW)
    ReplaceData {
        old_segment_path: String,
        new_entry: ManifestEntry,
    },
    /// Add a delete file to an existing data file (MoR)
    AddDelete {
        old_segment_path: String,
        updated_entry: ManifestEntry,
    },
    /// Add entirely new data (Inserts)
    AddData { new_entry: ManifestEntry },
}

pub struct MergePlanner {}

impl Default for MergePlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl MergePlanner {
    pub fn new() -> Self {
        Self {}
    }

    /// Prune segments using Bloom Filters to quickly discard segments that cannot possibly match the source keys.
    pub fn prune_segments(
        &self,
        base_path: &str,
        candidate_entries: &[ManifestEntry],
        key_columns: &[&str],
        source_batch: &RecordBatch,
    ) -> Result<Vec<ManifestEntry>> {
        let mut pruned = Vec::new();

        // Optimization: For multiple key columns, we could check all, but checking the primary one (first) is often enough for Bloom.
        let key_col_name = if let Some(k) = key_columns.first() {
            k
        } else {
            return Ok(candidate_entries.to_vec());
        };

        let col = match source_batch.column_by_name(key_col_name) {
            Some(c) => c,
            None => return Ok(candidate_entries.to_vec()), // column not found, fallback
        };

        // Pre-convert a sample of keys to serde_json::Value for Bloom testing
        let mut test_values = Vec::new();
        // Take up to 10 keys to test bloom filters. If a segment matches ANY of them, we keep it.
        // For large batches, it's safer to just check a few distinct keys or all if it's small.
        // But to be completely correct, a segment might match key 999 and not key 0.
        // We MUST check all keys. To avoid checking 10k keys per segment, we collect unique keys first.
        let mut unique_keys = std::collections::HashSet::new();
        for i in 0..source_batch.num_rows() {
            let val = crate::core::manifest::ManifestValue::from_array(col, i);
            let json_val = match val {
                crate::core::manifest::ManifestValue::String(s) => serde_json::Value::String(s),
                crate::core::manifest::ManifestValue::Int32(i) => serde_json::json!(i),
                crate::core::manifest::ManifestValue::Int64(i) => serde_json::json!(i),
                crate::core::manifest::ManifestValue::Float32(f) => serde_json::json!(f),
                crate::core::manifest::ManifestValue::Float64(f) => serde_json::json!(f),
                crate::core::manifest::ManifestValue::Boolean(b) => serde_json::Value::Bool(b),
                crate::core::manifest::ManifestValue::Null => serde_json::Value::Null,
            };
            if !json_val.is_null() && unique_keys.insert(json_val.to_string()) {
                test_values.push(json_val);
            }
        }

        for entry in candidate_entries {
            let seg_id = entry
                .file_path
                .split('/')
                .next_back()
                .unwrap_or(&entry.file_path)
                .replace(".parquet", "");
            let read_config = SegmentConfig::new("", &seg_id);
            let store = crate::core::storage::create_object_store(base_path)?;
            let reader = crate::core::reader::HybridReader::new(read_config, store, base_path);

            let mut possible = false;
            for val in &test_values {
                let might_match = self
                    .runtime_block_on(reader.check_bloom_filter(key_col_name, val))
                    .unwrap_or(true);
                if might_match {
                    possible = true;
                    break;
                }
            }

            if possible {
                pruned.push(entry.clone());
            } else {
                tracing::info!(
                    "Bloom Filter pruned segment: {} for column: {}",
                    seg_id,
                    key_col_name
                );
            }
        }

        Ok(pruned)
    }

    /// Execute the merge according to MergeMode
    pub fn execute_merge(
        &self,
        base_path: &str,
        mode: MergeMode,
        key_columns: &[&str],
        source_keys_encoded: &[Vec<u8>],
        _source_key_set: &HashSet<Vec<u8>>,
        source_batch: &RecordBatch,
        candidate_entries: &[ManifestEntry],
    ) -> Result<Vec<MergeCommitAction>> {
        // maps seg_id -> Vec<(source_row_idx, segment_row_idx)>
        let mut updates_by_segment: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        let mut unmatched_rows: HashSet<usize> = (0..source_batch.num_rows()).collect();

        // Prepare projected schema for reading ONLY key columns
        let schema = source_batch.schema();
        let mut key_fields = Vec::new();
        for &k in key_columns {
            key_fields.push(schema.field_with_name(k)?.clone());
        }
        let key_schema = Arc::new(arrow::datatypes::Schema::new(key_fields));

        // 1. Identify which rows in source match which segments
        for entry in candidate_entries {
            let seg_id = entry
                .file_path
                .split('/')
                .next_back()
                .unwrap_or(&entry.file_path)
                .replace(".parquet", "");
            tracing::info!("Checking Segment {} for merge overlaps", seg_id);

            let read_config = SegmentConfig::new("", &seg_id);
            let store = crate::core::storage::create_object_store(base_path)?;
            let reader = crate::core::reader::HybridReader::new(read_config, store, base_path);

            let mut original_batches = Vec::new();
            use futures::StreamExt;
            // ONLY load the key columns into memory to prevent OOM
            let mut stream = self.runtime_block_on(reader.stream_all(Some(key_schema.clone())))?;
            while let Some(b) = self.runtime_block_on(stream.next()) {
                original_batches.push(b?);
            }

            if original_batches.is_empty() {
                continue;
            }

            let original_batch = arrow::compute::concat_batches(&key_schema, &original_batches)?;

            // Encode segment keys
            let mut col_indices = Vec::new();
            let mut fields = Vec::new();
            for &col_name in key_columns {
                let idx = key_schema.index_of(col_name)?;
                col_indices.push(idx);
                fields.push(SortField::new(key_schema.field(idx).data_type().clone()));
            }

            let converter = RowConverter::new(fields)?;
            let key_arrays: Vec<Arc<dyn Array>> = col_indices
                .iter()
                .map(|&idx| original_batch.column(idx).clone())
                .collect();
            let seg_rows = converter.convert_columns(&key_arrays)?;

            for (row_idx, encoded_key) in source_keys_encoded.iter().enumerate() {
                if !unmatched_rows.contains(&row_idx) {
                    continue;
                }

                // Linear scan for now, since we need the exact row matched.
                // Could be optimized by storing seg_rows in a HashMap -> row_index
                let mut found_seg_idx = None;
                for i in 0..original_batch.num_rows() {
                    if seg_rows.row(i).as_ref() == encoded_key.as_slice() {
                        found_seg_idx = Some(i);
                        break;
                    }
                }

                if let Some(seg_idx) = found_seg_idx {
                    updates_by_segment
                        .entry(seg_id.clone())
                        .or_default()
                        .push((row_idx, seg_idx));
                    unmatched_rows.remove(&row_idx);
                }
            }
        }

        let mut commit_actions = Vec::new();
        let format_version = 2; // Iceberg v2

        let delete_writer = crate::core::iceberg::iceberg_delete::IcebergDeleteWriter::new(
            base_path.to_string(),
            format_version,
        );

        // 2. Process matched updates
        for (seg_id, matched_rows) in updates_by_segment {
            tracing::info!(
                "Updating Segment {} with {} rows (Mode: {:?})",
                seg_id,
                matched_rows.len(),
                mode
            );

            // Find the original ManifestEntry
            let mut original_entry = None;
            for e in candidate_entries {
                if e.file_path.contains(&seg_id) {
                    original_entry = Some(e.clone());
                    break;
                }
            }
            let original_entry = original_entry.unwrap();
            let old_path = original_entry.file_path.clone();

            if mode == MergeMode::CopyOnWrite {
                // CoW: We MUST read the entire segment to rewrite it without the deleted rows.
                let read_config = SegmentConfig::new("", &seg_id);
                let store = crate::core::storage::create_object_store(base_path)?;
                let reader = crate::core::reader::HybridReader::new(read_config, store, base_path);

                let mut original_batches = Vec::new();
                use futures::StreamExt;
                let mut stream = self.runtime_block_on(reader.stream_all(None))?;
                while let Some(b) = self.runtime_block_on(stream.next()) {
                    original_batches.push(b?);
                }
                let original_batch =
                    arrow::compute::concat_batches(&source_batch.schema(), &original_batches)?;

                let mut deleted_segment_rows = HashSet::new();
                let mut source_row_indices = Vec::new();
                for &(src_idx, seg_idx) in &matched_rows {
                    deleted_segment_rows.insert(seg_idx);
                    source_row_indices.push(src_idx);
                }

                let mut keep_indices_builder = BooleanBuilder::new();
                for i in 0..original_batch.num_rows() {
                    keep_indices_builder.append_value(!deleted_segment_rows.contains(&i));
                }
                let keep_mask = keep_indices_builder.finish();
                let filtered_batch =
                    arrow::compute::filter_record_batch(&original_batch, &keep_mask)?;

                let mut indices_builder = UInt32Builder::new();
                for &idx in &source_row_indices {
                    indices_builder.append_value(idx as u32);
                }
                let indices_arr = indices_builder.finish();
                let updates_batch = arrow::compute::take_record_batch(source_batch, &indices_arr)?;

                let new_batch = arrow::compute::concat_batches(
                    &source_batch.schema(),
                    &[filtered_batch, updates_batch],
                )?;

                let new_seg_id = format!("seg_{}", Uuid::new_v4());
                let new_config = SegmentConfig::new(base_path, &new_seg_id);
                let writer = crate::core::segment::HybridSegmentWriter::new(new_config);
                writer.write_batch(&new_batch)?;

                commit_actions.push(MergeCommitAction::ReplaceData {
                    old_segment_path: old_path,
                    new_entry: writer.to_manifest_entry(),
                });
            } else {
                // MoR: We DO NOT need to read the old segment! We already have the deleted row positions from Step 1.
                let mut deleted_positions = Vec::new();
                let mut deleted_file_paths = Vec::new();

                for &(src_idx, seg_idx) in &matched_rows {
                    deleted_positions.push(seg_idx as i64);
                    deleted_file_paths.push(old_path.clone());
                    // For MoR, we STILL need to insert the updated rows as new data!
                    // Add them to unmatched_rows so they are inserted at the end.
                    unmatched_rows.insert(src_idx);
                }

                if !deleted_positions.is_empty() {
                    let file_path_col = arrow::array::StringArray::from(deleted_file_paths);
                    let pos_col = arrow::array::Int64Array::from(deleted_positions);

                    let partition_data = if !original_entry.partition_values.is_empty() {
                        let path = std::path::Path::new(&original_entry.file_path);
                        let rel_path = path
                            .parent()
                            .and_then(|p| p.to_str())
                            .unwrap_or("")
                            .to_string();
                        Some((rel_path, original_entry.partition_values.clone()))
                    } else {
                        None
                    };

                    let delete_file =
                        self.runtime_block_on(delete_writer.write_position_delete(
                            partition_data,
                            &file_path_col,
                            &pos_col,
                        ))?;

                    let mut updated_entry = original_entry.clone();
                    updated_entry.delete_files.push(delete_file);

                    commit_actions.push(MergeCommitAction::AddDelete {
                        old_segment_path: old_path,
                        updated_entry,
                    });
                }
            }
        }

        // 3. Insert unmatched rows (or updated rows in MoR)
        if !unmatched_rows.is_empty() {
            tracing::info!("Inserting {} new rows", unmatched_rows.len());
            let mut unmatched_vec: Vec<usize> = unmatched_rows.into_iter().collect();
            unmatched_vec.sort_unstable(); // preserve order

            let mut indices_builder = UInt32Builder::new();
            for &idx in &unmatched_vec {
                indices_builder.append_value(idx as u32);
            }
            let indices_arr = indices_builder.finish();
            let inserts_batch = arrow::compute::take_record_batch(source_batch, &indices_arr)?;

            let new_seg_id = format!("seg_{}", Uuid::new_v4());
            let new_config = SegmentConfig::new(base_path, &new_seg_id);
            let writer = crate::core::segment::HybridSegmentWriter::new(new_config);
            writer.write_batch(&inserts_batch)?;

            commit_actions.push(MergeCommitAction::AddData {
                new_entry: writer.to_manifest_entry(),
            });
        }

        Ok(commit_actions)
    }

    fn runtime_block_on<T, F: std::future::Future<Output = T>>(&self, future: F) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime for merge operation")
            .block_on(future)
    }
}
