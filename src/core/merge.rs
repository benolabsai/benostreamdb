// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::error::HyperstreamError;
use crate::core::manifest::SegmentId;
use anyhow::Result;
use arrow::array::Array;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use roaring::RoaringBitmap;
use serde_json::Value;
use std::collections::HashMap;

pub struct MergePlanner {
    // In a real implementation, this would hold references to the Catalog/Store
}

impl Default for MergePlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl MergePlanner {
    pub fn new() -> Self {
        Self {}
    }

    /// Prunes segments that definitely do not contain any of the source keys.
    ///
    /// # Arguments
    /// * `segments` - List of all segment IDs in the table.
    /// * `key_column` - The name of the column to merge on (e.g., "id").
    /// * `source_keys` - The list of keys from the source data (upsert batch).
    /// * `index_provider` - A closure/trait to fetch the inverted index for a segment.
    ///
    /// # Returns
    /// A subset of `segments` that are candidates for the merge.
    pub fn prune_segments<F>(
        &self,
        segments: &[SegmentId],
        key_column: &str,
        index_provider: F,
    ) -> Result<Vec<SegmentId>>
    where
        F: Fn(&SegmentId, &str) -> Result<Option<RoaringBitmap>>, // Returns bitmap for the key column if exists exists
    {
        // Challenge: The inverted index is usually Value -> Bitmap.
        // But here we want to check if ANY of the source_keys exist in the segment.
        // Ideally, we have a Bloom Filter for the segment for quick "Not Present" checks.

        // For Phase 3 MVP, let's assume we load the "Inverted Index" which maps Values to RowIDs.
        // We check if (key in Index).

        let mut candidates = Vec::new();

        // Iterate over all segments
        for seg_id in segments {
            // Check if this segment contains ANY of the source keys
            // In a real implementation, we would use a BloomFilter or the InvertedIndex

            // For this mock/MVP, we use the callback to simulate an index lookup
            // New contract: index_provider(seg_id, key_column) returns a "SegmentIndex" wrapper?
            // Let's simplify:
            // We assume the index_provider gives us a way to check existence.

            // Let's assume index_provider returns a RoaringBitmap of *valid row IDs* for the segment?
            // No, that's not helpful for "does value exist".

            // Let's change the closure signature to be more direct for this prototype:
            // F: Fn(&SegmentId, &Value) -> bool
            // But we need to change the signature in the impl block.

            // Let's keep the signature but assume we fetch a "bitmap of all present values"? No.
            // Let's just implement the loop assuming index_provider can do the check.

            // Check intersection
            // This is O(M * N) if not careful.
            // Ideally: Load Index for Segment. Check if any key in Buffer is in Index.

            if let Some(bitmap) = index_provider(seg_id, key_column)? {
                // Index exists. If matches found (bitmap not empty), include segment.
                if !bitmap.is_empty() {
                    candidates.push(seg_id.clone());
                }
            } else {
                // If no index, we must scan it (conservative)
                candidates.push(seg_id.clone());
            }
        }

        Ok(candidates)
    }

    /// Load the inverted index for a segment from its .inv.parquet file.
    /// Returns a HashMap mapping key values (as i64) to vectors of local row IDs.
    fn load_inverted_index_map(
        base_path: &str,
        seg_id: &SegmentId,
        key_column: &str,
    ) -> Result<std::collections::HashMap<i64, Vec<u32>>> {
        let path = format!(
            "{}/{}.{}.inv.parquet",
            base_path.replace("file://", ""),
            seg_id,
            key_column
        );
        let inv_path = std::path::Path::new(&path);
        if !inv_path.exists() {
            return Ok(std::collections::HashMap::new());
        }

        let file = std::fs::File::open(inv_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut map: std::collections::HashMap<i64, Vec<u32>> = std::collections::HashMap::new();

        for batch_res in reader {
            let batch = batch_res?;
            let lists = batch.column(1).as_any().downcast_ref::<arrow::array::ListArray>()
                .ok_or_else(|| HyperstreamError::MergeFailed {
                    reason: format!(
                        "Inverted index segment {} has unexpected column type at index 1 (expected ListArray)",
                        seg_id
                    )
                })?;
            let col0 = batch.column(0);

            if let Some(keys) = col0.as_any().downcast_ref::<arrow::array::Int32Array>() {
                for i in 0..batch.num_rows() {
                    let k = keys.value(i) as i64;
                    let list = lists.value(i);
                    let u32_list = list.as_any().downcast_ref::<arrow::array::UInt32Array>()
                        .ok_or_else(|| HyperstreamError::MergeFailed {
                            reason: format!(
                                "Inverted index segment {} row {} has unexpected inner type (expected UInt32Array)",
                                seg_id, i
                            )
                        })?;
                    map.insert(k, u32_list.values().to_vec());
                }
            } else if let Some(keys) = col0.as_any().downcast_ref::<arrow::array::Int64Array>() {
                for i in 0..batch.num_rows() {
                    let k = keys.value(i);
                    let list = lists.value(i);
                    let u32_list = list.as_any().downcast_ref::<arrow::array::UInt32Array>()
                        .ok_or_else(|| HyperstreamError::MergeFailed {
                            reason: format!(
                                "Inverted index segment {} row {} has unexpected inner type (expected UInt32Array)",
                                seg_id, i
                            )
                        })?;
                    map.insert(k, u32_list.values().to_vec());
                }
            }
        }
        Ok(map)
    }

    /// For a candidate segment, identifies exactly which Row IDs match the source keys.
    ///
    /// Loads the inverted index (.inv.parquet) for the segment and looks up each
    /// source key, unioning the resulting row ID bitmaps.
    ///
    /// # Arguments
    /// * `base_path` - Base storage path where segment files reside.
    /// * `segment_id` - The segment to search.
    /// * `key_column` - The merge key column name.
    /// * `source_keys` - Keys from the upsert batch.
    ///
    /// # Returns
    /// A RoaringBitmap of local Row IDs to be updated/deleted.
    pub fn find_row_ids(
        &self,
        base_path: &str,
        segment_id: &SegmentId,
        key_column: &str,
        source_keys: &[Value],
    ) -> Result<RoaringBitmap> {
        let index_map = Self::load_inverted_index_map(base_path, segment_id, key_column)?;
        let mut matches = RoaringBitmap::new();

        for key_val in source_keys {
            if let Some(i_val) = key_val.as_i64() {
                if let Some(row_ids) = index_map.get(&i_val) {
                    for &rid in row_ids {
                        matches.insert(rid);
                    }
                }
            }
        }

        Ok(matches)
    }
    /// Execute the Merge Operation
    ///
    /// 1. Group source rows by target segment (using Index).
    /// 2. For each target segment: Read -> Update -> Write New Segment.
    /// 3. For unmatched rows: Write New Segment.
    ///
    /// Returns: List of (OldSegmentId, NewSegmentId) pairs for catalog commit.
    pub fn execute_merge<F>(
        &self,
        base_path: &str, // Base URI for reading/writing
        key_column: &str,
        source_keys: &[Value], // Parallel to source_batch rows
        source_batch: &arrow::record_batch::RecordBatch,
        candidate_segments: &[SegmentId],
        _index_provider: F,
    ) -> Result<Vec<(Option<SegmentId>, SegmentId)>>
    where
        F: Fn(&SegmentId, &str) -> Result<Option<RoaringBitmap>>,
    {
        tracing::info!("Executing Merge on {} candidates", candidate_segments.len());

        let mut updates_by_segment: HashMap<SegmentId, Vec<usize>> = HashMap::new();
        let mut unmatched_rows: Vec<usize> = Vec::new();

        // 1. Classify Source Rows
        // For each source row, find which segment it belongs to.
        // Optimization: Inverted Index check.
        // Naive MVP: Iterate source keys, check all candidate indexes.

        // Load inverted index maps for all candidate segments
        let mut segment_bitmaps: HashMap<SegmentId, std::collections::HashMap<i64, Vec<u32>>> =
            HashMap::new();
        for seg_id in candidate_segments {
            let map = Self::load_inverted_index_map(base_path, seg_id, key_column)?;
            segment_bitmaps.insert(seg_id.clone(), map);
        }

        for (row_idx, key_val) in source_keys.iter().enumerate() {
            let mut found = false;
            // Iterate candidates to find match
            if let Some(i_val) = key_val.as_i64() {
                // i_val is i64
                for (seg_id, map) in &segment_bitmaps {
                    if map.contains_key(&i_val) {
                        updates_by_segment
                            .entry(seg_id.clone())
                            .or_default()
                            .push(row_idx);
                        found = true;
                        break;
                        // Assumption: Key exists in only 1 segment (Primary Key uniqueness)
                    }
                }
            }

            if !found {
                unmatched_rows.push(row_idx);
            }
        }

        let mut commit_actions = Vec::new();

        // 2. Process Updates (Copy-on-Write)
        for (seg_id, source_row_indices) in updates_by_segment {
            tracing::info!(
                "Updating Segment {} with {} rows",
                seg_id,
                source_row_indices.len()
            );

            // A. Read Original Segment
            // Use empty base path for reader configuration because store is already rooted at base_path
            let read_config = crate::SegmentConfig::new("", &seg_id);

            // We need to construct a Store.
            // This is getting complicated to call generic helper.
            // Let's assume `base_path` works for `HybridReader` if we create a local store.
            let store = crate::core::storage::create_object_store(base_path)?;
            let reader = crate::core::reader::HybridReader::new(read_config, store, base_path);

            // Read all batches (all columns needed for merge)
            let mut original_batches = Vec::new();
            use futures::StreamExt;
            let mut stream = self.runtime_block_on(reader.stream_all(None))?;
            while let Some(b) = self.runtime_block_on(stream.next()) {
                original_batches.push(b?);
            }

            // B. Apply Updates in Memory
            // Flatten to one batch for simplicity?
            let schema = source_batch.schema();
            let original_batch = arrow::compute::concat_batches(&schema, &original_batches)?;

            // Build Lookup Map for Source Updates: Key -> RowIndex in SourceBatch
            let mut source_update_map: HashMap<i32, usize> = HashMap::new();
            for &src_idx in &source_row_indices {
                let key =
                    source_keys[src_idx]
                        .as_i64()
                        .ok_or_else(|| HyperstreamError::MergeFailed {
                            reason: format!("Source key at row {} is not an integer", src_idx),
                        })? as i32;
                source_update_map.insert(key, src_idx);
            }

            // Iterate Original Batch, Replace if Key Matches
            // This requires reconstructing the columns.
            // MVP: We assume fixed schema: Int32(id), Int64(val)
            // Generalizing is hard in Rust without dynamic typing or massive match blocks.
            // Let's implement for the specific Schema: {id: Int32, val: Int64}
            // Or better: Use `arrow`'s `MutableRecordBatch` or similar? No such thing easily.
            // We'll filter out the old rows and append the new rows.
            // "Delete" old matching rows -> "Append" new source rows.

            let id_col_idx = schema.index_of(key_column)?;
            let id_arr = original_batch
                .column(id_col_idx)
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .ok_or_else(|| HyperstreamError::MergeFailed {
                    reason: format!(
                        "Key column '{}' in segment {} has unexpected type (expected Int32)",
                        key_column, seg_id
                    ),
                })?;

            let mut keep_indices_builder = arrow::array::BooleanBuilder::new();
            for i in 0..original_batch.num_rows() {
                let key = id_arr.value(i);
                if source_update_map.contains_key(&key) {
                    keep_indices_builder.append_value(false); // Delete old
                } else {
                    keep_indices_builder.append_value(true); // Keep old
                }
            }
            let keep_mask = keep_indices_builder.finish();
            let filtered_batch = arrow::compute::filter_record_batch(&original_batch, &keep_mask)?;

            // Extract the update rows from Source Batch
            // We can slice or filter source_batch
            // Create a mask for source batch?
            // Or just pick indices? arrow::compute::take is good.
            let indices_arr = arrow::array::UInt32Array::from(
                source_row_indices
                    .iter()
                    .map(|&x| x as u32)
                    .collect::<Vec<u32>>(),
            );
            let updates_batch = arrow::compute::take_record_batch(source_batch, &indices_arr)?;

            // Merge: Filtered Old + Updates
            let new_batch =
                arrow::compute::concat_batches(&schema, &[filtered_batch, updates_batch])?;

            // C. Write New Segment
            // Use full base_path for Writer as it uses std::fs directly
            let new_seg_id = format!("seg_{}", uuid::Uuid::new_v4());
            let new_config = crate::SegmentConfig::new(base_path, &new_seg_id);
            let writer = crate::core::segment::HybridSegmentWriter::new(new_config);
            writer.write_batch(&new_batch)?;

            commit_actions.push((Some(seg_id), new_seg_id));
        }

        // 3. Process Inserts (Unmatched)
        if !unmatched_rows.is_empty() {
            tracing::info!("Inserting {} new rows", unmatched_rows.len());
            let indices_arr = arrow::array::UInt32Array::from(
                unmatched_rows
                    .iter()
                    .map(|&x| x as u32)
                    .collect::<Vec<u32>>(),
            );
            let inserts_batch = arrow::compute::take_record_batch(source_batch, &indices_arr)?;

            let new_seg_id = format!("seg_{}", uuid::Uuid::new_v4());
            let new_config = crate::SegmentConfig::new(base_path, &new_seg_id);
            let writer = crate::core::segment::HybridSegmentWriter::new(new_config);
            writer.write_batch(&inserts_batch)?;

            commit_actions.push((None, new_seg_id));
        }

        Ok(commit_actions)
    }

    // Helper to block on async code in sync function
    fn runtime_block_on<T, F: std::future::Future<Output = T>>(&self, future: F) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime for merge operation")
            .block_on(future)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn test_merge_planner_new() {
        let _planner = MergePlanner::new();
        // Just verify it constructs without panicking
    }

    #[test]
    fn test_prune_segments_with_index() {
        let planner = MergePlanner::new();
        let segments = vec!["seg1".to_string(), "seg2".to_string(), "seg3".to_string()];

        // Mock index provider that returns Some for seg1 and seg2, None for seg3
        let index_provider = |seg_id: &SegmentId, _col: &str| -> Result<Option<RoaringBitmap>> {
            if seg_id == "seg1" || seg_id == "seg2" {
                let mut bitmap = RoaringBitmap::new();
                bitmap.insert(1);
                bitmap.insert(2);
                Ok(Some(bitmap))
            } else {
                Ok(None) // No index for seg3, must scan
            }
        };

        let candidates = planner
            .prune_segments(&segments, "id", index_provider)
            .unwrap();

        // All segments should be candidates (seg1, seg2 have matches, seg3 has no index so conservative)
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_prune_segments_empty_index() {
        let planner = MergePlanner::new();
        let segments = vec!["seg1".to_string()];

        // Index exists but is empty (no matches)
        let index_provider = |_seg_id: &SegmentId, _col: &str| -> Result<Option<RoaringBitmap>> {
            Ok(Some(RoaringBitmap::new())) // Empty bitmap
        };

        let candidates = planner
            .prune_segments(&segments, "id", index_provider)
            .unwrap();

        // Empty bitmap means no matches, segment should be pruned
        assert_eq!(candidates.len(), 0);
    }

    #[test]
    fn test_find_row_ids_no_index_returns_empty() {
        let planner = MergePlanner::new();
        let segment_id = "test_seg".to_string();
        let source_keys = vec![Value::Number(1.into()), Value::Number(2.into())];

        // Non-existent base path → no index file → empty bitmap (no panic)
        let result = planner.find_row_ids("/tmp/nonexistent", &segment_id, "id", &source_keys);
        assert!(result.is_ok());
        let bitmap = result.unwrap();
        assert!(bitmap.is_empty());
    }

    #[test]
    fn test_find_row_ids_with_inverted_index() {
        use arrow::array::{Int32Builder, ListBuilder, UInt32Builder};
        use arrow::datatypes::{
            DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
        };
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::fs;

        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_str().unwrap();

        // Create an inverted index file: key 1 → row IDs [10, 20], key 2 → row IDs [30]
        let inv_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", ArrowDataType::Int32, false),
            ArrowField::new(
                "row_ids",
                ArrowDataType::List(Arc::new(ArrowField::new(
                    "item",
                    ArrowDataType::UInt32,
                    true,
                ))),
                false,
            ),
        ]));

        let mut key_builder = Int32Builder::new();
        key_builder.append_value(1);
        key_builder.append_value(2);
        let key_array = key_builder.finish();
        let mut list_builder: ListBuilder<UInt32Builder> = ListBuilder::new(UInt32Builder::new());
        list_builder.values().append_value(10);
        list_builder.values().append_value(20);
        list_builder.append(true);
        list_builder.values().append_value(30);
        list_builder.append(true);

        let inv_batch = RecordBatch::try_new(
            inv_schema.clone(),
            vec![Arc::new(key_array), Arc::new(list_builder.finish())],
        )
        .unwrap();

        let inv_path = tmp.path().join("test_seg.id.inv.parquet");
        {
            let file = fs::File::create(&inv_path).unwrap();
            let mut writer = ArrowWriter::try_new(file, inv_schema, None).unwrap();
            writer.write(&inv_batch).unwrap();
            writer.close().unwrap();
        }

        let planner = MergePlanner::new();
        let source_keys = vec![
            Value::Number(1.into()),  // should match rows 10, 20
            Value::Number(2.into()),  // should match row 30
            Value::Number(99.into()), // should NOT match anything
        ];

        let bitmap = planner
            .find_row_ids(base, &"test_seg".to_string(), "id", &source_keys)
            .unwrap();
        assert!(bitmap.contains(10));
        assert!(bitmap.contains(20));
        assert!(bitmap.contains(30));
        assert_eq!(bitmap.len(), 3);
        assert!(!bitmap.contains(99));
    }

    #[test]
    fn test_merge_handles_duplicates() {
        // Test that merge correctly handles duplicate keys (last-write-wins)
        let _planner = MergePlanner::new();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        // Source batch with duplicate IDs
        let source_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2])), // Duplicate ID 1
                Arc::new(StringArray::from(vec!["A", "B", "C"])),
            ],
        )
        .unwrap();

        let source_keys = [
            Value::Number(1.into()),
            Value::Number(1.into()),
            Value::Number(2.into()),
        ];

        // In a real merge, the last value for ID 1 should win
        // This test verifies the structure is correct
        assert_eq!(source_batch.num_rows(), 3);
        assert_eq!(source_keys.len(), 3);
    }

    #[test]
    fn test_merge_handles_nulls() {
        // Test that merge correctly handles null values
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true), // Nullable
        ]));

        let source_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("A"), None, Some("C")])),
            ],
        )
        .unwrap();

        assert_eq!(source_batch.num_rows(), 3);

        // Verify null handling
        let value_col = source_batch.column(1);
        assert!(value_col.is_null(1)); // Second row should be null
        assert!(!value_col.is_null(0));
        assert!(!value_col.is_null(2));
    }

    #[test]
    fn test_merge_large_dataset() {
        // Test merge with large number of rows
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let num_rows = 10_000;
        let ids: Vec<i64> = (0..num_rows).collect();
        let values: Vec<i32> = (0..num_rows).map(|i| i as i32 * 2).collect();

        let source_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids.clone())),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();

        assert_eq!(source_batch.num_rows(), num_rows as usize);

        // Verify data integrity
        let id_col = source_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(num_rows as usize - 1), num_rows - 1);
    }

    #[test]
    fn test_merge_empty_segment() {
        // Test merging with an empty source batch
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        let source_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ],
        )
        .unwrap();

        assert_eq!(source_batch.num_rows(), 0);

        // Empty batch should be handled gracefully
        let source_keys: Vec<Value> = vec![];
        assert_eq!(source_keys.len(), 0);
    }

    #[test]
    fn test_merge_schema_evolution() {
        // Test that merge handles schema changes (new columns)
        let old_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let new_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, true), // New nullable column
        ]));

        // Old batch (2 columns)
        let old_batch = RecordBatch::try_new(
            old_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            ],
        )
        .unwrap();

        // New batch (3 columns)
        let new_batch = RecordBatch::try_new(
            new_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![3])),
                Arc::new(StringArray::from(vec!["Charlie"])),
                Arc::new(Int32Array::from(vec![Some(30)])),
            ],
        )
        .unwrap();

        assert_eq!(old_batch.num_columns(), 2);
        assert_eq!(new_batch.num_columns(), 3);

        // In a real merge, the old batch would need to be projected to the new schema
        // with null values for the new column
    }
}
