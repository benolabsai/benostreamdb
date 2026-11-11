use anyhow::Result;
use arrow::array::{Array, BooleanBuilder, UInt32Builder};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
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

pub fn align_batch_to_schema(
    batch: &RecordBatch,
    target_schema: &arrow::datatypes::SchemaRef,
) -> Result<RecordBatch> {
    if batch.schema() == *target_schema {
        return Ok(batch.clone());
    }
    let mut cols = Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        if let Some(col) = batch.column_by_name(field.name()) {
            if col.data_type() == field.data_type() {
                cols.push(col.clone());
            } else {
                cols.push(arrow::compute::cast(col.as_ref(), field.data_type())?);
            }
        } else {
            cols.push(arrow::array::new_null_array(
                field.data_type(),
                batch.num_rows(),
            ));
        }
    }
    Ok(RecordBatch::try_new(target_schema.clone(), cols)?)
}

pub struct MergePlanner {
    /// Single current-thread runtime reused for every async turn driven by
    /// this planner. Building a runtime per operation was the dominant cost
    /// for merges touching many segments.
    runtime: OnceLock<tokio::runtime::Runtime>,
    store: Option<Arc<dyn object_store::ObjectStore>>,
    data_store: Option<Arc<dyn object_store::ObjectStore>>,
    partition_spec: crate::core::manifest::PartitionSpec,
    iceberg_schema: Option<crate::core::manifest::Schema>,
    table_schema: Option<arrow::datatypes::SchemaRef>,
    primary_key: Vec<String>,
    index_all: bool,
    index_columns: Vec<String>,
    index_configs: HashMap<String, crate::core::table::ColumnIndexConfig>,
    default_device: Option<String>,
    global_centroids: HashMap<String, Vec<Vec<f32>>>,
    format_version: i32,
}

impl Default for MergePlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl MergePlanner {
    pub fn new() -> Self {
        Self {
            runtime: OnceLock::new(),
            store: None,
            data_store: None,
            partition_spec: crate::core::manifest::PartitionSpec::default(),
            iceberg_schema: None,
            table_schema: None,
            primary_key: Vec::new(),
            index_all: false,
            index_columns: Vec::new(),
            index_configs: HashMap::new(),
            default_device: None,
            global_centroids: HashMap::new(),
            format_version: 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_table_context(
        mut self,
        store: Arc<dyn object_store::ObjectStore>,
        data_store: Option<Arc<dyn object_store::ObjectStore>>,
        partition_spec: crate::core::manifest::PartitionSpec,
        iceberg_schema: Option<crate::core::manifest::Schema>,
        table_schema: Option<arrow::datatypes::SchemaRef>,
        primary_key: Vec<String>,
        index_all: bool,
        index_columns: Vec<String>,
        index_configs: HashMap<String, crate::core::table::ColumnIndexConfig>,
        default_device: Option<String>,
        global_centroids: HashMap<String, Vec<Vec<f32>>>,
        format_version: i32,
    ) -> Self {
        self.store = Some(store);
        self.data_store = data_store;
        self.partition_spec = partition_spec;
        self.iceberg_schema = iceberg_schema;
        self.table_schema = table_schema;
        self.primary_key = primary_key;
        self.index_all = index_all;
        self.index_columns = index_columns;
        self.index_configs = index_configs;
        self.default_device = default_device;
        self.global_centroids = global_centroids;
        self.format_version = format_version;
        self
    }

    fn resolve_store(&self, base_path: &str) -> Result<Arc<dyn object_store::ObjectStore>> {
        match &self.store {
            Some(s) => Ok(s.clone()),
            None => crate::core::storage::create_object_store(base_path),
        }
    }

    fn should_build_indexes(&self, batch: &RecordBatch) -> bool {
        self.index_all
            || !self.index_columns.is_empty()
            || !self.index_configs.is_empty()
            || !self.primary_key.is_empty()
            || batch.schema().fields().iter().any(|f| {
                matches!(
                    f.data_type(),
                    arrow::datatypes::DataType::FixedSizeList(_, _)
                        | arrow::datatypes::DataType::List(_)
                )
            })
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

        // Pre-convert unique keys to serde_json::Value for Bloom testing.
        let mut test_values = Vec::new();
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

        let store = self.resolve_store(base_path)?;

        for entry in candidate_entries {
            let rel_parquet = crate::core::segment::relativize_against(base_path, &entry.file_path);
            let seg_id = crate::core::segment::segment_id_from_path(&rel_parquet).to_string();
            let mut read_config = SegmentConfig::new("", &seg_id)
                .with_parquet_path(rel_parquet)
                .with_index_files(entry.index_files.clone())
                .with_delete_files(entry.delete_files.clone())
                .with_record_count(entry.record_count as u64)
                .with_file_checksum(entry.file_checksum.clone());
            if let Some(ds) = self.data_store.clone() {
                read_config = read_config.with_data_store(ds);
            }
            let mut reader =
                crate::core::reader::HybridReader::new(read_config, store.clone(), base_path);
            if let Some(ref ischema) = self.iceberg_schema {
                reader = reader.with_iceberg_schema(ischema.clone());
            }

            // Check every candidate key within a single runtime turn; the
            // reader caches Parquet metadata, so repeated checks per segment
            // no longer re-open the file or re-create a runtime.
            let possible = self.runtime_block_on(async {
                for val in &test_values {
                    if reader
                        .check_bloom_filter(key_col_name, val)
                        .await
                        .unwrap_or(true)
                    {
                        return true;
                    }
                }
                false
            });

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

        let store = self.resolve_store(base_path)?;
        let is_local_uri = !base_path.contains("://") || base_path.starts_with("file://");
        let base_for_writer = base_path
            .strip_prefix("file://")
            .unwrap_or(base_path)
            .trim_end_matches('/');

        // 1. Identify which rows in source match which segments
        let mut deleted_by_segment: HashMap<String, roaring::RoaringBitmap> = HashMap::new();
        for entry in candidate_entries {
            let rel_parquet = crate::core::segment::relativize_against(base_path, &entry.file_path);
            let seg_id = crate::core::segment::segment_id_from_path(&rel_parquet).to_string();
            tracing::info!("Checking Segment {} for merge overlaps", seg_id);

            let del_reader = if !entry.delete_files.is_empty() {
                let mut del_config = SegmentConfig::new("", &seg_id)
                    .with_parquet_path(rel_parquet.clone())
                    .with_delete_files(entry.delete_files.clone())
                    .with_record_count(entry.record_count as u64);
                if let Some(ds) = self.data_store.clone() {
                    del_config = del_config.with_data_store(ds);
                }
                let mut r =
                    crate::core::reader::HybridReader::new(del_config, store.clone(), base_path);
                if let Some(ref ischema) = self.iceberg_schema {
                    r = r.with_iceberg_schema(ischema.clone());
                }
                Some(r)
            } else {
                None
            };

            let mut read_config = SegmentConfig::new("", &seg_id)
                .with_parquet_path(rel_parquet)
                .with_index_files(entry.index_files.clone())
                .with_record_count(entry.record_count as u64)
                .with_file_checksum(entry.file_checksum.clone());
            if let Some(ds) = self.data_store.clone() {
                read_config = read_config.with_data_store(ds);
            }
            let mut reader =
                crate::core::reader::HybridReader::new(read_config, store.clone(), base_path);
            if let Some(ref ischema) = self.iceberg_schema {
                reader = reader.with_iceberg_schema(ischema.clone());
            }

            // ONLY load the key columns into memory to prevent OOM.
            // Drain the whole stream within a single runtime turn instead of
            // re-entering block_on for every batch poll.
            // Read without delete skipping so `i` equals physical Parquet row offset,
            // while skipping rows already present in `deleted_bm`.
            let (original_batches, deleted_bm) = self.runtime_block_on(async {
                let del_bm = match del_reader {
                    Some(dr) => dr.load_merged_deletes().await.unwrap_or_default(),
                    None => roaring::RoaringBitmap::new(),
                };
                let mut stream = reader
                    .stream_raw_row_groups(None, Some(key_schema.clone()))
                    .await?;
                let mut batches = Vec::new();
                while let Some(b) = stream.next().await {
                    batches.push(b?);
                }
                Ok::<(Vec<RecordBatch>, roaring::RoaringBitmap), anyhow::Error>((batches, del_bm))
            })?;

            if original_batches.is_empty() {
                continue;
            }
            if !deleted_bm.is_empty() {
                deleted_by_segment.insert(seg_id.clone(), deleted_bm.clone());
            }

            let aligned_key_batches: Vec<RecordBatch> = original_batches
                .iter()
                .map(|b| align_batch_to_schema(b, &key_schema))
                .collect::<Result<Vec<_>>>()?;
            let original_batch = arrow::compute::concat_batches(&key_schema, &aligned_key_batches)?;

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

            // Index segment rows by their encoded key for O(1) lookups.
            // First occurrence wins.
            let mut seg_key_index: HashMap<Vec<u8>, usize> =
                HashMap::with_capacity(original_batch.num_rows());
            for i in 0..original_batch.num_rows() {
                if deleted_bm.contains(i as u32) {
                    continue;
                }
                seg_key_index
                    .entry(seg_rows.row(i).as_ref().to_vec())
                    .or_insert(i);
            }

            for (row_idx, encoded_key) in source_keys_encoded.iter().enumerate() {
                if !unmatched_rows.contains(&row_idx) {
                    continue;
                }

                if let Some(&seg_idx) = seg_key_index.get(encoded_key.as_slice()) {
                    updates_by_segment
                        .entry(seg_id.clone())
                        .or_default()
                        .push((row_idx, seg_idx));
                    unmatched_rows.remove(&row_idx);
                }
            }
        }

        let mut commit_actions = Vec::new();
        let format_version = self.format_version.max(2);

        let delete_writer = crate::core::iceberg::iceberg_delete::IcebergDeleteWriter::new(
            base_path.to_string(),
            format_version,
            store.clone(),
        );

        let cow_target_schema = self
            .table_schema
            .clone()
            .unwrap_or_else(|| source_batch.schema());

        // 2. Process matched updates
        for (seg_id, matched_rows) in updates_by_segment {
            tracing::info!(
                "Updating Segment {} with {} rows (Mode: {:?})",
                seg_id,
                matched_rows.len(),
                mode
            );

            // Find the original ManifestEntry using normalized relative path / segment_id
            let mut original_entry = None;
            for e in candidate_entries {
                let rel = crate::core::segment::relativize_against(base_path, &e.file_path);
                let cand_id = crate::core::segment::segment_id_from_path(&rel);
                if cand_id == seg_id || e.file_path == seg_id {
                    original_entry = Some(e.clone());
                    break;
                }
            }
            let Some(original_entry) = original_entry else {
                tracing::warn!(
                    "No manifest entry found for segment {}; skipping its update",
                    seg_id
                );
                continue;
            };
            let old_path = original_entry.file_path.clone();

            if mode == MergeMode::CopyOnWrite {
                // CoW: We MUST read the entire segment to rewrite it without the deleted/updated rows.
                let rel_parquet =
                    crate::core::segment::relativize_against(base_path, &original_entry.file_path);
                let mut read_config = SegmentConfig::new("", &seg_id)
                    .with_parquet_path(rel_parquet.clone())
                    .with_record_count(original_entry.record_count as u64)
                    .with_file_checksum(original_entry.file_checksum.clone());
                if let Some(ds) = self.data_store.clone() {
                    read_config = read_config.with_data_store(ds);
                }
                let mut reader =
                    crate::core::reader::HybridReader::new(read_config, store.clone(), base_path);
                if let Some(ref ischema) = self.iceberg_schema {
                    reader = reader.with_iceberg_schema(ischema.clone());
                }

                let original_batches = self.runtime_block_on(async {
                    let mut stream = reader.stream_raw_row_groups(None, None).await?;
                    let mut batches = Vec::new();
                    while let Some(b) = stream.next().await {
                        batches.push(b?);
                    }
                    Ok::<Vec<RecordBatch>, anyhow::Error>(batches)
                })?;
                let aligned_original_batches: Vec<RecordBatch> = original_batches
                    .iter()
                    .map(|b| align_batch_to_schema(b, &cow_target_schema))
                    .collect::<Result<Vec<_>>>()?;
                let original_batch =
                    arrow::compute::concat_batches(&cow_target_schema, &aligned_original_batches)?;

                let mut deleted_segment_rows = HashSet::new();
                let mut source_row_indices = Vec::new();
                for &(src_idx, seg_idx) in &matched_rows {
                    deleted_segment_rows.insert(seg_idx);
                    source_row_indices.push(src_idx);
                }
                let already_deleted = deleted_by_segment.get(&seg_id);

                let mut keep_indices_builder = BooleanBuilder::new();
                for i in 0..original_batch.num_rows() {
                    let was_deleted = already_deleted.is_some_and(|bm| bm.contains(i as u32));
                    keep_indices_builder
                        .append_value(!was_deleted && !deleted_segment_rows.contains(&i));
                }
                let keep_mask = keep_indices_builder.finish();
                let filtered_batch =
                    arrow::compute::filter_record_batch(&original_batch, &keep_mask)?;

                let mut indices_builder = UInt32Builder::new();
                for &idx in &source_row_indices {
                    indices_builder.append_value(idx as u32);
                }
                let indices_arr = indices_builder.finish();
                let raw_updates_batch =
                    arrow::compute::take_record_batch(source_batch, &indices_arr)?;
                let updates_batch = align_batch_to_schema(&raw_updates_batch, &cow_target_schema)?;

                let new_batch = arrow::compute::concat_batches(
                    &cow_target_schema,
                    &[filtered_batch, updates_batch],
                )?;

                let rel_dir = rel_parquet
                    .rfind('/')
                    .map(|pos| rel_parquet[..pos].trim_matches('/'))
                    .unwrap_or("");
                let hive_dir = if !original_entry.partition_values.is_empty()
                    && !self.partition_spec.fields.is_empty()
                {
                    let derived = self
                        .partition_spec
                        .partition_to_path(&original_entry.partition_values);
                    if derived.is_empty() {
                        rel_dir.to_string()
                    } else {
                        derived
                    }
                } else {
                    rel_dir.to_string()
                };

                let new_seg_id = format!("seg_{}", Uuid::new_v4());
                let rel_new_parquet = crate::core::segment::ensure_prefix(
                    &hive_dir,
                    &format!("{new_seg_id}.parquet"),
                );
                let full_base = if hive_dir.is_empty() {
                    base_for_writer.to_string()
                } else {
                    format!("{}/{}", base_for_writer, hive_dir)
                };
                if is_local_uri && !full_base.is_empty() {
                    std::fs::create_dir_all(&full_base)?;
                }
                let new_config = SegmentConfig::new(&full_base, &new_seg_id)
                    .with_index_all(self.index_all)
                    .with_columns_to_index(self.index_columns.clone())
                    .with_default_device(self.default_device.clone())
                    .with_parquet_path(rel_new_parquet)
                    .with_partition_values(
                        original_entry
                            .partition_values
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    );
                let mut writer = crate::core::segment::HybridSegmentWriter::new(new_config)
                    .with_index_configs(self.index_configs.clone())
                    .with_global_centroids(self.global_centroids.clone());
                writer.primary_key = self.primary_key.clone();
                writer.set_store(store.clone());
                writer.write_batch(&new_batch)?;
                if self.should_build_indexes(&new_batch) {
                    writer.build_indexes(&new_batch, 0)?;
                    self.runtime_block_on(writer.finish_indexing())?;
                }
                let mut new_entry = writer.to_manifest_entry();
                self.runtime_block_on(writer.upload_to_store())?;
                new_entry.file_path =
                    crate::core::segment::ensure_prefix(&hive_dir, &new_entry.file_path);
                for idx in &mut new_entry.index_files {
                    idx.file_path = crate::core::segment::ensure_prefix(&hive_dir, &idx.file_path);
                }

                commit_actions.push(MergeCommitAction::ReplaceData {
                    old_segment_path: old_path,
                    new_entry,
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
                    // V3 tables express merge-on-read deletes as deletion
                    // vectors: one `deletion-vector-v1` Puffin blob per data
                    // file per snapshot. V2 tables keep Parquet position-delete
                    // files, which every V2 reader can decode.
                    let delete_file = if format_version >= 3 {
                        let rel = crate::core::segment::relativize_against(
                            base_path,
                            &original_entry.file_path,
                        );
                        let rel_parent = rel
                            .rfind('/')
                            .map(|pos| rel[..pos].to_string())
                            .unwrap_or_default();
                        self.runtime_block_on(
                            crate::core::iceberg::delete::write_deletion_vector_puffin(
                                &store,
                                base_path,
                                &original_entry.file_path,
                                &rel_parent,
                                &deleted_positions,
                                &original_entry.partition_values,
                            ),
                        )?
                    } else {
                        let file_path_col = arrow::array::StringArray::from(deleted_file_paths);
                        let pos_col = arrow::array::Int64Array::from(deleted_positions);

                        let partition_data = if !original_entry.partition_values.is_empty() {
                            let rel = crate::core::segment::relativize_against(
                                base_path,
                                &original_entry.file_path,
                            );
                            let rel_path = rel
                                .rfind('/')
                                .map(|pos| rel[..pos].to_string())
                                .unwrap_or_default();
                            Some((rel_path, original_entry.partition_values.clone()))
                        } else {
                            None
                        };

                        self.runtime_block_on(delete_writer.write_position_delete(
                            partition_data,
                            &file_path_col,
                            &pos_col,
                        ))?
                    };

                    let mut updated_entry = original_entry.clone();
                    updated_entry.delete_files.push(delete_file);

                    commit_actions.push(MergeCommitAction::AddDelete {
                        old_segment_path: old_path,
                        updated_entry,
                    });
                }
            }
        }

        // 3. Insert unmatched rows (or updated rows in MoR), honoring partition_spec and index settings
        if !unmatched_rows.is_empty() {
            tracing::info!("Inserting {} new rows", unmatched_rows.len());
            let mut unmatched_vec: Vec<usize> = unmatched_rows.into_iter().collect();
            unmatched_vec.sort_unstable(); // preserve order

            let mut indices_builder = UInt32Builder::new();
            for &idx in &unmatched_vec {
                indices_builder.append_value(idx as u32);
            }
            let indices_arr = indices_builder.finish();
            let raw_inserts = arrow::compute::take_record_batch(source_batch, &indices_arr)?;
            let aligned_inserts = if let Some(ref ts) = self.table_schema {
                align_batch_to_schema(&raw_inserts, ts)?
            } else {
                raw_inserts
            };

            let partitioned = self.partition_spec.partition_batch(&aligned_inserts)?;
            for (partition_values, part_batch) in partitioned {
                if part_batch.num_rows() == 0 {
                    continue;
                }
                let hive_dir = self.partition_spec.partition_to_path(&partition_values);
                let new_seg_id = format!("seg_{}", Uuid::new_v4());
                let rel_new_parquet = crate::core::segment::ensure_prefix(
                    &hive_dir,
                    &format!("{new_seg_id}.parquet"),
                );
                let full_base = if hive_dir.is_empty() {
                    base_for_writer.to_string()
                } else {
                    format!("{}/{}", base_for_writer, hive_dir)
                };
                if is_local_uri && !full_base.is_empty() {
                    std::fs::create_dir_all(&full_base)?;
                }

                let new_config = SegmentConfig::new(&full_base, &new_seg_id)
                    .with_index_all(self.index_all)
                    .with_columns_to_index(self.index_columns.clone())
                    .with_default_device(self.default_device.clone())
                    .with_parquet_path(rel_new_parquet)
                    .with_partition_values(partition_values);
                let mut writer = crate::core::segment::HybridSegmentWriter::new(new_config)
                    .with_index_configs(self.index_configs.clone())
                    .with_global_centroids(self.global_centroids.clone());
                writer.primary_key = self.primary_key.clone();
                writer.set_store(store.clone());
                writer.write_batch(&part_batch)?;
                if self.should_build_indexes(&part_batch) {
                    writer.build_indexes(&part_batch, 0)?;
                    self.runtime_block_on(writer.finish_indexing())?;
                }
                let mut new_entry = writer.to_manifest_entry();
                self.runtime_block_on(writer.upload_to_store())?;
                new_entry.file_path =
                    crate::core::segment::ensure_prefix(&hive_dir, &new_entry.file_path);
                for idx in &mut new_entry.index_files {
                    idx.file_path = crate::core::segment::ensure_prefix(&hive_dir, &idx.file_path);
                }

                commit_actions.push(MergeCommitAction::AddData { new_entry });
            }
        }

        Ok(commit_actions)
    }

    // `Runtime::build()` has no infallible form (it fails only if the OS cannot
    // give us a reactor), and `join()` on a thread that panicked has no way to
    // produce a `T` other than re-panicking. Both are intentional, documented
    // invariants; see NO_PANIC_POLICY.md.
    #[allow(clippy::expect_used)]
    fn runtime_block_on<T: Send, F: std::future::Future<Output = T> + Send>(&self, future: F) -> T {
        // A nested `block_on` on a freshly built runtime panics with
        // "Cannot start a runtime from within a runtime" when the current
        // thread is already driving an async context.
        match tokio::runtime::Handle::try_current() {
            // Fast path: reuse the ambient multi-threaded runtime in place.
            // `block_in_place` parks this worker so the rest of the pool keeps
            // running; no runtime or thread is created per call.
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(future))
            }
            // A current-thread runtime would panic on `block_in_place`; offload
            // the future to a dedicated thread with its own runtime.
            Ok(_) => std::thread::scope(|s| {
                s.spawn(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to create Tokio runtime for merge operation")
                        .block_on(future)
                })
                .join()
                .expect("Merge runtime thread panicked")
            }),
            // No ambient runtime: reuse one current-thread runtime for the
            // whole planner lifetime.
            Err(_) => self
                .runtime
                .get_or_init(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to create Tokio runtime for merge operation")
                })
                .block_on(future),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the "Cannot start a runtime from within a runtime"
    /// panic (see tests/python/test_compound_pk.py CI failure): the sync merge
    /// entry points must not panic even when invoked from inside a running
    /// tokio runtime.
    #[test]
    fn runtime_block_on_inside_runtime_does_not_panic() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let planner = MergePlanner::new();
            let value = planner.runtime_block_on(async { 42u8 });
            assert_eq!(value, 42);
        });
    }

    #[test]
    fn runtime_block_on_outside_runtime_works() {
        let planner = MergePlanner::new();
        let value = planner.runtime_block_on(async { "ok" });
        assert_eq!(value, "ok");
    }
}
