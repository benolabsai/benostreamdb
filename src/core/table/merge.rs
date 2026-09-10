// Copyright (c) 2026 Richard Albright. All rights reserved.

/// ACID Merge (Upsert) implementation for Table.
///
/// Supports:
/// - Merge-on-Read (MoR): generates deletion vectors and appends new records.
/// - Merge-on-Write (MoW): rewrites affected Parquet segments with updated records.
use anyhow::Result;
use arrow::record_batch::RecordBatch;
use serde_json::Value;

use super::Table;
use crate::core::manifest::{ManifestEntry, ManifestManager};

/// Merge strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    MergeOnRead,
    MergeOnWrite,
}

impl Table {
    /// Merge (Upsert) batches into the table
    pub fn merge(
        &self,
        batches: Vec<RecordBatch>,
        key_column: &str,
        mode: MergeMode,
    ) -> Result<()> {
        match mode {
            MergeMode::MergeOnRead => self.merge_on_read(batches, key_column),
            MergeMode::MergeOnWrite => self.merge_on_write(batches, key_column),
        }
    }

    fn merge_on_read(&self, batches: Vec<RecordBatch>, key_column: &str) -> Result<()> {
        let key_cols: Vec<&str> = key_column.split(',').collect();
        self.runtime().block_on(async {
            for batch in &batches {
                let schema = batch.schema();
                let col_indices: Vec<usize> = key_cols
                    .iter()
                    .map(|&c| schema.index_of(c))
                    .collect::<Result<Vec<usize>, _>>()?;

                for i in 0..batch.num_rows() {
                    let mut filters = Vec::new();
                    for (&col_name, &col_idx) in key_cols.iter().zip(col_indices.iter()) {
                        let col = batch.column(col_idx);
                        let val = if let Some(arr) =
                            col.as_any().downcast_ref::<arrow::array::Int32Array>()
                        {
                            format!("{}", arr.value(i))
                        } else if let Some(arr) =
                            col.as_any().downcast_ref::<arrow::array::Int64Array>()
                        {
                            format!("{}", arr.value(i))
                        } else if let Some(arr) =
                            col.as_any().downcast_ref::<arrow::array::StringArray>()
                        {
                            format!("'{}'", arr.value(i))
                        } else {
                            // Fallback for other types
                            "".to_string()
                        };

                        if !val.is_empty() {
                            filters.push(format!("{} = {}", col_name, val));
                        }
                    }

                    if !filters.is_empty() {
                        let filter_expr = filters.join(" AND ");
                        self.delete_async(&filter_expr).await?;
                    }
                }
            }

            // Step 2: Write the new data (Append)
            self.write_async(batches).await?;
            Ok(())
        })
    }

    fn merge_on_write(&self, batches: Vec<RecordBatch>, key_column: &str) -> Result<()> {
        // MoW uses MergePlanner to rewrite segments
        use crate::core::merge::MergePlanner;

        if batches.is_empty() {
            return Ok(());
        }

        let schema = batches[0].schema();
        let source_batch = arrow::compute::concat_batches(&schema, &batches)?;

        // Extract keys as JSON values for MergePlanner
        let mut source_keys = Vec::new();
        let col_idx = schema.index_of(key_column)?;
        let col = source_batch.column(col_idx);

        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
            for i in 0..arr.len() {
                source_keys.push(Value::Number(arr.value(i).into()));
            }
        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
            for i in 0..arr.len() {
                source_keys.push(Value::Number(arr.value(i).into()));
            }
        } else {
            return Err(anyhow::anyhow!("MoW currently only supports integer keys"));
        }

        self.runtime().block_on(async {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

            let segment_ids: Vec<String> = all_entries
                .iter()
                .map(|e| {
                    e.file_path
                        .split('/')
                        .next_back()
                        .unwrap_or(&e.file_path)
                        .replace(".parquet", "")
                })
                .collect();

            let planner = MergePlanner::new();
            let commit_actions = planner.execute_merge(
                &self.uri,
                key_column,
                &source_keys,
                &source_batch,
                &segment_ids,
                |_, _| Ok(None),
            )?;

            let mut new_entries = Vec::new();
            let mut removed_paths = Vec::new();

            for (old_seg, new_seg) in commit_actions {
                if let Some(old) = old_seg {
                    removed_paths.push(format!("{}.parquet", old));
                }

                let entry = ManifestEntry {
                    file_path: format!("{}.parquet", new_seg),
                    ..Default::default()
                };
                new_entries.push(entry);
            }

            manifest_manager
                .commit(
                    &new_entries,
                    &removed_paths,
                    crate::core::manifest::CommitMetadata::default(),
                )
                .await?;
            Ok(())
        })
    }
}
