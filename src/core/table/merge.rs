// Copyright (c) 2026 Richard Albright. All rights reserved.

/// ACID Merge (Upsert) implementation for Table.
///
/// Supports:
/// - Merge-on-Read (MoR): generates deletion vectors and appends new records.
/// - Merge-on-Write (MoW): rewrites affected Parquet segments with updated records.
use anyhow::Result;
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};
use std::collections::HashSet;
use std::sync::Arc;

use super::Table;
use crate::core::manifest::ManifestManager;
use crate::core::merge::{MergeCommitAction, MergePlanner};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    MergeOnRead,
    CopyOnWrite,
}

impl Table {
    /// Merge (Upsert) batches into the table
    pub fn merge(
        &self,
        batches: Vec<RecordBatch>,
        key_column: &str,
        mode: MergeMode,
    ) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }

        let schema = batches[0].schema();
        let source_batch = arrow::compute::concat_batches(&schema, &batches)?;

        let key_cols: Vec<&str> = key_column.split(',').collect();

        // Build RowConverter to encode source keys for fast comparison
        let mut col_indices = Vec::new();
        let mut fields = Vec::new();
        for &col_name in &key_cols {
            let idx = schema.index_of(col_name)?;
            col_indices.push(idx);
            fields.push(SortField::new(schema.field(idx).data_type().clone()));
        }

        let converter = RowConverter::new(fields)?;
        let key_arrays: Vec<Arc<dyn Array>> = col_indices
            .iter()
            .map(|&idx| source_batch.column(idx).clone())
            .collect();
        let source_rows = converter.convert_columns(&key_arrays)?;

        let mut source_keys_encoded = Vec::with_capacity(source_batch.num_rows());
        let mut source_key_set = HashSet::new();
        for i in 0..source_batch.num_rows() {
            let encoded = source_rows.row(i).as_ref().to_vec();
            source_keys_encoded.push(encoded.clone());
            source_key_set.insert(encoded);
        }

        self.runtime().block_on(async {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (_manifest, candidate_entries, _) = manifest_manager.load_latest_full().await?;

            let planner = MergePlanner::new();

            // Prune segments
            let pruned_entries =
                planner.prune_segments(&self.uri, &candidate_entries, &key_cols, &source_batch)?;

            // Execute merge
            let commit_actions = planner.execute_merge(
                &self.uri,
                mode,
                &key_cols,
                &source_keys_encoded,
                &source_key_set,
                &source_batch,
                &pruned_entries,
            )?;

            let mut add_entries = Vec::new();
            let mut remove_paths = Vec::new();

            for action in commit_actions {
                match action {
                    MergeCommitAction::ReplaceData {
                        old_segment_path,
                        new_entry,
                    } => {
                        remove_paths.push(old_segment_path);
                        add_entries.push(new_entry);
                    }
                    MergeCommitAction::AddDelete {
                        old_segment_path,
                        updated_entry,
                    } => {
                        remove_paths.push(old_segment_path);
                        add_entries.push(updated_entry);
                    }
                    MergeCommitAction::AddData { new_entry } => {
                        add_entries.push(new_entry);
                    }
                }
            }

            if !add_entries.is_empty() || !remove_paths.is_empty() {
                manifest_manager
                    .commit(
                        &add_entries,
                        &remove_paths,
                        crate::core::manifest::CommitMetadata::default(),
                    )
                    .await?;
            }
            Ok(())
        })
    }
}
