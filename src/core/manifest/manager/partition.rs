// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Partition specification management: updating partition specs and batch partitioning.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::super::types::*;
use super::ManifestManager;

fn is_already_exists(err: &object_store::Error) -> bool {
    err.to_string().contains("already exists")
}

impl ManifestManager {
    /// Update partition specification
    pub async fn update_partition_spec(&self, new_spec: PartitionSpec) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;

            let new_manifest = Manifest::new_with_spec(
                new_ver,
                current_manifest.entries.clone(),
                Some(current_ver),
                current_manifest.schemas.clone(),
                current_manifest.current_schema_id,
                new_spec.clone(),
            );

            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;

            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };

            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{} (Partition Spec Update)", new_ver);
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE
                        .insert(file_key, Arc::new(new_manifest.clone()))
                        .await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    attempt += 1;
                    if attempt >= max_retries {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(anyhow::anyhow!("Failed to commit partition spec update"))
    }
}

impl PartitionSpec {
    pub fn partition_batch(
        &self,
        batch: &RecordBatch,
    ) -> Result<Vec<(HashMap<String, Value>, RecordBatch)>> {
        if self.fields.is_empty() {
            return Ok(vec![(HashMap::new(), batch.clone())]);
        }

        // 1. Group row indices by partition key
        let mut row_groups: HashMap<Vec<Value>, Vec<u32>> = HashMap::new();

        for i in 0..batch.num_rows() {
            let mut key = Vec::with_capacity(self.fields.len());
            for field in &self.fields {
                // Determine source columns
                let source_ids = field.get_source_ids();
                let mut cols = Vec::new();

                // Prioritize finding by Name (most intuitive for users)
                let mut found = false;
                if let Ok(idx) = batch.schema().index_of(&field.name) {
                    cols.push(batch.column(idx));
                    found = true;
                }

                // Fallback to Iceberg IDs only if name lookup failed
                if !found {
                    for id in &source_ids {
                        let idx = batch.schema().fields().iter().position(|f| {
                            f.metadata()
                                .get("iceberg.id")
                                .and_then(|id_str| id_str.parse::<i32>().ok())
                                .map(|found_id| found_id == *id)
                                .unwrap_or(false)
                        });
                        if let Some(i) = idx {
                            cols.push(batch.column(i));
                        }
                    }
                }

                if cols.is_empty() {
                    anyhow::bail!("Cannot find source column for partition field '{}'. Ensure the column exists in the batch or is named correctly.", field.name);
                }

                // Take the first matching column for the partition key value
                let col = &cols[0];
                let manifest_val = crate::core::manifest::ManifestValue::from_array(col, i);
                key.push(manifest_val.to_json_value());
            }

            row_groups.entry(key).or_default().push(i as u32);
        }

        // 2. Build partition values map and slice batches
        let mut result = Vec::new();
        for (key_vec, row_indices) in row_groups {
            let mut partition_values = HashMap::new();
            for (idx, field) in self.fields.iter().enumerate() {
                partition_values.insert(field.name.clone(), key_vec[idx].clone());
            }

            // Use arrow compute take to slice the batch
            let indices = arrow::array::UInt32Array::from(row_indices);
            let sliced_columns: Vec<std::sync::Arc<dyn arrow::array::Array>> = batch
                .columns()
                .iter()
                .map(|col| {
                    let taken = arrow::compute::take(col.as_ref(), &indices, None)
                        .unwrap_or_else(|_| col.clone());
                    std::sync::Arc::new(taken) as std::sync::Arc<dyn arrow::array::Array>
                })
                .collect();
            let sliced = RecordBatch::try_new(batch.schema(), sliced_columns)?;

            result.push((partition_values, sliced));
        }

        Ok(result)
    }
}
