// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Schema management: updating schemas, identifier fields, and index specifications.

use anyhow::Result;
use object_store::path::Path;
use std::collections::HashMap;
use std::sync::Arc;

use super::super::types::*;
use super::ManifestManager;

fn is_already_exists(err: &object_store::Error) -> bool {
    err.to_string().contains("already exists")
}

impl ManifestManager {
    pub async fn update_schema(
        &self,
        new_schemas: Vec<Schema>,
        new_schema_id: i32,
        last_column_id: Option<i32>,
    ) -> Result<Manifest> {
        let dist_lock_path = Path::from(format!("{}/commit.lock", self.manifest_dir));
        let dist_lock =
            crate::core::lock::FileBasedLock::new(self.store.clone(), dist_lock_path, 30);
        let _dist_guard = dist_lock.acquire().await?;

        let max_retries = 10;
        let mut attempt = 0;

        loop {
            // Optimistic Concurrency Control Loop
            let (current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;

            // Re-use entries from latest
            let entries = current_manifest.entries.clone();

            let mut new_manifest = Manifest::new_with_spec(
                new_ver,
                entries,
                Some(current_ver),
                new_schemas.clone(),
                new_schema_id,
                current_manifest.partition_spec.clone(),
            );
            new_manifest.manifest_list_path = current_manifest.manifest_list_path.clone();
            new_manifest.last_column_id = last_column_id.unwrap_or(current_manifest.last_column_id);

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
                    tracing::info!("Committed Manifest v{} (Schema Update)", new_ver);
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE
                        .insert(file_key, Arc::new(new_manifest.clone()))
                        .await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    // Conflict, retry
                    if attempt >= max_retries {
                        break;
                    }
                    attempt += 1;
                    let base_delay = 10 * (2u64.pow(attempt.min(5) as u32));
                    let jitter = rand::random::<u64>() % base_delay;
                    tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
                    continue;
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }
        Err(anyhow::anyhow!(
            "Failed to commit schema update after {} attempts",
            max_retries
        ))
    }

    /// Update the primary key (identifier fields) for the table.
    /// This creates a new schema version with the updated field IDs.
    #[tracing::instrument(skip(self, new_ids))]
    pub async fn update_identifier_fields(&self, new_ids: Vec<i32>) -> Result<Manifest> {
        let (current_manifest, _) = self.load_latest().await?;
        let mut schemas = current_manifest.schemas.clone();

        // Get latest schema and update its identifier fields
        if let Some(latest_schema) = schemas.last_mut() {
            // Check if it's already exactly the same to avoid redundant commits
            if latest_schema.identifier_field_ids == new_ids {
                return Ok(current_manifest);
            }

            // Create a new schema version
            let mut new_schema = latest_schema.clone();
            new_schema.schema_id += 1;
            new_schema.identifier_field_ids = new_ids.clone();

            // AUTOMATIC EVOLUTION: Set identifier fields to required: true (NOT NULL)
            for field in &mut new_schema.fields {
                if new_ids.contains(&field.id) {
                    field.required = true;
                }
            }

            let new_schema_id = new_schema.schema_id;
            schemas.push(new_schema);

            self.update_schema(schemas, new_schema_id, None).await
        } else {
            Err(anyhow::anyhow!(
                "No existing schema found to update identifier fields"
            ))
        }
    }

    /// Update indexing specifications for columns
    pub async fn update_index_specs(
        &self,
        column_indexes: HashMap<String, Vec<IndexAlgorithm>>,
    ) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = self.load_latest().await?;
            let current_schema = current_manifest
                .schemas
                .last()
                .ok_or_else(|| anyhow::anyhow!("No schema found in manifest"))?;

            let mut new_fields = current_schema.fields.clone();
            let mut changed = false;

            for field in &mut new_fields {
                if let Some(new_indexes) = column_indexes.get(&field.name) {
                    field.indexes = new_indexes.clone();
                    changed = true;
                }
            }

            if !changed {
                return Ok(current_manifest);
            }

            let new_schema = Schema {
                schema_id: current_schema.schema_id + 1,
                fields: new_fields,
                identifier_field_ids: current_schema.identifier_field_ids.clone(),
            };

            let mut new_schemas = current_manifest.schemas.clone();
            new_schemas.push(new_schema.clone());

            let new_ver = current_ver + 1;
            let mut new_manifest = Manifest::new_with_spec(
                new_ver,
                current_manifest.entries.clone(),
                Some(current_ver),
                new_schemas,
                new_schema.schema_id,
                current_manifest.partition_spec.clone(),
            );
            new_manifest.manifest_list_path = current_manifest.manifest_list_path.clone();
            new_manifest.properties = current_manifest.properties.clone();

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
                    tracing::info!("Committed Manifest v{} (Index Spec Update)", new_ver);
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
        Err(anyhow::anyhow!("Failed to commit index spec update"))
    }
}
