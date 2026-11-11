// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_glue::{
    types::Column, types::StorageDescriptor, types::TableInput, Client as GlueClient,
};
use std::collections::HashMap;

use super::common::{
    catalog_conflict_error, catalog_not_found_error, ensure_storage_table_initialized,
    extract_metadata_location_from_updates, load_table_metadata_from_location,
    persist_updates_to_storage, table_root_from_metadata_location,
    validate_requirements_against_metadata,
};
use super::{Catalog, TableMetadata};
use arrow::datatypes::SchemaRef;

/// AWS Glue Catalog client (SDK mode)
#[derive(Clone)]
pub struct GlueCatalogClient {
    client: GlueClient,
    catalog_id: Option<String>, // AWS account ID (optional)
}

impl GlueCatalogClient {
    /// Create a new Glue Catalog client
    /// Uses default AWS credentials from environment
    pub async fn new(catalog_id: Option<String>) -> Result<Self> {
        Self::with_region(catalog_id, None).await
    }

    /// Create a Glue Catalog client with an optional explicit AWS region
    pub async fn with_region(catalog_id: Option<String>, region: Option<String>) -> Result<Self> {
        let mut loader = aws_config::defaults(BehaviorVersion::latest());
        if let Some(r) = region.filter(|r| !r.trim().is_empty()) {
            loader = loader.region(Region::new(r));
        }
        let config = loader.load().await;
        let client = GlueClient::new(&config);
        Ok(Self { client, catalog_id })
    }

    /// Create with explicit AWS config
    pub fn from_config(config: &aws_config::SdkConfig, catalog_id: Option<String>) -> Self {
        let client = GlueClient::new(config);
        Self { client, catalog_id }
    }

    async fn commit_table_internal(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        let mut get_req = self
            .client
            .get_table()
            .database_name(namespace)
            .name(table_name);

        if let Some(catalog_id) = &self.catalog_id {
            get_req = get_req.catalog_id(catalog_id);
        }

        let resp = get_req.send().await.map_err(|e| {
            if let aws_sdk_glue::error::SdkError::ServiceError(ref se) = e {
                if se.err().is_entity_not_found_exception() {
                    return catalog_not_found_error(
                        "glue",
                        format!("Glue table '{namespace}.{table_name}' not found"),
                    );
                }
            }
            anyhow::anyhow!("Failed to get Glue table for commit: {}", e)
        })?;

        let existing_table = resp
            .table()
            .ok_or_else(|| catalog_not_found_error("glue", "Table not found in Glue response"))?;

        let version_id = existing_table.version_id().map(str::to_string);
        let existing_params: HashMap<String, String> =
            existing_table.parameters().cloned().unwrap_or_default();

        let existing_meta_loc = existing_params
            .get("metadata_location")
            .cloned()
            .or_else(|| {
                existing_table
                    .storage_descriptor()
                    .and_then(|sd| sd.location().map(str::to_string))
            })
            .unwrap_or_default();

        let existing_meta = if existing_meta_loc.is_empty() {
            None
        } else {
            Some(load_table_metadata_from_location(&existing_meta_loc).await?)
        };

        validate_requirements_against_metadata(&requirements, existing_meta.as_ref(), "glue")?;

        let table_root = table_root_from_metadata_location(&existing_meta_loc);
        let new_metadata_location = match extract_metadata_location_from_updates(&updates) {
            Some(loc) => Some(loc),
            None if !table_root.is_empty() => {
                Some(persist_updates_to_storage(&table_root, &updates).await?)
            }
            None => None,
        };

        let storage_descriptor = existing_table
            .storage_descriptor()
            .cloned()
            .unwrap_or_else(|| StorageDescriptor::builder().build());
        let table_type = existing_table
            .table_type()
            .unwrap_or("EXTERNAL_TABLE")
            .to_string();

        let (table_input, expected_version_id) = build_glue_update_table_input(
            table_name,
            &existing_params,
            &existing_meta_loc,
            new_metadata_location.as_deref(),
            version_id,
            storage_descriptor,
            &table_type,
        )?;

        let mut update_req = self
            .client
            .update_table()
            .database_name(namespace)
            .table_input(table_input)
            .set_version_id(expected_version_id);

        if let Some(catalog_id) = &self.catalog_id {
            update_req = update_req.catalog_id(catalog_id);
        }

        update_req.send().await.map_err(|e| {
            if let aws_sdk_glue::error::SdkError::ServiceError(ref se) = e {
                if se.err().is_concurrent_modification_exception() {
                    return catalog_conflict_error(
                        "glue",
                        format!("Glue concurrent modification on '{namespace}.{table_name}': {e}"),
                    );
                }
                if se.err().is_entity_not_found_exception() {
                    return catalog_not_found_error(
                        "glue",
                        format!("Glue table '{namespace}.{table_name}' not found"),
                    );
                }
            }
            anyhow::anyhow!("Failed to update Glue table: {}", e)
        })?;

        Ok(())
    }
}

pub(crate) fn build_glue_update_table_input(
    table_name: &str,
    existing_params: &HashMap<String, String>,
    existing_meta_loc: &str,
    new_metadata_location: Option<&str>,
    expected_version_id: Option<String>,
    storage_descriptor: StorageDescriptor,
    table_type: &str,
) -> Result<(TableInput, Option<String>)> {
    let mut parameters = existing_params.clone();
    parameters
        .entry("table_type".to_string())
        .or_insert_with(|| "ICEBERG".to_string());

    if let Some(new_loc) = new_metadata_location {
        if !existing_meta_loc.is_empty() && existing_meta_loc != new_loc {
            parameters.insert(
                "previous_metadata_location".to_string(),
                existing_meta_loc.to_string(),
            );
        }
        parameters.insert("metadata_location".to_string(), new_loc.to_string());
    }

    let table_input = TableInput::builder()
        .name(table_name)
        .storage_descriptor(storage_descriptor)
        .table_type(table_type)
        .set_parameters(Some(parameters))
        .build()
        .map_err(|e| anyhow!("Failed to build Glue table input: {}", e))?;

    Ok((table_input, expected_version_id))
}

#[async_trait]
impl Catalog for GlueCatalogClient {
    async fn create_namespace(&self, namespace: &str) -> Result<()> {
        let db_input = aws_sdk_glue::types::DatabaseInput::builder()
            .name(namespace)
            .build()
            .map_err(|e| anyhow!("Failed to build Glue database input: {}", e))?;
        let mut req = self.client.create_database().database_input(db_input);
        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }
        match req.send().await {
            Ok(_) => Ok(()),
            Err(aws_sdk_glue::error::SdkError::ServiceError(se))
                if se.err().is_already_exists_exception() =>
            {
                Ok(())
            }
            Err(e) => Err(anyhow!("Failed to create Glue database: {}", e)),
        }
    }

    async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        let mut req = self.client.get_database().name(namespace);
        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }
        match req.send().await {
            Ok(_) => Ok(true),
            Err(aws_sdk_glue::error::SdkError::ServiceError(se))
                if se.err().is_entity_not_found_exception() =>
            {
                Ok(false)
            }
            Err(e) => Err(anyhow!(
                "Failed to check Glue database '{}': {}",
                namespace,
                e
            )),
        }
    }

    async fn drop_namespace(&self, namespace: &str) -> Result<()> {
        let tables = self.list_tables(namespace).await?;
        if !tables.is_empty() {
            return Err(catalog_conflict_error(
                "glue",
                format!("Glue database '{namespace}' is not empty"),
            ));
        }

        let mut req = self.client.delete_database().name(namespace);
        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }
        match req.send().await {
            Ok(_) => Ok(()),
            Err(aws_sdk_glue::error::SdkError::ServiceError(se))
                if se.err().is_entity_not_found_exception() =>
            {
                Err(catalog_not_found_error(
                    "glue",
                    format!("Glue database '{namespace}' not found"),
                ))
            }
            Err(aws_sdk_glue::error::SdkError::ServiceError(se))
                if se.err().is_concurrent_modification_exception() =>
            {
                Err(catalog_conflict_error(
                    "glue",
                    format!("Glue database '{namespace}' concurrent modification"),
                ))
            }
            Err(e) => Err(anyhow!(
                "Failed to drop Glue database '{}': {}",
                namespace,
                e
            )),
        }
    }

    async fn create_table(
        &self,
        namespace: &str, // Glue database name
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        let clean_root = location.map(|loc| loc.trim_end_matches('/').to_string());
        let initial_metadata_location = if let Some(ref root) = clean_root {
            Some(ensure_storage_table_initialized(root, &schema).await?)
        } else {
            None
        };

        // Extract columns from Arrow schema for Glue
        let columns: Vec<Column> = schema
            .fields()
            .iter()
            .map(|f| {
                Column::builder()
                    .name(f.name())
                    .r#type(f.data_type().to_string())
                    .build()
                    .map_err(|e| {
                        anyhow::anyhow!("Glue column build failed for '{}': {e}", f.name())
                    })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let storage_descriptor = StorageDescriptor::builder()
            .set_location(clean_root)
            .set_columns(Some(columns))
            .build();

        let mut table_input = TableInput::builder()
            .name(table_name)
            .storage_descriptor(storage_descriptor)
            .table_type("EXTERNAL_TABLE");

        let mut parameters = HashMap::new();
        parameters.insert("table_type".to_string(), "ICEBERG".to_string());
        if let Some(meta_loc) = initial_metadata_location {
            parameters.insert("metadata_location".to_string(), meta_loc);
        }
        table_input = table_input.set_parameters(Some(parameters));

        let mut req = self
            .client
            .create_table()
            .database_name(namespace)
            .table_input(
                table_input
                    .build()
                    .map_err(|e| anyhow!("Failed to build table input: {}", e))?,
            );

        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }

        req.send().await.map_err(
            |e: aws_sdk_glue::error::SdkError<
                aws_sdk_glue::operation::create_table::CreateTableError,
            >| {
                if let aws_sdk_glue::error::SdkError::ServiceError(ref se) = e {
                    if se.err().is_already_exists_exception() {
                        return catalog_conflict_error(
                            "glue",
                            format!("Glue table '{namespace}.{table_name}' already exists"),
                        );
                    }
                }
                anyhow::anyhow!("Failed to create Glue table: {}", e)
            },
        )?;

        Ok(())
    }

    async fn drop_table(&self, namespace: &str, table_name: &str) -> Result<()> {
        let mut req = self
            .client
            .delete_table()
            .database_name(namespace)
            .name(table_name);
        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }
        match req.send().await {
            Ok(_) => Ok(()),
            Err(aws_sdk_glue::error::SdkError::ServiceError(se))
                if se.err().is_entity_not_found_exception() =>
            {
                Ok(())
            }
            Err(aws_sdk_glue::error::SdkError::ServiceError(se))
                if se.err().is_concurrent_modification_exception() =>
            {
                Err(catalog_conflict_error(
                    "glue",
                    format!(
                        "Concurrent modification dropping Glue table '{namespace}.{table_name}'"
                    ),
                ))
            }
            Err(e) => Err(anyhow!(
                "Failed to drop Glue table '{namespace}.{table_name}': {e}"
            )),
        }
    }

    async fn list_tables(&self, namespace: &str) -> Result<Vec<String>> {
        let mut names = Vec::new();
        let mut next_token: Option<String> = None;

        loop {
            let mut req = self
                .client
                .get_tables()
                .database_name(namespace)
                .set_next_token(next_token.take());
            if let Some(catalog_id) = &self.catalog_id {
                req = req.catalog_id(catalog_id);
            }

            let resp = req.send().await.map_err(|e| {
                if let aws_sdk_glue::error::SdkError::ServiceError(ref se) = e {
                    if se.err().is_entity_not_found_exception() {
                        return catalog_not_found_error(
                            "glue",
                            format!("Glue database '{namespace}' not found"),
                        );
                    }
                }
                anyhow!("Failed to list Glue tables in '{namespace}': {e}")
            })?;

            for tbl in resp.table_list() {
                names.push(tbl.name().to_string());
            }

            match resp.next_token() {
                Some(tok) if !tok.is_empty() => next_token = Some(tok.to_string()),
                _ => break,
            }
        }

        names.sort();
        Ok(names)
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let mut req = self
            .client
            .get_table()
            .database_name(namespace)
            .name(table_name);

        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }

        let resp = req.send().await.map_err(
            |e: aws_sdk_glue::error::SdkError<
                aws_sdk_glue::operation::get_table::GetTableError,
            >| {
                if let aws_sdk_glue::error::SdkError::ServiceError(ref se) = e {
                    if se.err().is_entity_not_found_exception() {
                        return catalog_not_found_error(
                            "glue",
                            format!("Glue table '{namespace}.{table_name}' not found"),
                        );
                    }
                }
                anyhow::anyhow!("Failed to get Glue table: {}", e)
            },
        )?;

        let table = resp
            .table()
            .ok_or_else(|| catalog_not_found_error("glue", "Table not found in Glue response"))?;

        let location = table
            .parameters()
            .and_then(|p| p.get("metadata_location"))
            .cloned()
            .or_else(|| {
                table
                    .storage_descriptor()
                    .and_then(|sd| sd.location().map(str::to_string))
            })
            .ok_or_else(|| anyhow!("No metadata_location in Glue table parameters"))?;

        load_table_metadata_from_location(&location).await
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Err(anyhow!(
            "AWS Glue does not support branching. Use Nessie for Git-like branching."
        ))
    }

    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        match self.load_table(namespace, table_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, Vec::new())
            .await
    }

    async fn commit_table_with_requirements(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, requirements)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glue_update_table_input_preserves_version_id_and_previous_location() -> Result<()> {
        let mut existing = HashMap::new();
        existing.insert("custom_key".to_string(), "val".to_string());
        existing.insert(
            "metadata_location".to_string(),
            "s3://bucket/tbl/metadata/v1.metadata.json".to_string(),
        );

        let sd = StorageDescriptor::builder()
            .location("s3://bucket/tbl")
            .build();

        let (input, ver) = build_glue_update_table_input(
            "events",
            &existing,
            "s3://bucket/tbl/metadata/v1.metadata.json",
            Some("s3://bucket/tbl/metadata/v2.metadata.json"),
            Some("v-123".to_string()),
            sd,
            "EXTERNAL_TABLE",
        )?;

        assert_eq!(ver.as_deref(), Some("v-123"));
        let params = input
            .parameters()
            .ok_or_else(|| anyhow!("missing params"))?;
        assert_eq!(params.get("custom_key").map(String::as_str), Some("val"));
        assert_eq!(
            params.get("table_type").map(String::as_str),
            Some("ICEBERG")
        );
        assert_eq!(
            params.get("previous_metadata_location").map(String::as_str),
            Some("s3://bucket/tbl/metadata/v1.metadata.json")
        );
        assert_eq!(
            params.get("metadata_location").map(String::as_str),
            Some("s3://bucket/tbl/metadata/v2.metadata.json")
        );
        Ok(())
    }
}
