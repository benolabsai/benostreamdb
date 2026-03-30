// Copyright (c) 2026 Richard Albright. All rights reserved.

use aws_config::BehaviorVersion;
use aws_sdk_glue::{Client as GlueClient, types::TableInput, types::StorageDescriptor, types::Column};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::collections::HashMap;

use super::{Catalog, TableMetadata};
use arrow::datatypes::SchemaRef;

/// AWS Glue Catalog client
#[derive(Clone)]
pub struct GlueCatalogClient {
    client: GlueClient,
    catalog_id: Option<String>,  // AWS account ID (optional)
}

impl GlueCatalogClient {
    /// Create a new Glue Catalog client
    /// Uses default AWS credentials from environment
    pub async fn new(catalog_id: Option<String>) -> Result<Self> {
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = GlueClient::new(&config);
        Ok(Self { client, catalog_id })
    }
    
    /// Create with explicit AWS config
    pub fn from_config(config: &aws_config::SdkConfig, catalog_id: Option<String>) -> Self {
        let client = GlueClient::new(config);
        Self { client, catalog_id }
    }
}

#[async_trait]
impl Catalog for GlueCatalogClient {
    async fn create_table(
        &self,
        namespace: &str,  // Glue database name
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        // 1. Initialize table manifest if location is provided
        if let Some(loc) = location {
             crate::Table::create_async(loc.to_string(), schema.clone()).await?;
        }

        // 2. Extract columns from Arrow schema for Glue
        let columns: Vec<Column> = schema.fields().iter().map(|f| {
            Column::builder()
                .name(f.name())
                .r#type(f.data_type().to_string())
                .build()
                .unwrap() // Glue Column build is usually infallible for simple cases
        }).collect();
        
        // Create storage descriptor
        let storage_descriptor = StorageDescriptor::builder()
            .set_location(location.map(|s| s.to_string()))
            .set_columns(Some(columns))
            .build();
        
        // Create table input
        let mut table_input = TableInput::builder()
            .name(table_name)
            .storage_descriptor(storage_descriptor)
            .table_type("EXTERNAL_TABLE");
        
        // Add Iceberg-specific parameters
        let mut parameters = HashMap::new();
        parameters.insert("table_type".to_string(), "ICEBERG".to_string());
        if let Some(loc) = location {
            parameters.insert("metadata_location".to_string(), loc.to_string());
        }
        table_input = table_input.set_parameters(Some(parameters));
        
        // Create table
        let mut req = self.client.create_table()
            .database_name(namespace)
            .table_input(table_input.build().map_err(|e| anyhow!("Failed to build table input: {}", e))?);
        
        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }
        
        req.send().await
            .map_err(|e: aws_sdk_glue::error::SdkError<aws_sdk_glue::operation::create_table::CreateTableError>| anyhow::anyhow!("Failed to create Glue table: {}", e))?;
        
        Ok(())
    }
    
    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let mut req = self.client.get_table()
            .database_name(namespace)
            .name(table_name);
        
        if let Some(catalog_id) = &self.catalog_id {
            req = req.catalog_id(catalog_id);
        }
        
        let resp = req.send().await
            .map_err(|e: aws_sdk_glue::error::SdkError<aws_sdk_glue::operation::get_table::GetTableError>| anyhow::anyhow!("Failed to get Glue table: {}", e))?;
        
        let table = resp.table()
            .ok_or_else(|| anyhow!("Table not found in Glue response"))?;
        
        // Extract metadata_location from parameters
        let params = table.parameters()
            .ok_or_else(|| anyhow!("No parameters in Glue table"))?;
        
        let location = params.get("metadata_location")
            .ok_or_else(|| anyhow!("No metadata_location in Glue table parameters"))?
            .clone();
        
        Ok(TableMetadata::minimal(location))
    }
    
    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        // Glue doesn't support branches
        Err(anyhow!("AWS Glue does not support branching. Use Nessie for Git-like branching."))
    }
    
    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        match self.load_table(namespace, table_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn commit_table(&self, namespace: &str, table_name: &str, updates: Vec<serde_json::Value>) -> Result<()> {
        // For Glue, we need to update the table's metadata_location parameter
        // First, get the current table to preserve existing settings
        let mut get_req = self.client.get_table()
            .database_name(namespace)
            .name(table_name);
        
        if let Some(catalog_id) = &self.catalog_id {
            get_req = get_req.catalog_id(catalog_id);
        }
        
        let resp = get_req.send().await
            .map_err(|e| anyhow::anyhow!("Failed to get Glue table for commit: {}", e))?;
        
        let existing_table = resp.table()
            .ok_or_else(|| anyhow!("Table not found in Glue response"))?;
        
        // Extract the new metadata location from updates (look for set-snapshot-ref or similar)
        let mut new_metadata_location: Option<String> = None;
        for update in &updates {
            if let Some(action) = update.get("action").and_then(|v| v.as_str()) {
                if action == "set-current-snapshot" || action == "add-snapshot" {
                    // The snapshot update implies we need to update metadata location
                    // In a full implementation, we would compute the new path from snapshot
                    if let Some(snapshot) = update.get("snapshot") {
                        if let Some(manifest_list) = snapshot.get("manifest-list").and_then(|v| v.as_str()) {
                            // Derive metadata location from manifest list path
                            // e.g., s3://bucket/table/_manifest/snap-1.avro -> s3://bucket/table/metadata/vN.metadata.json
                            if let Some(table_root) = manifest_list.rsplit_once("/_manifest/") {
                                new_metadata_location = Some(format!("{}/metadata/v{}.metadata.json", 
                                    table_root.0,
                                    snapshot.get("snapshot-id").and_then(|v| v.as_i64()).unwrap_or(1)
                                ));
                            }
                        }
                    }
                }
            }
        }
        
        // Update parameters with new metadata location
        let mut parameters: HashMap<String, String> = existing_table.parameters().cloned()
            .unwrap_or_default();
        
        if let Some(new_loc) = new_metadata_location {
            parameters.insert("metadata_location".to_string(), new_loc);
        }
        
        // Preserve existing storage descriptor
        let storage_descriptor = existing_table.storage_descriptor()
            .cloned()
            .unwrap_or_else(|| StorageDescriptor::builder().build());
        
        // Build updated table input
        let table_input = TableInput::builder()
            .name(table_name)
            .storage_descriptor(storage_descriptor)
            .table_type(existing_table.table_type().unwrap_or("EXTERNAL_TABLE"))
            .set_parameters(Some(parameters))
            .build()
            .map_err(|e| anyhow!("Failed to build table input: {}", e))?;
        
        // Update table
        let mut update_req = self.client.update_table()
            .database_name(namespace)
            .table_input(table_input);
        
        if let Some(catalog_id) = &self.catalog_id {
            update_req = update_req.catalog_id(catalog_id);
        }
        
        update_req.send().await
            .map_err(|e| anyhow::anyhow!("Failed to update Glue table: {}", e))?;
        
        Ok(())
    }
}
