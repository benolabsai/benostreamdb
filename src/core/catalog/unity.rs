// Copyright (c) 2026 Richard Albright. All rights reserved.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
// use std::collections::HashMap; // Unused

use super::{Catalog, TableMetadata};
use arrow::datatypes::SchemaRef;

/// Unity Catalog client (Databricks)
#[derive(Clone)]
pub struct UnityCatalogClient {
    base_url: String,
    token: String,
    client: Client,
}

// Request structures
#[derive(Serialize)]
struct CreateTableRequest {
    name: String,
    catalog_name: String,
    schema_name: String,
    table_type: String,  // "EXTERNAL"
    data_source_format: String,  // "DELTA" or "ICEBERG"
    storage_location: String,
    columns: Vec<ColumnInfo>,
}

#[derive(Serialize)]
struct ColumnInfo {
    name: String,
    type_text: String,
    type_name: String,
    position: i32,
}

// Response structures
#[derive(Deserialize)]
#[allow(dead_code)]
struct TableResponse {
    name: String,
    catalog_name: String,
    schema_name: String,
    table_type: String,
    data_source_format: String,
    storage_location: String,
}

impl UnityCatalogClient {
    /// Create a new Unity Catalog client
    /// 
    /// # Arguments
    /// * `base_url` - Databricks workspace URL (e.g., "https://my-workspace.cloud.databricks.com")
    /// * `token` - Databricks personal access token
    pub fn new(base_url: String, token: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            token,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl Catalog for UnityCatalogClient {
    async fn create_table(
        &self,
        namespace: &str,  // catalog.schema
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        // 1. Initialize table manifest if location is provided
        if let Some(loc) = location {
             crate::Table::create_async(loc.to_string(), schema.clone()).await?;
        }

        // Parse namespace into catalog.schema
        let parts: Vec<&str> = namespace.split('.').collect();
        let (catalog, schema_name) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            ("main", namespace)
        };

        // 2. Extract columns from Arrow schema for Unity
        let columns: Vec<ColumnInfo> = schema.fields().iter().enumerate().map(|(i, f)| {
            ColumnInfo {
                name: f.name().to_string(),
                type_text: f.data_type().to_string(),
                type_name: f.data_type().to_string().to_uppercase(),
                position: i as i32,
            }
        }).collect();

        let req = CreateTableRequest {
            name: table_name.to_string(),
            catalog_name: catalog.to_string(),
            schema_name: schema_name.to_string(),
            table_type: "EXTERNAL".to_string(),
            data_source_format: "ICEBERG".to_string(),
            storage_location: location.unwrap_or("").to_string(),
            columns,
        };

        let url = format!("{}/api/2.1/unity-catalog/tables", self.base_url);
        let resp = self.client.post(&url)
            .bearer_auth(&self.token)
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to create Unity table ({}): {}", status, error));
        }

        Ok(())
    }
    
    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        // Parse namespace
        let parts: Vec<&str> = namespace.split('.').collect();
        let (catalog, schema_name) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            ("main", namespace)
        };

        let full_name = format!("{}.{}.{}", catalog, schema_name, table_name);
        let url = format!("{}/api/2.1/unity-catalog/tables/{}", self.base_url, full_name);
        
        let resp = self.client.get(&url)
            .bearer_auth(&self.token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to load Unity table ({}): {}", status, error));
        }

        let table: TableResponse = resp.json().await?;
        
        Ok(TableMetadata::minimal(table.storage_location))
    }
    
    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        // Unity Catalog doesn't support branches (yet)
        Err(anyhow!("Unity Catalog does not support branching. Use Nessie for Git-like branching."))
    }
    
    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        match self.load_table(namespace, table_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn commit_table(&self, namespace: &str, table_name: &str, updates: Vec<serde_json::Value>) -> Result<()> {
        // Unity Catalog uses PATCH to update table metadata
        // Extract catalog.schema from namespace
        let parts: Vec<&str> = namespace.split('.').collect();
        let (catalog_name, schema_name) = if parts.len() >= 2 {
            (parts[0], parts[1])
        } else {
            return Err(anyhow!("Invalid namespace format for Unity: expected 'catalog.schema'"));
        };
        
        // Extract new metadata location from updates
        let mut new_location: Option<String> = None;
        for update in &updates {
            if let Some(action) = update.get("action").and_then(|v| v.as_str()) {
                if action == "add-snapshot" {
                    if let Some(snapshot) = update.get("snapshot") {
                        if let Some(manifest_list) = snapshot.get("manifest-list").and_then(|v| v.as_str()) {
                            if let Some(table_root) = manifest_list.rsplit_once("/_manifest/") {
                                new_location = Some(table_root.0.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        let storage_location = new_location.ok_or_else(|| anyhow!("No new location in updates"))?;
        
        let url = format!("{}/api/2.1/unity-catalog/tables/{}.{}.{}", 
            self.base_url, catalog_name, schema_name, table_name);
        
        let resp = self.client.patch(&url)
            .bearer_auth(&self.token)
            .json(&serde_json::json!({
                "storage_location": storage_location
            }))
            .send()
            .await?;
        
        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to update Unity table ({}): {}", status, error));
        }
        
        Ok(())
    }
}
