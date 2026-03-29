// Copyright (c) 2026 Richard Albright. All rights reserved.

use serde::{Deserialize, Serialize};
use reqwest::Client;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::collections::HashMap;

use super::{Catalog, TableMetadata};
use arrow::datatypes::SchemaRef;

/// REST Catalog client implementing Iceberg REST Catalog specification
#[derive(Clone)]
pub struct RestCatalogClient {
    base_url: String,
    client: Client,
    prefix: String,  // Optional prefix (e.g., "warehouse")
}

// Request structures
#[derive(Serialize)]
struct CreateTableRequest {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none", rename = "partition-spec")]
    partition_spec: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "write-order")]
    write_order: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<HashMap<String, String>>,
}

// Response structures
#[derive(Deserialize)]
struct LoadTableResponse {
    #[allow(dead_code)]
    #[serde(rename = "metadata-location")]
    metadata_location: String,
    metadata: TableMetadata,
}

#[derive(Serialize)]
struct UpdateTableRequest {
    updates: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requirements: Option<Vec<serde_json::Value>>,
}

impl RestCatalogClient {
    /// Create a new REST Catalog client
    pub fn new(base_url: String, prefix: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
            prefix: prefix.unwrap_or_default(),
        }
    }
    
    /// Build full URL with optional prefix
    fn build_url(&self, path: &str) -> String {
        if self.prefix.is_empty() {
            format!("{}/v1{}", self.base_url, path)
        } else {
            format!("{}/v1/{}{}", self.base_url, self.prefix, path)
        }
    }
}

#[async_trait]
impl Catalog for RestCatalogClient {
    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        let url = self.build_url(&format!("/namespaces/{}/tables", namespace));
        

        // 2. Convert schema to JSON for REST API
        // For now, we use our internal manifest schema format as a proxy for Iceberg schema
        let manifest_schema = crate::core::manifest::Schema::from_arrow(&schema, 1);
        let schema_json = serde_json::to_value(&manifest_schema)?;
        
        let req = CreateTableRequest {
            name: table_name.to_string(),
            location: location.map(|s| s.to_string()),
            schema: schema_json,
            partition_spec: None,
            write_order: None,
            properties: None,
        };
        
        let resp = self.client.post(&url)
            .json(&req)
            .send()
            .await?;
        
        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to create table ({}): {}", status, error));
        }
        
        Ok(())
    }
    
    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let url = self.build_url(&format!("/namespaces/{}/tables/{}", namespace, table_name));
        
        let resp = self.client.get(&url).send().await?;
        
        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to load table ({}): {}", status, error));
        }
        
        let table_resp: LoadTableResponse = resp.json().await?;
        
        // Note: metadata.location remains the table root, while table_resp.metadata_location is the file URI.
        
        Ok(table_resp.metadata)
    }

    async fn commit_table(&self, namespace: &str, table_name: &str, updates: Vec<serde_json::Value>) -> Result<()> {
        let url = self.build_url(&format!("/namespaces/{}/tables/{}", namespace, table_name));
        
        let req = UpdateTableRequest {
            updates,
            requirements: None,
        };

        let resp = self.client.post(&url)
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to commit table update ({}): {}", status, error));
        }

        Ok(())
    }
    
    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        // REST Catalog doesn't support branches (Nessie-specific feature)
        Err(anyhow!("REST Catalog does not support branching. Use Nessie for Git-like branching."))
    }
    
    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        match self.load_table(namespace, table_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}
