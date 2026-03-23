use serde::{Deserialize, Serialize};
use reqwest::Client;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use async_trait::async_trait;

use super::{Catalog, TableMetadata};
use arrow::datatypes::SchemaRef;

#[derive(Clone)]
pub struct NessieClient {
    base_url: String,
    client: Client,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Branch {
    #[serde(rename = "type")]
    pub ref_type: String, // "BRANCH" or "TAG"
    pub name: String,
    pub hash: Option<String>,
}


// Internal structures for Nessie API
#[derive(Serialize)]
#[allow(dead_code)]
struct CreateReferenceRequest {
    #[serde(rename = "type")]
    ref_type: String, // "BRANCH"
    name: String,
    #[serde(rename = "sourceRefName")]
    source_ref_name: Option<String>,
}

#[derive(Serialize)]
struct CommitRequest {
    #[serde(rename = "branch")]
    branch: BranchRef,
    operations: Vec<Operation>,
    #[serde(rename = "commitMeta")]
    meta: CommitMeta,
}

#[derive(Serialize)]
struct BranchRef {
    #[serde(rename = "type")]
    ref_type: String, // "BRANCH"
    name: String,
    hash: String,
}

#[derive(Serialize)]
struct CommitMeta {
    message: String,
    author: String,
    properties: HashMap<String, String>,
}

#[derive(Serialize)]
struct Operation {
    #[serde(rename = "type")]
    op_type: String, // "PUT"
    key: Key,
    content: Content,
}

#[derive(Serialize)]
struct Key {
    elements: Vec<String>,
}

#[derive(Serialize)]
struct Content {
    #[serde(rename = "type")]
    content_type: String, // "ICEBERG_TABLE"
    #[serde(rename = "metadataLocation")]
    metadata_location: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "snapshotId")]
    snapshot_id: i64,
    #[serde(rename = "schemaId")]
    schema_id: i32,
    #[serde(rename = "specId")]
    spec_id: i32,
    #[serde(rename = "sortOrderId")]
    sort_order_id: i32,
    #[serde(rename = "sqlText", skip_serializing_if = "Option::is_none")]
    sql_text: Option<String>,
    #[serde(rename = "versionId", skip_serializing_if = "Option::is_none")]
    version_id: Option<i32>,
    #[serde(rename = "dialect", skip_serializing_if = "Option::is_none")]
    dialect: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ContentResponse {
    content: Option<ContentDetails>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ContentDetails {
    metadata_location: String,
}

#[derive(Deserialize)]
struct ReferenceResponse {
    reference: Branch,
}


#[async_trait]
impl Catalog for NessieClient {
    async fn create_table(
        &self,
        namespace: &str, // e.g. "main" (branch)
        table_name: &str, // e.g. "db.table"
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        let loc = location.unwrap_or(""); 
        
        // 1. Initialize table manifest if location is provided
        if !loc.is_empty() {
             crate::Table::create_async(loc.to_string(), schema.clone()).await?;
        }

        // 2. Register in Nessie
        self.create_table_internal(namespace, table_name, loc, schema).await
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let branch = self.get_reference(namespace).await?;
        let _elements: Vec<String> = table_name.split('.').map(|s| s.to_string()).collect();
        
        // Construct ContentKey
        let url = format!("{}/api/v2/trees/{}@{}/content", self.base_url, namespace, branch.hash.as_deref().unwrap_or(""));
        let query = [
             ("key", table_name), 
        ];

        // This is a simplification; Nessie's content API is more complex
        // For strictly verifying existence or fetching metadata location:
        let resp = self.client.get(&url).query(&query).send().await?;
         if !resp.status().is_success() {
             return Err(anyhow!("Table not found or error loading"));
        }
        
        let content_resp: ContentResponse = resp.json().await?;
        if let Some(details) = content_resp.content {
             Ok(TableMetadata::minimal(details.metadata_location))
        } else {
             Err(anyhow!("Table content not found"))
        }
    }
    
    async fn create_branch(&self, branch_name: &str, source_ref: Option<&str>) -> Result<()> {
        self.create_branch_internal(branch_name, source_ref).await
    }
    
    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        match self.load_table(namespace, table_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn commit_table(&self, namespace: &str, table_name: &str, updates: Vec<serde_json::Value>) -> Result<()> {
        // For Nessie, we create a commit with an updated ICEBERG_TABLE content
        let branch = self.get_reference(namespace).await?;
        let hash = branch.hash.clone().ok_or_else(|| anyhow!("Branch {} has no hash", namespace))?;
        
        // Extract metadata from updates
        let mut metadata_location = String::new();
        let mut snapshot_id: i64 = -1;
        let mut schema_id: i32 = 0;
        
        for update in &updates {
            if let Some(action) = update.get("action").and_then(|v| v.as_str()) {
                if action == "add-snapshot" {
                    if let Some(snapshot) = update.get("snapshot") {
                        if let Some(manifest_list) = snapshot.get("manifest-list").and_then(|v| v.as_str()) {
                            if let Some(table_root) = manifest_list.rsplit_once("/_manifest/") {
                                let snap_id = snapshot.get("snapshot-id").and_then(|v| v.as_i64()).unwrap_or(1);
                                metadata_location = format!("{}/metadata/v{}.metadata.json", table_root.0, snap_id);
                                snapshot_id = snap_id;
                                schema_id = snapshot.get("schema-id").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                            }
                        }
                    }
                }
            }
        }
        
        if metadata_location.is_empty() {
            return Err(anyhow!("Could not determine new metadata location from updates"));
        }
        
        let elements: Vec<String> = table_name.split('.').map(|s| s.to_string()).collect();
        
        let op = Operation {
            op_type: "PUT".to_string(),
            key: Key { elements },
            content: Content {
                content_type: "ICEBERG_TABLE".to_string(),
                metadata_location,
                id: None,
                snapshot_id,
                schema_id,
                spec_id: 0,
                sort_order_id: 0,
                sql_text: None,
                version_id: None,
                dialect: None,
            }
        };
        
        self.commit_operation(namespace, &hash, branch.name, op, format!("Update table {}", table_name)).await
    }
}

impl NessieClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    pub async fn create_branch_internal(&self, name: &str, source_ref: Option<&str>) -> Result<()> {
        let url = format!("{}/api/v2/trees", self.base_url);
        
        let query = [
            ("name", name),
            ("type", "BRANCH"),
        ];

        let body = if let Some(source_name) = source_ref {
            let source = self.get_reference(source_name).await?;
            serde_json::to_value(&source)?
        } else {
             serde_json::json!({
                 "type": "BRANCH",
                 "name": "main" 
             })
        };

        let resp = self.client.post(&url)
            .query(&query)
            .json(&body)
            .send()
            .await?;
            
        if !resp.status().is_success() {
             let error_text = resp.text().await?;
             return Err(anyhow!("Failed to create branch: {}", error_text));
        }
        Ok(())
    }

    pub async fn get_reference(&self, name: &str) -> Result<Branch> {
        let url = format!("{}/api/v2/trees/{}", self.base_url, name);
        let resp = self.client.get(&url).send().await?;
        
        if !resp.status().is_success() {
             let error_text = resp.text().await?;
             return Err(anyhow!("Failed to get reference '{}': {}", name, error_text));
        }

        let wrapper: ReferenceResponse = resp.json().await?;
        Ok(wrapper.reference)
    }

    pub async fn create_table_internal(
        &self, 
        branch_name: &str, 
        table_name: &str, 
        location: &str,
        _schema: SchemaRef
    ) -> Result<()> {
        let branch = self.get_reference(branch_name).await?;
        let hash = branch.hash.ok_or_else(|| anyhow!("Branch {} has no hash", branch_name))?;

        let elements: Vec<String> = table_name.split('.').map(|s| s.to_string()).collect();

        let op = Operation {
            op_type: "PUT".to_string(),
            key: Key { elements },
            content: Content {
                content_type: "ICEBERG_TABLE".to_string(),
                metadata_location: location.to_string(),
                id: None,
                snapshot_id: -1,
                schema_id: 0,
                spec_id: 0,
                sort_order_id: 0,
                sql_text: None,
                version_id: None,
                dialect: None,
            }
        };

        self.commit_operation(branch_name, &hash, branch.name, op, format!("Create table {}", table_name)).await
    }

    pub async fn create_view(
        &self,
        branch_name: &str,
        view_name: &str,
        metadata_location: &str,
        sql_text: &str,
        dialect: &str,
    ) -> Result<()> {
        let branch = self.get_reference(branch_name).await?;
        let hash = branch.hash.ok_or_else(|| anyhow!("Branch {} has no hash", branch_name))?;

        let elements: Vec<String> = view_name.split('.').map(|s| s.to_string()).collect();

        let op = Operation {
            op_type: "PUT".to_string(),
            key: Key { elements },
            content: Content {
                content_type: "ICEBERG_VIEW".to_string(),
                metadata_location: metadata_location.to_string(),
                id: None,
                snapshot_id: -1, 
                schema_id: 0,
                spec_id: 0,
                sort_order_id: 0,
                sql_text: Some(sql_text.to_string()),
                version_id: Some(1),
                dialect: Some(dialect.to_string()),
            }
        };

        self.commit_operation(branch_name, &hash, branch.name, op, format!("Create view {}", view_name)).await
    }

    async fn commit_operation(
        &self,
        branch_name: &str,
        hash: &str,
        ref_name: String,
        op: Operation,
        message: String
    ) -> Result<()> {
        let commit = CommitRequest {
            branch: BranchRef {
                ref_type: "BRANCH".to_string(),
                name: ref_name,
                hash: hash.to_string(),
            },
            operations: vec![op],
            meta: CommitMeta {
                message,
                author: "hyperstream".to_string(),
                properties: HashMap::new(),
            },
        };

        let url = format!("{}/api/v2/trees/{}@{}/history/commit", self.base_url, branch_name, hash);
        
        let resp = self.client.post(&url).json(&commit).send().await?;

        if !resp.status().is_success() {
             let error_text = resp.text().await?;
             return Err(anyhow!("Failed to commit operation: {}", error_text));
        }
        Ok(())
    }
}
