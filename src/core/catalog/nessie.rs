// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::common::{
    apply_table_updates, catalog_conflict_error, catalog_not_found_error,
    ensure_storage_table_initialized, extract_metadata_location_from_updates,
    load_table_metadata_from_location, persist_updates_to_storage,
    table_root_from_metadata_location, validate_requirements_against_metadata,
};
use super::{Catalog, TableMetadata};
use crate::core::error::BenoStreamError;
use arrow::datatypes::SchemaRef;

#[derive(Clone)]
pub struct NessieClient {
    base_url: String,
    token: Option<String>,
    client: Client,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Branch {
    #[serde(rename = "type")]
    pub ref_type: String, // "BRANCH" or "TAG"
    pub name: String,
    pub hash: Option<String>,
}

// Internal structures for Nessie API
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
    op_type: String, // "PUT" or "DELETE"
    key: Key,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Content>,
}

#[derive(Serialize, Deserialize)]
struct Key {
    elements: Vec<String>,
}

#[derive(Serialize)]
struct Content {
    #[serde(rename = "type")]
    content_type: String, // "ICEBERG_TABLE" or "ICEBERG_VIEW"
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
struct ContentResponse {
    content: Option<ContentDetails>,
}

#[derive(Deserialize)]
struct ContentDetails {
    #[serde(rename = "metadataLocation", alias = "metadata_location")]
    metadata_location: String,
    #[serde(default)]
    id: Option<String>,
}

#[derive(Deserialize)]
struct ReferenceResponse {
    reference: Branch,
}

#[derive(Deserialize)]
struct NessieEntriesResponse {
    #[serde(default)]
    entries: Vec<NessieEntry>,
}

#[derive(Deserialize)]
struct NessieEntry {
    #[serde(rename = "type", default)]
    content_type: String,
    name: Key,
}

fn nessie_http_error(status: reqwest::StatusCode, operation: &str, body: String) -> anyhow::Error {
    let lower = body.to_ascii_lowercase();
    let effective_status = if status == reqwest::StatusCode::CONFLICT
        || lower.contains("referenceconflictexception")
        || lower.contains("reference conflict")
    {
        409
    } else {
        status.as_u16()
    };
    anyhow::Error::new(BenoStreamError::CatalogRequest {
        catalog: "nessie".to_string(),
        status: Some(effective_status),
        reason: format!("Nessie {operation} failed ({status}): {body}"),
    })
}

#[async_trait]
impl Catalog for NessieClient {
    async fn create_namespace(&self, namespace: &str) -> Result<()> {
        if self.get_reference(namespace).await.is_ok() {
            return Ok(());
        }
        self.create_branch_internal(namespace, Some("main")).await
    }

    async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        let url = format!(
            "{}/api/v2/trees/{}",
            self.base_url,
            urlencoding::encode(namespace)
        );
        let resp = self.prepare(self.client.get(&url)).send().await?;
        if resp.status().is_success() {
            Ok(true)
        } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(false)
        } else {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            Err(nessie_http_error(status, "namespace_exists", err))
        }
    }

    async fn drop_namespace(&self, namespace: &str) -> Result<()> {
        if namespace == "main" {
            return Err(catalog_conflict_error(
                "nessie",
                "Cannot drop default 'main' branch",
            ));
        }
        let branch = self.get_reference(namespace).await?;
        let tables = self.list_tables(namespace).await?;
        if !tables.is_empty() {
            return Err(catalog_conflict_error(
                "nessie",
                format!("Nessie branch '{namespace}' is not empty"),
            ));
        }
        let ref_with_hash = match branch.hash.as_deref() {
            Some(h) if !h.is_empty() => format!("{}@{h}", branch.name),
            _ => branch.name,
        };
        let url = format!(
            "{}/api/v2/trees/{}",
            self.base_url,
            urlencoding::encode(&ref_with_hash)
        );
        let resp = self.prepare(self.client.delete(&url)).send().await?;
        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::NOT_FOUND {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(nessie_http_error(status, "drop_namespace", err));
        }
        Ok(())
    }

    async fn create_table(
        &self,
        namespace: &str,  // e.g. "main" (branch)
        table_name: &str, // e.g. "db.table" or "table"
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        if self.load_table(namespace, table_name).await.is_ok() {
            return Err(catalog_conflict_error(
                "nessie",
                format!("Table '{table_name}' already exists on Nessie ref '{namespace}'"),
            ));
        }

        let clean_root = location
            .map(|l| l.trim_end_matches('/').to_string())
            .filter(|l| !l.is_empty());
        let initial_meta_loc = if let Some(ref root) = clean_root {
            ensure_storage_table_initialized(root, &schema).await?
        } else {
            String::new()
        };

        self.create_table_internal(namespace, table_name, &initial_meta_loc, schema)
            .await
    }

    async fn drop_table(&self, namespace: &str, table_name: &str) -> Result<()> {
        let branch = match self.get_reference(namespace).await {
            Ok(b) => b,
            Err(_) => return Ok(()),
        };
        let hash = branch
            .hash
            .clone()
            .ok_or_else(|| anyhow!("Branch {} has no hash", namespace))?;

        let existing = self.fetch_content(namespace, &hash, table_name).await;
        if matches!(existing, Err(ref e) if e.downcast_ref::<BenoStreamError>().is_some_and(|b| matches!(b, BenoStreamError::CatalogRequest { status: Some(404), .. })))
        {
            return Ok(());
        }
        existing?;

        let elements: Vec<String> = table_name.split('.').map(|s| s.to_string()).collect();
        let op = Operation {
            op_type: "DELETE".to_string(),
            key: Key { elements },
            content: None,
        };

        self.commit_operation(
            namespace,
            &hash,
            branch.name,
            op,
            format!("Drop table {table_name}"),
        )
        .await
    }

    async fn list_tables(&self, namespace: &str) -> Result<Vec<String>> {
        let branch = self.get_reference(namespace).await?;
        let ref_with_hash = match branch.hash.as_deref() {
            Some(h) if !h.is_empty() => format!("{}@{h}", branch.name),
            _ => branch.name,
        };
        let url = format!(
            "{}/api/v2/trees/{}/entries",
            self.base_url,
            urlencoding::encode(&ref_with_hash)
        );
        let resp = self.prepare(self.client.get(&url)).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(nessie_http_error(status, "list_tables", err));
        }

        let parsed: NessieEntriesResponse = resp.json().await?;
        let mut names: Vec<String> = parsed
            .entries
            .into_iter()
            .filter(|e| e.content_type.is_empty() || e.content_type == "ICEBERG_TABLE")
            .map(|e| e.name.elements.join("."))
            .filter(|s| !s.is_empty())
            .collect();
        names.sort();
        Ok(names)
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let branch = self.get_reference(namespace).await?;
        let hash = branch.hash.as_deref().unwrap_or("");
        let details = self.fetch_content(namespace, hash, table_name).await?;
        load_table_metadata_from_location(&details.metadata_location).await
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

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_with_requirements(namespace, table_name, updates, Vec::new())
            .await
    }

    async fn commit_table_with_requirements(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        let branch = self.get_reference(namespace).await?;
        let hash = branch
            .hash
            .clone()
            .ok_or_else(|| anyhow!("Branch {} has no hash", namespace))?;

        let existing_details = self.fetch_content(namespace, &hash, table_name).await?;
        let existing_meta =
            load_table_metadata_from_location(&existing_details.metadata_location).await?;

        validate_requirements_against_metadata(&requirements, Some(&existing_meta), "nessie")?;

        let mut next_meta = existing_meta.clone();
        apply_table_updates(&mut next_meta, &updates)?;

        let table_root = table_root_from_metadata_location(&existing_details.metadata_location);
        let new_meta_loc = match extract_metadata_location_from_updates(&updates) {
            Some(loc) => loc,
            None if !table_root.is_empty() => {
                persist_updates_to_storage(&table_root, &updates).await?
            }
            None => existing_details.metadata_location.clone(),
        };

        let elements: Vec<String> = table_name.split('.').map(|s| s.to_string()).collect();

        let op = Operation {
            op_type: "PUT".to_string(),
            key: Key { elements },
            content: Some(Content {
                content_type: "ICEBERG_TABLE".to_string(),
                metadata_location: new_meta_loc,
                id: existing_details.id,
                snapshot_id: next_meta.current_snapshot_id.unwrap_or(-1),
                schema_id: next_meta.current_schema_id,
                spec_id: next_meta.default_spec_id,
                sort_order_id: next_meta.default_sort_order_id,
                sql_text: None,
                version_id: None,
                dialect: None,
            }),
        };

        self.commit_operation(
            namespace,
            &hash,
            branch.name,
            op,
            format!("Update table {table_name}"),
        )
        .await
    }
}

impl NessieClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            token: None,
            // Bounded timeout: an unresponsive catalog must not block forever.
            client: super::catalog_http_client(),
        }
    }

    pub fn with_token(base_url: String, token: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            token: token.filter(|t| !t.trim().is_empty()),
            // Bounded timeout: an unresponsive catalog must not block forever.
            client: super::catalog_http_client(),
        }
    }

    fn prepare(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref tok) = self.token {
            builder.bearer_auth(tok)
        } else {
            builder
        }
    }

    async fn fetch_content(
        &self,
        namespace: &str,
        hash: &str,
        table_name: &str,
    ) -> Result<ContentDetails> {
        let ref_with_hash = if hash.is_empty() {
            namespace.to_string()
        } else {
            format!("{namespace}@{hash}")
        };
        // Nessie API v2 addresses a single keyed content as
        // `/trees/{ref}/contents/{keyElements}` — plural, with the key as path
        // segments. The older `/content?key=…` form no longer exists and answers
        // 404 on every request, which made `load_table` fail after a successful
        // `create_table`.
        let key_path = table_name
            .split('.')
            .map(urlencoding::encode)
            .collect::<Vec<_>>()
            .join("/");
        let url = format!(
            "{}/api/v2/trees/{}/contents/{}",
            self.base_url,
            urlencoding::encode(&ref_with_hash),
            key_path
        );
        let resp = self.prepare(self.client.get(&url)).send().await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(catalog_not_found_error(
                "nessie",
                format!("Nessie table '{table_name}' not found on ref '{namespace}'"),
            ));
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(nessie_http_error(status, "load_table", err));
        }

        let content_resp: ContentResponse = resp.json().await?;
        content_resp.content.ok_or_else(|| {
            catalog_not_found_error(
                "nessie",
                format!("Nessie table '{table_name}' has no content on ref '{namespace}'"),
            )
        })
    }

    pub async fn create_branch_internal(&self, name: &str, source_ref: Option<&str>) -> Result<()> {
        let url = format!("{}/api/v2/trees", self.base_url);
        let query = [("name", name), ("type", "BRANCH")];

        let body = if let Some(source_name) = source_ref {
            let source = self.get_reference(source_name).await?;
            serde_json::to_value(&source)?
        } else {
            serde_json::json!({
                "type": "BRANCH",
                "name": "main"
            })
        };

        let resp = self
            .prepare(self.client.post(&url).query(&query).json(&body))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_text = resp.text().await.unwrap_or_default();
            return Err(nessie_http_error(status, "create_branch", error_text));
        }
        Ok(())
    }

    pub async fn get_reference(&self, name: &str) -> Result<Branch> {
        let url = format!(
            "{}/api/v2/trees/{}",
            self.base_url,
            urlencoding::encode(name)
        );
        let resp = self.prepare(self.client.get(&url)).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_text = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::NOT_FOUND {
                return Err(catalog_not_found_error(
                    "nessie",
                    format!("Reference '{name}' not found: {error_text}"),
                ));
            }
            return Err(nessie_http_error(status, "get_reference", error_text));
        }

        let wrapper: ReferenceResponse = resp.json().await?;
        Ok(wrapper.reference)
    }

    pub async fn create_table_internal(
        &self,
        branch_name: &str,
        table_name: &str,
        location: &str,
        _schema: SchemaRef,
    ) -> Result<()> {
        let branch = self.get_reference(branch_name).await?;
        let hash = branch
            .hash
            .ok_or_else(|| anyhow!("Branch {} has no hash", branch_name))?;

        let elements: Vec<String> = table_name.split('.').map(|s| s.to_string()).collect();

        let op = Operation {
            op_type: "PUT".to_string(),
            key: Key { elements },
            content: Some(Content {
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
            }),
        };

        self.commit_operation(
            branch_name,
            &hash,
            branch.name,
            op,
            format!("Create table {}", table_name),
        )
        .await
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
        let hash = branch
            .hash
            .ok_or_else(|| anyhow!("Branch {} has no hash", branch_name))?;

        let elements: Vec<String> = view_name.split('.').map(|s| s.to_string()).collect();

        let op = Operation {
            op_type: "PUT".to_string(),
            key: Key { elements },
            content: Some(Content {
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
            }),
        };

        self.commit_operation(
            branch_name,
            &hash,
            branch.name,
            op,
            format!("Create view {}", view_name),
        )
        .await
    }

    async fn commit_operation(
        &self,
        branch_name: &str,
        hash: &str,
        ref_name: String,
        op: Operation,
        message: String,
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
                author: "benostream".to_string(),
                properties: HashMap::new(),
            },
        };

        let ref_with_hash = format!("{branch_name}@{hash}");
        let url = format!(
            "{}/api/v2/trees/{}/history/commit",
            self.base_url,
            urlencoding::encode(&ref_with_hash)
        );

        let resp = self
            .prepare(self.client.post(&url).json(&commit))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_text = resp.text().await.unwrap_or_default();
            return Err(nessie_http_error(status, "commit", error_text));
        }
        Ok(())
    }
}
