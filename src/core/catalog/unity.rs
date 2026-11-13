// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::common::{
    catalog_conflict_error, catalog_not_found_error, ensure_storage_table_initialized,
    extract_metadata_location_from_updates, load_table_metadata_from_location,
    persist_updates_to_storage, table_root_from_metadata_location,
    validate_requirements_against_metadata,
};
use super::rest::RestCatalogClient;
use super::{Catalog, TableMetadata};
use crate::core::error::BenoStreamError;
use arrow::datatypes::SchemaRef;

/// Unity Catalog client (Databricks).
///
/// Supports both:
/// - **Standard Unity Catalog Tables REST** (`/api/2.1/unity-catalog/...`)
/// - **Databricks Unity Iceberg REST** (`/api/2.1/unity-catalog/iceberg-rest`),
///   which speaks full Apache Iceberg REST `/v1` semantics (`requirements`, `updates`,
///   and atomic `409 Conflict` retries).
#[derive(Clone)]
pub struct UnityCatalogClient {
    base_url: String,
    token: String,
    client: Client,
    iceberg_rest: Option<RestCatalogClient>,
    default_catalog: String,
}

// Request structures
#[derive(Serialize)]
struct CreateTableRequest {
    name: String,
    catalog_name: String,
    schema_name: String,
    table_type: String,         // "EXTERNAL"
    data_source_format: String, // "DELTA" or "ICEBERG"
    storage_location: String,
    columns: Vec<ColumnInfo>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    properties: HashMap<String, String>,
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
    #[serde(default)]
    storage_location: String,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Deserialize)]
struct ListUnityTablesResponse {
    #[serde(default)]
    tables: Vec<TableResponse>,
    #[serde(default)]
    next_page_token: Option<String>,
}

fn unity_http_error(status: reqwest::StatusCode, operation: &str, body: String) -> anyhow::Error {
    anyhow::Error::new(BenoStreamError::CatalogRequest {
        catalog: "unity".to_string(),
        status: Some(status.as_u16()),
        reason: format!("Failed Unity {operation} ({status}): {body}"),
    })
}

fn parse_unity_namespace<'a>(namespace: &'a str, default_catalog: &'a str) -> (&'a str, &'a str) {
    if let Some((catalog, schema)) = namespace.split_once('.') {
        (catalog, schema)
    } else {
        (default_catalog, namespace)
    }
}

fn arrow_type_to_unity_type_name(dt: &arrow::datatypes::DataType) -> &'static str {
    match dt {
        arrow::datatypes::DataType::Boolean => "BOOLEAN",
        arrow::datatypes::DataType::Int8 | arrow::datatypes::DataType::UInt8 => "BYTE",
        arrow::datatypes::DataType::Int16 | arrow::datatypes::DataType::UInt16 => "SHORT",
        arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::UInt32 => "INT",
        arrow::datatypes::DataType::Int64 | arrow::datatypes::DataType::UInt64 => "LONG",
        arrow::datatypes::DataType::Float32 => "FLOAT",
        arrow::datatypes::DataType::Float64 => "DOUBLE",
        arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8 => "STRING",
        arrow::datatypes::DataType::Binary
        | arrow::datatypes::DataType::LargeBinary
        | arrow::datatypes::DataType::FixedSizeBinary(_) => "BINARY",
        arrow::datatypes::DataType::Date32 | arrow::datatypes::DataType::Date64 => "DATE",
        arrow::datatypes::DataType::Timestamp(_, _) => "TIMESTAMP",
        _ => "STRING",
    }
}

impl UnityCatalogClient {
    /// Create a standard Unity Catalog client (`/api/2.1/unity-catalog/tables/...`).
    pub fn new(base_url: String, token: String) -> Self {
        let trimmed = base_url.trim_end_matches('/').to_string();
        if trimmed.ends_with("/iceberg-rest") || trimmed.ends_with("/iceberg") {
            return Self::new_iceberg_rest(trimmed, token, None, None);
        }
        Self {
            base_url: trimmed,
            token,
            // Bounded timeout: an unresponsive catalog must not block forever.
            client: super::catalog_http_client(),
            iceberg_rest: None,
            default_catalog: "main".to_string(),
        }
    }

    /// Create a Unity Catalog client configured for Unity Iceberg REST.
    ///
    /// The Iceberg REST mount point is not standardized across Unity deployments:
    ///
    /// | deployment | Iceberg REST base |
    /// |---|---|
    /// | open-source `unitycatalog/unitycatalog` | `{root}/api/2.1/unity-catalog/iceberg` |
    /// | Databricks Unity | `{root}/api/2.1/unity-catalog/iceberg-rest` |
    ///
    /// Both are registered as probe candidates (open-source first) and the first one that
    /// answers `GET /v1/config` is cached by [`RestCatalogClient`], so callers do not need to
    /// know which flavour they are talking to. An explicit `base_url` that already ends in
    /// `/iceberg` or `/iceberg-rest` is probed first, preserving custom mount points.
    ///
    /// `warehouse` defaults to `main` and is always sent on `/v1/config`; the open-source
    /// server rejects a bare `/v1/config` with `400 Must supply a proper catalog in
    /// warehouse property`.
    pub fn new_iceberg_rest(
        base_url: String,
        token: String,
        warehouse: Option<String>,
        prefix: Option<String>,
    ) -> Self {
        let trimmed = base_url.trim_end_matches('/').to_string();
        let explicit_rest = trimmed.ends_with("/iceberg-rest") || trimmed.ends_with("/iceberg");
        let root = if let Some(stripped) = trimmed.strip_suffix("/iceberg-rest") {
            stripped.to_string()
        } else if let Some(stripped) = trimmed.strip_suffix("/iceberg") {
            stripped.to_string()
        } else {
            trimmed.clone()
        };

        let mut candidates: Vec<String> = Vec::with_capacity(3);
        if explicit_rest {
            candidates.push(trimmed.clone());
        }
        candidates.push(format!("{root}/api/2.1/unity-catalog/iceberg"));
        candidates.push(format!("{root}/api/2.1/unity-catalog/iceberg-rest"));

        let effective_warehouse = warehouse
            .as_deref()
            .map(str::trim)
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .or_else(|| {
                prefix
                    .as_deref()
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "main".to_string());
        let discover = prefix.is_none();

        let rest_client =
            RestCatalogClient::with_bearer_token(candidates[0].clone(), prefix, token.clone())
                .with_warehouse(Some(effective_warehouse.clone()), discover)
                .with_base_url_candidates(candidates);

        Self {
            base_url: root,
            token,
            // Bounded timeout: an unresponsive catalog must not block forever.
            client: super::catalog_http_client(),
            iceberg_rest: Some(rest_client),
            default_catalog: effective_warehouse,
        }
    }

    /// The Iceberg REST base URL actually in use, once `/v1/config` autodetection has run.
    ///
    /// Returns `None` for a standard (non-Iceberg-REST) Unity client.
    pub async fn resolved_iceberg_rest_base(&self) -> Option<String> {
        match &self.iceberg_rest {
            Some(rest) => Some(rest.base_url().await),
            None => None,
        }
    }

    /// Override the default catalog name used when unqualified namespaces are passed.
    pub fn with_default_catalog(mut self, catalog: impl Into<String>) -> Self {
        self.default_catalog = catalog.into();
        self
    }

    async fn fetch_unity_table(&self, full_name: &str) -> Result<TableResponse> {
        let url = format!(
            "{}/api/2.1/unity-catalog/tables/{}",
            self.base_url,
            urlencoding::encode(full_name)
        );

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::NOT_FOUND {
                return Err(catalog_not_found_error(
                    "unity",
                    format!("Unity table '{full_name}' not found: {error}"),
                ));
            }
            return Err(unity_http_error(status, "load_table", error));
        }

        Ok(resp.json().await?)
    }

    fn resolve_rest_namespace<'a>(&'a self, namespace: &'a str) -> &'a str {
        if let Some((_cat, schema)) = namespace.split_once('.') {
            schema
        } else {
            namespace
        }
    }
}

#[async_trait]
impl Catalog for UnityCatalogClient {
    async fn create_namespace(&self, namespace: &str) -> Result<()> {
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .create_namespace(self.resolve_rest_namespace(namespace))
                .await;
        }

        let (catalog, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);
        let req = serde_json::json!({
            "name": schema_name,
            "catalog_name": catalog,
        });
        let url = format!("{}/api/2.1/unity-catalog/schemas", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.token)
            .json(&req)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            return Err(unity_http_error(status, "create_namespace", error));
        }
        Ok(())
    }

    async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .namespace_exists(self.resolve_rest_namespace(namespace))
                .await;
        }

        let (catalog, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);
        let url = format!(
            "{}/api/2.1/unity-catalog/schemas/{}",
            self.base_url,
            urlencoding::encode(&format!("{catalog}.{schema_name}"))
        );
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(true)
        } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(false)
        } else {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            Err(unity_http_error(status, "namespace_exists", error))
        }
    }

    async fn drop_namespace(&self, namespace: &str) -> Result<()> {
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .drop_namespace(self.resolve_rest_namespace(namespace))
                .await;
        }

        let tables = self.list_tables(namespace).await?;
        if !tables.is_empty() {
            return Err(catalog_conflict_error(
                "unity",
                format!("Unity schema '{namespace}' is not empty"),
            ));
        }

        let (catalog, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);
        let url = format!(
            "{}/api/2.1/unity-catalog/schemas/{}",
            self.base_url,
            urlencoding::encode(&format!("{catalog}.{schema_name}"))
        );
        let resp = self
            .client
            .delete(&url)
            .bearer_auth(&self.token)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            return Err(unity_http_error(status, "drop_namespace", error));
        }
        Ok(())
    }

    async fn create_table(
        &self,
        namespace: &str, // catalog.schema or schema
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .create_table(
                    self.resolve_rest_namespace(namespace),
                    table_name,
                    schema,
                    location,
                )
                .await;
        }

        let clean_root = location.map(|loc| loc.trim_end_matches('/').to_string());
        let initial_metadata_location = if let Some(ref root) = clean_root {
            Some(ensure_storage_table_initialized(root, &schema).await?)
        } else {
            None
        };

        let (catalog, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);

        let columns: Vec<ColumnInfo> = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| ColumnInfo {
                name: f.name().to_string(),
                type_text: f.data_type().to_string(),
                type_name: arrow_type_to_unity_type_name(f.data_type()).to_string(),
                position: i as i32,
            })
            .collect();

        let mut properties = HashMap::new();
        properties.insert("table_type".to_string(), "ICEBERG".to_string());
        if let Some(meta_loc) = initial_metadata_location {
            properties.insert("metadata_location".to_string(), meta_loc);
        }

        let req = CreateTableRequest {
            name: table_name.to_string(),
            catalog_name: catalog.to_string(),
            schema_name: schema_name.to_string(),
            table_type: "EXTERNAL".to_string(),
            data_source_format: "ICEBERG".to_string(),
            storage_location: clean_root.unwrap_or_default(),
            columns,
            properties,
        };

        let url = format!("{}/api/2.1/unity-catalog/tables", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.token)
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            return Err(unity_http_error(status, "create_table", error));
        }

        Ok(())
    }

    async fn drop_table(&self, namespace: &str, table_name: &str) -> Result<()> {
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .drop_table(self.resolve_rest_namespace(namespace), table_name)
                .await;
        }

        let (catalog, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);
        let full_name = format!("{catalog}.{schema_name}.{table_name}");
        let url = format!(
            "{}/api/2.1/unity-catalog/tables/{}",
            self.base_url,
            urlencoding::encode(&full_name)
        );
        let resp = self
            .client
            .delete(&url)
            .bearer_auth(&self.token)
            .send()
            .await?;
        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::NOT_FOUND {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            return Err(unity_http_error(status, "drop_table", error));
        }
        Ok(())
    }

    async fn list_tables(&self, namespace: &str) -> Result<Vec<String>> {
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .list_tables(self.resolve_rest_namespace(namespace))
                .await;
        }

        let (catalog, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);
        let mut names = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut url = format!(
                "{}/api/2.1/unity-catalog/tables?catalog_name={}&schema_name={}",
                self.base_url,
                urlencoding::encode(catalog),
                urlencoding::encode(schema_name)
            );
            if let Some(ref tok) = page_token {
                url.push_str("&page_token=");
                url.push_str(&urlencoding::encode(tok));
            }
            let resp = self
                .client
                .get(&url)
                .bearer_auth(&self.token)
                .send()
                .await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let error = resp.text().await.unwrap_or_default();
                return Err(unity_http_error(status, "list_tables", error));
            }
            let parsed: ListUnityTablesResponse = resp.json().await?;
            names.extend(parsed.tables.into_iter().map(|t| t.name));
            match parsed.next_page_token.filter(|t| !t.is_empty()) {
                Some(tok) => page_token = Some(tok),
                None => break,
            }
        }

        names.sort();
        Ok(names)
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .load_table(self.resolve_rest_namespace(namespace), table_name)
                .await;
        }

        let (catalog, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);
        let full_name = format!("{catalog}.{schema_name}.{table_name}");
        let table = self.fetch_unity_table(&full_name).await?;

        let meta_loc = table
            .properties
            .get("metadata_location")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or(table.storage_location);

        load_table_metadata_from_location(&meta_loc).await
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Err(anyhow!(
            "Unity Catalog does not support branching. Use Nessie for Git-like branching."
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
        if let Some(ref rest) = self.iceberg_rest {
            return rest
                .commit_table_with_requirements(
                    self.resolve_rest_namespace(namespace),
                    table_name,
                    updates,
                    requirements,
                )
                .await;
        }

        let (catalog_name, schema_name) = parse_unity_namespace(namespace, &self.default_catalog);
        let full_name = format!("{catalog_name}.{schema_name}.{table_name}");
        let existing_table = self.fetch_unity_table(&full_name).await?;

        let existing_meta_loc = existing_table
            .properties
            .get("metadata_location")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| existing_table.storage_location.clone());

        let table_root = if !existing_table.storage_location.is_empty() {
            table_root_from_metadata_location(&existing_table.storage_location)
        } else {
            table_root_from_metadata_location(&existing_meta_loc)
        };

        let existing_meta = if existing_meta_loc.is_empty() {
            None
        } else {
            Some(load_table_metadata_from_location(&existing_meta_loc).await?)
        };

        validate_requirements_against_metadata(&requirements, existing_meta.as_ref(), "unity")?;

        let new_meta_loc = match extract_metadata_location_from_updates(&updates) {
            Some(loc) => loc,
            None if !table_root.is_empty() => {
                persist_updates_to_storage(&table_root, &updates).await?
            }
            None => {
                return Err(anyhow!(
                    "Could not determine metadata location or table root for Unity table '{full_name}'"
                ));
            }
        };

        let mut next_props = existing_table.properties;
        next_props
            .entry("table_type".to_string())
            .or_insert_with(|| "ICEBERG".to_string());
        if !existing_meta_loc.is_empty() && existing_meta_loc != new_meta_loc {
            next_props.insert(
                "previous_metadata_location".to_string(),
                existing_meta_loc.clone(),
            );
        }
        next_props.insert("metadata_location".to_string(), new_meta_loc);

        let url = format!(
            "{}/api/2.1/unity-catalog/tables/{}",
            self.base_url,
            urlencoding::encode(&full_name)
        );

        let patch_body = serde_json::json!({
            "storage_location": table_root,
            "properties": next_props,
        });

        let resp = self
            .client
            .patch(&url)
            .bearer_auth(&self.token)
            .json(&patch_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            return Err(unity_http_error(status, "commit_table", error));
        }

        Ok(())
    }
}
