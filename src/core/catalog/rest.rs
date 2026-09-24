// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::{Catalog, TableMetadata};
use arrow::datatypes::SchemaRef;

/// Authentication method for REST Catalog
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestCatalogAuth {
    /// Static Bearer Token
    BearerToken(String),
    /// OAuth2 Client Credentials flow
    OAuth2 {
        token_endpoint: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    },
}

#[derive(Clone, Debug)]
struct CachedToken {
    token: String,
    expires_at: Instant,
}

/// REST Catalog client implementing Iceberg REST Catalog specification
#[derive(Clone)]
pub struct RestCatalogClient {
    base_url: String,
    client: Client,
    prefix: String, // Optional prefix (e.g., "warehouse")
    auth: Option<RestCatalogAuth>,
    token_cache: Arc<RwLock<Option<CachedToken>>>,
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

#[derive(Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

impl RestCatalogClient {
    /// Create a new unauthenticated REST Catalog client
    pub fn new(base_url: String, prefix: Option<String>) -> Self {
        Self::with_auth(base_url, prefix, None)
    }

    /// Create a new REST Catalog client with authentication
    pub fn with_auth(
        base_url: String,
        prefix: Option<String>,
        auth: Option<RestCatalogAuth>,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
            prefix: prefix.unwrap_or_default(),
            auth,
            token_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a client with a pre-configured static Bearer token
    pub fn with_bearer_token(base_url: String, prefix: Option<String>, token: String) -> Self {
        Self::with_auth(base_url, prefix, Some(RestCatalogAuth::BearerToken(token)))
    }

    /// Create a client configured for OAuth2 client credentials grant (Polaris / Lakekeeper)
    pub fn with_oauth2(
        base_url: String,
        prefix: Option<String>,
        token_endpoint: Option<String>,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    ) -> Self {
        let endpoint = token_endpoint
            .unwrap_or_else(|| format!("{}/v1/oauth/tokens", base_url.trim_end_matches('/')));
        Self::with_auth(
            base_url,
            prefix,
            Some(RestCatalogAuth::OAuth2 {
                token_endpoint: endpoint,
                client_id,
                client_secret,
                scope,
            }),
        )
    }

    /// Get current active Bearer token, fetching or refreshing via OAuth2 if required
    pub async fn get_token(&self) -> Result<Option<String>> {
        let auth = match &self.auth {
            Some(a) => a,
            None => return Ok(None),
        };

        match auth {
            RestCatalogAuth::BearerToken(token) => Ok(Some(token.clone())),
            RestCatalogAuth::OAuth2 {
                token_endpoint,
                client_id,
                client_secret,
                scope,
            } => {
                // Fast path: check valid cached token
                {
                    let cache = self.token_cache.read().await;
                    if let Some(ref cached) = *cache {
                        if Instant::now() < cached.expires_at {
                            return Ok(Some(cached.token.clone()));
                        }
                    }
                }

                // Slow path: acquire write lock and refresh
                let mut write_cache = self.token_cache.write().await;
                if let Some(ref cached) = *write_cache {
                    if Instant::now() < cached.expires_at {
                        return Ok(Some(cached.token.clone()));
                    }
                }

                let mut params = vec![
                    ("grant_type", "client_credentials"),
                    ("client_id", client_id.as_str()),
                    ("client_secret", client_secret.as_str()),
                ];
                if let Some(ref s) = scope {
                    params.push(("scope", s.as_str()));
                }

                let resp = self
                    .client
                    .post(token_endpoint)
                    .form(&params)
                    .send()
                    .await
                    .map_err(|e| anyhow!("OAuth2 token request failed: {}", e))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let err = resp.text().await.unwrap_or_default();
                    return Err(anyhow!(
                        "OAuth2 token request returned {} - {}",
                        status,
                        err
                    ));
                }

                let token_resp: OAuthTokenResponse = resp
                    .json()
                    .await
                    .map_err(|e| anyhow!("Failed to parse OAuth2 token response: {}", e))?;

                let ttl_secs = token_resp.expires_in.unwrap_or(3600);
                // Refresh 60 seconds before actual expiration
                let safety_margin = std::cmp::min(60, ttl_secs / 2);
                let valid_duration = Duration::from_secs(ttl_secs.saturating_sub(safety_margin));
                let expires_at = Instant::now() + valid_duration;

                let token = token_resp.access_token;
                *write_cache = Some(CachedToken {
                    token: token.clone(),
                    expires_at,
                });

                Ok(Some(token))
            }
        }
    }

    /// Decorate request builder with Authorization header if configured
    async fn prepare_request(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder> {
        if let Some(token) = self.get_token().await? {
            Ok(builder.bearer_auth(token))
        } else {
            Ok(builder)
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

        // Convert schema to JSON for REST API
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

        let builder = self
            .prepare_request(self.client.post(&url).json(&req))
            .await?;
        let resp = builder.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to create table ({}): {}", status, error));
        }

        Ok(())
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let url = self.build_url(&format!("/namespaces/{}/tables/{}", namespace, table_name));

        let builder = self.prepare_request(self.client.get(&url)).await?;
        let resp = builder.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!("Failed to load table ({}): {}", status, error));
        }

        let table_resp: LoadTableResponse = resp.json().await?;

        // Note: metadata.location remains the table root, while table_resp.metadata_location is the file URI.
        Ok(table_resp.metadata)
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        let url = self.build_url(&format!("/namespaces/{}/tables/{}", namespace, table_name));

        let req = UpdateTableRequest {
            updates,
            requirements: None,
        };

        let builder = self
            .prepare_request(self.client.post(&url).json(&req))
            .await?;
        let resp = builder.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await?;
            return Err(anyhow!(
                "Failed to commit table update ({}): {}",
                status,
                error
            ));
        }

        Ok(())
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        // REST Catalog doesn't support branches (Nessie-specific feature)
        Err(anyhow!(
            "REST Catalog does not support branching. Use Nessie for Git-like branching."
        ))
    }

    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        match self.load_table(namespace, table_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rest_catalog_bearer_token() -> Result<()> {
        let client = RestCatalogClient::with_bearer_token(
            "http://localhost:8181".to_string(),
            None,
            "static-secret-token".to_string(),
        );

        let token = client.get_token().await?;
        assert_eq!(token, Some("static-secret-token".to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn test_rest_catalog_token_caching() -> Result<()> {
        let client = RestCatalogClient::with_oauth2(
            "http://localhost:8181".to_string(),
            Some("polaris".to_string()),
            None,
            "principal_id".to_string(),
            "principal_secret".to_string(),
            Some("PRINCIPAL_ROLE:ALL".to_string()),
        );

        // Pre-populate valid cache
        {
            let mut cache = client.token_cache.write().await;
            *cache = Some(CachedToken {
                token: "cached-polaris-token".to_string(),
                expires_at: Instant::now() + Duration::from_secs(300),
            });
        }

        let token = client.get_token().await?;
        assert_eq!(token, Some("cached-polaris-token".to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn test_create_catalog_async_oauth2() -> Result<()> {
        let mut config = HashMap::new();
        config.insert("url".to_string(), "http://localhost:8181".to_string());
        config.insert("prefix".to_string(), "polaris_warehouse".to_string());
        config.insert(
            "credential".to_string(),
            "polaris_client:polaris_secret".to_string(),
        );
        config.insert("scope".to_string(), "PRINCIPAL_ROLE:ALL".to_string());

        let catalog = crate::core::catalog::create_catalog_async(
            crate::core::catalog::CatalogType::Rest,
            config,
        )
        .await?;

        assert!(!catalog.table_exists("ns", "table").await.unwrap_or(true));

        Ok(())
    }
}
