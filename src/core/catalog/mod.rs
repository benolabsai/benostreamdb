// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

pub mod common;
pub mod config;
pub mod glue;
pub mod hive;
pub mod jdbc;
pub mod memory;
pub mod nessie;
pub mod policy;
pub mod rest;
pub mod unity;

use anyhow::Result;
use async_trait::async_trait;

use arrow::datatypes::SchemaRef;
pub use config::CatalogConfig;
pub use policy::{
    get_catalog_policy, lock_catalog_read_only, resolve_permissions_for_untrusted_connector,
    set_catalog_policy, set_catalog_schema_access, CatalogPolicy, SchemaPermissions,
    CATALOG_POLICY_FILE, IMMUTABLE_SYSTEM_SCHEMAS, SCRATCH_SCHEMA,
};

/// Abstract Catalog Interface (Iceberg Compatible)
///
/// This trait isolates the application from specific catalog implementations (Nessie, REST, Glue, etc.)
#[async_trait]
pub trait Catalog: Send + Sync {
    /// Create a namespace (PostgreSQL schema / Iceberg namespace) in the catalog.
    ///
    /// Catalogs that have no explicit namespace concept (e.g. Nessie, where a
    /// namespace is a branch) may leave the default no-op implementation.
    async fn create_namespace(&self, _namespace: &str) -> Result<()> {
        Ok(())
    }

    /// Check whether a namespace exists in the catalog.
    async fn namespace_exists(&self, _namespace: &str) -> Result<bool> {
        Ok(true)
    }

    /// Drop an empty namespace from the catalog.
    async fn drop_namespace(&self, _namespace: &str) -> Result<()> {
        Ok(())
    }

    /// Create a new table in the catalog
    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()>;

    /// Drop a table from the catalog.
    ///
    /// Catalogs that cannot drop tables may leave the default no-op.
    async fn drop_table(&self, _namespace: &str, _table_name: &str) -> Result<()> {
        Ok(())
    }

    /// List all table names in a namespace.
    ///
    /// Catalogs that do not implement listing return an empty list by default.
    async fn list_tables(&self, _namespace: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Load table metadata
    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata>;

    /// Create a new branch (Git-like semantics)
    async fn create_branch(&self, branch_name: &str, source_ref: Option<&str>) -> Result<()>;

    /// Check if a table exists
    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool>;

    /// Commit table updates (Iceberg atomic swap)
    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()>;

    /// Commit table updates with Iceberg REST `requirements` assertions (e.g.
    /// `assert-table-uuid`, `assert-ref-snapshot-id`). Default implementation
    /// delegates to [`Catalog::commit_table`] for catalogs without native
    /// requirement checking.
    async fn commit_table_with_requirements(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        _requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table(namespace, table_name, updates).await
    }
}

use crate::core::metadata::TableMetadata;

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Catalog type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CatalogType {
    Memory,
    Nessie,
    Rest,
    Glue,
    Hive,
    Unity,
    Jdbc,
    #[serde(alias = "s3_tables", alias = "s3-tables")]
    S3Tables,
}

impl FromStr for CatalogType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "memory" | "mem" => Ok(CatalogType::Memory),
            "nessie" => Ok(CatalogType::Nessie),
            "rest" => Ok(CatalogType::Rest),
            "glue" => Ok(CatalogType::Glue),
            "hive" => Ok(CatalogType::Hive),
            "unity" => Ok(CatalogType::Unity),
            "jdbc" => Ok(CatalogType::Jdbc),
            "s3tables" | "s3_tables" | "s3-tables" => Ok(CatalogType::S3Tables),
            _ => Err(anyhow::anyhow!("Unknown catalog type: {}", s)),
        }
    }
}

/// Extract an AWS region from an S3 Tables bucket ARN
/// (`arn:aws:s3tables:<region>:<account>:bucket/<name>`).
pub(crate) fn extract_s3tables_region_from_arn(arn: &str) -> Option<String> {
    let mut parts = arn.trim().split(':');
    let arn_prefix = parts.next()?;
    let _partition = parts.next()?;
    let service = parts.next()?;
    let region = parts.next()?;
    if arn_prefix == "arn" && service == "s3tables" && is_valid_aws_region(region) {
        Some(region.to_string())
    } else {
        None
    }
}

pub(crate) fn is_valid_aws_region(region: &str) -> bool {
    !region.is_empty()
        && region.len() <= 64
        && region
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

pub(crate) fn resolve_aws_region_from_config(
    config: &std::collections::HashMap<String, String>,
) -> Option<String> {
    config
        .get("region")
        .or_else(|| config.get("rest.signing-region"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            config
                .get("warehouse")
                .and_then(|w| extract_s3tables_region_from_arn(w))
        })
        .or_else(|| {
            std::env::var("AWS_REGION")
                .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
}

fn parse_bool_flag(config: &std::collections::HashMap<String, String>, key: &str) -> Option<bool> {
    config
        .get(key)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "true" | "1" | "yes"))
}

/// Create a catalog instance based on type and configuration
pub async fn create_catalog_async(
    catalog_type: CatalogType,
    config: std::collections::HashMap<String, String>,
) -> Result<Box<dyn Catalog>> {
    // SSRF guard (opt-in via `BSDB_SSRF_GUARD=1`): catalog endpoints are
    // user-supplied and fetched server-side. Reject internal/loopback hosts
    // before constructing any client.
    if crate::core::storage::ssrf_guard_enabled() {
        for key in ["url", "uri", "oauth2-server-uri", "oauth2_server_uri"] {
            if let Some(v) = config.get(key) {
                if v.contains("://") {
                    crate::core::storage::validate_external_uri(v)?;
                }
            }
        }
    }

    match catalog_type {
        CatalogType::Memory => {
            let warehouse = config.get("warehouse").cloned();
            if let Some(uri) = config
                .get("uri")
                .or_else(|| config.get("url"))
                .or_else(|| config.get("name"))
            {
                Ok(Box::new(memory::MemoryCatalogClient::for_uri(
                    uri, warehouse,
                )))
            } else if let Some(wh) = warehouse {
                Ok(Box::new(memory::MemoryCatalogClient::with_warehouse(wh)))
            } else {
                Ok(Box::new(memory::MemoryCatalogClient::new()))
            }
        }
        CatalogType::Nessie => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' config for Nessie catalog"))?;
            let token = config.get("token").cloned();
            Ok(Box::new(nessie::NessieClient::with_token(
                url.clone(),
                token,
            )))
        }
        CatalogType::Rest => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' or 'uri' config for REST catalog"))?;
            let prefix = config.get("prefix").cloned();
            let warehouse = config.get("warehouse").cloned();

            let sigv4_requested = config.get("auth").is_some_and(|a| {
                a.eq_ignore_ascii_case("sigv4") || a.eq_ignore_ascii_case("aws_sigv4")
            }) || parse_bool_flag(&config, "rest.sigv4-enabled")
                .unwrap_or(false);

            let auth = if sigv4_requested {
                let region = resolve_aws_region_from_config(&config).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Missing AWS 'region' (or 'rest.signing-region' / AWS_REGION) for SigV4 REST catalog"
                    )
                })?;
                if !is_valid_aws_region(&region) {
                    anyhow::bail!("Invalid AWS region '{region}' for SigV4 REST catalog");
                }
                let service = config
                    .get("service")
                    .or_else(|| config.get("rest.signing-name"))
                    .cloned()
                    .unwrap_or_else(|| "execute-api".to_string());
                Some(rest::RestCatalogAuth::AwsSigV4 { region, service })
            } else if let Some(token) = config.get("token").cloned() {
                Some(rest::RestCatalogAuth::BearerToken(token))
            } else if let Some(credential) = config.get("credential") {
                let parts: Vec<&str> = credential.splitn(2, ':').collect();
                if parts.len() == 2 {
                    let client_id = parts[0].to_string();
                    let client_secret = parts[1].to_string();
                    let oauth2_url = config
                        .get("oauth2-server-uri")
                        .or_else(|| config.get("oauth2_server_uri"))
                        .cloned();
                    let scope = config.get("scope").cloned();
                    let token_endpoint = oauth2_url.unwrap_or_else(|| {
                        format!("{}/v1/oauth/tokens", url.trim_end_matches('/'))
                    });
                    Some(rest::RestCatalogAuth::OAuth2 {
                        token_endpoint,
                        client_id,
                        client_secret,
                        scope,
                    })
                } else {
                    anyhow::bail!(
                        "Invalid 'credential' format. Expected <client_id>:<client_secret>"
                    );
                }
            } else if let (Some(client_id), Some(client_secret)) = (
                config.get("client_id").or_else(|| config.get("client-id")),
                config
                    .get("client_secret")
                    .or_else(|| config.get("client-secret")),
            ) {
                let oauth2_url = config
                    .get("oauth2-server-uri")
                    .or_else(|| config.get("oauth2_server_uri"))
                    .cloned();
                let scope = config.get("scope").cloned();
                let token_endpoint = oauth2_url
                    .unwrap_or_else(|| format!("{}/v1/oauth/tokens", url.trim_end_matches('/')));
                Some(rest::RestCatalogAuth::OAuth2 {
                    token_endpoint,
                    client_id: client_id.clone(),
                    client_secret: client_secret.clone(),
                    scope,
                })
            } else {
                None
            };

            let discover = parse_bool_flag(&config, "discover_config")
                .unwrap_or_else(|| warehouse.is_some() && prefix.is_none());

            Ok(Box::new(
                rest::RestCatalogClient::with_auth(url.clone(), prefix, auth)
                    .with_warehouse(warehouse, discover),
            ))
        }
        CatalogType::S3Tables => {
            let warehouse = config
                .get("warehouse")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let explicit_url = config.get("url").or_else(|| config.get("uri")).cloned();
            if warehouse.is_none() && explicit_url.is_none() {
                anyhow::bail!(
                    "Missing 'warehouse' (S3 Tables bucket ARN) or 'url' config for S3Tables catalog"
                );
            }

            let region = resolve_aws_region_from_config(&config).ok_or_else(|| {
                anyhow::anyhow!(
                    "Missing 'region' (or regional 'warehouse' ARN / AWS_REGION) for S3Tables catalog"
                )
            })?;
            if !is_valid_aws_region(&region) {
                anyhow::bail!("Invalid AWS region '{region}' for S3Tables catalog");
            }

            let url = explicit_url
                .unwrap_or_else(|| format!("https://s3tables.{region}.amazonaws.com/iceberg"));
            if crate::core::storage::ssrf_guard_enabled() && url.contains("://") {
                crate::core::storage::validate_external_uri(&url)?;
            }

            let prefix = config.get("prefix").cloned();
            let discover = parse_bool_flag(&config, "discover_config")
                .unwrap_or_else(|| warehouse.is_some() && prefix.is_none());

            Ok(Box::new(
                rest::RestCatalogClient::with_sigv4(url, prefix, region, "s3tables")
                    .with_warehouse(warehouse, discover),
            ))
        }
        CatalogType::Glue => {
            let mode_rest = config
                .get("mode")
                .is_some_and(|m| m.eq_ignore_ascii_case("rest"));
            if mode_rest {
                let region = resolve_aws_region_from_config(&config).ok_or_else(|| {
                    anyhow::anyhow!("Missing 'region' (or AWS_REGION) for Glue Iceberg REST mode")
                })?;
                if !is_valid_aws_region(&region) {
                    anyhow::bail!("Invalid AWS region '{region}' for Glue Iceberg REST mode");
                }
                let url = config
                    .get("url")
                    .or_else(|| config.get("uri"))
                    .cloned()
                    .unwrap_or_else(|| format!("https://glue.{region}.amazonaws.com/iceberg"));
                if crate::core::storage::ssrf_guard_enabled() && url.contains("://") {
                    crate::core::storage::validate_external_uri(&url)?;
                }
                let prefix = config.get("prefix").cloned();
                let warehouse = config
                    .get("warehouse")
                    .or_else(|| config.get("catalog_id"))
                    .cloned();
                let discover = parse_bool_flag(&config, "discover_config")
                    .unwrap_or_else(|| warehouse.is_some() && prefix.is_none());

                Ok(Box::new(
                    rest::RestCatalogClient::with_sigv4(url, prefix, region, "glue")
                        .with_warehouse(warehouse, discover),
                ))
            } else {
                let catalog_id = config.get("catalog_id").cloned();
                let region = config.get("region").cloned();
                let client = glue::GlueCatalogClient::with_region(catalog_id, region).await?;
                Ok(Box::new(client))
            }
        }
        CatalogType::Hive => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' config for Hive Metastore"))?;
            Ok(Box::new(hive::HiveMetastoreClient::new(url.clone())?))
        }
        CatalogType::Unity => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' config for Unity Catalog"))?;
            let token = config
                .get("token")
                .ok_or_else(|| anyhow::anyhow!("Missing 'token' config for Unity Catalog"))?;
            let warehouse = config
                .get("warehouse")
                .or_else(|| config.get("catalog"))
                .cloned();
            let prefix = config.get("prefix").cloned();
            let trimmed_url = url.trim_end_matches('/');
            let iceberg_rest_mode = config.get("mode").is_some_and(|m| {
                m.eq_ignore_ascii_case("iceberg-rest") || m.eq_ignore_ascii_case("rest")
            }) || trimmed_url.ends_with("/iceberg-rest")
                || trimmed_url.ends_with("/iceberg");

            if iceberg_rest_mode {
                Ok(Box::new(unity::UnityCatalogClient::new_iceberg_rest(
                    url.clone(),
                    token.clone(),
                    warehouse,
                    prefix,
                )))
            } else {
                let mut client = unity::UnityCatalogClient::new(url.clone(), token.clone());
                if let Some(cat) = warehouse {
                    client = client.with_default_catalog(cat);
                }
                Ok(Box::new(client))
            }
        }
        CatalogType::Jdbc => {
            let uri = config
                .get("uri")
                .or_else(|| config.get("url"))
                .ok_or_else(|| anyhow::anyhow!("Missing 'uri' config for JDBC catalog"))?;
            let warehouse = config.get("warehouse").cloned();
            let catalog_name = config
                .get("catalog_name")
                .unwrap_or(&"default".to_string())
                .clone();
            let client = jdbc::JdbcCatalogClient::new(uri.clone(), warehouse, catalog_name).await?;
            Ok(Box::new(client))
        }
    }
}

/// Synchronous catalog factory (for non-async contexts)
/// Note: Glue requires async initialization, so this will block
pub fn create_catalog(
    catalog_type: CatalogType,
    config: std::collections::HashMap<String, String>,
) -> Result<Box<dyn Catalog>> {
    tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("Failed to create tokio runtime for catalog: {}", e))?
        .block_on(create_catalog_async(catalog_type, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_s3tables_arn_region_extraction_and_validation() {
        assert_eq!(
            extract_s3tables_region_from_arn(
                "arn:aws:s3tables:us-east-1:123456789012:bucket/analytics"
            ),
            Some("us-east-1".to_string())
        );
        assert_eq!(
            extract_s3tables_region_from_arn(
                "arn:aws-us-gov:s3tables:us-gov-west-1:123456789012:bucket/gov"
            ),
            Some("us-gov-west-1".to_string())
        );
        assert_eq!(
            extract_s3tables_region_from_arn("arn:aws:s3:::plain-bucket"),
            None
        );
        assert_eq!(
            extract_s3tables_region_from_arn(
                "arn:aws:s3tables:us-east-1.evil.com/path:123:bucket/bad"
            ),
            None
        );
        assert!(is_valid_aws_region("eu-central-1"));
        assert!(!is_valid_aws_region("us-east-1/evil"));
    }

    #[test]
    fn test_catalog_type_s3tables_aliases() -> Result<()> {
        assert_eq!(CatalogType::from_str("s3tables")?, CatalogType::S3Tables);
        assert_eq!(CatalogType::from_str("s3_tables")?, CatalogType::S3Tables);
        assert_eq!(CatalogType::from_str("s3-tables")?, CatalogType::S3Tables);
        Ok(())
    }

    #[tokio::test]
    async fn test_create_catalog_async_phase_b_modes_and_ssrf() -> Result<()> {
        let mut s3_cfg = HashMap::new();
        s3_cfg.insert(
            "warehouse".to_string(),
            "arn:aws:s3tables:us-west-2:123456789012:bucket/prod".to_string(),
        );
        assert!(create_catalog_async(CatalogType::S3Tables, s3_cfg)
            .await
            .is_ok());

        let mut s3_ssrf = HashMap::new();
        s3_ssrf.insert(
            "warehouse".to_string(),
            "arn:aws:s3tables:us-west-2:123456789012:bucket/prod".to_string(),
        );
        s3_ssrf.insert("url".to_string(), "http://127.0.0.1:9999".to_string());
        assert!(create_catalog_async(CatalogType::S3Tables, s3_ssrf)
            .await
            .is_err());

        let mut glue_rest = HashMap::new();
        glue_rest.insert("mode".to_string(), "rest".to_string());
        glue_rest.insert("region".to_string(), "eu-central-1".to_string());
        glue_rest.insert("catalog_id".to_string(), "123456789012".to_string());
        assert!(create_catalog_async(CatalogType::Glue, glue_rest)
            .await
            .is_ok());

        let mut unity_rest = HashMap::new();
        unity_rest.insert(
            "url".to_string(),
            "https://dbc.example.com/api/2.1/unity-catalog/iceberg-rest".to_string(),
        );
        unity_rest.insert("token".to_string(), "pat-123".to_string());
        unity_rest.insert("warehouse".to_string(), "main".to_string());
        assert!(create_catalog_async(CatalogType::Unity, unity_rest)
            .await
            .is_ok());

        let mut rest_sigv4 = HashMap::new();
        rest_sigv4.insert("url".to_string(), "https://api.example.com".to_string());
        rest_sigv4.insert("auth".to_string(), "sigv4".to_string());
        rest_sigv4.insert("region".to_string(), "us-east-1".to_string());
        rest_sigv4.insert("service".to_string(), "s3tables".to_string());
        assert!(create_catalog_async(CatalogType::Rest, rest_sigv4)
            .await
            .is_ok());

        Ok(())
    }
}
