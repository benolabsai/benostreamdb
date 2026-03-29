// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod nessie;
pub mod rest;
pub mod glue;
pub mod hive;
pub mod unity;
pub mod config;
pub mod jdbc;

use anyhow::Result;
use async_trait::async_trait;

pub use config::CatalogConfig;
use arrow::datatypes::SchemaRef;

/// Abstract Catalog Interface (Iceberg Compatible)
/// 
/// This trait isolates the application from specific catalog implementations (Nessie, REST, Glue, etc.)
#[async_trait]
pub trait Catalog: Send + Sync {
    /// Create a new table in the catalog
    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()>;

    /// Load table metadata
    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata>;
    
    /// Create a new branch (Git-like semantics)
    async fn create_branch(&self, branch_name: &str, source_ref: Option<&str>) -> Result<()>;
    
    /// Check if a table exists
    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool>;

    /// Commit table updates (Iceberg atomic swap)
    async fn commit_table(&self, namespace: &str, table_name: &str, updates: Vec<serde_json::Value>) -> Result<()>;
}

use crate::core::metadata::TableMetadata;

use std::str::FromStr;
use serde::{Deserialize, Serialize};

/// Catalog type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CatalogType {
    Nessie,
    Rest,
    Glue,
    Hive,
    Unity,
    Jdbc,
}

impl FromStr for CatalogType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nessie" => Ok(CatalogType::Nessie),
            "rest" => Ok(CatalogType::Rest),
            "glue" => Ok(CatalogType::Glue),
            "hive" => Ok(CatalogType::Hive),
            "unity" => Ok(CatalogType::Unity),
            "jdbc" => Ok(CatalogType::Jdbc),
            _ => Err(anyhow::anyhow!("Unknown catalog type: {}", s)),
        }
    }
}

/// Create a catalog instance based on type and configuration
pub async fn create_catalog_async(
    catalog_type: CatalogType,
    config: std::collections::HashMap<String, String>,
) -> Result<Box<dyn Catalog>> {
    match catalog_type {
        CatalogType::Nessie => {
            let url = config.get("url")
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' config for Nessie catalog"))?;
            Ok(Box::new(nessie::NessieClient::new(url.clone())))
        }
        CatalogType::Rest => {
            let url = config.get("url")
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' config for REST catalog"))?;
            let prefix = config.get("prefix").cloned();
            Ok(Box::new(rest::RestCatalogClient::new(url.clone(), prefix)))
        }
        CatalogType::Glue => {
            let catalog_id = config.get("catalog_id").cloned();
            let client = glue::GlueCatalogClient::new(catalog_id).await?;
            Ok(Box::new(client))
        }
        CatalogType::Hive => {
            let url = config.get("url")
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' config for Hive Metastore"))?;
            Ok(Box::new(hive::HiveMetastoreClient::new(url.clone())?))
        }
        CatalogType::Unity => {
            let url = config.get("url")
                .ok_or_else(|| anyhow::anyhow!("Missing 'url' config for Unity Catalog"))?;
            let token = config.get("token")
                .ok_or_else(|| anyhow::anyhow!("Missing 'token' config for Unity Catalog"))?;
            Ok(Box::new(unity::UnityCatalogClient::new(url.clone(), token.clone())))
        }
        CatalogType::Jdbc => {
            let uri = config.get("uri")
                .ok_or_else(|| anyhow::anyhow!("Missing 'uri' config for JDBC catalog"))?;
            let warehouse = config.get("warehouse").cloned();
            let catalog_name = config.get("catalog_name").unwrap_or(&"default".to_string()).clone();
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
        .unwrap()
        .block_on(create_catalog_async(catalog_type, config))
}
