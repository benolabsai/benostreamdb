// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{Result, anyhow, Context};
use async_trait::async_trait;
use sqlx::{AnyPool, any::AnyPoolOptions, Row};
use arrow::datatypes::SchemaRef;
use super::{Catalog, TableMetadata};

/// JDBC Implementation of Iceberg Catalog
pub struct JdbcCatalogClient {
    pool: AnyPool,
    warehouse: String,
    catalog_name: String,
}

impl JdbcCatalogClient {
    pub async fn new(uri: String, warehouse: Option<String>, catalog_name: String) -> Result<Self> {
        // sqlx-any uses the scheme to determine the driver
        let pool = AnyPoolOptions::new()
            .max_connections(5)
            .connect(&uri)
            .await
            .context(format!("Failed to connect to JDBC catalog database at {}", uri))?;
        
        let client = Self {
            pool,
            warehouse: warehouse.unwrap_or_else(|| "/tmp/hyperstream_warehouse".to_string()),
            catalog_name,
        };
        
        client.setup().await?;
        
        Ok(client)
    }

    async fn setup(&self) -> Result<()> {
        // Create tables if they don't exist (using standard SQL)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS iceberg_tables (
                catalog_name VARCHAR(255) NOT NULL,
                table_namespace VARCHAR(255) NOT NULL,
                table_name VARCHAR(255) NOT NULL,
                metadata_location VARCHAR(255),
                previous_metadata_location VARCHAR(255),
                PRIMARY KEY (catalog_name, table_namespace, table_name)
            )"
        ).execute(&self.pool).await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS namespaces (
                catalog_name VARCHAR(255) NOT NULL,
                namespace VARCHAR(255) NOT NULL,
                PRIMARY KEY (catalog_name, namespace)
            )"
        ).execute(&self.pool).await?;

        Ok(())
    }
}

#[async_trait]
impl Catalog for JdbcCatalogClient {
    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        // 1. Determine location if not provided
        let table_location = location.map(|s| s.to_string()).unwrap_or_else(|| {
             format!("{}/{}/{}", self.warehouse, namespace, table_name)
        });
        
        // 2. Initialize table manifest in storage
        crate::Table::create_async(table_location.clone(), schema.clone()).await?;
        
        // Initial metadata location
        let metadata_location = format!("{}/manifest.json", table_location); 

        // 3. Register in DB
        sqlx::query(
            "INSERT INTO iceberg_tables (catalog_name, table_namespace, table_name, metadata_location) 
             VALUES (?, ?, ?, ?)"
        )
        .bind(&self.catalog_name)
        .bind(namespace)
        .bind(table_name)
        .bind(&metadata_location)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let row = sqlx::query(
            "SELECT metadata_location FROM iceberg_tables 
             WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
        )
        .bind(&self.catalog_name)
        .bind(namespace)
        .bind(table_name)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| anyhow!("Table {}.{} not found in JDBC catalog: {}", namespace, table_name, e))?;

        let metadata_location: String = row.get(0);
        
        // Extract base location from metadata location (manifest.json)
        let location = if metadata_location.contains("/manifest.json") {
             metadata_location.replace("/manifest.json", "")
        } else {
             metadata_location
        };

        Ok(TableMetadata::minimal(location))
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Err(anyhow!("JDBC Catalog does not support branching."))
    }

    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
         let row = sqlx::query(
            "SELECT COUNT(*) FROM iceberg_tables 
             WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
        )
        .bind(&self.catalog_name)
        .bind(namespace)
        .bind(table_name)
        .fetch_one(&self.pool)
        .await?;
        
        // Handle potential differences in count return type (i32/i64)
        let count: i64 = match row.try_get::<i64, _>(0) {
            Ok(c) => c,
            Err(_) => row.get::<i32, _>(0) as i64,
        };
        Ok(count > 0)
    }

    async fn commit_table(&self, namespace: &str, table_name: &str, updates: Vec<serde_json::Value>) -> Result<()> {
        // Extract new metadata location from updates
        let mut new_metadata_location: Option<String> = None;
        for update in &updates {
            if let Some(action) = update.get("action").and_then(|v| v.as_str()) {
                if action == "add-snapshot" {
                    if let Some(snapshot) = update.get("snapshot") {
                        if let Some(manifest_list) = snapshot.get("manifest-list").and_then(|v| v.as_str()) {
                            if let Some(table_root) = manifest_list.rsplit_once("/_manifest/") {
                                let snap_id = snapshot.get("snapshot-id").and_then(|v| v.as_i64()).unwrap_or(1);
                                new_metadata_location = Some(format!("{}/metadata/v{}.metadata.json", table_root.0, snap_id));
                            }
                        }
                    }
                }
            }
        }
        
        let new_loc = new_metadata_location.ok_or_else(|| anyhow!("No new metadata location in updates"))?;
        
        // Atomic UPDATE with previous_metadata_location for history
        sqlx::query(
            "UPDATE iceberg_tables 
             SET previous_metadata_location = metadata_location, 
                 metadata_location = ? 
             WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
        )
        .bind(&new_loc)
        .bind(&self.catalog_name)
        .bind(namespace)
        .bind(table_name)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
}
