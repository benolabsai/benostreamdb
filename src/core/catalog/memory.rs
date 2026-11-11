// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{anyhow, Result};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::common::{
    apply_table_updates, catalog_conflict_error, catalog_not_found_error,
    ensure_storage_table_initialized, extract_metadata_location_from_updates,
    validate_requirements_against_metadata,
};
use super::{Catalog, TableMetadata};

#[derive(Debug, Clone)]
struct MemoryTableEntry {
    location: String,
    metadata_location: String,
    metadata: TableMetadata,
}

#[derive(Debug, Default)]
struct MemoryCatalogState {
    namespaces: HashSet<String>,
    tables: HashMap<(String, String), MemoryTableEntry>,
}

/// Process-wide named in-memory catalogs so separate client handles referencing
/// the same `memory://<name>` URI observe each other's atomic commits.
static MEMORY_CATALOG_REGISTRY: Lazy<RwLock<HashMap<String, Arc<RwLock<MemoryCatalogState>>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// In-process Iceberg Catalog backed by `RwLock`-synchronized state.
///
/// Mirrors [`apache/iceberg-rust`'s `iceberg-catalog-memory`](https://github.com/apache/iceberg-rust)
/// for deterministic zero-infrastructure unit tests and concurrent multi-writer
/// Iceberg `requirements` (HTTP-409 equivalent) convergence tests.
#[derive(Clone)]
pub struct MemoryCatalogClient {
    state: Arc<RwLock<MemoryCatalogState>>,
    warehouse: String,
}

impl Default for MemoryCatalogClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryCatalogClient {
    /// Create an isolated in-memory catalog with an isolated `memory://` warehouse.
    pub fn new() -> Self {
        Self::with_warehouse(format!("memory://bsdb-mem-{}", uuid::Uuid::new_v4()))
    }

    /// Create an isolated in-memory catalog storing table data under `warehouse`.
    pub fn with_warehouse(warehouse: String) -> Self {
        Self {
            state: Arc::new(RwLock::new(MemoryCatalogState::default())),
            warehouse: warehouse.trim_end_matches('/').to_string(),
        }
    }

    /// Open or create a shared named in-memory catalog keyed by `name_or_uri`.
    pub fn for_uri(name_or_uri: &str, warehouse: Option<String>) -> Self {
        let key = name_or_uri.trim().to_string();
        let state = {
            let mut reg = MEMORY_CATALOG_REGISTRY.write();
            reg.entry(key)
                .or_insert_with(|| Arc::new(RwLock::new(MemoryCatalogState::default())))
                .clone()
        };
        let wh = warehouse.unwrap_or_else(|| {
            if name_or_uri.contains("://") {
                name_or_uri.trim_end_matches('/').to_string()
            } else {
                format!("memory://bsdb-mem-{}", name_or_uri.trim_matches('/'))
            }
        });
        Self {
            state,
            warehouse: wh,
        }
    }

    /// Return the current committed `metadata-location` pointer for diagnostics and tests.
    pub fn metadata_location(&self, namespace: &str, table_name: &str) -> Option<String> {
        let state = self.state.read();
        state
            .tables
            .get(&(namespace.to_string(), table_name.to_string()))
            .map(|e| e.metadata_location.clone())
    }

    async fn commit_table_internal(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        let key = (namespace.to_string(), table_name.to_string());
        let mut guard = self.state.write();

        let existing = guard.tables.get(&key);
        let existing_meta = existing.map(|e| &e.metadata);

        validate_requirements_against_metadata(&requirements, existing_meta, "memory")?;

        let entry = guard.tables.get_mut(&key).ok_or_else(|| {
            catalog_not_found_error(
                "memory",
                format!(
                    "Table '{}.{}' not found in memory catalog",
                    namespace, table_name
                ),
            )
        })?;

        let mut next_meta = entry.metadata.clone();
        apply_table_updates(&mut next_meta, &updates)?;

        if let Some(new_loc) = extract_metadata_location_from_updates(&updates) {
            entry.metadata_location = new_loc;
        }
        if next_meta.location.is_empty() {
            next_meta.location = entry.location.clone();
        }
        entry.metadata = next_meta;

        Ok(())
    }
}

#[async_trait]
impl Catalog for MemoryCatalogClient {
    async fn create_namespace(&self, namespace: &str) -> Result<()> {
        let mut guard = self.state.write();
        guard.namespaces.insert(namespace.to_string());
        Ok(())
    }

    async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        let guard = self.state.read();
        Ok(guard.namespaces.contains(namespace))
    }

    async fn drop_namespace(&self, namespace: &str) -> Result<()> {
        let mut guard = self.state.write();
        let has_tables = guard.tables.keys().any(|(ns, _)| ns == namespace);
        if has_tables {
            return Err(catalog_conflict_error(
                "memory",
                format!("Namespace '{namespace}' is not empty"),
            ));
        }
        guard.namespaces.remove(namespace);
        Ok(())
    }

    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        let key = (namespace.to_string(), table_name.to_string());
        {
            let guard = self.state.read();
            if guard.tables.contains_key(&key) {
                return Err(catalog_conflict_error(
                    "memory",
                    format!("Table '{}.{}' already exists", namespace, table_name),
                ));
            }
        }

        let table_location = location
            .map(|s| s.trim_end_matches('/').to_string())
            .unwrap_or_else(|| format!("{}/{}/{}", self.warehouse, namespace, table_name));

        let metadata_location = ensure_storage_table_initialized(&table_location, &schema).await?;
        let mut metadata =
            super::common::load_table_metadata_from_location(&metadata_location).await?;
        if metadata.location.is_empty() {
            metadata.location = table_location.clone();
        }

        let mut guard = self.state.write();
        if guard.tables.contains_key(&key) {
            return Err(catalog_conflict_error(
                "memory",
                format!("Table '{}.{}' already exists", namespace, table_name),
            ));
        }
        guard.namespaces.insert(namespace.to_string());
        guard.tables.insert(
            key,
            MemoryTableEntry {
                location: table_location,
                metadata_location,
                metadata,
            },
        );
        Ok(())
    }

    async fn drop_table(&self, namespace: &str, table_name: &str) -> Result<()> {
        let mut guard = self.state.write();
        guard
            .tables
            .remove(&(namespace.to_string(), table_name.to_string()));
        Ok(())
    }

    async fn list_tables(&self, namespace: &str) -> Result<Vec<String>> {
        let guard = self.state.read();
        let mut names: Vec<String> = guard
            .tables
            .keys()
            .filter(|(ns, _)| ns == namespace)
            .map(|(_, tbl)| tbl.clone())
            .collect();
        names.sort();
        Ok(names)
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let guard = self.state.read();
        let entry = guard
            .tables
            .get(&(namespace.to_string(), table_name.to_string()))
            .ok_or_else(|| {
                catalog_not_found_error(
                    "memory",
                    format!(
                        "Table '{}.{}' not found in memory catalog",
                        namespace, table_name
                    ),
                )
            })?;
        Ok(entry.metadata.clone())
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Err(anyhow!(
            "Memory Catalog does not support branching. Use Nessie for Git-like branching."
        ))
    }

    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        let guard = self.state.read();
        Ok(guard
            .tables
            .contains_key(&(namespace.to_string(), table_name.to_string())))
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, Vec::new())
            .await
    }

    async fn commit_table_with_requirements(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, requirements)
            .await
    }
}
