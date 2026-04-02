// Copyright (c) 2026 Richard Albright. All rights reserved.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::core::manifest::{Schema, PartitionSpec, SortOrder};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct SnapshotLogEntry {
    pub timestamp_ms: i64,
    pub snapshot_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct MetadataLogEntry {
    pub timestamp_ms: i64,
    pub metadata_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct Snapshot {
    pub snapshot_id: i64,
    pub parent_snapshot_id: Option<i64>,
    pub timestamp_ms: i64,
    pub sequence_number: Option<i64>,
    pub summary: HashMap<String, String>,
    pub manifest_list: String,
    pub schema_id: Option<i32>,
    
    // V3 Row Lineage (Iceberg spec v3 required)
    /// The first _row_id assigned to the first row in the first data file
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,
    
    /// The upper bound of the number of rows with assigned row IDs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added_rows: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct TableMetadata {
    pub format_version: i32,
    pub table_uuid: String,
    pub location: String,
    pub last_sequence_number: i64,
    pub last_updated_ms: i64,
    pub last_column_id: i32,
    
    // Schema
    pub current_schema_id: i32,
    pub schemas: Vec<Schema>,
    
    // Partitioning
    pub default_spec_id: i32,
    pub partition_specs: Vec<PartitionSpec>,
    
    // Sorting
    pub default_sort_order_id: i32,
    pub sort_orders: Vec<SortOrder>,
    
    // Primary Key (Iceberg Identifier Fields)
    #[serde(rename = "identifier-field-ids", default, skip_serializing_if = "Vec::is_empty")]
    pub identifier_field_ids: Vec<i32>,
    
    // Properties
    #[serde(default)]
    pub properties: HashMap<String, String>,
    
    // Snapshots
    #[serde(default)]
    pub current_snapshot_id: Option<i64>,
    #[serde(default)]
    pub snapshots: Vec<Snapshot>,
    
    #[serde(default)]
    pub snapshot_log: Vec<SnapshotLogEntry>,
    
    #[serde(default)]
    pub metadata_log: Vec<MetadataLogEntry>,
    
    // V3 Row Lineage (Iceberg spec v3 required)
    /// A long higher than all assigned row IDs; the next snapshot's first-row-id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_row_id: Option<i64>,
}

impl TableMetadata {
    /// Create a minimal metadata skeleton (used by catalogs that only know the location)
    pub fn minimal(location: String) -> Self {
        Self {
            format_version: 2,
            location,
            ..Default::default()
        }
    }

    pub fn new(
        format_version: i32,
        uuid: String,
        location: String,
        schema: Schema,
        partition_spec: PartitionSpec,
        sort_order: SortOrder,
    ) -> Self {
        Self {
            format_version,
            table_uuid: uuid,
            location,
            last_sequence_number: 0,
            last_updated_ms: chrono::Utc::now().timestamp_millis(),
            last_column_id: schema.fields.iter().map(|f| f.id).max().unwrap_or(0),
            current_schema_id: schema.schema_id,
            schemas: vec![schema.clone()],
            default_spec_id: partition_spec.spec_id,
            partition_specs: vec![partition_spec],
            default_sort_order_id: sort_order.order_id,
            sort_orders: vec![sort_order],
            properties: HashMap::new(),
            current_snapshot_id: None,
            snapshots: Vec::new(),
            snapshot_log: Vec::new(),
            metadata_log: Vec::new(),
            next_row_id: if format_version >= 3 { Some(0) } else { None },
            identifier_field_ids: schema.identifier_field_ids.clone(),
        }
    }

    /// Add a new snapshot and update current_snapshot_id
    pub fn add_snapshot(&mut self, snapshot: Snapshot) {
        self.last_updated_ms = snapshot.timestamp_ms;
        self.current_snapshot_id = Some(snapshot.snapshot_id);
        self.snapshot_log.push(SnapshotLogEntry {
            timestamp_ms: snapshot.timestamp_ms,
            snapshot_id: snapshot.snapshot_id,
        });
        self.snapshots.push(snapshot);
    }

    /// Add a new schema
    pub fn add_schema(&mut self, schema: Schema) {
        self.last_column_id = schema.fields.iter().map(|f| f.id).max().unwrap_or(self.last_column_id);
        self.current_schema_id = schema.schema_id;
        self.schemas.push(schema);
    }

    /// Add a new partition spec
    pub fn add_partition_spec(&mut self, spec: PartitionSpec) {
        self.default_spec_id = spec.spec_id;
        self.partition_specs.push(spec);
    }

    /// Save metadata to ObjectStore as a new version v<X>.metadata.json
    pub async fn save_to_store(&mut self, storage: &dyn object_store::ObjectStore, version: i32) -> anyhow::Result<String> {
        let path = object_store::path::Path::from(format!("metadata/v{}.metadata.json", version));
        let json = serde_json::to_vec(self)?;
        
        storage.put(&path, json.into()).await?;
        
        // Also update version-hint.text
        let hint_path = object_store::path::Path::from("metadata/version-hint.text");
        storage.put(&hint_path, version.to_string().into()).await?;
        
        Ok(path.to_string())
    }

    /// Load metadata from ObjectStore
    pub async fn load_from_store(storage: &dyn object_store::ObjectStore, path: &object_store::path::Path) -> anyhow::Result<Self> {
        let res = storage.get(path).await?;
        let bytes = res.bytes().await?;
        let metadata: Self = serde_json::from_slice(&bytes)?;
        Ok(metadata)
    }

    /// Load the latest metadata version using version-hint.text
    pub async fn load_latest(storage: &dyn object_store::ObjectStore) -> anyhow::Result<Self> {
        let hint_path = object_store::path::Path::from("metadata/version-hint.text");
        let res = storage.get(&hint_path).await?;
        let bytes = res.bytes().await?;
        let version_str = String::from_utf8(bytes.to_vec())?;
        let version = version_str.trim().parse::<i32>()?;
        
        let path = object_store::path::Path::from(format!("metadata/v{}.metadata.json", version));
        Self::load_from_store(storage, &path).await
    }
}
