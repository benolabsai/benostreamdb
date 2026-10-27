// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::manifest::{PartitionSpec, Schema, SortOrder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Iceberg's parser rejects an explicit JSON `null` for this key, so it is
    /// omitted entirely for a root snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_snapshot_id: Option<i64>,
    pub timestamp_ms: i64,
    pub sequence_number: Option<i64>,
    pub summary: HashMap<String, String>,
    pub manifest_list: String,
    #[serde(skip_serializing_if = "Option::is_none")]
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

    /// Highest partition field id ever assigned. Iceberg requires this key and
    /// reserves ids from 1000 upwards, so an unpartitioned table reports 999.
    #[serde(default = "default_last_partition_id")]
    pub last_partition_id: i32,

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
    #[serde(
        rename = "identifier-field-ids",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
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

/// First partition field id Iceberg assigns; unpartitioned tables report one less.
const ICEBERG_PARTITION_DATA_ID_START: i32 = 1000;

fn default_last_partition_id() -> i32 {
    ICEBERG_PARTITION_DATA_ID_START - 1
}

impl TableMetadata {
    /// Create a minimal metadata skeleton (used by catalogs that only know the location)
    pub fn minimal(location: String) -> Self {
        Self {
            format_version: 2,
            location,
            last_partition_id: default_last_partition_id(),
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
            last_partition_id: partition_spec
                .fields
                .iter()
                .filter_map(|f| f.field_id)
                .max()
                .unwrap_or(default_last_partition_id())
                .max(default_last_partition_id()),
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
        // Iceberg validates that `last-sequence-number` covers every snapshot.
        if let Some(sequence_number) = snapshot.sequence_number {
            self.last_sequence_number = self.last_sequence_number.max(sequence_number);
        }
        self.snapshot_log.push(SnapshotLogEntry {
            timestamp_ms: snapshot.timestamp_ms,
            snapshot_id: snapshot.snapshot_id,
        });
        self.snapshots.push(snapshot);
    }

    /// Add a new schema
    pub fn add_schema(&mut self, schema: Schema) {
        self.last_column_id = schema
            .fields
            .iter()
            .map(|f| f.id)
            .max()
            .unwrap_or(self.last_column_id);
        self.current_schema_id = schema.schema_id;
        self.schemas.push(schema);
    }

    /// Add a new partition spec
    pub fn add_partition_spec(&mut self, spec: PartitionSpec) {
        self.default_spec_id = spec.spec_id;
        if let Some(field_id) = spec.fields.iter().filter_map(|f| f.field_id).max() {
            self.last_partition_id = self.last_partition_id.max(field_id);
        }
        self.partition_specs.push(spec);
    }

    /// Fill in the invariants Iceberg's `TableMetadataParser` enforces but that
    /// older BenoStreamDB manifests may not have recorded.
    ///
    /// Kept at the write boundary (rather than in the constructors) so tables
    /// written by earlier releases are upgraded as they are next committed.
    fn normalize_iceberg_metadata(&mut self) {
        // Iceberg rejects an empty `sort-orders` array.
        if self.sort_orders.is_empty() {
            self.sort_orders.push(SortOrder {
                order_id: self.default_sort_order_id,
                fields: Vec::new(),
            });
        }

        // The default partition spec must exist, even when unpartitioned.
        if self.partition_specs.is_empty() {
            self.partition_specs.push(PartitionSpec {
                spec_id: self.default_spec_id,
                fields: Vec::new(),
            });
        }

        // Every partition field needs a concrete `source-id` and `field-id`:
        // Iceberg parses both as required integers.
        let mut next_field_id = self.last_partition_id.max(default_last_partition_id()) + 1;
        let mut max_field_id = self.last_partition_id.max(default_last_partition_id());
        for spec in &mut self.partition_specs {
            for field in &mut spec.fields {
                if field.source_id.is_none() {
                    field.source_id = field.source_ids.first().copied();
                }
                if field.field_id.is_none() {
                    field.field_id = Some(next_field_id);
                    next_field_id += 1;
                }
                if let Some(field_id) = field.field_id {
                    max_field_id = max_field_id.max(field_id);
                }
            }
        }
        self.last_partition_id = max_field_id;

        // `last-sequence-number` must cover every snapshot in the table.
        let max_sequence_number = self
            .snapshots
            .iter()
            .filter_map(|snapshot| snapshot.sequence_number)
            .max()
            .unwrap_or(0);
        self.last_sequence_number = self.last_sequence_number.max(max_sequence_number);

        // Row lineage (`first-row-id` / `added-rows`) is a v3 snapshot field.
        // Emitting it in a v2 table is outside the spec, even though tolerant
        // readers ignore it.
        if self.format_version < 3 {
            for snapshot in &mut self.snapshots {
                snapshot.first_row_id = None;
                snapshot.added_rows = None;
            }
        }
    }

    /// Save metadata to ObjectStore as a new version `v<X>.metadata.json`
    pub async fn save_to_store(
        &mut self,
        storage: &dyn object_store::ObjectStore,
        version: i32,
    ) -> anyhow::Result<String> {
        self.normalize_iceberg_metadata();
        let path = object_store::path::Path::from(format!("metadata/v{}.metadata.json", version));
        let json = serde_json::to_vec(self)?;

        storage.put(&path, json.into()).await?;

        // Advance `version-hint.text` monotonically. Data commits and
        // properties-only commits both write metadata, and a writer that read an
        // older head can finish last; regressing the hint would hide the newer
        // metadata (and its table properties) from external readers.
        let hint_path = object_store::path::Path::from("metadata/version-hint.text");
        let current_version = match storage.get(&hint_path).await {
            Ok(res) => res
                .bytes()
                .await
                .ok()
                .and_then(|bytes| String::from_utf8(bytes.to_vec()).ok())
                .and_then(|value| value.trim().parse::<i32>().ok()),
            Err(_) => None,
        };
        if current_version.is_none_or(|current| version >= current) {
            storage.put(&hint_path, version.to_string().into()).await?;
        }

        Ok(path.to_string())
    }

    /// Load metadata from ObjectStore
    pub async fn load_from_store(
        storage: &dyn object_store::ObjectStore,
        path: &object_store::path::Path,
    ) -> anyhow::Result<Self> {
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
