// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{anyhow, Result};
use arrow::datatypes::SchemaRef;
use object_store::path::Path as ObjPath;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::core::error::BenoStreamError;
use crate::core::manifest::{ManifestManager, PartitionSpec, Schema, SortOrder};
use crate::core::metadata::{MetadataLogEntry, Snapshot, SnapshotLogEntry, TableMetadata};
use crate::core::storage::create_object_store;

/// Canonical Iceberg REST `TableRequirement` predicates validated atomically
/// against existing table metadata during `commit_table_with_requirements`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum TableRequirement {
    AssertCreate,
    AssertTableUuid {
        uuid: String,
    },
    AssertRefSnapshotId {
        #[serde(rename = "ref")]
        ref_name: String,
        #[serde(rename = "snapshot-id")]
        snapshot_id: Option<i64>,
    },
    AssertCurrentSchemaId {
        #[serde(rename = "current-schema-id")]
        current_schema_id: i32,
    },
    AssertLastAssignedFieldId {
        #[serde(rename = "last-assigned-field-id")]
        last_assigned_field_id: i32,
    },
    AssertLastAssignedPartitionId {
        #[serde(rename = "last-assigned-partition-id")]
        last_assigned_partition_id: i32,
    },
    AssertDefaultSpecId {
        #[serde(rename = "default-spec-id")]
        default_spec_id: i32,
    },
    AssertDefaultSortOrderId {
        #[serde(rename = "default-sort-order-id")]
        default_sort_order_id: i32,
    },
    #[serde(other)]
    Unknown,
}

pub fn catalog_conflict_error(catalog: &str, reason: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(BenoStreamError::CatalogRequest {
        catalog: catalog.to_string(),
        status: Some(409),
        reason: reason.into(),
    })
}

pub fn catalog_bad_request_error(catalog: &str, reason: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(BenoStreamError::CatalogRequest {
        catalog: catalog.to_string(),
        status: Some(400),
        reason: reason.into(),
    })
}

pub fn catalog_not_found_error(catalog: &str, reason: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(BenoStreamError::CatalogRequest {
        catalog: catalog.to_string(),
        status: Some(404),
        reason: reason.into(),
    })
}

/// Parse raw JSON Iceberg requirements into typed [`TableRequirement`] values.
pub fn parse_requirements(
    requirements: &[serde_json::Value],
    catalog_name: &str,
) -> Result<Vec<TableRequirement>> {
    requirements
        .iter()
        .map(|val| {
            serde_json::from_value::<TableRequirement>(val.clone()).map_err(|e| {
                catalog_bad_request_error(catalog_name, format!("Invalid table requirement: {e}"))
            })
        })
        .collect()
}

/// Validate parsed [`TableRequirement`]s against existing [`TableMetadata`].
pub fn validate_requirements(
    requirements: &[TableRequirement],
    existing_meta: Option<&TableMetadata>,
    catalog_name: &str,
) -> Result<()> {
    for req in requirements {
        match req {
            TableRequirement::AssertCreate => {
                if existing_meta.is_some() {
                    return Err(catalog_conflict_error(
                        catalog_name,
                        "Requirement failed: assert-create (table already exists)",
                    ));
                }
            }
            TableRequirement::AssertTableUuid { uuid } => {
                let actual = existing_meta.map(|m| m.table_uuid.as_str()).unwrap_or("");
                if actual != uuid {
                    return Err(catalog_conflict_error(
                        catalog_name,
                        format!("Requirement failed: assert-table-uuid expected '{uuid}', got '{actual}'"),
                    ));
                }
            }
            TableRequirement::AssertRefSnapshotId {
                ref_name,
                snapshot_id,
            } => {
                if ref_name == "main" {
                    match existing_meta {
                        Some(meta) => {
                            if meta.current_snapshot_id != *snapshot_id {
                                return Err(catalog_conflict_error(
                                    catalog_name,
                                    format!(
                                        "Requirement failed: assert-ref-snapshot-id on '{}' expected {:?}, got {:?}",
                                        ref_name, snapshot_id, meta.current_snapshot_id
                                    ),
                                ));
                            }
                        }
                        None => {
                            if snapshot_id.is_some() {
                                return Err(catalog_conflict_error(
                                    catalog_name,
                                    format!(
                                        "Requirement failed: assert-ref-snapshot-id on '{}' expected {:?}, got missing table",
                                        ref_name, snapshot_id
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
            TableRequirement::AssertCurrentSchemaId { current_schema_id } => {
                let actual = existing_meta.map(|m| m.current_schema_id);
                if actual != Some(*current_schema_id) {
                    return Err(catalog_conflict_error(
                        catalog_name,
                        format!(
                            "Requirement failed: assert-current-schema-id expected {current_schema_id}, got {actual:?}"
                        ),
                    ));
                }
            }
            TableRequirement::AssertLastAssignedFieldId {
                last_assigned_field_id,
            } => {
                let actual = existing_meta.map(|m| m.last_column_id);
                if actual != Some(*last_assigned_field_id) {
                    return Err(catalog_conflict_error(
                        catalog_name,
                        format!(
                            "Requirement failed: assert-last-assigned-field-id expected {last_assigned_field_id}, got {actual:?}"
                        ),
                    ));
                }
            }
            TableRequirement::AssertLastAssignedPartitionId {
                last_assigned_partition_id,
            } => {
                let actual = existing_meta.map(|m| m.last_partition_id);
                if actual != Some(*last_assigned_partition_id) {
                    return Err(catalog_conflict_error(
                        catalog_name,
                        format!(
                            "Requirement failed: assert-last-assigned-partition-id expected {last_assigned_partition_id}, got {actual:?}"
                        ),
                    ));
                }
            }
            TableRequirement::AssertDefaultSpecId { default_spec_id } => {
                let actual = existing_meta.map(|m| m.default_spec_id);
                if actual != Some(*default_spec_id) {
                    return Err(catalog_conflict_error(
                        catalog_name,
                        format!(
                            "Requirement failed: assert-default-spec-id expected {default_spec_id}, got {actual:?}"
                        ),
                    ));
                }
            }
            TableRequirement::AssertDefaultSortOrderId {
                default_sort_order_id,
            } => {
                let actual = existing_meta.map(|m| m.default_sort_order_id);
                if actual != Some(*default_sort_order_id) {
                    return Err(catalog_conflict_error(
                        catalog_name,
                        format!(
                            "Requirement failed: assert-default-sort-order-id expected {default_sort_order_id}, got {actual:?}"
                        ),
                    ));
                }
            }
            TableRequirement::Unknown => {}
        }
    }
    Ok(())
}

/// Parse and validate raw JSON Iceberg requirements against `existing_meta`.
pub fn validate_requirements_against_metadata(
    requirements: &[serde_json::Value],
    existing_meta: Option<&TableMetadata>,
    catalog_name: &str,
) -> Result<()> {
    if requirements.is_empty() {
        return Ok(());
    }
    let parsed = parse_requirements(requirements, catalog_name)?;
    validate_requirements(&parsed, existing_meta, catalog_name)
}

/// Derive the Iceberg table root directory URI from a `metadata_location` pointer.
///
/// Handles canonical `<root>/metadata/v{N}.metadata.json`, legacy
/// `<root>/manifest.json`, and bare table root URIs consistently.
pub fn table_root_from_metadata_location(meta_loc: &str) -> String {
    let trimmed = meta_loc.trim();
    if let Some((root, rest)) = trimmed.rsplit_once("/metadata/") {
        if rest.ends_with(".json") {
            return root.trim_end_matches('/').to_string();
        }
    }
    if let Some(root) = trimmed.strip_suffix("/manifest.json") {
        return root.trim_end_matches('/').to_string();
    }
    trimmed.trim_end_matches('/').to_string()
}

/// Extract the target `metadata-location` from Iceberg REST table updates.
///
/// Prefers explicit `set-metadata-location` updates; falls back to deriving
/// `<table_root>/metadata/v{N}.metadata.json` from `add-snapshot` or
/// `set-current-snapshot` manifest list paths.
pub fn extract_metadata_location_from_updates(updates: &[serde_json::Value]) -> Option<String> {
    for update in updates {
        if update.get("action").and_then(|v| v.as_str()) == Some("set-metadata-location") {
            if let Some(loc) = update.get("metadata-location").and_then(|v| v.as_str()) {
                return Some(loc.to_string());
            }
        }
    }

    for update in updates {
        let action = update.get("action").and_then(|v| v.as_str());
        if matches!(action, Some("add-snapshot" | "set-current-snapshot")) {
            if let Some(snapshot) = update.get("snapshot") {
                if let Some(manifest_list) = snapshot.get("manifest-list").and_then(|v| v.as_str())
                {
                    if let Some((table_root, _)) = manifest_list.rsplit_once("/_manifest/") {
                        let ver = snapshot
                            .get("sequence-number")
                            .and_then(|v| v.as_i64())
                            .or_else(|| snapshot.get("snapshot-id").and_then(|v| v.as_i64()))
                            .unwrap_or(1);
                        return Some(format!("{}/metadata/v{}.metadata.json", table_root, ver));
                    }
                    if let Some((table_root, _)) = manifest_list.rsplit_once("/metadata/") {
                        let ver = snapshot
                            .get("sequence-number")
                            .and_then(|v| v.as_i64())
                            .or_else(|| snapshot.get("snapshot-id").and_then(|v| v.as_i64()))
                            .unwrap_or(1);
                        return Some(format!("{}/metadata/v{}.metadata.json", table_root, ver));
                    }
                }
            }
        }
    }

    None
}

/// Resolve [`TableMetadata`] stored at `meta_loc` (either a `vN.metadata.json` URI or table root).
///
/// When metadata JSON cannot be read (e.g. external tables pre-registered as bare roots
/// without BenoStream metadata files), falls back to [`TableMetadata::minimal`] anchored
/// at [`table_root_from_metadata_location`].
pub async fn load_table_metadata_from_location(meta_loc: &str) -> Result<TableMetadata> {
    let table_root = table_root_from_metadata_location(meta_loc);
    if table_root.is_empty() {
        return Ok(TableMetadata::minimal(String::new()));
    }

    let store = match create_object_store(&table_root) {
        Ok(s) => s,
        Err(_) => return Ok(TableMetadata::minimal(table_root)),
    };

    if let Some((_, file_name)) = meta_loc.trim().rsplit_once("/metadata/") {
        if file_name.ends_with(".json") && !file_name.contains('/') {
            let rel_path = ObjPath::from(format!("metadata/{file_name}"));
            if let Ok(mut meta) = TableMetadata::load_from_store(store.as_ref(), &rel_path).await {
                if meta.location.is_empty() {
                    meta.location = table_root;
                }
                return Ok(meta);
            }
        }
    }

    if let Ok(mut meta) = TableMetadata::load_latest(store.as_ref()).await {
        if meta.location.is_empty() {
            meta.location = table_root;
        }
        return Ok(meta);
    }

    Ok(TableMetadata::minimal(table_root))
}

/// Determine initial metadata file URI for a newly created table at `table_location`.
pub async fn resolve_initial_metadata_location(table_location: &str) -> String {
    let clean_root = table_location.trim_end_matches('/');
    if let Ok(store) = create_object_store(table_location) {
        if let Ok(Some(ver)) = TableMetadata::latest_version_in_store(store.as_ref()).await {
            return format!("{clean_root}/metadata/v{ver}.metadata.json");
        }
    }
    format!("{clean_root}/metadata/v1.metadata.json")
}

/// Ensure a BenoStreamDB / Iceberg table exists in object storage at `table_location`,
/// initializing `_manifest/v1.json` and `metadata/v1.metadata.json` only if absent.
///
/// Tolerates tables pre-created by a caller (e.g., `bsdb-search` creating storage
/// before catalog registration) as well as concurrent initialization races, returning
/// the initial metadata JSON location.
pub async fn ensure_storage_table_initialized(
    table_location: &str,
    schema: &SchemaRef,
) -> Result<String> {
    let clean_root = table_location.trim_end_matches('/').to_string();
    let store = create_object_store(&clean_root)?;
    let mm = ManifestManager::new(store.clone(), "", &clean_root);
    let (_, manifest_ver) = mm.load_latest_direct().await?;
    let meta_ver = TableMetadata::latest_version_in_store(store.as_ref())
        .await
        .ok()
        .flatten();

    if manifest_ver == 0 && meta_ver.is_none() {
        match crate::Table::create_async(clean_root.clone(), schema.clone()).await {
            Ok(_) => {}
            Err(e) if e.to_string().contains("already exists") => {}
            Err(e) => return Err(e),
        }
    }

    Ok(resolve_initial_metadata_location(&clean_root).await)
}

/// Apply standard Iceberg catalog `updates` payloads to an in-memory [`TableMetadata`].
pub fn apply_table_updates(meta: &mut TableMetadata, updates: &[serde_json::Value]) -> Result<()> {
    for update in updates {
        let action = update
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'action' field in table update: {update}"))?;

        match action {
            "assign-uuid" => {
                if let Some(uuid) = update.get("uuid").and_then(|v| v.as_str()) {
                    meta.table_uuid = uuid.to_string();
                }
            }
            "upgrade-format-version" => {
                if let Some(fv) = update.get("format-version").and_then(|v| v.as_i64()) {
                    let fv = fv as i32;
                    meta.format_version = meta.format_version.max(fv);
                    if meta.format_version >= 3 && meta.next_row_id.is_none() {
                        meta.next_row_id = Some(0);
                    }
                }
            }
            "add-schema" => {
                let schema_val = update
                    .get("schema")
                    .ok_or_else(|| anyhow!("add-schema missing 'schema'"))?;
                let schema: Schema = serde_json::from_value(schema_val.clone())?;
                if let Some(lci) = update.get("last-column-id").and_then(|v| v.as_i64()) {
                    meta.last_column_id = meta.last_column_id.max(lci as i32);
                } else {
                    let max_id = schema.fields.iter().map(|f| f.id).max().unwrap_or(0);
                    meta.last_column_id = meta.last_column_id.max(max_id);
                }
                if !meta.schemas.iter().any(|s| s.schema_id == schema.schema_id) {
                    meta.schemas.push(schema);
                }
            }
            "set-current-schema" => {
                if let Some(id) = update.get("schema-id").and_then(|v| v.as_i64()) {
                    let target = if id == -1 {
                        meta.schemas
                            .last()
                            .map(|s| s.schema_id)
                            .unwrap_or(meta.current_schema_id)
                    } else {
                        id as i32
                    };
                    meta.current_schema_id = target;
                }
            }
            "add-spec" => {
                let spec_val = update
                    .get("spec")
                    .ok_or_else(|| anyhow!("add-spec missing 'spec'"))?;
                let spec: PartitionSpec = serde_json::from_value(spec_val.clone())?;
                if let Some(fid) = spec.fields.iter().filter_map(|f| f.field_id).max() {
                    meta.last_partition_id = meta.last_partition_id.max(fid);
                }
                if !meta
                    .partition_specs
                    .iter()
                    .any(|s| s.spec_id == spec.spec_id)
                {
                    meta.partition_specs.push(spec);
                }
            }
            "set-default-spec" => {
                if let Some(id) = update.get("spec-id").and_then(|v| v.as_i64()) {
                    let target = if id == -1 {
                        meta.partition_specs
                            .last()
                            .map(|s| s.spec_id)
                            .unwrap_or(meta.default_spec_id)
                    } else {
                        id as i32
                    };
                    meta.default_spec_id = target;
                }
            }
            "add-sort-order" => {
                let order_val = update
                    .get("sort-order")
                    .ok_or_else(|| anyhow!("add-sort-order missing 'sort-order'"))?;
                let order: SortOrder = serde_json::from_value(order_val.clone())?;
                if !meta
                    .sort_orders
                    .iter()
                    .any(|o| o.order_id == order.order_id)
                {
                    meta.sort_orders.push(order);
                }
            }
            "set-default-sort-order" => {
                if let Some(id) = update.get("sort-order-id").and_then(|v| v.as_i64()) {
                    let target = if id == -1 {
                        meta.sort_orders
                            .last()
                            .map(|o| o.order_id)
                            .unwrap_or(meta.default_sort_order_id)
                    } else {
                        id as i32
                    };
                    meta.default_sort_order_id = target;
                }
            }
            "add-snapshot" => {
                let snap_val = update
                    .get("snapshot")
                    .ok_or_else(|| anyhow!("add-snapshot missing 'snapshot'"))?;
                let snap: Snapshot = serde_json::from_value(snap_val.clone())?;
                let candidate_next_row_id = snap
                    .first_row_id
                    .zip(snap.added_rows)
                    .map(|(first, added)| first.saturating_add(added));
                meta.merge_snapshot_monotonic(snap, candidate_next_row_id);
            }
            "set-current-snapshot" => {
                if let Some(snap_id) = update.get("snapshot-id").and_then(|v| v.as_i64()) {
                    meta.current_snapshot_id = (snap_id >= 0).then_some(snap_id);
                    if let Some(id) = meta.current_snapshot_id {
                        let ts = meta
                            .snapshots
                            .iter()
                            .find(|s| s.snapshot_id == id)
                            .map(|s| s.timestamp_ms)
                            .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
                        if meta.snapshot_log.last().map(|e| e.snapshot_id) != Some(id) {
                            meta.snapshot_log.push(SnapshotLogEntry {
                                timestamp_ms: ts,
                                snapshot_id: id,
                            });
                        }
                        meta.last_updated_ms = meta.last_updated_ms.max(ts);
                    }
                }
            }
            "set-snapshot-ref" => {
                if update.get("ref-name").and_then(|v| v.as_str()) == Some("main") {
                    if let Some(id) = update.get("snapshot-id").and_then(|v| v.as_i64()) {
                        meta.current_snapshot_id = (id >= 0).then_some(id);
                    }
                }
            }
            "remove-snapshots" => {
                if let Some(ids) = update.get("snapshot-ids").and_then(|v| v.as_array()) {
                    let drop_set: HashSet<i64> = ids.iter().filter_map(|v| v.as_i64()).collect();
                    meta.snapshots
                        .retain(|s| !drop_set.contains(&s.snapshot_id));
                    meta.snapshot_log
                        .retain(|e| !drop_set.contains(&e.snapshot_id));
                }
            }
            "set-properties" => {
                if let Some(map_val) = update.get("updates").and_then(|v| v.as_object()) {
                    let parsed: HashMap<String, String> = map_val
                        .iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect();
                    meta.properties.extend(parsed);
                }
            }
            "remove-properties" => {
                if let Some(removals) = update.get("removals").and_then(|v| v.as_array()) {
                    for key in removals.iter().filter_map(|v| v.as_str()) {
                        meta.properties.remove(key);
                    }
                }
            }
            "set-location" => {
                if let Some(loc) = update.get("location").and_then(|v| v.as_str()) {
                    meta.location = loc.to_string();
                }
            }
            "set-metadata-location" => {
                if let Some(loc) = update.get("metadata-location").and_then(|v| v.as_str()) {
                    let ts = chrono::Utc::now().timestamp_millis();
                    if meta.metadata_log.last().map(|m| m.metadata_file.as_str()) != Some(loc) {
                        meta.metadata_log.push(MetadataLogEntry {
                            timestamp_ms: ts,
                            metadata_file: loc.to_string(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    meta.last_updated_ms = meta
        .last_updated_ms
        .max(chrono::Utc::now().timestamp_millis());
    Ok(())
}

/// Persist property/schema/snapshot updates directly to table storage when a caller invokes
/// [`super::Catalog::commit_table`] without having pre-written a `v{N}.metadata.json` file.
pub async fn persist_updates_to_storage(
    table_root: &str,
    updates: &[serde_json::Value],
) -> Result<String> {
    let store = create_object_store(table_root)?;
    let updates_owned = updates.to_vec();
    let (rel_path, _) = TableMetadata::commit_update_atomic(store.as_ref(), 1, move |existing| {
        let mut meta = existing.unwrap_or_else(|| TableMetadata::minimal(table_root.into()));
        if meta.location.is_empty() {
            meta.location = table_root.to_string();
        }
        apply_table_updates(&mut meta, &updates_owned)?;
        Ok(crate::core::metadata::MetadataUpdateOutcome::Updated(meta))
    })
    .await?;
    Ok(format!(
        "{}/{}",
        table_root.trim_end_matches('/'),
        rel_path.trim_start_matches('/')
    ))
}
