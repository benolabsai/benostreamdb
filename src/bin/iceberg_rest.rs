// Copyright (c) 2026 Richard Albright. All rights reserved.

use ax_lib::{
    routing::{get, post},
    Router, Json,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use futures::StreamExt;
use object_store::ObjectStore;
use hyperstreamdb::core::manifest::ManifestManager;
use hyperstreamdb::core::metadata::TableMetadata;

#[derive(Debug, Serialize, Deserialize)]
pub struct CatalogConfig {
    pub overrides: std::collections::HashMap<String, String>,
    pub defaults: std::collections::HashMap<String, String>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/v1/config", get(get_config))
        .route("/v1/:prefix/namespaces", get(list_namespaces))
        .route("/v1/:prefix/namespaces/:namespace/tables", get(list_tables).post(create_table))
        .route("/v1/:prefix/namespaces/:namespace/tables/:table", get(get_table).post(update_table))
        .route("/v1/:prefix/namespaces/:namespace/tables/:table/register", post(register_table));

    let port = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8181);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Iceberg REST Server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    ax_lib::serve(listener, app).await.unwrap();
}

async fn get_config() -> impl IntoResponse {
    let mut overrides = std::collections::HashMap::new();
    overrides.insert("prefix".to_string(), "hdb".to_string());
    
    let config = CatalogConfig {
        overrides,
        defaults: std::collections::HashMap::new(),
    };
    
    Json(config)
}

async fn list_namespaces(ax_lib::extract::Path(prefix): ax_lib::extract::Path<String>) -> impl IntoResponse {
    println!("Catalog prefix: {}", prefix);
    let uri = std::env::var("HYPERSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp".to_string());
    let store = hyperstreamdb::core::storage::create_object_store(&uri).expect("Failed to create object store");
    
    // Discover namespaces by listing top-level directories
    let mut namespaces = std::collections::HashSet::new();
    let mut stream = store.list(None);
    while let Some(meta) = stream.next().await {
        if let Ok(meta) = meta {
            let path_str: String = meta.location.to_string();
            if let Some(ns) = path_str.split('/').next() {
                if !ns.is_empty() && !ns.starts_with('_') && !ns.contains('.') {
                    namespaces.insert(vec![ns.to_string()]);
                }
            }
        }
    }
    
    if namespaces.is_empty() {
        namespaces.insert(vec!["default".to_string()]);
    }

    let response = serde_json::json!({
        "namespaces": namespaces.into_iter().collect::<Vec<_>>()
    });
    Json(response)
}

async fn list_tables(
    ax_lib::extract::Path((prefix, namespace)): ax_lib::extract::Path<(String, String)>
) -> impl IntoResponse {
    println!("Catalog prefix: {}, namespace: {}", prefix, namespace);
    let uri = std::env::var("HYPERSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp".to_string());
    let store = hyperstreamdb::core::storage::create_object_store(&uri).expect("Failed to create object store");
    
    let mut tables = Vec::new();
    let prefix_path = object_store::path::Path::from(namespace.as_str());
    let mut stream = store.list(Some(&prefix_path));
    
    let mut seen_tables = std::collections::HashSet::new();
    while let Some(meta) = stream.next().await {
        if let Ok(meta) = meta {
            let path_str: String = meta.location.to_string();
            if path_str.contains("/_manifest/") {
                if let Some(table_name) = path_str.strip_prefix(&(namespace.clone() + "/")) {
                    let parts: Vec<&str> = table_name.split('/').collect();
                    if let Some(name) = parts.first() {
                         if seen_tables.insert(name.to_string()) {
                            tables.push(serde_json::json!({
                                "namespace": [namespace],
                                "name": name
                            }));
                         }
                    }
                }
            }
        }
    }

    let response = serde_json::json!({
        "identifiers": tables
    });
    Json(response)
}

// Replaced by hyperstreamdb::core::metadata::TableMetadata

async fn get_table(
    ax_lib::extract::Path((prefix, namespace, table)): ax_lib::extract::Path<(String, String, String)>
) -> impl IntoResponse {
    let uri = std::env::var("HYPERSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp".to_string());
    println!("Prefix: {}, Getting metadata for {}.{} (Storage: {})", prefix, namespace, table, uri);
    
    let table_path = format!("{}/{}", namespace, table);
    let table_full_uri = format!("{}/{}", uri.trim_end_matches('/'), table_path);
    
    let store = hyperstreamdb::core::storage::create_object_store(&table_full_uri).expect("Failed to create object store");
    let manager = ManifestManager::new(store.clone(), "", &table_full_uri);
    
    // Try to load official TableMetadata first
    let metadata_result = TableMetadata::load_latest(store.as_ref()).await;
    
    // Get version from hint for filename
    let version = if let Ok(res) = store.get(&object_store::path::Path::from("metadata/version-hint.text")).await {
        let bytes = res.bytes().await.unwrap_or_default();
        String::from_utf8(bytes.to_vec()).unwrap_or_else(|_| "0".to_string()).trim().parse::<i32>().unwrap_or(0)
    } else {
        0
    };

    match metadata_result {
        Ok(metadata) => {
            (StatusCode::OK, Json(serde_json::json!({
                "metadata-location": format!("{}/metadata/v{}.metadata.json", table_full_uri, version),
                "metadata": metadata,
                "config": {}
            }))).into_response()
        },
        Err(e) => {
            eprintln!("Official metadata NOT found: {}. Falling back to manifest scan.", e);
            // Fallback to legacy manifest scanning
            match manager.load_latest().await {
                Ok((manifest, version)) => {
                    let mut metadata = TableMetadata::new(
                        2,
                        uuid::Uuid::new_v4().to_string(),
                        table_full_uri.clone(),
                        manifest.schemas.last().cloned().unwrap_or_default(),
                        manifest.partition_spec.clone(),
                        manifest.sort_orders.first().cloned().unwrap_or_default()
                    );
                    metadata.last_sequence_number = version as i64;
                    metadata.last_updated_ms = manifest.timestamp_ms;
                    
                    (StatusCode::OK, Json(serde_json::json!({
                        "metadata-location": format!("{}/metadata/v{}.metadata.json", table_full_uri, version),
                        "metadata": metadata,
                        "config": {}
                    }))).into_response()
                },
                Err(e2) => {
                    eprintln!("Failed to load manifest for {}: {}", table, e2);
                    (StatusCode::NOT_FOUND, Json(serde_json::json!({
                        "error": {
                            "message": format!("Table not found: {} / {}", e, e2),
                            "type": "NoSuchTableException",
                            "code": 404
                        }
                    }))).into_response()
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct CreateTableRequest {
    pub name: String,
    pub location: Option<String>,
    pub schema: serde_json::Value,
    pub partition_spec: Option<serde_json::Value>,
    pub sort_order: Option<serde_json::Value>,
    pub properties: Option<std::collections::HashMap<String, String>>,
}

async fn create_table(
    ax_lib::extract::Path((prefix, namespace)): ax_lib::extract::Path<(String, String)>,
    Json(payload): Json<CreateTableRequest>
) -> impl IntoResponse {
    println!("Creating table {}.{}.{}", prefix, namespace, payload.name);

    let base_uri = std::env::var("HYPERSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp".to_string());
    
    // Determine location
    let location = payload.location.unwrap_or_else(|| {
        format!("{}/{}/{}", base_uri.trim_end_matches('/'), namespace, payload.name)
    });

    // Convert schema
    let arrow_schema = match hyperstreamdb::core::iceberg::iceberg_json_to_arrow_schema(&payload.schema) {
        Ok(s) => s,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": {
                    "message": format!("Invalid schema: {}", e),
                    "type": "BadRequestException",
                    "code": 400
                }
            }))).into_response();
        }
    };

    // Create table
    match hyperstreamdb::Table::create_async(location.clone(), arrow_schema.clone()).await {
        Ok(_) => {
            // Load the newly created metadata
            let store = hyperstreamdb::core::storage::create_object_store(&location).expect("Failed to create object store");
            let metadata = TableMetadata::load_latest(store.as_ref()).await.unwrap_or_else(|_| {
                 // Fallback if load fails (shouldn't happen on fresh create)
                 TableMetadata::new(
                    2, 
                    uuid::Uuid::new_v4().to_string(), 
                    location.clone(), 
                    hyperstreamdb::core::manifest::Schema::from_arrow(&arrow_schema, 0), 
                    hyperstreamdb::core::manifest::PartitionSpec::default(), 
                    hyperstreamdb::core::manifest::SortOrder::default()
                 )
            });

            (StatusCode::OK, Json(serde_json::json!({
                "metadata-location": format!("{}/metadata/v{}.metadata.json", location, metadata.last_sequence_number),
                "metadata": metadata,
                "config": {}
            }))).into_response()
        },
        Err(e) => {
             eprintln!("Failed to create table: {}", e);
             (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": {
                    "message": format!("Failed to create table: {}", e),
                    "type": "RESTException",
                    "code": 500
                }
            }))).into_response()
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TableUpdateRequest {
    pub updates: Vec<TableUpdateAction>,
    pub requirements: Option<Vec<TableRequirement>>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "kebab-case")]
pub enum TableUpdateAction {
    AddSchema { schema: serde_json::Value, #[serde(rename = "last-column-id")] last_column_id: Option<i32> },
    SetCurrentSchema { #[serde(rename = "schema-id")] schema_id: i32 },
    AddSnapshot { snapshot: serde_json::Value },
    #[serde(rename = "remove-sidecar-index")]
    RemoveSidecarIndex {
        #[serde(rename = "index-type")]
        index_type: Option<String>,
        #[serde(rename = "column-name")]
        column_name: Option<String>,
    },
    #[serde(rename = "add-sidecar-index")]
    AddSidecarIndex {
        #[serde(rename = "file-path")]
        file_path: String,
        #[serde(rename = "index-file")]
        index_file: hyperstreamdb::core::manifest::IndexFile,
    },
    AddPartitionSpec { spec: hyperstreamdb::core::manifest::PartitionSpec },
    SetDefaultSpec { #[serde(rename = "spec-id")] spec_id: i32 },
    AddSortOrder { #[serde(rename = "sort-order")] sort_order: hyperstreamdb::core::manifest::SortOrder },
    SetDefaultSortOrder { #[serde(rename = "sort-order-id")] sort_order_id: i32 },
    SetProperties { updates: std::collections::HashMap<String, String> },
    RemoveProperties { removals: Vec<String> },
    UpgradeFormatVersion { #[serde(rename = "format-version")] format_version: i32 },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum TableRequirement {
    AssertCreate,
    #[serde(other)]
    Unknown,
}

async fn update_table(
    ax_lib::extract::Path((prefix, namespace, table)): ax_lib::extract::Path<(String, String, String)>,
    Json(payload): Json<TableUpdateRequest>
) -> impl IntoResponse {
    println!("Updating table {}.{}.{} with {} updates", prefix, namespace, table, payload.updates.len());
    
    let uri = std::env::var("HYPERSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp".to_string());
    let store = hyperstreamdb::core::storage::create_object_store(&uri).expect("Failed to create object store");
    
    // TODO: Support complex namespaces with '/' which might be URL encoded
    let table_path = format!("{}/{}", namespace, table);
    let manager = ManifestManager::new(store.clone(), &table_path, &uri);
    
    // 1. Apply Updates
    let mut new_entries = Vec::new();
    let mut updated_schemas = None;
    let mut updated_schema_id = None;
    
    let mut updated_partition_specs = None;
    let mut updated_default_spec_id = None;
    let mut updated_properties = None;
    let mut removed_properties = None;
    let mut updated_sort_orders = None;
    let mut updated_default_sort_order_id = None;
    let mut updated_last_column_id = None;
    
    // Load current manifest to get schemas for decoding
    let (current_manifest, _) = manager.load_latest().await.unwrap_or((hyperstreamdb::core::manifest::Manifest::default(), 0));
    
    // Pre-load all existing entries if we might be modifying them (rewriting)
    let mut all_existing_entries = if payload.updates.iter().any(|u| matches!(u, TableUpdateAction::RemoveSidecarIndex { .. } | TableUpdateAction::AddSidecarIndex { .. })) {
        Some(manager.load_all_entries(&current_manifest).await.unwrap_or_default())
    } else {
        None
    };

    for action in payload.updates {
        match action {
            TableUpdateAction::AddSidecarIndex { file_path, index_file } => {
                println!("Processing AddSidecarIndex...");
                if let Some(entries) = &mut all_existing_entries {
                    if let Some(entry) = entries.iter_mut().find(|e| e.file_path == file_path) {
                        entry.index_files.push(index_file);
                        new_entries.push(entry.clone());
                    } else {
                        println!("Warning: Data file not found for index attachement: {}", file_path);
                    }
                }
            },
            TableUpdateAction::RemoveSidecarIndex { index_type, column_name } => {
                println!("Processing RemoveSidecarIndex...");
                if let Some(entries) = &mut all_existing_entries {
                    // Filter and modify entries
                    for entry in entries.iter_mut() {
                        if !entry.index_files.is_empty() {
                            let old_len = entry.index_files.len();
                            entry.index_files.retain(|idx| {
                                let match_type = index_type.as_ref().is_none_or(|t| idx.index_type == *t);
                                let match_col = column_name.as_ref().is_none_or(|c| idx.column_name.as_ref() == Some(c));
                                !(match_type && match_col) // Keep if NOT matching removal criteria
                            });
                            if entry.index_files.len() < old_len {
                                // Entry modified. We treat this as a rewrite:
                                new_entries.push(entry.clone());
                            }
                        }
                    }
                }
            },
            TableUpdateAction::AddSnapshot { snapshot } => {
                println!("Processing AddSnapshot...");
                if let Some(manifest_list) = snapshot.get("manifest-list").and_then(|s| s.as_str()) {
                     println!("Fetching manifest list: {}", manifest_list);
                     let ml_path = object_store::path::Path::from(manifest_list);
                     // Try reading
                     let res = store.get(&ml_path).await;
                     
                     if let Ok(get_res) = res {
                         let bytes = get_res.bytes().await.unwrap_or_default();
                    if let Ok(list) = hyperstreamdb::core::iceberg::read_manifest_list(&bytes[..]) {
                             let mut snapshot_data_entries = Vec::new();
                             let mut snapshot_delete_entries = Vec::new();

                             for entry in list {
                                 let raw_path = entry.manifest_path.as_str();
                                 println!("Processing manifest: {}", raw_path);
                                 
                                 // Handle file:// URI scheme and Relativize
                                 let root_path_str = if uri.starts_with("file://") {
                                      uri.strip_prefix("file://").unwrap_or(&uri)
                                 } else {
                                      ""
                                 };

                                 let path_no_scheme = raw_path.strip_prefix("file://").unwrap_or(raw_path);

                                 let clean_path = if !root_path_str.is_empty() && path_no_scheme.starts_with(root_path_str) {
                                     path_no_scheme.strip_prefix(root_path_str).unwrap_or(path_no_scheme).trim_start_matches('/')
                                 } else {
                                     path_no_scheme
                                 };
                                 
                                 let m_path = object_store::path::Path::from(clean_path);
                                 
                                 match store.get(&m_path).await {
                                     Ok(m_res) => {
                                         println!("Read manifest file: {}", clean_path);
                                         let m_bytes = m_res.bytes().await.unwrap_or_default();
                                         if let Ok(m_entries) = hyperstreamdb::core::iceberg::read_manifest(&m_bytes[..]) {
                                             println!("Manifest contains {} entries", m_entries.len());
                                             // Convert entries
                                             let schema = current_manifest.schemas.last().or(current_manifest.schemas.first());
                                             
                                             if let Some(s) = schema {
                                                 for ie in m_entries {
                                                     // Only add active files
                                                     if ie.status == 1 || ie.status == 0 { // ADDED or EXISTING
                                                         match hyperstreamdb::core::iceberg::convert_iceberg_to_object(&ie, s, &current_manifest.partition_spec) {
                                                             Ok(hyperstreamdb::core::iceberg::IcebergManifestObject::Data(me)) => {
                                                                 snapshot_data_entries.push(me);
                                                             },
                                                             Ok(hyperstreamdb::core::iceberg::IcebergManifestObject::Delete(df)) => {
                                                                 snapshot_delete_entries.push(df);
                                                             },
                                                             Err(e) => {
                                                                 println!("Warning: Failed to convert iceberg object: {}", e);
                                                             }
                                                         }
                                                     }
                                                 }
                                             } else {
                                                 println!("Warning: No schema available to decode stats");
                                             }
                                         } else {
                                             println!("Failed to parse Avro manifest: {}", clean_path);
                                         }
                                     },
                                     Err(e) => {
                                         println!("Failed to read manifest file {}: {}", clean_path, e);
                                     }
                                 }
                             }

                             // Associate Delete Files with Data Files
                             for data_entry in &mut snapshot_data_entries {
                                 for delete_entry in &snapshot_delete_entries {
                                     // Simple partition matching
                                     // For equality deletes: applies to all in partition
                                     // For position deletes: typically applies to specific file, but here we blindly link if partition matches
                                     // Ideally position deletes link by file_path but Iceberg V2 splits them. 
                                     // We'll link all partition-compatible deletes for now.
                                     if data_entry.partition_values == delete_entry.partition_values {
                                         data_entry.delete_files.push(delete_entry.clone());
                                     }
                                 }
                             }
                             
                             new_entries.extend(snapshot_data_entries);
                         }
                     } else {
                         println!("Failed to read manifest list: {}", manifest_list);
                     }
                }
            },
            TableUpdateAction::AddSchema { schema, last_column_id } => {
                println!("Processing AddSchema...");
                if let Ok(arrow_schema) = hyperstreamdb::core::iceberg::iceberg_json_to_arrow_schema(&schema) {
                    let id = schema.get("schema-id").and_then(|v| v.as_i64()).map(|v| v as i32)
                        .unwrap_or(current_manifest.schemas.len() as i32 + 1);
                    let new_schema = hyperstreamdb::core::manifest::Schema::from_arrow(&arrow_schema, id);
                    
                    let mut schemas = current_manifest.schemas.clone();
                    schemas.push(new_schema);
                    updated_schemas = Some(schemas);
                    if let Some(lci) = last_column_id {
                        updated_last_column_id = Some(lci);
                    }
                }
            },
            TableUpdateAction::SetCurrentSchema { schema_id } => {
                 updated_schema_id = Some(schema_id);
            },
            TableUpdateAction::AddPartitionSpec { spec } => {
                let mut specs = updated_partition_specs.clone().unwrap_or_else(|| current_manifest.partition_specs.clone());
                // Assign Schema ID if not matching? Iceberg usually sends complete object.
                // We just append.
                specs.push(spec);
                updated_partition_specs = Some(specs);
            },
            TableUpdateAction::SetDefaultSpec { spec_id } => {
                updated_default_spec_id = Some(spec_id);
            },
            TableUpdateAction::SetProperties { updates } => {
                 let mut props: std::collections::HashMap<String, String> = updated_properties.take().unwrap_or_default();
                 props.extend(updates);
                 updated_properties = Some(props);
            },
            TableUpdateAction::RemoveProperties { removals } => {
                 let mut rems: Vec<String> = removed_properties.take().unwrap_or_default();
                 rems.extend(removals);
                 removed_properties = Some(rems);
            },
            TableUpdateAction::AddSortOrder { sort_order } => {
                let mut orders = updated_sort_orders.clone().unwrap_or_else(|| current_manifest.sort_orders.clone());
                orders.push(sort_order);
                updated_sort_orders = Some(orders);
            },
            TableUpdateAction::SetDefaultSortOrder { sort_order_id } => {
                updated_default_sort_order_id = Some(sort_order_id);
            },
            TableUpdateAction::UpgradeFormatVersion { format_version } => {
                println!("Upgrading table format version to {}", format_version);
                // We track version in Manifest but mostly ignore it for now
                // Ideally update Manifest struct to hold format_version
            },
            _ => { println!("Ignoring update action: {:?}", action); }
        }
    }

    // 2. Commit
    let commit_metadata = hyperstreamdb::core::manifest::CommitMetadata {
        updated_schemas,
        updated_schema_id,
        updated_partition_specs,
        updated_default_spec_id,
        updated_properties,
        updated_sort_orders,
        updated_default_sort_order_id,
        removed_properties,
        updated_last_column_id,
        is_fast_append: false,
    };

    match manager.commit(&new_entries, &[], commit_metadata).await {
         Ok(_) => {
             // Return updated metadata
             get_table(ax_lib::extract::Path((prefix, namespace, table))).await.into_response()
         },
         Err(e) => {
             eprintln!("Failed to commit update: {}", e);
             (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": {
                    "message": format!("Commit failed: {}", e),
                    "type": "CommitFailedException",
                    "code": 500
                }
            }))).into_response()
         }
    }
}

async fn register_table(
    ax_lib::extract::Path((prefix, namespace, table)): ax_lib::extract::Path<(String, String, String)>,
    Json(payload): Json<serde_json::Value>
) -> impl IntoResponse {
    println!("Registering table {}.{}.{} with payload: {:?}", prefix, namespace, table, payload);
    // TODO: Persistence logic for external table registration
    StatusCode::ACCEPTED
}

// Alias for convenience since I noticed axum was used in gateway.rs but I might need to verify the exact module path if it changed
mod ax_lib {
    pub use axum::*;
}
