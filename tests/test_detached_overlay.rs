// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Integration tests for the Detached Overlay Catalog.
//!
//! These exercise mounting an external, read-only Iceberg table with a separate
//! writable overlay prefix, the persisted `detached_catalog.json` sidecar, and
//! the no-op sync path when the upstream snapshot has not advanced.

use arrow::datatypes::{DataType, Field, Schema};
use benostreamdb::Table;
use std::sync::Arc;
use tempfile::tempdir;

/// Write a minimal, snapshot-less Iceberg table metadata file under
/// `{root}/metadata/v1.metadata.json` and return the table directory URI.
fn write_minimal_iceberg_table(root: &std::path::Path) -> anyhow::Result<String> {
    let meta_dir = root.join("metadata");
    std::fs::create_dir_all(&meta_dir)?;

    let location = format!("file://{}", root.display());
    let metadata = serde_json::json!({
        "format-version": 2,
        "table-uuid": "00000000-0000-0000-0000-000000000001",
        "location": location,
        "last-sequence-number": 0,
        "last-updated-ms": 1_700_000_000_000i64,
        "current-snapshot-id": serde_json::Value::Null,
        "snapshots": [],
        "schemas": [{
            "schema-id": 0,
            "type": "struct",
            "fields": [
                {"id": 1, "name": "id", "type": "long", "required": true},
                {"id": 2, "name": "text", "type": "string", "required": false}
            ]
        }],
        "current-schema-id": 0,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "default-spec-id": 0
    });

    std::fs::write(
        meta_dir.join("v1.metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;

    Ok(location)
}

#[tokio::test]
async fn test_mount_detached_overlay_catalog() -> anyhow::Result<()> {
    let upstream = tempdir()?;
    let overlay = tempdir()?;

    let upstream_uri = write_minimal_iceberg_table(upstream.path())?;
    let overlay_uri = format!("file://{}", overlay.path().display());

    let table = Table::mount_external_iceberg(upstream_uri.clone(), overlay_uri.clone()).await?;

    // The overlay schema mirrors the upstream Iceberg schema.
    let schema = table.arrow_schema();
    assert!(schema.field_with_name("id").is_ok());
    assert!(schema.field_with_name("text").is_ok());

    // The detached catalog pins the upstream identity and snapshot.
    let catalog = table
        .detached_catalog()
        .await?
        .expect("detached catalog should be present after mount");
    assert_eq!(catalog.upstream_table_uri, upstream_uri);
    assert_eq!(catalog.overlay_storage_uri, overlay_uri);
    assert_eq!(catalog.upstream_snapshot_id, None);
    assert_eq!(catalog.upstream_schema_id, 0);
    assert!(catalog.last_synced_manifest_version >= 1);

    // The sidecar is physically written into the overlay prefix.
    assert!(overlay.path().join("detached_catalog.json").exists());

    // No upstream change -> sync is a no-op.
    assert!(!table.sync_detached_overlay_async().await?);

    Ok(())
}

#[tokio::test]
async fn test_sync_on_non_overlay_table_errors() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

    let table = Table::create_async(uri, schema).await?;

    assert!(table.detached_catalog().await?.is_none());
    assert!(
        table.sync_detached_overlay_async().await.is_err(),
        "syncing a non-overlay table must fail rather than silently no-op"
    );

    Ok(())
}
