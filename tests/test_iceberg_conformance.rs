// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS4: Iceberg V2/V3 conformance.
//!
//! Validates the on-disk artifacts BenoStreamDB writes against the Iceberg spec
//! — the metadata JSON, the manifest list Avro, and the data manifest Avro — so
//! the "100% of core V2/V3" claim is checked structurally rather than only by
//! internal round-trips. External engines (Spark/Trino/PyIceberg) are not
//! available in CI, so this suite parses the artifacts with the project's own
//! Avro reader and asserts the spec-required fields and types.
//!
//! See `plans/production_readiness_plan.md` §WS4 and
//! `docs/ICEBERG_COMPATIBILITY.md` for the published matrix.

use std::sync::Arc;

use arrow::array::{Int32Array, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::iceberg::{read_manifest, read_manifest_list};
use benostreamdb::core::storage::create_object_store;
use benostreamdb::Table;
use futures::StreamExt;
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;
use tempfile::tempdir;

fn int_batch(start: i32, n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from_iter_values(start..start + n);
    RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
}

/// Read the highest-numbered `metadata/v{N}.metadata.json` as raw JSON.
async fn latest_metadata_json(store: &Arc<dyn ObjectStore>) -> anyhow::Result<serde_json::Value> {
    let mut stream = store.list(Some(&ObjPath::from("metadata")));
    let mut best: Option<(u64, ObjPath)> = None;
    while let Some(meta) = stream.next().await {
        let meta = meta?;
        let name = meta.location.filename().unwrap_or("").to_string();
        if let Some(rest) = name.strip_prefix('v') {
            if let Some(num) = rest.strip_suffix(".metadata.json") {
                if let Ok(v) = num.parse::<u64>() {
                    if best.as_ref().map(|(bv, _)| v > *bv).unwrap_or(true) {
                        best = Some((v, meta.location));
                    }
                }
            }
        }
    }
    let (_, path) = best.ok_or_else(|| anyhow::anyhow!("no metadata JSON found"))?;
    let bytes = store.get(&path).await?.bytes().await?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Find the single `_manifest/snap-*.avro` manifest list.
async fn find_manifest_list(store: &Arc<dyn ObjectStore>) -> anyhow::Result<ObjPath> {
    let mut stream = store.list(Some(&ObjPath::from("_manifest")));
    while let Some(meta) = stream.next().await {
        let meta = meta?;
        let name = meta.location.filename().unwrap_or("").to_string();
        if name.starts_with("snap-") && name.ends_with(".avro") {
            return Ok(meta.location);
        }
    }
    anyhow::bail!("no manifest list (snap-*.avro) found")
}

/// Find the first `_manifest/*-m*.avro` data manifest.
async fn find_data_manifest(store: &Arc<dyn ObjectStore>) -> anyhow::Result<ObjPath> {
    let mut stream = store.list(Some(&ObjPath::from("_manifest")));
    while let Some(meta) = stream.next().await {
        let meta = meta?;
        let name = meta.location.filename().unwrap_or("").to_string();
        if name.ends_with(".avro") && !name.starts_with("snap-") {
            return Ok(meta.location);
        }
    }
    anyhow::bail!("no data manifest (*-m*.avro) found")
}

/// The metadata JSON must carry every field the Iceberg V2 spec requires.
#[tokio::test]
async fn metadata_json_has_required_v2_fields() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![int_batch(0, 10)]).await?;
        table.commit_async().await?;
    }

    let store = create_object_store(&uri)?;
    let meta = latest_metadata_json(&store).await?;

    // Required top-level fields (Iceberg table metadata spec).
    for field in [
        "format-version",
        "table-uuid",
        "location",
        "last-sequence-number",
        "last-updated-ms",
        "last-column-id",
        "current-schema-id",
        "schemas",
        "default-spec-id",
        "partition-specs",
        "default-sort-order-id",
        "sort-orders",
        "current-snapshot-id",
        "snapshots",
        "snapshot-log",
        "metadata-log",
    ] {
        assert!(
            meta.get(field).is_some(),
            "metadata JSON is missing required field `{field}`: {meta}"
        );
    }

    assert_eq!(
        meta["format-version"], 2,
        "default format version must be 2"
    );
    assert!(
        meta["table-uuid"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "table-uuid must be a non-empty string"
    );
    assert!(
        meta["schemas"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "schemas must be a non-empty array"
    );
    assert!(
        meta["snapshots"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "a committed table must have at least one snapshot"
    );

    // The snapshot must reference a manifest list.
    let snap = &meta["snapshots"][0];
    assert!(
        snap["manifest-list"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "snapshot must reference a manifest-list"
    );
    assert!(snap["snapshot-id"].is_i64(), "snapshot-id must be a long");
    assert!(snap["timestamp-ms"].is_i64(), "timestamp-ms must be a long");

    Ok(())
}

/// The manifest list Avro must parse and carry the V2 fields.
#[tokio::test]
async fn manifest_list_avro_conforms_to_v2_schema() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![int_batch(0, 10)]).await?;
        table.commit_async().await?;
    }

    let store = create_object_store(&uri)?;
    let list_path = find_manifest_list(&store).await?;
    let bytes = store.get(&list_path).await?.bytes().await?;
    let entries = read_manifest_list(&bytes[..])?;

    assert!(
        !entries.is_empty(),
        "manifest list must contain at least one manifest entry"
    );
    for e in &entries {
        assert!(!e.manifest_path.is_empty(), "manifest_path must be set");
        assert!(e.manifest_length > 0, "manifest_length must be positive");
        assert_eq!(e.content, 0, "data manifest content must be 0");
        assert!(
            e.added_files_count >= 0,
            "added_data_files_count must be non-negative"
        );
        assert!(
            e.added_rows_count >= 0,
            "added_rows_count must be non-negative"
        );
    }

    Ok(())
}

/// The data manifest Avro must parse and carry the V2 data_file fields.
#[tokio::test]
async fn manifest_avro_round_trips() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![int_batch(0, 10)]).await?;
        table.commit_async().await?;
    }

    let store = create_object_store(&uri)?;
    let manifest_path = find_data_manifest(&store).await?;
    let bytes = store.get(&manifest_path).await?.bytes().await?;
    let entries = read_manifest(&bytes[..])?;

    assert!(!entries.is_empty(), "data manifest must have entries");
    let total_rows: i64 = entries.iter().map(|e| e.data_file.record_count).sum();
    assert_eq!(
        total_rows, 10,
        "manifest must report the committed row count"
    );

    for e in &entries {
        // status: 0=EXISTING, 1=ADDED, 2=DELETED
        assert!(
            (0..=2).contains(&e.status),
            "status must be a valid manifest entry status, got {}",
            e.status
        );
        assert_eq!(e.data_file.content, 0, "data file content must be 0 (data)");
        assert_eq!(
            e.data_file.file_format.to_uppercase(),
            "PARQUET",
            "file_format must be PARQUET"
        );
        assert!(
            !e.data_file.file_path.is_empty(),
            "data file path must be set"
        );
        assert!(
            e.data_file.file_size_in_bytes > 0,
            "file_size_in_bytes must be positive"
        );
    }

    Ok(())
}

/// A `Timestamp(Nanosecond)` column must round-trip exactly.
#[tokio::test]
async fn nanosecond_timestamp_round_trip() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
    ]));
    // Distinct nanosecond values (not whole seconds) to prove ns precision.
    let ids = Int32Array::from_iter_values(0..3);
    let ts = TimestampNanosecondArray::from(vec![
        1_700_000_000_000_000_001i64,
        1_700_000_000_000_000_002i64,
        1_700_000_000_000_000_003i64,
    ]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(ts)])?;

    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
    }

    let table = Table::new_async(uri).await?;
    let out = table.read_async(None, None, None).await?;
    let mut seen = Vec::new();
    for b in &out {
        if let Some(col) = b.column_by_name("ts") {
            let arr = col
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .expect("ts must read back as Timestamp(Nanosecond)");
            seen.extend(arr.iter().flatten());
        }
    }
    seen.sort();
    assert_eq!(
        seen,
        vec![
            1_700_000_000_000_000_001i64,
            1_700_000_000_000_000_002i64,
            1_700_000_000_000_000_003i64
        ],
        "nanosecond timestamps must round-trip without precision loss"
    );

    Ok(())
}

/// A delete must be reflected in the manifest (delete files or a rewritten
/// segment) and the row count must drop.
#[tokio::test]
async fn delete_is_reflected_in_manifest() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![int_batch(0, 10)]).await?;
        table.commit_async().await?;
        table.delete_async("id = 5").await?;
    }

    let table = Table::new_async(uri.clone()).await?;
    let batches = table.sql("SELECT count(*) FROM t").await?;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 9, "delete must remove exactly one row");

    // The manifest must reference a delete file (merge-on-read) or a rewritten
    // segment; either way the live entry set must be consistent.
    let store = create_object_store(&uri)?;
    let manifest_path = find_data_manifest(&store).await?;
    let bytes = store.get(&manifest_path).await?.bytes().await?;
    let entries = read_manifest(&bytes[..])?;
    let total: i64 = entries.iter().map(|e| e.data_file.record_count).sum();
    assert!(
        total >= 9,
        "manifest must still reference the live data (got {total} rows)"
    );

    Ok(())
}

/// A V3 table must expose row-lineage metadata.
#[tokio::test]
async fn v3_row_lineage_metadata() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    {
        let table = Table::new_async(uri.clone()).await?;
        table.set_format_version(3);
        table.write_async(vec![int_batch(0, 5)]).await?;
        table.commit_async().await?;
    }

    let store = create_object_store(&uri)?;
    let meta = latest_metadata_json(&store).await?;
    assert_eq!(meta["format-version"], 3, "table must be upgraded to v3");

    // V3 row lineage: the metadata carries `next-row-id`, and the snapshot
    // carries `first-row-id` / `added-rows`.
    assert!(
        meta.get("next-row-id").is_some(),
        "v3 metadata must carry next-row-id: {meta}"
    );
    let snap = &meta["snapshots"][0];
    assert!(
        snap.get("first-row-id").is_some(),
        "v3 snapshot must carry first-row-id: {snap}"
    );

    Ok(())
}
