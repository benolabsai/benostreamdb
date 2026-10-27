// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Verifies that a table written by BenoStreamDB is a valid *Iceberg* table.
//!
//! Zero-copy means an Iceberg reader (Trino's `iceberg` connector via
//! `system.register_table`, PyIceberg, Spark, ...) reads the same Parquet files
//! BenoStreamDB wrote, without exporting or rewriting anything. That only works
//! when three things are true, and this test pins all three:
//!
//! 1. `metadata/vN.metadata.json` satisfies `TableMetadataParser`
//!    (`type: struct` schemas, `schema-id`, `last-partition-id`, a non-empty
//!    `sort-orders` list, no explicit `null` parent snapshots, and a
//!    `last-sequence-number` that covers every snapshot).
//! 2. Manifest Avro files carry the user metadata `ManifestReader` reads
//!    (`schema`, `partition-spec`, `partition-spec-id`, `format-version`,
//!    `content`).
//! 3. Data files carry Parquet field ids, so columns are mapped by id rather
//!    than by position.
//!
//! Nested (vector) columns are covered because they are the most fragile case:
//! Iceberg encodes them as `{"type":"list","element-id":..,"element":"double"}`.
//!
//! Delete files are covered separately: Iceberg requires one content type per
//! manifest, so a snapshot with deletes must reference a `content = 1` manifest
//! that contains only delete entries.

use anyhow::{Context, Result};
use arrow::array::{Array, Int32Array, Int64Array, ListArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::table::merge::MergeMode;
use benostreamdb::Table;
use parquet::file::reader::FileReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn docs_batch() -> RecordBatch {
    let values = ListArray::from_iter_primitive::<arrow::datatypes::Float64Type, _, _>(vec![
        Some(vec![Some(1.0), Some(0.0)]),
        Some(vec![Some(0.0), Some(1.0)]),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["alpha", "beta"])),
            Arc::new(values),
        ],
    )
    .expect("docs batch")
}

fn latest_metadata_path(table_dir: &Path) -> Result<PathBuf> {
    let hint = std::fs::read_to_string(table_dir.join("metadata/version-hint.text"))?;
    let version: i32 = hint.trim().parse()?;
    Ok(table_dir.join(format!("metadata/v{version}.metadata.json")))
}

fn local_path(uri: &str) -> Result<PathBuf> {
    let path = uri
        .strip_prefix("file://")
        .with_context(|| format!("expected a file:// URI, got {uri}"))?;
    Ok(PathBuf::from(path))
}

fn avro_record_field<'a>(
    value: &'a apache_avro::types::Value,
    name: &str,
) -> Option<&'a apache_avro::types::Value> {
    match value {
        apache_avro::types::Value::Record(fields) => fields
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value),
        _ => None,
    }
}

fn avro_i64(value: &apache_avro::types::Value) -> Option<i64> {
    match value {
        apache_avro::types::Value::Int(v) => Some(*v as i64),
        apache_avro::types::Value::Long(v) => Some(*v),
        apache_avro::types::Value::Union(_, inner) => avro_i64(inner),
        _ => None,
    }
}

fn avro_string(value: &apache_avro::types::Value) -> Option<String> {
    match value {
        apache_avro::types::Value::String(v) => Some(v.clone()),
        apache_avro::types::Value::Union(_, inner) => avro_string(inner),
        _ => None,
    }
}

/// Freshest `_manifest/vN.json` (the engine's internal manifest, still the
/// source of truth it reads deletes back from).
fn latest_engine_manifest(table_dir: &Path) -> Result<serde_json::Value> {
    let manifest_dir = table_dir.join("_manifest");
    let (_, path) = std::fs::read_dir(&manifest_dir)?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let version = name
                .strip_prefix('v')?
                .strip_suffix(".json")?
                .parse::<u32>()
                .ok()?;
            Some((version, entry.path()))
        })
        .max_by_key(|(version, _)| *version)
        .context("a _manifest/vN.json must exist")?;
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

#[tokio::test]
async fn written_table_is_readable_as_standard_iceberg() -> Result<()> {
    let root = tempfile::tempdir()?;
    let table_dir = root.path().join("docs");
    let uri = format!("file://{}", table_dir.display());

    let table = Table::new_async(uri).await?;
    table.write_async(vec![docs_batch()]).await?;
    table.commit_async().await?;

    // ── 1. Table metadata ──────────────────────────────────────────────────
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(latest_metadata_path(&table_dir)?)?)?;

    assert_eq!(metadata["format-version"], 2);
    assert_eq!(
        metadata["last-partition-id"], 999,
        "Iceberg requires last-partition-id; 999 is the unpartitioned value"
    );
    assert!(
        !metadata["sort-orders"]
            .as_array()
            .expect("sort-orders must be an array")
            .is_empty(),
        "Iceberg rejects an empty sort-orders list"
    );

    let current_schema_id = metadata["current-schema-id"].as_i64().unwrap();
    let schema = metadata["schemas"]
        .as_array()
        .and_then(|schemas| {
            schemas
                .iter()
                .find(|s| s["schema-id"].as_i64() == Some(current_schema_id))
        })
        .expect("current schema must be present in `schemas`");
    assert_eq!(schema["type"], "struct", "Iceberg struct schema marker");
    assert!(
        schema.get("schema_id").is_none(),
        "`schema_id` must be kebab-case `schema-id`"
    );

    let vec_field = schema["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "vec")
        .expect("vec column");
    assert_eq!(vec_field["type"]["type"], "list");
    assert_eq!(vec_field["type"]["element"], "double");
    assert_eq!(
        vec_field["type"]["element-id"], 301,
        "nested ids must match the manifest schema (3 * 100 + 1)"
    );

    let snapshots = metadata["snapshots"].as_array().unwrap();
    assert!(!snapshots.is_empty());
    let max_sequence_number = snapshots
        .iter()
        .filter_map(|snapshot| snapshot["sequence-number"].as_i64())
        .max()
        .unwrap_or(0);
    assert!(
        metadata["last-sequence-number"].as_i64().unwrap() >= max_sequence_number,
        "Iceberg rejects snapshots newer than last-sequence-number"
    );
    for snapshot in snapshots {
        assert!(
            snapshot.get("parent-snapshot-id").is_none(),
            "Iceberg's parser rejects an explicit null parent-snapshot-id"
        );
        // Row lineage is a v3 snapshot field; a v2 table must not carry it.
        assert!(
            snapshot.get("first-row-id").is_none() && snapshot.get("added-rows").is_none(),
            "v3 row-lineage fields must not appear in a v2 snapshot"
        );
    }

    // ── 2. Manifest list + manifest Avro metadata ──────────────────────────
    let manifest_list_uri = snapshots[0]["manifest-list"]
        .as_str()
        .expect("manifest-list");
    let manifest_list_path = local_path(manifest_list_uri)?;
    let manifest_list_reader = apache_avro::Reader::new(std::fs::File::open(&manifest_list_path)?)?;
    let manifest_path = manifest_list_reader
        .into_iter()
        .next()
        .transpose()?
        .and_then(|record| avro_record_field(&record, "manifest_path").cloned())
        .and_then(|value| match value {
            apache_avro::types::Value::String(path) => Some(path),
            apache_avro::types::Value::Union(_, inner) => match *inner {
                apache_avro::types::Value::String(path) => Some(path),
                _ => None,
            },
            _ => None,
        })
        .expect("manifest list must contain a manifest_path");

    let manifest_reader =
        apache_avro::Reader::new(std::fs::File::open(local_path(&manifest_path)?)?)?;
    let manifest_metadata = manifest_reader.user_metadata();
    for key in [
        "schema",
        "partition-spec",
        "partition-spec-id",
        "format-version",
        "content",
    ] {
        assert!(
            manifest_metadata.contains_key(key),
            "ManifestReader requires the `{key}` Avro metadata key"
        );
    }
    let manifest_schema: serde_json::Value = serde_json::from_slice(&manifest_metadata["schema"])?;
    assert_eq!(
        manifest_schema["schema-id"], current_schema_id,
        "the manifest must describe the table's current schema"
    );

    // ── 3. Parquet field ids ───────────────────────────────────────────────
    let parquet_path = std::fs::read_dir(&table_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .expect("a data file must exist");
    let parquet_reader =
        parquet::file::reader::SerializedFileReader::new(std::fs::File::open(&parquet_path)?)?;
    let arrow_schema = parquet::arrow::parquet_to_arrow_schema(
        parquet_reader.metadata().file_metadata().schema_descr(),
        None,
    )?;

    for (name, expected_id) in [("id", "1"), ("name", "2"), ("vec", "3")] {
        let field = arrow_schema.field_with_name(name)?;
        assert_eq!(
            field.metadata().get("PARQUET:field_id").map(String::as_str),
            Some(expected_id),
            "column `{name}` must carry its Iceberg field id"
        );
    }

    let DataType::List(element) = arrow_schema.field_with_name("vec")?.data_type() else {
        panic!("vec must be a list");
    };
    assert_eq!(
        element
            .metadata()
            .get("PARQUET:field_id")
            .map(String::as_str),
        Some("301"),
        "vector elements need the nested Iceberg field id"
    );

    Ok(())
}

#[tokio::test]
async fn deletes_live_in_their_own_content_one_manifest() -> Result<()> {
    let root = tempfile::tempdir()?;
    let table_dir = root.path().join("orders");
    let uri = format!("file://{}", table_dir.display());

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let table = Table::create_async(uri, schema.clone()).await?;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E"])),
        ],
    )?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    // Writes a position delete file and commits a new snapshot.
    table.delete_async("id = 2").await?;
    // A second delete lands in a later snapshot: the first delete file must be
    // carried forward as EXISTING, not re-added.
    table.delete_async("id = 4").await?;

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(latest_metadata_path(&table_dir)?)?)?;
    let snapshot = metadata["snapshots"]
        .as_array()
        .and_then(|snapshots| snapshots.last())
        .expect("delete commit must add a snapshot");
    let list_path = local_path(snapshot["manifest-list"].as_str().expect("manifest-list"))?;
    let list_reader = apache_avro::Reader::new(std::fs::File::open(&list_path)?)?;

    let mut data_manifests = Vec::new();
    let mut delete_manifests = Vec::new();
    let mut manifest_added_deletes = 0;
    let mut manifest_existing_deletes = 0;
    for record in list_reader {
        let record = record?;
        let content = avro_record_field(&record, "content")
            .and_then(avro_i64)
            .unwrap_or(0);
        let path = avro_record_field(&record, "manifest_path")
            .and_then(avro_string)
            .expect("manifest_path");
        match content {
            0 => data_manifests.push(path),
            1 => {
                manifest_added_deletes += avro_record_field(&record, "added_data_files_count")
                    .and_then(avro_i64)
                    .unwrap_or(0);
                manifest_existing_deletes +=
                    avro_record_field(&record, "existing_data_files_count")
                        .and_then(avro_i64)
                        .unwrap_or(0);
                delete_manifests.push(path);
            }
            other => panic!("unexpected manifest content {other}"),
        }
    }

    // The snapshot's delete set must be exactly one manifest: older delete
    // manifests must not be carried forward alongside it.
    assert_eq!(
        delete_manifests.len(),
        1,
        "a snapshot must reference exactly its own delete manifest"
    );
    assert!(
        !data_manifests.is_empty(),
        "the data manifests must still be referenced"
    );

    // A data manifest must not contain delete entries any more, and its paths
    // are what delete rows have to reference.
    let mut data_file_paths = Vec::new();
    for path in &data_manifests {
        let reader = apache_avro::Reader::new(std::fs::File::open(local_path(path)?)?)?;
        for record in reader {
            let record = record?;
            let data_file = avro_record_field(&record, "data_file").expect("data_file");
            let content = avro_record_field(data_file, "content")
                .and_then(avro_i64)
                .unwrap_or(0);
            assert_eq!(content, 0, "data manifests must contain only data entries");
            if let Some(file_path) = avro_record_field(data_file, "file_path").and_then(avro_string)
            {
                data_file_paths.push(file_path);
            }
        }
    }
    assert!(
        !data_file_paths.is_empty(),
        "the data manifest must declare data files"
    );

    // The delete manifest is declared `content: 1` in Avro metadata and holds
    // only delete entries.
    let delete_path = local_path(&delete_manifests[0])?;
    let reader = apache_avro::Reader::new(std::fs::File::open(&delete_path)?)?;
    let avro_metadata = reader.user_metadata();
    assert_eq!(
        avro_metadata.get("content").map(Vec::as_slice),
        Some(&b"1"[..]),
        "ManifestReader routes entries by this key"
    );
    assert!(
        avro_metadata.contains_key("schema"),
        "delete manifests need the table schema too"
    );

    let mut added_entries = 0;
    let mut existing_entries = 0;
    for record in reader {
        let record = record?;
        let data_file = avro_record_field(&record, "data_file").expect("data_file");
        let content = avro_record_field(data_file, "content")
            .and_then(avro_i64)
            .unwrap_or(0);
        assert!(
            content == 1 || content == 2,
            "delete manifest entries must be position/equality deletes, got {content}"
        );
        // Parquet is what the spec's example writers use and what every client
        // implements; the spec permits Avro, but Parquet is the portable default.
        let format = avro_record_field(data_file, "file_format")
            .and_then(avro_string)
            .expect("file_format");
        assert_eq!(format, "PARQUET", "position deletes must be Parquet");

        // Entry status follows the spec: a file added by an earlier snapshot is
        // EXISTING (0) with a null snapshot id; a new one is ADDED (1).
        let status = avro_record_field(&record, "status")
            .and_then(avro_i64)
            .unwrap_or(1);
        match status {
            1 => {
                added_entries += 1;
                assert!(
                    avro_record_field(&record, "snapshot_id")
                        .and_then(avro_i64)
                        .is_some(),
                    "ADDED delete entries must carry a snapshot id"
                );
            }
            0 => {
                existing_entries += 1;
                assert!(
                    avro_record_field(&record, "snapshot_id")
                        .and_then(avro_i64)
                        .is_none(),
                    "EXISTING delete entries must leave snapshot_id null"
                );
            }
            other => panic!("unexpected delete manifest entry status {other}"),
        }
    }
    assert_eq!(
        added_entries, 1,
        "only the newly deleted file is ADDED in the latest snapshot"
    );
    assert_eq!(
        existing_entries, 1,
        "the earlier delete file is EXISTING, not re-added"
    );
    assert_eq!(
        manifest_added_deletes, 1,
        "the manifest-list entry must report one added delete file"
    );
    assert_eq!(
        manifest_existing_deletes, 1,
        "the manifest-list entry must report one existing delete file"
    );

    // The rows inside the delete file must name the absolute data-file URI that
    // the data manifest declares. Store-relative paths make readers silently
    // skip the delete instead of reporting an error.
    let delete_files: Vec<_> = std::fs::read_dir(&table_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("del-pos-") && name.ends_with(".parquet"))
        })
        .collect();
    assert_eq!(
        delete_files.len(),
        2,
        "one Parquet position-delete file per delete commit"
    );
    let mut referenced_paths = Vec::new();
    for delete_file in &delete_files {
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(delete_file)?,
        )?
        .build()?;
        for batch in reader {
            let batch = batch?;
            let column = batch
                .column_by_name("file_path")
                .expect("file_path column")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("file_path must be Utf8");
            for value in column.iter().flatten() {
                referenced_paths.push(value.to_string());
            }
        }
    }
    assert_eq!(
        referenced_paths.len(),
        2,
        "each delete file holds the one deleted row"
    );
    for referenced in &referenced_paths {
        assert!(
            data_file_paths.iter().any(|path| path == referenced),
            "delete row references `{referenced}`, which no data manifest declares"
        );
    }

    // BenoStreamDB's own manifest keeps the authoritative delete list, which is
    // how the engine still applies them internally.
    let engine_manifest = latest_engine_manifest(&table_dir)?;
    assert_eq!(
        engine_manifest["delete_files"]
            .as_array()
            .expect("delete_files")
            .len(),
        2,
        "the engine manifest must still record both delete files"
    );

    // A data-only append after the deletes exercises the append-only fast path:
    // the previous delete manifest must NOT be carried forward, and both delete
    // files are EXISTING (none added by the append).
    let appended = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![6])),
            Arc::new(StringArray::from(vec!["F"])),
        ],
    )?;
    table.write_async(vec![appended]).await?;
    table.commit_async().await?;

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(latest_metadata_path(&table_dir)?)?)?;
    let snapshot = metadata["snapshots"]
        .as_array()
        .and_then(|snapshots| snapshots.last())
        .expect("append must add a snapshot");
    let list_path = local_path(snapshot["manifest-list"].as_str().expect("manifest-list"))?;
    let list_reader = apache_avro::Reader::new(std::fs::File::open(&list_path)?)?;
    let mut delete_manifests = 0;
    let mut added_deletes = 0;
    let mut existing_deletes = 0;
    let mut data_manifests = 0;
    for record in list_reader {
        let record = record?;
        let content = avro_record_field(&record, "content")
            .and_then(avro_i64)
            .unwrap_or(0);
        if content == 0 {
            data_manifests += 1;
        } else if content == 1 {
            delete_manifests += 1;
            added_deletes += avro_record_field(&record, "added_data_files_count")
                .and_then(avro_i64)
                .unwrap_or(0);
            existing_deletes += avro_record_field(&record, "existing_data_files_count")
                .and_then(avro_i64)
                .unwrap_or(0);
        }
    }
    assert_eq!(
        delete_manifests, 1,
        "an append must not carry the previous delete manifest forward"
    );
    assert!(
        data_manifests >= 1,
        "the data manifests must still be listed"
    );
    assert_eq!(
        added_deletes, 0,
        "an append adds no delete files, so none are ADDED"
    );
    assert_eq!(
        existing_deletes, 2,
        "both delete files are EXISTING for the append snapshot"
    );

    Ok(())
}

/// MERGE and compaction commit manifests; they must also publish a metadata
/// snapshot, or external Iceberg readers keep seeing the pre-operation state.
#[test]
fn merge_and_compaction_publish_snapshots() -> Result<()> {
    let root = tempfile::tempdir()?;
    let table_dir = root.path().join("kv");
    let uri = format!("file://{}", table_dir.display());

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("val", DataType::Utf8, false),
    ]));
    // Sync constructor: `merge`/`compact` are synchronous APIs that need the
    // table's own runtime.
    let table = Table::create(uri, schema.clone())?;
    table.write(vec![RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )?])?;
    table.commit()?;
    let snapshots_after_first_commit = latest_snapshot_count(&table_dir)?;

    // MERGE: adds delete files and replacement rows; the spec calls it overwrite.
    table.merge(
        vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![2])),
                Arc::new(StringArray::from(vec!["B"])),
            ],
        )?],
        "id",
        MergeMode::MergeOnRead,
    )?;

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(latest_metadata_path(&table_dir)?)?)?;
    let snapshot = metadata["snapshots"]
        .as_array()
        .and_then(|snapshots| snapshots.last())
        .expect("merge must publish a snapshot");
    assert_eq!(
        snapshot["summary"]["operation"], "overwrite",
        "MERGE adds and removes files, so the spec operation is `overwrite`"
    );
    assert_eq!(
        latest_snapshot_count(&table_dir)?,
        snapshots_after_first_commit + 1,
        "merge must publish exactly one new snapshot"
    );
    assert_snapshot_points_at_engine_manifest(&table_dir, snapshot)?;

    // Produce a couple more segments so compaction has work to do, then compact.
    for ids in [vec![4_i64], vec![5_i64]] {
        let val: String = ids[0].to_string();
        table.write(vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(vec![val])),
            ],
        )?])?;
        table.commit()?;
    }
    let snapshots_before_compaction = latest_snapshot_count(&table_dir)?;
    table.compact(None)?;

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(latest_metadata_path(&table_dir)?)?)?;
    let snapshot = metadata["snapshots"]
        .as_array()
        .and_then(|snapshots| snapshots.last())
        .expect("compaction must publish a snapshot");
    assert_eq!(
        snapshot["summary"]["operation"], "replace",
        "compaction replaces files without changing data, so the spec operation is `replace`"
    );
    assert_eq!(
        latest_snapshot_count(&table_dir)?,
        snapshots_before_compaction + 1,
        "compaction must publish exactly one new snapshot"
    );
    assert_snapshot_points_at_engine_manifest(&table_dir, snapshot)?;

    Ok(())
}

fn latest_snapshot_count(table_dir: &Path) -> Result<usize> {
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(latest_metadata_path(table_dir)?)?)?;
    Ok(metadata["snapshots"]
        .as_array()
        .map(|snapshots| snapshots.len())
        .unwrap_or(0))
}

/// The published snapshot must reference the manifest list of the latest
/// committed manifest; otherwise external readers see a stale file set.
fn assert_snapshot_points_at_engine_manifest(
    table_dir: &Path,
    snapshot: &serde_json::Value,
) -> Result<()> {
    let engine = latest_engine_manifest(table_dir)?;
    let engine_list = engine["manifest_list_path"]
        .as_str()
        .context("committed manifest must have a manifest list")?;
    let published = snapshot["manifest-list"]
        .as_str()
        .expect("snapshot manifest-list");
    assert_eq!(
        published.rsplit('/').next(),
        engine_list.rsplit('/').next(),
        "published snapshot must point at the committed manifest list"
    );
    Ok(())
}
