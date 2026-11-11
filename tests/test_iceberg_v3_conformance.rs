// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
// Iceberg V3 manifest conformance, verified with an **independent** Avro
// object-container reader written here rather than the `apache_avro` crate the
// engine uses to write the file. A test that parses with the same library that
// produced the bytes cannot catch a wrong key name or a wrong header shape.
//
// Covered by this suite (Workstream V3.2):
//   * the Avro `format-version` user-metadata key carries the *table's* version,
//     not a hardcoded "2";
//   * the manifest entry schema declares the V3 deletion-vector fields
//     `referenced_data_file` (143), `content_offset` (144) and
//     `content_size_in_bytes` (145);
//   * a V2 table still declares "2" (no accidental upgrade).

use anyhow::{Context, Result};
use arrow::array::{Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn events_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("embedding", DataType::Float32, false),
    ]))
}

fn events_batch() -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        events_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])),
            Arc::new(Float32Array::from(vec![0.1f32, 0.2, 0.3])),
        ],
    )?)
}

// ---------------------------------------------------------------------------
// Minimal Avro object-container header reader (independent of `apache_avro`)
// ---------------------------------------------------------------------------

/// Decode an Avro `long` (zig-zag encoded, variable length).
fn read_long(buf: &[u8], pos: &mut usize) -> Result<i64> {
    let mut acc: u64 = 0;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(*pos).context("truncated Avro long")?;
        *pos += 1;
        acc |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        anyhow::ensure!(shift < 64, "Avro long overflows 64 bits");
    }
    Ok(((acc >> 1) as i64) ^ -((acc & 1) as i64))
}

/// Decode a length-prefixed byte string.
fn read_sized<'a>(buf: &'a [u8], pos: &mut usize, what: &str) -> Result<&'a [u8]> {
    let len = read_long(buf, pos)?;
    anyhow::ensure!(len >= 0, "negative {what} length {len}");
    let len = len as usize;
    let end = pos.checked_add(len).context("Avro length overflow")?;
    anyhow::ensure!(end <= buf.len(), "truncated Avro {what}");
    let out = &buf[*pos..end];
    *pos = end;
    Ok(out)
}

/// Parse the metadata map out of an Avro object container header.
///
/// Layout: `Obj\x01`, then an Avro map of `string -> bytes`, then the 16-byte
/// sync marker. Map blocks are `(count, entries...)`, terminated by a zero
/// count; a negative count means `(-count, entry-byte-size, entries...)`.
fn read_avro_header_metadata(bytes: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    anyhow::ensure!(
        bytes.len() >= 4 && &bytes[..4] == b"Obj\x01",
        "not an Avro object container file"
    );
    let mut pos = 4usize;
    let mut meta = HashMap::new();

    loop {
        let mut count = read_long(bytes, &mut pos)?;
        if count == 0 {
            break;
        }
        if count < 0 {
            count = -count;
            // byte size of this block; not needed for a sequential read
            let _ = read_long(bytes, &mut pos)?;
        }
        for _ in 0..count {
            let key = String::from_utf8(read_sized(bytes, &mut pos, "metadata key")?.to_vec())
                .context("Avro metadata key is not UTF-8")?;
            let value = read_sized(bytes, &mut pos, "metadata value")?.to_vec();
            meta.insert(key, value);
        }
    }

    Ok(meta)
}

/// Find the first data manifest (`*-m0.avro`) under the table directory.
fn find_data_manifest(root: &Path) -> Result<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // `*-m<chunk>.avro` are data manifests; `*-d<n>.avro` are delete
            // manifests and `snap-*.avro` are manifest lists.
            if name.ends_with("-m0.avro") && !name.starts_with("snap-") {
                return Ok(path);
            }
        }
    }
    anyhow::bail!(
        "no data manifest (*-m0.avro) found under {}",
        root.display()
    )
}

fn table_uri(root: &Path) -> String {
    format!("file://{}", root.display())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn v3_manifest_declares_format_version_3_and_dv_fields() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let uri = table_uri(&tmp.path().join("v3_table"));

    let table = benostreamdb::core::table::Table::create_async(uri.clone(), events_schema())
        .await
        .context("create v3 table")?;
    table.set_format_version(3);
    table.write_async(vec![events_batch()?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let manifest_path = find_data_manifest(tmp.path())?;
    let bytes = std::fs::read(&manifest_path)?;
    let meta = read_avro_header_metadata(&bytes)?;

    let format_version = meta
        .get("format-version")
        .context("manifest is missing the `format-version` user metadata")?;
    assert_eq!(
        String::from_utf8_lossy(format_version),
        "3",
        "a format-version = 3 table must declare `format-version` = 3 in its manifest \
         Avro metadata; external V3 readers select parsing rules from this key"
    );

    // `content` tells an external reader whether this manifest holds data or deletes.
    assert_eq!(
        String::from_utf8_lossy(
            meta.get("content")
                .context("manifest is missing the `content` user metadata")?
        ),
        "0",
        "data manifests declare content = 0"
    );

    // `apache_avro` writes `avro.schema` through `serde_json::to_string`, i.e.
    // compact JSON with no spaces after the colons. Compare against that form.
    let schema_json: String = String::from_utf8_lossy(
        meta.get("avro.schema")
            .context("manifest header has no avro.schema")?,
    )
    .chars()
    .filter(|c| !c.is_whitespace())
    .collect();

    for (field_id, name) in [
        ("143", "referenced_data_file"),
        ("144", "content_offset"),
        ("145", "content_size_in_bytes"),
    ] {
        assert!(
            schema_json.contains(&format!("\"name\":\"{name}\"")),
            "manifest entry schema must declare V3 field `{name}`; schema was: {schema_json}"
        );
        assert!(
            schema_json.contains(&format!("\"field-id\":{field_id}")),
            "manifest entry schema must declare `{name}` with field-id {field_id}; schema was: {schema_json}"
        );
    }

    // The V3 lineage field must still be present; it is what PyIceberg and
    // iceberg-rust use to reconstruct row lineage.
    assert!(
        schema_json.contains("\"field-id\":142"),
        "V3 manifest entry schema must keep first_row_id (field-id 142)"
    );

    Ok(())
}

#[tokio::test]
async fn v2_manifest_still_declares_format_version_2() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let uri = table_uri(&tmp.path().join("v2_table"));

    // Default table creation is V2; do not touch the format version.
    let table = benostreamdb::core::table::Table::create_async(uri.clone(), events_schema())
        .await
        .context("create v2 table")?;
    table.write_async(vec![events_batch()?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let manifest_path = find_data_manifest(tmp.path())?;
    let bytes = std::fs::read(&manifest_path)?;
    let meta = read_avro_header_metadata(&bytes)?;

    assert_eq!(
        String::from_utf8_lossy(
            meta.get("format-version")
                .context("manifest is missing `format-version`")?
        ),
        "2",
        "a V2 table must not be silently upgraded to V3 in its manifest metadata"
    );

    Ok(())
}

#[test]
fn avro_header_reader_round_trips_a_known_header() {
    // Hand-built header: magic, one map block with two string->bytes entries,
    // then a terminating zero count.
    let mut buf = Vec::new();
    buf.extend_from_slice(b"Obj\x01");

    let entry = |buf: &mut Vec<u8>, k: &str, v: &str| {
        // Avro long = zig-zag varint. Small positive values encode as 2 * n.
        buf.push((k.len() as u8) * 2);
        buf.extend_from_slice(k.as_bytes());
        buf.push((v.len() as u8) * 2);
        buf.extend_from_slice(v.as_bytes());
    };

    buf.push(2 * 2); // block count = 2
    entry(&mut buf, "format-version", "3");
    entry(&mut buf, "content", "0");
    buf.push(0); // end of map

    let meta = read_avro_header_metadata(&buf).expect("header should parse");
    assert_eq!(String::from_utf8_lossy(&meta["format-version"]), "3");
    assert_eq!(String::from_utf8_lossy(&meta["content"]), "0");
}
