// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! JSON-path inverted index: a rebuildable overlay over selected JSON paths in
//! a `Utf8` column.
//!
//! Each configured path (e.g. `$.user.id`) is extracted per row and indexed as
//! `(path, value) -> row_ids`, so equality/containment filters on those paths
//! avoid a full scan. The index is a Parquet file with schema
//! `(path: Utf8, value: Utf8, row_ids: List<UInt32>)`, following the same
//! Overlay Invariant as the scalar/BM25 indexes: it is advisory and rebuildable
//! from the Parquet data.
//!
//! Path syntax is the same subset as the `json_*` UDFs: an optional `$` root,
//! then `.key` / bare `key` segments and `[index]` subscripts.

use anyhow::{Context, Result};
use arrow::array::{Array, ListBuilder, StringBuilder, StringArray, UInt32Builder};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use roaring::RoaringBitmap;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::sync::Arc;

/// One step of a parsed JSON path.
#[derive(Debug, PartialEq, Eq)]
enum PathStep {
    Key(String),
    Index(usize),
}

/// Normalize a path to the canonical `$.a.b` form used as the index key, so a
/// configured path and a query path always agree.
pub fn normalize_path(path: &str) -> String {
    let p = path.trim();
    if p.is_empty() || p == "$" {
        return "$".to_string();
    }
    if p.starts_with('$') {
        p.to_string()
    } else {
        format!("$.{}", p)
    }
}

/// Parse a JSON path into steps (`$.a.b[0]`, `a.b[0]`, `$[0]`).
fn parse_path(path: &str) -> Vec<PathStep> {
    let mut steps = Vec::new();
    let mut chars = path.chars().peekable();
    if chars.peek() == Some(&'$') {
        chars.next();
    }
    let mut buf = String::new();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if !buf.is_empty() {
                    steps.push(PathStep::Key(std::mem::take(&mut buf)));
                }
            }
            '[' => {
                if !buf.is_empty() {
                    steps.push(PathStep::Key(std::mem::take(&mut buf)));
                }
                let mut idx = String::new();
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                    idx.push(c2);
                }
                let idx = idx.trim().trim_matches(|c| c == '"' || c == '\'');
                match idx.parse::<usize>() {
                    Ok(i) => steps.push(PathStep::Index(i)),
                    Err(_) => steps.push(PathStep::Key(idx.to_string())),
                }
            }
            _ => buf.push(c),
        }
    }
    if !buf.is_empty() {
        steps.push(PathStep::Key(buf));
    }
    steps
}

/// Resolve `steps` against `root`, returning the value if present.
///
/// A `Key` resolves against an object key or, when the current value is an
/// array, a numeric index — matching the `json_extract_path` UDF so the index
/// and the filter agree on what a path means.
fn resolve<'a>(root: &'a Value, steps: &[PathStep]) -> Option<&'a Value> {
    let mut cur = root;
    for step in steps {
        cur = match step {
            PathStep::Key(k) => match cur {
                Value::Object(o) => o.get(k)?,
                Value::Array(a) => a.get(k.parse::<usize>().ok()?)?,
                _ => return None,
            },
            PathStep::Index(i) => cur.get(*i)?,
        };
    }
    Some(cur)
}

/// Sentinel value recorded for a path that is present but whose value is not a
/// scalar (object/array/null). It makes `json_path_exists` a superset lookup:
/// every row where the path resolves is indexed, regardless of value type.
pub const PRESENT_MARKER: &str = "\u{0}__present__";

/// The indexed value form: strings are unquoted, everything else is JSON text.
fn value_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(_) | Value::Bool(_) => Some(v.to_string()),
        // Objects/arrays/null are not indexed as scalar values.
        _ => None,
    }
}

/// Record `row_id` under `(path, value)` for a resolved value, plus the
/// presence marker. Arrays are flattened one level so membership predicates
/// (`json_contains`/`json_exists` against an array) remain a superset of the
/// true matches — the index is advisory and the filter is re-applied above the
/// scan, so a superset is required for correctness.
fn index_resolved(
    postings: &mut BTreeMap<(String, String), Vec<u32>>,
    path: &str,
    v: &Value,
    row_id: u32,
) {
    postings
        .entry((path.to_string(), PRESENT_MARKER.to_string()))
        .or_default()
        .push(row_id);
    match v {
        Value::Array(a) => {
            for el in a {
                if let Some(text) = value_text(el) {
                    postings
                        .entry((path.to_string(), text))
                        .or_default()
                        .push(row_id);
                }
            }
        }
        other => {
            if let Some(text) = value_text(other) {
                postings
                    .entry((path.to_string(), text))
                    .or_default()
                    .push(row_id);
            }
        }
    }
}

/// Build the `(path, value, row_ids)` index for a `Utf8` column.
///
/// `out_path` is the destination Parquet file. Rows whose JSON is invalid or
/// whose path is absent are simply not indexed (the query falls back to a scan
/// for those rows via the differential oracle).
pub fn build_json_path_index(
    col_array: &Arc<dyn Array>,
    paths: &[String],
    row_offset: usize,
    out_path: &std::path::Path,
) -> Result<()> {
    let casted = arrow::compute::cast(col_array, &arrow::datatypes::DataType::Utf8)
        .context("Failed to cast column to Utf8 for JSON-path indexing")?;
    let array = casted
        .as_any()
        .downcast_ref::<StringArray>()
        .context("Invalid cast")?;

    let parsed_paths: Vec<(String, Vec<PathStep>)> = paths
        .iter()
        .map(|p| (normalize_path(p), parse_path(p)))
        .collect();

    // (path, value) -> sorted row ids.
    let mut postings: BTreeMap<(String, String), Vec<u32>> = BTreeMap::new();
    for (i, val) in array.iter().enumerate() {
        let Some(json) = val else { continue };
        let Ok(v) = serde_json::from_str::<Value>(json) else {
            continue;
        };
        let row_id = (row_offset + i) as u32;
        // Root value under the synthetic `$` path, so `json_exists` /
        // `json_contains` against a root array or scalar can use the index.
        index_resolved(&mut postings, "$", &v, row_id);
        for (path, steps) in &parsed_paths {
            if let Some(found) = resolve(&v, steps) {
                index_resolved(&mut postings, path, found, row_id);
            }
        }
    }

    let mut path_builder = StringBuilder::new();
    let mut value_builder = StringBuilder::new();
    let mut list_builder = ListBuilder::new(UInt32Builder::new());
    for ((path, value), mut rows) in postings {
        rows.sort_unstable();
        rows.dedup();
        path_builder.append_value(path);
        value_builder.append_value(value);
        for r in rows {
            list_builder.values().append_value(r);
        }
        list_builder.append(true);
    }

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("path", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new(
            "row_ids",
            arrow::datatypes::DataType::List(Arc::new(arrow::datatypes::Field::new(
                "item",
                arrow::datatypes::DataType::UInt32,
                true,
            ))),
            false,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(path_builder.finish()),
            Arc::new(value_builder.finish()),
            Arc::new(list_builder.finish()),
        ],
    )?;

    // Record the covered paths in the footer so a query on an unindexed path
    // falls back to a full scan instead of returning an empty (subset) bitmap.
    let mut covered: Vec<String> = parsed_paths.iter().map(|(p, _)| p.clone()).collect();
    covered.push("$".to_string());
    covered.sort();
    covered.dedup();
    let covered_json = serde_json::to_string(&covered)?;

    let tmp = format!("{}.tmp", out_path.to_str().context("Invalid UTF-8 in path")?);
    let file = File::create(&tmp)?;
    let props = parquet::file::properties::WriterProperties::builder()
        .set_key_value_metadata(Some(vec![parquet::file::metadata::KeyValue {
            key: "json_paths".to_string(),
            value: Some(covered_json),
        }]))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    std::fs::rename(&tmp, out_path)?;
    Ok(())
}

/// Load the index bytes and return the row bitmap for `(path, value)`.
pub fn load_json_path_bitmap(bytes: &[u8], path: &str, value: &str) -> Result<RoaringBitmap> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .context("Failed to open JSON-path index")?
        .build()?;

    let mut bitmap = RoaringBitmap::new();
    for batch in reader {
        let batch = batch?;
        let paths = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("path column")?;
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("value column")?;
        let rows = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .context("row_ids column")?;
        for i in 0..batch.num_rows() {
            if paths.value(i) == path && values.value(i) == value {
                let list = rows.value(i);
                let ids = list
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .context("row_ids item")?;
                for j in 0..ids.len() {
                    bitmap.insert(ids.value(j));
                }
            }
        }
    }
    Ok(bitmap)
}

/// Load the index bytes and return the row bitmap for every row where `path`
/// resolves (any value, including the presence marker). Used by
/// `json_path_exists` / `json_exists`.
pub fn load_json_path_exists_bitmap(bytes: &[u8], path: &str) -> Result<RoaringBitmap> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .context("Failed to open JSON-path index")?
        .build()?;

    let mut bitmap = RoaringBitmap::new();
    for batch in reader {
        let batch = batch?;
        let paths = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("path column")?;
        let rows = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .context("row_ids column")?;
        for i in 0..batch.num_rows() {
            if paths.value(i) == path {
                let list = rows.value(i);
                let ids = list
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .context("row_ids item")?;
                for j in 0..ids.len() {
                    bitmap.insert(ids.value(j));
                }
            }
        }
    }
    Ok(bitmap)
}

/// The set of paths this index covers, read from the Parquet footer. A query
/// on a path not in this set must fall back to a full scan.
pub fn indexed_paths(bytes: &[u8]) -> Result<Vec<String>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .context("Failed to open JSON-path index")?;
    if let Some(kv) = builder.metadata().file_metadata().key_value_metadata() {
        for e in kv {
            if e.key == "json_paths" {
                if let Some(v) = &e.value {
                    if let Ok(list) = serde_json::from_str::<Vec<String>>(v) {
                        return Ok(list);
                    }
                }
            }
        }
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_resolve() {
        let v: Value = serde_json::from_str(r#"{"a":{"b":[10,20,{"c":"x"}]}}"#).unwrap();
        assert_eq!(
            resolve(&v, &parse_path("$.a.b[2].c")),
            Some(&Value::String("x".into()))
        );
        assert_eq!(resolve(&v, &parse_path("$.a.b[1]")), Some(&Value::from(20)));
        assert_eq!(resolve(&v, &parse_path("$.a.missing")), None);
    }

    #[test]
    fn value_text_unquotes_strings() {
        assert_eq!(value_text(&Value::String("error".into())), Some("error".into()));
        assert_eq!(value_text(&Value::from(500)), Some("500".into()));
        assert_eq!(value_text(&Value::Bool(true)), Some("true".into()));
        assert_eq!(value_text(&Value::Null), None);
    }

    #[test]
    fn build_and_load_round_trip() {
        let json = StringArray::from(vec![
            Some(r#"{"level":"error","user":{"id":1}}"#),
            Some(r#"{"level":"info","user":{"id":2}}"#),
            Some(r#"{"level":"error","user":{"id":3}}"#),
            None,
        ]);
        let arr: Arc<dyn Array> = Arc::new(json);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("test.jsonpath.parquet");
        build_json_path_index(
            &arr,
            &["$.level".to_string(), "$.user.id".to_string()],
            0,
            &out,
        )
        .unwrap();

        let bytes = std::fs::read(&out).unwrap();
        let bm = load_json_path_bitmap(&bytes, "$.level", "error").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![0, 2]);
        let bm = load_json_path_bitmap(&bytes, "$.user.id", "2").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![1]);
        let bm = load_json_path_bitmap(&bytes, "$.level", "missing").unwrap();
        assert!(bm.is_empty());
    }

    #[test]
    fn exists_and_array_membership_are_supersets() {
        let json = StringArray::from(vec![
            Some(r#"{"level":"error","tags":["a","b"]}"#),
            Some(r#"{"level":"info","tags":["b"]}"#),
            Some(r#"{"level":{"nested":1}}"#),
            Some(r#"["error","warn"]"#),
        ]);
        let arr: Arc<dyn Array> = Arc::new(json);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("test.jsonpath.parquet");
        build_json_path_index(
            &arr,
            &["$.level".to_string(), "$.tags".to_string()],
            0,
            &out,
        )
        .unwrap();
        let bytes = std::fs::read(&out).unwrap();

        // `json_path_exists($.level)` matches every row where the path resolves,
        // including the non-scalar object on row 2.
        let bm = load_json_path_exists_bitmap(&bytes, "$.level").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![0, 1, 2]);

        // Array membership: `$.tags` contains "b" on rows 0 and 1.
        let bm = load_json_path_bitmap(&bytes, "$.tags", "b").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![0, 1]);

        // Root array element: `json_exists(col, 'error')` on row 3.
        let bm = load_json_path_bitmap(&bytes, "$", "error").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![3]);
    }
}
