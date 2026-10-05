// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Integration tests for the PostgreSQL `json` UDFs (`json_extract_path`,
//! `json_extract_path_text`, `json_contains`, `json_exists`, `json_typeof`,
//! `json_path_exists`, `json_path_query`) through SQL.

use arrow::array::{Array, BooleanArray, ListArray, StringArray};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::sql::session::BenoStreamSession;

fn session() -> BenoStreamSession {
    BenoStreamSession::new(None)
}

fn first_string(batches: &[RecordBatch]) -> Option<String> {
    let b = batches.first()?;
    let s = b.column(0).as_any().downcast_ref::<StringArray>()?;
    if s.is_null(0) {
        None
    } else {
        Some(s.value(0).to_string())
    }
}

fn first_bool(batches: &[RecordBatch]) -> Option<bool> {
    let b = batches.first()?;
    let s = b.column(0).as_any().downcast_ref::<BooleanArray>()?;
    if s.is_null(0) {
        None
    } else {
        Some(s.value(0))
    }
}

fn first_list(batches: &[RecordBatch]) -> Option<Vec<String>> {
    let b = batches.first()?;
    let l = b.column(0).as_any().downcast_ref::<ListArray>()?;
    if l.is_null(0) {
        return None;
    }
    let vals = l.value(0);
    let s = vals.as_any().downcast_ref::<StringArray>()?;
    Some((0..s.len()).map(|i| s.value(i).to_string()).collect())
}

#[tokio::test]
async fn json_extract_path_variadic() -> anyhow::Result<()> {
    let session = session();
    // json form keeps JSON quoting for a string value.
    let (batches, _) = session
        .sql(r#"SELECT json_extract_path('{"a":{"b":[10,20,{"c":"x"}]}}', 'a', 'b', '2', 'c') AS v;"#)
        .await?;
    assert_eq!(first_string(&batches).as_deref(), Some("\"x\""));

    // Array index via a numeric path element.
    let (batches, _) = session
        .sql(r#"SELECT json_extract_path('{"a":{"b":[10,20]}}', 'a', 'b', '1') AS v;"#)
        .await?;
    assert_eq!(first_string(&batches).as_deref(), Some("20"));
    Ok(())
}

#[tokio::test]
async fn json_extract_path_text_unquotes_strings() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql(r#"SELECT json_extract_path_text('{"user":{"id":42,"name":"ada"}}', 'user', 'name') AS v;"#)
        .await?;
    assert_eq!(first_string(&batches).as_deref(), Some("ada"));

    let (batches, _) = session
        .sql(r#"SELECT json_extract_path_text('{"user":{"id":42}}', 'user', 'id') AS v;"#)
        .await?;
    assert_eq!(first_string(&batches).as_deref(), Some("42"));
    Ok(())
}

#[tokio::test]
async fn json_extract_path_returns_null_for_missing_or_invalid() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql(r#"SELECT json_extract_path('{"a":1}', 'missing') AS v;"#)
        .await?;
    assert_eq!(first_string(&batches), None);

    let (batches, _) = session
        .sql(r#"SELECT json_extract_path('not json', 'a') AS v;"#)
        .await?;
    assert_eq!(first_string(&batches), None);

    let (batches, _) = session
        .sql("SELECT json_extract_path(NULL, 'a') AS v;")
        .await?;
    assert_eq!(first_string(&batches), None);
    Ok(())
}

#[tokio::test]
async fn json_contains_is_recursive_subset() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql(
            r#"SELECT json_contains('{"level":"error","tags":["a","b"]}', '{"level":"error"}') AS v;"#,
        )
        .await?;
    assert_eq!(first_bool(&batches), Some(true));

    let (batches, _) = session
        .sql(
            r#"SELECT json_contains('{"level":"error","tags":["a","b"]}', '{"tags":["b"]}') AS v;"#,
        )
        .await?;
    assert_eq!(first_bool(&batches), Some(true));

    let (batches, _) = session
        .sql(
            r#"SELECT json_contains('{"level":"error","tags":["a","b"]}', '{"level":"warn"}') AS v;"#,
        )
        .await?;
    assert_eq!(first_bool(&batches), Some(false));
    Ok(())
}

#[tokio::test]
async fn json_exists_tests_top_level_key() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql(r#"SELECT json_exists('{"a":1,"b":2}', 'a') AS v;"#)
        .await?;
    assert_eq!(first_bool(&batches), Some(true));

    let (batches, _) = session
        .sql(r#"SELECT json_exists('{"a":1,"b":2}', 'c') AS v;"#)
        .await?;
    assert_eq!(first_bool(&batches), Some(false));

    // Array element membership.
    let (batches, _) = session
        .sql(r#"SELECT json_exists('["x","y"]', 'y') AS v;"#)
        .await?;
    assert_eq!(first_bool(&batches), Some(true));
    Ok(())
}

#[tokio::test]
async fn json_typeof_names_the_type() -> anyhow::Result<()> {
    let session = session();
    for (json, expected) in [
        (r#"{"a":1}"#, "object"),
        ("[1,2]", "array"),
        (r#""s""#, "string"),
        ("42", "number"),
        ("true", "boolean"),
        ("null", "null"),
    ] {
        let (batches, _) = session
            .sql(&format!("SELECT json_typeof('{json}') AS v;"))
            .await?;
        assert_eq!(
            first_string(&batches).as_deref(),
            Some(expected),
            "for {json}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn json_path_exists_and_query_wildcards() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql(r#"SELECT json_path_exists('{"items":[{"id":1},{"id":2}]}', '$.items[*].id') AS v;"#)
        .await?;
    assert_eq!(first_bool(&batches), Some(true));

    let (batches, _) = session
        .sql(r#"SELECT json_path_exists('{"items":[{"id":1}]}', '$.items[*].missing') AS v;"#)
        .await?;
    assert_eq!(first_bool(&batches), Some(false));

    let (batches, _) = session
        .sql(r#"SELECT json_path_query('{"items":[{"id":1},{"id":2},{"id":3}]}', '$.items[*].id') AS v;"#)
        .await?;
    assert_eq!(
        first_list(&batches),
        Some(vec!["1".to_string(), "2".to_string(), "3".to_string()])
    );
    Ok(())
}

#[tokio::test]
async fn json_udfs_filter_rows() -> anyhow::Result<()> {
    let session = session();
    let (batches, _) = session
        .sql(
            r#"SELECT id FROM (VALUES
                 (1, '{"level":"error"}'),
                 (2, '{"level":"info"}'),
                 (3, '{"level":"error","code":500}')
               ) AS t(id, payload)
               WHERE json_contains(payload, '{"level":"error"}')
               ORDER BY id;"#,
        )
        .await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 2);
    Ok(())
}

#[test]
fn json_udfs_are_registered() {
    use benostreamdb::core::sql::udf;
    let names: Vec<String> = udf::all_json_udfs()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    for expected in [
        "json_extract_path",
        "json_extract_path_text",
        "json_contains",
        "json_exists",
        "json_typeof",
        "json_path_exists",
        "json_path_query",
    ] {
        assert!(names.contains(&expected.to_string()), "missing {expected}");
    }
}
