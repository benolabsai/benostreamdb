// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Integration test verifying SQL binding ergonomics and type coercion without arrow_cast.
//! Tests the 6 priorities from the Python Binding Ergonomics proposal:
//! 1. UDF boundary coercion (Int64/Float64 literals coerced to UInt64/UInt32/Float32 slots).
//! 2. Graph TVFs called with raw literals and without casts.
//! 3. Table function aliases (`graph_neighbors` <-> `neighbors`, `graph_shortest_path` <-> `shortest_path`, etc.).
//! 4. Semicolon and line comment tolerance in SQL strings.
//! 5. Vector literal and distance operator ergonomics without ::vector casts.
//! 6. Helpful diagnostic error on CREATE INDEX statements.

use std::sync::Arc;
use arrow::array::{Float32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::sql::session::BenoStreamSession;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

#[tokio::test]
async fn test_sql_binding_ergonomics_no_arrow_cast() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let uri = format!("file://{}", tmp.path().display());

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);

    // Schema with Int64 node IDs (common real-world case that previously required arrow_cast)
    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Int64, false),
        Field::new("target", DataType::Int64, false),
    ]));

    let src = Int64Array::from(vec![0, 0, 1, 2, 2, 2, 3]);
    let dst = Int64Array::from(vec![1, 2, 2, 0, 1, 3, 0]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(src), Arc::new(dst)],
    )?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    let session = BenoStreamSession::new(None);
    session.register_table("edges", Arc::new(table))?;

    // 1. Priority 1: UDF boundary coercion without arrow_cast(...)
    // Int64 source/target columns, integer literals 0 and 1, integer hops 2
    let (res_neighbors, _) = session
        .sql("SELECT graph_neighbors(source, target, 0, 1) AS n FROM edges")
        .await?;
    assert!(!res_neighbors.is_empty(), "graph_neighbors should return results without arrow_cast");

    let (res_sp, _) = session
        .sql("SELECT graph_shortest_path(source, target, 0, 3) AS sp FROM edges")
        .await?;
    assert!(!res_sp.is_empty(), "graph_shortest_path should return results without arrow_cast");

    let (res_louvain, _) = session
        .sql("SELECT graph_louvain_communities(source, target, 1.0) AS comm FROM edges")
        .await?;
    assert!(!res_louvain.is_empty(), "graph_louvain_communities should work with Float64 literal 1.0");

    let (res_pr, _) = session
        .sql("SELECT graph_pagerank(source, target, 0.85, 20) AS pr FROM edges")
        .await?;
    assert!(!res_pr.is_empty(), "graph_pagerank should work without arrow_cast for damping and iterations");

    // 2. Priority 2: Table-Valued Functions (TVFs) with raw integer literals
    let (tvf_neighbors, _) = session
        .sql("SELECT * FROM graph_neighbors('edges', 0, 1)")
        .await?;
    assert!(!tvf_neighbors.is_empty(), "TVF graph_neighbors should accept integer literal 0");

    let (tvf_sp, _) = session
        .sql("SELECT * FROM graph_shortest_path('edges', 0, 3)")
        .await?;
    assert!(!tvf_sp.is_empty(), "TVF graph_shortest_path should work with integer endpoints");

    // Aliased versions of table functions
    let (tvf_sp_alias, _) = session
        .sql("SELECT * FROM shortest_path('edges', 0, 3)")
        .await?;
    assert!(!tvf_sp_alias.is_empty(), "TVF shortest_path alias should work");

    let (tvf_pr, _) = session
        .sql("SELECT * FROM graph_pagerank('edges', 0.85, 20)")
        .await?;
    assert!(!tvf_pr.is_empty(), "TVF graph_pagerank should work");

    let (tvf_louvain, _) = session
        .sql("SELECT * FROM graph_louvain('edges', 1.0)")
        .await?;
    assert!(!tvf_louvain.is_empty(), "TVF graph_louvain should work");

    // 3. Priority 6 & DDL: CREATE INDEX executes via DDL interception
    let (res_idx, _) = session
        .sql("CREATE INDEX idx_edges ON edges (source)")
        .await?;
    // Catalog DDL returns an empty batch upon executing DDL successfully
    assert!(res_idx.is_empty() || res_idx[0].num_rows() == 0);
    Ok(())
}

#[tokio::test]
async fn test_vector_ergonomics_without_casts() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let uri = format!("file://{}", tmp.path().display());

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
            false,
        ),
    ]));

    let id_arr = Int64Array::from(vec![1, 2]);
    let val_arr = Float32Array::from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let emb_arr = arrow::array::FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        3,
        Arc::new(val_arr),
        None,
    )?;

    let batch = RecordBatch::try_new(schema, vec![Arc::new(id_arr), Arc::new(emb_arr)])?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    let session = BenoStreamSession::new(None);
    session.register_table("items", Arc::new(table))?;

    // Priority 5: Distance operators with bare list literals without '::vector'
    let (res, _) = session
        .sql("SELECT id, embedding <-> [1.0, 0.0, 0.0] AS dist FROM items ORDER BY dist LIMIT 1")
        .await?;
    assert_eq!(res.len(), 1);
    let dist_col = res[0].column(1).as_any().downcast_ref::<Float32Array>().unwrap();
    assert!((dist_col.value(0) - 0.0).abs() < 1e-5);

    // Also with cosine operator <=>
    let (res_cos, _) = session
        .sql("SELECT id, embedding <=> [1.0, 0.0, 0.0] AS dist FROM items ORDER BY dist LIMIT 1")
        .await?;
    assert_eq!(res_cos.len(), 1);

    Ok(())
}

#[test]
fn test_sanitize_sql_semicolon_and_comments() {
    use benostreamdb::core::sql::session::sanitize_sql_query;

    // Semicolon at end stripped
    let cleaned = sanitize_sql_query("SELECT 1;").unwrap();
    assert_eq!(cleaned, "SELECT 1");

    // Semicolon with trailing spaces
    let cleaned2 = sanitize_sql_query("SELECT 1 ;  \n").unwrap();
    assert_eq!(cleaned2, "SELECT 1");

    // Line comments allowed
    let cleaned3 = sanitize_sql_query("-- comment line\nSELECT 1 -- inline comment\n;").unwrap();
    assert!(cleaned3.contains("SELECT 1"));
    assert!(!cleaned3.contains("--"));

    // Multiple statements rejected
    assert!(sanitize_sql_query("SELECT 1; SELECT 2;").is_err());
}
