// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
// Fixture generator for the independent `iceberg-rust` oracle
// (`tests/iceberg_rust_oracle`). This is the only bridge between the two: the
// oracle itself never links BenoStreamDB, because a reader built from the
// writer's own assumptions proves nothing.
//
// Ignored by default. Run it, then point the oracle at the printed path:
//
//   BSDB_FIXTURE_DIR=/tmp/bsdb_oracle_fixture \
//     cargo test --test oracle_fixture -- --ignored --nocapture
//
//   BSDB_ORACLE_URI=file:///tmp/bsdb_oracle_fixture/v3 \
//   BSDB_ORACLE_SUITE=v3 BSDB_ORACLE_EXPECT_IDS=1,2 \
//     cargo test --manifest-path tests/iceberg_rust_oracle/Cargo.toml -- --nocapture

use anyhow::{Context, Result};
use arrow::array::{Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("embedding", DataType::Float32, false),
    ]))
}

fn batch(ids: &[i32]) -> Result<RecordBatch> {
    let titles: Vec<String> = ids.iter().map(|i| format!("doc-{i}")).collect();
    let embeddings: Vec<f32> = ids.iter().map(|i| *i as f32 / 10.0).collect();
    Ok(RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(titles)),
            Arc::new(Float32Array::from(embeddings)),
        ],
    )?)
}

async fn build(dir: &std::path::Path, version: i32) -> Result<()> {
    let uri = format!("file://{}", dir.display());
    let table = benostreamdb::core::table::Table::create_async(uri, schema())
        .await
        .with_context(|| format!("create table at {}", dir.display()))?;
    table.set_format_version(version);
    table.write_async(vec![batch(&[1, 2, 3, 4, 5])?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    // Delete ids 3,4,5 so the oracle also proves deletes are applied.
    table.delete_async("id >= 3").await?;
    table.wait_for_background_tasks_async().await?;
    Ok(())
}

#[tokio::test]
#[ignore = "fixture generator; run explicitly to feed tests/iceberg_rust_oracle"]
async fn write_oracle_fixtures() -> Result<()> {
    let root = std::env::var("BSDB_FIXTURE_DIR")
        .unwrap_or_else(|_| "/tmp/bsdb_oracle_fixture".to_string());
    let root = std::path::PathBuf::from(root);
    std::fs::create_dir_all(&root)?;

    let v2 = root.join("v2");
    let v3 = root.join("v3");
    for dir in [&v2, &v3] {
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
        std::fs::create_dir_all(dir)?;
    }

    build(&v2, 2).await?;
    build(&v3, 3).await?;

    println!("V2 table: file://{}", v2.display());
    println!("V3 table: file://{}", v3.display());
    Ok(())
}
