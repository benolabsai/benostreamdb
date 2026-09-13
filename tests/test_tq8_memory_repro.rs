// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Memory repro for the REST KNN path against the 100k-doc TQ8 benchmark
//! table. Isolates which component (index load / index search / row fetch)
//! allocates the bulk of memory.
//!
//! Run (data must exist at /tmp/hsdbg2/bench-repro):
//!   cargo test --release --test test_tq8_memory_repro -- --nocapture

use std::sync::Arc;

use hyperstreamdb::core::index::VectorValue;
use hyperstreamdb::core::planner::VectorSearchParams;
use hyperstreamdb::core::table::Table;
use object_store::{local::LocalFileSystem, ObjectStore};

fn rss_mb() -> u64 {
    for line in std::fs::read_to_string("/proc/self/status").unwrap().lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            return v.split_whitespace().next().unwrap().parse::<u64>().unwrap() / 1024;
        }
    }
    0
}

#[tokio::test]
async fn test_rest_knn_path_memory_100k() -> anyhow::Result<()> {
    let root = "/tmp/hsdbg2/bench-repro";
    if !std::path::Path::new(root).exists() {
        eprintln!("Skipping test_rest_knn_path_memory_100k: {} does not exist", root);
        return Ok(());
    }
    let uri = format!("file://{}", root);

    let _store: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(root)?);

    let query = VectorValue::Float32(vec![0.5f32; 64]);

    let r0 = rss_mb();
    println!("baseline RSS: {} MB", r0);

    // Phase C: full REST path (search + row fetch via hot row cache)
    let table = Table::new_async(uri).await?;
    println!("table opened, RSS {} MB", rss_mb());
    let params = VectorSearchParams::new("embedding", query.clone(), 10);
    let scored = table.execute_vector_search_as_scored(params).await?;
    let r3 = rss_mb();
    println!(
        "C) REST search phase: {} scored, RSS {} -> {} MB (delta {})",
        scored.len(),
        r0,
        r3,
        r3.saturating_sub(r0)
    );
    for (i, s) in scored.iter().take(5).enumerate() {
        println!("  [{}] row_id={} score={}", i, s.row_id, s.score);
    }

    let batches = table.fetch_results_by_id(scored, None).await?;
    let r4 = rss_mb();
    let nrows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "D) REST fetch phase: {} batches ({} rows), RSS {} -> {} MB (delta {})",
        batches.len(),
        nrows,
        r3,
        r4,
        r4.saturating_sub(r3)
    );

    Ok(())
}
