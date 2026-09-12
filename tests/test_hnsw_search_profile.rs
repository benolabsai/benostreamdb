// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Profile the HNSW-IVF search: measure the coarse (cluster selection) vs.
//! fine (HNSW graph search) split for a 100k-vector index. This identifies
//! the bottleneck for the HNSW Hot Cache optimization.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{FixedSizeListArray, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::core::index::VectorValue;
use hyperstreamdb::core::table::Table;
use tempfile::tempdir;

#[tokio::test]
async fn test_hnsw_search_profile_100k() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    table
        .add_index(
            "embedding".to_string(),
            hyperstreamdb::core::manifest::IndexAlgorithm::Hnsw {
                metric: "l2".to_string(),
                complexity: 16,
                quality: 200,
                build_device: None,
                search_device: None,
            },
        )
        .await?;

    let dim = 64usize;
    let n = 100_000usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim as i32),
            false,
        ),
    ]));

    // Generate 100k random 64-dim vectors.
    let mut rng = 42u64;
    let next_f32 = |rng: &mut u64| -> f32 {
        // Simple LCG for reproducibility.
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*rng >> 33) as f32) / (1u64 << 31) as f32
    };

    let ids: Vec<i32> = (0..n as i32).collect();
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..n)
        .map(|_| {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(Some(next_f32(&mut rng)));
            }
            Some(v)
        })
        .collect();

    let t_build_start = Instant::now();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vectors,
                dim as i32,
            )),
        ],
    )?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    let t_build = t_build_start.elapsed();

    // Clear caches so the first search loads the index from disk.
    hyperstreamdb::core::cache::HNSW_IVF_CACHE.invalidate_all();

    // Query vector (random).
    let mut query = Vec::with_capacity(dim);
    for _ in 0..dim {
        query.push(next_f32(&mut rng));
    }

    // Warm-up (loads the index into HNSW_IVF_CACHE).
    let t_warm = Instant::now();
    let _ = table
        .query()
        .vector_search("embedding", VectorValue::Float32(query.clone()), 10)
        .to_batches()
        .await?;
    let t_warmup = t_warm.elapsed();

    // Timed searches (index is now cached).
    let n_runs = 20;
    let t_search_start = Instant::now();
    for _ in 0..n_runs {
        let _ = table
            .query()
            .vector_search("embedding", VectorValue::Float32(query.clone()), 10)
            .to_batches()
            .await?;
    }
    let t_search_total = t_search_start.elapsed();
    let t_search_avg = t_search_total / n_runs as u32;

    println!("\n=== HNSW Search Profile (100k vectors, dim 64) ===");
    println!("Build + commit: {:?}", t_build);
    println!("Warm-up (load index): {:?}", t_warmup);
    println!("Avg search (cached, {} runs): {:?}", n_runs, t_search_avg);
    println!("============================================\n");

    Ok(())
}
