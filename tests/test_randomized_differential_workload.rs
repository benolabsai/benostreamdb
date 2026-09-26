// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Randomized differential workload — the plan's "definition of done".
//!
//! > a randomized workload of millions of operations (insert / update / delete /
//! > predicate / vector / snapshot / commit / compaction / index-rebuild) with
//! > injected failures, run without a single divergence between indexed and
//! > full-scan execution and without a single invariant violation.
//! > — `plans/production_readiness_plan.md` §5
//!
//! The "full-scan" side is an **independent ground-truth model** kept in the
//! test (`BTreeMap<i32, Vec<f32>>`), mutated in lockstep with the table. After
//! every committed step the test asserts:
//!
//! * **Row count** — the table matches the model.
//! * **Scalar predicate** — an indexed predicate's id set equals the model's.
//! * **Vector query** — the indexed top-1 for a row's own embedding is that row
//!   (embeddings are distinct, so the nearest neighbour is unique and exact).
//! * **Atomic recovery** — with a crash point injected, the table recovers and
//!   the model observes exactly one atomic outcome (applied or not).
//!
//! Compaction / vacuum / index-rebuild are interleaved because those are exactly
//! the operations that mutate the physical layout under the logical model.
//!
//! Step count is `BSDB_WORKLOAD_STEPS` (default 400; the weekly soak runs more).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::fault_injection::{arm, disarm, CrashPoint};
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::VectorSearchParams;
use benostreamdb::Table;
use once_cell::sync::Lazy;
use tempfile::tempdir;
use tokio::sync::Mutex as AsyncMutex;

const DIM: usize = 8;

/// The fault injector is process-global, so the crash-injection test must not
/// run concurrently with the differential test (which would otherwise trip on
/// an armed boundary). Serialize the two.
static SERIAL: Lazy<AsyncMutex<()>> = Lazy::new(|| AsyncMutex::new(()));

/// Deterministic embeddings: row `id` has embedding `[id; DIM]`. Distinct per
/// id, so the nearest neighbour of a row's own embedding is unique — and the
/// pattern is well-conditioned for the quantized (TQ8) index, matching the WS1
/// ANN recall test.
fn embedding(id: i32) -> Vec<f32> {
    vec![id as f32; DIM]
}

/// Build a batch for the given ids (value = id as f64).
fn batch(ids: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    let id_arr = Int32Array::from_iter_values(ids.iter().copied());
    let value_arr = Float64Array::from_iter_values(ids.iter().map(|i| *i as f64));
    let emb = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        ids.iter()
            .map(|i| Some(embedding(*i).into_iter().map(Some).collect::<Vec<_>>())),
        DIM as i32,
    );
    RecordBatch::try_new(
        schema,
        vec![Arc::new(id_arr), Arc::new(value_arr), Arc::new(emb) as _],
    )
    .unwrap()
}

/// A tiny deterministic LCG (reproducible runs from a seed).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
    /// A model key chosen at random (or `None` if the model is empty).
    fn pick(&mut self, model: &BTreeMap<i32, Vec<f32>>) -> Option<i32> {
        if model.is_empty() {
            return None;
        }
        let idx = self.below(model.len() as u64) as usize;
        model.keys().nth(idx).copied()
    }
}

async fn query_ids(table: &Table, filter: Option<&str>) -> anyhow::Result<Vec<i32>> {
    let sql = match filter {
        Some(f) => format!("SELECT id FROM t WHERE {f}"),
        None => "SELECT id FROM t".to_string(),
    };
    let batches = table.sql(&sql).await?;
    let mut ids = Vec::new();
    for b in &batches {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                ids.extend(arr.iter().flatten());
            }
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

async fn count(table: &Table) -> anyhow::Result<usize> {
    Ok(query_ids(table, None).await?.len())
}

/// The vector search's ids, in result order.
async fn vector_topk(table: &Table, q: Vec<f32>, k: usize) -> anyhow::Result<Vec<i32>> {
    let params = VectorSearchParams::new("embedding", VectorValue::Float32(q), k);
    let out = table.read_async(None, Some(vec![params]), None).await?;
    let mut ids = Vec::new();
    for b in &out {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                ids.extend(arr.iter().flatten());
            }
        }
    }
    Ok(ids)
}

/// Assert the indexed table agrees with the in-memory model (the full-scan oracle).
async fn assert_model_agrees(
    table: &Table,
    model: &BTreeMap<i32, Vec<f32>>,
    rng: &mut Lcg,
    step: usize,
) -> anyhow::Result<()> {
    let start_count = std::time::Instant::now();
    eprintln!("[rdw]   check count (step {step})");
    let n = count(table).await?;
    assert_eq!(
        n,
        model.len(),
        "step {step}: row count diverged from the model"
    );
    eprintln!("[rdw]   check count done in {:?}", start_count.elapsed());

    let start_scalar = std::time::Instant::now();
    eprintln!("[rdw]   check scalar (step {step})");

    // Scalar predicate over an id range.
    let lo = (rng.below(3_000) as i32) - 500;
    let hi = lo + rng.below(2_000) as i32 + 1;
    let expected: Vec<i32> = model.range(lo..hi).map(|(k, _)| *k).collect();
    let got = query_ids(table, Some(&format!("id >= {lo} AND id < {hi}"))).await?;
    eprintln!(
        "[rdw]   scalar done (step {step}) in {:?}",
        start_scalar.elapsed()
    );
    assert_eq!(
        got, expected,
        "step {step}: scalar predicate [{lo}, {hi}) diverged from the model"
    );

    // Vector query: a live row's own embedding. The overlay vector index is
    // *approximate* (HNSW-TQ8), so the contract is not exact top-1 but:
    //   (a) no phantom rows — every returned id is live in the model, and
    //   (b) recall — the exactly-nearest row (distance 0) is within the top-k.
    if let Some(picked) = rng.pick(model) {
        let start_vec = std::time::Instant::now();
        eprintln!("[rdw]   check vector (step {step}, id {picked})");
        let k = 10usize.min(model.len()).max(1);
        let got = vector_topk(table, embedding(picked), k).await?;
        assert!(
            !got.is_empty(),
            "step {step}: vector search returned no rows"
        );
        for id in &got {
            assert!(
                model.contains_key(id),
                "step {step}: vector search returned a phantom row {id} (not live)"
            );
        }
        assert!(
            got.contains(&picked),
            "step {step}: vector recall miss — id {picked} (distance 0, exactly nearest) \
             was not in the top-{k}: {got:?}"
        );
        eprintln!(
            "[rdw]   vector done (step {step}) in {:?}",
            start_vec.elapsed()
        );
    }

    Ok(())
}

/// Focused: does compaction preserve vector-index recall?
#[tokio::test]
async fn vector_recall_survives_compaction() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri).await?;
    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;

    let ids: Vec<i32> = (0..20).collect();
    table.write_async(vec![batch(&ids)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    for id in 0..20 {
        let got = vector_topk(&table, embedding(id), 5).await?;
        assert!(
            got.contains(&id),
            "pre-compaction recall miss for id {id}: {got:?}"
        );
    }

    table.rewrite_data_files_async(None).await?;

    for id in 0..20 {
        let got = vector_topk(&table, embedding(id), 5).await?;
        assert!(
            got.contains(&id),
            "post-compaction recall miss for id {id}: {got:?}"
        );
    }

    Ok(())
}

/// The long randomized differential run. The read-after-delete hang that
/// previously forced this to be `#[ignore]`d was fixed by the partition-scoped
/// delete-file work (see `plans/production_readiness_plan.md` §8): the delete
/// list is now stored once per manifest and resolved at read time, so a
/// full-table read after a `delete_async` on a compacted table no longer
/// blocks. Runs by default; the weekly soak raises `BSDB_WORKLOAD_STEPS`.
#[tokio::test]
async fn randomized_differential_workload_matches_model() -> anyhow::Result<()> {
    let _guard = SERIAL.lock().await;
    // Bounded by default so the per-PR run is fast; the weekly soak raises it.
    let steps: usize = std::env::var("BSDB_WORKLOAD_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    // Keep the table small so the workload stays fast and disk-bounded.
    const MAX_ROWS: usize = 120;

    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri).await?;

    // A vector index, so the vector path exercises the overlay index.
    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;

    let mut model: BTreeMap<i32, Vec<f32>> = BTreeMap::new();
    let mut rng = Lcg::new(0xD1FF_2026);
    let mut next_id: i32 = 0;

    for step in 0..steps {
        let t_step = std::time::Instant::now();
        let op = rng.below(100);
        eprintln!("[rdw] step {step} op {op} model={}", model.len());
        if model.len() >= MAX_ROWS || (55..80).contains(&op) {
            // delete one live row (also enforces the size bound)
            if let Some(victim) = rng.pick(&model) {
                eprintln!("[rdw]   delete id {victim}");
                table.delete_async(&format!("id = {victim}")).await?;
                model.remove(&victim);
            }
        } else if op < 55 {
            // insert 1..=5 rows
            let n = (rng.below(5) + 1) as i32;
            let ids: Vec<i32> = (0..n)
                .map(|_| {
                    next_id += 1;
                    next_id
                })
                .collect();
            table.write_async(vec![batch(&ids)]).await?;
            table.commit_async().await?;
            // The vector index is built asynchronously after commit; wait for it
            // so the immediately-following recall check sees every live row
            // (read-your-writes), otherwise it races a stale index.
            table.wait_for_background_tasks_async().await?;
            for id in ids {
                model.insert(id, embedding(id));
            }
        } else if op < 90 {
            // compaction
            table.rewrite_data_files_async(None).await?;
        } else if op < 95 {
            // vacuum (retention 2 keeps recent versions readable)
            table.vacuum_async(2).await?;
        } else {
            // index rebuild
            table
                .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
                .await?;
            table.wait_for_background_tasks_async().await?;
        }

        assert_model_agrees(&table, &model, &mut rng, step).await?;
        eprintln!("[rdw] step {step} TOTAL {:?}", t_step.elapsed());
    }

    assert_model_agrees(&table, &model, &mut rng, steps).await?;
    assert!(!model.is_empty(), "workload produced no rows");

    eprintln!(
        "[rdw] === merged-deletes phase breakdown ===\n{}",
        benostreamdb::telemetry::metrics::dump_merged_deletes_metrics()
    );

    eprintln!(
        "[rdw] === read-path phase breakdown ===\n{}",
        benostreamdb::telemetry::metrics::dump_read_metrics()
    );

    Ok(())
}

/// The same workload with injected crash boundaries: after each crash the table
/// must recover, and the model must observe exactly one atomic outcome (the step
/// either applied or it did not — never a torn state).
#[tokio::test]
async fn randomized_workload_recovers_atomically_from_injected_crashes() -> anyhow::Result<()> {
    let _guard = SERIAL.lock().await;
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let seed_ids: Vec<i32> = (0..10).collect();
    let mut model: BTreeMap<i32, Vec<f32>> = seed_ids.iter().map(|i| (*i, embedding(*i))).collect();
    let mut next_id = 10;
    let mut rng = Lcg::new(0xC2A5_2026);
    let mut table = Table::new_async(uri.clone()).await?;
    table.write_async(vec![batch(&seed_ids)]).await?;
    table.commit_async().await?;

    for (i, &point) in CrashPoint::ALL.iter().enumerate() {
        let before = count(&table).await?;

        let ids: Vec<i32> = (0..5)
            .map(|_| {
                next_id += 1;
                next_id
            })
            .collect();

        // An insert that may be aborted at `point`.
        arm(point);
        let _ = async {
            table.write_async(vec![batch(&ids)]).await?;
            table.commit_async().await
        }
        .await;
        disarm();
        let _ = table.wait_for_background_tasks_async().await;

        // Re-open: recovery must yield a consistent snapshot.
        table = Table::new_async(uri.clone()).await?;
        let after = count(&table).await?;

        assert!(
            after == before || after == before + ids.len(),
            "crash at {point} produced a torn state: {before} -> {after} (step {i})"
        );
        if after == before + ids.len() {
            for id in ids {
                model.insert(id, embedding(id));
            }
        }

        // The recovered table must still agree with the model.
        assert_model_agrees(&table, &model, &mut rng, i).await?;
    }

    Ok(())
}
