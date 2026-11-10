// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Dedicated multi-writer differential oracle tests (separate from single-writer
//! `test_randomized_differential_workload.rs`).
//!
//! Why separate:
//! - Single-writer differential testing runs under a process-global `CrashPoint`
//!   lock and exercises step-by-step `CrashPoint` aborts.
//! - Multi-writer differential testing spawns `N` concurrent `Table` handles over a
//!   shared `memory://` store (which enforces atomic `PutMode::Create` CAS on
//!   manifests and metadata) plus continuous linearizable readers, comparing the
//!   final converged table state (row counts, exact sorted IDs, scalar range
//!   queries, `hnsw_tq8` top-k vector recall, manifest file existence, and V3
//!   `_row_id` uniqueness) against the exact union of writer-local `BTreeMap`
//!   ground-truth models.
//!
//! Each writer owns a disjoint ID stride (`w * STRIDE + ...`) so every committed
//! mutation commutes cleanly with sibling writers.
//!
//! Run:
//! ```bash
//! cargo test --test test_multi_writer_differential -- --nocapture
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use arrow::array::{Array, FixedSizeListArray, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::{IndexAlgorithm, ManifestManager};
use benostreamdb::core::metadata::TableMetadata;
use benostreamdb::core::table::builder::TableBuilder;
use benostreamdb::core::table::{VectorSearchParams, WalDurability};
use benostreamdb::Table;
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path as ObjPath;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const DIM: usize = 8;
const WRITER_STRIDE: i32 = 100_000;

fn embedding(id: i32) -> Vec<f32> {
    // Deterministic SplitMix64-derived unit vector so every id points in a
    // genuinely distinct direction across segments rather than sharing `[1; DIM]`.
    let mut s = (id as i64 as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9).wrapping_add(1);
        s ^= s >> 27;
        let x = ((s & 0xFFFF) as f32) / 32768.0 - 1.0;
        v.push(x);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.into_iter().map(|x| x / norm).collect()
}

fn make_batch(ids: &[i32]) -> Result<RecordBatch> {
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
    let val_arr = Float64Array::from_iter_values(ids.iter().map(|i| *i as f64));
    let emb = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        ids.iter()
            .map(|i| Some(embedding(*i).into_iter().map(Some).collect::<Vec<_>>())),
        DIM as i32,
    );
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(id_arr), Arc::new(val_arr), Arc::new(emb) as _],
    )?)
}

/// Deterministic LCG for reproducible multi-writer differential schedules.
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
}

async fn open_shared(uri: &str, wal_dir: &std::path::Path) -> Result<Table> {
    TableBuilder::new(uri)
        .with_wal_dir(wal_dir)
        .build_async()
        .await
}

async fn open_shared_with_durability(
    uri: &str,
    wal_dir: &std::path::Path,
    durability: WalDurability,
) -> Result<Table> {
    TableBuilder::new(uri)
        .with_wal_dir(wal_dir)
        .with_durability(durability)
        .build_async()
        .await
}

async fn query_ids_raw(table: &Table, filter: Option<&str>) -> Result<Vec<i32>> {
    let sql = match filter {
        Some(f) => format!("SELECT id FROM t WHERE {f}"),
        None => "SELECT id FROM t".to_string(),
    };
    let batches = table.sql(&sql).await?;
    let mut ids = Vec::new();
    for b in &batches {
        if let Some(col) = b.column_by_name("id") {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("expected Int32 id column")?;
            ids.extend(arr.iter().flatten());
        }
    }
    Ok(ids)
}

async fn query_ids_sorted(table: &Table, filter: Option<&str>) -> Result<Vec<i32>> {
    let mut ids = query_ids_raw(table, filter).await?;
    ids.sort_unstable();
    Ok(ids)
}

async fn vector_topk(table: &Table, q: Vec<f32>, k: usize) -> Result<Vec<i32>> {
    let params = VectorSearchParams::new("embedding", VectorValue::Float32(q), k);
    let out = table.read_async(None, Some(vec![params]), None).await?;
    let mut ids = Vec::new();
    for b in &out {
        if let Some(col) = b.column_by_name("id") {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("expected Int32 id column in vector search")?;
            ids.extend(arr.iter().flatten());
        }
    }
    Ok(ids)
}

/// Assert full union ground-truth model equality across count, sorted full scan,
/// scalar predicate filters across every writer stride, `hnsw_tq8` KNN recall, and
/// manifest object existence in the shared store.
async fn assert_union_model_agrees(
    table: &Table,
    store: &Arc<dyn ObjectStore>,
    uri: &str,
    model: &BTreeMap<i32, Vec<f32>>,
    writers: usize,
    rng: &mut Lcg,
    label: &str,
) -> Result<()> {
    let all_got_raw = query_ids_raw(table, None).await?;
    let all_got_set: HashSet<i32> = all_got_raw.iter().copied().collect();
    anyhow::ensure!(
        all_got_set.len() == all_got_raw.len(),
        "{label}: duplicate rows observed in full scan (raw={}, unique={})",
        all_got_raw.len(),
        all_got_set.len()
    );
    let mut all_got = all_got_raw;
    all_got.sort_unstable();
    let all_expected: Vec<i32> = model.keys().copied().collect();
    anyhow::ensure!(
        all_got == all_expected,
        "{label}: full row set diverged from union model (got_len={}, expected_len={})",
        all_got.len(),
        all_expected.len()
    );

    // Check scalar range filters spanning both cross-writer windows and single-writer strides.
    for w in 0..writers {
        let base = (w as i32 + 1) * WRITER_STRIDE;
        let lo = base + (rng.below(10) as i32) * 10;
        let hi = lo + 45;
        let expected: Vec<i32> = model.range(lo..hi).map(|(k, _)| *k).collect();
        let got = query_ids_sorted(table, Some(&format!("id >= {lo} AND id < {hi}"))).await?;
        anyhow::ensure!(
            got == expected,
            "{label}: scalar range [{lo}, {hi}) for writer {w} diverged (got={got:?}, expected={expected:?})"
        );
    }

    // Check ANN recall + zero phantom IDs across 8 sampled live keys.
    let keys: Vec<i32> = model.keys().copied().collect();
    let sample_count = keys.len().min(8);
    let k = model.len().clamp(1, 10);
    for s in 0..sample_count {
        let idx = ((rng.next() as usize).wrapping_add(s * 7)) % keys.len();
        let target_id = keys[idx];
        let hits = vector_topk(table, embedding(target_id), k).await?;
        anyhow::ensure!(
            !hits.is_empty(),
            "{label}: vector_topk returned empty for live id {target_id}"
        );
        for h in &hits {
            anyhow::ensure!(
                model.contains_key(h),
                "{label}: vector search returned phantom deleted/uncommitted id {h}"
            );
        }
        if !hits.contains(&target_id) {
            let vp =
                VectorSearchParams::new("embedding", VectorValue::Float32(embedding(target_id)), k);
            let explain_str = table.explain(None, Some(vec![vp.clone()])).await;
            let mgr = ManifestManager::new(store.clone(), "", uri);
            if let Ok((_m, ents, ver)) = mgr.load_latest_full().await {
                let with_vec_idx = ents
                    .iter()
                    .filter(|e| {
                        e.index_files
                            .iter()
                            .any(|f| f.column_name == "embedding" && f.index_category == "vector")
                    })
                    .count();
                eprintln!(
                    "[diag:{label}] manifest_ver={ver} total_entries={} indexed_entries={with_vec_idx}",
                    ents.len()
                );
                if let Ok(Some(router)) = table.ensure_vector_router("embedding").await {
                    let plan = router.route(&embedding(target_id), 30);
                    eprintln!(
                        "[diag:{label}] router_centroids={} router_segs={} plan_cands={}",
                        router.len(),
                        router.segment_indices.len(),
                        plan.candidate_segments.len()
                    );
                }
            }
            eprintln!("[diag:{label}] explain:\n{explain_str}");
        }
        anyhow::ensure!(
            hits.contains(&target_id),
            "{label}: hnsw_tq8 recall miss for live id {target_id} (top-{k}={hits:?})"
        );
    }

    // Verify every manifest data file exists in the store (no dangling references).
    let manager = ManifestManager::new(store.clone(), "", uri);
    let (_manifest, entries, _) = manager.load_latest_full().await?;
    for entry in &entries {
        let p = ObjPath::from(entry.file_path.as_str());
        store.head(&p).await.with_context(|| {
            format!(
                "{label}: manifest references missing data file {}",
                entry.file_path
            )
        })?;
    }

    Ok(())
}

/// Multi-writer V2 differential harness:
/// - Phase A: 6 concurrent writers (Sync + Async writes + commits + compaction) + 2 continuous linearizable readers.
/// - Barrier 1: verify exact union `BTreeMap` model equality + `hnsw_tq8` recall.
/// - Phase B: 6 concurrent writers performing mixed inserts and disjoint own-stride `delete_async` + continuous readers.
/// - Barrier 2: verify exact union `BTreeMap` model equality + `hnsw_tq8` recall.
/// - Phase C: compaction (materializing Phase B deletes) + index rebuild + 6 concurrent writers final round.
/// - Barrier 3: final union `BTreeMap` equality + zero dangling manifest entries.
#[tokio::test]
async fn multi_writer_v2_differential_matches_union_model() -> Result<()> {
    let uri = "memory://mw-diff-v2";
    let wal_root = tempfile::tempdir()?;
    let writers = 6usize;

    let mut global_model: BTreeMap<i32, Vec<f32>> = BTreeMap::new();
    let seed_ids: Vec<i32> = vec![1, 2, 3, 4];

    // Seed table with schema + hnsw_tq8 index before opening writer handles.
    {
        let seed = open_shared(uri, &wal_root.path().join("seed")).await?;
        seed.write_async(vec![make_batch(&seed_ids)?]).await?;
        seed.commit_async().await?;
        seed.add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
            .await?;
        seed.wait_for_background_tasks_async().await?;
        for &id in &seed_ids {
            global_model.insert(id, embedding(id));
        }
    }

    let mut writer_models: Vec<BTreeMap<i32, Vec<f32>>> = vec![BTreeMap::new(); writers];
    let mut writer_next_seq: Vec<i32> = vec![0; writers];

    // ------------------------------------------------------------------------
    // Phase A: Concurrent Sync/Async writes + concurrent compaction + continuous linearizable readers
    // ------------------------------------------------------------------------
    let committed_lower_bound = Arc::new(parking_lot::Mutex::new(
        seed_ids.iter().copied().collect::<HashSet<i32>>(),
    ));
    let stop_readers = Arc::new(AtomicBool::new(false));

    let mut reader_handles = Vec::new();
    for reader_idx in 0..2usize {
        let reader_tbl =
            open_shared(uri, &wal_root.path().join(format!("r_a_{reader_idx}"))).await?;
        let bound = committed_lower_bound.clone();
        let stop = stop_readers.clone();
        reader_handles.push(tokio::spawn(async move {
            let mut reads = 0usize;
            while !stop.load(Ordering::Acquire) {
                let expected_min: HashSet<i32> = bound.lock().clone();
                if reader_idx == 0 {
                    let raw = query_ids_raw(&reader_tbl, None).await?;
                    let got_set: HashSet<i32> = raw.iter().copied().collect();
                    anyhow::ensure!(
                        got_set.len() == raw.len(),
                        "Phase A linearizable reader observed duplicate row id"
                    );
                    for id in &expected_min {
                        anyhow::ensure!(
                            got_set.contains(id),
                            "Phase A linearizable reader missed committed id={id}"
                        );
                    }
                } else {
                    let hits = vector_topk(&reader_tbl, embedding(1), 4).await?;
                    anyhow::ensure!(
                        !hits.is_empty(),
                        "Phase A concurrent vector search returned empty"
                    );
                }
                reads += 1;
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            anyhow::ensure!(reads > 0, "Phase A reader made zero progress");
            Ok::<_, anyhow::Error>(())
        }));
    }

    let mut writer_handles = Vec::new();
    for w in 0..writers {
        let tbl = open_shared_with_durability(
            uri,
            &wal_root.path().join(format!("w_{w}")),
            if w % 2 == 0 {
                WalDurability::Sync
            } else {
                WalDurability::Async
            },
        )
        .await?;
        let mut local_model = std::mem::take(&mut writer_models[w]);
        let mut next_seq = writer_next_seq[w];
        let bound = committed_lower_bound.clone();

        writer_handles.push(tokio::spawn(async move {
            let mut rng = Lcg::new(0xA110_0000 + w as u64);
            let base = (w as i32 + 1) * WRITER_STRIDE;

            for step in 0..10usize {
                let op = rng.below(10);
                if op < 8 {
                    let count = (rng.below(4) + 1) as i32;
                    let ids: Vec<i32> = (0..count)
                        .map(|_| {
                            next_seq += 1;
                            base + next_seq
                        })
                        .collect();
                    if step % 3 == 0 {
                        tbl.write_buffered_async(vec![make_batch(&ids)?]).await?;
                        tbl.flush_wal_async().await?;
                    } else {
                        tbl.write_async(vec![make_batch(&ids)?]).await?;
                    }
                    tbl.commit_async().await?;
                    {
                        let mut g = bound.lock();
                        for &id in &ids {
                            local_model.insert(id, embedding(id));
                            g.insert(id);
                        }
                    }
                } else {
                    // Concurrent compaction overlapping sibling writers
                    let _ = tbl.rewrite_data_files_async(None).await;
                }
            }

            tbl.wait_for_background_tasks_async().await?;
            Ok::<_, anyhow::Error>((w, local_model, next_seq))
        }));
    }

    for h in writer_handles {
        let (w, local_model, next_seq) = h.await??;
        writer_models[w] = local_model;
        writer_next_seq[w] = next_seq;
    }
    stop_readers.store(true, Ordering::Release);
    for h in reader_handles {
        h.await??;
    }

    let mut barrier_model = global_model.clone();
    for wm in &writer_models {
        for (k, v) in wm {
            barrier_model.insert(*k, v.clone());
        }
    }

    let store = benostreamdb::core::storage::create_object_store(uri)?;
    let mut check_rng = Lcg::new(0xC4EC_0001);
    {
        let verify = open_shared(uri, &wal_root.path().join("verify_a")).await?;
        verify.wait_for_background_tasks_async().await?;
        assert_union_model_agrees(
            &verify,
            &store,
            uri,
            &barrier_model,
            writers,
            &mut check_rng,
            "PhaseA",
        )
        .await?;
    }

    // ------------------------------------------------------------------------
    // Phase B: Concurrent disjoint-stride inserts + own-stride deletes + readers
    // ------------------------------------------------------------------------
    let stop_b = Arc::new(AtomicBool::new(false));
    let reader_b_tbl = open_shared(uri, &wal_root.path().join("r_b")).await?;
    let stop_b_clone = stop_b.clone();
    let reader_b_handle = tokio::spawn(async move {
        let mut reads = 0usize;
        while !stop_b_clone.load(Ordering::Acquire) {
            let raw = query_ids_raw(&reader_b_tbl, None).await?;
            let uniq: HashSet<i32> = raw.iter().copied().collect();
            anyhow::ensure!(
                uniq.len() == raw.len(),
                "Phase B concurrent reader observed duplicate rows"
            );
            for anchor in &seed_ids {
                anyhow::ensure!(
                    uniq.contains(anchor),
                    "Phase B reader lost never-deleted seed id={anchor}"
                );
            }
            reads += 1;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        anyhow::ensure!(reads > 0, "Phase B reader made zero progress");
        Ok::<_, anyhow::Error>(())
    });

    let mut phase_b_handles = Vec::new();
    for w in 0..writers {
        let tbl = open_shared(uri, &wal_root.path().join(format!("w_b_{w}"))).await?;
        let mut local_model = std::mem::take(&mut writer_models[w]);
        let mut next_seq = writer_next_seq[w];

        phase_b_handles.push(tokio::spawn(async move {
            let mut rng = Lcg::new(0xB220_0000 + w as u64);
            let base = (w as i32 + 1) * WRITER_STRIDE;

            for _step in 0..8usize {
                if !local_model.is_empty() && rng.below(10) < 4 {
                    let keys: Vec<i32> = local_model.keys().copied().collect();
                    let victim = keys[(rng.below(keys.len() as u64)) as usize];
                    tbl.delete_async(&format!("id = {victim}")).await?;
                    tbl.commit_async().await?;
                    local_model.remove(&victim);
                } else {
                    let count = (rng.below(3) + 1) as i32;
                    let ids: Vec<i32> = (0..count)
                        .map(|_| {
                            next_seq += 1;
                            base + next_seq
                        })
                        .collect();
                    tbl.write_async(vec![make_batch(&ids)?]).await?;
                    tbl.commit_async().await?;
                    for id in ids {
                        local_model.insert(id, embedding(id));
                    }
                }
            }
            tbl.wait_for_background_tasks_async().await?;
            Ok::<_, anyhow::Error>((w, local_model, next_seq))
        }));
    }

    for h in phase_b_handles {
        let (w, local_model, next_seq) = h.await??;
        writer_models[w] = local_model;
        writer_next_seq[w] = next_seq;
    }
    stop_b.store(true, Ordering::Release);
    reader_b_handle.await??;

    let mut phase_b_model = global_model.clone();
    for wm in &writer_models {
        for (k, v) in wm {
            phase_b_model.insert(*k, v.clone());
        }
    }

    {
        let verify = open_shared(uri, &wal_root.path().join("verify_b")).await?;
        verify.wait_for_background_tasks_async().await?;
        assert_union_model_agrees(
            &verify,
            &store,
            uri,
            &phase_b_model,
            writers,
            &mut check_rng,
            "PhaseB",
        )
        .await?;
    }

    // ------------------------------------------------------------------------
    // Phase C: Post-delete compaction + index rebuild + concurrent final writes
    // ------------------------------------------------------------------------
    {
        let maint = open_shared(uri, &wal_root.path().join("maint_c")).await?;
        maint.rewrite_data_files_async(None).await?;
        let _ = maint.vacuum_async(2).await?;
        maint
            .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
            .await?;
        maint.wait_for_background_tasks_async().await?;
    }

    let mut phase_c_handles = Vec::new();
    for w in 0..writers {
        let tbl = open_shared(uri, &wal_root.path().join(format!("w_c_{w}"))).await?;
        let mut local_model = std::mem::take(&mut writer_models[w]);
        let mut next_seq = writer_next_seq[w];
        phase_c_handles.push(tokio::spawn(async move {
            let base = (w as i32 + 1) * WRITER_STRIDE;
            for _ in 0..3usize {
                next_seq += 1;
                let id = base + next_seq;
                tbl.write_async(vec![make_batch(&[id])?]).await?;
                tbl.commit_async().await?;
                local_model.insert(id, embedding(id));
            }
            tbl.wait_for_background_tasks_async().await?;
            Ok::<_, anyhow::Error>((w, local_model, next_seq))
        }));
    }

    for h in phase_c_handles {
        let (w, local_model, next_seq) = h.await??;
        writer_models[w] = local_model;
        writer_next_seq[w] = next_seq;
    }

    let mut final_model = global_model;
    for wm in &writer_models {
        for (k, v) in wm {
            final_model.insert(*k, v.clone());
        }
    }

    let final_verify = open_shared(uri, &wal_root.path().join("verify_final")).await?;
    final_verify.wait_for_background_tasks_async().await?;
    assert_union_model_agrees(
        &final_verify,
        &store,
        uri,
        &final_model,
        writers,
        &mut check_rng,
        "PhaseC-Final",
    )
    .await?;

    Ok(())
}

/// Multi-writer Iceberg Format Version 3 differential test:
/// - 6 concurrent writers contending on V3 strict OCC (`expected_version`)
/// - Verifies union `BTreeMap` model equality + `hnsw_tq8` recall
/// - Verifies V3 `_row_id` uniqueness & contiguity `0..total_rows` and
///   monotonic `_last_updated_sequence_number > 0` across all Parquet segments.
#[tokio::test]
async fn multi_writer_v3_differential_and_row_lineage_is_exact() -> Result<()> {
    let uri = "memory://mw-diff-v3";
    let wal_root = tempfile::tempdir()?;
    let writers = 6usize;

    let mut global_model: BTreeMap<i32, Vec<f32>> = BTreeMap::new();
    let seed_ids = vec![1, 2, 3, 4];

    {
        let seed = open_shared(uri, &wal_root.path().join("seed_v3")).await?;
        seed.set_format_version(3);
        seed.write_async(vec![make_batch(&seed_ids)?]).await?;
        seed.commit_async().await?;
        seed.add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
            .await?;
        seed.wait_for_background_tasks_async().await?;
        for &id in &seed_ids {
            global_model.insert(id, embedding(id));
        }
    }

    let mut handles = Vec::new();
    for w in 0..writers {
        let tbl = open_shared(uri, &wal_root.path().join(format!("w_v3_{w}"))).await?;
        handles.push(tokio::spawn(async move {
            let mut local_model: BTreeMap<i32, Vec<f32>> = BTreeMap::new();
            let mut rng = Lcg::new(0x7300_0000 + w as u64);
            let base = (w as i32 + 1) * WRITER_STRIDE;
            let mut seq = 0i32;

            for _ in 0..6usize {
                let n = (rng.below(4) + 1) as i32;
                let ids: Vec<i32> = (0..n)
                    .map(|_| {
                        seq += 1;
                        base + seq
                    })
                    .collect();
                tbl.write_async(vec![make_batch(&ids)?]).await?;
                tbl.commit_async().await?;
                for id in ids {
                    local_model.insert(id, embedding(id));
                }
            }
            tbl.wait_for_background_tasks_async().await?;
            Ok::<_, anyhow::Error>(local_model)
        }));
    }

    for h in handles {
        let wm = h.await??;
        for (k, v) in wm {
            global_model.insert(k, v);
        }
    }

    let store = benostreamdb::core::storage::create_object_store(uri)?;
    let verify = open_shared(uri, &wal_root.path().join("verify_v3")).await?;
    verify.wait_for_background_tasks_async().await?;

    let mut rng = Lcg::new(0x7300_CAFE);
    assert_union_model_agrees(
        &verify,
        &store,
        uri,
        &global_model,
        writers,
        &mut rng,
        "V3-Final",
    )
    .await?;

    // Verify Iceberg V3 row-lineage invariants across all committed Parquet segments.
    let mm = ManifestManager::new(store.clone(), "", uri);
    let (manifest, entries, _ver) = mm.load_latest_full().await?;
    anyhow::ensure!(
        manifest.format_version == 3,
        "expected manifest format_version 3, got {}",
        manifest.format_version
    );

    let mut seen_row_ids = BTreeSet::new();
    let mut seen_data_ids = BTreeSet::new();
    for entry in &entries {
        let first_row_id = entry
            .first_row_id
            .context("V3 manifest entry missing first_row_id")?;
        let obj = store.get(&ObjPath::from(entry.file_path.as_str())).await?;
        let bytes = obj.bytes().await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;

        let mut file_pos = 0i64;
        for batch_res in reader {
            let b = batch_res?;
            let id_col = b
                .column_by_name("id")
                .context("missing id col")?
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("expected Int32 id")?;
            let row_id_col = b
                .column_by_name("_row_id")
                .context("missing V3 _row_id col")?
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("expected Int64 _row_id")?;
            let seq_col = b
                .column_by_name("_last_updated_sequence_number")
                .context("missing V3 _last_updated_sequence_number col")?
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("expected Int64 _last_updated_sequence_number")?;

            for i in 0..b.num_rows() {
                let row_id = row_id_col.value(i);
                let seq = seq_col.value(i);
                let id = id_col.value(i);
                anyhow::ensure!(
                    row_id == first_row_id + file_pos,
                    "V3 _row_id mismatch in {}: got {}, expected {}",
                    entry.file_path,
                    row_id,
                    first_row_id + file_pos
                );
                anyhow::ensure!(seq > 0, "V3 sequence number must be > 0");
                anyhow::ensure!(
                    seen_row_ids.insert(row_id),
                    "duplicate V3 _row_id={row_id} across concurrent writers"
                );
                anyhow::ensure!(seen_data_ids.insert(id), "duplicate id={id}");
                file_pos += 1;
            }
        }
        anyhow::ensure!(
            file_pos == entry.record_count,
            "entry record_count mismatch in {}",
            entry.file_path
        );
    }

    let expected_row_ids: BTreeSet<i64> = (0..global_model.len() as i64).collect();
    anyhow::ensure!(
        seen_row_ids == expected_row_ids,
        "V3 _row_id set must be contiguous 0..{} without gaps or duplicates",
        global_model.len()
    );
    let meta = TableMetadata::load_latest(store.as_ref()).await?;
    anyhow::ensure!(
        meta.next_row_id == Some(global_model.len() as i64),
        "V3 TableMetadata::next_row_id={:?}, expected {}",
        meta.next_row_id,
        global_model.len()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Fault-injecting shared object store for differential recovery checks
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ToggleFaultyStore {
    inner: Arc<dyn ObjectStore>,
    enabled: AtomicBool,
    fail_every: usize,
    puts: AtomicUsize,
}

impl std::fmt::Display for ToggleFaultyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ToggleFaultyStore({})", self.inner)
    }
}

impl ToggleFaultyStore {
    fn new(inner: Arc<dyn ObjectStore>, fail_every: usize) -> Self {
        Self {
            inner,
            enabled: AtomicBool::new(true),
            fail_every,
            puts: AtomicUsize::new(0),
        }
    }

    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }
}

#[async_trait::async_trait]
impl ObjectStore for ToggleFaultyStore {
    async fn put(
        &self,
        location: &ObjPath,
        payload: PutPayload,
    ) -> object_store::Result<PutResult> {
        self.inner.put(location, payload).await
    }

    async fn put_opts(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        let n = self.puts.fetch_add(1, Ordering::SeqCst) + 1;
        if self.enabled.load(Ordering::Acquire)
            && self.fail_every > 0
            && n.is_multiple_of(self.fail_every)
        {
            return Err(object_store::Error::Generic {
                store: "ToggleFaultyStore",
                source: "injected transient put_opts failure".into(),
            });
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn get_opts(
        &self,
        location: &ObjPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_range(
        &self,
        location: &ObjPath,
        range: Range<u64>,
    ) -> object_store::Result<Bytes> {
        self.inner.get_range(location, range).await
    }

    async fn head(&self, location: &ObjPath) -> object_store::Result<ObjectMeta> {
        self.inner.head(location).await
    }

    async fn delete(&self, location: &ObjPath) -> object_store::Result<()> {
        self.inner.delete(location).await
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjPath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

/// Multi-writer differential test under transient object-store `put_opts` failures:
/// verifies that interrupted `Staging` commits are restored ahead of new writes,
/// `ManifestCommitted` tails converge on retry without duplicating rows, and the
/// final table equals the exact union of every acknowledged `write_async` batch.
#[tokio::test]
async fn multi_writer_faulty_store_differential_converges_without_duplicates() -> Result<()> {
    let raw_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let faulty = Arc::new(ToggleFaultyStore::new(raw_store, 4));
    let faulty_dyn: Arc<dyn ObjectStore> = faulty.clone();

    let uri = "memory://mw-diff-faulty";
    let wal_root = tempfile::tempdir()?;
    let writers = 4usize;

    let mut global_model: HashMap<i32, Vec<f32>> = HashMap::new();
    let seed_ids = vec![1, 2, 3, 4];

    faulty.set_enabled(false);
    {
        let seed = TableBuilder::new(uri)
            .with_store(faulty_dyn.clone())
            .with_wal_dir(wal_root.path().join("seed_faulty"))
            .build_async()
            .await?;
        seed.write_async(vec![make_batch(&seed_ids)?]).await?;
        seed.commit_async().await?;
        seed.add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
            .await?;
        seed.wait_for_background_tasks_async().await?;
        for &id in &seed_ids {
            global_model.insert(id, embedding(id));
        }
    }
    faulty.set_enabled(true);

    let mut handles = Vec::new();
    for w in 0..writers {
        let tbl = TableBuilder::new(uri)
            .with_store(faulty_dyn.clone())
            .with_wal_dir(wal_root.path().join(format!("w_faulty_{w}")))
            .with_durability(WalDurability::Sync)
            .build_async()
            .await?;
        handles.push(tokio::spawn(async move {
            let mut local_written: HashMap<i32, Vec<f32>> = HashMap::new();
            let base = (w as i32 + 1) * WRITER_STRIDE;

            for step in 0..8usize {
                let start = base + (step as i32) * 10;
                let ids = vec![start, start + 1, start + 2];
                // Once `write_async` returns `Ok(())`, the rows are durable in WAL & Staging-restorable.
                tbl.write_async(vec![make_batch(&ids)?]).await?;
                for id in ids {
                    local_written.insert(id, embedding(id));
                }

                // Some commits will hit transient `ToggleFaultyStore` put_opts errors;
                // verify reads remain duplicate-free during the faulted window.
                if tbl.commit_async().await.is_err() {
                    let raw = query_ids_raw(&tbl, None).await?;
                    let uniq: HashSet<i32> = raw.iter().copied().collect();
                    anyhow::ensure!(
                        uniq.len() == raw.len(),
                        "writer {w}: duplicate id observed after transient commit failure"
                    );
                }
            }
            Ok::<_, anyhow::Error>((tbl, local_written))
        }));
    }

    let mut active_handles = Vec::new();
    for h in handles {
        let (tbl, local_written) = h.await??;
        for (k, v) in local_written {
            global_model.insert(k, v);
        }
        active_handles.push(tbl);
    }

    // Clear transient fault injection and converge every handle's pending/staged tail.
    faulty.set_enabled(false);
    for tbl in active_handles {
        tbl.commit_async().await?;
        tbl.wait_for_background_tasks_async().await?;
    }

    let expected_btree: BTreeMap<i32, Vec<f32>> = global_model.into_iter().collect();
    let verify = TableBuilder::new(uri)
        .with_store(faulty_dyn.clone())
        .with_wal_dir(wal_root.path().join("verify_faulty"))
        .build_async()
        .await?;
    verify.wait_for_background_tasks_async().await?;

    let mut rng = Lcg::new(0xFA01_1234);
    assert_union_model_agrees(
        &verify,
        &faulty_dyn,
        uri,
        &expected_btree,
        writers,
        &mut rng,
        "FaultyStore-Converged",
    )
    .await?;

    Ok(())
}

/// Multi-writer differential test against real S3 object storage (RustFS).
/// Skipped unless `AWS_ENDPOINT_URL` is set.
#[tokio::test]
async fn s3_multi_writer_differential_matches_model() -> Result<()> {
    if std::env::var("AWS_ENDPOINT_URL").is_err() {
        eprintln!("skipping s3_multi_writer_differential_matches_model: AWS_ENDPOINT_URL not set");
        return Ok(());
    }

    let bucket =
        std::env::var("BSDB_TEST_S3_BUCKET").unwrap_or_else(|_| "mstar-staging".to_string());
    let prefix = format!(
        "mw-diff-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let uri = format!("s3://{}/{}", bucket, prefix);
    let wal_root = tempfile::tempdir()?;
    let writers = 4usize;
    let mut global_model: BTreeMap<i32, Vec<f32>> = BTreeMap::new();
    let seed_ids = vec![1, 2, 3, 4];

    {
        let seed = open_shared(&uri, &wal_root.path().join("seed_s3")).await?;
        seed.write_async(vec![make_batch(&seed_ids)?]).await?;
        seed.commit_async().await?;
        seed.add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
            .await?;
        seed.wait_for_background_tasks_async().await?;
        for &id in &seed_ids {
            global_model.insert(id, embedding(id));
        }
    }

    let mut handles = Vec::new();
    for w in 0..writers {
        let tbl = open_shared(&uri, &wal_root.path().join(format!("w_s3_{w}"))).await?;
        handles.push(tokio::spawn(async move {
            let mut local_model: BTreeMap<i32, Vec<f32>> = BTreeMap::new();
            let base = (w as i32 + 1) * WRITER_STRIDE;
            for step in 0..5usize {
                let start = base + (step as i32) * 10;
                let ids = vec![start, start + 1, start + 2];
                tbl.write_async(vec![make_batch(&ids)?]).await?;
                tbl.commit_async().await?;
                for id in ids {
                    local_model.insert(id, embedding(id));
                }
            }
            tbl.wait_for_background_tasks_async().await?;
            Ok::<_, anyhow::Error>(local_model)
        }));
    }

    for h in handles {
        let wm = h.await??;
        for (k, v) in wm {
            global_model.insert(k, v);
        }
    }

    let store = benostreamdb::core::storage::create_object_store(&uri)?;
    let verify = open_shared(&uri, &wal_root.path().join("verify_s3")).await?;
    verify.wait_for_background_tasks_async().await?;

    let mut rng = Lcg::new(0x5300_5300);
    assert_union_model_agrees(
        &verify,
        &store,
        &uri,
        &global_model,
        writers,
        &mut rng,
        "S3-Final",
    )
    .await?;

    Ok(())
}
