// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Table-level, cluster-keyed vector index artifact.
//!
//! The per-segment index ([`crate::core::index::hnsw_ivf::HnswIvfIndex`]) is
//! owned by one Parquet file: each segment trains its own k-means, so a cluster
//! id is segment-local and a query must probe every segment to cover its true
//! neighbours. This module builds the other shape: **one** Puffin artifact, not
//! owned by any data file, keyed by *global* cluster, whose graph nodes carry
//! `(segment_idx, row_id)`.
//!
//! The decoupling that makes this possible is already in the engine: a graph
//! node identifies a row by `(segment_id, row_id)`, and
//! [`crate::core::table::Table::fetch_results_by_id`] groups results by segment
//! and reads rows by local id. So a cluster's members can be contiguous in the
//! artifact while remaining scattered across the Parquet files, and the payload
//! fetch still resolves them. The Parquet files are never rewritten.
//!
//! ## Artifact layout
//!
//! One Puffin container holds:
//!
//! | Blob | Contents |
//! |---|---|
//! | `centroids` | global centroids (Parquet list column) + metric/quantizer metadata |
//! | `segments` | `segment_idx -> segment_id` table |
//! | `cluster_{c}.hnsw.graph` | Arrow IPC HNSW graph over cluster `c`'s codes |
//! | `cluster_{c}.node_meta` | `(segment_idx, row_id)` per graph node |
//!
//! `node_meta` is the only genuinely new structure; the graph and quantizer
//! payloads are the existing ones.
//!
//! ## Memory bound
//!
//! The build never holds the whole corpus. Assignment streams Parquet batches
//! into a fixed number of hash buckets on disk (`cluster_id % BUCKETS`), and
//! each bucket is then loaded, grouped by cluster, and written out one cluster
//! at a time. Working set is `corpus_bytes / BUCKETS` plus one cluster's graph.

use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Array, ArrayRef, Float32Builder, ListBuilder, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::StreamExt;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

// `hnsw_ivf` defines its own L1/Hamming/Jaccard distance wrappers (the prelude
// exports same-named types from `hnsw_rs::dist`); the graph variants are typed
// against the `hnsw_ivf` ones, so import those explicitly to shadow the glob.
use crate::core::index::hnsw_ivf::{
    assign_to_centroids, DistHamming, DistJaccard, DistL1, HnswGraph,
};
use crate::core::index::hnsw_rs::prelude::*;
use crate::core::index::{global_centroids, Quantizer, QuantizerImpl, VectorMetric, VectorValue};
use crate::core::manifest::{IndexAlgorithm, ManifestEntry, TableIndexFile};
use crate::core::puffin::{
    PuffinReader, PuffinWriter, PUFFIN_BLOB_HNSW_IVF_CENTROIDS,
    PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META, PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS,
    PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS, PUFFIN_BLOB_VECTOR_HNSW_GRAPH,
};
use crate::core::search::ScoredResult;
use parquet::file::reader::FileReader;

/// Default number of global clusters when the caller does not choose one.
pub const DEFAULT_CLUSTER_COUNT: usize = 1024;

/// Default HNSW graph degree.
pub const DEFAULT_CLUSTER_M: usize = 16;

/// Default HNSW construction beam.
pub const DEFAULT_CLUSTER_EF_CONSTRUCTION: usize = 200;

/// Default target size for each sequential Puffin shard: 512 MiB (536,870,912 bytes).
///
/// Adheres to Apache Iceberg's standard default target file size (512 MiB)
/// and strictly respects cloud object storage single-PUT boundaries (e.g. AWS S3 5 GiB limit).
pub const DEFAULT_TARGET_SHARD_BYTES: usize = 536_870_912;

/// Get the target shard size in bytes from the environment (`BSDB_INDEX_TARGET_SHARD_BYTES`)
/// or fallback to [`DEFAULT_TARGET_SHARD_BYTES`].
pub fn target_shard_bytes() -> usize {
    std::env::var("BSDB_INDEX_TARGET_SHARD_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_TARGET_SHARD_BYTES)
}

/// Rows sampled for global k-means training.
///
/// 1–2 % of a 10M-row table. Training on a sample is what keeps the centroid
/// fit bounded; the full assignment pass still touches every row.
pub const DEFAULT_SAMPLE_BUDGET: usize = 200_000;

/// Number of on-disk hash buckets used to bound build memory.
fn build_bucket_count() -> usize {
    std::env::var("BSDB_CLUSTER_BUILD_BUCKETS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

/// Per-node locator: which data file, and which row inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMeta {
    /// Index into [`ClusterIndex::segment_table`].
    pub segment_idx: u32,
    /// Row position within that segment's Parquet file.
    pub row_id: u32,
}

/// Byte slices of one cluster's graph and node metadata inside a Puffin container.
#[derive(Debug, Clone, Copy)]
pub struct ClusterSlices {
    pub graph: (usize, usize),
    pub node_meta: (usize, usize),
}

/// A sequential Puffin shard of the cluster index.
#[derive(Clone)]
pub struct ClusterShard {
    /// Zero-based sequential shard index.
    pub shard_idx: usize,
    /// Storage path if loaded from or written to an object store.
    pub file_path: Option<String>,
    /// Range of global cluster IDs contained in this shard: [start, end] inclusive.
    pub cluster_range: (usize, usize),
    /// Container bytes (can be memory-mapped or owned).
    pub container: Bytes,
    /// Byte offsets and lengths for each cluster's blobs inside this shard.
    pub cluster_slices: HashMap<usize, ClusterSlices>,
}

struct MmapOwner(Arc<memmap2::Mmap>);

impl AsRef<[u8]> for MmapOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

/// A table-level, cluster-keyed vector index.
///
/// Holds the container shards and deserializes a cluster's graph on demand, so
/// opening the index costs one object-store read plus the centroid/segment
/// tables — not the whole graph. When the index exceeds the target shard size
/// (512 MiB default), it is sequentially partitioned into contiguous Puffin
/// shards.
pub struct ClusterIndex {
    /// Vector column this artifact indexes.
    pub column: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: VectorMetric,
    /// Global centroids, one per cluster.
    pub centroids: Vec<Vec<f32>>,
    /// Quantizer used for the graph codes, if any.
    pub quantizer: Option<QuantizerImpl>,
    /// `segment_idx -> segment_id`.
    pub segment_table: Vec<String>,
    /// Manifest version this artifact was built from.
    pub manifest_version: u64,
    /// Total indexed rows.
    pub record_count: i64,
    /// Sequential Puffin shards.
    pub shards: Vec<ClusterShard>,
    /// Mapping from global cluster_id to the containing shard index.
    pub cluster_to_shard: HashMap<usize, usize>,
    cache: moka::sync::Cache<usize, Arc<(HnswGraph, Vec<NodeMeta>)>>,
    /// Optional temporary directory backing memory-mapped build shards.
    pub _temp_dir: Option<std::path::PathBuf>,
}

impl Drop for ClusterIndex {
    fn drop(&mut self) {
        if let Some(ref p) = self._temp_dir {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}

impl std::fmt::Debug for ClusterIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterIndex")
            .field("column", &self.column)
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("clusters", &self.centroids.len())
            .field("segments", &self.segment_table.len())
            .field("shards", &self.shards.len())
            .field("record_count", &self.record_count)
            .finish()
    }
}

/// Sample vectors across every data file in a table.
///
/// Used to train global centroids for the Phase-1 validation path
/// (`Table::rebuild_indexes_with_global_centroids`). A fixed stride over the
/// whole table keeps the sample bounded and spread across segments; a prefix
/// sample would be one segment.
pub async fn sample_table_vectors(
    store: Arc<dyn ObjectStore>,
    entries: &[ManifestEntry],
    column: &str,
    max_samples: usize,
) -> Result<Vec<Vec<f32>>> {
    let total_rows: usize = entries.iter().map(|e| e.record_count.max(0) as usize).sum();
    if total_rows == 0 || max_samples == 0 {
        return Ok(Vec::new());
    }
    let stride = (total_rows / max_samples).max(1);
    let mut sample = Vec::with_capacity(max_samples.min(total_rows));
    let mut global_row = 0usize;

    for entry in entries {
        let mut on_batch = |_offset: usize, vectors: &[Vec<f32>]| -> Result<()> {
            for v in vectors {
                if global_row.is_multiple_of(stride) {
                    sample.push(v.clone());
                }
                global_row += 1;
            }
            Ok(())
        };
        stream_vector_batches(&store, &entry.file_path, column, &mut on_batch).await?;
    }
    Ok(sample)
}

impl ClusterIndex {
    /// Number of global clusters.
    pub fn cluster_count(&self) -> usize {
        self.centroids.len()
    }

    /// Number of data files covered by the artifact.
    pub fn segment_count(&self) -> usize {
        self.segment_table.len()
    }

    /// Total bytes all container shards occupy.
    pub fn size_in_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.container.len()).sum()
    }

    /// Number of sequential Puffin shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Read-only access to the sequential shards.
    pub fn shards(&self) -> &[ClusterShard] {
        &self.shards
    }

    /// Build the artifact from a table's data files.
    ///
    /// Two sequential passes over the vector column:
    ///
    /// 1. sample rows across every segment and train global centroids;
    /// 2. assign every row to its nearest centroid and hash it into an on-disk
    ///    bucket, then build one cluster at a time.
    ///
    /// When total serialized clusters exceed [`target_shard_bytes`], the artifact
    /// is sequentially partitioned into contiguous 512 MiB Puffin shards.
    /// The Parquet files are read, never written.
    pub async fn build_from_table(
        store: Arc<dyn ObjectStore>,
        entries: &[ManifestEntry],
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::build_from_table_with_target(
            store,
            entries,
            column,
            algo,
            k,
            m,
            ef_construction,
            None,
        )
        .await
    }

    /// Build the artifact from a table's data files with an optional target shard size override.
    pub async fn build_from_table_with_target(
        store: Arc<dyn ObjectStore>,
        entries: &[ManifestEntry],
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
        target_shard_bytes_override: Option<usize>,
    ) -> Result<Self> {
        if entries.is_empty() {
            anyhow::bail!("build_from_table requires at least one data file");
        }

        let segment_table: Vec<String> = entries.iter().map(segment_id_of).collect();
        let total_rows: usize = entries.iter().map(|e| e.record_count.max(0) as usize).sum();
        if total_rows == 0 {
            anyhow::bail!("build_from_table: table has no rows");
        }

        // ---- Pass 1: sample across all segments, train global centroids ----
        let stride = (total_rows / DEFAULT_SAMPLE_BUDGET).max(1);
        let mut sample: Vec<Vec<f32>> = Vec::with_capacity(DEFAULT_SAMPLE_BUDGET.min(total_rows));
        let mut dim = 0usize;
        let mut global_row = 0usize;

        for entry in entries {
            let mut on_batch = |_offset: usize, vectors: &[Vec<f32>]| -> Result<()> {
                if dim == 0 {
                    if let Some(v) = vectors.first() {
                        dim = v.len();
                    }
                }
                for v in vectors {
                    if global_row.is_multiple_of(stride) {
                        sample.push(v.clone());
                    }
                    global_row += 1;
                }
                Ok(())
            };
            stream_vector_batches(&store, &entry.file_path, column, &mut on_batch).await?;
        }

        if sample.is_empty() || dim == 0 {
            anyhow::bail!(
                "build_from_table: column '{}' produced no vectors (is it a vector column?)",
                column
            );
        }

        let centroids = global_centroids::train_global(&sample, k, None)?;
        let quantizer = train_quantizer(algo, &sample, dim, metric_for_algo(algo))?;
        let metric = metric_for_algo(algo);

        // ---- Pass 2: assign every row, range-bucket to disk ----
        let buckets = build_bucket_count();
        let tmp_dir = std::env::temp_dir().join(format!("bsdb-cluster-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp_dir)
            .with_context(|| format!("creating build temp dir {}", tmp_dir.display()))?;

        let mut bucket_files: Vec<std::fs::File> = Vec::with_capacity(buckets);
        for b in 0..buckets {
            let path = tmp_dir.join(format!("bucket_{b}.bin"));
            bucket_files.push(
                std::fs::File::create(&path)
                    .with_context(|| format!("creating bucket file {}", path.display()))?,
            );
        }

        let clusters_per_bucket = centroids.len().div_ceil(buckets);
        let mut assign_result: Result<()> = Ok(());
        for (seg_idx, entry) in entries.iter().enumerate() {
            let seg_idx = seg_idx as u32;
            let mut on_batch = |offset: usize, vectors: &[Vec<f32>]| -> Result<()> {
                let labels = assign_to_centroids(vectors, &centroids, metric)?;
                for (i, (label, v)) in labels.iter().zip(vectors.iter()).enumerate() {
                    let row_id = (offset + i) as u32;
                    let bucket = (*label / clusters_per_bucket).min(buckets - 1);
                    write_record(&mut bucket_files[bucket], *label as u32, seg_idx, row_id, v)?;
                }
                Ok(())
            };
            if let Err(e) =
                stream_vector_batches(&store, &entry.file_path, column, &mut on_batch).await
            {
                assign_result = Err(e);
                break;
            }
        }
        drop(bucket_files);

        if let Err(e) = assign_result {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Err(e);
        }

        // ---- Pass 3: Build one cluster at a time, streaming into sequential 512 MiB shards ----
        let target_bytes = target_shard_bytes_override.unwrap_or_else(target_shard_bytes);
        let mut shards_out: Vec<ClusterShard> = Vec::new();

        struct ActiveShard {
            shard_idx: usize,
            cluster_start: usize,
            cluster_end: usize,
            path: std::path::PathBuf,
            writer: PuffinWriter<std::io::BufWriter<std::fs::File>>,
            cluster_count: usize,
        }

        let seal_shard = |shard: ActiveShard| -> Result<ClusterShard> {
            let mut buf_writer = shard.writer.finish()?;
            buf_writer.flush()?;
            let file = std::fs::File::open(&shard.path)
                .with_context(|| format!("opening sealed shard {}", shard.path.display()))?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let container = Bytes::from_owner(MmapOwner(Arc::new(mmap)));
            let slices = cluster_slices_from_container(&container)?;
            let c_start = if shard.cluster_start == usize::MAX {
                0
            } else {
                shard.cluster_start
            };
            Ok(ClusterShard {
                shard_idx: shard.shard_idx,
                file_path: None,
                cluster_range: (c_start, shard.cluster_end),
                container,
                cluster_slices: slices,
            })
        };

        let shard0_path = tmp_dir.join("shard_0.puffin");
        let shard0_file = std::fs::File::create(&shard0_path)
            .with_context(|| format!("creating shard file {}", shard0_path.display()))?;
        let mut current_shard = ActiveShard {
            shard_idx: 0,
            cluster_start: usize::MAX,
            cluster_end: 0,
            path: shard0_path,
            writer: PuffinWriter::new(std::io::BufWriter::new(shard0_file))?,
            cluster_count: 0,
        };
        write_header_blobs(
            &mut current_shard.writer,
            &centroids,
            metric,
            quantizer.as_ref(),
            &segment_table,
        )?;

        let mut build_result: Result<()> = Ok(());
        for b in 0..buckets {
            let path = tmp_dir.join(format!("bucket_{b}.bin"));
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(e) => {
                    build_result = Err(anyhow::Error::from(e)
                        .context(format!("reading bucket {}", path.display())));
                    break;
                }
            };
            let grouped = match group_bucket(&bytes, dim) {
                Ok(grouped) => grouped,
                Err(e) => {
                    build_result = Err(e);
                    break;
                }
            };
            let mut cluster_ids: Vec<usize> = grouped.keys().copied().collect();
            cluster_ids.sort_unstable();

            let built_results: Vec<Result<Option<BuiltClusterBlobs>>> = cluster_ids
                .par_iter()
                .map(|&cluster_id| {
                    let members = &grouped[&cluster_id];
                    build_cluster_blobs(
                        cluster_id,
                        members,
                        quantizer.as_ref(),
                        metric,
                        m,
                        ef_construction,
                    )
                })
                .collect();

            let mut built_clusters = Vec::with_capacity(built_results.len());
            for res in built_results {
                match res {
                    Ok(Some(blobs)) => built_clusters.push(blobs),
                    Ok(None) => {}
                    Err(e) => {
                        build_result = Err(e);
                        break;
                    }
                }
            }
            if build_result.is_err() {
                break;
            }

            for blobs in built_clusters {
                let cluster_id = blobs.cluster_id;

                // If current shard has reached target size, seal it and roll to the next sequential shard
                if current_shard.cluster_count > 0
                    && current_shard.writer.current_offset() as usize >= target_bytes
                {
                    let next_idx = current_shard.shard_idx + 1;
                    let next_path = tmp_dir.join(format!("shard_{next_idx}.puffin"));
                    let next_file = match std::fs::File::create(&next_path) {
                        Ok(f) => f,
                        Err(e) => {
                            build_result = Err(anyhow::Error::from(e)
                                .context(format!("creating shard file {}", next_path.display())));
                            break;
                        }
                    };
                    let new_writer = match PuffinWriter::new(std::io::BufWriter::new(next_file)) {
                        Ok(w) => w,
                        Err(e) => {
                            build_result = Err(e);
                            break;
                        }
                    };
                    let old_shard = std::mem::replace(
                        &mut current_shard,
                        ActiveShard {
                            shard_idx: next_idx,
                            cluster_start: usize::MAX,
                            cluster_end: 0,
                            path: next_path,
                            writer: new_writer,
                            cluster_count: 0,
                        },
                    );
                    let sealed = match seal_shard(old_shard) {
                        Ok(s) => s,
                        Err(e) => {
                            build_result = Err(e);
                            break;
                        }
                    };
                    shards_out.push(sealed);
                }

                if let Err(e) = write_cluster_blobs_to_puffin(&mut current_shard.writer, &blobs) {
                    build_result = Err(e);
                    break;
                }

                if current_shard.cluster_start == usize::MAX {
                    current_shard.cluster_start = cluster_id;
                }
                current_shard.cluster_end = cluster_id;
                current_shard.cluster_count += 1;
            }

            if build_result.is_err() {
                break;
            }
            // Free the bucket as soon as it is consumed.
            let _ = std::fs::remove_file(&path);
        }

        if let Err(e) = build_result {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Err(e);
        }

        // Seal final shard
        let final_shard = seal_shard(current_shard)?;
        shards_out.push(final_shard);

        let mut idx = Self::from_shards(
            shards_out,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            0,
            total_rows as i64,
        )?;
        idx._temp_dir = Some(tmp_dir);
        Ok(idx)
    }

    /// Build an artifact directly from `(segment_idx, row_id, vector)` records.
    ///
    /// Used by tests and small tables; the streaming path is [`Self::build_from_table`].
    pub fn build_from_records(
        records: Vec<(u32, u32, Vec<f32>)>,
        segment_table: Vec<String>,
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::build_from_records_with_target(
            records,
            segment_table,
            column,
            algo,
            k,
            m,
            ef_construction,
            target_shard_bytes(),
        )
    }

    /// Build an artifact directly from records with an explicit target shard size.
    pub fn build_from_records_with_target(
        records: Vec<(u32, u32, Vec<f32>)>,
        segment_table: Vec<String>,
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
        target_bytes: usize,
    ) -> Result<Self> {
        if records.is_empty() {
            anyhow::bail!("build_from_records requires at least one record");
        }
        let record_count = records.len() as i64;
        let dim = records[0].2.len();
        let metric = metric_for_algo(algo);
        let vectors: Vec<Vec<f32>> = records.iter().map(|(_, _, v)| v.clone()).collect();
        let centroids = global_centroids::train_global(&vectors, k, None)?;
        let quantizer = train_quantizer(algo, &vectors, dim, metric)?;
        let labels = assign_to_centroids(&vectors, &centroids, metric)?;

        let mut grouped: HashMap<usize, Vec<(u32, u32, Vec<f32>)>> = HashMap::new();
        for ((seg, row, v), label) in records.into_iter().zip(labels) {
            grouped.entry(label).or_default().push((seg, row, v));
        }

        let mut sorted_cluster_ids: Vec<usize> = grouped.keys().copied().collect();
        sorted_cluster_ids.sort_unstable();

        let mut shards_out: Vec<ClusterShard> = Vec::new();
        struct ActiveShard {
            shard_idx: usize,
            cluster_start: usize,
            cluster_end: usize,
            writer: PuffinWriter<Cursor<Vec<u8>>>,
            cluster_count: usize,
        }

        let mut current_shard = ActiveShard {
            shard_idx: 0,
            cluster_start: usize::MAX,
            cluster_end: 0,
            writer: PuffinWriter::new(Cursor::new(Vec::new()))?,
            cluster_count: 0,
        };
        write_header_blobs(
            &mut current_shard.writer,
            &centroids,
            metric,
            quantizer.as_ref(),
            &segment_table,
        )?;

        for cluster_id in sorted_cluster_ids {
            let members = &grouped[&cluster_id];
            if members.is_empty() {
                continue;
            }

            if current_shard.cluster_count > 0
                && current_shard.writer.current_offset() as usize >= target_bytes
            {
                let next_idx = current_shard.shard_idx + 1;
                let new_writer = PuffinWriter::new(Cursor::new(Vec::new()))?;
                let old_shard = std::mem::replace(
                    &mut current_shard,
                    ActiveShard {
                        shard_idx: next_idx,
                        cluster_start: usize::MAX,
                        cluster_end: 0,
                        writer: new_writer,
                        cluster_count: 0,
                    },
                );
                let container = Bytes::from(old_shard.writer.finish()?.into_inner());
                let slices = cluster_slices_from_container(&container)?;
                shards_out.push(ClusterShard {
                    shard_idx: old_shard.shard_idx,
                    file_path: None,
                    cluster_range: (old_shard.cluster_start, old_shard.cluster_end),
                    container,
                    cluster_slices: slices,
                });
            }

            write_cluster_blobs(
                &mut current_shard.writer,
                cluster_id,
                members,
                quantizer.as_ref(),
                metric,
                m,
                ef_construction,
            )?;
            if current_shard.cluster_start == usize::MAX {
                current_shard.cluster_start = cluster_id;
            }
            current_shard.cluster_end = cluster_id;
            current_shard.cluster_count += 1;
        }

        let container = Bytes::from(current_shard.writer.finish()?.into_inner());
        let slices = cluster_slices_from_container(&container)?;
        let c_start = if current_shard.cluster_start == usize::MAX {
            0
        } else {
            current_shard.cluster_start
        };
        shards_out.push(ClusterShard {
            shard_idx: current_shard.shard_idx,
            file_path: None,
            cluster_range: (c_start, current_shard.cluster_end),
            container,
            cluster_slices: slices,
        });

        Self::from_shards(
            shards_out,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            0,
            record_count,
        )
    }

    /// Assemble an index from sequential shards.
    pub fn from_shards(
        shards: Vec<ClusterShard>,
        column: String,
        dim: usize,
        metric: VectorMetric,
        centroids: Vec<Vec<f32>>,
        quantizer: Option<QuantizerImpl>,
        segment_table: Vec<String>,
        manifest_version: u64,
        record_count: i64,
    ) -> Result<Self> {
        let mut cluster_to_shard = HashMap::new();
        for (s_idx, shard) in shards.iter().enumerate() {
            for cid in shard.cluster_slices.keys() {
                cluster_to_shard.insert(*cid, s_idx);
            }
        }
        Ok(Self {
            column,
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            record_count,
            shards,
            cluster_to_shard,
            cache: moka::sync::Cache::builder().max_capacity(64).build(),
            _temp_dir: None,
        })
    }

    /// Assemble an index from an already-serialized container (single shard).
    pub fn from_container(
        container: Bytes,
        column: String,
        dim: usize,
        metric: VectorMetric,
        centroids: Vec<Vec<f32>>,
        quantizer: Option<QuantizerImpl>,
        segment_table: Vec<String>,
        manifest_version: u64,
        record_count: i64,
    ) -> Result<Self> {
        let cluster_slices = cluster_slices_from_container(&container)?;
        let min_c = cluster_slices.keys().copied().min().unwrap_or(0);
        let max_c = cluster_slices.keys().copied().max().unwrap_or(0);
        let shard = ClusterShard {
            shard_idx: 0,
            file_path: None,
            cluster_range: (min_c, max_c),
            container,
            cluster_slices,
        };
        Self::from_shards(
            vec![shard],
            column,
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            record_count,
        )
    }

    /// Load an artifact (or sequential shard set) from the object store.
    pub async fn load(
        store: Arc<dyn ObjectStore>,
        path: &str,
        column: &str,
        manifest_version: u64,
    ) -> Result<Self> {
        let object_path = object_store::path::Path::from(path);
        let container = store
            .get(&object_path)
            .await
            .with_context(|| format!("opening cluster index artifact '{path}'"))?
            .bytes()
            .await
            .with_context(|| format!("reading cluster index artifact '{path}'"))?;

        // If this container is Shard 0 with a multi-shard descriptor blob, load all shards
        if let Ok(Some((offset, len))) = find_blob(&container, PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS) {
            let shard_meta_bytes = container.slice(offset..offset + len);
            if let Ok(shard_paths) = serde_json::from_slice::<Vec<String>>(&shard_meta_bytes) {
                if shard_paths.len() > 1 {
                    return Self::load_sharded(store, &shard_paths, column, manifest_version).await;
                }
            }
        }

        let (centroids, metric, quantizer) = read_centroids_blob(&container)?;
        let segment_table = read_segments_blob(&container)?;
        let dim = centroids.first().map(|c| c.len()).unwrap_or(0);
        let record_count = 0;

        Self::from_container(
            container,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            record_count,
        )
    }

    /// Load a sequential multi-shard index from the object store.
    pub async fn load_sharded(
        store: Arc<dyn ObjectStore>,
        shard_paths: &[String],
        column: &str,
        manifest_version: u64,
    ) -> Result<Self> {
        if shard_paths.is_empty() {
            anyhow::bail!("load_sharded requires at least one shard path");
        }

        // Shard 0 always contains the global centroids and segment table.
        let shard0_path = &shard_paths[0];
        let shard0_obj = object_store::path::Path::from(shard0_path.as_str());
        let shard0_container = store
            .get(&shard0_obj)
            .await
            .with_context(|| format!("opening shard 0 '{shard0_path}'"))?
            .bytes()
            .await
            .with_context(|| format!("reading shard 0 bytes '{shard0_path}'"))?;

        let (centroids, metric, quantizer) = read_centroids_blob(&shard0_container)?;
        let segment_table = read_segments_blob(&shard0_container)?;
        let dim = centroids.first().map(|c| c.len()).unwrap_or(0);

        let mut shards: Vec<ClusterShard> = Vec::with_capacity(shard_paths.len());
        let s0_slices = cluster_slices_from_container(&shard0_container)?;
        let s0_min = s0_slices.keys().copied().min().unwrap_or(0);
        let s0_max = s0_slices.keys().copied().max().unwrap_or(0);
        shards.push(ClusterShard {
            shard_idx: 0,
            file_path: Some(shard0_path.clone()),
            cluster_range: (s0_min, s0_max),
            container: shard0_container,
            cluster_slices: s0_slices,
        });

        for (i, path) in shard_paths.iter().enumerate().skip(1) {
            let obj_path = object_store::path::Path::from(path.as_str());
            let container = store
                .get(&obj_path)
                .await
                .with_context(|| format!("opening shard {i} '{path}'"))?
                .bytes()
                .await
                .with_context(|| format!("reading shard {i} bytes '{path}'"))?;
            let slices = cluster_slices_from_container(&container)?;
            let s_min = slices.keys().copied().min().unwrap_or(0);
            let s_max = slices.keys().copied().max().unwrap_or(0);
            shards.push(ClusterShard {
                shard_idx: i,
                file_path: Some(path.clone()),
                cluster_range: (s_min, s_max),
                container,
                cluster_slices: slices,
            });
        }

        Self::from_shards(
            shards,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            0,
        )
    }

    /// Persist the container shards to the object store.
    ///
    /// Single-shard indexes are written directly to `path`.
    /// Multi-shard indexes are partitioned sequentially into 512 MiB shards:
    /// `..._shard_0000.puffin`, `..._shard_0001.puffin`, etc.
    /// Returns the list of persisted shard paths in sequential order.
    pub async fn save(&self, store: Arc<dyn ObjectStore>, path: &str) -> Result<Vec<String>> {
        let mut saved_paths = Vec::new();
        if self.shards.len() <= 1 {
            let object_path = object_store::path::Path::from(path);
            let bytes = if let Some(shard) = self.shards.first() {
                shard.container.clone()
            } else {
                Bytes::new()
            };
            store
                .put(&object_path, bytes.into())
                .await
                .with_context(|| format!("writing cluster index artifact '{path}'"))?;
            saved_paths.push(path.to_string());
        } else {
            let stem = path.strip_suffix(".puffin").unwrap_or(path);
            let shard_names: Vec<String> = (0..self.shards.len())
                .map(|i| format!("{stem}_shard_{:04}.puffin", i))
                .collect();

            for (i, shard) in self.shards.iter().enumerate() {
                let shard_path = &shard_names[i];
                let object_path = object_store::path::Path::from(shard_path.as_str());

                let bytes_to_write = if i == 0 {
                    let mut writer = PuffinWriter::new(Cursor::new(Vec::new()))?;
                    let reader = PuffinReader::new(Cursor::new(shard.container.as_ref()))?;
                    for b in reader.footer().blobs.iter() {
                        let offset = b.offset as usize;
                        let length = b.length as usize;
                        let blob_data = &shard.container[offset..offset + length];
                        writer.add_blob(
                            b.r#type.clone(),
                            b.fields.clone(),
                            b.snapshot_id,
                            b.sequence_number,
                            blob_data,
                            b.properties.clone(),
                        )?;
                    }
                    let shard_json = serde_json::to_vec(&shard_names)?;
                    writer.add_blob(
                        PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS.to_string(),
                        Vec::new(),
                        0,
                        1,
                        &shard_json,
                        HashMap::from([(
                            "filename".to_string(),
                            "cluster_shards.json".to_string(),
                        )]),
                    )?;
                    Bytes::from(writer.finish()?.into_inner())
                } else {
                    shard.container.clone()
                };

                store
                    .put(&object_path, bytes_to_write.into())
                    .await
                    .with_context(|| format!("writing cluster index shard '{shard_path}'"))?;
                saved_paths.push(shard_path.clone());
            }
        }
        Ok(saved_paths)
    }

    /// Describe this artifact as a manifest [`TableIndexFile`].
    pub fn to_table_index_file(
        &self,
        path: &str,
        manifest_version: u64,
        algorithm: &str,
    ) -> TableIndexFile {
        let shard_files = if self.shards.len() <= 1 {
            vec![path.to_string()]
        } else {
            let stem = path.strip_suffix(".puffin").unwrap_or(path);
            (0..self.shards.len())
                .map(|i| format!("{stem}_shard_{:04}.puffin", i))
                .collect()
        };
        self.to_table_index_file_with_shards(path, shard_files, manifest_version, algorithm)
    }

    /// Describe this artifact as a manifest [`TableIndexFile`] with explicit shard files.
    pub fn to_table_index_file_with_shards(
        &self,
        path: &str,
        shard_files: Vec<String>,
        manifest_version: u64,
        algorithm: &str,
    ) -> TableIndexFile {
        TableIndexFile {
            format_version: 2,
            index_id: uuid::Uuid::new_v4().to_string(),
            index_category: "vector".to_string(),
            algorithm: algorithm.to_string(),
            column_name: self.column.clone(),
            source_manifest_version: manifest_version,
            source_segments: self.segment_table.clone(),
            source_record_count: self.record_count,
            index_checksum: None,
            file_path: path.to_string(),
            shard_files,
            blob_type: Some(blob_type_for(self.quantizer.as_ref()).to_string()),
            cluster_count: self.centroids.len(),
            dim: self.dim,
            metric: self.metric.as_str().to_string(),
            compression: "none".to_string(),
        }
    }

    /// Search the table-level index.
    ///
    /// Coarse-scans the global centroids, takes the top `n_probe`, traverses
    /// those cluster graphs across their respective sequential shards, and maps
    /// each node back to `(segment_id, row_id)`. The returned [`ScoredResult`]s
    /// feed the existing hydration path unchanged.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>> {
        if k == 0 || self.centroids.is_empty() || query.len() != self.dim {
            return Ok(Vec::new());
        }

        // Coarse: rank global centroids.
        let mut cluster_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, metric_distance(self.metric, query, c)))
            .collect();
        cluster_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let probes = n_probe.clamp(1, cluster_distances.len());
        let effective_query = self.quantize_query(query);
        let ef = ef_search.unwrap_or(40);
        let cluster_k = (k * 4).max(ef.min(k * 8));

        let mut candidates: Vec<ScoredResult> = Vec::new();
        for (cluster_id, _) in cluster_distances.into_iter().take(probes) {
            let Some(cluster) = self.get_cluster(cluster_id) else {
                continue;
            };
            let (graph, node_meta) = cluster.as_ref();
            let cluster_ef = ef.min((node_meta.len() * 3) / 4).max(cluster_k);
            for neighbour in graph.search(&effective_query, cluster_k, cluster_ef, None) {
                let local_id = neighbour.d_id;
                let Some(meta) = node_meta.get(local_id) else {
                    continue;
                };
                let Some(segment_id) = self.segment_table.get(meta.segment_idx as usize) else {
                    continue;
                };
                candidates.push(ScoredResult {
                    segment_id: segment_id.clone(),
                    row_id: meta.row_id,
                    score: neighbour.distance,
                });
            }
        }

        candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.dedup_by(|a, b| a.segment_id == b.segment_id && a.row_id == b.row_id);
        candidates.truncate(k);
        Ok(candidates)
    }

    /// Load a cluster's graph and node metadata from its containing shard, caching the deserialized form.
    fn get_cluster(&self, cluster_id: usize) -> Option<Arc<(HnswGraph, Vec<NodeMeta>)>> {
        if let Some(hit) = self.cache.get(&cluster_id) {
            return Some(hit);
        }
        let shard_idx = *self.cluster_to_shard.get(&cluster_id)?;
        let shard = self.shards.get(shard_idx)?;
        let slices = shard.cluster_slices.get(&cluster_id)?;
        let graph_bytes = shard
            .container
            .slice(slices.graph.0..slices.graph.0 + slices.graph.1);
        let meta_bytes = shard
            .container
            .slice(slices.node_meta.0..slices.node_meta.0 + slices.node_meta.1);

        let graph = match deserialize_graph(graph_bytes, self.quantizer.as_ref(), self.metric) {
            Ok(graph) => graph,
            Err(e) => {
                tracing::warn!(cluster_id, shard_idx, error = %e, "cluster_index: graph deserialization failed");
                return None;
            }
        };
        let node_meta = match node_meta_from_parquet(meta_bytes) {
            Ok(meta) => meta,
            Err(e) => {
                tracing::warn!(cluster_id, shard_idx, error = %e, "cluster_index: node metadata parse failed");
                return None;
            }
        };

        let arc = Arc::new((graph, node_meta));
        self.cache.insert(cluster_id, arc.clone());
        Some(arc)
    }

    /// Quantize the query the same way the graph codes were quantized.
    fn quantize_query(&self, query: &[f32]) -> VectorValue {
        match self.quantizer {
            Some(QuantizerImpl::BenoQuant(ref q)) => {
                let mut q_vec = query.to_vec();
                if self.metric == VectorMetric::Cosine {
                    normalize(&mut q_vec);
                }
                VectorValue::Binary(q.encode(&q_vec))
            }
            Some(QuantizerImpl::Sq(ref q)) => VectorValue::Binary(q.encode(query)),
            Some(QuantizerImpl::Pq(ref q)) => {
                let mut q_vec = query.to_vec();
                if self.metric == VectorMetric::InnerProduct || self.metric == VectorMetric::Cosine
                {
                    normalize(&mut q_vec);
                }
                VectorValue::Binary(q.encode(&q_vec))
            }
            None => VectorValue::Float32(query.to_vec()),
        }
    }
}

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

/// Segment id convention shared with the query path: basename minus extension.
fn segment_id_of(entry: &ManifestEntry) -> String {
    let name = entry
        .file_path
        .rsplit('/')
        .next()
        .unwrap_or(&entry.file_path);
    let name = name.strip_suffix(".puffin").unwrap_or(name);
    name.strip_suffix(".parquet").unwrap_or(name).to_string()
}

/// Metric implied by the index algorithm.
///
/// The algorithm does not carry a metric, so the artifact defaults to L2 — the
/// same default the per-segment builder uses. Callers that need another metric
/// should build with [`ClusterIndex::build_from_records`] or extend this.
fn metric_for_algo(_algo: &IndexAlgorithm) -> VectorMetric {
    VectorMetric::L2
}

/// Train the quantizer named by `algo` over the sample.
fn train_quantizer(
    algo: &IndexAlgorithm,
    sample: &[Vec<f32>],
    dim: usize,
    metric: VectorMetric,
) -> Result<Option<QuantizerImpl>> {
    use crate::core::index::benoquant::BenoQuantEncoder;
    use crate::core::index::pq::{PqConfig, PqEncoder};
    use crate::core::index::sq::SqEncoder;

    Ok(match algo {
        IndexAlgorithm::HnswTq8 { .. } => Some(QuantizerImpl::BenoQuant(BenoQuantEncoder::train(
            sample, 8, metric,
        )?)),
        IndexAlgorithm::HnswSq8 { .. } => {
            Some(QuantizerImpl::Sq(SqEncoder::train(sample, 8, metric)?))
        }
        IndexAlgorithm::HnswTq4 { .. } => Some(QuantizerImpl::BenoQuant(BenoQuantEncoder::train(
            sample, 4, metric,
        )?)),
        IndexAlgorithm::HnswPq { compression, .. } => {
            let subspaces = (dim / (*compression).max(1)).max(1);
            Some(QuantizerImpl::Pq(PqEncoder::train(
                sample,
                PqConfig {
                    m: subspaces,
                    k: 256,
                    dim,
                },
            )?))
        }
        _ => None,
    })
}

/// Puffin blob type for the graph payload.
fn blob_type_for(_quantizer: Option<&QuantizerImpl>) -> &'static str {
    PUFFIN_BLOB_VECTOR_HNSW_GRAPH
}

/// Write the centroids and segment-table blobs.
fn write_header_blobs<W: std::io::Write + std::io::Seek>(
    writer: &mut PuffinWriter<W>,
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    quantizer: Option<&QuantizerImpl>,
    segment_table: &[String],
) -> Result<()> {
    let centroids_bytes = centroids_to_parquet(centroids, metric, quantizer)?;
    writer.add_blob(
        PUFFIN_BLOB_HNSW_IVF_CENTROIDS.to_string(),
        Vec::new(),
        0,
        1,
        &centroids_bytes,
        HashMap::from([("filename".to_string(), "centroids.parquet".to_string())]),
    )?;

    let segments_bytes = segments_to_parquet(segment_table)?;
    writer.add_blob(
        PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS.to_string(),
        Vec::new(),
        0,
        1,
        &segments_bytes,
        HashMap::from([("filename".to_string(), "segments.parquet".to_string())]),
    )?;
    Ok(())
}

/// In-memory serialized blobs for a completed cluster graph and node metadata.
struct BuiltClusterBlobs {
    cluster_id: usize,
    graph_blob_type: &'static str,
    graph_bytes: Vec<u8>,
    meta_bytes: Vec<u8>,
}

/// Construct and serialize one cluster's HNSW graph and node metadata in memory.
///
/// Embarrassingly parallel: takes only immutable references and creates a thread-unique
/// temp dump file before returning the owned byte payloads.
fn build_cluster_blobs(
    cluster_id: usize,
    members: &[(u32, u32, Vec<f32>)],
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
    m: usize,
    ef_construction: usize,
) -> Result<Option<BuiltClusterBlobs>> {
    if members.is_empty() {
        return Ok(None);
    }

    let vectors: Vec<Vec<f32>> = members.iter().map(|(_, _, v)| v.clone()).collect();
    let graph = build_graph(&vectors, quantizer, metric, m, ef_construction);

    let tmp_base = std::env::temp_dir().join(format!(
        "bsdb-cluster-graph-{}-{}",
        uuid::Uuid::new_v4(),
        cluster_id
    ));
    let tmp_str = tmp_base.to_string_lossy().to_string();
    graph
        .file_dump(&tmp_str)
        .map_err(|e| anyhow::anyhow!("cluster {cluster_id}: HNSW dump failed: {e}"))?;
    let graph_path = format!("{tmp_str}.hnsw.graph");
    let graph_bytes =
        std::fs::read(&graph_path).with_context(|| format!("reading dumped graph {graph_path}"))?;
    let _ = std::fs::remove_file(&graph_path);

    let node_meta: Vec<NodeMeta> = members
        .iter()
        .map(|(seg, row, _)| NodeMeta {
            segment_idx: *seg,
            row_id: *row,
        })
        .collect();
    let meta_bytes = node_meta_to_parquet(&node_meta)?;

    Ok(Some(BuiltClusterBlobs {
        cluster_id,
        graph_blob_type: blob_type_for(quantizer),
        graph_bytes,
        meta_bytes,
    }))
}

/// Write pre-built cluster graph and node metadata blobs to a Puffin shard.
fn write_cluster_blobs_to_puffin<W: std::io::Write + std::io::Seek>(
    writer: &mut PuffinWriter<W>,
    blobs: &BuiltClusterBlobs,
) -> Result<()> {
    let cluster_props = HashMap::from([
        ("cluster_id".to_string(), blobs.cluster_id.to_string()),
        (
            "filename".to_string(),
            format!("cluster_{}.hnsw.graph", blobs.cluster_id),
        ),
    ]);
    writer.add_blob(
        blobs.graph_blob_type.to_string(),
        Vec::new(),
        0,
        1,
        &blobs.graph_bytes,
        cluster_props,
    )?;

    let meta_props = HashMap::from([
        ("cluster_id".to_string(), blobs.cluster_id.to_string()),
        (
            "filename".to_string(),
            format!("cluster_{}.node_meta.parquet", blobs.cluster_id),
        ),
    ]);
    writer.add_blob(
        PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META.to_string(),
        Vec::new(),
        0,
        1,
        &blobs.meta_bytes,
        meta_props,
    )?;
    Ok(())
}

/// Build and write one cluster's graph and node metadata sequentially.
fn write_cluster_blobs<W: std::io::Write + std::io::Seek>(
    writer: &mut PuffinWriter<W>,
    cluster_id: usize,
    members: &[(u32, u32, Vec<f32>)],
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
    m: usize,
    ef_construction: usize,
) -> Result<()> {
    if let Some(blobs) =
        build_cluster_blobs(cluster_id, members, quantizer, metric, m, ef_construction)?
    {
        write_cluster_blobs_to_puffin(writer, &blobs)?;
    }
    Ok(())
}

/// Build an HNSW graph over one cluster's vectors.
fn build_graph(
    vectors: &[Vec<f32>],
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
    m: usize,
    ef_construction: usize,
) -> HnswGraph {
    let max_layers = 16;
    let mut hnsw = if let Some(q) = quantizer {
        match q {
            QuantizerImpl::BenoQuant(tq) => {
                if tq.bits() == 4 {
                    HnswGraph::TurboQuant4(Hnsw::new(
                        m,
                        vectors.len(),
                        max_layers,
                        ef_construction,
                        crate::core::index::distance::DistL2u4,
                    ))
                } else {
                    HnswGraph::TurboQuant8(Hnsw::new(
                        m,
                        vectors.len(),
                        max_layers,
                        ef_construction,
                        crate::core::index::distance::DistL2u8,
                    ))
                }
            }
            QuantizerImpl::Sq(q) => HnswGraph::Sq8(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                crate::core::index::sq::DistSq8 {
                    encoder: Arc::new(q.clone()),
                },
            )),
            QuantizerImpl::Pq(pq) => {
                let dist = crate::core::index::pq::DistPqSdc {
                    pq: Arc::new(pq.clone()),
                };
                HnswGraph::Pq(Hnsw::new(
                    m,
                    vectors.len(),
                    max_layers,
                    ef_construction,
                    dist,
                ))
            }
        }
    } else {
        match metric {
            VectorMetric::L2 => HnswGraph::L2(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistL2,
            )),
            VectorMetric::Cosine => HnswGraph::Cosine(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistCosine,
            )),
            VectorMetric::InnerProduct => HnswGraph::Dot(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistDot::norm(),
            )),
            VectorMetric::L1 => HnswGraph::L1(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistL1,
            )),
            VectorMetric::Hamming => HnswGraph::Hamming(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistHamming,
            )),
            VectorMetric::Jaccard => HnswGraph::Jaccard(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistJaccard,
            )),
        }
    };

    hnsw.set_extend_candidates(true);
    hnsw.set_keeping_pruned(true);
    for (local_id, v) in vectors.iter().enumerate() {
        if let Some(q) = quantizer {
            hnsw.insert((VectorValue::Binary(q.encode(v)), local_id));
        } else {
            hnsw.insert((VectorValue::Float32(v.clone()), local_id));
        }
    }
    hnsw
}

/// Deserialize a graph payload, choosing the node type from the quantizer.
fn deserialize_graph(
    bytes: Bytes,
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
) -> Result<HnswGraph> {
    use crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw;

    let graph = if let Some(q) = quantizer {
        match q {
            QuantizerImpl::BenoQuant(tq) => {
                if tq.bits() == 4 {
                    HnswGraph::TurboQuant4Arrow(
                        ArrowHnsw::load_from_bytes_owned(
                            bytes,
                            crate::core::index::distance::DistL2u4,
                        )
                        .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
                    )
                } else {
                    HnswGraph::TurboQuant8Arrow(
                        ArrowHnsw::load_from_bytes_owned(
                            bytes,
                            crate::core::index::distance::DistL2u8,
                        )
                        .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
                    )
                }
            }
            QuantizerImpl::Sq(q) => HnswGraph::Sq8Arrow(
                ArrowHnsw::load_from_bytes_owned(
                    bytes,
                    crate::core::index::sq::DistSq8 {
                        encoder: Arc::new(q.clone()),
                    },
                )
                .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            QuantizerImpl::Pq(pq) => {
                let dist = crate::core::index::pq::DistPqSdc {
                    pq: Arc::new(pq.clone()),
                };
                HnswGraph::PqArrow(
                    ArrowHnsw::load_from_bytes_owned(bytes, dist)
                        .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
                )
            }
        }
    } else {
        match metric {
            VectorMetric::L2 => HnswGraph::L2Arrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistL2)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::Cosine => HnswGraph::CosineArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistCosine)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::InnerProduct => HnswGraph::DotArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistDot::norm())
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::L1 => HnswGraph::L1Arrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistL1)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::Hamming => HnswGraph::HammingArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistHamming)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::Jaccard => HnswGraph::JaccardArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistJaccard)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
        }
    };
    Ok(graph)
}

// ---------------------------------------------------------------------------
// Bucket I/O
// ---------------------------------------------------------------------------

/// Append one `[cluster u32][segment u32][row u32][f32 * dim]` record.
fn write_record(
    file: &mut std::fs::File,
    cluster_id: u32,
    segment_idx: u32,
    row_id: u32,
    vector: &[f32],
) -> Result<()> {
    use std::io::Write;
    file.write_all(&cluster_id.to_le_bytes())?;
    file.write_all(&segment_idx.to_le_bytes())?;
    file.write_all(&row_id.to_le_bytes())?;
    let mut buf = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

/// Group a bucket's records by cluster id.
fn group_bucket(bytes: &[u8], dim: usize) -> Result<HashMap<usize, Vec<(u32, u32, Vec<f32>)>>> {
    let header = 12usize;
    let record = header + dim * 4;
    if record == 0 || !bytes.len().is_multiple_of(record) {
        anyhow::bail!(
            "cluster build bucket is not a whole number of {record}-byte records ({} bytes)",
            bytes.len()
        );
    }
    let mut grouped: HashMap<usize, Vec<(u32, u32, Vec<f32>)>> = HashMap::new();
    for chunk in bytes.chunks_exact(record) {
        let cluster_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap_or([0; 4])) as usize;
        let segment_idx = u32::from_le_bytes(chunk[4..8].try_into().unwrap_or([0; 4]));
        let row_id = u32::from_le_bytes(chunk[8..12].try_into().unwrap_or([0; 4]));
        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim {
            let start = header + i * 4;
            let v = f32::from_le_bytes(chunk[start..start + 4].try_into().unwrap_or([0; 4]));
            vector.push(v);
        }
        grouped
            .entry(cluster_id)
            .or_default()
            .push((segment_idx, row_id, vector));
    }
    Ok(grouped)
}

// ---------------------------------------------------------------------------
// Parquet helpers
// ---------------------------------------------------------------------------

/// Stream a Parquet file's vector column in batches.
///
/// `on_batch(row_offset, vectors)` is called once per record batch, so the
/// caller never materializes a whole segment.
async fn stream_vector_batches<F>(
    store: &Arc<dyn ObjectStore>,
    path: &str,
    column: &str,
    on_batch: &mut F,
) -> Result<()>
where
    F: FnMut(usize, &[Vec<f32>]) -> Result<()>,
{
    let object_path = object_store::path::Path::from(path);
    let reader = ParquetObjectReader::new(store.clone(), object_path);
    let builder = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .with_context(|| format!("opening Parquet file '{path}'"))?;

    let col_idx = builder
        .schema()
        .fields()
        .iter()
        .position(|f| f.name() == column)
        .ok_or_else(|| anyhow::anyhow!("column '{column}' not found in '{path}'"))?;

    let mask = ProjectionMask::roots(builder.parquet_schema(), [col_idx]);
    let mut stream = builder
        .with_projection(mask)
        .build()
        .with_context(|| format!("building Parquet stream for '{path}'"))?;

    let mut row_offset = 0usize;
    while let Some(batch) = stream.next().await {
        let batch = batch.with_context(|| format!("reading batch from '{path}'"))?;
        if batch.num_rows() == 0 {
            continue;
        }
        let col = batch.column(0).clone();
        let vectors = extract_vectors(&col)?;
        on_batch(row_offset, &vectors)?;
        row_offset += batch.num_rows();
    }
    Ok(())
}

/// Extract `Vec<Vec<f32>>` from a FixedSizeList or List column.
fn extract_vectors(col: &ArrayRef) -> Result<Vec<Vec<f32>>> {
    match col.data_type() {
        DataType::FixedSizeList(_, _) => {
            let list = col
                .as_any()
                .downcast_ref::<arrow::array::FixedSizeListArray>()
                .context("invalid cast to FixedSizeListArray")?;
            Ok((0..list.len())
                .map(|i| values_to_f32(&list.value(i)))
                .collect())
        }
        DataType::List(_) => {
            let list = col
                .as_any()
                .downcast_ref::<arrow::array::ListArray>()
                .context("invalid cast to ListArray")?;
            Ok((0..list.len())
                .map(|i| values_to_f32(&list.value(i)))
                .collect())
        }
        other => anyhow::bail!("expected a vector column, found {other:?}"),
    }
}

fn values_to_f32(item: &ArrayRef) -> Vec<f32> {
    if let Some(floats) = item.as_any().downcast_ref::<arrow::array::Float32Array>() {
        floats.values().to_vec()
    } else if let Some(doubles) = item.as_any().downcast_ref::<arrow::array::Float64Array>() {
        doubles.values().iter().map(|&d| d as f32).collect()
    } else {
        vec![0.0; item.len()]
    }
}

fn centroids_to_parquet(
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    quantizer: Option<&QuantizerImpl>,
) -> Result<Vec<u8>> {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for centroid in centroids {
        for &v in centroid {
            builder.values().append_value(v);
        }
        builder.append(true);
    }
    let array = Arc::new(builder.finish());
    let schema = Arc::new(Schema::new(vec![Field::new(
        "centroid",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![
            parquet::file::metadata::KeyValue {
                key: "vector_metric".to_string(),
                value: Some(metric.as_str().to_string()),
            },
            parquet::file::metadata::KeyValue {
                key: "quantizer_config".to_string(),
                value: quantizer.and_then(|q| serde_json::to_string(q).ok()),
            },
        ]))
        .build();
    write_parquet_in_memory(schema, &batch, props)
}

fn segments_to_parquet(segment_table: &[String]) -> Result<Vec<u8>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("segment_idx", DataType::UInt32, false),
        Field::new("segment_id", DataType::Utf8, false),
    ]));
    let idx = UInt32Array::from((0..segment_table.len() as u32).collect::<Vec<_>>());
    let ids = StringArray::from(segment_table.to_vec());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(idx), Arc::new(ids)])?;
    write_parquet_in_memory(schema, &batch, WriterProperties::builder().build())
}

fn node_meta_to_parquet(meta: &[NodeMeta]) -> Result<Vec<u8>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("segment_idx", DataType::UInt32, false),
        Field::new("row_id", DataType::UInt32, false),
    ]));
    let seg = UInt32Array::from(meta.iter().map(|m| m.segment_idx).collect::<Vec<_>>());
    let row = UInt32Array::from(meta.iter().map(|m| m.row_id).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(seg), Arc::new(row)])?;
    write_parquet_in_memory(schema, &batch, WriterProperties::builder().build())
}

fn write_parquet_in_memory(
    schema: Arc<Schema>,
    batch: &RecordBatch,
    props: WriterProperties,
) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))?;
        writer.write(batch)?;
        writer.close()?;
    }
    Ok(buf)
}

fn node_meta_from_parquet(bytes: Bytes) -> Result<Vec<NodeMeta>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let seg = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("node_meta: segment_idx is not UInt32")?;
        let row = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("node_meta: row_id is not UInt32")?;
        for i in 0..batch.num_rows() {
            out.push(NodeMeta {
                segment_idx: seg.value(i),
                row_id: row.value(i),
            });
        }
    }
    Ok(out)
}

fn read_centroids_blob(
    container: &Bytes,
) -> Result<(Vec<Vec<f32>>, VectorMetric, Option<QuantizerImpl>)> {
    let (offset, length) = find_blob(container, PUFFIN_BLOB_HNSW_IVF_CENTROIDS)?
        .ok_or_else(|| anyhow::anyhow!("cluster index container has no centroids blob"))?;
    let bytes = container.slice(offset..offset + length);

    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
    let reader = builder.build()?;
    let mut centroids = Vec::new();
    for batch in reader {
        let batch = batch?;
        if batch.num_columns() == 0 {
            continue;
        }
        centroids.extend(extract_vectors(&batch.column(0).clone())?);
    }

    let mut metric = VectorMetric::L2;
    let mut quantizer = None;
    let file_reader = parquet::file::reader::SerializedFileReader::new(bytes)?;
    if let Some(kv) = file_reader.metadata().file_metadata().key_value_metadata() {
        for entry in kv {
            match entry.key.as_str() {
                "vector_metric" => {
                    if let Some(v) = &entry.value {
                        metric = v.parse::<VectorMetric>().unwrap_or(VectorMetric::L2);
                    }
                }
                "quantizer_config" => {
                    if let Some(v) = &entry.value {
                        if let Ok(mut q) = serde_json::from_str::<QuantizerImpl>(v) {
                            if let QuantizerImpl::Pq(ref mut pq) = q {
                                pq.init_lut();
                            }
                            quantizer = Some(q);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok((centroids, metric, quantizer))
}

fn read_segments_blob(container: &Bytes) -> Result<Vec<String>> {
    let (offset, length) = find_blob(container, PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS)?
        .ok_or_else(|| anyhow::anyhow!("cluster index container has no segments blob"))?;
    let bytes = container.slice(offset..offset + length);
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("segments: segment_id is not Utf8")?;
        for i in 0..batch.num_rows() {
            out.push(ids.value(i).to_string());
        }
    }
    Ok(out)
}

/// Find a blob's `(offset, length)` by type.
fn find_blob(container: &Bytes, blob_type: &str) -> Result<Option<(usize, usize)>> {
    let reader = PuffinReader::new(Cursor::new(container.as_ref()))?;
    for blob in reader.footer().blobs.iter() {
        if blob.r#type == blob_type {
            let offset = usize::try_from(blob.offset)
                .map_err(|_| anyhow::anyhow!("negative blob offset"))?;
            let length = usize::try_from(blob.length)
                .map_err(|_| anyhow::anyhow!("negative blob length"))?;
            if offset
                .checked_add(length)
                .is_none_or(|end| end > container.len())
            {
                anyhow::bail!("blob '{blob_type}' exceeds container size");
            }
            return Ok(Some((offset, length)));
        }
    }
    Ok(None)
}

/// Pair each cluster's graph blob with its node-metadata blob.
fn cluster_slices_from_container(container: &Bytes) -> Result<HashMap<usize, ClusterSlices>> {
    let reader = PuffinReader::new(Cursor::new(container.as_ref()))?;
    let mut graphs: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut metas: HashMap<usize, (usize, usize)> = HashMap::new();

    for blob in reader.footer().blobs.iter() {
        let Some(cluster_id) = blob
            .properties
            .get("cluster_id")
            .and_then(|v| v.parse::<usize>().ok())
        else {
            continue;
        };
        let offset =
            usize::try_from(blob.offset).map_err(|_| anyhow::anyhow!("negative blob offset"))?;
        let length =
            usize::try_from(blob.length).map_err(|_| anyhow::anyhow!("negative blob length"))?;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > container.len())
        {
            anyhow::bail!("cluster {cluster_id} blob exceeds container size");
        }
        if blob.r#type == PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META {
            metas.insert(cluster_id, (offset, length));
        } else if blob.r#type == PUFFIN_BLOB_VECTOR_HNSW_GRAPH {
            graphs.insert(cluster_id, (offset, length));
        }
    }

    let mut out = HashMap::new();
    for (cluster_id, graph) in graphs {
        if let Some(node_meta) = metas.get(&cluster_id) {
            out.insert(
                cluster_id,
                ClusterSlices {
                    graph,
                    node_meta: *node_meta,
                },
            );
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Distance under the artifact's metric, "smaller is closer".
fn metric_distance(metric: VectorMetric, q: &[f32], c: &[f32]) -> f32 {
    use crate::core::index::distance;
    match metric {
        VectorMetric::L2 => distance::l2_distance(q, c),
        VectorMetric::Cosine => distance::cosine_distance(q, c),
        VectorMetric::InnerProduct => -distance::dot_product(q, c),
        VectorMetric::L1 => distance::l1_distance(q, c),
        VectorMetric::Hamming => distance::hamming_distance(q, c),
        VectorMetric::Jaccard => distance::jaccard_distance(q, c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(n: usize, dim: usize, offset: f32) -> Vec<(u32, u32, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| offset + (i as f32) * 0.01 + (d as f32) * 0.001)
                    .collect();
                ((i % 3) as u32, i as u32, v)
            })
            .collect()
    }

    fn segments() -> Vec<String> {
        vec![
            "seg_0".to_string(),
            "seg_1".to_string(),
            "seg_2".to_string(),
        ]
    }

    #[test]
    fn build_and_search_round_trip() {
        let records = blob(300, 8, 0.0);
        let index = ClusterIndex::build_from_records(
            records,
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw().with_complexity(16).with_quality(200),
            8,
            16,
            100,
        )
        .unwrap();

        assert_eq!(index.cluster_count(), 8);
        assert_eq!(index.dim, 8);
        assert_eq!(index.segment_count(), 3);

        let query = vec![0.0; 8];
        let results = index.search(&query, 5, 4, Some(64)).unwrap();
        assert!(!results.is_empty(), "search returned no results");
        assert!(results.len() <= 5);
        // Scores must be ascending (distance).
        for pair in results.windows(2) {
            assert!(pair[0].score <= pair[1].score + 1e-6);
        }
        // Every hit must resolve to a real segment.
        for r in &results {
            assert!(index.segment_table.contains(&r.segment_id));
        }
    }

    #[test]
    fn search_rejects_wrong_dimension() {
        let index = ClusterIndex::build_from_records(
            blob(50, 4, 0.0),
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw().with_complexity(16).with_quality(100),
            4,
            8,
            50,
        )
        .unwrap();
        assert!(index.search(&[0.0; 3], 5, 2, None).unwrap().is_empty());
        assert!(index.search(&[0.0; 4], 0, 2, None).unwrap().is_empty());
    }

    #[test]
    fn node_meta_parquet_round_trip() {
        let meta = vec![
            NodeMeta {
                segment_idx: 0,
                row_id: 7,
            },
            NodeMeta {
                segment_idx: 2,
                row_id: 42,
            },
        ];
        let bytes = node_meta_to_parquet(&meta).unwrap();
        let decoded = node_meta_from_parquet(Bytes::from(bytes)).unwrap();
        assert_eq!(decoded, meta);
    }

    #[test]
    fn segments_parquet_round_trip() {
        let segs = segments();
        let bytes = segments_to_parquet(&segs).unwrap();
        let container = wrap_blob(PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS, &bytes);
        assert_eq!(read_segments_blob(&container).unwrap(), segs);
    }

    #[test]
    fn bucket_grouping_preserves_records() {
        let dim = 3;
        let mut buf = Vec::new();
        let mut file = tempfile::tempfile().unwrap();
        write_record(&mut file, 5, 1, 9, &[1.0, 2.0, 3.0]).unwrap();
        write_record(&mut file, 5, 2, 10, &[4.0, 5.0, 6.0]).unwrap();
        write_record(&mut file, 7, 0, 1, &[7.0, 8.0, 9.0]).unwrap();
        use std::io::{Read, Seek, SeekFrom};
        file.seek(SeekFrom::Start(0)).unwrap();
        file.read_to_end(&mut buf).unwrap();

        let grouped = group_bucket(&buf, dim).unwrap();
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[&5].len(), 2);
        assert_eq!(grouped[&5][0], (1, 9, vec![1.0, 2.0, 3.0]));
        assert_eq!(grouped[&7][0], (0, 1, vec![7.0, 8.0, 9.0]));
    }

    #[test]
    fn table_index_file_covers_segments() {
        let index = ClusterIndex::build_from_records(
            blob(30, 4, 0.0),
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw().with_complexity(16).with_quality(50),
            4,
            8,
            50,
        )
        .unwrap();
        let file = index.to_table_index_file("idx/cluster.puffin", 7, "cluster_hnsw");
        assert_eq!(file.cluster_count, 4);
        assert_eq!(file.dim, 4);
        assert_eq!(file.source_manifest_version, 7);
        assert!(file.covers_segments(&segments()));
        assert!(!file.covers_segments(&["seg_0".to_string()]));
    }

    /// Wrap a single blob in a minimal Puffin container for reader tests.
    fn wrap_blob(blob_type: &str, data: &[u8]) -> Bytes {
        let mut writer = PuffinWriter::new(Cursor::new(Vec::new())).unwrap();
        writer
            .add_blob(
                blob_type.to_string(),
                Vec::new(),
                0,
                1,
                data,
                HashMap::new(),
            )
            .unwrap();
        Bytes::from(writer.finish().unwrap().into_inner())
    }
}
