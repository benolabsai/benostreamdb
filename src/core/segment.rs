// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use std::fs::File;

use crate::SegmentConfig;
use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*; // used in join
                       // allow wide imports to find standard items
use crate::core::index::gpu::{set_thread_gpu_context, ComputeContext};
use crate::core::manifest::{ColumnStats, ManifestEntry, ManifestValue, VectorStats};
use object_store::ObjectStore;
use parquet::file::statistics::Statistics as ParquetStats;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

#[allow(clippy::expect_used)] // Thread-pool construction only fails on OS resource exhaustion; unrecoverable at startup.
fn indexing_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("indexing-rayon-{}", i))
            .num_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .clamp(2, 8),
            )
            .build()
            .expect("Failed to build indexing thread pool")
    })
}

/// Attach Iceberg-compatible Parquet field ids to a batch schema.
///
/// The Iceberg spec requires `field_id` on every data-file column so readers
/// map Parquet columns by id instead of by position. arrow-rs turns the
/// `PARQUET:field_id` metadata key into the Parquet field id, so annotate the
/// schema once per batch before writing.
///
/// Nested ids mirror [`crate::core::manifest::Schema::from_arrow`]: a child of
/// a top-level column `id` is `id * 100 + position + 1`, which is exactly how
/// the manifest schema derives its ids from the same Arrow schema.
fn with_parquet_field_ids(schema: &arrow::datatypes::Schema) -> arrow::datatypes::Schema {
    use arrow::datatypes::DataType;

    fn annotate(field: &arrow::datatypes::Field, id: i32) -> arrow::datatypes::Field {
        let mut metadata = field.metadata().clone();
        metadata
            .entry(parquet::arrow::PARQUET_FIELD_ID_META_KEY.to_string())
            .or_insert_with(|| id.to_string());

        let data_type = match field.data_type() {
            DataType::Struct(children) => DataType::Struct(
                children
                    .iter()
                    .enumerate()
                    .map(|(i, child)| annotate(child, id * 100 + i as i32 + 1))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            DataType::List(child) => DataType::List(Arc::new(annotate(child, id * 100 + 1))),
            DataType::LargeList(child) => {
                DataType::LargeList(Arc::new(annotate(child, id * 100 + 1)))
            }
            DataType::FixedSizeList(child, size) => {
                DataType::FixedSizeList(Arc::new(annotate(child, id * 100 + 1)), *size)
            }
            DataType::Map(entries, sorted) => {
                let entries = match entries.data_type() {
                    DataType::Struct(children) => {
                        let annotated = children
                            .iter()
                            .enumerate()
                            .map(|(i, child)| annotate(child, id * 100 + i as i32 + 1))
                            .collect::<Vec<_>>();
                        Arc::new(
                            arrow::datatypes::Field::new(
                                entries.name().clone(),
                                DataType::Struct(annotated.into()),
                                entries.is_nullable(),
                            )
                            .with_metadata(entries.metadata().clone()),
                        )
                    }
                    _ => entries.clone(),
                };
                DataType::Map(entries, *sorted)
            }
            other => other.clone(),
        };

        arrow::datatypes::Field::new(field.name().clone(), data_type, field.is_nullable())
            .with_metadata(metadata)
    }

    arrow::datatypes::Schema::new(
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, field)| annotate(field, i as i32 + 1))
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn segment_id_from_path(path: &str) -> &str {
    let name = path.split('/').next_back().unwrap_or(path);
    let name = name.strip_suffix(".puffin").unwrap_or(name);
    name.strip_suffix(".parquet").unwrap_or(name)
}

pub(crate) fn ensure_prefix(prefix: &str, path: &str) -> String {
    let clean = path.trim_start_matches('/');
    let prefix = prefix.trim_matches('/');
    if prefix.is_empty()
        || clean == prefix
        || clean
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
    {
        clean.to_string()
    } else {
        format!("{}/{}", prefix, clean)
    }
}

pub(crate) fn relativize_against(root_uri: &str, file_path: &str) -> String {
    if !file_path.starts_with('/') && !file_path.contains("://") {
        return file_path.to_string();
    }
    fn strip_scheme(s: &str) -> &str {
        s.strip_prefix("file://").unwrap_or(s)
    }
    let root = strip_scheme(root_uri).trim_end_matches('/');
    let p = strip_scheme(file_path);
    if !root.is_empty() {
        if let Some(rest) = p.strip_prefix(root) {
            if rest.is_empty() || rest.starts_with('/') {
                return rest.trim_start_matches('/').to_string();
            }
        }
    }
    if file_path.contains("://") {
        if let Ok(url) = url::Url::parse(file_path) {
            return url.path().trim_start_matches('/').to_string();
        }
    }
    file_path
        .split('/')
        .next_back()
        .unwrap_or(file_path)
        .to_string()
}

pub(crate) fn resolve_entry_index_path(entry_file_path: &str, index_file_path: &str) -> String {
    if index_file_path.contains("://") || index_file_path.starts_with('/') {
        return index_file_path.to_string();
    }
    let dir_path = match entry_file_path.rfind('/') {
        Some(pos) => &entry_file_path[..pos],
        None => "",
    };
    ensure_prefix(dir_path, index_file_path)
}

/// Returns `true` when `path` looks like an actual table data Parquet file
/// (including Hive-partitioned `col=val/seg_*.parquet`), excluding manifests,
/// WAL segments, delete files, and secondary index artifacts.
pub(crate) fn is_data_parquet_path(path: &str) -> bool {
    if !path.ends_with(".parquet") {
        return false;
    }
    let normalized = path.trim_start_matches('/');
    if normalized
        .split('/')
        .any(|comp| matches!(comp, "_manifest" | "_wal" | "deletes" | "metadata"))
    {
        return false;
    }
    let file_name = normalized.rsplit('/').next().unwrap_or(normalized);
    !file_name.starts_with("del-pos-")
        && !file_name.starts_with("del-eq-")
        && !file_name.ends_with("_deletes.parquet")
        && !file_name.ends_with(".inv.parquet")
        && !file_name.ends_with(".idx.parquet")
        && !file_name.ends_with(".centroids.parquet")
        && !file_name.ends_with(".mapping.parquet")
        && !file_name.contains(".hnsw.")
        && !file_name.contains(".cluster_")
}

pub(crate) fn normalize_index_family(category: &str) -> &str {
    match category {
        "lexical" | "inverted" | "bm25" => "lexical",
        other => other,
    }
}

pub(crate) enum ScalarPostingMap {
    Int8(std::collections::BTreeMap<i8, Vec<u32>>),
    Int16(std::collections::BTreeMap<i16, Vec<u32>>),
    Int32(std::collections::BTreeMap<i32, Vec<u32>>),
    Int64(std::collections::BTreeMap<i64, Vec<u32>>),
    UInt8(std::collections::BTreeMap<u8, Vec<u32>>),
    UInt16(std::collections::BTreeMap<u16, Vec<u32>>),
    UInt32(std::collections::BTreeMap<u32, Vec<u32>>),
    UInt64(std::collections::BTreeMap<u64, Vec<u32>>),
    Float32(HashMap<u32, Vec<u32>>),
    Float64(HashMap<u64, Vec<u32>>),
    Date32(std::collections::BTreeMap<i32, Vec<u32>>),
    TimestampDay(std::collections::BTreeMap<i32, Vec<u32>>),
    Boolean(std::collections::BTreeMap<bool, Vec<u32>>),
    Time32 {
        unit: arrow::datatypes::TimeUnit,
        postings: std::collections::BTreeMap<i32, Vec<u32>>,
    },
    Time64 {
        unit: arrow::datatypes::TimeUnit,
        postings: std::collections::BTreeMap<i64, Vec<u32>>,
    },
    Binary(std::collections::BTreeMap<Vec<u8>, Vec<u32>>),
    Decimal128 {
        precision: u8,
        scale: i8,
        postings: std::collections::BTreeMap<i128, Vec<u32>>,
    },
}

fn encode_delta_row_ids(
    mut row_ids: Vec<u32>,
    list_builder: &mut arrow::array::ListBuilder<arrow::array::UInt32Builder>,
) {
    row_ids.sort_unstable();
    row_ids.dedup();
    let mut last_id = 0u32;
    for row_id in row_ids {
        list_builder.values().append_value(row_id - last_id);
        last_id = row_id;
    }
    list_builder.append(true);
}

fn scalar_inv_schema(key_type: arrow::datatypes::DataType) -> arrow::datatypes::SchemaRef {
    Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("key", key_type, false),
        arrow::datatypes::Field::new(
            "row_ids",
            arrow::datatypes::DataType::List(Arc::new(arrow::datatypes::Field::new(
                "item",
                arrow::datatypes::DataType::UInt32,
                true,
            ))),
            false,
        ),
    ]))
}

impl ScalarPostingMap {
    fn into_parquet_bytes(self) -> Result<(Vec<u8>, bool)> {
        let mut list_builder = arrow::array::ListBuilder::new(arrow::array::UInt32Builder::new());
        let mut is_timestamp_day = false;

        let (key_array, key_type): (arrow::array::ArrayRef, arrow::datatypes::DataType) = match self
        {
            Self::Int8(map) => {
                let mut b = arrow::array::Int8Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Int8)
            }
            Self::Int16(map) => {
                let mut b = arrow::array::Int16Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Int16)
            }
            Self::Int32(map) => {
                let mut b = arrow::array::Int32Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Int32)
            }
            Self::Int64(map) => {
                let mut b = arrow::array::Int64Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Int64)
            }
            Self::UInt8(map) => {
                let mut b = arrow::array::UInt8Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::UInt8)
            }
            Self::UInt16(map) => {
                let mut b = arrow::array::UInt16Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::UInt16)
            }
            Self::UInt32(map) => {
                let mut b = arrow::array::UInt32Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::UInt32)
            }
            Self::UInt64(map) => {
                let mut b = arrow::array::UInt64Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::UInt64)
            }
            Self::Float32(map) => {
                let mut entries: Vec<(u32, Vec<u32>)> = map.into_iter().collect();
                entries
                    .sort_unstable_by(|a, b| f32::from_bits(a.0).total_cmp(&f32::from_bits(b.0)));
                let mut b = arrow::array::Float32Builder::with_capacity(entries.len());
                for (bits, rows) in entries {
                    b.append_value(f32::from_bits(bits));
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Float32)
            }
            Self::Float64(map) => {
                let mut entries: Vec<(u64, Vec<u32>)> = map.into_iter().collect();
                entries
                    .sort_unstable_by(|a, b| f64::from_bits(a.0).total_cmp(&f64::from_bits(b.0)));
                let mut b = arrow::array::Float64Builder::with_capacity(entries.len());
                for (bits, rows) in entries {
                    b.append_value(f64::from_bits(bits));
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Float64)
            }
            Self::Date32(map) => {
                let mut b = arrow::array::Date32Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Date32)
            }
            Self::TimestampDay(map) => {
                is_timestamp_day = true;
                let mut b = arrow::array::Date32Builder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Date32)
            }
            Self::Boolean(map) => {
                let mut b = arrow::array::BooleanBuilder::with_capacity(map.len());
                for (k, rows) in map {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Boolean)
            }
            Self::Time32 { unit, postings } => match unit {
                arrow::datatypes::TimeUnit::Second => {
                    let mut b = arrow::array::Time32SecondBuilder::with_capacity(postings.len());
                    for (k, rows) in postings {
                        b.append_value(k);
                        encode_delta_row_ids(rows, &mut list_builder);
                    }
                    (
                        Arc::new(b.finish()),
                        arrow::datatypes::DataType::Time32(unit),
                    )
                }
                arrow::datatypes::TimeUnit::Millisecond => {
                    let mut b =
                        arrow::array::Time32MillisecondBuilder::with_capacity(postings.len());
                    for (k, rows) in postings {
                        b.append_value(k);
                        encode_delta_row_ids(rows, &mut list_builder);
                    }
                    (
                        Arc::new(b.finish()),
                        arrow::datatypes::DataType::Time32(unit),
                    )
                }
                other => anyhow::bail!("Unsupported Time32 unit: {:?}", other),
            },
            Self::Time64 { unit, postings } => match unit {
                arrow::datatypes::TimeUnit::Microsecond => {
                    let mut b =
                        arrow::array::Time64MicrosecondBuilder::with_capacity(postings.len());
                    for (k, rows) in postings {
                        b.append_value(k);
                        encode_delta_row_ids(rows, &mut list_builder);
                    }
                    (
                        Arc::new(b.finish()),
                        arrow::datatypes::DataType::Time64(unit),
                    )
                }
                arrow::datatypes::TimeUnit::Nanosecond => {
                    let mut b =
                        arrow::array::Time64NanosecondBuilder::with_capacity(postings.len());
                    for (k, rows) in postings {
                        b.append_value(k);
                        encode_delta_row_ids(rows, &mut list_builder);
                    }
                    (
                        Arc::new(b.finish()),
                        arrow::datatypes::DataType::Time64(unit),
                    )
                }
                other => anyhow::bail!("Unsupported Time64 unit: {:?}", other),
            },
            Self::Binary(map) => {
                let mut b = arrow::array::BinaryBuilder::new();
                for (k, rows) in map {
                    b.append_value(&k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (Arc::new(b.finish()), arrow::datatypes::DataType::Binary)
            }
            Self::Decimal128 {
                precision,
                scale,
                postings,
            } => {
                let mut b = arrow::array::Decimal128Builder::with_capacity(postings.len())
                    .with_precision_and_scale(precision, scale)?;
                for (k, rows) in postings {
                    b.append_value(k);
                    encode_delta_row_ids(rows, &mut list_builder);
                }
                (
                    Arc::new(b.finish()),
                    arrow::datatypes::DataType::Decimal128(precision, scale),
                )
            }
        };

        let inv_schema = scalar_inv_schema(key_type);
        let list_array = Arc::new(list_builder.finish());
        let inv_batch = RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array])?;

        let mut kv_meta = vec![parquet::file::metadata::KeyValue {
            key: "analyzer".to_string(),
            value: Some("identity".to_string()),
        }];
        if is_timestamp_day {
            kv_meta.push(parquet::file::metadata::KeyValue {
                key: "source_type".to_string(),
                value: Some("timestamp".to_string()),
            });
        }

        let mut buffer = Vec::new();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_key_value_metadata(Some(kv_meta))
            .build();
        let mut writer = ArrowWriter::try_new(&mut buffer, inv_schema, Some(props))?;
        writer.write(&inv_batch)?;
        writer.close()?;

        Ok((buffer, is_timestamp_day))
    }
}

async fn read_puffin_footer_blobs(
    store: &dyn ObjectStore,
    store_path: &object_store::path::Path,
    local_path: Option<&str>,
) -> Result<Option<Vec<crate::core::puffin::PuffinBlobMetadata>>> {
    if let Some(lp) = local_path {
        if std::path::Path::new(lp).exists() {
            let file = File::open(lp)?;
            let reader = crate::core::puffin::PuffinReader::new(file)?;
            return Ok(Some(reader.footer().blobs.clone()));
        }
    }
    let meta = match store.head(store_path).await {
        Ok(m) => m,
        Err(e) if e.to_string().contains("not found") || e.to_string().contains("404") => {
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    };
    let size = meta.size;
    if size < 12 {
        return Ok(None);
    }
    let tail = store.get_range(store_path, (size - 12)..size).await?;
    if tail.len() != 12 || tail[8..12] != crate::core::puffin::PUFFIN_MAGIC {
        anyhow::bail!("Invalid Puffin magic at end of file {}", store_path);
    }
    let footer_size = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as u64;
    let flags = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
    if footer_size + 12 > size {
        anyhow::bail!("Corrupt Puffin footer size in {}", store_path);
    }
    let footer_start = size - 12 - footer_size;
    let footer_bytes = store
        .get_range(store_path, footer_start..(size - 12))
        .await?;
    let footer_payload = if flags & 0x01 != 0 {
        zstd::decode_all(&footer_bytes[..])
            .context("Failed to decompress Puffin footer with Zstd")?
    } else {
        footer_bytes.to_vec()
    };
    let footer: crate::core::puffin::PuffinFooter =
        serde_json::from_slice(&footer_payload).context("Failed to parse Puffin footer JSON")?;
    Ok(Some(footer.blobs))
}

async fn read_puffin_blob_bytes(
    store: &dyn ObjectStore,
    store_path: &object_store::path::Path,
    local_path: Option<&str>,
    blob: &crate::core::puffin::PuffinBlobMetadata,
) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut data = if let Some(lp) = local_path {
        let mut f = File::open(lp)?;
        f.seek(SeekFrom::Start(blob.offset as u64))?;
        let mut buf = vec![0u8; blob.length as usize];
        f.read_exact(&mut buf)?;
        buf
    } else {
        store
            .get_range(
                store_path,
                (blob.offset as u64)..((blob.offset + blob.length) as u64),
            )
            .await?
            .to_vec()
    };
    if let Some(ref codec) = blob.compression_codec {
        if codec == "zstd" {
            data = zstd::decode_all(&data[..])
                .context("Failed to decompress preserved Puffin blob with Zstd")?;
        } else {
            anyhow::bail!(
                "Unsupported compression codec '{}' on preserved Puffin blob",
                codec
            );
        }
    }
    Ok(data)
}

pub(crate) type JsonPathPostingData = (
    std::collections::BTreeMap<(String, String), Vec<u32>>,
    Vec<String>,
);

pub struct HybridSegmentWriter {
    pub(crate) config: SegmentConfig,
    // Store paths of created files for upload tracking
    pub(crate) generated_files: parking_lot::Mutex<Vec<String>>,
    // Accumulated Stats
    pub(crate) stats: parking_lot::Mutex<HashMap<String, crate::core::manifest::ColumnStats>>,
    pub record_count: std::sync::atomic::AtomicUsize,
    pub store: Option<Arc<dyn ObjectStore>>,
    pub primary_key: Vec<String>,
    pub index_configs: HashMap<String, crate::core::table::ColumnIndexConfig>,
    // Additive buffers for multi-batch indexing. The writer is constructed
    // per-flush and driven from a single task, but `build_indexes` fans columns
    // out through a rayon `into_par_iter`, so these locks can be touched
    // concurrently. `inverted_data` is therefore only held for a short merge
    // after each column tokenizes into a task-local map (see
    // `build_inverted_index`); the remaining buffers are written briefly, once
    // per column, and are effectively uncontended.
    //
    // Each posting entry is `(row_id, position)` — the ordinal token position
    // within the document. Storing positions enables phrase queries, span
    // queries, and highlighting, and also speeds up tf computation (count
    // positions per doc instead of counting duplicate row_ids).
    pub(crate) inverted_data:
        parking_lot::Mutex<HashMap<String, std::collections::BTreeMap<String, Vec<(u32, u32)>>>>,
    pub(crate) scalar_inverted_data: parking_lot::Mutex<HashMap<String, ScalarPostingMap>>,
    pub(crate) json_path_data: parking_lot::Mutex<HashMap<String, JsonPathPostingData>>,
    pub(crate) preserve_existing_index_files:
        parking_lot::Mutex<Vec<crate::core::manifest::IndexFile>>,
    pub index_metadata: parking_lot::Mutex<HashMap<String, String>>,
    pub(crate) vector_data: parking_lot::Mutex<HashMap<String, String>>,
    pub(crate) graph_data: parking_lot::Mutex<HashMap<String, String>>,
    pub(crate) file_checksum: parking_lot::Mutex<Option<String>>,
    pub(crate) puffin_index_entries: parking_lot::Mutex<Vec<crate::core::manifest::IndexFile>>,
    /// Table-level centroids to assign against, keyed by column.
    ///
    /// Empty for the default per-segment build. Populated by
    /// `Table::rebuild_indexes_with_global_centroids` so every segment in one
    /// rebuild pass shares the same cluster ids.
    pub(crate) global_centroids: parking_lot::Mutex<HashMap<String, Vec<Vec<f32>>>>,
    pub(crate) remote_staging_dir: std::sync::OnceLock<std::path::PathBuf>,
}

impl HybridSegmentWriter {
    pub fn new(config: SegmentConfig) -> Self {
        Self {
            config,
            generated_files: parking_lot::Mutex::new(Vec::new()),
            stats: parking_lot::Mutex::new(HashMap::new()),
            record_count: std::sync::atomic::AtomicUsize::new(0),
            store: None,
            primary_key: Vec::new(),
            index_configs: HashMap::new(),
            inverted_data: parking_lot::Mutex::new(HashMap::new()),
            scalar_inverted_data: parking_lot::Mutex::new(HashMap::new()),
            json_path_data: parking_lot::Mutex::new(HashMap::new()),
            preserve_existing_index_files: parking_lot::Mutex::new(Vec::new()),
            index_metadata: parking_lot::Mutex::new(HashMap::new()),
            vector_data: parking_lot::Mutex::new(HashMap::new()),
            graph_data: parking_lot::Mutex::new(HashMap::new()),
            file_checksum: parking_lot::Mutex::new(None),
            puffin_index_entries: parking_lot::Mutex::new(Vec::new()),
            global_centroids: parking_lot::Mutex::new(HashMap::new()),
            remote_staging_dir: std::sync::OnceLock::new(),
        }
    }

    pub fn with_preserve_existing_index_files(
        self,
        existing: Vec<crate::core::manifest::IndexFile>,
    ) -> Self {
        *self.preserve_existing_index_files.lock() = existing;
        self
    }

    fn get_or_init_remote_staging_dir(&self) -> Result<&std::path::PathBuf> {
        if let Some(dir) = self.remote_staging_dir.get() {
            return Ok(dir);
        }
        let temp_dir = std::env::temp_dir()
            .join("benostream_staging")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&temp_dir)?;
        let _ = self.remote_staging_dir.set(temp_dir);
        self.remote_staging_dir
            .get()
            .ok_or_else(|| anyhow::anyhow!("Failed to initialize remote staging directory"))
    }

    /// Assign this writer's vector builds against a table-level centroid set.
    pub fn with_global_centroids(mut self, centroids: HashMap<String, Vec<Vec<f32>>>) -> Self {
        self.global_centroids = parking_lot::Mutex::new(centroids);
        self
    }

    pub fn with_index_configs(
        mut self,
        configs: HashMap<String, crate::core::table::ColumnIndexConfig>,
    ) -> Self {
        self.index_configs = configs;
        self
    }

    pub fn with_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_existing_stats(self, stats: HashMap<String, ColumnStats>) -> Self {
        *self.stats.lock() = stats;
        self
    }

    pub fn with_record_count(self, count: usize) -> Self {
        self.record_count
            .store(count, std::sync::atomic::Ordering::SeqCst);
        self
    }

    pub fn set_store(&mut self, store: Arc<dyn ObjectStore>) {
        self.store = Some(store);
    }

    pub fn get_generated_files(&self) -> Vec<String> {
        self.generated_files.lock().clone()
    }

    pub fn get_stats(&self) -> HashMap<String, ColumnStats> {
        self.stats.lock().clone()
    }

    pub fn get_record_count(&self) -> usize {
        self.record_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn to_manifest_entry(&self) -> ManifestEntry {
        let stats = self.get_stats();
        let record_count = self.get_record_count() as i64;
        let files = self.get_generated_files();

        let mut parquet_file = String::new();
        let mut total_size = 0;

        // All secondary indexes are packed into the Puffin compound bundle;
        // the PuffinIndexWriter produces the IndexFile entries at bundle-finalize
        // time and stores them in puffin_index_entries.
        let index_files = self.puffin_index_entries.lock().clone();

        // Walk the generated files list to find the main Parquet data file and
        // its on-disk size; everything else is a Puffin artifact or temp file
        // and does not need a manifest entry.
        for f in &files {
            let filename = f.split('/').next_back().unwrap_or(f).to_string();
            if filename.ends_with(".parquet") {
                parquet_file = filename.clone();
                if let Ok(meta) = std::fs::metadata(f) {
                    total_size = meta.len() as i64;
                }
            }
        }

        ManifestEntry {
            file_path: self.config.parquet_path.clone().unwrap_or(parquet_file),
            file_size_bytes: total_size,
            record_count,
            index_files,
            delete_files: self.config.delete_files.clone(),
            column_stats: stats,
            partition_values: self.config.partition_values.clone().into_iter().collect(),
            clustering_strategy: None,
            clustering_columns: None,
            min_clustering_score: None,
            max_clustering_score: None,
            normalization_mins: None,
            normalization_maxs: None,
            file_checksum: self.file_checksum.lock().clone(),
            first_row_id: None,
        }
    }

    /// Compute vector statistics (BenoStream exclusive) while delegating
    /// scalar statistics to the Parquet writer metadata (Zero-Copy).
    fn compute_vector_stats(&self, _batch: &RecordBatch) -> Result<HashMap<String, VectorStats>> {
        // Bypassing vector stats computation (dim_min, dim_max) to maximize bulk ingestion throughput.
        // HNSW builds its own bounding boxes, and Parquet provides scalar stats.
        Ok(HashMap::new())
    }

    fn merge_parquet_stats(
        &self,
        metadata: &parquet::file::metadata::ParquetMetaData,
        vector_stats_map: HashMap<String, VectorStats>,
    ) -> Result<()> {
        let mut final_stats = self.stats.lock();

        if let Some(rg) = metadata.row_groups().first() {
            for col in rg.columns() {
                let col_name = col.column_path().string();
                let mut col_stats = ColumnStats::default();

                if let Some(stats) = col.statistics() {
                    // Extract common statistics regardless of type
                    // In parquet 57.x, these methods have _opt suffix on the enum
                    col_stats.null_count = stats.null_count_opt().unwrap_or(0) as i64;
                    col_stats.distinct_count = stats.distinct_count_opt().map(|v| v as i64);

                    match stats {
                        ParquetStats::Int32(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Int32(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Int32(v));
                        }
                        ParquetStats::Int64(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Int64(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Int64(v));
                        }
                        ParquetStats::Float(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Float32(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Float32(v));
                        }
                        ParquetStats::Double(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Float64(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Float64(v));
                        }
                        ParquetStats::ByteArray(s) => {
                            if let (Some(min_val), Some(max_val)) = (s.min_opt(), s.max_opt()) {
                                col_stats.min = std::str::from_utf8(min_val.as_ref())
                                    .ok()
                                    .map(|s| ManifestValue::String(s.to_string()));
                                col_stats.max = std::str::from_utf8(max_val.as_ref())
                                    .ok()
                                    .map(|s| ManifestValue::String(s.to_string()));
                            }
                        }
                        ParquetStats::Boolean(s) => {
                            col_stats.min = s.min_opt().map(|&v| ManifestValue::Boolean(v));
                            col_stats.max = s.max_opt().map(|&v| ManifestValue::Boolean(v));
                        }
                        _ => {}
                    }
                }

                // Merge in BenoStream-specific vector stats if applicable
                if let Some(v_stats) = vector_stats_map.get(&col_name) {
                    col_stats.vector_stats = Some(v_stats.clone());
                }

                final_stats.insert(col_name, col_stats);
            }
        }
        Ok(())
    }

    /// Write a batch of data to a Parquet file (fast path, no index building).
    /// Index building should be done asynchronously via build_indexes_async().
    pub fn write_batch(&self, batch: &RecordBatch) -> Result<()> {
        let is_remote =
            self.config.base_path.contains("://") && !self.config.base_path.starts_with("file://");
        let (path, _local_staging_dir) = if is_remote {
            let temp_dir = std::env::temp_dir()
                .join("benostream_staging")
                .join(uuid::Uuid::new_v4().to_string());
            std::fs::create_dir_all(&temp_dir)?;
            let filename = format!("{}.parquet", self.config.segment_id);
            (temp_dir.join(&filename), Some(temp_dir))
        } else {
            let base = self
                .config
                .base_path
                .strip_prefix("file://")
                .unwrap_or(&self.config.base_path);
            let base_path = std::path::Path::new(base);
            if !base.is_empty() {
                std::fs::create_dir_all(base_path)
                    .context("Failed to create local segment directory")?;
            }
            let p = if base.is_empty() {
                format!("{}.parquet", self.config.segment_id)
            } else {
                format!("{}/{}.parquet", base, self.config.segment_id)
            };
            (std::path::PathBuf::from(p), None)
        };

        let tmp_path = format!("{}.tmp", path.to_str().context("Invalid UTF-8 in path")?);

        // Zero-Copy Stats: Calculate vector stats using Rayon
        let vec_stats = self.compute_vector_stats(batch)?;

        // Write Data (Parquet) to temporary file
        let file = File::create(&tmp_path).context("Failed to create temporary segment file")?;
        let mut props_builder = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            // Page-level statistics. Chunk-level was enough for row-group
            // pruning, but the rerank path (`rescore_candidates`) reads a few
            // dozen rows scattered across a 1M-row group: without a page index
            // that is a 512 MB read per row group touched, and a 5M-row segment
            // has five of them. Page stats write the ColumnIndex/OffsetIndex so
            // `RowSelection` can skip to the page holding each candidate.
            // Page stats are a superset of chunk stats, so `merge_parquet_stats`
            // still sees the chunk min/max it needs.
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Page)
            // 32 KB pages hold 64 rows at 128 dims x f32. Small enough that a
            // 40-candidate rerank touches ~40 pages (~1.3 MB) instead of a whole
            // row group, large enough that the page index stays ~3 MB/segment
            // rather than ~13 MB at 8 KB pages.
            .set_data_page_size_limit(32 * 1024);

        // Enable Bloom Filters for Primary Keys if defined
        for pk in &self.primary_key {
            props_builder = props_builder.set_column_bloom_filter_enabled(
                parquet::schema::types::ColumnPath::from(pk.clone()),
                true,
            );
        }

        let props = props_builder.build();
        // Data files must carry Iceberg field ids. The writer derives the
        // Parquet schema (and therefore the ids) from this schema, while the
        // batch's arrays are written positionally, so only the schema needs the
        // annotation.
        let data_schema = Arc::new(with_parquet_field_ids(batch.schema().as_ref()));
        let mut writer = ArrowWriter::try_new(file, data_schema, Some(props))?;

        writer.write(batch)?;
        let metadata = writer.close()?; // Capture Zero-Copy metadata

        // Extract and Merge Parquet Stats
        self.merge_parquet_stats(&metadata, vec_stats)?;

        // Atomic rename
        std::fs::rename(&tmp_path, &path).context("Failed to atomically rename segment file")?;

        // Compute File Checksum
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;
        use std::io::Read;
        let mut f = File::open(&path)?;
        let mut buffer = [0; 65536];
        while let Ok(n) = f.read(&mut buffer) {
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        *self.file_checksum.lock() = Some(format!("{:x}", hasher.finalize()));

        {
            let mut files = self.generated_files.lock();
            files.push(path.to_str().context("Invalid UTF-8 in path")?.to_string());
        }

        tracing::info!(
            "Written data to {} ({} rows)",
            path.display(),
            batch.num_rows()
        );
        self.record_count
            .fetch_add(batch.num_rows(), std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Upload all generated files to ObjectStore if configured.
    /// Returns the final paths in the store.
    pub async fn upload_to_store(&self) -> Result<Vec<String>> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(self.get_generated_files()), // Local filesystem, already there
        };

        let files = self.get_generated_files();
        let mut final_paths = Vec::new();
        let rel_dir_from_parquet = self.config.parquet_path.as_deref().and_then(|p| {
            if !p.contains("://") && !p.starts_with('/') {
                Some(p.rfind('/').map(|pos| &p[..pos]).unwrap_or(""))
            } else {
                None
            }
        });

        for local_path in files {
            if !local_path.contains("benostream_staging") {
                // Not a staged file, assume it's already in the right place (or local)
                final_paths.push(local_path);
                continue;
            }

            let filename = local_path
                .split('/')
                .next_back()
                .context("Missing filename")?;

            let (store_path, output_path) = if let Some(rel_dir) = rel_dir_from_parquet {
                let rel_path = if rel_dir.is_empty() {
                    filename.to_string()
                } else {
                    format!("{}/{}", rel_dir, filename)
                };
                (object_store::path::Path::from(rel_path.as_str()), rel_path)
            } else {
                let remote_path = if self.config.base_path.contains("://") {
                    let mut base = self.config.base_path.clone();
                    if !base.ends_with('/') {
                        base.push('/');
                    }
                    format!("{}{}", base, filename)
                } else {
                    format!("{}/{}", self.config.base_path, filename)
                };

                // Parse remote_path to object_store::path::Path
                // e.g. s3://bucket/data/seg.parquet -> data/seg.parquet
                let sp = if remote_path.contains("://") {
                    let url = url::Url::parse(&remote_path)?;
                    object_store::path::Path::from(url.path().trim_start_matches('/'))
                } else {
                    object_store::path::Path::from(remote_path.clone())
                };
                (sp, remote_path)
            };

            let data = std::fs::read(&local_path)?;
            store.put(&store_path, data.into()).await?;

            // Cleanup local staging file
            let _ = std::fs::remove_file(&local_path);

            final_paths.push(output_path);
        }

        // Update generated_files with final paths
        {
            let mut g_files = self.generated_files.lock();
            *g_files = final_paths.clone();
        }

        Ok(final_paths)
    }

    /// Flush all buffered indexes (Inverted, HNSW, etc) to storage and track them.
    /// This should be called ONCE after all build_indexes() calls for a segment are complete.
    pub async fn finish_indexing(&self) -> Result<()> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No store configured for finishing indexes"))?;

        // Determine the directory prefix from the existing parquet path or base_path
        let parent_prefix_buf = if let Some(p) = &self.config.parquet_path {
            if let Some(pos) = p.rfind('/') {
                p[..pos + 1].to_string()
            } else {
                "".to_string()
            }
        } else if !self.config.base_path.is_empty() {
            let base = self
                .config
                .base_path
                .strip_prefix("file://")
                .unwrap_or(&self.config.base_path)
                .trim_end_matches('/');
            format!("{}/", base)
        } else {
            "".to_string()
        };
        let parent_prefix = parent_prefix_buf.as_str();

        let puffin_filename = format!("{}.puffin", self.config.segment_id);
        let puffin_path_str = format!("{}{}", parent_prefix, puffin_filename);

        let is_store_relative = self.config.parquet_path.as_ref().is_some_and(|p| {
            !p.contains("://")
                && !p.starts_with('/')
                && !puffin_path_str.contains("://")
                && !puffin_path_str.starts_with('/')
        });
        let is_local_file =
            !puffin_path_str.contains("://") || puffin_path_str.starts_with("file://");
        let local_puffin_target = puffin_path_str
            .strip_prefix("file://")
            .unwrap_or(&puffin_path_str);
        let use_local_atomic = !is_store_relative
            && is_local_file
            && std::path::Path::new(local_puffin_target)
                .parent()
                .is_some_and(|p| !p.as_os_str().is_empty() && p.exists());

        // Staging file on disk for streaming Puffin serialization (avoids gigabytes of RAM allocation).
        let staging_puffin_path = if use_local_atomic {
            format!(
                "{}_{}.tmp.puffin",
                local_puffin_target,
                uuid::Uuid::new_v4()
            )
        } else {
            std::env::temp_dir()
                .join(format!(
                    "{}_{}.tmp.puffin",
                    self.config.segment_id,
                    uuid::Uuid::new_v4()
                ))
                .to_string_lossy()
                .to_string()
        };

        let staging_file = std::fs::File::create(&staging_puffin_path)?;
        let mut puffin_writer = crate::core::puffin::PuffinIndexWriter::new(
            std::io::BufWriter::with_capacity(128 * 1024, staging_file),
            puffin_path_str.clone(),
            self.config.active_snapshot_id.unwrap_or(0),
            self.file_checksum
                .lock()
                .clone()
                .or_else(|| self.config.file_checksum.clone())
                .unwrap_or_default(),
            self.get_record_count() as i64,
        )?;

        // 1. Process Inverted Index Buffers - Drain the buffer into a local variable
        let inverted_data = {
            let mut inverted_lock = self.inverted_data.lock();
            std::mem::take(&mut *inverted_lock)
        }; // Guard is dropped here

        for (col_name, inverted_map) in inverted_data {
            tracing::info!(
                "Finishing Inverted Index for column '{}' ({} unique tokens)",
                col_name,
                inverted_map.len()
            );

            // BM25 bookkeeping: only string columns have an analyzer recorded
            // (build_inverted inserts into index_metadata for Utf8/LargeUtf8; the
            // Int32/Date32/Float inverted arms do not).
            let analyzer_name = self.index_metadata.lock().get(&col_name).cloned();

            if let Some(analyzer) = analyzer_name.as_ref() {
                // Per-row token counts from the inverted map, taken by reference
                // before the consuming loop below drains it. Rows with no tokens
                // are absent and read as 0 at query time.
                let mut doc_counts: HashMap<u32, u32> = HashMap::new();
                for postings in inverted_map.values() {
                    for &(row_id, _pos) in postings {
                        *doc_counts.entry(row_id).or_insert(0) += 1;
                    }
                }
                let mut doc_rows: Vec<(u32, u32)> = doc_counts.into_iter().collect();
                doc_rows.sort_unstable_by_key(|(row_id, _)| *row_id);

                let doclen_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                    arrow::datatypes::Field::new(
                        "row_id",
                        arrow::datatypes::DataType::UInt32,
                        false,
                    ),
                    arrow::datatypes::Field::new(
                        "token_count",
                        arrow::datatypes::DataType::UInt32,
                        false,
                    ),
                ]));
                let doc_rows_vec: Vec<u32> = doc_rows.iter().map(|(row_id, _)| *row_id).collect();
                let token_counts_vec: Vec<u32> = doc_rows.iter().map(|(_, count)| *count).collect();
                let doclen_batch = RecordBatch::try_new(
                    doclen_schema.clone(),
                    vec![
                        std::sync::Arc::new(arrow::array::UInt32Array::from(doc_rows_vec)),
                        std::sync::Arc::new(arrow::array::UInt32Array::from(token_counts_vec)),
                    ],
                )?;

                let mut doclen_buffer = Vec::new();
                {
                    let props = parquet::file::properties::WriterProperties::builder().build();
                    let mut writer =
                        ArrowWriter::try_new(&mut doclen_buffer, doclen_schema, Some(props))?;
                    writer.write(&doclen_batch)?;
                    writer.close()?;
                }
                puffin_writer.add_index_blob(
                    "lexical",
                    "doclen",
                    &col_name,
                    crate::core::puffin::PUFFIN_BLOB_LEXICAL_DOCLEN,
                    &doclen_buffer,
                    vec![],
                    HashMap::new(),
                )?;
                tracing::info!(
                    "  BM25 doc-length packed into Puffin bundle for column '{}' (analyzer: {})",
                    col_name,
                    analyzer
                );
            }

            // Build Arrow Arrays for Parquet:
            // Column 0: key (Utf8)
            // Column 1: row_ids (List<UInt32>) - unique doc IDs per term (delta-encoded)
            // Column 2: positions (List<List<UInt32>>) - ordinal token positions per doc
            let mut key_builder = arrow::array::StringBuilder::new();
            let value_builder = arrow::array::UInt32Builder::new();
            let mut list_builder = arrow::array::ListBuilder::new(value_builder);

            let pos_inner_builder = arrow::array::UInt32Builder::new();
            let pos_doc_builder = arrow::array::ListBuilder::new(pos_inner_builder);
            let mut positions_builder = arrow::array::ListBuilder::new(pos_doc_builder);

            for (key, mut postings) in inverted_map {
                key_builder.append_value(&key);
                postings.sort_unstable_by_key(|&(row_id, pos)| (row_id, pos));

                // Group postings by unique row_id
                let mut doc_postings: Vec<(u32, Vec<u32>)> = Vec::new();
                for (row_id, pos) in postings {
                    if let Some(last) = doc_postings.last_mut() {
                        if last.0 == row_id {
                            last.1.push(pos);
                            continue;
                        }
                    }
                    doc_postings.push((row_id, vec![pos]));
                }

                let mut last_id = 0;
                for (row_id, positions) in doc_postings {
                    list_builder.values().append_value(row_id - last_id);
                    last_id = row_id;

                    let doc_pos_builder = positions_builder.values();
                    for pos in positions {
                        doc_pos_builder.values().append_value(pos);
                    }
                    doc_pos_builder.append(true);
                }
                list_builder.append(true);
                positions_builder.append(true);
            }

            let key_array = std::sync::Arc::new(key_builder.finish());
            let list_array = std::sync::Arc::new(list_builder.finish());
            let pos_array = std::sync::Arc::new(positions_builder.finish());

            let inv_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new(
                    "row_ids",
                    arrow::datatypes::DataType::List(std::sync::Arc::new(
                        arrow::datatypes::Field::new(
                            "item",
                            arrow::datatypes::DataType::UInt32,
                            true,
                        ),
                    )),
                    false,
                ),
                arrow::datatypes::Field::new(
                    "positions",
                    arrow::datatypes::DataType::List(std::sync::Arc::new(
                        arrow::datatypes::Field::new(
                            "item",
                            arrow::datatypes::DataType::List(std::sync::Arc::new(
                                arrow::datatypes::Field::new(
                                    "item",
                                    arrow::datatypes::DataType::UInt32,
                                    true,
                                ),
                            )),
                            true,
                        ),
                    )),
                    false,
                ),
            ]));

            let inv_batch =
                RecordBatch::try_new(inv_schema.clone(), vec![key_array, list_array, pos_array])?;
            // Write to memory buffer then to store. String-indexed columns embed
            // the analyzer name in key/value metadata so readers re-tokenize
            // queries identically.
            let mut buffer = Vec::new();
            {
                let kv_meta = analyzer_name.as_ref().map(|name| {
                    vec![parquet::file::metadata::KeyValue {
                        key: "analyzer".to_string(),
                        value: Some(name.clone()),
                    }]
                });
                let props = parquet::file::properties::WriterProperties::builder()
                    .set_key_value_metadata(kv_meta)
                    .build();
                let mut writer = ArrowWriter::try_new(&mut buffer, inv_schema, Some(props))?;
                writer.write(&inv_batch)?;
                writer.close()?;
            }

            let mut props = HashMap::new();
            if let Some(ref a) = analyzer_name {
                props.insert("analyzer".to_string(), a.clone());
            }
            puffin_writer.add_index_blob(
                "lexical",
                "inverted",
                &col_name,
                crate::core::puffin::PUFFIN_BLOB_LEXICAL_BM25,
                &buffer,
                vec![],
                props,
            )?;
            tracing::info!(
                "  Inverted Index packed into Puffin bundle for column '{}'",
                col_name
            );
        }

        // 2. Process Vector Index Buffers (Out of Core)
        let vector_data = {
            let mut vector_lock = self.vector_data.lock();
            std::mem::take(&mut *vector_lock)
        };

        for (col_name, tmp_path) in vector_data {
            tracing::info!(
                "Finishing Vector Index for column '{}' from out-of-core file",
                col_name
            );
            let mut algos = self
                .index_configs
                .get(&col_name)
                .map(|c| c.algorithms.clone())
                .unwrap_or_else(|| {
                    self.config
                        .column_algorithms
                        .get(&col_name)
                        .cloned()
                        .unwrap_or_default()
                });
            if algos.is_empty() {
                algos.push(crate::core::manifest::IndexAlgorithm::default());
            }

            for (idx, algo) in algos.iter().enumerate() {
                let metric = match algo {
                    crate::core::manifest::IndexAlgorithm::Hnsw { metric, .. }
                    | crate::core::manifest::IndexAlgorithm::HnswPq { metric, .. }
                    | crate::core::manifest::IndexAlgorithm::HnswTq4 { metric, .. }
                    | crate::core::manifest::IndexAlgorithm::HnswTq8 { metric, .. }
                    | crate::core::manifest::IndexAlgorithm::HnswSq8 { metric, .. } => metric
                        .parse::<crate::core::index::VectorMetric>()
                        .with_context(|| {
                            format!("Invalid vector metric for {}: {}", col_name, metric)
                        })?,
                    _ => crate::core::index::VectorMetric::L2,
                };

                let algo_id = match algo {
                    crate::core::manifest::IndexAlgorithm::Hnsw { .. } => "hnsw",
                    crate::core::manifest::IndexAlgorithm::HnswPq { .. } => "pq",
                    crate::core::manifest::IndexAlgorithm::HnswTq4 { .. } => "tq4",
                    crate::core::manifest::IndexAlgorithm::HnswTq8 { .. } => "tq8",
                    crate::core::manifest::IndexAlgorithm::HnswSq8 { .. } => "sq8",
                    _ => "idx",
                };

                let suffix = if algos.len() > 1 {
                    format!("{}.{}.{}", col_name, algo_id, idx)
                } else {
                    format!("{}.{}", col_name, algo_id)
                };

                // Get the staging directory from the tmp_path
                let tmp_path_buf = std::path::PathBuf::from(&tmp_path);
                let local_staging_dir = tmp_path_buf
                    .parent()
                    .context("temporary vector file has no parent directory")?;

                let local_base_path =
                    local_staging_dir.join(format!("{}.{}", self.config.segment_id, suffix));

                // Streaming build and disk serialization: avoids keeping all bucket HNSW graphs in RAM.
                // When the table supplied global centroids, assign against those
                // instead of training per-segment k-means.
                let table_centroids = self.global_centroids.lock().get(&col_name).cloned();
                let saved_files = match table_centroids {
                    Some(centroids) if !centroids.is_empty() => {
                        crate::core::index::hnsw_ivf::HnswIvfIndex::build_and_save_from_file_with_centroids(
                            &tmp_path,
                            local_base_path.to_str().context("Invalid UTF-8 in path")?,
                            metric,
                            centroids,
                            None,
                            algo,
                        )?
                    }
                    _ => crate::core::index::hnsw_ivf::HnswIvfIndex::build_and_save_from_file(
                        &tmp_path,
                        local_base_path.to_str().context("Invalid UTF-8 in path")?,
                        metric,
                        None,
                        None,
                        algo,
                    )?,
                };

                {
                    let mut meta = self.index_metadata.lock();
                    meta.insert(
                        format!("{}.{}", self.config.segment_id, suffix),
                        algo.to_string(),
                    );
                }

                // Pack vector index artifacts into the Puffin bundle via streaming reader.
                for local_file in &saved_files {
                    let file_name = std::path::Path::new(local_file)
                        .file_name()
                        .context("saved index file has no file name")?
                        .to_string_lossy()
                        .to_string();

                    let blob_type = if file_name.ends_with(".hnsw.graph") {
                        crate::core::puffin::PUFFIN_BLOB_VECTOR_HNSW_GRAPH
                    } else if file_name.contains(".centroids.parquet") {
                        crate::core::puffin::PUFFIN_BLOB_HNSW_IVF_CENTROIDS
                    } else if file_name.contains(".mapping.parquet") {
                        crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_MAPPING
                    } else if file_name.contains(".cluster_") && file_name.ends_with(".hnsw.graph")
                    {
                        crate::core::puffin::PUFFIN_BLOB_HNSW_CLUSTER_GRAPH
                    } else {
                        continue;
                    };

                    let mut props = HashMap::new();
                    props.insert("filename".to_string(), file_name.clone());
                    puffin_writer.add_index_blob_from_file(
                        "vector",
                        algo_id,
                        &col_name,
                        blob_type,
                        local_file,
                        vec![],
                        props,
                    )?;
                    // Clean up the local temp file after packing.
                    let _ = std::fs::remove_file(local_file);
                }
            }

            // Cleanup the temporary raw vector file
            let _ = std::fs::remove_file(&tmp_path);
        }

        // 3. Process Graph Index Buffers
        let graph_data = {
            let mut graph_lock = self.graph_data.lock();
            std::mem::take(&mut *graph_lock)
        };

        for (col_name, tmp_path) in graph_data {
            tracing::info!(
                "Finishing Graph Index for virtual column '{}' from tmp file",
                col_name
            );

            // Get the staging directory from the tmp_path
            let tmp_path_buf = std::path::PathBuf::from(&tmp_path);
            let local_staging_dir = tmp_path_buf
                .parent()
                .context("temporary graph file has no parent directory")?;

            let local_base_path =
                local_staging_dir.join(format!("{}.{}", self.config.segment_id, col_name));

            let saved_files = crate::core::index::csr_graph::MmapCsrGraph::build_from_file(
                &tmp_path_buf,
                &local_base_path,
            )
            .map_err(|e| anyhow::anyhow!("Graph build failed: {}", e))?;

            // Pack graph index artifacts into the Puffin bundle via streaming reader.
            for local_file in &saved_files {
                let file_name = std::path::Path::new(local_file)
                    .file_name()
                    .context("saved index file has no file name")?
                    .to_string_lossy()
                    .to_string();

                let blob_type = if file_name.ends_with(".csr.offsets") {
                    crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_OFFSETS
                } else if file_name.ends_with(".csr.edges") {
                    crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_EDGES
                } else if file_name.ends_with(".csr.dict") {
                    crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_DICT
                } else {
                    continue;
                };

                let mut props = HashMap::new();
                props.insert("filename".to_string(), file_name.clone());
                puffin_writer.add_index_blob_from_file(
                    "graph_v2",
                    "csr_graph",
                    &col_name,
                    blob_type,
                    local_file,
                    vec![],
                    props,
                )?;
                // Clean up the local temp file after packing.
                let _ = std::fs::remove_file(local_file);
            }

            {
                let mut meta = self.index_metadata.lock();
                meta.insert(
                    format!("{}.{}", self.config.segment_id, col_name),
                    "csr_graph".to_string(),
                );
            }

            // Cleanup
            let _ = std::fs::remove_file(&tmp_path);
        }

        // 4. Pack accumulated scalar inverted-index buffers directly into Puffin
        // without intermediate staging files or single-batch clobbering.
        let scalar_inverted_data = {
            let mut lock = self.scalar_inverted_data.lock();
            std::mem::take(&mut *lock)
        };
        for (col_name, scalar_map) in scalar_inverted_data {
            let (buffer, is_timestamp_day) = scalar_map.into_parquet_bytes()?;
            let mut props = HashMap::new();
            props.insert("analyzer".to_string(), "identity".to_string());
            if is_timestamp_day {
                props.insert("source_type".to_string(), "timestamp".to_string());
            }
            puffin_writer.add_index_blob(
                "lexical",
                "inverted",
                &col_name,
                crate::core::puffin::PUFFIN_BLOB_LEXICAL_BM25,
                &buffer,
                vec![],
                props,
            )?;
            tracing::info!(
                "  Scalar Inverted Index packed into Puffin bundle for column '{}'",
                col_name
            );
        }

        // 4b. Pack accumulated JSON-path overlays into Puffin with accurate
        // segment-wide row_ids and metadata.
        let json_path_data = {
            let mut lock = self.json_path_data.lock();
            std::mem::take(&mut *lock)
        };
        for (col_name, (postings, covered)) in json_path_data {
            let buffer =
                crate::core::index::json_path::build_json_path_parquet_bytes(postings, covered)?;
            puffin_writer.add_index_blob(
                "json_path",
                "json_path",
                &col_name,
                crate::core::puffin::PUFFIN_BLOB_JSON_PATH,
                &buffer,
                vec![],
                HashMap::new(),
            )?;
            tracing::info!(
                "  JSON-path index packed into Puffin bundle for column '{}'",
                col_name
            );
        }

        // 5. Preserve untouched blobs from an existing Puffin bundle when
        // backfilling or selectively reindexing a subset of columns/families.
        let existing_index_files = {
            let mut lock = self.preserve_existing_index_files.lock();
            std::mem::take(&mut *lock)
        };
        if !existing_index_files.is_empty() {
            let rel_pq = self.config.parquet_path.as_deref().unwrap_or("");
            let rebuilt_families: std::collections::HashSet<(String, String)> = puffin_writer
                .entries()
                .iter()
                .map(|e| {
                    (
                        e.column_name.clone(),
                        normalize_index_family(&e.index_category).to_string(),
                    )
                })
                .collect();

            let to_preserve: Vec<crate::core::manifest::IndexFile> = existing_index_files
                .into_iter()
                .filter(|old_idx| {
                    let resolved = resolve_entry_index_path(rel_pq, &old_idx.file_path);
                    let same_puffin = resolved == puffin_path_str
                        || (resolved.ends_with(".puffin")
                            && segment_id_from_path(&resolved) == self.config.segment_id);
                    same_puffin
                        && !rebuilt_families.contains(&(
                            old_idx.column_name.clone(),
                            normalize_index_family(&old_idx.index_category).to_string(),
                        ))
                })
                .collect();

            if !to_preserve.is_empty() {
                let old_store_path = if puffin_path_str.contains("://") {
                    let url = url::Url::parse(&puffin_path_str)?;
                    object_store::path::Path::from(url.path().trim_start_matches('/'))
                } else {
                    object_store::path::Path::from(puffin_path_str.as_str())
                };
                let local_target_opt = if use_local_atomic {
                    Some(local_puffin_target)
                } else {
                    None
                };
                if let Some(footer_blobs) =
                    read_puffin_footer_blobs(store.as_ref(), &old_store_path, local_target_opt)
                        .await?
                {
                    let mut seen_ranges = std::collections::HashSet::new();
                    for old_idx in to_preserve {
                        let col_marker = format!(".{}.", old_idx.column_name);
                        let matched_blob = old_idx
                            .blob_offset
                            .zip(old_idx.blob_length)
                            .and_then(|(off, len)| {
                                footer_blobs
                                    .iter()
                                    .find(|b| b.offset == off && b.length == len)
                            })
                            .or_else(|| {
                                footer_blobs
                                    .iter()
                                    .find(|b| {
                                        old_idx.blob_type.as_deref() == Some(b.r#type.as_str())
                                            && b.properties
                                                .get("filename")
                                                .is_some_and(|f| f.contains(&col_marker))
                                    })
                                    .or_else(|| {
                                        footer_blobs.iter().find(|b| {
                                            old_idx.blob_type.as_deref() == Some(b.r#type.as_str())
                                        })
                                    })
                            });
                        let Some(blob_meta) = matched_blob else {
                            continue;
                        };
                        if !seen_ranges.insert((blob_meta.offset, blob_meta.length)) {
                            continue;
                        }
                        let blob_type = old_idx.blob_type.as_deref().unwrap_or(&blob_meta.r#type);
                        if blob_meta.compression_codec.is_none() && local_target_opt.is_none() {
                            puffin_writer
                                .add_index_blob_from_store_range(
                                    store.as_ref(),
                                    &old_store_path,
                                    blob_meta.offset as u64,
                                    blob_meta.length as u64,
                                    &old_idx.index_category,
                                    &old_idx.algorithm,
                                    &old_idx.column_name,
                                    blob_type,
                                    blob_meta.fields.clone(),
                                    blob_meta.properties.clone(),
                                )
                                .await?;
                        } else {
                            let bytes = read_puffin_blob_bytes(
                                store.as_ref(),
                                &old_store_path,
                                local_target_opt,
                                blob_meta,
                            )
                            .await?;
                            puffin_writer.add_index_blob(
                                &old_idx.index_category,
                                &old_idx.algorithm,
                                &old_idx.column_name,
                                blob_type,
                                &bytes,
                                blob_meta.fields.clone(),
                                blob_meta.properties.clone(),
                            )?;
                        }
                    }
                }
            }
        }

        let (mut buf_writer, puffin_entries) = puffin_writer.finish()?;
        use std::io::Write;
        buf_writer.flush()?;
        drop(buf_writer);

        if use_local_atomic {
            // Local lakehouse (with non-relative path): zero-copy atomic rename on disk!
            std::fs::rename(&staging_puffin_path, local_puffin_target)?;
        } else {
            let store_path = if puffin_path_str.contains("://") {
                let url = url::Url::parse(&puffin_path_str)?;
                object_store::path::Path::from(url.path().trim_start_matches('/'))
            } else {
                object_store::path::Path::from(puffin_path_str.clone())
            };
            let puffin_bytes = std::fs::read(&staging_puffin_path)?;
            store.put(&store_path, puffin_bytes.into()).await?;
            let _ = std::fs::remove_file(&staging_puffin_path);
        }
        {
            let mut files = self.generated_files.lock();
            files.push(puffin_path_str.clone());
        }
        {
            let mut entries = self.puffin_index_entries.lock();
            *entries = puffin_entries;
        }
        tracing::info!(
            "  Puffin compound index bundle written to storage: {}",
            puffin_path_str
        );

        Ok(())
    }

    /// Build indexes for a batch (can be called asynchronously after write_batch).
    /// This is the expensive operation that should run in background.
    pub fn build_indexes(&self, batch: &RecordBatch, row_offset: usize) -> Result<()> {
        crate::core::run_blocking(|| {
            indexing_pool().install(|| {
                tracing::info!(
                    "Building indexes for batch of {} rows at offset {}",
                    batch.num_rows(),
                    row_offset
                );
                let schema = batch.schema();
                let _fields = schema.fields();

                // Build Indexes

                batch
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .try_for_each(|(i, field)| {
                        let col_name = field.name();
                        let col = batch.column(i);

                        let is_pk = self.primary_key.contains(&col_name.to_string());
                        let is_vector = matches!(
                            col.data_type(),
                            arrow::datatypes::DataType::FixedSizeList(_, _)
                                | arrow::datatypes::DataType::List(_)
                        );
                        let in_config_list = self
                            .config
                            .columns_to_index
                            .as_ref()
                            .map(|cols| cols.contains(&col_name.to_string()))
                            .unwrap_or(false);

                        if self.config.index_all || is_pk || is_vector || in_config_list {
                            self.index_column(col_name, col, row_offset)
                        } else {
                            Ok(())
                        }
                    })?;

                // 2. Build Composite Indexes (Virtual Columns)
                let mut composite_tasks = Vec::new();
                let mut graph_tasks = Vec::new();
                for (col_name, config) in &self.index_configs {
                    if !config.enabled {
                        continue;
                    }
                    for alg in &config.algorithms {
                        if let crate::core::manifest::IndexAlgorithm::CompositeBitmap { columns } =
                            alg
                        {
                            composite_tasks.push((col_name.clone(), columns.clone()));
                        } else if let crate::core::manifest::IndexAlgorithm::CsrGraph {
                            src_column,
                            dst_column,
                        } = alg
                        {
                            graph_tasks.push((
                                col_name.clone(),
                                src_column.clone(),
                                dst_column.clone(),
                            ));
                        }
                    }
                }

                composite_tasks
                    .into_par_iter()
                    .try_for_each(|(col_name, columns)| {
                        let mut builder = arrow::array::StringBuilder::new();
                        let num_rows = batch.num_rows();

                        let mut arrays = Vec::new();
                        for col in &columns {
                            if let Some(arr) = batch.column_by_name(col) {
                                arrays.push(arr);
                            } else {
                                return Err(anyhow::anyhow!(
                                    "Column {} not found for composite index",
                                    col
                                ));
                            }
                        }

                        for row in 0..num_rows {
                            let mut joined = String::new();
                            for (i, arr) in arrays.iter().enumerate() {
                                if i > 0 {
                                    joined.push('\0');
                                }
                                let val =
                                    crate::core::manifest::ManifestValue::from_array(arr, row)
                                        .to_string();
                                joined.push_str(&val);
                            }
                            builder.append_value(&joined);
                        }
                        let composite_array =
                            std::sync::Arc::new(builder.finish()) as std::sync::Arc<dyn Array>;

                        self.index_column(&col_name, &composite_array, row_offset)
                    })?;

                // 3. Build Graph Indexes
                graph_tasks
                    .into_par_iter()
                    .try_for_each(|(col_name, src_col, dst_col)| {
                        let is_remote = self.config.base_path.is_empty()
                            || (self.config.base_path.contains("://")
                                && !self.config.base_path.starts_with("file://"));
                        let local_staging_dir = if is_remote {
                            self.get_or_init_remote_staging_dir()?.clone()
                        } else {
                            let path = self
                                .config
                                .base_path
                                .strip_prefix("file://")
                                .unwrap_or(&self.config.base_path);
                            let p = std::path::PathBuf::from(path);
                            if !path.is_empty() {
                                std::fs::create_dir_all(&p)?;
                            }
                            p
                        };

                        self.build_graph_index(
                            &col_name,
                            batch,
                            &src_col,
                            &dst_col,
                            row_offset,
                            &local_staging_dir,
                        )
                    })?;

                Ok(())
            }) // end indexing_pool
        }) // end block_in_place
    }

    /// Build index for a single column.
    /// Can be called during ingestion OR for post-hoc backfilling.
    pub fn index_column(
        &self,
        col_name: &str,
        col_array: &std::sync::Arc<dyn Array>,
        row_offset: usize,
    ) -> Result<()> {
        // Apply per-column device override if specified
        let desired_device = self
            .config
            .column_devices
            .get(col_name)
            .or(self.config.default_device.as_ref());

        if let Some(device_str) = desired_device {
            tracing::info!(
                "Applying device override for column {}: {}",
                col_name,
                device_str
            );

            // Avoid creating a new context if the existing one already matches the desired device backend.
            // CudaDevice::new() allocates memory, so we want to reuse it.
            let mut needs_new_context = true;
            if let Some(existing_ctx) = crate::core::index::gpu::get_thread_gpu_context() {
                if existing_ctx.backend_name() == device_str
                    || (device_str == "auto" && existing_ctx.is_available())
                {
                    needs_new_context = false;
                }
            }

            if needs_new_context {
                if let Ok(ctx) = ComputeContext::from_device_str(device_str) {
                    tracing::info!(
                        "Successfully set global GPU context to {:?} for column {}",
                        ctx.backend,
                        col_name
                    );
                    set_thread_gpu_context(Some(ctx));
                } else {
                    tracing::warn!("Failed to parse device string: {}", device_str);
                }
            }
        }

        // OPT-IN CHECK: Only index if configured, or if it's a Vector or Primary Key column
        let config = self.index_configs.get(col_name);
        let is_pk = self.primary_key.contains(&col_name.to_string());
        let is_vector = matches!(
            col_array.data_type(),
            arrow::datatypes::DataType::FixedSizeList(_, _) | arrow::datatypes::DataType::List(_)
        );
        let in_config_list = self
            .config
            .columns_to_index
            .as_ref()
            .map(|cols| cols.contains(&col_name.to_string()))
            .unwrap_or(false);

        if !is_pk
            && !is_vector
            && !self.config.index_all
            && !in_config_list
            && !config.map(|c| c.enabled).unwrap_or(false)
        {
            // Skip indexing for this column! (Massive speed gain for multi-column tables)
            return Ok(());
        }

        // Create a local staging directory if base_path is empty or a remote URI
        let is_remote = self.config.base_path.is_empty()
            || (self.config.base_path.contains("://")
                && !self.config.base_path.starts_with("file://"));
        let local_staging_dir = if is_remote {
            self.get_or_init_remote_staging_dir()?.clone()
        } else {
            let path = self
                .config
                .base_path
                .strip_prefix("file://")
                .unwrap_or(&self.config.base_path);
            let p = std::path::PathBuf::from(path);
            if !path.is_empty() {
                std::fs::create_dir_all(&p)?;
            }
            p
        };

        match col_array.data_type() {
            arrow::datatypes::DataType::List(_)
            | arrow::datatypes::DataType::FixedSizeList(_, _) => {
                self.build_vector_index(col_name, col_array, row_offset, &local_staging_dir)?;
            }
            _ => {
                self.build_inverted_index(col_name, col_array, row_offset, &local_staging_dir)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListArray, Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Does `ArrowWriter::close()` hand back metadata that still carries the
    /// chunk-level statistics we just asked it to write?
    ///
    /// `merge_parquet_stats` populates `column_stats` from exactly that
    /// metadata, so if the stats are absent here, every segment ends up with
    /// empty `column_stats` and the planner's stats pruning can never fire.
    #[test]
    fn parquet_close_metadata_carries_chunk_statistics() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let props = parquet::file::properties::WriterProperties::builder()
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
            .build();

        let mut buf: Vec<u8> = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        let metadata = writer.close().unwrap();

        let rg = metadata.row_groups().first().expect("one row group");
        let mut seen = Vec::new();
        for col in rg.columns() {
            seen.push((
                col.column_path().string(),
                col.statistics().is_some(),
                col.statistics()
                    .and_then(|s| s.null_count_opt())
                    .map(|_| "nulls"),
            ));
        }
        println!("close() metadata statistics: {seen:?}");
        assert!(
            seen.iter().any(|(_, has, _)| *has),
            "ArrowWriter::close() metadata carried no column statistics: {seen:?}"
        );
    }

    /// Does the real write path actually populate `column_stats`?
    ///
    /// The parquet metadata carries statistics (previous test), so if this
    /// comes back empty the loss is between `write_batch` and
    /// `to_manifest_entry`.
    #[tokio::test]
    async fn write_batch_populates_column_stats() -> Result<()> {
        use arrow::array::{Int64Array, StringArray};

        let dir = tempfile::tempdir()?;
        let base = dir.path().to_str().unwrap().to_string();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![40, 41, 42])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )?;

        let w = HybridSegmentWriter::new(SegmentConfig::new(&base, "seg_stats_test"));
        w.write_batch(&batch)?;
        let stats = w.get_stats();
        println!("column_stats after write_batch: {stats:?}");
        assert!(
            stats.get("id").and_then(|s| s.min.as_ref()).is_some(),
            "id min missing from column_stats: {stats:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_write_hybrid_segment() -> Result<()> {
        // 1. Setup Data: Int32 Column + Vector Column
        let dim = 4;
        let num_rows = 10;

        let id_array = Int32Array::from((0..num_rows).collect::<Vec<i32>>());

        // 10 vectors of dim 4
        let mut values = Vec::new();
        for i in 0..num_rows {
            for j in 0..dim {
                values.push((i + j) as f32);
            }
        }
        let values_array = Float32Array::from(values);
        let vectors_array = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(values_array),
            None,
        )?;

        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_array), Arc::new(vectors_array)],
        )?;

        // 2. Write Segment
        let tmp_dir = std::env::temp_dir();
        let config = SegmentConfig::new(
            tmp_dir.to_str().context("Invalid UTF-8 in path")?,
            "test_segment_001",
        )
        .with_index_all(true);
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());
        let writer = HybridSegmentWriter::new(config.clone()).with_store(store);

        writer.write_batch(&batch)?;

        // Build indexes (required for index files to be created)
        writer.build_indexes(&batch, 0)?;
        writer.finish_indexing().await?;

        // 3. Verify Files
        let base = format!("{}/{}", config.base_path, config.segment_id);

        // Parquet
        assert!(
            std::path::Path::new(&format!("{}.parquet", base)).exists(),
            "Parquet file should exist"
        );

        // All secondary indexes are packed into a single Puffin compound bundle.
        let puffin_path = format!("{}.puffin", base);
        assert!(
            std::path::Path::new(&puffin_path).exists(),
            "Puffin compound index bundle should exist"
        );
        let bytes = std::fs::read(&puffin_path)?;
        let reader = crate::core::puffin::PuffinReader::new(std::io::Cursor::new(bytes))?;
        let types: Vec<&str> = reader
            .footer()
            .blobs
            .iter()
            .map(|b| b.r#type.as_str())
            .collect();
        assert!(
            types.contains(&crate::core::puffin::PUFFIN_BLOB_LEXICAL_BM25),
            "Puffin bundle should contain the lexical (inverted) index for `id` (has {types:?})"
        );
        assert!(
            types.contains(&crate::core::puffin::PUFFIN_BLOB_HNSW_IVF_CENTROIDS),
            "Puffin bundle should contain the vector centroids (has {types:?})"
        );
        assert!(
            types.contains(&crate::core::puffin::PUFFIN_BLOB_VECTOR_HNSW_GRAPH),
            "Puffin bundle should contain the vector HNSW graph (has {types:?})"
        );

        Ok(())
    }

    #[test]
    fn segment_path_helpers_work_across_schemes() {
        assert_eq!(
            segment_id_from_path("file:///tmp/tbl/cat=a/seg_1.parquet"),
            "seg_1"
        );
        assert_eq!(segment_id_from_path("cat=a/seg_1.parquet"), "seg_1");
        assert_eq!(segment_id_from_path("seg_1"), "seg_1");

        assert_eq!(ensure_prefix("", "seg_1.puffin"), "seg_1.puffin");
        assert_eq!(ensure_prefix("cat=a", "seg_1.puffin"), "cat=a/seg_1.puffin");
        assert_eq!(
            ensure_prefix("cat=a/", "cat=a/seg_1.puffin"),
            "cat=a/seg_1.puffin"
        );

        assert_eq!(
            relativize_against("file:///tmp/tbl", "file:///tmp/tbl/cat=a/seg_1.parquet"),
            "cat=a/seg_1.parquet"
        );
        assert_eq!(
            relativize_against("/tmp/tbl/", "/tmp/tbl/cat=a/seg_1.parquet"),
            "cat=a/seg_1.parquet"
        );
        assert_eq!(
            relativize_against("s3://bucket/tbl/", "s3://bucket/tbl/cat=a/seg_1.parquet"),
            "cat=a/seg_1.parquet"
        );
        assert_eq!(
            relativize_against("memory://tbl", "cat=a/seg_1.parquet"),
            "cat=a/seg_1.parquet"
        );
        assert_eq!(
            relativize_against("/tmp/tbl_other", "/tmp/tbl/seg_1.parquet"),
            "seg_1.parquet"
        );

        assert_eq!(
            resolve_entry_index_path("cat=a/seg_1.parquet", "seg_1.puffin"),
            "cat=a/seg_1.puffin"
        );
        assert_eq!(
            resolve_entry_index_path("cat=a/seg_1.parquet", "cat=a/seg_1.puffin"),
            "cat=a/seg_1.puffin"
        );
        assert_eq!(
            resolve_entry_index_path("seg_1.parquet", "seg_1.puffin"),
            "seg_1.puffin"
        );
    }

    #[tokio::test]
    async fn upload_to_store_preserves_partition_prefix_on_s3_and_local() -> Result<()> {
        let mem_store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2]))])?;

        // Remote/prefixed S3-like table URI with store-relative parquet_path under Hive partition
        let cfg = SegmentConfig::new("s3://bucket/tbl/cat=a", "seg_part_remote")
            .with_parquet_path("cat=a/seg_part_remote.parquet".to_string())
            .with_index_all(true);
        let writer = HybridSegmentWriter::new(cfg).with_store(mem_store.clone());
        writer.write_batch(&batch)?;
        writer.build_indexes(&batch, 0)?;
        writer.finish_indexing().await?;
        let uploaded = writer.upload_to_store().await?;
        assert!(
            uploaded
                .iter()
                .any(|p| p == "cat=a/seg_part_remote.parquet"),
            "expected partitioned parquet upload, got {uploaded:?}"
        );
        assert!(
            mem_store
                .head(&object_store::path::Path::from(
                    "cat=a/seg_part_remote.parquet"
                ))
                .await
                .is_ok(),
            "partitioned parquet missing in store"
        );
        assert!(
            mem_store
                .head(&object_store::path::Path::from(
                    "cat=a/seg_part_remote.puffin"
                ))
                .await
                .is_ok(),
            "partitioned puffin bundle missing in store"
        );

        // Local directory with store-relative parquet_path under Hive partition
        let dir = tempfile::tempdir()?;
        let table_root = dir.path();
        let part_dir = table_root.join("cat=b");
        std::fs::create_dir_all(&part_dir)?;
        let local_store: Arc<dyn object_store::ObjectStore> = Arc::new(
            object_store::local::LocalFileSystem::new_with_prefix(table_root)?,
        );
        let local_cfg =
            SegmentConfig::new(part_dir.to_str().context("valid utf8")?, "seg_part_local")
                .with_parquet_path("cat=b/seg_part_local.parquet".to_string())
                .with_index_all(true);
        let local_writer = HybridSegmentWriter::new(local_cfg).with_store(local_store.clone());
        local_writer.write_batch(&batch)?;
        local_writer.build_indexes(&batch, 0)?;
        local_writer.finish_indexing().await?;
        assert!(table_root.join("cat=b/seg_part_local.parquet").exists());
        assert!(table_root.join("cat=b/seg_part_local.puffin").exists());

        Ok(())
    }
}
