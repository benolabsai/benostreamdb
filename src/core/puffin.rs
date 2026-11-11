// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{Context, Result};
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

/// Puffin File Magic Bytes: PFA1
pub const PUFFIN_MAGIC: [u8; 4] = [0x50, 0x46, 0x41, 0x31];

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PuffinBlobMetadata {
    pub r#type: String,
    pub fields: Vec<i32>,
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: i64,
    #[serde(rename = "sequence-number")]
    pub sequence_number: i64,
    pub offset: i64,
    pub length: i64,
    #[serde(rename = "compression-codec", skip_serializing_if = "Option::is_none")]
    pub compression_codec: Option<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PuffinFooter {
    pub blobs: Vec<PuffinBlobMetadata>,
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub properties: HashMap<String, String>,
}

/// Writer for Puffin files.
pub struct PuffinWriter<W: Write + Seek> {
    writer: W,
    blobs: Vec<PuffinBlobMetadata>,
    current_offset: i64,
}

/// Standard Apache Iceberg Puffin blob types.
///
/// The HNSW graph payload uses a single, quantization-agnostic blob type. The
/// quantizer (f32 / sq8 / tq8 / tq4 / pq) is carried by the manifest `algorithm`
/// field and the centroids `quantizer_config` — not by the Puffin blob type.
pub const PUFFIN_BLOB_VECTOR_HNSW_GRAPH: &str = "org.apache.iceberg.vector.hnsw-graph";
pub const PUFFIN_BLOB_LEXICAL_BM25: &str = "org.apache.iceberg.lexical.bm25";
pub const PUFFIN_BLOB_LEXICAL_DOCLEN: &str = "org.apache.iceberg.lexical.doclen";
pub const PUFFIN_BLOB_INDEX_ROARING: &str = "org.apache.iceberg.index.roaring-bitmap";
pub const PUFFIN_BLOB_JSON_PATH: &str = "org.apache.iceberg.json.path";
pub const PUFFIN_BLOB_GRAPH_CSR: &str = "org.apache.iceberg.graph.csr";
pub const PUFFIN_BLOB_GRAPH_CSR_OFFSETS: &str = "org.apache.iceberg.graph.csr.offsets";
pub const PUFFIN_BLOB_GRAPH_CSR_EDGES: &str = "org.apache.iceberg.graph.csr.edges";
pub const PUFFIN_BLOB_GRAPH_CSR_DICT: &str = "org.apache.iceberg.graph.csr.dict";
pub const PUFFIN_BLOB_HNSW_IVF_CENTROIDS: &str = "org.apache.iceberg.vector.centroids";
pub const PUFFIN_BLOB_HNSW_CLUSTER_GRAPH: &str = "org.apache.iceberg.vector.cluster-graph";
pub const PUFFIN_BLOB_HNSW_CLUSTER_MAPPING: &str = "org.apache.iceberg.vector.cluster-mapping";

// ---------------------------------------------------------------------------
// Table-level (cluster-keyed) index blobs.
//
// These live in a Puffin container that is *not* owned by any single data file.
// The `cluster-hnsw-*` types name the per-cluster graph payload; the companion
// `cluster-node-meta` blob carries the `(segment_idx, row_id)` pair for each
// graph node, which is what lets the existing row fetcher resolve a hit even
// though a cluster's members are spread across many Parquet files.
// ---------------------------------------------------------------------------
pub const PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META: &str =
    "org.apache.iceberg.vector.cluster-node-meta";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS: &str = "org.apache.iceberg.vector.cluster-segments";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS: &str = "org.apache.iceberg.vector.cluster-shards";

use crate::core::manifest::IndexFile;
use sha2::{Digest, Sha256};
use uuid::Uuid;

impl<W: Write + Seek> PuffinWriter<W> {
    pub fn new(mut writer: W) -> Result<Self> {
        writer.write_all(&PUFFIN_MAGIC)?;
        Ok(Self {
            writer,
            blobs: Vec::new(),
            current_offset: 4,
        })
    }

    /// Current byte offset written to the Puffin file.
    pub fn current_offset(&self) -> i64 {
        self.current_offset
    }

    pub fn add_blob(
        &mut self,
        type_name: String,
        fields: Vec<i32>,
        snapshot_id: i64,
        sequence_number: i64,
        data: &[u8],
        properties: HashMap<String, String>,
    ) -> Result<(i64, i64)> {
        let length = data.len() as i64;
        let offset = self.current_offset;

        self.writer.write_all(data)?;

        self.blobs.push(PuffinBlobMetadata {
            r#type: type_name,
            fields,
            snapshot_id,
            sequence_number,
            offset,
            length,
            compression_codec: None,
            properties,
        });

        self.current_offset += length;
        Ok((offset, length))
    }

    /// Add a blob by streaming from a reader, avoiding buffering large payloads in RAM.
    /// Computes the SHA-256 checksum on the fly and returns `(offset, length, sha256_checksum)`.
    pub fn add_blob_from_reader<R: Read>(
        &mut self,
        type_name: String,
        fields: Vec<i32>,
        snapshot_id: i64,
        sequence_number: i64,
        mut reader: R,
        properties: HashMap<String, String>,
    ) -> Result<(i64, i64, String)> {
        let offset = self.current_offset;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 65536];
        let mut total_len = 0i64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            self.writer.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            total_len += n as i64;
        }
        let checksum = format!("{:x}", hasher.finalize());

        self.blobs.push(PuffinBlobMetadata {
            r#type: type_name,
            fields,
            snapshot_id,
            sequence_number,
            offset,
            length: total_len,
            compression_codec: None,
            properties,
        });

        self.current_offset += total_len;
        Ok((offset, total_len, checksum))
    }

    pub fn finish(mut self) -> Result<W> {
        let footer = PuffinFooter {
            blobs: self.blobs,
            properties: HashMap::new(),
        };

        let footer_json = serde_json::to_string(&footer)?;
        let footer_bytes = footer_json.as_bytes();
        let footer_size = footer_bytes.len() as u32;

        // Spec order: Payload, Size (4), Flags (4), Magic (4)
        self.writer.write_all(footer_bytes)?;
        self.writer.write_all(&footer_size.to_le_bytes())?;

        let flags: u32 = 0; // Bit 0: 0 = uncompressed
        self.writer.write_all(&flags.to_le_bytes())?;
        self.writer.write_all(&PUFFIN_MAGIC)?;

        self.writer.flush()?;
        Ok(self.writer)
    }
}

/// High-level Puffin compound index writer that aggregates multiple secondary index
/// artifacts (vector, lexical BM25, roaring bitmaps, CSR graphs) for a segment into
/// a single `.puffin` container file with strict lineage bindings.
pub struct PuffinIndexWriter<W: Write + Seek> {
    writer: PuffinWriter<W>,
    entries: Vec<IndexFile>,
    container_file_path: String,
    source_snapshot_id: i64,
    source_data_checksum: String,
    source_record_count: i64,
}

impl<W: Write + Seek> PuffinIndexWriter<W> {
    pub fn new(
        writer: W,
        container_file_path: String,
        source_snapshot_id: i64,
        source_data_checksum: String,
        source_record_count: i64,
    ) -> Result<Self> {
        let p_writer = PuffinWriter::new(writer)?;
        Ok(Self {
            writer: p_writer,
            entries: Vec::new(),
            container_file_path,
            source_snapshot_id,
            source_data_checksum,
            source_record_count,
        })
    }

    /// Add an index artifact payload as a blob in the Puffin container and generate
    /// its corresponding `IndexFile` metadata entry.
    pub fn add_index_blob(
        &mut self,
        category: &str,
        algorithm: &str,
        column_name: &str,
        blob_type: &str,
        data: &[u8],
        fields: Vec<i32>,
        properties: HashMap<String, String>,
    ) -> Result<&IndexFile> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let index_checksum = format!("{:x}", hasher.finalize());

        let (offset, length) = self.writer.add_blob(
            blob_type.to_string(),
            fields,
            self.source_snapshot_id,
            1, // sequence number
            data,
            properties,
        )?;

        let entry = IndexFile {
            format_version: 2,
            index_id: Uuid::new_v4().to_string(),
            index_category: category.to_string(),
            algorithm: algorithm.to_string(),
            column_name: column_name.to_string(),
            source_snapshot_id: self.source_snapshot_id,
            source_data_checksum: self.source_data_checksum.clone(),
            source_record_count: self.source_record_count,
            index_checksum: Some(index_checksum),
            file_path: self.container_file_path.clone(),
            blob_type: Some(blob_type.to_string()),
            blob_offset: Some(offset),
            blob_length: Some(length),
            compression: "none".to_string(),
        };

        self.entries.push(entry);
        Ok(self.entries.last().ok_or_else(|| {
            crate::core::error::BenoStreamError::internal("entry was just pushed")
        })?)
    }

    /// Add an index artifact from an on-disk file, streaming its payload directly
    /// into the Puffin container in chunks without buffering the whole file in RAM.
    pub fn add_index_blob_from_file(
        &mut self,
        category: &str,
        algorithm: &str,
        column_name: &str,
        blob_type: &str,
        file_path: &str,
        fields: Vec<i32>,
        properties: HashMap<String, String>,
    ) -> Result<&IndexFile> {
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::with_capacity(65536, file);
        let (offset, length, index_checksum) = self.writer.add_blob_from_reader(
            blob_type.to_string(),
            fields,
            self.source_snapshot_id,
            1, // sequence number
            reader,
            properties,
        )?;

        let entry = IndexFile {
            format_version: 2,
            index_id: Uuid::new_v4().to_string(),
            index_category: category.to_string(),
            algorithm: algorithm.to_string(),
            column_name: column_name.to_string(),
            source_snapshot_id: self.source_snapshot_id,
            source_data_checksum: self.source_data_checksum.clone(),
            source_record_count: self.source_record_count,
            index_checksum: Some(index_checksum),
            file_path: self.container_file_path.clone(),
            blob_type: Some(blob_type.to_string()),
            blob_offset: Some(offset),
            blob_length: Some(length),
            compression: "none".to_string(),
        };

        self.entries.push(entry);
        Ok(self.entries.last().ok_or_else(|| {
            crate::core::error::BenoStreamError::internal("entry was just pushed")
        })?)
    }

    /// Stream an existing uncompressed blob range from `store` into this Puffin
    /// container in bounded chunks without buffering large blobs in RAM.
    #[allow(clippy::too_many_arguments)]
    pub async fn add_index_blob_from_store_range(
        &mut self,
        store: &dyn ObjectStore,
        src_path: &object_store::path::Path,
        src_offset: u64,
        src_length: u64,
        category: &str,
        algorithm: &str,
        column_name: &str,
        blob_type: &str,
        fields: Vec<i32>,
        properties: HashMap<String, String>,
    ) -> Result<&IndexFile> {
        const CHUNK_SIZE: u64 = 8 * 1024 * 1024;
        let dest_offset = self.writer.current_offset();
        let mut hasher = Sha256::new();
        let mut remaining = src_length;
        let mut cursor = src_offset;

        while remaining > 0 {
            let to_read = remaining.min(CHUNK_SIZE);
            let chunk = store
                .get_range(src_path, cursor..(cursor + to_read))
                .await?;
            if chunk.is_empty() {
                anyhow::bail!(
                    "Unexpected EOF reading preserved Puffin blob {} range {}..{}",
                    src_path,
                    src_offset,
                    src_offset + src_length
                );
            }
            self.writer.writer.write_all(&chunk)?;
            hasher.update(&chunk);
            let read_len = chunk.len() as u64;
            cursor += read_len;
            remaining = remaining.saturating_sub(read_len);
        }

        let index_checksum = format!("{:x}", hasher.finalize());
        let length_i64 = i64::try_from(src_length).context("Puffin blob length exceeds i64")?;
        self.writer.blobs.push(PuffinBlobMetadata {
            r#type: blob_type.to_string(),
            fields,
            snapshot_id: self.source_snapshot_id,
            sequence_number: 1,
            offset: dest_offset,
            length: length_i64,
            compression_codec: None,
            properties,
        });
        self.writer.current_offset += length_i64;

        let entry = IndexFile {
            format_version: 2,
            index_id: Uuid::new_v4().to_string(),
            index_category: category.to_string(),
            algorithm: algorithm.to_string(),
            column_name: column_name.to_string(),
            source_snapshot_id: self.source_snapshot_id,
            source_data_checksum: self.source_data_checksum.clone(),
            source_record_count: self.source_record_count,
            index_checksum: Some(index_checksum),
            file_path: self.container_file_path.clone(),
            blob_type: Some(blob_type.to_string()),
            blob_offset: Some(dest_offset),
            blob_length: Some(length_i64),
            compression: "none".to_string(),
        };

        self.entries.push(entry);
        Ok(self.entries.last().ok_or_else(|| {
            crate::core::error::BenoStreamError::internal("entry was just pushed")
        })?)
    }

    pub fn entries(&self) -> &[IndexFile] {
        &self.entries
    }

    /// Finish writing the Puffin container and return the underlying writer
    /// along with the generated `IndexFile` entries.
    pub fn finish(self) -> Result<(W, Vec<IndexFile>)> {
        let writer = self.writer.finish()?;
        Ok((writer, self.entries))
    }
}

impl PuffinIndexWriter<std::io::Cursor<Vec<u8>>> {
    /// Create a new in-memory PuffinIndexWriter.
    pub fn new_in_memory(
        container_file_path: String,
        source_snapshot_id: i64,
        source_data_checksum: String,
        source_record_count: i64,
    ) -> Result<Self> {
        Self::new(
            std::io::Cursor::new(Vec::new()),
            container_file_path,
            source_snapshot_id,
            source_data_checksum,
            source_record_count,
        )
    }

    /// Finish writing the in-memory Puffin container and return the serialized bytes
    /// along with the generated `IndexFile` entries.
    pub fn finish_to_bytes(self) -> Result<(Vec<u8>, Vec<IndexFile>)> {
        let (cursor, entries) = self.finish()?;
        Ok((cursor.into_inner(), entries))
    }
}

/// Convenience reader for index blobs packed inside a Puffin container.
pub struct PuffinIndexReader<R: Read + Seek> {
    reader: PuffinReader<R>,
}

impl<R: Read + Seek> PuffinIndexReader<R> {
    pub fn new(reader: R) -> Result<Self> {
        let p_reader = PuffinReader::new(reader)?;
        Ok(Self { reader: p_reader })
    }

    pub fn blobs(&self) -> &[PuffinBlobMetadata] {
        &self.reader.footer().blobs
    }

    pub fn read_blob_by_index(&mut self, blob_idx: usize) -> Result<Vec<u8>> {
        self.reader.read_blob(blob_idx)
    }

    pub fn read_blob_by_type(&mut self, blob_type: &str) -> Result<Option<Vec<u8>>> {
        let idx = self
            .reader
            .footer()
            .blobs
            .iter()
            .position(|b| b.r#type == blob_type);
        match idx {
            Some(i) => self.reader.read_blob(i).map(Some),
            None => Ok(None),
        }
    }
}

/// Reader for Puffin files.
pub struct PuffinReader<R: Read + Seek> {
    reader: R,
    header: PuffinFooter,
}

impl<R: Read + Seek> PuffinReader<R> {
    pub fn new(mut reader: R) -> Result<Self> {
        // Read footer size and magic from end
        reader.seek(SeekFrom::End(-12))?; // 12 bytes = Size(4) + Flags(4) + Magic(4)

        let mut size_buf = [0u8; 4];
        reader.read_exact(&mut size_buf)?;
        let footer_size = u32::from_le_bytes(size_buf);

        let mut flags_buf = [0u8; 4];
        reader.read_exact(&mut flags_buf)?;
        let flags = u32::from_le_bytes(flags_buf);

        let mut magic_buf = [0u8; 4];
        reader.read_exact(&mut magic_buf)?;
        if magic_buf != PUFFIN_MAGIC {
            anyhow::bail!("Invalid Puffin magic at end of file");
        }

        // Seek to start of footer payload
        reader.seek(SeekFrom::End(-12 - footer_size as i64))?;
        let mut footer_payload = vec![0u8; footer_size as usize];
        reader.read_exact(&mut footer_payload)?;

        // Handle footer-level compression flag (bit 0)
        // When set, the footer payload is zstd-compressed
        let footer_payload = if flags & 0x01 != 0 {
            zstd::decode_all(&footer_payload[..])
                .context("Failed to decompress Puffin footer with Zstd")?
        } else {
            footer_payload
        };

        let footer: PuffinFooter = serde_json::from_slice(&footer_payload)
            .context("Failed to parse Puffin footer JSON")?;

        Ok(Self {
            reader,
            header: footer,
        })
    }

    pub fn footer(&self) -> &PuffinFooter {
        &self.header
    }

    pub fn read_blob(&mut self, blob_idx: usize) -> Result<Vec<u8>> {
        let meta = self
            .header
            .blobs
            .get(blob_idx)
            .context("Blob index out of range")?;

        self.reader.seek(SeekFrom::Start(meta.offset as u64))?;
        let mut data = vec![0u8; meta.length as usize];
        self.reader.read_exact(&mut data)?;

        // Check per-blob compression codec
        if let Some(ref codec) = meta.compression_codec {
            if codec == "zstd" {
                data = zstd::decode_all(&data[..])
                    .context(format!("Failed to decompress blob {} with Zstd", blob_idx))?;
            } else {
                anyhow::bail!(
                    "Blob {} uses compression codec '{}' which is not yet supported",
                    blob_idx,
                    codec
                );
            }
        }

        Ok(data)
    }
}

/// Zero-copy, read-only object store over the blobs of one Puffin container.
///
/// A container is the natural unit for an index split across hundreds of blobs
/// (centroids plus a graph and a mapping per IVF cluster). Loading it used to
/// mean materializing every blob into `$TMPDIR` and parsing it back — a full
/// write and re-read of the bundle on every cold load (4.3 GB per load at 10M
/// vectors). Here the container is read once and each blob is exposed as a
/// `Bytes` slice of it, so the loaders copy nothing they do not keep.
pub struct PuffinContainerStore {
    inner: object_store::memory::InMemory,
    label: String,
}

impl PuffinContainerStore {
    /// Slice `container` into an in-memory store, returning it together with the
    /// loader base path for `column_name`'s vector index.
    pub async fn from_container(
        container: bytes::Bytes,
        container_path: &str,
        column_name: &str,
    ) -> Result<(Self, String)> {
        // (filename, offset, length), in container order.
        let entries: Vec<(String, i64, i64)> = {
            let reader = PuffinReader::new(std::io::Cursor::new(container.as_ref()))?;
            reader
                .footer()
                .blobs
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let name = b
                        .properties
                        .get("filename")
                        .cloned()
                        .unwrap_or_else(|| format!("blob_{i}"));
                    (name, b.offset, b.length)
                })
                .collect()
        };

        // A bundle holds one vector index per (column, algorithm). Identify this
        // one by its centroids blob rather than by blob order, which is not stable.
        let marker = format!(".{column_name}.");
        let base_path = entries
            .iter()
            .find_map(|(name, _, _)| {
                if name.contains(&marker) {
                    name.strip_suffix(".centroids.parquet").map(str::to_string)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Puffin bundle '{}' has no centroids blob for column '{}'",
                    container_path,
                    column_name
                )
            })?;

        let inner = object_store::memory::InMemory::new();
        for (name, offset, length) in &entries {
            let start = usize::try_from(*offset).map_err(|_| {
                anyhow::anyhow!("negative blob offset {offset} in '{container_path}'")
            })?;
            let len = usize::try_from(*length).map_err(|_| {
                anyhow::anyhow!("negative blob length {length} in '{container_path}'")
            })?;
            let end = start
                .checked_add(len)
                .filter(|e| *e <= container.len())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "blob '{}' [{start}..{start}+{len}] is outside the {}-byte container",
                        name,
                        container.len()
                    )
                })?;
            // `Bytes::slice` shares the container's allocation: no copy.
            inner
                .put(
                    &object_store::path::Path::from(name.as_str()),
                    container.slice(start..end).into(),
                )
                .await?;
        }

        Ok((
            Self {
                inner,
                label: container_path.to_string(),
            },
            base_path,
        ))
    }
}

impl std::fmt::Display for PuffinContainerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PuffinContainerStore({})", self.label)
    }
}

impl std::fmt::Debug for PuffinContainerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PuffinContainerStore({})", self.label)
    }
}

#[async_trait::async_trait]
impl ObjectStore for PuffinContainerStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        _payload: object_store::PutPayload,
        _opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        Err(object_store::Error::NotSupported {
            source: format!("PuffinContainerStore is read-only ({location})").into(),
        })
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        _opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        Err(object_store::Error::NotSupported {
            source: format!("PuffinContainerStore is read-only ({location})").into(),
        })
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn copy(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
    ) -> object_store::Result<()> {
        Err(object_store::Error::NotSupported {
            source: format!("PuffinContainerStore is read-only ({from} -> {to})").into(),
        })
    }

    async fn copy_if_not_exists(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
    ) -> object_store::Result<()> {
        Err(object_store::Error::NotSupported {
            source: format!("PuffinContainerStore is read-only ({from} -> {to})").into(),
        })
    }

    async fn delete(&self, location: &object_store::path::Path) -> object_store::Result<()> {
        Err(object_store::Error::NotSupported {
            source: format!("PuffinContainerStore is read-only ({location})").into(),
        })
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
}

/// Materialize every blob of a Puffin container into a temporary directory,
/// using each blob's `filename` property (falling back to `blob_{i}`).
///
/// This bridges the Puffin compound-bundle storage format to the existing
/// multi-file index loaders (HNSW-IVF, CSR graph), which expect a directory of
/// named files. The caller must keep the returned `TempDir` alive for the
/// duration of the load; once the index has copied its bytes into memory the
/// directory can be dropped.
///
/// Returns the temp directory and the list of `(blob_type, filename)` pairs
/// that were written.
pub async fn materialize_puffin_blobs(
    store: &Arc<dyn ObjectStore>,
    puffin_path: &str,
) -> Result<(tempfile::TempDir, Vec<(String, String)>)> {
    let bytes = store
        .get(&object_store::path::Path::from(puffin_path))
        .await
        .with_context(|| format!("Failed to read Puffin bundle '{}'", puffin_path))?
        .bytes()
        .await?;
    let mut reader = PuffinReader::new(std::io::Cursor::new(bytes.to_vec()))?;
    let blobs = reader.footer().blobs.clone();
    let temp_dir = tempfile::tempdir()?;
    let mut written = Vec::with_capacity(blobs.len());
    for (i, blob) in blobs.iter().enumerate() {
        let filename = blob
            .properties
            .get("filename")
            .cloned()
            .unwrap_or_else(|| format!("blob_{}", i));
        let data = reader.read_blob(i)?;
        std::fs::write(temp_dir.path().join(&filename), data)?;
        written.push((blob.r#type.clone(), filename));
    }
    Ok((temp_dir, written))
}

/// Puffin blob type identifier for Apache Iceberg V3 deletion vectors.
///
/// This is the spec identifier the Iceberg project uses to locate a deletion
/// vector inside a Puffin container. The previous value
/// (`"apache-datasketches-theta-v1"`) was the identifier for a *different*
/// blob type entirely, so no external reader would ever have found our DVs.
pub const DELETION_VECTOR_TYPE: &str = "deletion-vector-v1";

/// Magic sequence that prefixes every `deletion-vector-v1` blob payload.
///
/// Big-endian `0xD1D33964` (equivalently little-endian `1681511377` /
/// `0x6439D3D1`) — the value the Iceberg spec mandates. A reader that does not
/// see these four bytes must reject the blob rather than guess.
pub const DELETION_VECTOR_MAGIC: [u8; 4] = [0xD1, 0xD3, 0x39, 0x64];

/// Bytes of framing a `deletion-vector-v1` payload carries around the bitmap:
/// `4B` big-endian length + `4B` magic + `4B` big-endian CRC-32.
const DELETION_VECTOR_OVERHEAD: usize = 12;

/// Bytes of framing that sit *outside* the region the length field counts:
/// the `4B` length field itself plus the trailing `4B` CRC-32.
const DELETION_VECTOR_FRAME_BYTES: usize = 8;

/// CRC-32 (IEEE 802.3, reflected, polynomial `0xEDB88320`).
///
/// This is the same checksum Iceberg's reference implementation applies
/// (`java.util.zip.CRC32`). Implemented inline rather than pulled in as a
/// dependency: it is ~15 lines, and `deny.toml` audits every crate in the graph.
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Encode a deletion vector into the Iceberg V3 `deletion-vector-v1` payload.
///
/// Layout (all integers big-endian except the portable roaring bitmap, which is
/// little-endian by definition of the portable serialization format):
///
/// ```text
/// +----------------+--------+----------------------+-----------+
/// | length (4B BE) | magic  | portable 64-bit      | CRC-32    |
/// | = 4 + N        | 4B     | roaring bitmap (N)   | 4B BE     |
/// +----------------+--------+----------------------+-----------+
/// ```
///
/// The length covers magic + bitmap only — it excludes both the length field
/// and the trailing checksum. The CRC covers exactly the same bytes as the
/// length (magic + bitmap).
///
/// Iceberg V3 stores row positions as 64-bit values. BenoStreamDB row positions
/// are `u32` today (`RoaringBitmap`), so anything above `u32::MAX` is an error
/// rather than a silent truncation.
pub fn encode_deletion_vector_v1(bitmap: &roaring::RoaringBitmap) -> Result<Vec<u8>> {
    let mut treemap = roaring::RoaringTreemap::new();
    for position in bitmap.iter() {
        treemap.insert(position as u64);
    }

    let mut portable = Vec::with_capacity(DELETION_VECTOR_OVERHEAD + 16);
    treemap
        .serialize_into(&mut portable)
        .context("Failed to serialize portable 64-bit RoaringBitmap for deletion vector")?;

    let framed_len = DELETION_VECTOR_MAGIC.len() + portable.len();
    if framed_len > u32::MAX as usize {
        anyhow::bail!(
            "Deletion vector payload of {} bytes exceeds the 32-bit length field",
            framed_len
        );
    }

    let mut payload = Vec::with_capacity(DELETION_VECTOR_OVERHEAD + portable.len());
    payload.extend_from_slice(&(framed_len as u32).to_be_bytes());
    payload.extend_from_slice(&DELETION_VECTOR_MAGIC);
    payload.extend_from_slice(&portable);

    let checksum = crc32_ieee(&payload[4..4 + framed_len]);
    payload.extend_from_slice(&checksum.to_be_bytes());
    Ok(payload)
}

/// Decode a `deletion-vector-v1` payload into the set of deleted row positions.
///
/// Validates the length field, the magic sequence, and the CRC-32 before any
/// bitmap bytes are trusted. Every rejection is
/// `BenoStreamError::DeletionVectorDeserialize` so a corrupt DV surfaces as a
/// typed error instead of a silently short delete list (which would resurrect
/// deleted rows).
pub fn decode_deletion_vector_v1(data: &[u8]) -> Result<roaring::RoaringBitmap> {
    use crate::core::error::BenoStreamError;

    let reject = |reason: String| -> anyhow::Error {
        BenoStreamError::DeletionVectorDeserialize { reason }.into()
    };

    if data.len() < DELETION_VECTOR_OVERHEAD {
        return Err(reject(format!(
            "payload of {} bytes is shorter than the {}-byte deletion-vector framing",
            data.len(),
            DELETION_VECTOR_OVERHEAD
        )));
    }

    let framed_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if framed_len + DELETION_VECTOR_FRAME_BYTES != data.len() {
        return Err(reject(format!(
            "length field declares {} bytes but the payload is {} bytes (length + CRC framing adds {})",
            framed_len,
            data.len(),
            DELETION_VECTOR_FRAME_BYTES
        )));
    }
    if data[4..8] != DELETION_VECTOR_MAGIC {
        return Err(reject(format!(
            "magic {:02X?} does not match the deletion-vector-v1 sequence {:02X?}",
            &data[4..8],
            DELETION_VECTOR_MAGIC
        )));
    }

    let bitmap_bytes = &data[8..4 + framed_len];
    let expected = u32::from_be_bytes([
        data[4 + framed_len],
        data[5 + framed_len],
        data[6 + framed_len],
        data[7 + framed_len],
    ]);
    let actual = crc32_ieee(&data[4..4 + framed_len]);
    if expected != actual {
        return Err(reject(format!(
            "CRC-32 mismatch: payload declares {expected:#010X} but computes {actual:#010X}"
        )));
    }

    let treemap = roaring::RoaringTreemap::deserialize_from(bitmap_bytes)
        .map_err(|e| reject(format!("failed to deserialize portable RoaringBitmap: {e}")))?;

    let mut bitmap = roaring::RoaringBitmap::new();
    for position in treemap.iter() {
        let position_u32 = u32::try_from(position).map_err(|_| {
            reject(format!(
                "deletion vector position {position} exceeds u32::MAX, which this reader cannot represent"
            ))
        })?;
        bitmap.insert(position_u32);
    }
    Ok(bitmap)
}

/// Read a deletion vector blob from a Puffin file and return the deleted row positions.
pub fn read_deletion_vector_from_puffin<R: Read + Seek>(
    reader: &mut PuffinReader<R>,
    blob_idx: usize,
) -> Result<roaring::RoaringBitmap> {
    let data = reader.read_blob(blob_idx)?;
    decode_deletion_vector_v1(&data)
}

/// Decode a deletion vector from a byte range already fetched from an object store.
///
/// This is the read path `HybridReader` uses: a DV manifest entry carries
/// `content_offset` / `content_size_in_bytes`, so only the blob is fetched, not
/// the whole Puffin file.
pub fn read_deletion_vector_from_bytes(data: &[u8]) -> Result<roaring::RoaringBitmap> {
    decode_deletion_vector_v1(data)
}

#[cfg(test)]
mod deletion_vector_tests {
    use super::*;

    fn bitmap_of(positions: &[u32]) -> roaring::RoaringBitmap {
        let mut bm = roaring::RoaringBitmap::new();
        for &p in positions {
            bm.insert(p);
        }
        bm
    }

    #[test]
    fn deletion_vector_v1_round_trips() {
        let original = bitmap_of(&[0, 1, 2, 7, 63, 64, 65, 4096, 1_000_000]);
        let payload = encode_deletion_vector_v1(&original).unwrap();
        let decoded = decode_deletion_vector_v1(&payload).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn deletion_vector_v1_round_trips_empty_and_max_position() {
        for positions in [vec![], vec![0u32], vec![u32::MAX]] {
            let original = bitmap_of(&positions);
            let payload = encode_deletion_vector_v1(&original).unwrap();
            assert_eq!(decode_deletion_vector_v1(&payload).unwrap(), original);
        }
    }

    #[test]
    fn deletion_vector_v1_framing_is_spec_shaped() {
        let payload = encode_deletion_vector_v1(&bitmap_of(&[5, 6])).unwrap();
        let framed_len =
            u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        // length covers magic + bitmap, excluding the length field and the CRC
        assert_eq!(framed_len, payload.len() - DELETION_VECTOR_FRAME_BYTES);
        assert_eq!(framed_len, payload.len() - 8);
        assert_eq!(&payload[4..8], &DELETION_VECTOR_MAGIC);
        assert_eq!(
            u32::from_be_bytes([
                payload[4 + framed_len],
                payload[5 + framed_len],
                payload[6 + framed_len],
                payload[7 + framed_len],
            ]),
            crc32_ieee(&payload[4..4 + framed_len])
        );
    }

    #[test]
    fn deletion_vector_v1_rejects_bad_magic() {
        let mut payload = encode_deletion_vector_v1(&bitmap_of(&[3])).unwrap();
        payload[4] ^= 0xFF;
        let err = decode_deletion_vector_v1(&payload).unwrap_err();
        assert!(
            err.to_string().contains("magic"),
            "expected a magic error, got: {err}"
        );
    }

    #[test]
    fn deletion_vector_v1_rejects_corrupted_bitmap() {
        let mut payload = encode_deletion_vector_v1(&bitmap_of(&[3, 9, 27])).unwrap();
        let framed_len = payload.len() - DELETION_VECTOR_OVERHEAD;
        // flip a bitmap byte (starts right after the magic) without touching framing
        payload[8] ^= 0xFF;
        // recompute nothing: the CRC must catch it
        let err = decode_deletion_vector_v1(&payload).unwrap_err();
        assert!(
            err.to_string().contains("CRC-32"),
            "expected a CRC error, got: {err}"
        );
        // sanity: restoring the byte makes it decode again
        payload[8] ^= 0xFF;
        assert!(decode_deletion_vector_v1(&payload).is_ok());
        let _ = framed_len;
    }

    #[test]
    fn deletion_vector_v1_rejects_truncated_and_overlong_payloads() {
        let payload = encode_deletion_vector_v1(&bitmap_of(&[11, 12, 13])).unwrap();

        let truncated = &payload[..payload.len() - 1];
        assert!(decode_deletion_vector_v1(truncated).is_err());

        let mut overlong = payload.clone();
        overlong.push(0x00);
        assert!(decode_deletion_vector_v1(&overlong).is_err());

        assert!(decode_deletion_vector_v1(&[]).is_err());
        assert!(decode_deletion_vector_v1(&payload[..8]).is_err());
    }

    #[test]
    fn deletion_vector_type_is_the_iceberg_spec_identifier() {
        assert_eq!(DELETION_VECTOR_TYPE, "deletion-vector-v1");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_puffin_writer_and_reader() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let puffin_path = temp_dir.path().join("test.puffin");
        let payload1_path = temp_dir.path().join("payload1.bin");
        let payload2_path = temp_dir.path().join("payload2.bin");

        let data1 = vec![42u8; 100_000];
        let data2 = vec![99u8; 200_000];
        std::fs::write(&payload1_path, &data1)?;
        std::fs::write(&payload2_path, &data2)?;

        let puffin_file = std::fs::File::create(&puffin_path)?;
        let mut writer = PuffinIndexWriter::new(
            std::io::BufWriter::new(puffin_file),
            puffin_path.to_string_lossy().to_string(),
            1,
            "chk123".to_string(),
            300,
        )?;

        let mut props1 = HashMap::new();
        props1.insert("filename".to_string(), "payload1.bin".to_string());
        writer.add_index_blob_from_file(
            "vector",
            "tq8",
            "embedding",
            PUFFIN_BLOB_VECTOR_HNSW_GRAPH,
            payload1_path.to_str().unwrap(),
            vec![],
            props1,
        )?;

        let mut props2 = HashMap::new();
        props2.insert("filename".to_string(), "payload2.bin".to_string());
        writer.add_index_blob_from_file(
            "vector",
            "centroids",
            "embedding",
            PUFFIN_BLOB_HNSW_IVF_CENTROIDS,
            payload2_path.to_str().unwrap(),
            vec![],
            props2,
        )?;

        let (mut file_writer, entries) = writer.finish()?;
        use std::io::Write;
        file_writer.flush()?;
        drop(file_writer);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].blob_length, Some(100_000));
        assert_eq!(entries[1].blob_length, Some(200_000));

        // Read back with PuffinReader
        let read_file = std::fs::File::open(&puffin_path)?;
        let mut reader = PuffinReader::new(read_file)?;
        let blobs = reader.footer().blobs.clone();
        assert_eq!(blobs.len(), 2);

        let read_data1 = reader.read_blob(0)?;
        assert_eq!(read_data1, data1);
        let read_data2 = reader.read_blob(1)?;
        assert_eq!(read_data2, data2);

        Ok(())
    }
}
