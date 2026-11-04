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
pub const PUFFIN_BLOB_VECTOR_HNSW_TQ8: &str = "org.apache.iceberg.vector.hnsw-tq8";
pub const PUFFIN_BLOB_VECTOR_HNSW_TQ4: &str = "org.apache.iceberg.vector.hnsw-tq4";
pub const PUFFIN_BLOB_VECTOR_HNSW_PQ: &str = "org.apache.iceberg.vector.hnsw-pq";
pub const PUFFIN_BLOB_VECTOR_HNSW: &str = "org.apache.iceberg.vector.hnsw";
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
pub const PUFFIN_BLOB_VECTOR_CLUSTER_HNSW_TQ8: &str =
    "org.apache.iceberg.vector.cluster-hnsw-tq8";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_HNSW_TQ4: &str =
    "org.apache.iceberg.vector.cluster-hnsw-tq4";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_HNSW_PQ: &str = "org.apache.iceberg.vector.cluster-hnsw-pq";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_HNSW: &str = "org.apache.iceberg.vector.cluster-hnsw";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META: &str =
    "org.apache.iceberg.vector.cluster-node-meta";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS: &str =
    "org.apache.iceberg.vector.cluster-segments";
pub const PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS: &str =
    "org.apache.iceberg.vector.cluster-shards";

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

/// Deletion Vector blob type identifier
pub const DELETION_VECTOR_TYPE: &str = "apache-datasketches-theta-v1";

/// Read a deletion vector from a Puffin file and return the set of deleted row positions
///
/// Deletion vectors in Iceberg v3 use RoaringBitmap format to efficiently store deleted positions
pub fn read_deletion_vector_from_puffin<R: Read + Seek>(
    reader: &mut PuffinReader<R>,
    blob_idx: usize,
) -> Result<roaring::RoaringBitmap> {
    let data = reader.read_blob(blob_idx)?;

    // Deletion vectors are stored as RoaringBitmap serialized format
    let bitmap = roaring::RoaringBitmap::deserialize_from(&data[..])
        .context("Failed to deserialize deletion vector RoaringBitmap")?;

    Ok(bitmap)
}

/// Read a deletion vector from a byte range (for async/object store scenarios)
pub fn read_deletion_vector_from_bytes(data: &[u8]) -> Result<roaring::RoaringBitmap> {
    let bitmap = roaring::RoaringBitmap::deserialize_from(data)
        .context("Failed to deserialize deletion vector RoaringBitmap")?;

    Ok(bitmap)
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
            PUFFIN_BLOB_VECTOR_HNSW_TQ8,
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
