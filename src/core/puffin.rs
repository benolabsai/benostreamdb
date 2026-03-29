// Copyright (c) 2026 Richard Albright. All rights reserved.

use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::io::{Read, Write, Seek, SeekFrom};
use std::collections::HashMap;

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

impl<W: Write + Seek> PuffinWriter<W> {
    pub fn new(mut writer: W) -> Result<Self> {
        writer.write_all(&PUFFIN_MAGIC)?;
        Ok(Self {
            writer,
            blobs: Vec::new(),
            current_offset: 4,
        })
    }

    pub fn add_blob(
        &mut self,
        type_name: String,
        fields: Vec<i32>,
        snapshot_id: i64,
        sequence_number: i64,
        data: &[u8],
        properties: HashMap<String, String>,
    ) -> Result<()> {
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
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
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
        Ok(())
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
        // let flags = u32::from_le_bytes(flags_buf);
        
        let mut magic_buf = [0u8; 4];
        reader.read_exact(&mut magic_buf)?;
        if magic_buf != PUFFIN_MAGIC {
            anyhow::bail!("Invalid Puffin magic at end of file");
        }
        
        // Seek to start of footer payload
        reader.seek(SeekFrom::End(-12 - footer_size as i64))?;
        let mut footer_payload = vec![0u8; footer_size as usize];
        reader.read_exact(&mut footer_payload)?;
        
        // TODO: Handle compression flag if bit 0 is set
        
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
        let meta = self.header.blobs.get(blob_idx)
            .context("Blob index out of range")?;
            
        self.reader.seek(SeekFrom::Start(meta.offset as u64))?;
        let mut data = vec![0u8; meta.length as usize];
        self.reader.read_exact(&mut data)?;
        
        Ok(data)
    }
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
