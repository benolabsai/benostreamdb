// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Iceberg integration module for HyperStreamDB.
//!
//! Provides reading, writing, and conversion of Iceberg table metadata,
//! manifests, schemas, delete files, and partition transforms.

pub mod types;
pub mod manifest;
pub mod schema;
pub mod delete;
pub mod writer;
pub mod transform;
pub mod value;

// Re-export `iceberg_delete` for backward compatibility with existing callers
// that reference `crate::core::iceberg::iceberg_delete::IcebergDeleteWriter`.
// The actual implementation lives in `delete` module.
pub mod iceberg_delete {
    pub use super::delete::IcebergDeleteWriter;
}

// ── Types ──────────────────────────────────────────────────────────────────
pub use types::{
    IcebergTableMetadata,
    IcebergSnapshot,
    IcebergManifestListEntry,
    IcebergManifestEntry,
    IcebergDataFile,
    IcebergManifestObject,
    DeleteContent,
};

// ── Manifest reading & conversion ─────────────────────────────────────────
pub use manifest::{
    read_manifest_list,
    read_manifest,
    convert_iceberg_to_object,
};

// ── Schema ─────────────────────────────────────────────────────────────────
pub use schema::{
    iceberg_json_to_arrow_schema,
    iceberg_partition_spec_to_hyperstream,
};

// ── Delete readers ────────────────────────────────────────────────────────
pub use delete::{
    PositionDeleteReader,
    EqualityDeleteReader,
};

// ── Writers ────────────────────────────────────────────────────────────────
pub use writer::{
    IcebergWriter,
    GpuPuffinWriter,
    MANIFEST_LIST_SCHEMA_V2,
};

// ── Transforms ─────────────────────────────────────────────────────────────
pub use transform::{
    IcebergTransform,
    murmur3_32_x86,
};

// ── Value conversion ──────────────────────────────────────────────────────
pub use value::{
    avro_to_json,
    decode_iceberg_value,
    parse_avro_value_bytes,
    parse_avro_value_bytes_with_type,
    json_to_avro_value,
};
