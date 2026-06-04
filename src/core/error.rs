// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Structured error types for HyperStreamDB.
//!
//! All public APIs return `Result<T, HyperstreamError>`. Internal implementation
//! details may still use `anyhow::Error` for convenience, converting at module
//! boundaries via `?` or `.map_err(HyperstreamError::from_anyhow)`.

use std::fmt;

#[derive(Debug)]
pub enum HyperstreamError {
    /// I/O or object-store failure (S3, Azure, GCS, local fs).
    Io { source: std::io::Error },

    /// Object store returned an error (not found, permission denied, etc.).
    ObjectStore { source: object_store::Error },

    /// Arrow error (schema cast, array downcast, FFI, etc.).
    Arrow { source: arrow::error::ArrowError },

    /// DataFusion planning or execution error.
    DataFusion {
        source: datafusion::error::DataFusionError,
    },

    /// Invalid URI or path for storage backend.
    InvalidUri { uri: String, reason: String },

    // ─── Schema errors ───────────────────────────────────────────────
    /// Requested column does not exist in the table schema.
    ColumnNotFound {
        column: String,
        table: Option<String>,
    },

    /// Duplicate column detected during schema merge.
    DuplicateColumn { column: String },

    /// Schema promotion failed (type incompatibility, unexpected deletion, etc.).
    SchemaIncompatible { reason: String },

    /// Table or schema not found in catalog.
    TableNotFound { namespace: String, name: String },

    /// NULL value in a NOT NULL / primary key column.
    NullConstraintViolation { column: String },

    // ─── Index errors ────────────────────────────────────────────────
    /// Vector index file not found at expected path.
    IndexNotFound { path: String },

    /// Index build failed (empty vectors, unsupported type, etc.).
    IndexBuildFailed { index: String, reason: String },

    /// GPU device not available or not enabled.
    GpuNotAvailable { requested: String },

    /// GPU compilation / pipeline creation failure.
    GpuCompileFailed { reason: String },

    /// Unsupported vector dimension or metric for the requested index.
    UnsupportedVectorConfig { detail: String },

    // ─── Catalog errors ──────────────────────────────────────────────
    /// Catalog misconfiguration (missing URL, token, etc.).
    CatalogConfig { catalog: String, reason: String },

    /// Catalog request failed (network, auth, server error).
    CatalogRequest {
        catalog: String,
        status: Option<u16>,
        reason: String,
    },

    /// Branching not supported by this catalog type.
    BranchNotSupported { catalog: String },

    /// Unknown catalog type string.
    UnknownCatalogType { typ: String },

    // ─── Manifest errors ────────────────────────────────────────────
    /// Manifest version mismatch (file version != expected).
    ManifestVersionMismatch { expected: u64, actual: u64 },

    /// Manifest commit failed after all retries.
    ManifestCommitFailed { reason: String },

    /// Manifest rollback failed.
    ManifestRollbackFailed { reason: String },

    // ─── Lock errors ────────────────────────────────────────────────
    /// Failed to acquire distributed lock after max retries.
    LockAcquireFailed { resource: String, retries: u32 },

    // ─── Query / planner errors ─────────────────────────────────────
    /// Failed to parse or evaluate filter expression.
    FilterError { reason: String },

    /// Physical expression creation failed.
    PhysicalExprError { reason: String },

    // ─── Reader / scan errors ───────────────────────────────────────
    /// Segment not found in manifest.
    SegmentNotFound { segment_id: String },

    /// Data integrity / checksum mismatch.
    DataIntegrityCheckFailed { segment_id: String, reason: String },

    /// Bitmap iterator exhausted (internal state corruption).
    BitmapExhausted,

    // ─── Write errors ───────────────────────────────────────────────
    /// Primary key uniqueness violation.
    PrimaryKeyViolation { key: String },

    /// Write operation failed.
    WriteFailed { reason: String },

    /// Merge/upsert operation failed.
    MergeFailed { reason: String },

    // ─── WAL errors ─────────────────────────────────────────────────
    /// WAL file could not be opened.
    WalOpenError {
        path: String,
        source: std::io::Error,
    },

    /// WAL channel closed unexpectedly.
    WalChannelClosed,

    /// WAL compaction failed.
    WalCompactionFailed { reason: String },

    // ─── Puffin errors ──────────────────────────────────────────────
    /// Puffin footer parse failure.
    PuffinParseError { reason: String },

    /// Puffin blob index out of range.
    PuffinBlobOutOfRange { requested: usize, max: usize },

    /// Deletion vector deserialize failure.
    DeletionVectorDeserialize { reason: String },

    // ─── Iceberg errors ─────────────────────────────────────────────
    /// Invalid Iceberg schema.
    IcebergSchemaError { reason: String },

    /// Iceberg equality-delete position mismatch.
    IcebergDeleteError { reason: String },

    // ─── License errors ─────────────────────────────────────────────
    /// Invalid license key format.
    InvalidLicense { reason: String },

    /// License has expired.
    LicenseExpired { expired_at: String },

    /// Enterprise feature requires valid license.
    EnterpriseFeatureRequired { feature: String },

    // ─── Embedding errors ───────────────────────────────────────────
    /// Embedding API returned invalid / unexpected response.
    EmbeddingApiError { reason: String },

    // ─── Concurrency / background task errors ───────────────────────
    /// Background task (indexer, compactor, etc.) failed.
    BackgroundTaskFailed { task: String, reason: String },

    /// Semaphore or concurrency limit reached.
    ConcurrencyLimit,

    // ─── Catch-all ──────────────────────────────────────────────────
    /// Wrapped anyhow error for internal implementation details.
    Internal { source: anyhow::Error },
}

// ─── Result type alias ────────────────────────────────────────────────────────

pub type Result<T> = std::result::Result<T, HyperstreamError>;

// ─── Display ───────────────────────────────────────────────────────────────────

impl fmt::Display for HyperstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { source } => write!(f, "I/O error: {source}"),
            Self::ObjectStore { source } => write!(f, "Object store error: {source}"),
            Self::Arrow { source } => write!(f, "Arrow error: {source}"),
            Self::DataFusion { source } => write!(f, "DataFusion error: {source}"),
            Self::InvalidUri { uri, reason } => write!(f, "Invalid URI '{uri}': {reason}"),

            Self::ColumnNotFound { column, table } => {
                if let Some(t) = table {
                    write!(f, "Column '{column}' not found in table '{t}'")
                } else {
                    write!(f, "Column '{column}' not found")
                }
            }
            Self::DuplicateColumn { column } => write!(f, "Duplicate column '{column}'"),
            Self::SchemaIncompatible { reason } => write!(f, "Schema incompatible: {reason}"),
            Self::TableNotFound { namespace, name } => {
                write!(f, "Table '{namespace}.{name}' not found in catalog")
            }
            Self::NullConstraintViolation { column } => {
                write!(f, "NULL constraint violation on column '{column}'")
            }

            Self::IndexNotFound { path } => write!(f, "Index file not found at '{path}'"),
            Self::IndexBuildFailed { index, reason } => {
                write!(f, "Index build failed for '{index}': {reason}")
            }
            Self::GpuNotAvailable { requested } => {
                write!(f, "GPU device '{requested}' not available")
            }
            Self::GpuCompileFailed { reason } => write!(f, "GPU compile failed: {reason}"),
            Self::UnsupportedVectorConfig { detail } => {
                write!(f, "Unsupported vector configuration: {detail}")
            }

            Self::CatalogConfig { catalog, reason } => {
                write!(f, "Catalog '{catalog}' config error: {reason}")
            }
            Self::CatalogRequest {
                catalog,
                status,
                reason,
            } => {
                if let Some(s) = status {
                    write!(f, "Catalog '{catalog}' request failed (HTTP {s}): {reason}")
                } else {
                    write!(f, "Catalog '{catalog}' request failed: {reason}")
                }
            }
            Self::BranchNotSupported { catalog } => {
                write!(f, "Branching not supported by catalog '{catalog}'")
            }
            Self::UnknownCatalogType { typ } => {
                write!(f, "Unknown catalog type: {typ}")
            }

            Self::ManifestVersionMismatch { expected, actual } => {
                write!(
                    f,
                    "Manifest version mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ManifestCommitFailed { reason } => {
                write!(f, "Manifest commit failed: {reason}")
            }
            Self::ManifestRollbackFailed { reason } => {
                write!(f, "Manifest rollback failed: {reason}")
            }

            Self::LockAcquireFailed { resource, retries } => {
                write!(
                    f,
                    "Failed to acquire lock on '{resource}' after {retries} retries"
                )
            }

            Self::FilterError { reason } => write!(f, "Filter error: {reason}"),
            Self::PhysicalExprError { reason } => write!(f, "Physical expression error: {reason}"),

            Self::SegmentNotFound { segment_id } => {
                write!(f, "Segment '{segment_id}' not found in manifest")
            }
            Self::DataIntegrityCheckFailed { segment_id, reason } => write!(
                f,
                "Data integrity check failed for segment '{segment_id}': {reason}"
            ),
            Self::BitmapExhausted => write!(f, "Bitmap iterator exhausted"),

            Self::PrimaryKeyViolation { key } => {
                write!(f, "Primary key violation: duplicate key '{key}'")
            }
            Self::WriteFailed { reason } => write!(f, "Write failed: {reason}"),

            Self::MergeFailed { reason } => write!(f, "Merge/upsert failed: {reason}"),

            Self::WalOpenError { path, source } => {
                write!(f, "WAL open error at '{path}': {source}")
            }
            Self::WalChannelClosed => write!(f, "WAL channel closed"),
            Self::WalCompactionFailed { reason } => {
                write!(f, "WAL compaction failed: {reason}")
            }

            Self::PuffinParseError { reason } => write!(f, "Puffin parse error: {reason}"),
            Self::PuffinBlobOutOfRange { requested, max } => {
                write!(
                    f,
                    "Puffin blob index out of range: requested {requested}, max {max}"
                )
            }
            Self::DeletionVectorDeserialize { reason } => {
                write!(f, "Deletion vector deserialize error: {reason}")
            }

            Self::IcebergSchemaError { reason } => {
                write!(f, "Iceberg schema error: {reason}")
            }
            Self::IcebergDeleteError { reason } => {
                write!(f, "Iceberg delete error: {reason}")
            }

            Self::InvalidLicense { reason } => write!(f, "Invalid license: {reason}"),
            Self::LicenseExpired { expired_at } => {
                write!(f, "License expired at {expired_at}")
            }
            Self::EnterpriseFeatureRequired { feature } => {
                write!(f, "Enterprise feature '{feature}' requires valid license")
            }

            Self::EmbeddingApiError { reason } => {
                write!(f, "Embedding API error: {reason}")
            }

            Self::BackgroundTaskFailed { task, reason } => {
                write!(f, "Background task '{task}' failed: {reason}")
            }
            Self::ConcurrencyLimit => write!(f, "Concurrency limit reached"),

            Self::Internal { source } => write!(f, "Internal error: {source}"),
        }
    }
}

impl std::error::Error for HyperstreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source } => Some(source),
            Self::ObjectStore { source } => Some(source),
            Self::Arrow { source } => Some(source),
            Self::DataFusion { source } => Some(source),
            Self::WalOpenError { source, .. } => Some(source),
            Self::Internal { source } => Some(source.as_ref()),
            _ => None,
        }
    }
}

// ─── From conversions for common upstream types ────────────────────────────────

impl From<std::io::Error> for HyperstreamError {
    fn from(source: std::io::Error) -> Self {
        Self::Io { source }
    }
}

impl From<object_store::Error> for HyperstreamError {
    fn from(source: object_store::Error) -> Self {
        Self::ObjectStore { source }
    }
}

impl From<arrow::error::ArrowError> for HyperstreamError {
    fn from(source: arrow::error::ArrowError) -> Self {
        Self::Arrow { source }
    }
}

impl From<datafusion::error::DataFusionError> for HyperstreamError {
    fn from(source: datafusion::error::DataFusionError) -> Self {
        Self::DataFusion { source }
    }
}

impl From<anyhow::Error> for HyperstreamError {
    fn from(source: anyhow::Error) -> Self {
        // Note: we can't downcast and re-wrap here because the upstream error
        // types (io::Error, object_store::Error, etc.) do not implement Clone.
        // The direct `From` impls on those types will catch them before they
        // become anyhow::Error.  This impl is the catch-all for anything that
        // survived the anyhow context chain.
        Self::Internal { source }
    }
}

// ─── Convenience constructors ─────────────────────────────────────────────────

impl HyperstreamError {
    /// Wrap a generic message into an `Internal` anyhow error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal {
            source: anyhow::Error::msg(msg.into()),
        }
    }

    /// Wrap a message with a source error (any type implementing `std::error::Error`).
    pub fn internal_context(
        msg: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Internal {
            source: anyhow::Error::new(source).context(msg.into()),
        }
    }
}
