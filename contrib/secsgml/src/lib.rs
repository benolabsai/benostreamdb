//! # secsgml
//!
//! High-performance native Rust port of `secsgml` / `secsgml2` for parsing SEC EDGAR
//! SGML submissions (`.nc`, `.sgml`, `.txt`).
//!
//! Features:
//! - Pure, safe Rust implementation with zero required C compiler toolchains.
//! - SIMD-accelerated boundary scans and tag dispatch via `memchr`.
//! - High-speed UU-decoding with 60-byte fast path and malformed line tolerance.
//! - Normalization of submission header trees (SIC code extraction, SEC Act extraction, tag standardization).
//! - Automatic decoding and extraction of binary attachments (GIF, JPG, PDF) and XML/XBRL text payloads.
//! - Exact compatibility with the `secsgml2` Python API.

pub mod error;
pub mod metadata;
pub mod parser;
pub mod utils;
pub mod uu;

pub use error::SecSgmlError;
pub use metadata::{
    build_metadata_tree, parse_submission_metadata, standardize_submission_metadata,
    SubmissionEvent, SubmissionEventType,
};
pub use parser::{parse_sgml, parse_sgml_content_into_memory, Document, DocumentMeta};
pub use utils::calculate_documents_locations_in_tar;
pub use uu::{
    decode_uuencoded_content, find_uu_body_bounds, is_uuencoded, strip_uu_wrappers, uudecode,
};
