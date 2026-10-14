#![deny(clippy::unwrap_used, clippy::expect_used)]

//! BenoStreamDB MCP Core Library
//!
//! Provides the reusable Model Context Protocol (MCP) server engine for BenoStreamDB,
//! supporting extensible tool registries, custom table schemas, and pluggable
//! file extractors for domain verticals (Finance, Healthcare, Legal, etc.).

pub mod ingest_formats;
pub mod protocol;
pub mod server;

pub use protocol::{JsonRpcRequest, JsonRpcResponse};
pub use server::{CustomSchemaResolver, McpExtension, McpServer, SCRATCH_SCHEMA};
