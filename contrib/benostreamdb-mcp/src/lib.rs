#![deny(clippy::unwrap_used, clippy::expect_used)]

//! BenoStreamDB MCP Core Library
//!
//! Provides the reusable Model Context Protocol (MCP) server engine for BenoStreamDB,
//! supporting extensible tool registries, custom table schemas, and pluggable
//! file extractors for domain verticals (Finance, Healthcare, Legal, etc.).

pub mod ingest_formats;
pub mod protocol;
pub mod server;

pub use benostreamdb::core::catalog::{
    get_catalog_policy, lock_catalog_read_only, resolve_permissions_for_untrusted_connector,
    set_catalog_policy, set_catalog_schema_access, CatalogPolicy, CATALOG_POLICY_FILE,
    IMMUTABLE_SYSTEM_SCHEMAS,
};
pub use protocol::{JsonRpcRequest, JsonRpcResponse};
pub use server::{
    resolve_default_warehouse, validate_local_path, CustomSchemaResolver, McpExtension, McpServer,
    McpServerConfig, SchemaPermissions, SCRATCH_SCHEMA,
};
