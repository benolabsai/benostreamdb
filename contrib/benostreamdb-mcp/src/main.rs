#![deny(clippy::unwrap_used, clippy::expect_used)]

use anyhow::Result;
use benostreamdb_mcp::{resolve_default_warehouse, McpServer, McpServerConfig};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging to STDERR (stdout is reserved for MCP JSON-RPC messages)
    let subscriber = FmtSubscriber::builder()
        .with_writer(std::io::stderr)
        .with_max_level(Level::DEBUG)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    info!("BenoStreamDB MCP Server initializing...");

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "BenoStreamDB Model Context Protocol (MCP) Server\n\n\
            USAGE:\n    \
                benostreamdb-mcp [OPTIONS]\n\n\
            OPTIONS:\n    \
                --warehouse <PATH>              Persistent local warehouse path for Iceberg tables\n    \
                --readwrite-schemas <SCHEMAS>   Comma-separated list of writable schemas (default: scratch)\n    \
                --readonly-schemas <SCHEMAS>    Comma-separated list of read-only schemas\n    \
                --default-write-schema <SCHEMA> Default schema for unqualified table creations\n    \
                --ephemeral                     Use ephemeral temporary directory for this session\n    \
                --config <JSON_STRING>          Full JSON configuration string\n    \
                --config-file <PATH>            Path to JSON configuration file\n    \
                -h, --help                      Print this help message\n\n\
            ENVIRONMENT VARIABLES:\n    \
                BSDB_WAREHOUSE                  Local warehouse path\n    \
                BSDB_READWRITE_SCHEMAS          Comma-separated list of writable schemas\n    \
                BSDB_READONLY_SCHEMAS           Comma-separated list of read-only schemas\n    \
                BSDB_DEFAULT_WRITE_SCHEMA       Default schema for writes\n    \
                BSDB_EPHEMERAL                  Set to 1 or true for ephemeral mode\n    \
                BSDB_MCP_CONFIG                 JSON string with full server configuration\n"
        );
        return Ok(());
    }

    let mut config = McpServerConfig::resolve_from_args_and_env(&args)?;

    if config.warehouse.is_none() && !config.ephemeral.unwrap_or(false) {
        let warehouse_path = resolve_default_warehouse();
        config.warehouse = Some(warehouse_path.clone());
        info!("Defaulted warehouse path to: {}", warehouse_path);
    }

    let mcp_server = McpServer::with_config(config);
    mcp_server.run().await?;

    info!("Server loop exited");

    Ok(())
}
