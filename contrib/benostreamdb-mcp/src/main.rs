#![deny(clippy::unwrap_used, clippy::expect_used)]

mod protocol;
mod server;

use crate::server::McpServer;
use anyhow::Result;
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

    if std::env::var("BSDB_WAREHOUSE").is_err() {
        let temp_dir = std::env::temp_dir().join(format!(
            "benostreamdb_mcp_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&temp_dir);
        std::env::set_var("BSDB_WAREHOUSE", temp_dir.to_str().unwrap_or(""));
        info!("Set BSDB_WAREHOUSE to temporary directory: {:?}", temp_dir);
    }

    let mcp_server = McpServer::new();
    mcp_server.run().await?;

    info!("Server loop exited");

    Ok(())
}
