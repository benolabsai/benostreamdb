mod protocol;
mod server;

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use crate::server::McpServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging to STDERR (stdout is reserved for MCP JSON-RPC messages)
    let subscriber = FmtSubscriber::builder()
        .with_writer(std::io::stderr)
        .with_max_level(Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    info!("BenoStreamDB MCP Server initializing...");

    if std::env::var("BSDB_WAREHOUSE").is_err() {
        let temp_dir = std::env::temp_dir().join(format!("benostreamdb_mcp_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_var("BSDB_WAREHOUSE", temp_dir.to_str().unwrap());
        info!("Set BSDB_WAREHOUSE to temporary directory: {:?}", temp_dir);
    }

    let mcp_server = McpServer::new();
    mcp_server.run().await?;

    info!("Server loop exited");

    Ok(())
}
