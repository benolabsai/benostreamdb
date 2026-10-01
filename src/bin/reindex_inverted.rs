// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Rebuild the position-aware inverted index for a column, in place.
//!
//! Upgrades legacy 2-column (position-less) `.inv.parquet` sidecars to the
//! 3-column position-aware format without touching the (potentially huge)
//! vector indexes. See `Table::reindex_inverted_column`.
//!
//! Example:
//! ```text
//! cargo run --release --bin reindex_inverted -- \
//!     --table-uri file:///home/me/data/wiki_graph_db/nodes --column title
//! ```

use anyhow::Result;
use benostreamdb::Table;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Table URI, e.g. `file:///home/me/data/wiki_graph_db/nodes`.
    #[arg(long)]
    table_uri: String,

    /// Column whose inverted index should be rebuilt.
    #[arg(long, default_value = "title")]
    column: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    println!("Opening table {} ...", args.table_uri);
    let table = Table::builder(args.table_uri.clone()).build_async().await?;

    println!(
        "Reindexing inverted index for column '{}' (position-aware) ...",
        args.column
    );
    let rebuilt = table.reindex_inverted_column(&args.column).await?;
    println!("Rebuilt {rebuilt} segment(s).");

    Ok(())
}
