// Copyright (c) 2026 Richard Albright. All rights reserved.

fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim().to_uppercase();
    let mut num_str = String::new();
    let mut unit_str = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            num_str.push(c);
        } else if c.is_ascii_alphabetic() {
            unit_str.push(c);
        }
    }

    let num = num_str.parse::<f64>().map_err(|_| format!("Invalid number in size: {}", s))?;

    let multiplier = match unit_str.as_str() {
        "" | "B" => 1.0,
        "KB" | "K" => 1024.0,
        "MB" | "M" => 1024.0 * 1024.0,
        "GB" | "G" => 1024.0 * 1024.0 * 1024.0,
        "TB" | "T" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return Err(format!("Unknown unit in size: {}. Use B, KB, MB, GB, or TB.", unit_str)),
    };

    Ok((num * multiplier) as u64)
}

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "hyperstreamdb-admin")]
#[command(about = "Administrative tooling for HyperStreamDB", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run compaction on a table to merge small data files and optimize query performance
    Compact {
        /// URI of the HyperStreamDB table
        #[arg(long, short)]
        uri: String,

        /// Target file size (e.g., 64MB, 1GB)
        #[arg(long, default_value = "64MB", value_parser = parse_size)]
        target_file_size: u64,
    },
    /// Remove unreferenced data files to reclaim storage space
    Vacuum {
        /// URI of the HyperStreamDB table
        #[arg(long, short)]
        uri: String,

        /// Number of versions of history to retain
        #[arg(long, default_value_t = 10)]
        retain_versions: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Compact { uri, target_file_size } => {
            tracing::info!("Starting compaction on table: {}", uri);
            let table = hyperstreamdb::Table::new_async(uri.clone()).await?;
            let opts = hyperstreamdb::core::compaction::CompactionOptions {
                target_file_size_bytes: *target_file_size as i64,
                ..Default::default()
            };
            table.rewrite_data_files_async(Some(opts)).await?;
            println!("Compaction completed successfully for {}", uri);
        }
        Commands::Vacuum { uri, retain_versions } => {
            tracing::info!("Starting vacuum on table: {}", uri);
            let table = hyperstreamdb::Table::new_async(uri.clone()).await?;
            let deleted = table.vacuum_async(*retain_versions).await?;
            println!("Vacuum completed for {}. Deleted {} unreferenced files.", uri, deleted);
        }
    }

    Ok(())
}
