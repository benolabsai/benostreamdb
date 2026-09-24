// Copyright (c) 2026 Richard Albright. All rights reserved.

// No-panic policy for production binaries (see NO_PANIC_POLICY.md).
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use arrow::util::pretty::print_batches;
use benostreamdb::core::sql::session::BenoStreamSession;
use benostreamdb::core::table::Table;
use clap::{Parser, Subcommand};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "bsdb")]
#[command(about = "BenoStreamDB CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a SQL REPL (default)
    Repl,
    /// Execute a single SQL query
    Query {
        #[arg(short, long)]
        query: String,
    },
    /// Table management commands
    Table {
        #[command(subcommand)]
        command: TableCommands,
    },
    /// Register a table from a URI (Legacy/Convenience)
    Register {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        uri: String,
    },
}

#[derive(Subcommand)]
enum TableCommands {
    /// Inspect table metadata
    Inspect {
        /// Table URI
        #[arg(short, long)]
        uri: String,
    },
    /// Compact small files into larger segments
    Compact {
        /// Table URI
        #[arg(short, long)]
        uri: String,
    },
    /// Remove old files
    Vacuum {
        /// Table URI
        #[arg(short, long)]
        uri: String,
        /// Remove files older than N days
        #[arg(long, default_value_t = 7)]
        older_than_days: u64,
    },
    /// Bulk-ingest parquet files (native orchestrator: plan → parallel → commit)
    Ingest {
        /// Table URI
        #[arg(short, long)]
        uri: String,
        /// Input file(s) to ingest (.parquet/.csv/.json/.ndjson/.arrow/.ipc)
        #[arg(short, long, num_args = 1..)]
        input: Vec<String>,
        /// Rows per work unit
        #[arg(long, default_value_t = 1_000_000)]
        chunk_rows: usize,
        /// Max work units in flight
        #[arg(long, default_value_t = 4)]
        parallelism: usize,
        /// Build indexes for every column
        #[arg(long, default_value_t = false)]
        index_all: bool,
        /// Print the planned work units and exit (do not execute)
        #[arg(long, default_value_t = false)]
        plan: bool,
        /// Restrict to a single row range [start, end) of the first input
        /// (serverless thin-runner mode)
        #[arg(long)]
        row_start: Option<usize>,
        #[arg(long)]
        row_end: Option<usize>,
        /// Run compaction after the ingest
        #[arg(long, default_value_t = false)]
        compact: bool,
        /// Return freed heap pages to the OS after a unit once RSS exceeds this
        /// budget (GB). Falls back to BSDB_INGEST_MEMORY_BUDGET_GB when unset.
        #[arg(long)]
        memory_budget_gb: Option<f64>,
        /// Multi-machine mode: claim units from a shared object-store lease
        /// queue instead of a fixed local list. Run on N machines with the same
        /// --input to split the work.
        #[arg(long, default_value_t = false)]
        coordinate: bool,
        /// Lease TTL (seconds) for --coordinate. A dead node's lease expires
        /// after this and another node steals its unit.
        #[arg(long, default_value_t = 300)]
        lease_ttl_secs: u64,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = benostreamdb::telemetry::tracing::init_tracing("bsdb")?;

    let cli = Cli::parse();
    let session = BenoStreamSession::new(None);

    match cli.command {
        Some(Commands::Query { query }) => {
            run_query(&session, &query).await;
        }
        Some(Commands::Table { command }) => match command {
            TableCommands::Inspect { uri } => inspect_table(&uri).await?,
            TableCommands::Compact { uri } => compact_table(&uri).await?,
            TableCommands::Vacuum {
                uri,
                older_than_days,
            } => vacuum_table(&uri, older_than_days).await?,
            TableCommands::Ingest {
                uri,
                input,
                chunk_rows,
                parallelism,
                index_all,
                plan,
                row_start,
                row_end,
                compact,
                memory_budget_gb,
                coordinate,
                lease_ttl_secs,
            } => {
                ingest_table(
                    &uri,
                    &input,
                    chunk_rows,
                    parallelism,
                    index_all,
                    plan,
                    row_start,
                    row_end,
                    compact,
                    memory_budget_gb,
                    coordinate,
                    lease_ttl_secs,
                )
                .await?
            }
        },
        Some(Commands::Register { name, uri }) => {
            println!("Registering table '{}' at '{}'", name, uri);
            let table = Table::new_async(uri).await?;
            session.register_table(&name, Arc::new(table))?;
            println!("Table registered.");
        }
        Some(Commands::Repl) | None => {
            run_repl(session).await?;
        }
    }

    Ok(())
}

async fn inspect_table(uri: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Inspecting table: {}", uri);
    let table = Table::new_async(uri.to_string()).await?;
    let stats = table.get_table_statistics_async().await?;

    println!("--- Table Statistics ---");
    println!("Row Count: {}", stats.row_count);
    println!("File Count: {}", stats.file_count);
    println!("Total Size: {} bytes", stats.total_size_bytes);
    println!("Index Coverage: {:?}", stats.index_coverage);
    Ok(())
}

async fn compact_table(uri: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Compacting table: {}", uri);
    let table = Table::new_async(uri.to_string()).await?;
    let start = Instant::now();
    table.rewrite_data_files_async(None).await?;
    println!("Compaction completed in {:.2?}", start.elapsed());
    Ok(())
}

async fn vacuum_table(uri: &str, days: u64) -> Result<(), Box<dyn std::error::Error>> {
    println!("Vacuuming table: {} (older than {} days)", uri, days);
    let table = Table::new_async(uri.to_string()).await?;
    let start = Instant::now();
    let deleted_count = table.vacuum_async(days as usize).await?;
    println!(
        "Vacuum completed in {:.2?}. Deleted {} files.",
        start.elapsed(),
        deleted_count
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn ingest_table(
    uri: &str,
    input: &[String],
    chunk_rows: usize,
    parallelism: usize,
    index_all: bool,
    plan: bool,
    row_start: Option<usize>,
    row_end: Option<usize>,
    compact: bool,
    memory_budget_gb: Option<f64>,
    coordinate: bool,
    lease_ttl_secs: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    use benostreamdb::core::table::IngestOptions;

    let table = Table::new_async(uri.to_string()).await?;

    if plan {
        let units = table.plan_ingest(input, chunk_rows)?;
        println!("Planned {} work unit(s):", units.len());
        for (p, s, e) in &units {
            println!("  {} [{}..{})", p, s, e);
        }
        return Ok(());
    }

    let opts = IngestOptions {
        chunk_rows,
        parallelism,
        index_all,
        resume: true,
        compact_after: compact,
        memory_budget_bytes: memory_budget_gb
            .filter(|gb| *gb > 0.0)
            .map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as u64),
    };

    let start = Instant::now();
    let report = if coordinate {
        let coordinator = table.object_store_coordinator(
            input,
            chunk_rows,
            std::time::Duration::from_secs(lease_ttl_secs.max(1)),
        )?;
        table.ingest_coordinated_async(opts, coordinator).await?
    } else {
        match (row_start, row_end) {
            (Some(s), Some(e)) => {
                let path = input
                    .first()
                    .ok_or("--row-start/--row-end require --input")?;
                table.ingest_range_async(path, s, e, opts).await?
            }
            _ => table.ingest_async(input, opts).await?,
        }
    };
    println!(
        "Ingest complete in {:.2?}: {} unit(s) committed, {} skipped, {} rows, {} segment(s)",
        start.elapsed(),
        report.units_committed,
        report.units_skipped,
        report.rows_ingested,
        report.segments.len()
    );
    Ok(())
}

async fn run_repl(session: BenoStreamSession) -> Result<(), Box<dyn std::error::Error>> {
    let mut rl = DefaultEditor::new()?;
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }

    println!("Welcome to BenoStreamDB CLI (bsdb)");
    println!("Type 'exit' or 'quit' to leave.");
    println!("Type 'register table_name uri' to register a table.");

    loop {
        let readline = rl.readline("bsdb> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                rl.add_history_entry(line)?;

                if line.eq_ignore_ascii_case("exit") || line.eq_ignore_ascii_case("quit") {
                    break;
                }

                if line.is_empty() {
                    continue;
                }

                if line.to_lowercase().starts_with("register ") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() == 3 {
                        let name = parts[1];
                        let uri = parts[2];
                        match Table::new_async(uri.to_string()).await {
                            Ok(table) => {
                                if let Err(e) = session.register_table(name, Arc::new(table)) {
                                    println!("Error registering table: {}", e);
                                } else {
                                    println!("Table '{}' registered.", name);
                                }
                            }
                            Err(e) => println!("Error creating table: {}", e),
                        }
                        continue;
                    }
                }

                run_query(&session, line).await;
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    rl.save_history("history.txt")?;
    Ok(())
}

async fn run_query(session: &BenoStreamSession, query: &str) {
    let start = Instant::now();
    match session.sql(query).await {
        Ok((batches, _schema)) => {
            let duration = start.elapsed();
            if batches.is_empty() {
                println!("Query returned 0 rows in {:.2?}", duration);
            } else {
                let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                println!("Query returned {} rows in {:.2?}", num_rows, duration);
                if let Err(e) = print_batches(&batches) {
                    eprintln!("failed to print result batches: {e}");
                }
            }
        }
        Err(e) => {
            println!("Error executing query: {}", e);
        }
    }
}
