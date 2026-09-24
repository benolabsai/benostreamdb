// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::license::verify_license as validate_license;
use crate::core::table::Table;
/// Incremental index update support for BenoStreamDB
///
/// This module provides continuous indexing capabilities:
/// when new data is appended to the write buffer, only the new rows
/// are indexed rather than rebuilding the entire index from scratch.
use anyhow::Result;

/// Enterprise index builder with continuous/incremental indexing
pub struct ContinuousIndexBuilder {
    license_key: Option<String>,
}

impl ContinuousIndexBuilder {
    pub fn new(license_key: Option<String>) -> Result<Self> {
        if let Some(key) = &license_key {
            validate_license(key)?;
        }
        Ok(Self { license_key })
    }

    pub fn is_licensed(&self) -> bool {
        self.license_key.is_some()
    }

    /// Perform an incremental index update on a table
    ///
    /// Flushes the write buffer and backfills indexes for the new data only,
    /// avoiding a full index rebuild. Requires an enterprise license.
    pub fn perform_incremental_update(&self, table: &Table) -> Result<()> {
        if !self.is_licensed() {
            anyhow::bail!("Incremental updates require an enterprise license.");
        }

        // Check if there's uncommitted data to index
        let buffered_rows = table.write_buffer_row_count();
        if buffered_rows == 0 {
            log::info!("[Continuous Indexing] No uncommitted data in write buffer, skipping incremental update");
            return Ok(());
        }

        log::info!(
            "[Continuous Indexing] Flushing {} rows from write buffer and updating indexes",
            buffered_rows
        );

        // Get the indexed columns to backfill
        let index_columns = table.get_index_columns();
        if index_columns.is_empty() {
            log::info!("[Continuous Indexing] No indexed columns configured, skipping");
            return Ok(());
        }

        // Flush the write buffer to disk, then backfill indexes for the new segment
        table
            .commit()
            .map_err(|e| anyhow::anyhow!("Commit failed: {}", e))?;

        table
            .backfill_indexes(index_columns)
            .map_err(|e| anyhow::anyhow!("Backfill failed: {}", e))?;

        log::info!("[Continuous Indexing] Incremental update completed successfully");
        Ok(())
    }
}
