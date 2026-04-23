// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use anyhow::Context;
use arrow::record_batch::RecordBatch;
// use arrow::array::Array; // Unused
use arrow::ipc::writer::StreamWriter;
use arrow::ipc::reader::StreamReader;
use arrow::datatypes::SchemaRef;
use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};

/// Write-Ahead Log for durability of in-memory writes.
/// Uses Arrow IPC Stream format for append-only logging.
/// Supports multiple processes by using unique log files in a shared directory.
pub struct WriteAheadLog {
    dir: PathBuf,
    path: PathBuf,
    writer: Option<StreamWriter<File>>,
    schema: Option<SchemaRef>,
    tx: Option<mpsc::Sender<LogOp>>,
}

enum LogOp {
    Append(RecordBatch, oneshot::Sender<Result<()>>),
}

impl std::fmt::Debug for WriteAheadLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteAheadLog")
            .field("dir", &self.dir)
            .field("path", &self.path)
            .field("schema", &self.schema)
            .finish()
    }
}

impl WriteAheadLog {
    /// Open or create a WAL directory.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        // Create a unique filename for this instance to avoid clobbering by other processes
        let id = uuid::Uuid::new_v4();
        let path = dir.join(format!("log_{}.arrow", id));
        
        Self {
            dir,
            path,
            writer: None,
            schema: None,
            tx: None,
        }
    }

    /// Start a background worker for asynchronous writes.
    pub fn spawn_worker(&mut self) -> Result<()> {
        if self.tx.is_some() {
            return Ok(());
        }

        let (tx, mut rx) = mpsc::channel::<LogOp>(1024);
        self.tx = Some(tx);

        // Move state into worker
        let path = self.path.clone();
        let mut writer_opt: Option<StreamWriter<File>> = self.writer.take();
        
        tokio::spawn(async move {
            let mut pending_syncs = Vec::new();
            let mut batch_count = 0;
            
            // Configurable sync batching: sync every N batches or every T milliseconds
            let sync_batch_size = std::env::var("HYPERSTREAM_WAL_SYNC_BATCH_SIZE")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<usize>().unwrap_or(10);
            let sync_interval_ms: u64 = std::env::var("HYPERSTREAM_WAL_SYNC_INTERVAL_MS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100);
            
            loop {
                // Wait for a message or a wait timeout to sync
                let timeout = tokio::time::sleep(std::time::Duration::from_millis(sync_interval_ms));
                
                tokio::select! {
                    msg = rx.recv() => {
                        match msg {
                            Some(LogOp::Append(batch, reply_tx)) => {
                                // Ensure writer
                                if writer_opt.is_none() {
                                    let file = match OpenOptions::new()
                                        .create(true)
                                        .append(true)
                                        .open(&path) {
                                            Ok(f) => f,
                                            Err(e) => {
                                                let _ = reply_tx.send(Err(anyhow::anyhow!("Failed to open WAL: {}", e)));
                                                continue;
                                            }
                                        };
                                    writer_opt = match StreamWriter::try_new(file, &batch.schema()) {
                                        Ok(w) => Some(w),
                                        Err(e) => {
                                             let _ = reply_tx.send(Err(anyhow::anyhow!("Failed to create WAL writer: {}", e)));
                                             continue;
                                        }
                                    };
                                }

                                if let Some(writer) = &mut writer_opt {
                                    if let Err(e) = writer.write(&batch) {
                                        let _ = reply_tx.send(Err(anyhow::anyhow!("WAL write failed: {}", e)));
                                    } else {
                                        pending_syncs.push(reply_tx);
                                        batch_count += 1;
                                        
                                        // Sync if we've reached batch size threshold
                                        if batch_count >= sync_batch_size {
                                            // Use fdatasync for better performance (only syncs data, not metadata)
                                            if let Err(e) = writer.get_ref().sync_data() {
                                                eprintln!("WAL sync_data failed: {}", e);
                                                // Fallback to sync_all if fdatasync not available
                                                let _ = writer.get_ref().sync_all();
                                            }
                                            
                                            // Reply to all pending syncs
                                            for tx in pending_syncs.drain(..) {
                                                let _ = tx.send(Ok(()));
                                            }
                                            batch_count = 0;
                                        }
                                    }
                                }
                            }
                            None => break, // Channel closed
                        }
                    }
                    _ = timeout => {
                        // Periodic sync - sync any pending writes
                        if !pending_syncs.is_empty() {
                            if let Some(writer) = &mut writer_opt {
                                // Use fdatasync for better performance
                                if let Err(e) = writer.get_ref().sync_data() {
                                    eprintln!("WAL sync_data failed: {}", e);
                                    let _ = writer.get_ref().sync_all();
                                }
                            }
                            for tx in pending_syncs.drain(..) {
                                let _ = tx.send(Ok(()));
                            }
                            batch_count = 0;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Append a batch to the log asynchronously.
    pub async fn append_async(&self, batch: RecordBatch) -> Result<()> {
        if let Some(tx) = &self.tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(LogOp::Append(batch, reply_tx)).await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;
            
            reply_rx.await
                .map_err(|_| anyhow::anyhow!("WAL worker dropped request"))?
        } else {
            // Fallback to sync? Or error?
            anyhow::bail!("WAL worker not started. Call spawn_worker() first.");
        }
    }

    /// Replay all log files in the WAL directory and return an iterator of batches.
    /// This should be used on startup for memory-efficient recovery.
    pub fn replay_stream(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>>>> {
        if !self.dir.exists() {
            return Ok(Box::new(std::iter::empty()));
        }

        // 1. List all .arrow files in the directory
        let entries = std::fs::read_dir(&self.dir)?;
        let mut wal_files = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
                wal_files.push(path);
            }
        }

        // Sort for deterministic replay
        wal_files.sort();

        let mut all_iterators = Vec::new();

        for path in wal_files {
            let file = File::open(&path)?;
            if file.metadata()?.len() == 0 {
                continue;
            }
            
            let reader = BufReader::new(file);
            let ipc_reader = StreamReader::try_new(reader, None)?;
            all_iterators.push(ipc_reader);
        }

        Ok(Box::new(all_iterators.into_iter().flatten().map(|res| res.map_err(anyhow::Error::from))))
    }

    /// Replay all log files in the WAL directory and return all batches.
    /// Legacy method, consider using replay_stream for large logs.
    pub fn replay(&self) -> Result<(Vec<RecordBatch>, Vec<String>)> {
        let stream = self.replay_stream()?;
        let mut batches = Vec::new();
        for b in stream {
            batches.push(b?);
        }
        
        // Return paths for cleanup (simplified for now)
        let mut paths = Vec::new();
        if self.dir.exists() {
            for entry in std::fs::read_dir(&self.dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
                    paths.push(path.to_str().unwrap().to_string());
                }
            }
        }

        Ok((batches, paths))
    }

    /// Initialize the writer with a schema.
    /// Must be called before first append.
    fn ensure_writer(&mut self, schema: SchemaRef) -> Result<()> {
        if self.writer.is_some() {
            return Ok(());
        }

        self.schema = Some(schema.clone());

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .context("Failed to open WAL file")?;

        // If file is new/empty, we need to write the schema header?
        // Actually, IPC Stream format writes schema at the start.
        // But if we are appending to an existing file, we can't just create a new StreamWriter
        // because it writes a header every time.
        
        // Strategy:
        // For simple WAL, we can just keep the file open.
        // If we close and reopen, we must be careful.
        // BUT: Arrow IPC Stream format allows concatenating messages? 
        // Standard StreamWriter writes schema header.
        
        // Better approach for crash recovery:
        // Always write to a NEW file for a new "session" or just overwrite if we flushed?
        // Actually, if we are recovering, we read everything, put it in memory, 
        // and can effectively TRUNCATE the log and start fresh for new writes since they are now in memory.
        
        // So:
        // 1. replay() reads existing data.
        // 2. truncate() clears the file (since data is now in memory).
        // 3. append() starts a fresh stream.

        let writer = StreamWriter::try_new(file, &schema)?;
        self.writer = Some(writer);
        Ok(())
    }

    pub fn append(&mut self, batch: &RecordBatch) -> Result<()> {
        // Ensure writer exists
        self.ensure_writer(batch.schema())?;
        
        if let Some(writer) = &mut self.writer {
            writer.write(batch)?;
            // Sync to disk for durability!
            writer.get_ref().sync_all()?; 
        }
        Ok(())
    }

    /// Clear the log file owned by this instance.
    /// Should be called after data is successfully persisted (flushed) to main storage.
    pub fn truncate(&mut self) -> Result<()> {
        self.writer = None; // Drop writer
        if self.path.exists() {
             std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }

    /// Delete specific log files (e.g. after replaying old logs on startup)
    pub fn cleanup_files(&self, paths: &[PathBuf]) -> Result<()> {
        for path in paths {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    /// Check if WAL should be compacted based on file size
    pub fn should_compact(&self) -> Result<bool> {
        if !self.path.exists() {
            return Ok(false);
        }

        let metadata = std::fs::metadata(&self.path)?;
        let size_mb = metadata.len() / (1024 * 1024);
        
        let threshold_mb: u64 = std::env::var("HYPERSTREAM_WAL_COMPACT_MB")
            .unwrap_or_else(|_| "1024".to_string())
            .parse()
            .unwrap_or(1024);
            
        Ok(size_mb > threshold_mb)
    }

    /// Compact WAL by consolidating all batches into one
    /// This reduces recovery time and file size
    pub fn compact(&mut self) -> Result<()> {
        // 1. Replay all batches
        let (batches, _) = self.replay()?;
        if batches.is_empty() || batches.len() == 1 {
            // Already compact or empty
            return Ok(());
        }

        // 2. Concatenate batches
        let schema = batches[0].schema();
        let consolidated = arrow::compute::concat_batches(&schema, &batches)
            .context("Failed to concatenate batches during WAL compaction")?;

        // 3. Write to temp file
        let temp_path = self.path.with_extension("arrow.tmp");
        let temp_file = File::create(&temp_path)
            .context("Failed to create temp WAL file")?;
        let mut temp_writer = StreamWriter::try_new(temp_file, &schema)?;
        temp_writer.write(&consolidated)?;
        temp_writer.finish()?;
        drop(temp_writer);

        // 4. Atomic replace
        std::fs::rename(&temp_path, &self.path)
            .context("Failed to replace WAL file")?;

        // 5. Reset writer (will be recreated on next append)
        self.writer = None;
        
        println!("WAL: Compacted {} batches into 1 batch", batches.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_test_batch(start: i32, count: i32) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let ids = Int32Array::from((start..start + count).collect::<Vec<i32>>());
        RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
    }

    #[test]
    fn test_wal_basic_operations() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("test_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        // Test 1: Create new WAL and append
        let mut wal = WriteAheadLog::new(&wal_dir);
        let batch1 = create_test_batch(0, 10);
        wal.append(&batch1)?;
        
        // Test 2: Replay should return the batch
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].num_rows(), 10);
        
        // Test 3: Append another batch
        let batch2 = create_test_batch(10, 10);
        wal.append(&batch2)?;
        
        // Test 4: Replay should return both batches
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].num_rows(), 10);
        assert_eq!(replayed[1].num_rows(), 10);
        
        // Test 5: Truncate should clear the log
        wal.truncate()?;
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 0);
        
        Ok(())
    }

    #[test]
    fn test_wal_compaction() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("test_compact_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        let mut wal = WriteAheadLog::new(&wal_dir);
        
        // Write multiple small batches
        for i in 0..5 {
            let batch = create_test_batch(i * 10, 10);
            wal.append(&batch)?;
        }
        
        // Verify we have 5 batches
        let (before_compact, _) = wal.replay()?;
        assert_eq!(before_compact.len(), 5);
        
        // Compact the WAL
        wal.compact()?;
        
        // After compaction, should have 1 batch with all rows
        let (after_compact, _) = wal.replay()?;
        assert_eq!(after_compact.len(), 1);
        assert_eq!(after_compact[0].num_rows(), 50);
        
        Ok(())
    }

    #[test]
    fn test_wal_crash_recovery() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("test_crash_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        // Simulate: Write some data and "crash" (drop the WAL)
        {
            let mut wal = WriteAheadLog::new(&wal_dir);
            let batch1 = create_test_batch(0, 100);
            wal.append(&batch1)?;
            let batch2 = create_test_batch(100, 100);
            wal.append(&batch2)?;
            // WAL goes out of scope here (simulating crash)
        }
        
        // Simulate: Restart and replay
        {
            let wal = WriteAheadLog::new(&wal_dir);
            let (recovered, _) = wal.replay()?;
            assert_eq!(recovered.len(), 2);
            assert_eq!(recovered[0].num_rows(), 100);
            assert_eq!(recovered[1].num_rows(), 100);
        }
        
        Ok(())
    }

    #[test]
    fn test_wal_empty_dir() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("empty_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        let wal = WriteAheadLog::new(&wal_dir);
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 0);
        
        Ok(())
    }

    #[test]
    fn test_wal_nonexistent_dir() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("nonexistent_wal");
        
        let wal = WriteAheadLog::new(&wal_dir);
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 0);
        
        Ok(())
    }

    #[test]
    fn test_wal_large_batches() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("large_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        let mut wal = WriteAheadLog::new(&wal_dir);
        
        // Write a large batch (100K rows)
        let large_batch = create_test_batch(0, 100_000);
        wal.append(&large_batch)?;
        
        // Verify replay
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].num_rows(), 100_000);
        
        Ok(())
    }

    #[test]
    fn test_wal_should_compact() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("should_compact_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        let mut wal = WriteAheadLog::new(&wal_dir);
        std::env::set_var("HYPERSTREAM_WAL_COMPACT_MB", "100");
        
        // Initially should not need compaction
        assert!(!wal.should_compact()?);
        
        // Write many batches to exceed 100MB threshold
        // Each batch is ~400KB (100K i32 values), so we need ~250 batches
        for i in 0..260 {
            let batch = create_test_batch(i * 100_000, 100_000);
            wal.append(&batch)?;
        }
        
        // Now should need compaction
        assert!(wal.should_compact()?);
        
        Ok(())
    }

    #[test]
    fn test_wal_multiple_schemas() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("multi_schema_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        let mut wal = WriteAheadLog::new(&wal_dir);
        
        // Write batch with one schema
        let schema1 = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let batch1 = RecordBatch::try_new(
            schema1,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;
        wal.append(&batch1)?;
        
        // Truncate and start fresh
        wal.truncate()?;
        
        // Write batch with different schema
        let schema2 = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));
        let batch2 = RecordBatch::try_new(
            schema2,
            vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
        )?;
        wal.append(&batch2)?;
        
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].schema().field(0).name(), "value");
        
        Ok(())
    }

    #[test]
    fn test_wal_compaction_preserves_data() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("compact_preserve_wal");
        std::fs::create_dir_all(&wal_dir)?;
        
        let mut wal = WriteAheadLog::new(&wal_dir);
        
        // Write specific data
        let expected_values: Vec<i32> = (0..100).collect();
        for chunk in expected_values.chunks(10) {
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
                vec![Arc::new(Int32Array::from(chunk.to_vec()))],
            )?;
            wal.append(&batch)?;
        }
        
        // Compact
        wal.compact()?;
        
        // Verify all data is preserved
        let (replayed, _) = wal.replay()?;
        assert_eq!(replayed.len(), 1);
        let ids = replayed[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        
        let actual_values: Vec<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();
        assert_eq!(actual_values, expected_values);
        
        Ok(())
    }
}
