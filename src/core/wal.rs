// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::Context;
use anyhow::Result;
use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

static WAL_FILE_SEQ: AtomicU64 = AtomicU64::new(0);

fn next_wal_filename(dir: &Path) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = WAL_FILE_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    dir.join(format!("log_{:020}_{:06}_{:08}.arrow", ts, pid, seq))
}

#[derive(Debug, Clone)]
struct WalSegmentEntry {
    path: PathBuf,
    txs: HashSet<uuid::Uuid>,
    has_untagged: bool,
}

#[derive(Debug, Default)]
struct WalSegmentRegistry {
    active: Option<WalSegmentEntry>,
    sealed: Vec<WalSegmentEntry>,
}

/// Recovered batches grouped by source WAL segment file.
#[derive(Debug, Clone)]
pub struct WalRecoveredSegment {
    pub path: PathBuf,
    pub batches: Vec<RecordBatch>,
}

const IPC_CONTINUATION_MARKER: [u8; 4] = [0xFF; 4];
const MAX_WAL_SCHEMA_META_BYTES: u64 = 64 * 1024 * 1024;

enum OpenedWalSegment {
    /// Segment file was deleted concurrently before opening.
    Vanished,
    /// Segment contains no complete batches (0-byte or crashed mid-schema-header).
    Empty,
    /// Segment opened at its first record batch with torn-EOF tail protection.
    Reader(Box<SafeIpcStreamIter<File>>),
}

enum OpenedWalStream<R: std::io::Read> {
    /// Stream contains no complete batches (0-byte or crashed mid-schema-header).
    Empty,
    /// Stream opened at its first record batch with torn-EOF tail protection.
    Reader(Box<SafeIpcStreamIter<R>>),
}

struct SafeIpcStreamIter<R: std::io::Read> {
    label: String,
    inner: StreamReader<BufReader<R>>,
    finished: bool,
}

impl<R: std::io::Read> Iterator for SafeIpcStreamIter<R> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.inner.next() {
            None => {
                self.finished = true;
                None
            }
            Some(Ok(batch)) => Some(Ok(batch)),
            Some(Err(arrow::error::ArrowError::IoError(_, ref io_err)))
                if io_err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                tracing::warn!(
                    path = %self.label,
                    error = %io_err,
                    "Truncated trailing WAL IPC frame at EOF after ungraceful shutdown; preserving prior complete batches"
                );
                self.finished = true;
                None
            }
            Some(Err(e)) => {
                self.finished = true;
                Some(Err(anyhow::Error::from(e)))
            }
        }
    }
}

fn open_wal_reader<R: std::io::Read>(
    label: &str,
    length: u64,
    inner_reader: R,
) -> Result<OpenedWalStream<R>> {
    if length == 0 {
        return Ok(OpenedWalStream::Empty);
    }

    let mut reader = BufReader::new(inner_reader);
    let prefix = reader.fill_buf()?;
    if prefix.is_empty() {
        return Ok(OpenedWalStream::Empty);
    }

    let Some(first4) = prefix.get(..4) else {
        if IPC_CONTINUATION_MARKER.starts_with(prefix) || prefix.iter().all(|&b| b == 0) {
            tracing::warn!(
                path = %label,
                length,
                "WAL segment has truncated IPC header (< 4 bytes); ignoring torn header"
            );
            return Ok(OpenedWalStream::Empty);
        }
        anyhow::bail!(
            "Corrupt WAL segment {}: invalid {}-byte header",
            label,
            prefix.len()
        );
    };

    if first4 != IPC_CONTINUATION_MARKER {
        if length <= 8 && prefix.iter().all(|&b| b == 0) {
            tracing::warn!(
                path = %label,
                length,
                "WAL segment contains only zeroed header bytes; ignoring empty preallocated header"
            );
            return Ok(OpenedWalStream::Empty);
        }
        anyhow::bail!(
            "Corrupt WAL segment {}: missing Arrow IPC continuation marker",
            label
        );
    }

    let Some(meta_slice) = prefix.get(4..8) else {
        tracing::warn!(
            path = %label,
            length,
            "WAL segment truncated mid-schema length (< 8 bytes); ignoring torn header"
        );
        return Ok(OpenedWalStream::Empty);
    };
    let meta_arr: [u8; 4] = meta_slice
        .try_into()
        .map_err(|_| anyhow::anyhow!("Invalid WAL metadata length slice"))?;
    let meta_len = i32::from_le_bytes(meta_arr);
    if meta_len == 0 {
        return Ok(OpenedWalStream::Empty);
    }
    if meta_len < 0 || (meta_len as u64) > MAX_WAL_SCHEMA_META_BYTES {
        anyhow::bail!(
            "Corrupt WAL segment {}: invalid schema metadata length {meta_len}",
            label
        );
    }

    match StreamReader::try_new(reader, None) {
        Ok(ipc_reader) => Ok(OpenedWalStream::Reader(Box::new(SafeIpcStreamIter {
            label: label.to_string(),
            inner: ipc_reader,
            finished: false,
        }))),
        Err(arrow::error::ArrowError::IoError(_, ref io_err))
            if io_err.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            tracing::warn!(
                path = %label,
                length,
                error = %io_err,
                "WAL segment schema header truncated at EOF; ignoring torn segment"
            );
            Ok(OpenedWalStream::Empty)
        }
        Err(e) => Err(anyhow::Error::from(e)),
    }
}

fn open_wal_segment(path: &Path) -> Result<OpenedWalSegment> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(
                path = %path.display(),
                "WAL segment vanished before replay (concurrently truncated); skipping"
            );
            return Ok(OpenedWalSegment::Vanished);
        }
        Err(e) => return Err(e.into()),
    };

    let length = match file.metadata() {
        Ok(metadata) => metadata.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(
                path = %path.display(),
                "WAL segment vanished before metadata read (concurrently truncated); skipping"
            );
            return Ok(OpenedWalSegment::Vanished);
        }
        Err(e) => return Err(e.into()),
    };

    let label = path.display().to_string();
    match open_wal_reader(&label, length, file)? {
        OpenedWalStream::Empty => Ok(OpenedWalSegment::Empty),
        OpenedWalStream::Reader(iter) => Ok(OpenedWalSegment::Reader(iter)),
    }
}

/// Replay raw WAL segment bytes in-memory without touching the filesystem.
///
/// Exposed for libFuzzer coverage of header validation, torn-EOF recovery, and
/// `benostream:tx_id` metadata parsing; never part of the stable public API.
#[doc(hidden)]
pub fn fuzz_replay_wal_bytes(data: &[u8]) -> Result<Vec<RecordBatch>> {
    match open_wal_reader(
        "<fuzz-wal-segment>",
        data.len() as u64,
        std::io::Cursor::new(data),
    )? {
        OpenedWalStream::Empty => Ok(Vec::new()),
        OpenedWalStream::Reader(iter) => {
            let mut batches = Vec::new();
            for item in iter {
                let batch = item?;
                let _ = extract_wal_tx(&batch);
                batches.push(batch);
            }
            Ok(batches)
        }
    }
}

/// Flush the WAL writer's buffered bytes to stable storage.
///
/// Prefers `fdatasync` (`File::sync_data`); if that fails, falls back to a full
/// `fsync` (`File::sync_all`). Returns `Err` only when *both* fail — the caller
/// must then acknowledge the write as **failed**, never as durable, so a failed
/// fsync can never be reported as a successful commit.
fn sync_wal(writer: &mut StreamWriter<File>) -> Result<()> {
    match writer.get_ref().sync_data() {
        Ok(()) => Ok(()),
        Err(e) => {
            tracing::warn!("WAL sync_data failed ({e}); falling back to sync_all");
            writer
                .get_ref()
                .sync_all()
                .map_err(|e2| anyhow::anyhow!("WAL fsync failed — sync_data: {e}; sync_all: {e2}"))
        }
    }
}

/// Write-Ahead Log for durability of in-memory writes.
/// Uses Arrow IPC Stream format for append-only logging.
/// Supports multiple processes by using unique log files in a shared directory.
pub struct WriteAheadLog {
    dir: PathBuf,
    path: PathBuf,
    writer: Option<StreamWriter<File>>,
    schema: Option<SchemaRef>,
    tx: Option<mpsc::Sender<LogOp>>,
    config: WalConfig,
    segments: Arc<parking_lot::Mutex<WalSegmentRegistry>>,
}

#[derive(Debug, Clone)]
pub struct WalConfig {
    pub compact_threshold_mb: u64,
    pub sync_batch_size: usize,
    pub sync_interval_ms: u64,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            compact_threshold_mb: 1024,
            sync_batch_size: 10,
            sync_interval_ms: 100,
        }
    }
}

impl WalConfig {
    pub fn from_env() -> Self {
        Self {
            compact_threshold_mb: std::env::var("BSDB_WAL_COMPACT_MB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1024),
            sync_batch_size: std::env::var("BSDB_WAL_SYNC_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            sync_interval_ms: std::env::var("BSDB_WAL_SYNC_INTERVAL_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecordHeader {
    pub tx_id: uuid::Uuid,
    pub sequence_number: u64,
}

/// Tag an Arrow RecordBatch with transaction identity metadata for the WAL.
pub fn tag_batch_with_wal_tx(
    batch: &RecordBatch,
    tx_id: uuid::Uuid,
    seq: u64,
) -> Result<RecordBatch> {
    let mut metadata = batch.schema().metadata().clone();
    metadata.insert("benostream:tx_id".to_string(), tx_id.to_string());
    metadata.insert("benostream:seq".to_string(), seq.to_string());
    let schema = Arc::new(batch.schema().as_ref().clone().with_metadata(metadata));
    RecordBatch::try_new(schema, batch.columns().to_vec()).map_err(Into::into)
}

/// Extract transaction identity metadata from an Arrow RecordBatch if present.
pub fn extract_wal_tx(batch: &RecordBatch) -> Option<WalRecordHeader> {
    let schema = batch.schema();
    let meta = schema.metadata();
    let tx_id_str = meta.get("benostream:tx_id")?;
    let seq_str = meta.get("benostream:seq")?;
    let tx_id = uuid::Uuid::parse_str(tx_id_str).ok()?;
    let sequence_number = seq_str.parse::<u64>().ok()?;
    Some(WalRecordHeader {
        tx_id,
        sequence_number,
    })
}

enum LogOp {
    Append(RecordBatch, oneshot::Sender<Result<()>>),
    AppendSync(RecordBatch, oneshot::Sender<Result<()>>),
    Flush(oneshot::Sender<Result<()>>),
    Rotate(oneshot::Sender<Result<()>>),
    TruncateAll(oneshot::Sender<Result<()>>),
    Compact(oneshot::Sender<Result<()>>),
}

fn seal_active_writer(
    writer_opt: &mut Option<StreamWriter<File>>,
    current_schema: &mut Option<SchemaRef>,
    segments: &Arc<parking_lot::Mutex<WalSegmentRegistry>>,
) {
    if let Some(mut old_w) = writer_opt.take() {
        let _ = old_w.finish();
        let _ = old_w.get_ref().sync_data();
    }
    *current_schema = None;
    let mut reg = segments.lock();
    if let Some(active) = reg.active.take() {
        reg.sealed.push(active);
    }
}

fn sync_and_notify_pending(
    writer_opt: &mut Option<StreamWriter<File>>,
    pending_syncs: &mut Vec<oneshot::Sender<Result<()>>>,
) -> Result<()> {
    match writer_opt.as_mut().map(sync_wal).unwrap_or(Ok(())) {
        Ok(()) => {
            for tx in pending_syncs.drain(..) {
                let _ = tx.send(Ok(()));
            }
            Ok(())
        }
        Err(e) => {
            let msg = e.to_string();
            for tx in pending_syncs.drain(..) {
                let _ = tx.send(Err(anyhow::anyhow!("{msg}")));
            }
            Err(anyhow::anyhow!("{msg}"))
        }
    }
}

fn ensure_worker_writer<'a>(
    dir: &Path,
    batch_schema: &SchemaRef,
    writer_opt: &'a mut Option<StreamWriter<File>>,
    current_schema: &mut Option<SchemaRef>,
    pending_syncs: &mut Vec<oneshot::Sender<Result<()>>>,
    batch_count: &mut usize,
    segments: &Arc<parking_lot::Mutex<WalSegmentRegistry>>,
) -> Result<&'a mut StreamWriter<File>> {
    let schema_changed = match current_schema {
        None => true,
        Some(curr) => curr.as_ref() != batch_schema.as_ref(),
    };

    if schema_changed || writer_opt.is_none() {
        if writer_opt.is_some() {
            let sync_res = sync_and_notify_pending(writer_opt, pending_syncs);
            *batch_count = 0;
            seal_active_writer(writer_opt, current_schema, segments);
            sync_res?;
        }
        let target_path = next_wal_filename(dir);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target_path)
            .map_err(|e| anyhow::anyhow!("Failed to open WAL: {}", e))?;
        let writer = StreamWriter::try_new(file, batch_schema)
            .map_err(|e| anyhow::anyhow!("Failed to create WAL writer: {}", e))?;
        *current_schema = Some(batch_schema.clone());
        {
            let mut reg = segments.lock();
            reg.active = Some(WalSegmentEntry {
                path: target_path,
                txs: HashSet::new(),
                has_untagged: false,
            });
        }
        *writer_opt = Some(writer);
    }

    writer_opt
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("WAL writer unexpectedly absent after initialization"))
}

fn record_batch_tx(batch: &RecordBatch, segments: &Arc<parking_lot::Mutex<WalSegmentRegistry>>) {
    let mut reg = segments.lock();
    if let Some(active) = reg.active.as_mut() {
        if let Some(hdr) = extract_wal_tx(batch) {
            active.txs.insert(hdr.tx_id);
        } else {
            active.has_untagged = true;
        }
    }
}

fn remove_all_wal_files(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(())
}

fn compact_wal_directory(
    dir: &Path,
    segments: &Arc<parking_lot::Mutex<WalSegmentRegistry>>,
) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }

    let mut wal_files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
            wal_files.push(path);
        }
    }
    wal_files.sort();

    if wal_files.is_empty() {
        return Ok(());
    }

    let mut all_batches = Vec::new();
    for path in &wal_files {
        match open_wal_segment(path)? {
            OpenedWalSegment::Vanished | OpenedWalSegment::Empty => continue,
            OpenedWalSegment::Reader(ipc_reader) => {
                for b in ipc_reader {
                    all_batches.push(b?);
                }
            }
        }
    }

    if all_batches.len() <= 1 && wal_files.len() <= 1 {
        return Ok(());
    }
    if all_batches.is_empty() {
        for path in &wal_files {
            let _ = std::fs::remove_file(path);
        }
        let mut reg = segments.lock();
        reg.sealed.retain(|e| !wal_files.contains(&e.path));
        return Ok(());
    }

    // Group batches sharing an exact schema (including WAL tx metadata) so compaction
    // consolidates batches without ever stripping or mixing per-tx `benostream:tx_id` tags.
    let mut new_entries: Vec<(PathBuf, RecordBatch)> = Vec::new();
    let mut grouped: Vec<(SchemaRef, Vec<RecordBatch>)> = Vec::new();
    for batch in all_batches {
        let schema = batch.schema();
        if let Some((_, group)) = grouped
            .iter_mut()
            .find(|(s, _)| s.as_ref() == schema.as_ref())
        {
            group.push(batch);
        } else {
            grouped.push((schema, vec![batch]));
        }
    }

    let single_schema_multiple_batches = grouped.len() == 1 && grouped[0].1.len() > 1;
    let has_multi_batch_group = grouped.iter().any(|(_, g)| g.len() > 1);
    if !has_multi_batch_group && wal_files.len() <= grouped.len() {
        return Ok(());
    }

    for (schema, group) in grouped {
        let consolidated = if group.len() == 1 {
            group
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("empty group"))?
        } else {
            arrow::compute::concat_batches(&schema, &group)
                .context("Failed to concatenate batches during WAL compaction")?
        };
        let target_path = next_wal_filename(dir);
        let temp_path = target_path.with_extension("arrow.tmp");
        {
            let temp_file = File::create(&temp_path).context("Failed to create temp WAL file")?;
            let mut temp_writer = StreamWriter::try_new(temp_file, &schema)?;
            temp_writer.write(&consolidated)?;
            temp_writer.finish()?;
            temp_writer
                .into_inner()
                .map_err(|e| anyhow::anyhow!("WAL compaction flush error: {e}"))?
                .sync_all()?;
        }
        std::fs::rename(&temp_path, &target_path).context("Failed to replace WAL file")?;
        new_entries.push((target_path, consolidated));
    }

    for stale_path in &wal_files {
        match std::fs::remove_file(stale_path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }

    {
        let mut reg = segments.lock();
        reg.sealed.retain(|e| !wal_files.contains(&e.path));
        for (path, batch) in &new_entries {
            let mut txs = HashSet::new();
            let mut has_untagged = false;
            if let Some(hdr) = extract_wal_tx(batch) {
                txs.insert(hdr.tx_id);
            } else {
                has_untagged = true;
            }
            reg.sealed.push(WalSegmentEntry {
                path: path.clone(),
                txs,
                has_untagged,
            });
        }
    }

    if single_schema_multiple_batches {
        tracing::info!("WAL: Compacted multiple batches into 1 batch");
    } else {
        tracing::info!(
            "WAL: Compacted {} file(s) into {} file(s)",
            wal_files.len(),
            new_entries.len()
        );
    }

    Ok(())
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
        let path = next_wal_filename(&dir);

        Self {
            dir,
            path,
            writer: None,
            schema: None,
            tx: None,
            config: WalConfig::default(),
            segments: Arc::new(parking_lot::Mutex::new(WalSegmentRegistry::default())),
        }
    }

    pub fn with_config(mut self, config: WalConfig) -> Self {
        self.config = config;
        self
    }

    /// Register a WAL segment recovered on startup so watermark pruning can delete it
    /// once all its transactions (including assigned synthetic UUIDs for legacy untagged
    /// records) have committed to the manifest.
    pub fn register_recovered_segment(&self, path: PathBuf, txs: HashSet<uuid::Uuid>) {
        let mut reg = self.segments.lock();
        if let Some(existing) = reg.sealed.iter_mut().find(|e| e.path == path) {
            existing.txs.extend(txs);
        } else {
            reg.sealed.push(WalSegmentEntry {
                path,
                txs,
                has_untagged: false,
            });
        }
    }

    /// Prune sealed WAL segments whose transaction sets are completely contained in
    /// `committed`. Active in-progress segments and sealed segments containing uncommitted
    /// or untagged writes are never deleted.
    pub fn prune_committed_segments(
        &self,
        committed: &HashSet<uuid::Uuid>,
    ) -> Result<Vec<PathBuf>> {
        let candidates: Vec<WalSegmentEntry> = {
            let mut reg = self.segments.lock();
            let mut keep = Vec::with_capacity(reg.sealed.len());
            let mut prune = Vec::new();
            for entry in reg.sealed.drain(..) {
                if !entry.has_untagged && entry.txs.is_subset(committed) {
                    prune.push(entry);
                } else {
                    keep.push(entry);
                }
            }
            reg.sealed = keep;
            prune
        };

        let mut removed = Vec::new();
        let mut failed = Vec::new();
        let mut first_err: Option<anyhow::Error> = None;

        for entry in candidates {
            match std::fs::remove_file(&entry.path) {
                Ok(()) => removed.push(entry.path),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    removed.push(entry.path);
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(anyhow::Error::from(e));
                    }
                    failed.push(entry);
                }
            }
        }

        if !failed.is_empty() {
            let mut reg = self.segments.lock();
            reg.sealed.extend(failed);
        }

        if let Some(err) = first_err {
            return Err(err);
        }

        Ok(removed)
    }

    /// Start a background worker for asynchronous writes.
    pub fn spawn_worker(&mut self) -> Result<()> {
        if self.tx.is_some() {
            return Ok(());
        }

        let (tx, mut rx) = mpsc::channel::<LogOp>(1024);
        self.tx = Some(tx);

        // Move state into worker
        let dir = self.dir.clone();
        let mut writer_opt: Option<StreamWriter<File>> = self.writer.take();
        let config = self.config.clone();
        let mut current_schema: Option<SchemaRef> = self.schema.take();
        let segments = self.segments.clone();

        tokio::spawn(async move {
            let mut pending_syncs: Vec<oneshot::Sender<Result<()>>> = Vec::new();
            let mut batch_count: usize = 0;

            let sync_batch_size = config.sync_batch_size;
            let sync_interval_ms = config.sync_interval_ms;

            loop {
                let timeout =
                    tokio::time::sleep(std::time::Duration::from_millis(sync_interval_ms));

                tokio::select! {
                    msg = rx.recv() => {
                        match msg {
                            Some(LogOp::Append(batch, reply_tx)) => {
                                let writer = match ensure_worker_writer(
                                    &dir,
                                    &batch.schema(),
                                    &mut writer_opt,
                                    &mut current_schema,
                                    &mut pending_syncs,
                                    &mut batch_count,
                                    &segments,
                                ) {
                                    Ok(w) => w,
                                    Err(e) => {
                                        let _ = reply_tx.send(Err(e));
                                        continue;
                                    }
                                };

                                let t_wal_write = std::time::Instant::now();
                                let rows = batch.num_rows();
                                if let Err(e) = writer.write(&batch) {
                                    let _ = reply_tx.send(Err(anyhow::anyhow!("WAL write failed: {}", e)));
                                } else {
                                    record_batch_tx(&batch, &segments);
                                    tracing::debug!(
                                        rows,
                                        wal_writer_write_ms = t_wal_write.elapsed().as_millis(),
                                        "wal_writer write timing"
                                    );
                                    pending_syncs.push(reply_tx);
                                    batch_count += 1;

                                    if batch_count >= sync_batch_size {
                                        let _ = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                                        batch_count = 0;
                                    }
                                }
                            }
                            Some(LogOp::AppendSync(batch, reply_tx)) => {
                                let writer = match ensure_worker_writer(
                                    &dir,
                                    &batch.schema(),
                                    &mut writer_opt,
                                    &mut current_schema,
                                    &mut pending_syncs,
                                    &mut batch_count,
                                    &segments,
                                ) {
                                    Ok(w) => w,
                                    Err(e) => {
                                        let _ = reply_tx.send(Err(e));
                                        continue;
                                    }
                                };

                                if let Err(e) = writer.write(&batch) {
                                    let _ = reply_tx.send(Err(anyhow::anyhow!("WAL write failed: {}", e)));
                                } else {
                                    record_batch_tx(&batch, &segments);
                                    pending_syncs.push(reply_tx);
                                    let _ = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                                    batch_count = 0;
                                }
                            }
                            Some(LogOp::Flush(reply_tx)) => {
                                let res = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                                batch_count = 0;
                                let _ = reply_tx.send(res);
                            }
                            Some(LogOp::Rotate(reply_tx)) => {
                                let sync_res = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                                batch_count = 0;
                                seal_active_writer(&mut writer_opt, &mut current_schema, &segments);
                                let _ = reply_tx.send(sync_res);
                            }
                            Some(LogOp::TruncateAll(reply_tx)) => {
                                seal_active_writer(&mut writer_opt, &mut current_schema, &segments);
                                for tx in pending_syncs.drain(..) {
                                    let _ = tx.send(Ok(()));
                                }
                                batch_count = 0;
                                {
                                    let mut reg = segments.lock();
                                    reg.active = None;
                                    reg.sealed.clear();
                                }
                                let _ = reply_tx.send(remove_all_wal_files(&dir));
                            }
                            Some(LogOp::Compact(reply_tx)) => {
                                let sync_res = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                                batch_count = 0;
                                seal_active_writer(&mut writer_opt, &mut current_schema, &segments);
                                let res = match sync_res {
                                    Ok(()) => compact_wal_directory(&dir, &segments),
                                    Err(e) => Err(e),
                                };
                                let _ = reply_tx.send(res);
                            }
                            None => break,
                        }
                    }
                    _ = timeout => {
                        if !pending_syncs.is_empty() {
                            let _ = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                            batch_count = 0;
                        }
                    }
                }
            }

            // Final cleanup when channel closes — drain any remaining queued messages.
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    LogOp::Append(batch, reply_tx) | LogOp::AppendSync(batch, reply_tx) => {
                        match ensure_worker_writer(
                            &dir,
                            &batch.schema(),
                            &mut writer_opt,
                            &mut current_schema,
                            &mut pending_syncs,
                            &mut batch_count,
                            &segments,
                        ) {
                            Ok(writer) => {
                                if let Err(e) = writer.write(&batch) {
                                    let _ = reply_tx
                                        .send(Err(anyhow::anyhow!("WAL write failed: {}", e)));
                                } else {
                                    record_batch_tx(&batch, &segments);
                                    pending_syncs.push(reply_tx);
                                }
                            }
                            Err(e) => {
                                let _ = reply_tx.send(Err(e));
                            }
                        }
                    }
                    LogOp::Flush(reply_tx) => {
                        let res = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                        let _ = reply_tx.send(res);
                    }
                    LogOp::Rotate(reply_tx) => {
                        let res = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                        seal_active_writer(&mut writer_opt, &mut current_schema, &segments);
                        let _ = reply_tx.send(res);
                    }
                    LogOp::TruncateAll(reply_tx) => {
                        seal_active_writer(&mut writer_opt, &mut current_schema, &segments);
                        for tx in pending_syncs.drain(..) {
                            let _ = tx.send(Ok(()));
                        }
                        let mut reg = segments.lock();
                        reg.active = None;
                        reg.sealed.clear();
                        drop(reg);
                        let _ = reply_tx.send(remove_all_wal_files(&dir));
                    }
                    LogOp::Compact(reply_tx) => {
                        let sync_res = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
                        seal_active_writer(&mut writer_opt, &mut current_schema, &segments);
                        let res = match sync_res {
                            Ok(()) => compact_wal_directory(&dir, &segments),
                            Err(e) => Err(e),
                        };
                        let _ = reply_tx.send(res);
                    }
                }
            }

            let _ = sync_and_notify_pending(&mut writer_opt, &mut pending_syncs);
            if let Some(mut writer) = writer_opt {
                let _ = writer.finish();
                if let Err(e) = writer.get_ref().sync_all() {
                    tracing::error!("Final WAL sync failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Append a batch to the log asynchronously.
    pub async fn append_async(&self, batch: RecordBatch) -> Result<()> {
        if let Some(tx) = &self.tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(LogOp::Append(batch, reply_tx))
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker dropped request"))?
        } else {
            // Fallback to sync? Or error?
            anyhow::bail!("WAL worker not started. Call spawn_worker() first.");
        }
    }

    /// Append a batch to the WAL without waiting for the fsync reply.
    /// Returns as soon as the batch is handed off to the WAL worker channel.
    /// The worker still persists and syncs on its own schedule — durability is maintained
    /// but the write path is not blocked by disk I/O latency.
    pub async fn append_fire_and_forget(&self, batch: RecordBatch) -> Result<()> {
        if let Some(tx) = &self.tx {
            // We still create a reply channel so the WAL worker can function normally,
            // but we intentionally drop reply_rx — the worker reply will silently fail,
            // which is fine because we don't need to wait for it.
            let (reply_tx, _reply_rx) = oneshot::channel();
            tx.send(LogOp::Append(batch, reply_tx))
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;
            Ok(())
        } else {
            anyhow::bail!("WAL worker not started. Call spawn_worker() first.");
        }
    }

    /// Append a batch to the WAL and immediately sync to disk before returning.
    /// This provides strict synchronous durability guarantees.
    pub async fn append_sync(&self, batch: RecordBatch) -> Result<()> {
        if let Some(tx) = &self.tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(LogOp::AppendSync(batch, reply_tx))
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker dropped request"))?
        } else {
            anyhow::bail!("WAL worker not started. Call spawn_worker() first.");
        }
    }

    /// Explicitly flush and sync all pending WAL writes to durable storage.
    pub async fn flush_async(&self) -> Result<()> {
        if let Some(tx) = &self.tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(LogOp::Flush(reply_tx))
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker dropped request"))?
        } else {
            Ok(())
        }
    }

    /// Flush, finish, and seal the active WAL writer segment so subsequent
    /// writes open a new segment file and the sealed segment can be safely
    /// watermark-pruned after commit without dropping concurrent writes.
    pub async fn rotate_async(&self) -> Result<()> {
        if let Some(tx) = &self.tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(LogOp::Rotate(reply_tx))
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker dropped request"))?
        } else {
            Ok(())
        }
    }

    /// Synchronize with the worker to seal and remove every WAL segment file
    /// (used on full table truncate).
    pub async fn truncate_all_async(&self) -> Result<()> {
        if let Some(tx) = &self.tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(LogOp::TruncateAll(reply_tx))
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker dropped request"))?
        } else {
            {
                let mut reg = self.segments.lock();
                reg.active = None;
                reg.sealed.clear();
            }
            remove_all_wal_files(&self.dir)
        }
    }

    /// Consolidate WAL segments through the owning worker so active writer fds
    /// are closed cleanly prior to compaction.
    pub async fn compact_async(&self) -> Result<()> {
        if let Some(tx) = &self.tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(LogOp::Compact(reply_tx))
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker channel closed"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("WAL worker dropped request"))?
        } else {
            compact_wal_directory(&self.dir, &self.segments)
        }
    }

    /// Replay all log files in the WAL directory, preserving per-segment paths and batches.
    pub fn replay_with_segments(&self) -> Result<(Vec<RecordBatch>, Vec<WalRecoveredSegment>)> {
        if !self.dir.exists() {
            return Ok((Vec::new(), Vec::new()));
        }

        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Vec::new(), Vec::new()));
            }
            Err(e) => return Err(e.into()),
        };

        let mut wal_files = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
                wal_files.push(path);
            }
        }
        wal_files.sort();

        let mut all_batches = Vec::new();
        let mut segments = Vec::with_capacity(wal_files.len());

        for path in wal_files {
            match open_wal_segment(&path)? {
                OpenedWalSegment::Vanished => continue,
                OpenedWalSegment::Empty => {
                    segments.push(WalRecoveredSegment {
                        path,
                        batches: Vec::new(),
                    });
                }
                OpenedWalSegment::Reader(ipc_reader) => {
                    let mut seg_batches = Vec::new();
                    for item in ipc_reader {
                        let b = item?;
                        all_batches.push(b.clone());
                        seg_batches.push(b);
                    }
                    segments.push(WalRecoveredSegment {
                        path,
                        batches: seg_batches,
                    });
                }
            }
        }

        Ok((all_batches, segments))
    }

    /// Replay all log files in the WAL directory and return an iterator of batches.
    /// This should be used on startup for memory-efficient recovery.
    pub fn replay_stream(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>>>> {
        if !self.dir.exists() {
            return Ok(Box::new(std::iter::empty()));
        }

        // 1. List all .arrow files in the directory
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Box::new(std::iter::empty()));
            }
            Err(e) => return Err(e.into()),
        };
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
            // A concurrent commit truncates the log after applying it, so a
            // segment can disappear between the directory scan above and this
            // open. That is a *completed* segment, not corruption — reporting it
            // as corruption told the caller to intervene manually in a scenario
            // the multi-process tests exercise routinely.
            match open_wal_segment(&path)? {
                OpenedWalSegment::Vanished | OpenedWalSegment::Empty => continue,
                OpenedWalSegment::Reader(ipc_reader) => {
                    all_iterators.push(*ipc_reader);
                }
            }
        }

        Ok(Box::new(all_iterators.into_iter().flatten()))
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
            // Same race as `replay_stream`: a concurrent truncate can remove the
            // directory between the existence check and the scan.
            let entries = match std::fs::read_dir(&self.dir) {
                Ok(entries) => Some(entries),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(e.into()),
            };
            for entry in entries.into_iter().flatten() {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
                    paths.push(path.to_str().context("Invalid UTF-8 in path")?.to_string());
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

    /// Clear the log files owned by this WAL directory.
    /// Should be called after data is successfully persisted (flushed) to main storage.
    pub fn truncate(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.finish();
        }
        self.schema = None;
        {
            let mut reg = self.segments.lock();
            reg.active = None;
            reg.sealed.clear();
        }
        remove_all_wal_files(&self.dir)
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

    /// Check if WAL should be compacted based on total size of all log files in the directory.
    /// Previous implementation only checked the current instance's file, missing stale logs
    /// from earlier instances that accumulate after crashes.
    pub fn should_compact(&self) -> Result<bool> {
        if !self.dir.exists() {
            return Ok(false);
        }

        let mut total_bytes: u64 = 0;
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("arrow") {
                total_bytes += entry.metadata()?.len();
            }
        }

        let threshold_bytes = self.config.compact_threshold_mb * 1024 * 1024;
        Ok(total_bytes > threshold_bytes)
    }

    /// Compact WAL by consolidating all batches into one (synchronous mode).
    pub fn compact(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.finish();
            let _ = writer.get_ref().sync_data();
        }
        self.schema = None;

        // 1. Replay all batches
        let (batches, _) = self.replay()?;
        if batches.is_empty() || batches.len() == 1 {
            return Ok(());
        }

        // 2. Concatenate batches
        let schema = batches[0].schema();
        let consolidated = arrow::compute::concat_batches(&schema, &batches)
            .context("Failed to concatenate batches during WAL compaction")?;

        // 3. Write to temp file
        let temp_path = self.path.with_extension("arrow.tmp");
        let temp_file = File::create(&temp_path).context("Failed to create temp WAL file")?;
        let mut temp_writer = StreamWriter::try_new(temp_file, &schema)?;
        temp_writer.write(&consolidated)?;
        temp_writer.finish()?;
        drop(temp_writer);

        // 4. Atomic replace
        std::fs::rename(&temp_path, &self.path).context("Failed to replace WAL file")?;

        // 5. Delete stale .arrow files from previous WAL instances.
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let stale_path = entry.path();
            if stale_path.is_file()
                && stale_path.extension().and_then(|s| s.to_str()) == Some("arrow")
                && stale_path != self.path
            {
                tracing::debug!("WAL: Removing stale log file: {:?}", stale_path);
                std::fs::remove_file(&stale_path)?;
            }
        }

        self.writer = None;

        tracing::info!("WAL: Compacted {} batches into 1 batch", batches.len());
        Ok(())
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        // Only finish if we have a direct writer (synchronous mode)
        // If tx is Some, the worker handles the finish.
        if self.tx.is_none() {
            if let Some(mut writer) = self.writer.take() {
                let _ = writer.finish();
                if let Ok(file) = writer.into_inner() {
                    let _ = file.sync_all();
                }
            }
        }
    }
}

/// A StreamingBuffer aggregates a continuous stream of row batches and periodically flushes them
/// via the Table's WAL and commits to Iceberg snapshots based on a size threshold or time interval.
pub struct StreamingBuffer {
    tx: mpsc::Sender<RecordBatch>,
}

impl StreamingBuffer {
    pub fn new(
        table: Arc<crate::core::table::Table>,
        batch_size_threshold: usize,
        interval_ms: u64,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<RecordBatch>(1024);

        tokio::spawn(async move {
            let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));
            let mut buffered_rows = 0;

            loop {
                tokio::select! {
                    Some(batch) = rx.recv() => {
                        let rows = batch.num_rows();
                        if let Err(e) = table.write_async(vec![batch]).await {
                            tracing::error!("StreamingBuffer write error: {}", e);
                        }
                        buffered_rows += rows;

                        if buffered_rows >= batch_size_threshold {
                            if let Err(e) = table.commit_async().await {
                                tracing::error!("StreamingBuffer commit error: {}", e);
                            }
                            buffered_rows = 0;
                        }
                    }
                    _ = tick.tick() => {
                        if buffered_rows > 0 {
                            if let Err(e) = table.commit_async().await {
                                tracing::error!("StreamingBuffer commit error: {}", e);
                            }
                            buffered_rows = 0;
                        }
                    }
                }
            }
        });

        Self { tx }
    }

    pub async fn insert(&self, batch: RecordBatch) -> Result<()> {
        self.tx
            .send(batch)
            .await
            .map_err(|_| anyhow::anyhow!("StreamingBuffer channel closed"))
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
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
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

        let config = WalConfig {
            compact_threshold_mb: 100,
            ..Default::default()
        };
        let mut wal = WriteAheadLog::new(&wal_dir).with_config(config);

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
        let schema1 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch1 =
            RecordBatch::try_new(schema1, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])?;
        wal.append(&batch1)?;

        // Truncate and start fresh
        wal.truncate()?;

        // Write batch with different schema
        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch2 =
            RecordBatch::try_new(schema2, vec![Arc::new(Int32Array::from(vec![10, 20, 30]))])?;
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

    #[test]
    fn test_wal_truncated_eof_tail_preserves_complete_prefix() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("torn_tail_wal");
        std::fs::create_dir_all(&wal_dir)?;

        let batch1 = create_test_batch(0, 10);
        let batch2 = create_test_batch(10, 50);

        let mut one_batch_buf = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut one_batch_buf, &batch1.schema())?;
            w.write(&batch1)?;
        }
        let one_batch_len = one_batch_buf.len();

        let mut two_batch_buf = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut two_batch_buf, &batch1.schema())?;
            w.write(&batch1)?;
            w.write(&batch2)?;
        }
        assert!(two_batch_buf.len() > one_batch_len + 32);

        // Truncate inside the second frame across continuation, meta_len, flatbuffer, and body cuts.
        for cut_offset in [
            one_batch_len + 2,
            one_batch_len + 6,
            one_batch_len + 24,
            two_batch_buf.len() - 4,
        ] {
            let sub_dir = wal_dir.join(format!("cut_{cut_offset}"));
            std::fs::create_dir_all(&sub_dir)?;
            let seg_path = sub_dir.join("log_0001.arrow");
            std::fs::write(&seg_path, &two_batch_buf[..cut_offset])?;

            let wal = WriteAheadLog::new(&sub_dir);
            let (batches, segs) = wal.replay_with_segments()?;
            assert_eq!(
                batches.len(),
                1,
                "cut_offset={cut_offset} must preserve batch1"
            );
            assert_eq!(batches[0].num_rows(), 10);
            assert_eq!(segs.len(), 1);
            assert_eq!(segs[0].batches.len(), 1);

            let (stream_batches, _) = wal.replay()?;
            assert_eq!(stream_batches.len(), 1);
            assert_eq!(stream_batches[0].num_rows(), 10);

            let dummy_reg = Arc::new(parking_lot::Mutex::new(WalSegmentRegistry::default()));
            compact_wal_directory(&sub_dir, &dummy_reg)?;
        }

        Ok(())
    }

    #[test]
    fn test_wal_truncated_schema_header_yields_empty() -> Result<()> {
        let temp_dir = tempdir()?;
        let wal_dir = temp_dir.path().join("torn_header_wal");
        std::fs::create_dir_all(&wal_dir)?;

        let batch = create_test_batch(0, 5);
        let mut full = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut full, &batch.schema())?;
            w.write(&batch)?;
        }

        for header_cut in [0usize, 3, 6, 20] {
            let sub_dir = wal_dir.join(format!("hdr_{header_cut}"));
            std::fs::create_dir_all(&sub_dir)?;
            std::fs::write(sub_dir.join("log_0001.arrow"), &full[..header_cut])?;

            let wal = WriteAheadLog::new(&sub_dir);
            let (batches, segs) = wal.replay_with_segments()?;
            assert!(batches.is_empty(), "header_cut={header_cut}");
            assert_eq!(segs.len(), 1);
            assert!(segs[0].batches.is_empty());

            let (stream_batches, _) = wal.replay()?;
            assert!(stream_batches.is_empty());
        }

        Ok(())
    }

    #[test]
    fn test_wal_corrupt_middle_or_header_errors() -> Result<()> {
        // Case 1: garbage non-IPC header fails instead of being treated as torn EOF.
        let dir1 = tempdir()?;
        std::fs::write(dir1.path().join("log_0001.arrow"), b"CORRUPT_HEADER_BYTES")?;
        let wal1 = WriteAheadLog::new(dir1.path());
        assert!(wal1.replay_with_segments().is_err());
        assert!(wal1.replay().is_err());

        // Case 2: corrupted middle flatbuffer bytes inside batch 1 must fail.
        let dir2 = tempdir()?;
        let batch1 = create_test_batch(0, 10);
        let batch2 = create_test_batch(10, 10);

        let mut header_only = Vec::new();
        {
            let _w = StreamWriter::try_new(&mut header_only, &batch1.schema())?;
        }
        let header_len = header_only.len();

        let mut two_batches = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut two_batches, &batch1.schema())?;
            w.write(&batch1)?;
            w.write(&batch2)?;
            w.finish()?;
        }
        // Overwrite batch 1's flatbuffer bytes right after its [CONTINUATION(4), meta_len(4)].
        let flatbuf_start = header_len + 8;
        for byte in &mut two_batches[flatbuf_start..flatbuf_start + 16] {
            *byte = 0xAA;
        }
        std::fs::write(dir2.path().join("log_0001.arrow"), &two_batches)?;

        let wal2 = WriteAheadLog::new(dir2.path());
        assert!(wal2.replay_with_segments().is_err());
        assert!(wal2.replay().is_err());

        Ok(())
    }

    #[test]
    fn test_fuzz_replay_wal_bytes_handles_edge_cases() -> Result<()> {
        // Empty, truncated continuation/zero headers, and zeroed preallocated <=8B frames.
        assert!(fuzz_replay_wal_bytes(&[])?.is_empty());
        assert!(fuzz_replay_wal_bytes(&[0xFF, 0xFF])?.is_empty());
        assert!(fuzz_replay_wal_bytes(&[0x00, 0x00])?.is_empty());
        assert!(fuzz_replay_wal_bytes(&[0xFF, 0xFF, 0xFF, 0xFF, 0x10])?.is_empty());
        assert!(fuzz_replay_wal_bytes(&[0x00; 8])?.is_empty());
        // Zero schema metadata length (EOS marker immediately after continuation)
        assert!(fuzz_replay_wal_bytes(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0])?.is_empty());

        // Reject invalid continuation marker, negative metadata length, and oversized metadata (>64 MiB).
        assert!(fuzz_replay_wal_bytes(b"NOT_ARROW_IPC").is_err());
        assert!(fuzz_replay_wal_bytes(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]).is_err());
        let oversized = (65u32 * 1024 * 1024).to_le_bytes();
        assert!(fuzz_replay_wal_bytes(&[
            0xFF,
            0xFF,
            0xFF,
            0xFF,
            oversized[0],
            oversized[1],
            oversized[2],
            oversized[3]
        ])
        .is_err());

        // Valid multi-batch WAL stream with tx tags + truncated EOF tail preserves prefix batch cleanly.
        let tx_id = uuid::Uuid::new_v4();
        let b1 = tag_batch_with_wal_tx(&create_test_batch(0, 5), tx_id, 1)?;
        let b2 = tag_batch_with_wal_tx(&create_test_batch(5, 7), tx_id, 1)?;

        let mut one = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut one, &b1.schema())?;
            w.write(&b1)?;
        }
        let mut two = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut two, &b1.schema())?;
            w.write(&b1)?;
            w.write(&b2)?;
            w.finish()?;
        }

        let full = fuzz_replay_wal_bytes(&two)?;
        assert_eq!(full.len(), 2);
        assert_eq!(extract_wal_tx(&full[0]).unwrap().tx_id, tx_id);
        assert_eq!(extract_wal_tx(&full[1]).unwrap().sequence_number, 1);

        let torn = fuzz_replay_wal_bytes(&two[..one.len() + 12])?;
        assert_eq!(torn.len(), 1);
        assert_eq!(torn[0].num_rows(), 5);

        Ok(())
    }
}
