// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Reactive lakehouse streaming: `Table::subscribe()`.
//!
//! A table owns a `tokio::sync::broadcast` channel. Every successful commit
//! publishes the committed [`RecordBatch`]es plus a [`TableEvent::Commit`]
//! marker, so an in-process consumer can react to new data without polling.
//!
//! [`Table::subscribe_filtered`] additionally evaluates a SQL predicate against
//! each incoming batch and only yields the rows that match — the "live query"
//! primitive. Filtering is a *superset-safe* post-filter: the predicate is
//! re-applied above the batch, so a subscriber never sees a row that fails it.

use std::sync::Arc;

use anyhow::{bail, Result};
use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use tokio::sync::broadcast;

use super::Table;

/// Capacity of the per-table broadcast channel. A slow subscriber that falls
/// further behind than this is told it lagged (and skips ahead) rather than
/// blocking the writer.
const SUBSCRIBER_CAPACITY: usize = 1024;

/// Create the per-table broadcast channel. Called once by the table builder;
/// clones share the returned sender.
pub(crate) fn new_subscriber_channel() -> Arc<broadcast::Sender<TableEvent>> {
    Arc::new(broadcast::channel(SUBSCRIBER_CAPACITY).0)
}

/// An event published on a table's subscription channel.
#[derive(Debug, Clone)]
pub enum TableEvent {
    /// A batch of rows that was just committed.
    Batch(RecordBatch),
    /// A commit completed; `rows` is the total number of rows in the commit.
    Commit { rows: usize },
}

/// A live subscription to a table's committed changes.
///
/// Created by [`Table::subscribe`] / [`Table::subscribe_filtered`]. Dropping it
/// unsubscribes; [`Subscription::close`] unsubscribes deterministically (so the
/// table stops counting this receiver and can skip publishing).
pub struct Subscription {
    rx: Option<broadcast::Receiver<TableEvent>>,
    filter: Option<String>,
}

impl Subscription {
    /// Unsubscribe now. Idempotent; further `recv`/`try_recv` calls error.
    pub fn close(&mut self) {
        self.rx = None;
    }

    /// Whether this subscription has been closed.
    pub fn is_closed(&self) -> bool {
        self.rx.is_none()
    }

    /// Await the next event, applying the subscription's predicate filter (if
    /// any) to `Batch` events. Batches that match no rows are skipped.
    pub async fn recv(&mut self) -> Result<TableEvent> {
        loop {
            let rx = match self.rx.as_mut() {
                Some(rx) => rx,
                None => bail!("subscription closed"),
            };
            match rx.recv().await {
                Ok(TableEvent::Batch(batch)) => match &self.filter {
                    None => return Ok(TableEvent::Batch(batch)),
                    Some(filter) => {
                        if let Some(filtered) = filter_batch(&batch, filter).await? {
                            return Ok(TableEvent::Batch(filtered));
                        }
                        // No rows matched — keep waiting.
                    }
                },
                // A filtered subscription is about rows: suppress the
                // commit markers (they carry no rows to test against).
                Ok(TableEvent::Commit { .. }) if self.filter.is_some() => continue,
                Ok(event) => return Ok(event),
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(
                        skipped = n,
                        "subscription lagged; skipped {n} event(s) to catch up"
                    );
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    bail!("subscription closed: the table was dropped")
                }
            }
        }
    }

    /// Non-blocking variant of [`Subscription::recv`]. Returns `Ok(None)` when
    /// no event is currently available.
    pub fn try_recv(&mut self) -> Result<Option<TableEvent>> {
        let rx = match self.rx.as_mut() {
            Some(rx) => rx,
            None => bail!("subscription closed"),
        };
        match rx.try_recv() {
            Ok(TableEvent::Batch(batch)) => match &self.filter {
                None => Ok(Some(TableEvent::Batch(batch))),
                Some(filter) => {
                    // `try_recv` cannot await the async filter; run it on the
                    // current thread's runtime via a blocking bridge.
                    let filtered = futures::executor::block_on(filter_batch(&batch, filter))?;
                    Ok(filtered.map(TableEvent::Batch))
                }
            },
            Ok(TableEvent::Commit { .. }) if self.filter.is_some() => Ok(None),
            Ok(event) => Ok(Some(event)),
            Err(broadcast::error::TryRecvError::Empty) => Ok(None),
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(skipped = n, "subscription lagged on try_recv");
                Ok(None)
            }
            Err(broadcast::error::TryRecvError::Closed) => {
                bail!("subscription closed: the table was dropped")
            }
        }
    }
}

impl Table {
    /// Subscribe to this table's committed changes.
    pub fn subscribe(&self) -> Subscription {
        Subscription {
            rx: Some(self.subscribers.subscribe()),
            filter: None,
        }
    }

    /// Subscribe to committed changes, yielding only rows matching `filter`
    /// (a SQL predicate, e.g. `"age > 30"`).
    pub fn subscribe_filtered(&self, filter: impl Into<String>) -> Subscription {
        Subscription {
            rx: Some(self.subscribers.subscribe()),
            filter: Some(filter.into()),
        }
    }

    /// Publish a successful commit to all subscribers. Cheap no-op when nobody
    /// is subscribed.
    pub(crate) fn publish_committed(&self, batches: &[RecordBatch]) {
        if self.subscribers.receiver_count() == 0 {
            return;
        }
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        for batch in batches {
            // A send error only means every receiver has gone away.
            let _ = self.subscribers.send(TableEvent::Batch(batch.clone()));
        }
        let _ = self.subscribers.send(TableEvent::Commit { rows });
    }
}

/// Evaluate a SQL predicate against a batch, returning the matching rows (or
/// `None` when nothing matches). The predicate is re-applied above the batch,
/// so the result is exact.
async fn filter_batch(batch: &RecordBatch, filter: &str) -> Result<Option<RecordBatch>> {
    if batch.num_rows() == 0 {
        return Ok(None);
    }
    let schema = batch.schema();
    let ctx = SessionContext::new();
    let mem = MemTable::try_new(schema.clone(), vec![vec![batch.clone()]])?;
    ctx.register_table("t", Arc::new(mem))?;
    let df = ctx
        .sql(&format!("SELECT * FROM t WHERE {filter}"))
        .await
        .map_err(|e| anyhow::anyhow!("invalid subscription filter '{filter}': {e}"))?;
    let batches = df.collect().await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    if total == 0 {
        return Ok(None);
    }
    Ok(Some(concat_batches(&schema, &batches)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch(values: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap()
    }

    #[tokio::test]
    async fn filter_batch_keeps_only_matching_rows() {
        let b = batch(&[1, 5, 10, 20]);
        let out = filter_batch(&b, "v >= 10").await.unwrap().unwrap();
        assert_eq!(out.num_rows(), 2);
        let col = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(col.value(0), 10);
        assert_eq!(col.value(1), 20);
    }

    #[tokio::test]
    async fn filter_batch_returns_none_when_nothing_matches() {
        let b = batch(&[1, 2, 3]);
        assert!(filter_batch(&b, "v > 100").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn filter_batch_rejects_invalid_predicate() {
        let b = batch(&[1, 2, 3]);
        assert!(filter_batch(&b, "this is not sql").await.is_err());
    }

    async fn temp_table(tag: &str) -> (Table, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("bsdb_sub_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let table = Table::create_async(dir.to_str().unwrap().to_string(), schema)
            .await
            .unwrap();
        (table, dir)
    }

    #[tokio::test]
    async fn close_unsubscribes_and_errors_afterwards() {
        let (table, dir) = temp_table("close").await;
        let mut sub = table.subscribe();
        assert!(!sub.is_closed());
        sub.close();
        assert!(sub.is_closed());
        // A closed subscription errors rather than blocking forever.
        assert!(sub.recv().await.is_err());
        assert!(sub.try_recv().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn subscribe_receives_committed_batch_then_commit() {
        let (table, dir) = temp_table("basic").await;
        let mut sub = table.subscribe();
        table.write_async(vec![batch(&[1, 2, 3])]).await.unwrap();
        table.commit_async().await.unwrap();

        match sub.recv().await.unwrap() {
            TableEvent::Batch(b) => assert_eq!(b.num_rows(), 3),
            other => panic!("expected Batch, got {other:?}"),
        }
        match sub.recv().await.unwrap() {
            TableEvent::Commit { rows } => assert_eq!(rows, 3),
            other => panic!("expected Commit, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn filtered_subscription_skips_non_matching_commits() {
        let (table, dir) = temp_table("filtered").await;
        let mut sub = table.subscribe_filtered("v >= 10");

        // First commit matches nothing -> the subscriber must not yield it.
        table.write_async(vec![batch(&[1, 2, 3])]).await.unwrap();
        table.commit_async().await.unwrap();
        // Second commit has matching rows.
        table.write_async(vec![batch(&[5, 10, 20])]).await.unwrap();
        table.commit_async().await.unwrap();

        match sub.recv().await.unwrap() {
            TableEvent::Batch(b) => {
                assert_eq!(b.num_rows(), 2, "only v>=10 rows should be delivered");
            }
            other => panic!("expected filtered Batch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
