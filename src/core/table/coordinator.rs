// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Multi-machine work coordination for the ingest orchestrator (A4).
//!
//! The single-machine path ([`Table::ingest_async`](crate::core::table::Table::ingest_async))
//! plans a static unit list and runs it through a bounded pool. Multi-machine
//! mode lets N independent processes (on N machines) share one work queue: each
//! claims a unit under an object-store lease, builds it, commits via the OCC
//! CAS, and marks it done. A dead node's lease expires and another node steals
//! it.
//!
//! The lease primitive is the existing [`FileBasedLock`] (object-store CAS with
//! heartbeats and expiry-steal) — the same "custom coordination" the incumbents
//! build with Raft, minus the cluster. No broker, no etcd, no consensus service.
//!
//! ## Correctness notes
//! - **At-least-once**: a stolen or retried unit may be built twice. The OCC
//!   manifest CAS serializes commits, and the shared `_ingest_done` markers
//!   dedupe completed units, so a re-run skips them.
//! - **Fencing**: a stale owner's commit fails on manifest version conflict.
//! - **Local filesystem caveat**: `FileBasedLock`'s expiry-steal falls back to a
//!   non-atomic delete→create on stores without conditional update (e.g. local
//!   FS). Use S3/GCS/Azure for production multi-machine coordination.

use anyhow::Result;
use async_trait::async_trait;
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::ObjectStore;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use crate::core::lock::{FileBasedLock, LockGuard};

/// A unit of ingest work: a row range within one input file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkUnit {
    /// Input file path.
    pub path: String,
    /// Inclusive start row.
    pub row_start: usize,
    /// Exclusive end row (`0` for whole-file streaming formats).
    pub row_end: usize,
}

impl WorkUnit {
    /// Construct a work unit.
    pub fn new(path: impl Into<String>, row_start: usize, row_end: usize) -> Self {
        Self {
            path: path.into(),
            row_start,
            row_end,
        }
    }

    /// Stable identity of this unit (used for the resume sidecar and markers).
    pub fn key(&self) -> String {
        format!("{}:{}:{}", self.path, self.row_start, self.row_end)
    }

    /// Filesystem-safe, deterministic name for this unit's lease/marker object.
    ///
    /// FNV-1a over the key: deterministic across machines and Rust versions
    /// (unlike `DefaultHasher`), and fixed-length so long paths don't overflow
    /// object-name limits.
    fn slug(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.key().as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{hash:016x}")
    }
}

/// A claimed work unit, held under an object-store lease.
///
/// Dropping the lease (via [`WorkCoordinator::release`] or on error) releases
/// the underlying lock so another node can claim the unit.
pub struct Lease {
    /// The claimed unit.
    pub unit: WorkUnit,
    guard: LockGuard,
}

/// Lease-based work coordination across machines.
#[async_trait]
pub trait WorkCoordinator: Send + Sync {
    /// Total number of units in the queue.
    fn total_units(&self) -> usize;

    /// Claim the next unleased, incomplete unit. `None` means the queue is
    /// drained (every unit is either complete or currently leased elsewhere).
    async fn claim(&self, completed: &HashSet<String>) -> Result<Option<Lease>>;

    /// Mark a unit complete and release its lease.
    async fn complete(&self, lease: Lease) -> Result<()>;

    /// Release a lease without completing it (the unit becomes retryable).
    fn release(&self, lease: Lease);

    /// Units already marked complete (shared across machines).
    async fn completed(&self) -> Result<HashSet<String>>;
}

/// Object-store lease backend: the default, zero-dependency coordinator.
///
/// Each unit is guarded by a [`FileBasedLock`] at `_ingest_leases/<slug>.lock`;
/// completion is recorded as a marker object at `_ingest_done/<slug>.done`
/// whose body is the unit key.
pub struct ObjectStoreCoordinator {
    store: Arc<dyn ObjectStore>,
    units: Vec<WorkUnit>,
    lease_prefix: Path,
    done_prefix: Path,
    ttl: Duration,
    clock_skew_ms: u64,
}

impl ObjectStoreCoordinator {
    /// Create a coordinator over `units`, leasing with `ttl`.
    pub fn new(store: Arc<dyn ObjectStore>, units: Vec<WorkUnit>, ttl: Duration) -> Self {
        Self {
            store,
            units,
            lease_prefix: Path::from("_ingest_leases"),
            done_prefix: Path::from("_ingest_done"),
            ttl,
            clock_skew_ms: 5_000,
        }
    }

    /// Override the NTP clock-skew allowance (ms) used when judging expiry.
    pub fn with_clock_skew(mut self, ms: u64) -> Self {
        self.clock_skew_ms = ms;
        self
    }

    fn lease_path(&self, unit: &WorkUnit) -> Path {
        Path::from(format!("{}/{}.lock", self.lease_prefix, unit.slug()))
    }

    fn done_path(&self, unit: &WorkUnit) -> Path {
        Path::from(format!("{}/{}.done", self.done_prefix, unit.slug()))
    }
}

#[async_trait]
impl WorkCoordinator for ObjectStoreCoordinator {
    fn total_units(&self) -> usize {
        self.units.len()
    }

    async fn claim(&self, completed: &HashSet<String>) -> Result<Option<Lease>> {
        for unit in &self.units {
            if completed.contains(&unit.key()) {
                continue;
            }
            // Clamp to a 1-second minimum: a 0-second TTL would make the
            // heartbeat spin and race its own CAS renewals.
            let lock = FileBasedLock::new(
                self.store.clone(),
                self.lease_path(unit),
                self.ttl.as_secs().max(1),
            )
            .with_clock_skew(self.clock_skew_ms);
            if let Some(guard) = lock.try_acquire().await? {
                return Ok(Some(Lease {
                    unit: unit.clone(),
                    guard,
                }));
            }
        }
        Ok(None)
    }

    async fn complete(&self, lease: Lease) -> Result<()> {
        let key = lease.unit.key();
        self.store
            .put(&self.done_path(&lease.unit), key.into_bytes().into())
            .await?;
        // Dropping the guard releases the lease (and stops its heartbeat).
        drop(lease.guard);
        Ok(())
    }

    fn release(&self, lease: Lease) {
        drop(lease.guard);
    }

    async fn completed(&self) -> Result<HashSet<String>> {
        let mut done = HashSet::new();
        let mut stream = self.store.list(Some(&self.done_prefix));
        loop {
            match stream.try_next().await {
                Ok(Some(meta)) => {
                    if let Ok(res) = self.store.get(&meta.location).await {
                        if let Ok(bytes) = res.bytes().await {
                            if let Ok(key) = String::from_utf8(bytes.to_vec()) {
                                done.insert(key);
                            }
                        }
                    }
                }
                Ok(None) => break,
                // A missing prefix simply means nothing is done yet.
                Err(object_store::Error::NotFound { .. }) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn units() -> Vec<WorkUnit> {
        vec![
            WorkUnit::new("a.parquet", 0, 100),
            WorkUnit::new("a.parquet", 100, 200),
            WorkUnit::new("b.parquet", 0, 50),
        ]
    }

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    #[test]
    fn slug_is_deterministic_and_fixed_length() {
        let u = WorkUnit::new(
            "some/very/long/path/to/a/file.parquet",
            1_000_000,
            2_000_000,
        );
        assert_eq!(u.slug(), u.slug());
        assert_eq!(u.slug().len(), 16);
        assert_ne!(u.slug(), WorkUnit::new("other.parquet", 0, 1).slug());
    }

    #[tokio::test]
    async fn claims_are_disjoint_across_coordinators() {
        let s = store();
        let c1 = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let c2 = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let done = HashSet::new();
        let l1 = c1.claim(&done).await.unwrap().unwrap();
        let l2 = c2.claim(&done).await.unwrap().unwrap();
        assert_ne!(
            l1.unit.key(),
            l2.unit.key(),
            "two coordinators must not claim the same unit"
        );
    }

    #[tokio::test]
    async fn completed_units_are_skipped() {
        let s = store();
        let c = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let done = HashSet::new();
        let lease = c.claim(&done).await.unwrap().unwrap();
        let key = lease.unit.key();
        c.complete(lease).await.unwrap();

        let done = c.completed().await.unwrap();
        assert!(done.contains(&key), "completed marker should be visible");

        // A fresh coordinator must not re-claim the completed unit.
        let c2 = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let l2 = c2.claim(&done).await.unwrap().unwrap();
        assert_ne!(l2.unit.key(), key);
    }

    #[tokio::test]
    async fn released_lease_is_reclaimable() {
        let s = store();
        let c = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let done = HashSet::new();
        let lease = c.claim(&done).await.unwrap().unwrap();
        let key = lease.unit.key();
        c.release(lease);
        // The guard's Drop deletes the lock asynchronously; give it a moment.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let c2 = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let l2 = c2.claim(&done).await.unwrap().unwrap();
        assert_eq!(l2.unit.key(), key, "released unit should be re-claimable");
    }

    #[tokio::test]
    async fn expired_lease_is_stolen() {
        // Simulate a dead node: write an already-expired lease for the first
        // unit directly, then verify a live coordinator steals it. (Driving a
        // real lease to expiry would need a spinning 0-TTL heartbeat, which
        // races its own CAS renewals.)
        let s = store();
        let c = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let unit0 = units()[0].clone();
        let expired = crate::core::lock::LockPayload {
            owner: "dead-node".to_string(),
            expires_at: 0,
        };
        s.put(
            &c.lease_path(&unit0),
            serde_json::to_vec(&expired).unwrap().into(),
        )
        .await
        .unwrap();

        let done = HashSet::new();
        let lease = c.claim(&done).await.unwrap().unwrap();
        assert_eq!(
            lease.unit.key(),
            unit0.key(),
            "expired lease should be stolen"
        );
    }

    #[tokio::test]
    async fn drained_queue_returns_none() {
        let s = store();
        let c = ObjectStoreCoordinator::new(s.clone(), units(), Duration::from_secs(60));
        let mut done = HashSet::new();
        // Complete every unit.
        while let Some(lease) = c.claim(&done).await.unwrap() {
            done.insert(lease.unit.key());
            c.complete(lease).await.unwrap();
        }
        assert_eq!(done.len(), 3);
        assert!(c.claim(&done).await.unwrap().is_none());
    }
}
