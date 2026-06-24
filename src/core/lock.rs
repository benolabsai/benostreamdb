use anyhow::Result;
use object_store::{path::Path, ObjectStore, PutMode, PutOptions, UpdateVersion};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LockPayload {
    pub owner: String,
    pub expires_at: u64,
}

pub struct FileBasedLock {
    store: Arc<dyn ObjectStore>,
    path: Path,
    owner: String,
    ttl_seconds: u64,
    clock_skew_ms: u64,
}

pub struct LockGuard {
    store: Arc<dyn ObjectStore>,
    path: Path,
    owner: String,
    abort_tx: Option<oneshot::Sender<()>>,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if let Some(tx) = self.abort_tx.take() {
            let _ = tx.send(());
        }
        let store = self.store.clone();
        let path = self.path.clone();
        let owner = self.owner.clone();

        tokio::spawn(async move {
            match store.get(&path).await {
                Ok(res) => {
                    if let Ok(bytes) = res.bytes().await {
                        if let Ok(payload) = serde_json::from_slice::<LockPayload>(&bytes) {
                            if payload.owner == owner {
                                let _ = store.delete(&path).await;
                            } else {
                                tracing::warn!(
                                    "Lock release skipped: lock is now owned by '{}', not '{}'.",
                                    payload.owner,
                                    owner
                                );
                            }
                        }
                    }
                }
                Err(_) => {} // Already deleted
            }
        });
    }
}

impl FileBasedLock {
    pub fn new(store: Arc<dyn ObjectStore>, path: Path, ttl_seconds: u64) -> Self {
        Self {
            store,
            path,
            owner: Uuid::new_v4().to_string(),
            ttl_seconds,
            clock_skew_ms: 5000, // 5 seconds default NTP drift allowance
        }
    }

    pub fn with_clock_skew(mut self, ms: u64) -> Self {
        self.clock_skew_ms = ms;
        self
    }

    pub async fn try_acquire(&self) -> Result<Option<LockGuard>> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let payload = LockPayload {
            owner: self.owner.clone(),
            expires_at: now + self.ttl_seconds,
        };
        let bytes = serde_json::to_vec(&payload)?;

        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };

        match self
            .store
            .put_opts(&self.path, bytes.clone().into(), opts.clone())
            .await
        {
            Ok(_) => Ok(Some(self.spawn_heartbeat())),
            Err(object_store::Error::AlreadyExists { .. }) => {
                // Check if expired
                match self.store.get(&self.path).await {
                    Ok(get_res) => {
                        let meta = get_res.meta.clone();
                        let current_bytes = get_res.bytes().await?;
                        if let Ok(current_payload) =
                            serde_json::from_slice::<LockPayload>(&current_bytes)
                        {
                            let current_time_ms =
                                SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
                            let expires_ms =
                                (current_payload.expires_at * 1000) + self.clock_skew_ms;

                            if current_time_ms > expires_ms {
                                // It's expired. Try to steal it using UpdateVersion if supported (S3/GCS)
                                let update_opts = PutOptions {
                                    mode: PutMode::Update(UpdateVersion {
                                        e_tag: meta.e_tag,
                                        version: meta.version,
                                    }),
                                    ..Default::default()
                                };

                                match self
                                    .store
                                    .put_opts(&self.path, bytes.clone().into(), update_opts)
                                    .await
                                {
                                    Ok(_) => return Ok(Some(self.spawn_heartbeat())),
                                    Err(object_store::Error::NotImplemented)
                                    | Err(object_store::Error::NotSupported { .. }) => {
                                        // SAFETY NOTE: This fallback path has a TOCTOU (time-of-check-time-of-use)
                                        // race condition. The delete → sleep → create_exclusive sequence is NOT atomic.
                                        // Two concurrent callers can both delete the expired lock and race to re-create it.
                                        // The random jitter reduces but does not eliminate this risk.
                                        // This is inherent to stores without conditional-update (CAS) support (e.g., LocalFileSystem).
                                        // For production distributed locking, use S3/GCS/Azure which support PutMode::Update.
                                        tracing::warn!("Lock steal using non-atomic fallback (TOCTOU risk). Consider using S3/GCS/Azure for production locking.");
                                        let _ = self.store.delete(&self.path).await;
                                        let jitter = rand::random::<u64>() % 50;
                                        tokio::time::sleep(std::time::Duration::from_millis(
                                            jitter,
                                        ))
                                        .await;

                                        match self
                                            .store
                                            .put_opts(&self.path, bytes.clone().into(), opts)
                                            .await
                                        {
                                            Ok(_) => return Ok(Some(self.spawn_heartbeat())),
                                            Err(_) => return Ok(None),
                                        }
                                    }
                                    Err(_) => return Ok(None),
                                }
                            }
                        }
                    }
                    Err(_) => {} // File deleted between our PutMode::Create and get
                }
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn spawn_heartbeat(&self) -> LockGuard {
        let (tx, mut rx) = oneshot::channel();
        let store = self.store.clone();
        let path = self.path.clone();
        let owner = self.owner.clone();
        let ttl_seconds = self.ttl_seconds;

        // Heartbeat wakes up at TTL/2 to renew
        let interval_ms = (ttl_seconds * 1000) / 2;

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut rx => {
                        // Drop guard signaled cancellation
                        break;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(interval_ms)) => {
                        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                        let payload = LockPayload {
                            owner: owner.clone(),
                            expires_at: now + ttl_seconds,
                        };
                        if let Ok(bytes) = serde_json::to_vec(&payload) {
                            // Fetch meta to do an atomic PutMode::Update if supported
                            if let Ok(get_res) = store.get(&path).await {
                                let update_opts = PutOptions {
                                    mode: PutMode::Update(UpdateVersion {
                                        e_tag: get_res.meta.e_tag,
                                        version: get_res.meta.version,
                                    }),
                                    ..Default::default()
                                };
                                let _ = store.put_opts(&path, bytes.into(), update_opts).await;
                            }
                        }
                    }
                }
            }
        });

        LockGuard {
            store: self.store.clone(),
            path: self.path.clone(),
            owner: self.owner.clone(),
            abort_tx: Some(tx),
        }
    }

    pub async fn acquire(&self) -> Result<LockGuard> {
        let max_retries = 100;
        for attempt in 0..max_retries {
            if let Some(guard) = self.try_acquire().await? {
                return Ok(guard);
            }
            let base_delay = 50 * (2u64.pow(attempt.min(6) as u32));
            let jitter = rand::random::<u64>() % base_delay;
            tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
        }
        Err(anyhow::anyhow!(
            "Failed to acquire distributed lock after {} attempts",
            max_retries
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_lock_acquire_and_release() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("test.lock");

        {
            let lock = FileBasedLock::new(store.clone(), path.clone(), 30);
            let _guard = lock
                .try_acquire()
                .await
                .unwrap()
                .expect("Should acquire lock");

            assert!(
                FileBasedLock::new(store.clone(), path.clone(), 30)
                    .try_acquire()
                    .await
                    .unwrap()
                    .is_none(),
                "Second acquire should fail"
            );
        } // guard drops here

        // Give the spawned drop task a tiny bit of time to execute
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        assert!(
            FileBasedLock::new(store.clone(), path.clone(), 30)
                .try_acquire()
                .await
                .unwrap()
                .is_some(),
            "Should acquire after release"
        );
    }

    #[tokio::test]
    async fn test_lock_expiration_steal() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("test_expire.lock");

        // Lock with 0 TTL (expires immediately)
        let lock1 = FileBasedLock::new(store.clone(), path.clone(), 0).with_clock_skew(0);
        let _guard = lock1.try_acquire().await.unwrap().expect("Should acquire");

        // Even though we have a guard, the heartbeat sleeps for TTL/2 (0ms), so it tries to renew
        // but the expiration is immediate. We sleep past the TTL + clock_skew.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let lock2 = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        assert!(
            lock2.try_acquire().await.unwrap().is_some(),
            "Should steal expired lock"
        );
    }
}
