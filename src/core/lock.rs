use object_store::{ObjectStore, PutMode, PutOptions, path::Path, UpdateVersion};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
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
}

impl FileBasedLock {
    pub fn new(store: Arc<dyn ObjectStore>, path: Path, ttl_seconds: u64) -> Self {
        Self {
            store,
            path,
            owner: Uuid::new_v4().to_string(),
            ttl_seconds,
        }
    }

    pub async fn try_acquire(&self) -> Result<bool> {
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

        match self.store.put_opts(&self.path, bytes.clone().into(), opts.clone()).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::AlreadyExists { .. }) => {
                // Check if expired
                match self.store.get(&self.path).await {
                    Ok(get_res) => {
                        let meta = get_res.meta.clone();
                        let current_bytes = get_res.bytes().await?;
                        if let Ok(current_payload) = serde_json::from_slice::<LockPayload>(&current_bytes) {
                            if now > current_payload.expires_at {
                                // It's expired. Try to steal it using UpdateVersion if supported (S3/GCS)
                                let update_opts = PutOptions {
                                    mode: PutMode::Update(UpdateVersion {
                                        e_tag: meta.e_tag,
                                        version: meta.version,
                                    }),
                                    ..Default::default()
                                };
                                
                                match self.store.put_opts(&self.path, bytes.clone().into(), update_opts).await {
                                    Ok(_) => return Ok(true),
                                    Err(object_store::Error::NotImplemented) | Err(object_store::Error::NotSupported { .. }) => {
                                        // Fallback for LocalFileSystem or stores that don't support conditional updates
                                        let _ = self.store.delete(&self.path).await;
                                        let jitter = rand::random::<u64>() % 50;
                                        tokio::time::sleep(std::time::Duration::from_millis(jitter)).await;
                                        
                                        match self.store.put_opts(&self.path, bytes.clone().into(), opts).await {
                                            Ok(_) => return Ok(true),
                                            Err(_) => return Ok(false),
                                        }
                                    }
                                    Err(_) => return Ok(false),
                                }
                            }
                        }
                    }
                    Err(_) => {} // File deleted between our PutMode::Create and get
                }
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }
    
    pub async fn acquire(&self) -> Result<()> {
        let max_retries = 100;
        for attempt in 0..max_retries {
            if self.try_acquire().await? {
                return Ok(());
            }
            let base_delay = 50 * (2u64.pow(attempt.min(6) as u32));
            let jitter = rand::random::<u64>() % base_delay;
            tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
        }
        Err(anyhow::anyhow!("Failed to acquire distributed lock after {} attempts", max_retries))
    }

    pub async fn release(&self) -> Result<()> {
        match self.store.get(&self.path).await {
            Ok(res) => {
                let bytes = res.bytes().await?;
                if let Ok(payload) = serde_json::from_slice::<LockPayload>(&bytes) {
                    if payload.owner == self.owner {
                        let _ = self.store.delete(&self.path).await;
                    }
                }
            }
            Err(_) => {}
        }
        Ok(())
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
        let lock = FileBasedLock::new(store.clone(), path.clone(), 30);

        assert!(lock.try_acquire().await.unwrap(), "Should acquire lock");
        assert!(!FileBasedLock::new(store.clone(), path.clone(), 30).try_acquire().await.unwrap(), "Second acquire should fail");
        
        lock.release().await.unwrap();
        
        assert!(FileBasedLock::new(store.clone(), path.clone(), 30).try_acquire().await.unwrap(), "Should acquire after release");
    }

    #[tokio::test]
    async fn test_lock_expiration_steal() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("test_expire.lock");
        
        // Lock with 0 TTL (expires immediately)
        let lock1 = FileBasedLock::new(store.clone(), path.clone(), 0);
        assert!(lock1.try_acquire().await.unwrap());

        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

        let lock2 = FileBasedLock::new(store.clone(), path.clone(), 30);
        assert!(lock2.try_acquire().await.unwrap(), "Should steal expired lock");
    }
}
