// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

#[async_trait]
pub trait EmbeddingFunction: Send + Sync {
    /// Vectorize a list of strings.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Get the dimension of the embeddings produced by this function.
    fn dimension(&self) -> usize;

    /// Get the name of this embedding function.
    fn name(&self) -> &str;
}

pub struct EmbeddingRegistry {
    functions: HashMap<String, Arc<dyn EmbeddingFunction>>,
}

impl Default for EmbeddingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, func: Arc<dyn EmbeddingFunction>) {
        self.functions.insert(name, func);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn EmbeddingFunction>> {
        self.functions.get(name).cloned()
    }
}

lazy_static::lazy_static! {
    pub static ref GLOBAL_REGISTRY: parking_lot::RwLock<EmbeddingRegistry> = parking_lot::RwLock::new(EmbeddingRegistry::new());
}

pub fn register_embedded_func(name: String, func: Arc<dyn EmbeddingFunction>) {
    let mut registry = GLOBAL_REGISTRY.write();
    {
        registry.register(name, func);
    }
}

pub fn get_embedded_func(name: &str) -> Option<Arc<dyn EmbeddingFunction>> {
    GLOBAL_REGISTRY.read().get(name)
}

/// A bridge to call Python embedding functions from Rust.
type EmbeddingCallback = Box<dyn Fn(Vec<String>) -> Result<Vec<Vec<f32>>> + Send + Sync>;

pub struct PythonCallbackFunction {
    name: String,
    callback: EmbeddingCallback,
    dim: usize,
}

#[async_trait]
impl EmbeddingFunction for PythonCallbackFunction {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Use spawn_blocking to avoid blocking the Tokio executor thread,
        // which is critical when the Python callback holds the GIL.
        let callback_ref = &self.callback;

        tokio::task::block_in_place(|| (callback_ref)(texts))
    }

    fn dimension(&self) -> usize {
        self.dim
    }
    fn name(&self) -> &str {
        &self.name
    }
}

impl PythonCallbackFunction {
    pub fn new(
        name: String,
        dim: usize,
        callback: impl Fn(Vec<String>) -> Result<Vec<Vec<f32>>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            callback: Box::new(callback),
            dim,
        }
    }
}

pub struct RemoteFunction {
    name: String,
    endpoint: String,
    api_key: String,
    dim: usize,
    client: reqwest::Client,
}

impl RemoteFunction {
    pub fn new(name: String, endpoint: String, api_key: String, dim: usize) -> Self {
        Self {
            name,
            endpoint,
            api_key,
            dim,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingFunction for RemoteFunction {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "input": texts,
                "model": self.name
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API returned error status {}: {}", status, body);
        }

        let res_body: serde_json::Value = response.json().await?;

        let data_array = res_body["data"].as_array().ok_or_else(|| {
            anyhow::anyhow!(
                "Invalid response from embedding API: missing 'data' array. Response: {}",
                serde_json::to_string(&res_body).unwrap_or_default()
            )
        })?;

        let embeddings: Result<Vec<Vec<f32>>> = data_array
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let embedding = d["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'embedding' array in data[{}]", i))?;
                embedding
                    .iter()
                    .enumerate()
                    .map(|(j, v)| {
                        v.as_f64().map(|f| f as f32).ok_or_else(|| {
                            anyhow::anyhow!("Non-numeric value at data[{}].embedding[{}]", i, j)
                        })
                    })
                    .collect()
            })
            .collect();

        embeddings
    }

    fn dimension(&self) -> usize {
        self.dim
    }
    fn name(&self) -> &str {
        &self.name
    }
}
