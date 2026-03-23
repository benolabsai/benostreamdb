use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use async_trait::async_trait;

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
    pub static ref GLOBAL_REGISTRY: std::sync::RwLock<EmbeddingRegistry> = std::sync::RwLock::new(EmbeddingRegistry::new());
}

pub fn register_embedded_func(name: String, func: Arc<dyn EmbeddingFunction>) {
    if let Ok(mut registry) = GLOBAL_REGISTRY.write() {
        registry.register(name, func);
    }
}

pub fn get_embedded_func(name: &str) -> Option<Arc<dyn EmbeddingFunction>> {
    GLOBAL_REGISTRY.read().ok()?.get(name)
}

// --- Implementations ---

#[cfg(feature = "candle")]
pub struct CandleFunction {
    name: String,
    model: Arc<candle_transformers::models::bert::BertModel>,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    dim: usize,
}

#[async_trait]
impl EmbeddingFunction for CandleFunction {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Implementation using candle...
        // For now, return stub if actually implemented
        unimplemented!("Candle implementation details to be fleshed out in next step")
    }

    fn dimension(&self) -> usize { self.dim }
    fn name(&self) -> &str { &self.name }
}

/// A bridge to call Python embedding functions from Rust.
pub struct PythonCallbackFunction {
    name: String,
    callback: Box<dyn Fn(Vec<String>) -> Result<Vec<Vec<f32>>> + Send + Sync>,
    dim: usize,
}

#[async_trait]
impl EmbeddingFunction for PythonCallbackFunction {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        (self.callback)(texts)
    }

    fn dimension(&self) -> usize { self.dim }
    fn name(&self) -> &str { &self.name }
}

impl PythonCallbackFunction {
    pub fn new(name: String, dim: usize, callback: impl Fn(Vec<String>) -> Result<Vec<Vec<f32>>> + Send + Sync + 'static) -> Self {
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

#[async_trait]
impl EmbeddingFunction for RemoteFunction {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Implementation for different cloud providers
        // For now, a generic OpenAI-compatible implementation
        let response = self.client.post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "input": texts,
                "model": self.name
            }))
            .send()
            .await?;
            
        let res_body: serde_json::Value = response.json().await?;
        // Parse OpenAI-style response
        let embeddings = res_body["data"].as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response from embedding API"))?
            .iter()
            .map(|d| {
                d["embedding"].as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
            
        Ok(embeddings)
    }

    fn dimension(&self) -> usize { self.dim }
    fn name(&self) -> &str { &self.name }
}
