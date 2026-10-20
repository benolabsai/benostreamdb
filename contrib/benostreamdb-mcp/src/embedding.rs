//! Local and HTTP embedding support for the BenoStreamDB MCP server.
//!
//! Local inference uses [Candle](https://github.com/huggingface/candle) — pure
//! Rust, no ONNX Runtime, no Python. The model is downloaded once from the
//! Hugging Face Hub (`hf-hub`) into a private cache and reused. HTTP providers
//! (Ollama, OpenAI-compatible, Hugging Face Inference, TEI) remain available for
//! GPU or managed inference.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Default local model. Retrieval-tuned, 384-d, small enough for CPU.
pub const DEFAULT_EMBEDDING_MODEL: &str = "BAAI/bge-small-en-v1.5";

/// The BGE recommendation for short retrieval queries. Documents are embedded
/// without a prefix; only queries receive this instruction.
pub const DEFAULT_BGE_QUERY_PREFIX: &str =
    "Represent this sentence for searching relevant passages: ";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    Local,
    Http,
    None,
}

impl EmbeddingProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Http => "http",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// `None` = auto: HTTP when `url` is set, otherwise local.
    pub provider: Option<EmbeddingProvider>,
    /// Hugging Face model id for local inference, or a model id sent verbatim
    /// to an HTTP endpoint.
    pub model: String,
    /// Expected vector dimension. Discovered from the model/config when
    /// possible; set explicitly only to truncate an MRL model.
    pub dim: Option<usize>,
    /// Private Hugging Face cache. Defaults to `<benostreamdb data dir>/models`.
    pub cache_dir: PathBuf,
    /// Local inference batch size.
    pub batch_size: usize,
    /// Tokenizer truncation length.
    pub max_length: usize,
    /// Query-side instruction prefix. Empty string disables it.
    pub query_prefix: Option<String>,
    /// Document-side instruction prefix. Empty string disables it.
    pub document_prefix: Option<String>,
    /// HTTP endpoint for `provider = "http"`.
    pub url: Option<String>,
    /// Bearer token for the HTTP endpoint.
    pub api_key: Option<String>,

    // Retained for configuration compatibility; local Candle inference is CPU-only.
    pub device: String,
    pub quantized: bool,
    pub intra_threads: Option<usize>,
    pub show_download_progress: bool,
    pub model_path: Option<PathBuf>,
    pub pooling: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: None,
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            dim: None,
            cache_dir: resolve_default_model_cache(),
            batch_size: 16,
            max_length: 512,
            query_prefix: None,
            document_prefix: None,
            url: None,
            api_key: None,
            device: "cpu".to_string(),
            quantized: false,
            intra_threads: None,
            show_download_progress: false,
            model_path: None,
            pooling: None,
        }
    }
}

impl EmbeddingConfig {
    /// Resolve provider precedence: explicit config wins; otherwise an HTTP URL
    /// implies HTTP; otherwise use the in-process Candle model.
    pub fn resolved_provider(&self) -> EmbeddingProvider {
        match self.provider {
            Some(p) => p,
            None => {
                if self
                    .url
                    .as_deref()
                    .map(|u| !u.trim().is_empty())
                    .unwrap_or(false)
                {
                    EmbeddingProvider::Http
                } else {
                    EmbeddingProvider::Local
                }
            }
        }
    }

    /// Overlay environment variables onto this config.
    pub fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_PROVIDER") {
            self.provider = parse_provider(&v).or(self.provider);
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_MODEL") {
            if !v.trim().is_empty() {
                self.model = v.trim().to_string();
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_DIM") {
            if let Ok(d) = v.trim().parse::<usize>() {
                if d > 0 {
                    self.dim = Some(d);
                }
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_CACHE_DIR") {
            if !v.trim().is_empty() {
                self.cache_dir = PathBuf::from(v.trim());
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_BATCH_SIZE") {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    self.batch_size = n;
                }
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_MAX_LENGTH") {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    self.max_length = n;
                }
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_QUERY_PREFIX") {
            self.query_prefix = Some(v);
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_DOCUMENT_PREFIX") {
            self.document_prefix = Some(v);
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_URL") {
            if !v.trim().is_empty() {
                self.url = Some(v.trim().to_string());
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_API_KEY") {
            if !v.trim().is_empty() {
                self.api_key = Some(v.trim().to_string());
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_SHOW_DOWNLOAD_PROGRESS") {
            self.show_download_progress = parse_bool(&v).unwrap_or(self.show_download_progress);
        }
    }

    /// The instruction prefix to prepend to queries.
    pub fn effective_query_prefix(&self) -> String {
        if let Some(p) = &self.query_prefix {
            return p.clone();
        }
        default_prefixes(&self.model).0.unwrap_or("").to_string()
    }

    /// The instruction prefix to prepend to documents.
    pub fn effective_document_prefix(&self) -> String {
        if let Some(p) = &self.document_prefix {
            return p.clone();
        }
        default_prefixes(&self.model).1.unwrap_or("").to_string()
    }

    /// Whether this config can produce embeddings at all.
    pub fn is_enabled(&self) -> bool {
        self.resolved_provider() != EmbeddingProvider::None
    }

    /// Stable identity persisted in table metadata.
    pub fn canonical_model_id(&self) -> Result<String> {
        match self.resolved_provider() {
            EmbeddingProvider::Local => Ok(huggingface_id(&self.model)),
            EmbeddingProvider::Http | EmbeddingProvider::None => {
                let model = self.model.trim();
                if model.is_empty() {
                    return Err(anyhow!("embedding.model is empty"));
                }
                Ok(model.to_string())
            }
        }
    }

    /// Best-known vector dimension. Local dimensions come from the model
    /// catalogue; HTTP dimensions are inferred for common model ids and
    /// otherwise learned from the first response.
    pub fn resolved_dim(&self) -> Option<usize> {
        if let Some(dim) = self.dim.filter(|d| *d > 0) {
            return Some(dim);
        }
        match self.resolved_provider() {
            EmbeddingProvider::Local => local_model_dim(&self.model),
            EmbeddingProvider::Http => known_http_dim(&self.model),
            EmbeddingProvider::None => None,
        }
    }
}

/// Status/inspection payload for the `list_embedding_models` MCP tool.
pub fn embedding_status(
    config: &EmbeddingConfig,
    loaded: bool,
    active_dim: Option<usize>,
) -> Value {
    let provider = config.resolved_provider();
    let dim = active_dim.or_else(|| config.resolved_dim());
    let providers = vec!["local", "http", "none"];

    json!({
        "provider": provider.as_str(),
        "model": config.model.clone(),
        "resolved_model": config.canonical_model_id().ok(),
        "dim": dim,
        "cache_dir": config.cache_dir.to_string_lossy().to_string(),
        "device": config.device.clone(),
        "loaded": loaded,
        "query_prefix": config.effective_query_prefix(),
        "document_prefix": config.effective_document_prefix(),
        "http_url": config.url.clone(),
        "http_api_key_set": config
            .api_key
            .as_deref()
            .map(|key| !key.is_empty())
            .unwrap_or(false),
        "providers": providers,
        "local_models": list_local_models(),
    })
}

/// Common local models exposed in the status payload. Candle does not ship a
/// model registry; this is a curated shortlist, not an allowlist.
pub fn list_local_models() -> Vec<Value> {
    let models = [
        ("BAAI/bge-small-en-v1.5", 384),
        ("BAAI/bge-base-en-v1.5", 768),
        ("BAAI/bge-large-en-v1.5", 1024),
        ("BAAI/bge-m3", 1024),
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
        ("intfloat/multilingual-e5-small", 384),
        ("intfloat/multilingual-e5-base", 768),
        ("intfloat/multilingual-e5-large", 1024),
        ("nomic-ai/nomic-embed-text-v1.5", 768),
        ("mixedbread-ai/mxbai-embed-large-v1", 1024),
        ("jinaai/jina-embeddings-v2-base-code", 768),
    ];
    models
        .into_iter()
        .map(|(model, dim)| {
            json!({
                "huggingface_id": model,
                "dim": dim,
            })
        })
        .collect()
}

/// A loaded local Candle model plus the metadata needed to embed queries and
/// documents consistently.
pub struct LocalEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    pub model_id: String,
    pub dim: usize,
    query_prefix: String,
    document_prefix: String,
    batch_size: usize,
    device: Device,
}

impl LocalEmbedder {
    /// Download (once) and load a Hugging Face BERT-family model. This is a
    /// blocking operation and must be called from `spawn_blocking`.
    pub fn load(config: &EmbeddingConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.cache_dir).with_context(|| {
            format!(
                "failed to create embedding model cache directory {}",
                config.cache_dir.display()
            )
        })?;

        let model_id = huggingface_id(&config.model);
        let api = ApiBuilder::new()
            .with_cache_dir(config.cache_dir.clone())
            .build()
            .map_err(|e| anyhow!("failed to initialize the Hugging Face Hub client: {e}"))?;
        let repo = api.model(model_id.clone());

        let config_path = repo
            .get("config.json")
            .map_err(|e| anyhow!("failed to download {model_id}/config.json: {e}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow!("failed to download {model_id}/tokenizer.json: {e}"))?;
        let weights_path = repo.get("model.safetensors").map_err(|e| {
            anyhow!(
                "failed to download {model_id}/model.safetensors: {e}. \
                 Candle requires safetensors weights; use a repo that publishes them \
                 or convert pytorch_model.bin once."
            )
        })?;

        let bert_config: BertConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| anyhow!("failed to mmap {model_id}/model.safetensors: {e}"))?
        };
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| anyhow!("failed to load the {model_id} BERT model: {e}"))?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("failed to load {}: {e}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_length.max(1),
                ..Default::default()
            }))
            .map_err(|e| anyhow!("failed to configure tokenizer truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let dim = config
            .dim
            .filter(|d| *d > 0)
            .unwrap_or(bert_config.hidden_size);
        Ok(Self {
            model,
            tokenizer,
            model_id,
            dim,
            query_prefix: config.effective_query_prefix(),
            document_prefix: config.effective_document_prefix(),
            batch_size: config.batch_size.max(1),
            device,
        })
    }

    /// Embed a batch of documents (document-side prefix applied when the model
    /// uses one).
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = texts
            .iter()
            .map(|text| format!("{}{}", self.document_prefix, text))
            .collect();
        self.embed_texts(&prefixed)
    }

    /// Embed a search query (query-side instruction prefix applied).
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let input = format!("{}{}", self.query_prefix, text);
        let mut vectors = self.embed_texts(&[input])?;
        vectors
            .pop()
            .ok_or_else(|| anyhow!("embedding model returned no vector"))
    }

    fn embed_texts(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let encodings = self
                .tokenizer
                .encode_batch(chunk.to_vec(), true)
                .map_err(|e| anyhow!("tokenization failed: {e}"))?;
            if encodings.is_empty() {
                continue;
            }
            let batch = encodings.len();
            let seq_len = encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .max()
                .unwrap_or(0)
                .max(1);

            let mut ids = Vec::with_capacity(batch * seq_len);
            let mut masks = Vec::with_capacity(batch * seq_len);
            let mut types = Vec::with_capacity(batch * seq_len);
            for encoding in &encodings {
                let encoding_ids = encoding.get_ids();
                let encoding_mask = encoding.get_attention_mask();
                let encoding_types = encoding.get_type_ids();
                for position in 0..seq_len {
                    ids.push(*encoding_ids.get(position).unwrap_or(&0));
                    masks.push(*encoding_mask.get(position).unwrap_or(&0));
                    types.push(*encoding_types.get(position).unwrap_or(&0));
                }
            }

            let input_ids = Tensor::from_vec(ids, (batch, seq_len), &self.device)?;
            let attention_mask = Tensor::from_vec(masks, (batch, seq_len), &self.device)?;
            let token_type_ids = Tensor::from_vec(types, (batch, seq_len), &self.device)?;

            let output = self
                .model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))
                .map_err(|e| anyhow!("BERT forward pass failed: {e}"))?;
            // BGE v1.5 uses CLS pooling followed by L2 normalization.
            let cls = output
                .i((.., 0, ..))
                .map_err(|e| anyhow!("failed to select the CLS token: {e}"))?
                .to_vec2::<f32>()
                .map_err(|e| anyhow!("failed to read the embedding tensor: {e}"))?;

            for mut vector in cls {
                let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for value in &mut vector {
                        *value /= norm;
                    }
                }
                all.push(vector);
            }
        }
        Ok(all)
    }
}

/// Canonical Hugging Face id for the local catalogue. Arbitrary ids pass through.
fn huggingface_id(model: &str) -> String {
    let raw = model.trim();
    let key = raw.to_ascii_lowercase();
    if key.contains("bge-small") || key.contains("bgesmall") {
        "BAAI/bge-small-en-v1.5".to_string()
    } else if key.contains("bge-base") || key.contains("bgebase") {
        "BAAI/bge-base-en-v1.5".to_string()
    } else if key.contains("bge-large") || key.contains("bgelarge") {
        "BAAI/bge-large-en-v1.5".to_string()
    } else if key.contains("bge-m3") || key.contains("bgem3") {
        "BAAI/bge-m3".to_string()
    } else if key.contains("allminilml6") || key.contains("all-minilm-l6") {
        "sentence-transformers/all-MiniLM-L6-v2".to_string()
    } else if key.contains("multilingual-e5-small") || key.contains("multilinguale5small") {
        "intfloat/multilingual-e5-small".to_string()
    } else if key.contains("multilingual-e5-base") || key.contains("multilinguale5base") {
        "intfloat/multilingual-e5-base".to_string()
    } else if key.contains("multilingual-e5-large") || key.contains("multilinguale5large") {
        "intfloat/multilingual-e5-large".to_string()
    } else if key.contains("nomic-embed-text-v1.5") || key.contains("nomicembedtextv15") {
        "nomic-ai/nomic-embed-text-v1.5".to_string()
    } else if key.contains("mxbai-embed-large") || key.contains("mxbaiembedlarge") {
        "mixedbread-ai/mxbai-embed-large-v1".to_string()
    } else if key.contains("jina-embeddings-v2-base-code")
        || key.contains("jinaembeddingsv2basecode")
    {
        "jinaai/jina-embeddings-v2-base-code".to_string()
    } else {
        raw.to_string()
    }
}

fn local_model_dim(model: &str) -> Option<usize> {
    let key = huggingface_id(model).to_ascii_lowercase();
    match key.as_str() {
        "baai/bge-small-en-v1.5" => Some(384),
        "baai/bge-base-en-v1.5" => Some(768),
        "baai/bge-large-en-v1.5" => Some(1024),
        "baai/bge-m3" => Some(1024),
        "sentence-transformers/all-minilm-l6-v2" => Some(384),
        "intfloat/multilingual-e5-small" => Some(384),
        "intfloat/multilingual-e5-base" => Some(768),
        "intfloat/multilingual-e5-large" => Some(1024),
        "nomic-ai/nomic-embed-text-v1.5" => Some(768),
        "mixedbread-ai/mxbai-embed-large-v1" => Some(1024),
        "jinaai/jina-embeddings-v2-base-code" => Some(768),
        _ => None,
    }
}

fn known_http_dim(model: &str) -> Option<usize> {
    match model.trim().to_ascii_lowercase().as_str() {
        "text-embedding-3-small" => Some(1536),
        "text-embedding-3-large" => Some(3072),
        "text-embedding-ada-002" => Some(1536),
        "nomic-embed-text" => Some(768),
        "all-minilm" | "all-minilm-l6-v2" => Some(384),
        "mxbai-embed-large" => Some(1024),
        "bge-m3" => Some(1024),
        "baai/bge-small-en-v1.5" | "bge-small-en-v1.5" => Some(384),
        "baai/bge-base-en-v1.5" | "bge-base-en-v1.5" => Some(768),
        "baai/bge-large-en-v1.5" | "bge-large-en-v1.5" => Some(1024),
        "intfloat/multilingual-e5-small" | "multilingual-e5-small" => Some(384),
        "intfloat/multilingual-e5-base" | "multilingual-e5-base" => Some(768),
        "intfloat/multilingual-e5-large" | "multilingual-e5-large" => Some(1024),
        "nomic-ai/nomic-embed-text-v1.5" | "nomic-embed-text-v1.5" => Some(768),
        "mixedbread-ai/mxbai-embed-large-v1" | "mxbai-embed-large-v1" => Some(1024),
        "jinaai/jina-embeddings-v2-base-code" | "jina-embeddings-v2-base-code" => Some(768),
        _ => None,
    }
}

fn default_prefixes(model: &str) -> (Option<&'static str>, Option<&'static str>) {
    let model = model.trim().to_ascii_lowercase();
    let model = model
        .strip_prefix("sentence-transformers/")
        .unwrap_or(&model);
    if model.contains("bge") || model.contains("mxbai") || model.contains("snowflake-arctic") {
        return (Some(DEFAULT_BGE_QUERY_PREFIX), None);
    }
    if model.contains("e5") {
        return (Some("query: "), Some("passage: "));
    }
    if model.contains("nomic-embed-text") || model.contains("nomicembedtext") {
        return (Some("search_query: "), Some("search_document: "));
    }
    (None, None)
}

fn parse_provider(value: &str) -> Option<EmbeddingProvider> {
    match value.trim().to_ascii_lowercase().as_str() {
        "local" | "candle" => Some(EmbeddingProvider::Local),
        "http" | "remote" | "ollama" | "openai" | "huggingface" | "hf" | "tei" => {
            Some(EmbeddingProvider::Http)
        }
        "none" | "off" | "disabled" => Some(EmbeddingProvider::None),
        _ => None,
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Default model cache inside the BenoStreamDB private data directory.
pub fn resolve_default_model_cache() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".benostreamdb")))
        .unwrap_or_else(|_| std::env::temp_dir().join(".benostreamdb"));
    base.join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_bge_small_local() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model, DEFAULT_EMBEDDING_MODEL);
        assert_eq!(config.resolved_provider(), EmbeddingProvider::Local);
        assert_eq!(config.resolved_dim(), Some(384));
        assert_eq!(config.effective_query_prefix(), DEFAULT_BGE_QUERY_PREFIX);
        assert_eq!(config.effective_document_prefix(), "");
    }

    #[test]
    fn url_implies_http_when_provider_is_unset() {
        let config = EmbeddingConfig {
            url: Some("http://localhost:11434/api/embeddings".to_string()),
            ..Default::default()
        };
        assert_eq!(config.resolved_provider(), EmbeddingProvider::Http);
    }

    #[test]
    fn huggingface_ids_and_variant_names_normalize() {
        assert_eq!(huggingface_id("BGESmallENV15Q"), "BAAI/bge-small-en-v1.5");
        assert_eq!(huggingface_id("BGELargeENV15Q"), "BAAI/bge-large-en-v1.5");
        assert_eq!(
            huggingface_id("sentence-transformers/all-MiniLM-L6-v2"),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(huggingface_id("vendor/custom-embed"), "vendor/custom-embed");
    }

    #[test]
    fn local_model_dimensions_are_known() {
        assert_eq!(local_model_dim("BAAI/bge-small-en-v1.5"), Some(384));
        assert_eq!(local_model_dim("BAAI/bge-large-en-v1.5"), Some(1024));
        assert_eq!(local_model_dim("unknown/model"), None);
    }

    #[test]
    fn prefixes_are_model_family_specific() {
        assert_eq!(
            default_prefixes("BAAI/bge-small-en-v1.5"),
            (Some(DEFAULT_BGE_QUERY_PREFIX), None)
        );
        assert_eq!(
            default_prefixes("intfloat/multilingual-e5-small"),
            (Some("query: "), Some("passage: "))
        );
        assert_eq!(default_prefixes("text-embedding-3-small"), (None, None));
    }

    /// End-to-end local model check. Ignored by default because the first run
    /// downloads the model into the private cache directory.
    #[test]
    #[ignore = "downloads the default Hugging Face model on first run"]
    fn local_embedder_loads_default_model_and_embeds() {
        let config = EmbeddingConfig {
            cache_dir: std::env::temp_dir().join("benostreamdb-candle-test-models"),
            ..Default::default()
        };
        let mut embedder =
            LocalEmbedder::load(&config).unwrap_or_else(|e| panic!("load local embedder: {e}"));
        assert_eq!(embedder.dim, 384);

        let docs = embedder
            .embed_documents(&["BenoStreamDB indexes Parquet files in place.".to_string()])
            .unwrap_or_else(|e| panic!("embed document: {e}"));
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].len(), 384);
        let norm = docs[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);

        let query = embedder
            .embed_query("vector search")
            .unwrap_or_else(|e| panic!("embed query: {e}"));
        assert_eq!(query.len(), 384);
    }
}
