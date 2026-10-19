//! Local, no-LLM embedding support for the BenoStreamDB MCP server.
//!
//! The default is an in-process Hugging Face model (`BAAI/bge-small-en-v1.5`,
//! quantized ONNX) run through ONNX Runtime via `fastembed`. Models are
//! downloaded once into the BenoStreamDB private data directory and reused on
//! subsequent starts. An HTTP provider remains available for Ollama /
//! OpenAI-compatible / TEI endpoints and for models that are not in the
//! `fastembed` catalogue.

use anyhow::{anyhow, Context, Result};
use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding,
    TextInitOptions, TokenizerFiles, UserDefinedEmbeddingModel,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Default retriever model. `bge-small-en-v1.5` is retrieval-tuned, 384-d, and
/// small enough to run comfortably on CPU. MiniLM is intentionally *not* a
/// default: it is a general similarity model, not a retrieval model.
pub const DEFAULT_EMBEDDING_MODEL: &str = "BAAI/bge-small-en-v1.5";

/// The BGE recommendation for short retrieval queries. Documents are embedded
/// without a prefix; only queries receive this instruction.
pub const DEFAULT_BGE_QUERY_PREFIX: &str =
    "Represent this sentence for searching relevant passages: ";

/// How embeddings are produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    /// In-process ONNX inference (`fastembed`).
    Local,
    /// HTTP endpoint: Ollama `{model, prompt}` or OpenAI-compatible `{model, input}`.
    Http,
    /// Explicitly disable embedding generation.
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

/// Embedding configuration, settable from MCP JSON (`embedding` object),
/// environment variables, or CLI flags.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// `None` = auto: HTTP when `url` is set, otherwise local.
    pub provider: Option<EmbeddingProvider>,
    /// Hugging Face model id (e.g. `BAAI/bge-small-en-v1.5`) or a
    /// `fastembed::EmbeddingModel` variant name.
    pub model: String,
    /// Prefer the quantized ONNX variant when one exists. Strongly recommended
    /// for CPU and cold start.
    pub quantized: bool,
    /// Expected vector dimension. Usually discovered from the model; required
    /// for user-defined ONNX models that do not ship a readable `config.json`.
    pub dim: Option<usize>,
    /// Private model cache. Defaults to `<benostreamdb data dir>/models`.
    pub cache_dir: PathBuf,
    /// ONNX inference batch size.
    pub batch_size: usize,
    /// Tokenizer truncation length.
    pub max_length: usize,
    /// Query-side instruction prefix. Empty string disables it.
    pub query_prefix: Option<String>,
    /// Document-side instruction prefix. Empty string disables it.
    pub document_prefix: Option<String>,
    /// `cpu`, `auto`, or `cuda`. Local inference is CPU-only in this build.
    pub device: String,
    /// Cap ONNX Runtime intra-op threads; `None` uses all cores.
    pub intra_threads: Option<usize>,
    /// Show Hugging Face download progress on first use.
    pub show_download_progress: bool,
    /// HTTP endpoint for `provider = "http"` (Ollama or OpenAI-compatible).
    pub url: Option<String>,
    /// Bearer token for the HTTP endpoint.
    pub api_key: Option<String>,
    /// Local directory or `.onnx` file for arbitrary HF models that are not in
    /// the `fastembed` catalogue. The directory must contain `tokenizer.json`.
    pub model_path: Option<PathBuf>,
    /// Pooling for user-defined models: `mean` (default) or `cls`.
    pub pooling: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: None,
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            quantized: true,
            dim: None,
            cache_dir: resolve_default_model_cache(),
            batch_size: 32,
            max_length: 512,
            query_prefix: None,
            document_prefix: None,
            device: "cpu".to_string(),
            intra_threads: None,
            show_download_progress: true,
            url: None,
            api_key: None,
            model_path: None,
            pooling: None,
        }
    }
}

impl EmbeddingConfig {
    /// Resolve provider precedence: explicit config wins; otherwise an HTTP URL
    /// implies HTTP; otherwise use the in-process model.
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
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_QUANTIZED") {
            self.quantized = parse_bool(&v).unwrap_or(self.quantized);
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_DIM") {
            if let Ok(d) = v.trim().parse::<usize>() {
                self.dim = Some(d);
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
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_DEVICE") {
            if !v.trim().is_empty() {
                self.device = v.trim().to_ascii_lowercase();
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_INTRA_THREADS") {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    self.intra_threads = Some(n);
                }
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_SHOW_DOWNLOAD_PROGRESS") {
            self.show_download_progress = parse_bool(&v).unwrap_or(self.show_download_progress);
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
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_MODEL_PATH") {
            if !v.trim().is_empty() {
                self.model_path = Some(PathBuf::from(v.trim()));
            }
        }
        if let Ok(v) = std::env::var("BSDB_EMBEDDING_POOLING") {
            self.pooling = Some(v.trim().to_ascii_lowercase());
        }
    }

    /// The instruction prefix to prepend to queries.
    pub fn effective_query_prefix(&self) -> String {
        if let Some(p) = &self.query_prefix {
            return p.clone();
        }
        match self.resolved_model() {
            Ok(m) => default_prefixes(&m).0.unwrap_or("").to_string(),
            Err(_) => String::new(),
        }
    }

    /// The instruction prefix to prepend to documents.
    pub fn effective_document_prefix(&self) -> String {
        if let Some(p) = &self.document_prefix {
            return p.clone();
        }
        match self.resolved_model() {
            Ok(m) => default_prefixes(&m).1.unwrap_or("").to_string(),
            Err(_) => String::new(),
        }
    }

    /// Whether this config can produce embeddings at all.
    pub fn is_enabled(&self) -> bool {
        self.resolved_provider() != EmbeddingProvider::None
    }

    /// Stable identity persisted in table metadata. Local models use the
    /// canonical Hugging Face id (falling back to the fastembed variant name);
    /// HTTP models use the configured model string verbatim.
    pub fn canonical_model_id(&self) -> Result<String> {
        match self.resolved_provider() {
            EmbeddingProvider::Http | EmbeddingProvider::None => Ok(self.model.trim().to_string()),
            EmbeddingProvider::Local => {
                let model = self.resolved_model()?;
                Ok(huggingface_id(&model)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{model:?}")))
            }
        }
    }

    /// Best-known vector dimension. HTTP dimensions are inferred for common
    /// OpenAI/Ollama model ids and otherwise learned from the first response.
    pub fn resolved_dim(&self) -> Option<usize> {
        if let Some(dim) = self.dim.filter(|d| *d > 0) {
            return Some(dim);
        }
        match self.resolved_provider() {
            EmbeddingProvider::Local => self.resolved_model().ok().and_then(|model| {
                TextEmbedding::get_model_info(&model)
                    .ok()
                    .map(|info| info.dim)
            }),
            EmbeddingProvider::Http => known_http_dim(&self.model),
            EmbeddingProvider::None => None,
        }
    }

    /// Resolve a configured Hugging Face id / fastembed variant to a concrete
    /// fastembed model. Returns an actionable error for models outside the
    /// local catalogue so the caller can switch to HTTP or `model_path`.
    pub fn resolved_model(&self) -> Result<EmbeddingModel> {
        let raw = self.model.trim();
        if raw.is_empty() {
            return Err(anyhow!(
                "embedding.model is empty; set it to a Hugging Face id such as {DEFAULT_EMBEDDING_MODEL}"
            ));
        }

        // Accept `hf:owner/model` as well as `owner/model`.
        let raw = raw.strip_prefix("hf:").unwrap_or(raw);

        // 1. Exact fastembed variant name (honour it as-is: an explicit
        //    `...Q` model is already the user's choice).
        if let Ok(model) = raw.parse::<EmbeddingModel>() {
            return Ok(model);
        }

        // 2. Common Hugging Face ids.
        let key = raw.to_ascii_lowercase();
        let base = match key.as_str() {
            "baai/bge-small-en-v1.5" | "bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "baai/bge-base-en-v1.5" | "bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "baai/bge-large-en-v1.5" | "bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
            "baai/bge-small-zh-v1.5" | "bge-small-zh-v1.5" => EmbeddingModel::BGESmallZHV15,
            "baai/bge-large-zh-v1.5" | "bge-large-zh-v1.5" => EmbeddingModel::BGELargeZHV15,
            "baai/bge-m3" | "bge-m3" => EmbeddingModel::BGEM3,

            "sentence-transformers/all-minilm-l6-v2" | "all-minilm-l6-v2" => {
                EmbeddingModel::AllMiniLML6V2
            }
            "sentence-transformers/all-minilm-l12-v2" | "all-minilm-l12-v2" => {
                EmbeddingModel::AllMiniLML12V2
            }
            "sentence-transformers/all-mpnet-base-v2" | "all-mpnet-base-v2" => {
                EmbeddingModel::AllMpnetBaseV2
            }
            "sentence-transformers/paraphrase-multilingual-minilm-l12-v2" => {
                EmbeddingModel::ParaphraseMLMiniLML12V2
            }
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" => {
                EmbeddingModel::ParaphraseMLMpnetBaseV2
            }

            "intfloat/multilingual-e5-small" | "multilingual-e5-small" => {
                EmbeddingModel::MultilingualE5Small
            }
            "intfloat/multilingual-e5-base" | "multilingual-e5-base" => {
                EmbeddingModel::MultilingualE5Base
            }
            "intfloat/multilingual-e5-large" | "multilingual-e5-large" => {
                EmbeddingModel::MultilingualE5Large
            }

            "nomic-ai/nomic-embed-text-v1" | "nomic-embed-text-v1" => {
                EmbeddingModel::NomicEmbedTextV1
            }
            "nomic-ai/nomic-embed-text-v1.5" | "nomic-embed-text-v1.5" => {
                EmbeddingModel::NomicEmbedTextV15
            }

            "mixedbread-ai/mxbai-embed-large-v1" | "mxbai-embed-large-v1" => {
                EmbeddingModel::MxbaiEmbedLargeV1
            }
            "alibaba-nlp/gte-base-en-v1.5" | "gte-base-en-v1.5" => EmbeddingModel::GTEBaseENV15,
            "alibaba-nlp/gte-large-en-v1.5" | "gte-large-en-v1.5" => EmbeddingModel::GTELargeENV15,

            "jinaai/jina-embeddings-v2-base-code" | "jina-embeddings-v2-base-code" => {
                EmbeddingModel::JinaEmbeddingsV2BaseCode
            }
            "jinaai/jina-embeddings-v2-base-en" | "jina-embeddings-v2-base-en" => {
                EmbeddingModel::JinaEmbeddingsV2BaseEN
            }

            "snowflake/snowflake-arctic-embed-xs" => EmbeddingModel::SnowflakeArcticEmbedXS,
            "snowflake/snowflake-arctic-embed-s" => EmbeddingModel::SnowflakeArcticEmbedS,
            "snowflake/snowflake-arctic-embed-m" => EmbeddingModel::SnowflakeArcticEmbedM,
            "snowflake/snowflake-arctic-embed-m-long" => EmbeddingModel::SnowflakeArcticEmbedMLong,
            "snowflake/snowflake-arctic-embed-l" => EmbeddingModel::SnowflakeArcticEmbedL,

            _ => {
                return Err(anyhow!(
                    "embedding model '{}' is not in the local fastembed catalogue. \
                     Set embedding.provider = \"http\" with embedding.url for a TEI/Ollama/OpenAI-compatible \
                     endpoint, or set embedding.model_path to a local directory containing model.onnx and tokenizer.json.",
                    self.model
                ));
            }
        };

        Ok(if self.quantized {
            quantized_variant(base.clone()).unwrap_or(base)
        } else {
            base
        })
    }

    fn pooling(&self) -> Pooling {
        match self
            .pooling
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("mean") => Pooling::Mean,
            Some("cls") | None => Pooling::Cls,
            Some(other) => {
                tracing::warn!(
                    "Unknown embedding.pooling '{}'; falling back to CLS pooling",
                    other
                );
                Pooling::Cls
            }
        }
    }
}

/// All embedding models available for in-process inference, independent of any
/// LLM/HTTP endpoint. `model` may be set to either `huggingface_id` or
/// `fastembed_name`.
pub fn list_local_models() -> Vec<Value> {
    TextEmbedding::list_supported_models()
        .into_iter()
        .map(|info| {
            json!({
                "huggingface_id": huggingface_id(&info.model),
                "fastembed_name": format!("{:?}", info.model),
                "dim": info.dim,
                "description": info.description,
                "source": info.model_code,
            })
        })
        .collect()
}

/// Status/inspection payload for the `list_embedding_models` MCP tool.
pub fn embedding_status(
    config: &EmbeddingConfig,
    loaded: bool,
    active_dim: Option<usize>,
) -> Value {
    let provider = config.resolved_provider();
    let resolved_model = config.resolved_model().ok();
    let resolved_name = resolved_model.as_ref().map(|m| format!("{:?}", m));
    let dim = active_dim.or(config.dim).or_else(|| {
        resolved_model
            .as_ref()
            .and_then(|m| TextEmbedding::get_model_info(m).ok())
            .map(|info| info.dim)
    });

    json!({
        "provider": provider.as_str(),
        "model": config.model.clone(),
        "resolved_model": resolved_name,
        "dim": dim,
        "quantized": config.quantized,
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
        "providers": ["local", "http", "none"],
        "local_models": list_local_models(),
    })
}

/// Canonical Hugging Face id for the models we expose by default. `None` means
/// the fastembed variant name is the canonical selector.
fn huggingface_id(model: &EmbeddingModel) -> Option<&'static str> {
    use EmbeddingModel::*;
    Some(match model {
        BGESmallENV15 | BGESmallENV15Q => "BAAI/bge-small-en-v1.5",
        BGEBaseENV15 | BGEBaseENV15Q => "BAAI/bge-base-en-v1.5",
        BGELargeENV15 | BGELargeENV15Q => "BAAI/bge-large-en-v1.5",
        BGESmallZHV15 => "BAAI/bge-small-zh-v1.5",
        BGELargeZHV15 => "BAAI/bge-large-zh-v1.5",
        BGEM3 => "BAAI/bge-m3",
        AllMiniLML6V2 | AllMiniLML6V2Q => "sentence-transformers/all-MiniLM-L6-v2",
        AllMiniLML12V2 | AllMiniLML12V2Q => "sentence-transformers/all-MiniLM-L12-v2",
        AllMpnetBaseV2 => "sentence-transformers/all-mpnet-base-v2",
        ParaphraseMLMiniLML12V2 | ParaphraseMLMiniLML12V2Q => {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
        ParaphraseMLMpnetBaseV2 => "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        MultilingualE5Small => "intfloat/multilingual-e5-small",
        MultilingualE5Base => "intfloat/multilingual-e5-base",
        MultilingualE5Large => "intfloat/multilingual-e5-large",
        NomicEmbedTextV1 => "nomic-ai/nomic-embed-text-v1",
        NomicEmbedTextV15 | NomicEmbedTextV15Q => "nomic-ai/nomic-embed-text-v1.5",
        MxbaiEmbedLargeV1 | MxbaiEmbedLargeV1Q => "mixedbread-ai/mxbai-embed-large-v1",
        GTEBaseENV15 | GTEBaseENV15Q => "Alibaba-NLP/gte-base-en-v1.5",
        GTELargeENV15 | GTELargeENV15Q => "Alibaba-NLP/gte-large-en-v1.5",
        JinaEmbeddingsV2BaseCode => "jinaai/jina-embeddings-v2-base-code",
        JinaEmbeddingsV2BaseEN => "jinaai/jina-embeddings-v2-base-en",
        SnowflakeArcticEmbedXS | SnowflakeArcticEmbedXSQ => "snowflake/snowflake-arctic-embed-xs",
        SnowflakeArcticEmbedS | SnowflakeArcticEmbedSQ => "snowflake/snowflake-arctic-embed-s",
        SnowflakeArcticEmbedM | SnowflakeArcticEmbedMQ => "snowflake/snowflake-arctic-embed-m",
        SnowflakeArcticEmbedMLong | SnowflakeArcticEmbedMLongQ => {
            "snowflake/snowflake-arctic-embed-m-long"
        }
        SnowflakeArcticEmbedL | SnowflakeArcticEmbedLQ => "snowflake/snowflake-arctic-embed-l",
        _ => return None,
    })
}

fn known_http_dim(model: &str) -> Option<usize> {
    match model.trim().to_ascii_lowercase().as_str() {
        "text-embedding-3-small" => Some(1536),
        "text-embedding-3-large" => Some(3072),
        "text-embedding-ada-002" => Some(1536),
        "nomic-embed-text" => Some(768),
        "all-minilm" => Some(384),
        "mxbai-embed-large" => Some(1024),
        "bge-m3" => Some(1024),
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

/// A loaded local embedding model plus the metadata needed to embed queries and
/// documents consistently.
pub struct LocalEmbedder {
    model: TextEmbedding,
    pub model_id: String,
    pub dim: usize,
    query_prefix: String,
    document_prefix: String,
    batch_size: usize,
}

impl LocalEmbedder {
    /// Load (or download once and cache) a local embedding model. This is a
    /// blocking operation and must be called from `spawn_blocking`.
    pub fn load(config: &EmbeddingConfig) -> Result<Self> {
        if !config.device.eq_ignore_ascii_case("cpu") && !config.device.eq_ignore_ascii_case("auto")
        {
            tracing::warn!(
                "embedding.device='{}' requested, but local fastembed inference is CPU-only in this build",
                config.device
            );
        }

        std::fs::create_dir_all(&config.cache_dir).with_context(|| {
            format!(
                "failed to create embedding model cache directory {}",
                config.cache_dir.display()
            )
        })?;

        let (model, dim) = if let Some(path) = &config.model_path {
            let model = load_user_defined(path, config)?;
            let dim = config
                .dim
                .filter(|d| *d > 0)
                .or_else(|| read_hidden_size(path).ok())
                .ok_or_else(|| {
                    anyhow!(
                        "embedding.dim is required for user-defined model path {} \
                         (could not read hidden_size from config.json)",
                        path.display()
                    )
                })?;
            (model, dim)
        } else {
            let model_name = config.resolved_model()?;
            let info = TextEmbedding::get_model_info(&model_name)
                .map_err(|e| anyhow!("failed to resolve embedding model metadata: {e}"))?;
            let mut options = TextInitOptions::new(model_name.clone())
                .with_cache_dir(config.cache_dir.clone())
                .with_show_download_progress(config.show_download_progress)
                .with_max_length(config.max_length);
            if let Some(threads) = config.intra_threads {
                options = options.with_intra_threads(threads);
            }
            let model = TextEmbedding::try_new(options)
                .map_err(|e| anyhow!("failed to load embedding model '{}': {e}", config.model))?;
            (model, config.dim.filter(|d| *d > 0).unwrap_or(info.dim))
        };

        let (default_query, default_document) = config
            .resolved_model()
            .map(|m| default_prefixes(&m))
            .unwrap_or((None, None));
        let query_prefix = config
            .query_prefix
            .clone()
            .or_else(|| default_query.map(str::to_string))
            .unwrap_or_default();
        let document_prefix = config
            .document_prefix
            .clone()
            .or_else(|| default_document.map(str::to_string))
            .unwrap_or_default();

        Ok(Self {
            model,
            model_id: config.model.clone(),
            dim,
            query_prefix,
            document_prefix,
            batch_size: config.batch_size.max(1),
        })
    }

    /// Embed document chunks. The document prefix (if any) is applied here so
    /// callers cannot accidentally embed documents as queries.
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prepared: Vec<String> = if self.document_prefix.is_empty() {
            texts.to_vec()
        } else {
            texts
                .iter()
                .map(|t| format!("{}{}", self.document_prefix, t))
                .collect()
        };
        self.model
            .embed(prepared, Some(self.batch_size))
            .map_err(|e| anyhow!("local document embedding failed: {e}"))
    }

    /// Embed a single search query with the model's query instruction prefix.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let prepared = format!("{}{}", self.query_prefix, text);
        let mut vectors = self
            .model
            .embed(vec![prepared], Some(1))
            .map_err(|e| anyhow!("local query embedding failed: {e}"))?;
        vectors
            .pop()
            .ok_or_else(|| anyhow!("embedding model returned no vector"))
    }
}

fn load_user_defined(path: &Path, config: &EmbeddingConfig) -> Result<TextEmbedding> {
    let (onnx_path, base_dir) = if path.is_file() {
        (path.to_path_buf(), path.parent().unwrap_or(Path::new(".")))
    } else {
        let candidates = [
            path.join("model.onnx"),
            path.join("onnx/model.onnx"),
            path.join("model_quantized.onnx"),
            path.join("onnx/model_quantized.onnx"),
        ];
        let onnx = candidates
            .into_iter()
            .find(|p| p.is_file())
            .ok_or_else(|| {
                anyhow!(
                    "no ONNX model found under {} (expected model.onnx or onnx/model.onnx)",
                    path.display()
                )
            })?;
        (onnx, path)
    };

    let read = |name: &str| -> Result<Vec<u8>> {
        std::fs::read(base_dir.join(name))
            .with_context(|| format!("failed to read {}", base_dir.join(name).display()))
    };
    let optional = |name: &str| -> Vec<u8> {
        std::fs::read(base_dir.join(name)).unwrap_or_else(|_| b"{}".to_vec())
    };

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: optional("special_tokens_map.json"),
        tokenizer_config_file: optional("tokenizer_config.json"),
    };

    let onnx = std::fs::read(&onnx_path)
        .with_context(|| format!("failed to read ONNX model {}", onnx_path.display()))?;

    let model = UserDefinedEmbeddingModel::new(onnx, tokenizer_files)
        .with_pooling(config.pooling())
        .with_quantization(QuantizationMode::None);

    let options = InitOptionsUserDefined::new().with_max_length(config.max_length);
    TextEmbedding::try_new_from_user_defined(model, options)
        .map_err(|e| anyhow!("failed to load user-defined embedding model: {e}"))
}

fn read_hidden_size(path: &Path) -> Result<usize> {
    let config_path = if path.is_file() {
        path.parent().unwrap_or(Path::new(".")).join("config.json")
    } else {
        path.join("config.json")
    };
    let text = std::fs::read_to_string(&config_path)?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    value
        .get("hidden_size")
        .or_else(|| value.get("d_model"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("hidden_size missing from {}", config_path.display()))
}

/// Query/document instruction prefixes for models that require them.
fn default_prefixes(model: &EmbeddingModel) -> (Option<&'static str>, Option<&'static str>) {
    match model {
        // BGE v1.5 English/Chinese and BGE-M3: query-side instruction only.
        EmbeddingModel::BGESmallENV15
        | EmbeddingModel::BGESmallENV15Q
        | EmbeddingModel::BGEBaseENV15
        | EmbeddingModel::BGEBaseENV15Q
        | EmbeddingModel::BGELargeENV15
        | EmbeddingModel::BGELargeENV15Q
        | EmbeddingModel::BGESmallZHV15
        | EmbeddingModel::BGELargeZHV15
        | EmbeddingModel::BGEM3
        | EmbeddingModel::MxbaiEmbedLargeV1
        | EmbeddingModel::MxbaiEmbedLargeV1Q
        | EmbeddingModel::SnowflakeArcticEmbedXS
        | EmbeddingModel::SnowflakeArcticEmbedXSQ
        | EmbeddingModel::SnowflakeArcticEmbedS
        | EmbeddingModel::SnowflakeArcticEmbedSQ
        | EmbeddingModel::SnowflakeArcticEmbedM
        | EmbeddingModel::SnowflakeArcticEmbedMQ
        | EmbeddingModel::SnowflakeArcticEmbedMLong
        | EmbeddingModel::SnowflakeArcticEmbedMLongQ
        | EmbeddingModel::SnowflakeArcticEmbedL
        | EmbeddingModel::SnowflakeArcticEmbedLQ => (Some(DEFAULT_BGE_QUERY_PREFIX), None),

        // E5 family: asymmetric query/passage prefixes.
        EmbeddingModel::MultilingualE5Small
        | EmbeddingModel::MultilingualE5Base
        | EmbeddingModel::MultilingualE5Large => (Some("query: "), Some("passage: ")),

        // Nomic v1/v1.5: task prefixes on both sides.
        EmbeddingModel::NomicEmbedTextV1
        | EmbeddingModel::NomicEmbedTextV15
        | EmbeddingModel::NomicEmbedTextV15Q => (Some("search_query: "), Some("search_document: ")),

        // Symmetric similarity / code models: no instruction prefix.
        _ => (None, None),
    }
}

fn quantized_variant(model: EmbeddingModel) -> Option<EmbeddingModel> {
    use EmbeddingModel::*;
    Some(match model {
        BGESmallENV15 => BGESmallENV15Q,
        BGEBaseENV15 => BGEBaseENV15Q,
        BGELargeENV15 => BGELargeENV15Q,
        AllMiniLML6V2 => AllMiniLML6V2Q,
        AllMiniLML12V2 => AllMiniLML12V2Q,
        NomicEmbedTextV15 => NomicEmbedTextV15Q,
        ParaphraseMLMiniLML12V2 => ParaphraseMLMiniLML12V2Q,
        MxbaiEmbedLargeV1 => MxbaiEmbedLargeV1Q,
        GTEBaseENV15 => GTEBaseENV15Q,
        GTELargeENV15 => GTELargeENV15Q,
        SnowflakeArcticEmbedXS => SnowflakeArcticEmbedXSQ,
        SnowflakeArcticEmbedS => SnowflakeArcticEmbedSQ,
        SnowflakeArcticEmbedM => SnowflakeArcticEmbedMQ,
        SnowflakeArcticEmbedMLong => SnowflakeArcticEmbedMLongQ,
        SnowflakeArcticEmbedL => SnowflakeArcticEmbedLQ,
        _ => return None,
    })
}

fn parse_provider(value: &str) -> Option<EmbeddingProvider> {
    match value.trim().to_ascii_lowercase().as_str() {
        "local" | "fastembed" | "onnx" => Some(EmbeddingProvider::Local),
        "http" | "remote" | "ollama" | "openai" | "tei" => Some(EmbeddingProvider::Http),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_bge_small_and_local() {
        let config = EmbeddingConfig {
            url: None,
            ..Default::default()
        };
        assert_eq!(config.model, DEFAULT_EMBEDDING_MODEL);
        assert!(config.quantized);
        assert_eq!(config.resolved_provider(), EmbeddingProvider::Local);
        assert_eq!(
            config
                .resolved_model()
                .unwrap_or_else(|e| panic!("default model should resolve: {e}")),
            EmbeddingModel::BGESmallENV15Q
        );
    }

    #[test]
    fn bge_query_prefix_is_applied_only_to_queries() {
        let config = EmbeddingConfig::default();
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
    fn mini_lm_is_selectable_but_not_default() {
        let config = EmbeddingConfig {
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            quantized: false,
            ..Default::default()
        };
        assert_eq!(
            config
                .resolved_model()
                .unwrap_or_else(|e| panic!("MiniLM should resolve: {e}")),
            EmbeddingModel::AllMiniLML6V2
        );
        assert_eq!(config.effective_query_prefix(), "");
    }

    #[test]
    fn huggingface_ids_resolve_to_expected_models_and_dims() {
        let cases = [
            (
                "BAAI/bge-small-en-v1.5",
                EmbeddingModel::BGESmallENV15Q,
                384,
            ),
            ("BAAI/bge-base-en-v1.5", EmbeddingModel::BGEBaseENV15Q, 768),
            (
                "BAAI/bge-large-en-v1.5",
                EmbeddingModel::BGELargeENV15Q,
                1024,
            ),
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                EmbeddingModel::AllMiniLML6V2Q,
                384,
            ),
            (
                "intfloat/multilingual-e5-small",
                EmbeddingModel::MultilingualE5Small,
                384,
            ),
            (
                "nomic-ai/nomic-embed-text-v1.5",
                EmbeddingModel::NomicEmbedTextV15Q,
                768,
            ),
            (
                "jinaai/jina-embeddings-v2-base-code",
                EmbeddingModel::JinaEmbeddingsV2BaseCode,
                768,
            ),
        ];

        for (id, expected, dim) in cases {
            let config = EmbeddingConfig {
                model: id.to_string(),
                quantized: true,
                ..Default::default()
            };
            let model = config
                .resolved_model()
                .unwrap_or_else(|e| panic!("{id} should resolve: {e}"));
            assert_eq!(model, expected, "unexpected model for {id}");
            let info = TextEmbedding::get_model_info(&model)
                .unwrap_or_else(|e| panic!("{id} should have model info: {e}"));
            assert_eq!(info.dim, dim, "unexpected dimension for {id}");
        }
    }

    #[test]
    fn quantized_flag_can_be_disabled_for_local_models() {
        let config = EmbeddingConfig {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            quantized: false,
            ..Default::default()
        };
        assert_eq!(
            config
                .resolved_model()
                .unwrap_or_else(|e| panic!("non-quantized model should resolve: {e}")),
            EmbeddingModel::BGESmallENV15
        );
    }

    /// End-to-end local model check. Ignored by default because the first run
    /// downloads the quantized ONNX model into the private cache directory.
    #[test]
    #[ignore = "downloads the default ONNX model on first run"]
    fn local_embedder_loads_default_model_and_embeds() {
        let config = EmbeddingConfig {
            show_download_progress: false,
            ..Default::default()
        };
        let mut embedder =
            LocalEmbedder::load(&config).unwrap_or_else(|e| panic!("load local embedder: {e}"));
        assert_eq!(embedder.dim, 384);
        assert!(config.cache_dir.is_dir(), "model cache dir must exist");

        let docs = embedder
            .embed_documents(&["BenoStreamDB indexes Parquet files in place.".to_string()])
            .unwrap_or_else(|e| panic!("embed document: {e}"));
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].len(), 384);
        let norm = docs[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "document embedding must be L2-normalized"
        );

        let query = embedder
            .embed_query("vector search")
            .unwrap_or_else(|e| panic!("embed query: {e}"));
        assert_eq!(query.len(), 384);
    }
}
