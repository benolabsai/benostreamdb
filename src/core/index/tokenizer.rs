// Copyright (c) 2026 Richard Albright. All rights reserved.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for text tokenization (used by Inverted Index)
pub trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Simple whitespace-based tokenizer
pub struct WhitespaceTokenizer;
impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_lowercase()).collect()
    }
}

/// Standard tokenizer (splits on non-alphanumeric, removes punctuation)
pub struct StandardTokenizer;
impl Tokenizer for StandardTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// Identity tokenizer (returns the whole string, used for category/exact matching)
pub struct IdentityTokenizer;
impl Tokenizer for IdentityTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        vec![text.to_string()]
    }
}

/// Registry for tokenizers (to allow user configuration)
pub struct TokenizerRegistry {
    tokenizers: HashMap<String, Arc<dyn Tokenizer>>,
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerRegistry {
    pub fn new() -> Self {
        let mut tr = Self {
            tokenizers: HashMap::new(),
        };
        tr.register("whitespace", Arc::new(WhitespaceTokenizer));
        tr.register("standard", Arc::new(StandardTokenizer));
        tr.register("identity", Arc::new(IdentityTokenizer));
        tr
    }

    pub fn register(&mut self, name: &str, tokenizer: Arc<dyn Tokenizer>) {
        self.tokenizers.insert(name.to_string(), tokenizer);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizers.get(name).cloned()
    }
}

lazy_static::lazy_static! {
    pub static ref GLOBAL_TOKENIZER_REGISTRY: TokenizerRegistry = TokenizerRegistry::new();
}

/// Configuration for indexing a specific column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub column: String,
    pub tokenizer: Option<String>, // If None, no indexing (unless it's a primary key)
    pub enabled: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            column: "".to_string(),
            tokenizer: None, // NO indexing by default (saves ingestion time!)
            enabled: false,
        }
    }
}
