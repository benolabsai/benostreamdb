// Copyright (c) 2026 Richard Albright. All rights reserved.

//! English analyzer: standard tokenization with English stop-word removal.
//!
//! Mirrors Lucene's basic `english` analyzer semantics closely enough for
//! keyword/BM25 search: alphanumeric tokenization (lowercased) followed by a
//! small stop-word filter.

use super::tokenizer::Tokenizer;

/// Lucene "english" analyzer stop-word list (basic English set).
///
/// Kept in alphabetical order; membership is checked via binary search.
pub const ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "nor", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Returns true if the (already lowercased) token is an English stop word.
pub fn is_english_stop_word(token: &str) -> bool {
    ENGLISH_STOP_WORDS.binary_search(&token).is_ok()
}

/// Standard tokenizer plus English stop-word removal.
#[derive(Debug, Clone, Default)]
pub struct EnglishTokenizer;

impl Tokenizer for EnglishTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .filter(|t| !is_english_stop_word(t))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_lowercased_alphanumeric() {
        let tokens = EnglishTokenizer.tokenize("Hello, World! 123");
        assert_eq!(tokens, vec!["hello", "world", "123"]);
    }

    #[test]
    fn removes_stop_words() {
        let tokens = EnglishTokenizer.tokenize("the quick brown fox jumps over the lazy dog");
        assert_eq!(
            tokens,
            vec!["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        );
    }

    #[test]
    fn stop_word_removal_is_case_insensitive() {
        let tokens = EnglishTokenizer.tokenize("The THE the The");
        assert!(tokens.is_empty());
    }

    #[test]
    fn punctuation_only_input_yields_no_tokens() {
        let tokens = EnglishTokenizer.tokenize("!@#$%^&*()");
        assert!(tokens.is_empty());
    }

    #[test]
    fn stop_word_check_is_lowercased() {
        assert!(is_english_stop_word("the"));
        assert!(!is_english_stop_word("therefore"));
        assert!(!is_english_stop_word("fox"));
    }

    #[test]
    fn numeric_and_mixed_tokens_survive() {
        let tokens = EnglishTokenizer.tokenize("the 42 and value-42 is");
        assert_eq!(tokens, vec!["42", "value", "42"]);
    }
}
