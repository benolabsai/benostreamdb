// Copyright (c) 2026 Richard Albright. All rights reserved.
// Portions Copyright The Apache Software Foundation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Configuration parameters for vector search operations.
//! Adapted from Apache Iceberg Rust project (v0.9.0+)

use datafusion::common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion::config::ConfigOptions;
use datafusion::error::Result;

/// Configuration parameters for vector search operations
/// Adapted from Apache Iceberg Rust project (v0.9.0+)
#[derive(Debug, Clone)]
pub struct VectorSearchConfig {
    /// HNSW search beam width (ef_search parameter)
    pub ef_search: Option<usize>,
    /// Number of IVF clusters to search (probes parameter)
    pub probes: Option<usize>,
    /// Whether to use vector indexes (default: true)
    pub use_index: bool,
    /// Enable LIMIT pushdown optimization (Iceberg pattern)
    pub limit_pushdown: bool,
    /// Enable row group skipping based on statistics
    pub skip_row_groups: bool,
    /// Cache manifest metadata for repeated queries
    pub cache_manifests: bool,
    /// Enable single-threaded fast path for small result sets
    pub fast_path: bool,
}

impl ExtensionOptions for VectorSearchConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> datafusion::common::Result<()> {
        match key {
            "ef_search" => {
                self.ef_search = Some(value.parse::<usize>().map_err(|e| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "Invalid ef_search value: {}",
                        e
                    ))
                })?);
            }
            "probes" => {
                self.probes = Some(value.parse::<usize>().map_err(|e| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "Invalid probes value: {}",
                        e
                    ))
                })?);
            }
            "use_index" => {
                self.use_index = value.parse::<bool>().map_err(|e| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "Invalid use_index value: {}",
                        e
                    ))
                })?;
            }
            "limit_pushdown" => {
                self.limit_pushdown = value.parse::<bool>().map_err(|e| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "Invalid limit_pushdown value: {}",
                        e
                    ))
                })?;
            }
            "skip_row_groups" => {
                self.skip_row_groups = value.parse::<bool>().map_err(|e| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "Invalid skip_row_groups value: {}",
                        e
                    ))
                })?;
            }
            "cache_manifests" => {
                self.cache_manifests = value.parse::<bool>().map_err(|e| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "Invalid cache_manifests value: {}",
                        e
                    ))
                })?;
            }
            "fast_path" => {
                self.fast_path = value.parse::<bool>().map_err(|e| {
                    datafusion::common::DataFusionError::Configuration(format!(
                        "Invalid fast_path value: {}",
                        e
                    ))
                })?;
            }
            _ => {
                return Err(datafusion::common::DataFusionError::Configuration(format!(
                    "Unknown configuration key: {}",
                    key
                )));
            }
        }
        Ok(())
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![
            ConfigEntry {
                key: "ef_search".to_string(),
                value: self.ef_search.map(|v| v.to_string()),
                description: "HNSW search beam width",
            },
            ConfigEntry {
                key: "probes".to_string(),
                value: self.probes.map(|v| v.to_string()),
                description: "Number of IVF clusters to search",
            },
            ConfigEntry {
                key: "use_index".to_string(),
                value: Some(self.use_index.to_string()),
                description: "Whether to use vector indexes",
            },
            ConfigEntry {
                key: "limit_pushdown".to_string(),
                value: Some(self.limit_pushdown.to_string()),
                description: "Enable LIMIT pushdown optimization",
            },
            ConfigEntry {
                key: "skip_row_groups".to_string(),
                value: Some(self.skip_row_groups.to_string()),
                description: "Enable row group skipping based on statistics",
            },
            ConfigEntry {
                key: "cache_manifests".to_string(),
                value: Some(self.cache_manifests.to_string()),
                description: "Cache manifest metadata for repeated queries",
            },
            ConfigEntry {
                key: "fast_path".to_string(),
                value: Some(self.fast_path.to_string()),
                description: "Enable single-threaded fast path for small result sets",
            },
        ]
    }
}

impl ConfigExtension for VectorSearchConfig {
    const PREFIX: &'static str = "benostreamdb";
}

impl VectorSearchConfig {
    /// Create a new VectorSearchConfig with default values
    pub fn new() -> Self {
        Self {
            ef_search: None,
            probes: None,
            use_index: true,
            limit_pushdown: true,  // Enable by default
            skip_row_groups: true, // Enable by default
            cache_manifests: true, // Enable by default
            fast_path: true,       // Enable by default
        }
    }

    /// Read configuration from DataFusion session config
    pub fn from_session_config(config: &ConfigOptions) -> Self {
        // Try to read from registered extensions first
        if let Some(ext_config) = config.extensions.get::<VectorSearchConfig>() {
            return ext_config.clone();
        }

        // Fallback: return defaults
        // Users can register via:
        //   config.options.extensions.insert(VectorSearchConfig::new());
        //   session.config_options().set("benostreamdb.ef_search", "128").unwrap();
        Self::new()
    }

    /// Extract the body of a SQL optimizer hint comment from a query.
    ///
    /// Recognizes the standard `/*+ ... */` form and returns the inner text
    /// (trimmed). Returns `None` when the query contains no hint comment.
    pub fn extract_sql_hints(query: &str) -> Option<String> {
        let start = query.find("/*+")?;
        let rest = &query[start + 3..];
        let end = rest.find("*/")?;
        let body = rest[..end].trim();
        if body.is_empty() {
            None
        } else {
            Some(body.to_string())
        }
    }

    /// Parse configuration from SQL hints.
    ///
    /// Accepts either the bare key/value list or the wrapped form:
    ///   `/*+ INDEX_HINT(ef_search=128, probes=10) */`
    ///   `/*+ ef_search=128, probes=10 */`
    ///
    /// Unknown keys are ignored; malformed values for known keys are an error.
    /// Parsing delegates to [`ExtensionOptions::set`] so the hint path and the
    /// session-config path stay in lock-step.
    pub fn from_sql_hints(hints: &str) -> Result<Self> {
        let mut config = Self::new();

        // Strip an optional `NAME(...)` wrapper, e.g. `INDEX_HINT(...)`.
        let body = hints.trim();
        let body = match (body.find('('), body.ends_with(')')) {
            (Some(open), true) => &body[open + 1..body.len() - 1],
            _ => body,
        };

        let known_keys: Vec<String> = config.entries().into_iter().map(|e| e.key).collect();

        for part in body.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (key, value) = match part.split_once('=') {
                Some(kv) => kv,
                None => continue,
            };
            let key = key.trim();
            let value = value.trim();
            if !known_keys.iter().any(|k| k == key) {
                // Ignore hints that belong to other extensions.
                continue;
            }
            config.set(key, value)?;
        }

        Ok(config)
    }

    /// Enable manifest caching for better performance on repeated queries (Iceberg v0.4.0+)
    pub fn with_manifest_caching(mut self, enable: bool) -> Self {
        self.cache_manifests = enable;
        self
    }

    /// Enable row group skipping to reduce I/O (Iceberg v0.4.0+)
    pub fn with_row_group_skipping(mut self, enable: bool) -> Self {
        self.skip_row_groups = enable;
        self
    }

    /// Enable fast path for single-threaded execution on small result sets (Iceberg v0.9.0+)
    pub fn with_fast_path(mut self, enable: bool) -> Self {
        self.fast_path = enable;
        self
    }
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod hint_tests {
    use super::*;

    #[test]
    fn extracts_hint_body_from_query() {
        let q = "SELECT * FROM t /*+ INDEX_HINT(ef_search=128, probes=10) */ ORDER BY v <-> '[1,2]' LIMIT 5";
        let hints = VectorSearchConfig::extract_sql_hints(q).unwrap();
        assert_eq!(hints, "INDEX_HINT(ef_search=128, probes=10)");
    }

    #[test]
    fn no_hint_returns_none() {
        assert!(VectorSearchConfig::extract_sql_hints("SELECT 1").is_none());
        assert!(VectorSearchConfig::extract_sql_hints("SELECT 1 /*+ */").is_none());
    }

    #[test]
    fn parses_wrapped_hint() {
        let cfg =
            VectorSearchConfig::from_sql_hints("INDEX_HINT(ef_search=128, probes=10)").unwrap();
        assert_eq!(cfg.ef_search, Some(128));
        assert_eq!(cfg.probes, Some(10));
    }

    #[test]
    fn parses_bare_hint_and_all_keys() {
        let cfg = VectorSearchConfig::from_sql_hints(
            "ef_search=64, probes=4, use_index=false, limit_pushdown=false, \
             skip_row_groups=false, cache_manifests=false, fast_path=false",
        )
        .unwrap();
        assert_eq!(cfg.ef_search, Some(64));
        assert_eq!(cfg.probes, Some(4));
        assert!(!cfg.use_index);
        assert!(!cfg.limit_pushdown);
        assert!(!cfg.skip_row_groups);
        assert!(!cfg.cache_manifests);
        assert!(!cfg.fast_path);
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let cfg = VectorSearchConfig::from_sql_hints("ef_search=32, other_ext=foo").unwrap();
        assert_eq!(cfg.ef_search, Some(32));
    }

    #[test]
    fn malformed_known_value_is_an_error() {
        assert!(VectorSearchConfig::from_sql_hints("ef_search=not_a_number").is_err());
    }

    #[test]
    fn session_config_round_trips_hint_config() {
        let cfg = VectorSearchConfig::from_sql_hints("ef_search=256").unwrap();
        let mut session_config = datafusion::prelude::SessionConfig::new();
        session_config.options_mut().extensions.insert(cfg);
        let read_back = VectorSearchConfig::from_session_config(session_config.options());
        assert_eq!(read_back.ef_search, Some(256));
    }
}
