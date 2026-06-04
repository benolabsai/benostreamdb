// Copyright (c) 2026 Richard Albright. All rights reserved.

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
    const PREFIX: &'static str = "hyperstreamdb";
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
        //   session.config_options().set("hyperstreamdb.ef_search", "128").unwrap();
        Self::new()
    }

    /// Parse configuration from SQL hints (future extension)
    /// Format: /*+ INDEX_HINT(ef_search=128, probes=10) */
    pub fn from_sql_hints(hints: &str) -> Result<Self> {
        let mut config = Self::new();

        // Simple parsing for MVP - look for key=value pairs
        for part in hints.split(',') {
            let part = part.trim();
            if let Some((key, value)) = part.split_once('=') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "ef_search" => {
                        if let Ok(ef) = value.parse::<usize>() {
                            config.ef_search = Some(ef);
                        }
                    }
                    "probes" => {
                        if let Ok(probes) = value.parse::<usize>() {
                            config.probes = Some(probes);
                        }
                    }
                    "use_index" => {
                        if let Ok(use_idx) = value.parse::<bool>() {
                            config.use_index = use_idx;
                        }
                    }
                    _ => {} // Ignore unknown parameters
                }
            }
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
