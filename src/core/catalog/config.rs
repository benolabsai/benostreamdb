// Copyright (c) 2026 Richard Albright. All rights reserved.

use serde::Deserialize;
use std::collections::HashMap;
use anyhow::{Result, Context};
use std::fs;
use super::CatalogType;

#[derive(Debug, Deserialize)]
pub struct CatalogConfig {
    pub catalog_type: CatalogType,
    pub config: HashMap<String, String>,
}

impl CatalogConfig {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read catalog config file: {}", path))?;
        
        // Use toml to deserialize
        let config: CatalogConfig = toml::from_str(&content)
            .with_context(|| format!("Failed to parse catalog config file: {}", path))?;
            
        Ok(config)
    }

    pub fn load_default() -> Result<Self> {
        // 1. Check environment variable
        if let Ok(path) = std::env::var("HYPERSTREAM_CONFIG") {
            if fs::metadata(&path).is_ok() {
                return Self::load_from_file(&path);
            }
        }

        // 2. Check current directory
        if fs::metadata("hyperstream.toml").is_ok() {
            return Self::load_from_file("hyperstream.toml");
        }

        // 3. Check home directory
        if let Some(mut home) = dirs::home_dir() {
            home.push(".hyperstream");
            home.push("config.toml");
            if home.exists() {
                return Self::load_from_file(home.to_str().unwrap());
            }
        }

        // 4. Fallback/Error
        anyhow::bail!("No configuration file found. Checked ENV 'HYPERSTREAM_CONFIG', ./hyperstream.toml, and ~/.hyperstream/config.toml")
    }
}
