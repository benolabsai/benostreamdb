// Copyright (c) 2026 Richard Albright. All rights reserved.

use super::CatalogType;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

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
        if let Ok(path) = std::env::var("BENOSTREAM_CONFIG") {
            if fs::metadata(&path).is_ok() {
                return Self::load_from_file(&path);
            }
        }

        // 2. Check current directory
        if fs::metadata("benostream.toml").is_ok() {
            return Self::load_from_file("benostream.toml");
        }

        // 3. Check home directory
        if let Some(mut home) = dirs::home_dir() {
            home.push(".benostream");
            home.push("config.toml");
            if home.exists() {
                let home_str = home.to_str().ok_or_else(|| {
                    anyhow::anyhow!("~/.benostream/config.toml path is not valid UTF-8")
                })?;
                return Self::load_from_file(home_str);
            }
        }

        // 4. Fallback/Error
        anyhow::bail!("No configuration file found. Checked ENV 'BENOSTREAM_CONFIG', ./benostream.toml, and ~/.benostream/config.toml")
    }
}
