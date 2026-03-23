use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NessieConfig {
    pub uri: String,
    pub ref_name: String, // branch or tag, defaults to "main"
    pub auth_type: Option<String>, // "bearer" or none
    pub auth_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NessieClient {
    client: reqwest::Client,
    config: NessieConfig,
}

impl NessieClient {
    pub fn new(config: NessieConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    /// Resolve a reference (branch/tag) to a hash
    pub async fn resolve_reference(&self, ref_name: &str) -> Result<String> {
        let url = format!("{}/api/v2/trees/{}", self.config.uri, ref_name);
        
        let mut builder = self.client.get(&url);
        if let Some(token) = &self.config.auth_token {
             builder = builder.header("Authorization", format!("Bearer {}", token));
        }

        let resp = builder.send().await.context("Failed to connect to Nessie")?;
        
        if !resp.status().is_success() {
             let status = resp.status();
             let text = resp.text().await.unwrap_or_default();
             anyhow::bail!("Nessie request failed: {} - {}", status, text);
        }

        let json: serde_json::Value = resp.json().await.context("Failed to parse Nessie response")?;
        
        // Extract hash
        if let Some(hash) = json.get("hash").and_then(|v| v.as_str()) {
            Ok(hash.to_string())
        } else {
            anyhow::bail!("Nessie response missing 'hash' field: {}", json);
        }
    }

    /// Load a table's metadata location from Nessie for a specific snapshot/hash
    pub async fn load_table_metadata(&self, namespace: String, table: String, ref_hash: Option<String>) -> Result<String> {
        // Use provided hash or default to config ref
        let reference = if let Some(h) = &ref_hash { h } else { &self.config.ref_name };
        
        // GET /api/v2/trees/{ref}/contents?key={namespace}.{table}
        // Note: Key format depends on Nessie version, but typically dotted.
        // If namespace has dots, might need escaping. Assuming simple for now.
        let key = if namespace.is_empty() {
             table.to_string()
        } else {
             format!("{}.{}", namespace, table)
        };
        
        let url = format!("{}/api/v2/trees/{}/contents", self.config.uri, reference);
        
        let mut builder = self.client.get(&url)
            .query(&[("key", &key)]);

        if let Some(token) = &self.config.auth_token {
             builder = builder.header("Authorization", format!("Bearer {}", token));
        }

        let resp = builder.send().await.context("Failed to fetch table content from Nessie")?;
             
        if !resp.status().is_success() {
             anyhow::bail!("Nessie request failed: {}", resp.status());
        }

        let json: serde_json::Value = resp.json().await?;
        
        // Response is a map of Key -> Content
        // We expect one entry
        if let Some(content) = json.get("contents").and_then(|c| c.as_array()).and_then(|a| a.first()) {
             if let Some(meta_loc) = content.get("content").and_then(|c| c.get("metadataLocation")).and_then(|v| v.as_str()) {
                 return Ok(meta_loc.to_string());
             }
        }
        
        anyhow::bail!("Table {} not found or missing metadataLocation in Nessie response", key);
    }
}
