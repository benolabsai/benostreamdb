// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::Utc;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

/// Master Public Key for BenoStreamDB Enterprise
/// Loaded from the `BSDB_LICENSE_PUBLIC_KEY` environment variable for security.
static MASTER_PUBLIC_KEY: Lazy<Option<VerifyingKey>> = Lazy::new(|| {
    let hex_str = std::env::var("BSDB_LICENSE_PUBLIC_KEY").ok()?;

    let bytes = hex::decode(hex_str.trim()).ok()?;

    let bytes_arr: [u8; 32] = bytes.as_slice().try_into().ok()?;
    VerifyingKey::from_bytes(&bytes_arr).ok()
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicensePayload {
    pub customer_id: String,
    pub expiry_timestamp: i64,
    pub features: Vec<String>,
}

impl LicensePayload {
    pub fn is_feature_enabled(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature) && !self.is_expired()
    }

    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() > self.expiry_timestamp
    }
}

pub fn verify_license(key: &str) -> Result<LicensePayload> {
    // DEVELOPER OVERRIDE: For local testing without public keys,
    // allow keys starting with HSDB-TEST- as valid mock licenses.
    if key.starts_with("HSDB-TEST-") {
        return Ok(LicensePayload {
            customer_id: "test-user".to_string(),
            expiry_timestamp: Utc::now().timestamp() + 3600,
            features: vec!["continuous_indexing".to_string()],
        });
    }

    // Format: base64(json_payload).base64(signature)
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid license key format. Expected 'payload.signature'");
    }

    let payload_bytes = BASE64
        .decode(parts[0])
        .context("Failed to decode license payload")?;
    let signature_bytes = BASE64
        .decode(parts[1])
        .context("Failed to decode license signature")?;

    let signature = Signature::from_slice(&signature_bytes).context("Invalid signature length")?;

    // Verify signature using the environment-loaded Public Key
    let public_key = MASTER_PUBLIC_KEY.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Enterprise License Public Key not found. Please set BSDB_LICENSE_PUBLIC_KEY environment variable."))?;

    public_key
        .verify(&payload_bytes, &signature)
        .context("License signature verification failed")?;

    let payload: LicensePayload =
        serde_json::from_slice(&payload_bytes).context("Failed to parse license payload")?;

    if payload.is_expired() {
        let expiry = chrono::DateTime::<Utc>::from_timestamp(payload.expiry_timestamp, 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        anyhow::bail!("This license key expired on {}", expiry);
    }

    Ok(payload)
}
