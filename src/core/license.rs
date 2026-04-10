// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;
use ed25519_dalek::{VerifyingKey, Signature, Verifier};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chrono::Utc;

/// Master Public Key for HyperStreamDB Enterprise
/// Loaded from the `HDB_LICENSE_PUBLIC_KEY` environment variable for security.
static MASTER_PUBLIC_KEY: Lazy<VerifyingKey> = Lazy::new(|| {
    let hex_str = std::env::var("HDB_LICENSE_PUBLIC_KEY")
        .expect("CRITICAL: HDB_LICENSE_PUBLIC_KEY environment variable is required for Enterprise features");
    
    let bytes = hex::decode(hex_str.trim())
        .expect("CRITICAL: Invalid HDB_LICENSE_PUBLIC_KEY hex format");
    
    VerifyingKey::from_bytes(bytes.as_slice().try_into().expect("Invalid public key length"))
        .expect("CRITICAL: Failed to construct VerifyingKey from environment bytes")
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
    // Format: base64(json_payload).base64(signature)
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid license key format. Expected 'payload.signature'");
    }

    let payload_bytes = BASE64.decode(parts[0])
        .context("Failed to decode license payload")?;
    let signature_bytes = BASE64.decode(parts[1])
        .context("Failed to decode license signature")?;

    let signature = Signature::from_slice(&signature_bytes)
        .context("Invalid signature length")?;

    // Verify signature using the environment-loaded Public Key
    MASTER_PUBLIC_KEY.verify(&payload_bytes, &signature)
        .context("License signature verification failed")?;

    let payload: LicensePayload = serde_json::from_slice(&payload_bytes)
        .context("Failed to parse license payload")?;

    if payload.is_expired() {
        let expiry = chrono::DateTime::<Utc>::from_timestamp(payload.expiry_timestamp, 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        anyhow::bail!("This license key expired on {}", expiry);
    }

    Ok(payload)
}
