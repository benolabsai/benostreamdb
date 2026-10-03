// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Stateless authentication shared by the HTTP gateways (search, Flight SQL).
//!
//! Two modes, both of which avoid maintaining a user database:
//!
//! * **API key** — a shared secret compared in constant time (`BSDB_API_KEY`).
//! * **JWT** — a bearer token verified against a shared secret (HS256,
//!   `BSDB_JWT_SECRET`) or an RSA public key (RS256, `BSDB_JWT_PUBLIC_KEY`,
//!   PEM). This is the "delegate to your IdP" path: the IdP signs, we only
//!   verify the signature and read `exp`/`nbf`/`aud`/`iss`/`sub` claims.
//!
//! When neither is configured the guard is a no-op; set `BSDB_AUTH_REQUIRED=1`
//! to fail closed instead.

use base64::Engine as _;
use serde_json::Value;
use subtle::ConstantTimeEq;

/// Authentication configuration, resolved once at startup.
#[derive(Clone, Default)]
pub struct AuthConfig {
    api_key: Option<String>,
    jwt_secret: Option<Vec<u8>>,
    jwt_public_key: Option<Vec<u8>>,
    jwt_audience: Option<String>,
    jwt_issuer: Option<String>,
    /// Fail closed when no credential is configured.
    required: bool,
    /// Exempt `/metrics` from auth (for an internal Prometheus scraper).
    metrics_public: bool,
}

impl AuthConfig {
    /// Build from environment variables.
    pub fn from_env() -> Self {
        let truthy = |k: &str| {
            matches!(
                std::env::var(k).as_deref(),
                Ok("1") | Ok("true") | Ok("yes")
            )
        };
        Self {
            api_key: std::env::var("BSDB_API_KEY").ok().filter(|s| !s.is_empty()),
            jwt_secret: std::env::var("BSDB_JWT_SECRET")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| s.into_bytes()),
            jwt_public_key: std::env::var("BSDB_JWT_PUBLIC_KEY")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| s.into_bytes()),
            jwt_audience: std::env::var("BSDB_JWT_AUDIENCE")
                .ok()
                .filter(|s| !s.is_empty()),
            jwt_issuer: std::env::var("BSDB_JWT_ISSUER")
                .ok()
                .filter(|s| !s.is_empty()),
            required: truthy("BSDB_AUTH_REQUIRED"),
            metrics_public: truthy("BSDB_METRICS_PUBLIC"),
        }
    }

    /// Whether any credential source is configured.
    pub fn enabled(&self) -> bool {
        self.api_key.is_some() || self.jwt_secret.is_some() || self.jwt_public_key.is_some()
    }

    /// Whether auth is required even when no credential is configured.
    pub fn required(&self) -> bool {
        self.required
    }

    /// Whether `/metrics` is exempt from auth.
    pub fn metrics_public(&self) -> bool {
        self.metrics_public
    }

    /// Verify a bearer token (API key or JWT). Returns the subject on success.
    pub fn verify(&self, token: &str) -> Result<String, String> {
        // API key first (constant-time).
        if let Some(expected) = &self.api_key {
            if expected.as_bytes().ct_eq(token.as_bytes()).into() {
                return Ok("api-key".to_string());
            }
        }
        // JWT.
        if self.jwt_secret.is_some() || self.jwt_public_key.is_some() {
            return self.verify_jwt(token);
        }
        Err("no credential configured".to_string())
    }

    fn verify_jwt(&self, token: &str) -> Result<String, String> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err("malformed JWT".to_string());
        }
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header_bytes = engine
            .decode(parts[0])
            .map_err(|_| "bad JWT header encoding".to_string())?;
        let payload_bytes = engine
            .decode(parts[1])
            .map_err(|_| "bad JWT payload encoding".to_string())?;
        let signature = engine
            .decode(parts[2])
            .map_err(|_| "bad JWT signature encoding".to_string())?;

        let header: Value =
            serde_json::from_slice(&header_bytes).map_err(|_| "bad JWT header".to_string())?;
        let alg = header
            .get("alg")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "JWT missing alg".to_string())?;

        let signing_input = format!("{}.{}", parts[0], parts[1]);
        match alg {
            "HS256" => {
                let secret = self
                    .jwt_secret
                    .as_ref()
                    .ok_or_else(|| "HS256 token but no BSDB_JWT_SECRET".to_string())?;
                verify_hs256(secret, signing_input.as_bytes(), &signature)?;
            }
            "RS256" => {
                let pem = self
                    .jwt_public_key
                    .as_ref()
                    .ok_or_else(|| "RS256 token but no BSDB_JWT_PUBLIC_KEY".to_string())?;
                verify_rs256(pem, signing_input.as_bytes(), &signature)?;
            }
            other => return Err(format!("unsupported JWT alg '{other}'")),
        }

        let claims: Value =
            serde_json::from_slice(&payload_bytes).map_err(|_| "bad JWT payload".to_string())?;
        self.check_claims(&claims)?;
        Ok(claims
            .get("sub")
            .and_then(|v| v.as_str())
            .unwrap_or("jwt")
            .to_string())
    }

    fn check_claims(&self, claims: &Value) -> Result<(), String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Some(exp) = claims.get("exp").and_then(|v| v.as_u64()) {
            if now >= exp {
                return Err("JWT expired".to_string());
            }
        }
        if let Some(nbf) = claims.get("nbf").and_then(|v| v.as_u64()) {
            if now < nbf {
                return Err("JWT not yet valid".to_string());
            }
        }
        if let Some(expected) = &self.jwt_issuer {
            let iss = claims.get("iss").and_then(|v| v.as_str()).unwrap_or("");
            if iss != expected {
                return Err("JWT issuer mismatch".to_string());
            }
        }
        if let Some(expected) = &self.jwt_audience {
            let ok = match claims.get("aud") {
                Some(Value::String(s)) => s == expected,
                Some(Value::Array(a)) => a.iter().any(|v| v.as_str() == Some(expected.as_str())),
                _ => false,
            };
            if !ok {
                return Err("JWT audience mismatch".to_string());
            }
        }
        Ok(())
    }
}

/// HMAC-SHA256 verification (constant-time).
fn verify_hs256(secret: &[u8], signing_input: &[u8], signature: &[u8]) -> Result<(), String> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret).map_err(|_| "invalid HMAC key".to_string())?;
    mac.update(signing_input);
    mac.verify_slice(signature)
        .map_err(|_| "JWT signature mismatch".to_string())
}

/// RSA PKCS#1 v1.5 SHA-256 verification using a PEM public key.
fn verify_rs256(pem: &[u8], signing_input: &[u8], signature: &[u8]) -> Result<(), String> {
    use ring::signature::{UnparsedPublicKey, RSA_PKCS1_2048_8192_SHA256};
    let der = pem_to_der(pem)?;
    let key = UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, der);
    key.verify(signing_input, signature)
        .map_err(|_| "JWT signature mismatch".to_string())
}

/// Extract the DER bytes from a PEM `PUBLIC KEY` (SPKI) block.
fn pem_to_der(pem: &[u8]) -> Result<Vec<u8>, String> {
    let text = std::str::from_utf8(pem).map_err(|_| "public key is not UTF-8".to_string())?;
    let body: String = text
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .map_err(|_| "public key is not valid base64".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hs256_token(secret: &[u8], claims: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = engine.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = engine.encode(claims.as_bytes());
        let signing_input = format!("{header}.{payload}");
        let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
        mac.update(signing_input.as_bytes());
        let sig = engine.encode(mac.finalize().into_bytes());
        format!("{signing_input}.{sig}")
    }

    #[test]
    fn api_key_constant_time_match() {
        let cfg = AuthConfig {
            api_key: Some("s3cret".to_string()),
            ..Default::default()
        };
        assert!(cfg.verify("s3cret").is_ok());
        assert!(cfg.verify("wrong").is_err());
    }

    #[test]
    fn hs256_round_trip_and_expiry() {
        let secret = b"top-secret";
        let cfg = AuthConfig {
            jwt_secret: Some(secret.to_vec()),
            ..Default::default()
        };
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;
        let good = hs256_token(secret, &format!(r#"{{"sub":"alice","exp":{future}}}"#));
        assert_eq!(cfg.verify(&good).unwrap(), "alice");

        let expired = hs256_token(secret, r#"{"sub":"bob","exp":1}"#);
        assert!(cfg.verify(&expired).is_err());

        let other = AuthConfig {
            jwt_secret: Some(b"different".to_vec()),
            ..Default::default()
        };
        assert!(other.verify(&good).is_err());
    }

    #[test]
    fn audience_and_issuer_enforced() {
        let secret = b"k";
        let cfg = AuthConfig {
            jwt_secret: Some(secret.to_vec()),
            jwt_audience: Some("benostreamdb".to_string()),
            jwt_issuer: Some("https://idp.example".to_string()),
            ..Default::default()
        };
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 60;
        let ok = hs256_token(
            secret,
            &format!(
                r#"{{"sub":"a","exp":{future},"aud":"benostreamdb","iss":"https://idp.example"}}"#
            ),
        );
        assert!(cfg.verify(&ok).is_ok());
        let bad_aud = hs256_token(
            secret,
            &format!(r#"{{"sub":"a","exp":{future},"aud":"other","iss":"https://idp.example"}}"#),
        );
        assert!(cfg.verify(&bad_aud).is_err());
    }
}
