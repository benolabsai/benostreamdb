// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    sign, PayloadChecksumKind, SignableBody, SignableRequest, SigningParams, SigningSettings,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use reqwest::{Client, Method};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};

use super::common::table_root_from_metadata_location;
use super::{Catalog, TableMetadata};
use crate::core::error::BenoStreamError;
use arrow::datatypes::SchemaRef;

/// Authentication method for REST Catalog
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestCatalogAuth {
    /// Static Bearer Token
    BearerToken(String),
    /// OAuth2 Client Credentials flow
    OAuth2 {
        token_endpoint: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    },
    /// AWS Signature Version 4 (Amazon S3 Tables / AWS Glue Iceberg REST)
    AwsSigV4 { region: String, service: String },
}

/// Iceberg `/v1/config` catalog discovery response
#[derive(Deserialize, Debug, Clone, Default)]
pub struct CatalogConfigResponse {
    #[serde(default)]
    pub overrides: HashMap<String, String>,
    #[serde(default)]
    pub defaults: HashMap<String, String>,
}

#[derive(Clone, Debug)]
struct CachedToken {
    token: String,
    expires_at: Instant,
}

/// Update action that pins the table at a caller-written metadata file.
///
/// This is **not** part of the Apache Iceberg REST spec (`set-location` is the
/// standard action, and it names the table base directory, not a metadata
/// file). Databricks Unity and several hosted catalogs accept it; conforming
/// servers such as the open-source Unity Catalog reject it and write the new
/// metadata file themselves. Support is detected once per client and cached.
const METADATA_LOCATION_ACTION: &str = "set-metadata-location";

/// REST Catalog client implementing Iceberg REST Catalog specification
#[derive(Clone)]
pub struct RestCatalogClient {
    /// Active Iceberg REST base URL. Mutable so the config probe can cache the winner.
    base_url: Arc<RwLock<String>>,
    /// Ordered fallback base URLs used only if `/v1/config` 404s on the primary.
    probe_candidates: Vec<String>,
    client: Client,
    prefix: Arc<RwLock<String>>,
    prefix_explicit: bool,
    warehouse: Option<String>,
    discover_on_first_use: bool,
    config_discovered: Arc<RwLock<bool>>,
    discover_gate: Arc<Mutex<()>>,
    config_defaults: Arc<RwLock<HashMap<String, String>>>,
    auth: Option<RestCatalogAuth>,
    token_cache: Arc<RwLock<Option<CachedToken>>>,
    aws_credentials: Arc<RwLock<Option<SharedCredentialsProvider>>>,
    /// `Some(true)` when the catalog accepts [`METADATA_LOCATION_ACTION`],
    /// `Some(false)` when it rejected it and manages metadata itself.
    metadata_location_action_supported: Arc<RwLock<Option<bool>>>,
}

/// Outcome of probing one candidate `/v1/config` endpoint.
enum ConfigProbe {
    Hit(CatalogConfigResponse),
    /// The mount point isn't there (400/404/405/501); another candidate may serve it.
    Missing(anyhow::Error),
    /// The endpoint answered but refused the request (auth, 5xx) — surface it if nothing hits.
    Refused(anyhow::Error),
}

// Request structures
#[derive(Serialize)]
struct CreateNamespaceRequest {
    namespace: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
struct CreateTableRequest {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none", rename = "partition-spec")]
    partition_spec: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "write-order")]
    write_order: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<HashMap<String, String>>,
}

// Response structures
#[derive(Deserialize)]
struct TableIdentifier {
    #[allow(dead_code)]
    namespace: Option<Vec<String>>,
    name: String,
}

#[derive(Deserialize)]
struct ListTablesResponse {
    identifiers: Vec<TableIdentifier>,
    #[serde(default, rename = "next-page-token")]
    next_page_token: Option<String>,
}

#[derive(Deserialize)]
struct LoadTableResponse {
    #[allow(dead_code)]
    #[serde(default, rename = "metadata-location")]
    metadata_location: String,
    metadata: TableMetadata,
}

#[derive(Serialize)]
struct UpdateTableRequest {
    updates: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requirements: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Deserialize)]
struct IcebergErrorEnvelope {
    error: Option<IcebergErrorBody>,
}

#[derive(Deserialize)]
struct IcebergErrorBody {
    message: Option<String>,
}

fn encode_segment(segment: &str) -> String {
    urlencoding::encode(segment).into_owned()
}

/// Compute AWS SigV4 signing headers for an HTTP request.
pub(crate) fn compute_sigv4_headers(
    method: &str,
    url: &str,
    extra_headers: &[(&str, &str)],
    body: &[u8],
    creds: &Credentials,
    region: &str,
    service: &str,
    now: SystemTime,
) -> Result<Vec<(String, String)>> {
    let identity: Identity = creds.clone().into();
    let mut settings = SigningSettings::default();
    settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;

    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(service)
        .time(now)
        .settings(settings)
        .build()
        .map_err(|e| anyhow!("Failed to build SigV4 signing parameters: {e}"))?;
    let params: SigningParams<'_> = signing_params.into();

    let signable = SignableRequest::new(
        method,
        url,
        extra_headers.iter().copied(),
        SignableBody::Bytes(body),
    )
    .map_err(|e| anyhow!("Failed to construct SigV4 signable request: {e}"))?;

    let (instructions, _signature) = sign(signable, &params)
        .map_err(|e| anyhow!("Failed to compute SigV4 signature: {e}"))?
        .into_parts();

    Ok(instructions
        .headers()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect())
}

async fn error_from_response(resp: reqwest::Response, operation: &str) -> anyhow::Error {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    let msg = serde_json::from_str::<IcebergErrorEnvelope>(&body)
        .ok()
        .and_then(|env| env.error.and_then(|e| e.message))
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| {
            if body.is_empty() {
                format!("HTTP {status}")
            } else {
                body
            }
        });
    anyhow::Error::new(BenoStreamError::CatalogRequest {
        catalog: "rest".to_string(),
        status: Some(status),
        reason: format!("{operation}: {msg}"),
    })
}

impl RestCatalogClient {
    /// Create a new unauthenticated REST Catalog client
    pub fn new(base_url: String, prefix: Option<String>) -> Self {
        Self::with_auth(base_url, prefix, None)
    }

    /// Create a new REST Catalog client with authentication
    pub fn with_auth(
        base_url: String,
        prefix: Option<String>,
        auth: Option<RestCatalogAuth>,
    ) -> Self {
        let prefix_explicit = prefix
            .as_deref()
            .is_some_and(|p| !p.trim_matches('/').is_empty());
        let clean_prefix = prefix.unwrap_or_default().trim_matches('/').to_string();
        Self {
            base_url: Arc::new(RwLock::new(base_url.trim_end_matches('/').to_string())),
            probe_candidates: Vec::new(),
            // Bounded timeout: an unresponsive catalog must not block forever.
            client: super::catalog_http_client(),
            prefix: Arc::new(RwLock::new(clean_prefix)),
            prefix_explicit,
            warehouse: None,
            discover_on_first_use: false,
            config_discovered: Arc::new(RwLock::new(false)),
            discover_gate: Arc::new(Mutex::new(())),
            config_defaults: Arc::new(RwLock::new(HashMap::new())),
            auth,
            token_cache: Arc::new(RwLock::new(None)),
            aws_credentials: Arc::new(RwLock::new(None)),
            metadata_location_action_supported: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a client with a pre-configured static Bearer token
    pub fn with_bearer_token(base_url: String, prefix: Option<String>, token: String) -> Self {
        Self::with_auth(base_url, prefix, Some(RestCatalogAuth::BearerToken(token)))
    }

    /// Create a client configured for OAuth2 client credentials grant (Polaris / Lakekeeper)
    pub fn with_oauth2(
        base_url: String,
        prefix: Option<String>,
        token_endpoint: Option<String>,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    ) -> Self {
        let endpoint = token_endpoint
            .unwrap_or_else(|| format!("{}/v1/oauth/tokens", base_url.trim_end_matches('/')));
        Self::with_auth(
            base_url,
            prefix,
            Some(RestCatalogAuth::OAuth2 {
                token_endpoint: endpoint,
                client_id,
                client_secret,
                scope,
            }),
        )
    }

    /// Create a REST Catalog client using AWS SigV4 request signing (S3 Tables / Glue Iceberg REST).
    pub fn with_sigv4(
        base_url: String,
        prefix: Option<String>,
        region: impl Into<String>,
        service: impl Into<String>,
    ) -> Self {
        Self::with_auth(
            base_url,
            prefix,
            Some(RestCatalogAuth::AwsSigV4 {
                region: region.into(),
                service: service.into(),
            }),
        )
    }

    /// Attach ordered fallback Iceberg REST base URLs.
    ///
    /// The primary [`base_url`](Self::new) is probed first, then each candidate in order.
    /// The first mount point that answers `GET /v1/config` is cached as the active base,
    /// so the probe cost is paid at most once per client. Different deployments mount
    /// Iceberg REST at different paths (open-source `unitycatalog/unitycatalog` uses
    /// `.../unity-catalog/iceberg`, Databricks Unity uses `.../unity-catalog/iceberg-rest`).
    pub fn with_base_url_candidates(mut self, candidates: Vec<String>) -> Self {
        self.probe_candidates = candidates
            .into_iter()
            .map(|c| c.trim_end_matches('/').to_string())
            .filter(|c| !c.is_empty())
            .collect();
        self
    }

    /// The currently active Iceberg REST base URL (the probe winner once discovered).
    pub async fn base_url(&self) -> String {
        self.base_url.read().await.clone()
    }

    /// Attach a catalog `warehouse` identifier (e.g., S3 Tables bucket ARN) and optionally
    /// trigger `/v1/config` discovery before the first table/namespace request.
    pub fn with_warehouse(
        mut self,
        warehouse: Option<String>,
        discover_on_first_use: bool,
    ) -> Self {
        self.warehouse = warehouse.filter(|w| !w.trim().is_empty());
        self.discover_on_first_use = discover_on_first_use;
        self
    }

    /// Provide explicit static AWS credentials for SigV4 signing (primarily for deterministic tests).
    pub fn with_aws_credentials(self, creds: Credentials) -> Self {
        let provider = SharedCredentialsProvider::new(creds);
        self.with_aws_credentials_provider(provider)
    }

    /// Provide an explicit AWS [`SharedCredentialsProvider`] for SigV4 signing.
    pub fn with_aws_credentials_provider(mut self, provider: SharedCredentialsProvider) -> Self {
        self.aws_credentials = Arc::new(RwLock::new(Some(provider)));
        self
    }

    /// Return the active URL routing prefix (populated either explicitly or via `/v1/config`).
    pub async fn prefix(&self) -> String {
        self.prefix.read().await.clone()
    }

    /// Return merged `defaults` / `overrides` learned from `/v1/config`.
    pub async fn config_defaults(&self) -> HashMap<String, String> {
        self.config_defaults.read().await.clone()
    }

    /// Resolve AWS credentials provider lazily via `aws_config` default chain.
    async fn resolve_aws_credentials(&self) -> Result<Credentials> {
        let existing = { self.aws_credentials.read().await.clone() };
        let provider = match existing {
            Some(p) => p,
            None => {
                let sdk_cfg = aws_config::load_defaults(BehaviorVersion::latest()).await;
                let p = sdk_cfg.credentials_provider().ok_or_else(|| {
                    anyhow!("No AWS credentials provider configured for SigV4 catalog signing")
                })?;
                *self.aws_credentials.write().await = Some(p.clone());
                p
            }
        };

        provider
            .provide_credentials()
            .await
            .map_err(|e| anyhow!("Failed to load AWS credentials for SigV4 signing: {e}"))
    }

    /// Get current active Bearer token, fetching or refreshing via OAuth2 if required
    pub async fn get_token(&self) -> Result<Option<String>> {
        let auth = match &self.auth {
            Some(a) => a,
            None => return Ok(None),
        };

        match auth {
            RestCatalogAuth::BearerToken(token) => Ok(Some(token.clone())),
            RestCatalogAuth::OAuth2 {
                token_endpoint,
                client_id,
                client_secret,
                scope,
            } => {
                // Fast path: check valid cached token
                {
                    let cache = self.token_cache.read().await;
                    if let Some(ref cached) = *cache {
                        if Instant::now() < cached.expires_at {
                            return Ok(Some(cached.token.clone()));
                        }
                    }
                }

                // Slow path: acquire write lock and refresh
                let mut write_cache = self.token_cache.write().await;
                if let Some(ref cached) = *write_cache {
                    if Instant::now() < cached.expires_at {
                        return Ok(Some(cached.token.clone()));
                    }
                }

                let mut params = vec![
                    ("grant_type", "client_credentials"),
                    ("client_id", client_id.as_str()),
                    ("client_secret", client_secret.as_str()),
                ];
                if let Some(ref s) = scope {
                    params.push(("scope", s.as_str()));
                }

                let resp = self
                    .client
                    .post(token_endpoint)
                    .form(&params)
                    .send()
                    .await
                    .map_err(|e| anyhow!("OAuth2 token request failed: {}", e))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let err = resp.text().await.unwrap_or_default();
                    return Err(anyhow!(
                        "OAuth2 token request returned {} - {}",
                        status,
                        err
                    ));
                }

                let token_resp: OAuthTokenResponse = resp
                    .json()
                    .await
                    .map_err(|e| anyhow!("Failed to parse OAuth2 token response: {}", e))?;

                let ttl_secs = token_resp.expires_in.unwrap_or(3600);
                // Refresh 60 seconds before actual expiration
                let safety_margin = std::cmp::min(60, ttl_secs / 2);
                let valid_duration = Duration::from_secs(ttl_secs.saturating_sub(safety_margin));
                let expires_at = Instant::now() + valid_duration;

                let token = token_resp.access_token;
                *write_cache = Some(CachedToken {
                    token: token.clone(),
                    expires_at,
                });

                Ok(Some(token))
            }
            RestCatalogAuth::AwsSigV4 { .. } => Ok(None),
        }
    }

    /// Execute an HTTP request with configured authentication (Bearer, OAuth2, or AWS SigV4).
    async fn send_request(
        &self,
        method: Method,
        url: &str,
        json_body: Option<&serde_json::Value>,
    ) -> Result<reqwest::Response> {
        let body_bytes = match json_body {
            Some(val) => serde_json::to_vec(val)?,
            None => Vec::new(),
        };

        let mut builder = self.client.request(method.clone(), url);
        let extra_headers: Vec<(&str, &str)> = if json_body.is_some() {
            builder = builder.header(reqwest::header::CONTENT_TYPE, "application/json");
            vec![("content-type", "application/json")]
        } else {
            Vec::new()
        };

        match &self.auth {
            Some(RestCatalogAuth::AwsSigV4 { region, service }) => {
                let creds = self.resolve_aws_credentials().await?;
                let sig_headers = compute_sigv4_headers(
                    method.as_str(),
                    url,
                    &extra_headers,
                    &body_bytes,
                    &creds,
                    region,
                    service,
                    SystemTime::now(),
                )?;
                for (k, v) in sig_headers {
                    builder = builder.header(k, v);
                }
            }
            Some(_) => {
                if let Some(token) = self.get_token().await? {
                    builder = builder.bearer_auth(token);
                }
            }
            None => {}
        }

        if json_body.is_some() {
            builder = builder.body(body_bytes);
        }

        Ok(builder.send().await?)
    }

    /// Fetch Iceberg REST `/v1/config` overrides and defaults, updating active prefix unless overridden.
    ///
    /// The primary base URL is tried first, then any configured fallbacks (see
    /// [`Self::with_base_url_candidates`]). A candidate that answers `2xx` wins and is cached
    /// as the active base, so the probe cost is paid once per client.
    ///
    /// `warehouse` is always included when one is known: the open-source Unity Catalog server
    /// rejects a bare `/v1/config` with `400 Must supply a proper catalog in warehouse
    /// property`.
    pub async fn fetch_config(&self, warehouse: Option<&str>) -> Result<CatalogConfigResponse> {
        let wh = warehouse
            .or(self.warehouse.as_deref())
            .map(str::trim)
            .filter(|w| !w.is_empty());

        let mut bases: Vec<String> = Vec::with_capacity(self.probe_candidates.len() + 1);
        bases.push(self.base_url.read().await.clone());
        for cand in &self.probe_candidates {
            if !bases.contains(cand) {
                bases.push(cand.clone());
            }
        }

        let mut refused: Option<anyhow::Error> = None;
        let mut missing: Option<anyhow::Error> = None;

        for base in bases {
            match self.probe_config_from(&base, wh).await {
                ConfigProbe::Hit(cfg) => {
                    {
                        let mut active = self.base_url.write().await;
                        if *active != base {
                            tracing::info!(
                                from = %*active,
                                to = %base,
                                "Iceberg REST /v1/config base path selected"
                            );
                            *active = base;
                        }
                    }
                    self.apply_config(&cfg).await;
                    return Ok(cfg);
                }
                ConfigProbe::Missing(err) => missing = Some(err),
                ConfigProbe::Refused(err) => {
                    if refused.is_none() {
                        refused = Some(err);
                    }
                }
            }
        }

        Err(refused.or(missing).unwrap_or_else(|| {
            anyhow!("Iceberg REST discovery failed: no candidate base URL configured")
        }))
    }

    /// Probe a single candidate base URL for `/v1/config`.
    async fn probe_config_from(&self, base: &str, warehouse: Option<&str>) -> ConfigProbe {
        let url = match warehouse {
            Some(w) => format!("{base}/v1/config?warehouse={}", encode_segment(w)),
            None => format!("{base}/v1/config"),
        };

        let resp = match self.send_request(Method::GET, &url, None).await {
            Ok(resp) => resp,
            // Transport-level failure is about the host, not the path — do not retry siblings.
            Err(err) => return ConfigProbe::Refused(err),
        };

        let status = resp.status();
        if !status.is_success() {
            let err = error_from_response(resp, "GET /v1/config").await;
            let missing = matches!(
                status,
                reqwest::StatusCode::BAD_REQUEST
                    | reqwest::StatusCode::NOT_FOUND
                    | reqwest::StatusCode::METHOD_NOT_ALLOWED
                    | reqwest::StatusCode::NOT_IMPLEMENTED
            );
            return if missing {
                ConfigProbe::Missing(err)
            } else {
                ConfigProbe::Refused(err)
            };
        }

        match resp.json::<CatalogConfigResponse>().await {
            Ok(cfg) => ConfigProbe::Hit(cfg),
            Err(err) => ConfigProbe::Refused(err.into()),
        }
    }

    /// Apply a `/v1/config` response to the routing prefix and merged defaults.
    async fn apply_config(&self, cfg: &CatalogConfigResponse) {
        if !self.prefix_explicit {
            if let Some(pfx) = cfg
                .overrides
                .get("prefix")
                .or_else(|| cfg.defaults.get("prefix"))
            {
                let clean = pfx.trim_matches('/').to_string();
                if !clean.is_empty() {
                    *self.prefix.write().await = clean;
                }
            }
        }

        {
            let mut defaults = self.config_defaults.write().await;
            for (k, v) in &cfg.defaults {
                defaults.insert(k.clone(), v.clone());
            }
            for (k, v) in &cfg.overrides {
                defaults.insert(k.clone(), v.clone());
            }
        }
        *self.config_discovered.write().await = true;
    }

    async fn ensure_config_discovered(&self) -> Result<()> {
        // Autodetection always needs one probe, even when discovery wasn't requested.
        if (!self.discover_on_first_use && self.probe_candidates.is_empty())
            || *self.config_discovered.read().await
        {
            return Ok(());
        }
        let _guard = self.discover_gate.lock().await;
        if *self.config_discovered.read().await {
            return Ok(());
        }
        self.fetch_config(self.warehouse.as_deref()).await?;
        Ok(())
    }

    /// Build full URL with optional prefix
    async fn build_url(&self, path: &str) -> String {
        let base = self.base_url.read().await.clone();
        let pfx = self.prefix.read().await;
        if pfx.is_empty() {
            format!("{base}/v1{}", path)
        } else {
            format!("{base}/v1/{}{}", pfx.trim_matches('/'), path)
        }
    }

    async fn commit_table_internal(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Option<Vec<serde_json::Value>>,
    ) -> Result<()> {
        self.ensure_config_discovered().await?;
        let url = self
            .build_url(&format!(
                "/namespaces/{}/tables/{}",
                encode_segment(namespace),
                encode_segment(table_name)
            ))
            .await;

        let mut updates = updates;
        if *self.metadata_location_action_supported.read().await == Some(false) {
            drop_metadata_location_action(&mut updates);
        }

        let resp = self
            .post_commit(&url, &updates, requirements.as_deref())
            .await?;
        if resp.status().is_success() {
            if has_metadata_location_action(&updates) {
                *self.metadata_location_action_supported.write().await = Some(true);
            }
            return Ok(());
        }

        let status = resp.status();
        let err = error_from_response(resp, "commit_table").await;
        let text = err.to_string();
        // Conforming catalogs reject the unknown action with different wording:
        // Unity says "Cannot convert metadata update action to json", while Polaris
        // maps it to a bare `WebApplicationException: HTTP 400 Bad Request`.
        let is_protocol_mismatch = (status == reqwest::StatusCode::BAD_REQUEST
            || status == reqwest::StatusCode::INTERNAL_SERVER_ERROR)
            && (text.contains("metadata update action")
                || text.contains("WebApplicationException")
                || text.contains("HTTP 400 Bad Request"));
        if !is_protocol_mismatch || !has_metadata_location_action(&updates) {
            return Err(err);
        }

        // The catalog implements the Iceberg REST spec, which has no
        // `set-metadata-location` action: it writes the next metadata file itself
        // from the snapshot updates. Retry once without the action and remember
        // the verdict so later commits skip the doomed first attempt.
        // Bind before logging: holding a guard inside the macro would leak its
        // non-`Sync` temporary across the awaits below.
        let catalog_base = self.base_url.read().await.clone();
        tracing::info!(
            catalog = %catalog_base,
            action = METADATA_LOCATION_ACTION,
            "Iceberg REST catalog rejects the action; letting the catalog manage metadata files"
        );
        *self.metadata_location_action_supported.write().await = Some(false);
        drop_metadata_location_action(&mut updates);

        let resp = self
            .post_commit(&url, &updates, requirements.as_deref())
            .await?;
        if !resp.status().is_success() {
            return Err(error_from_response(resp, "commit_table").await);
        }
        Ok(())
    }

    /// POST a `CommitTableRequest` and hand back the raw response.
    async fn post_commit(
        &self,
        url: &str,
        updates: &[serde_json::Value],
        requirements: Option<&[serde_json::Value]>,
    ) -> Result<reqwest::Response> {
        let body = serde_json::to_value(UpdateTableRequest {
            updates: updates.to_vec(),
            requirements: requirements.map(|r| r.to_vec()),
        })?;
        self.send_request(Method::POST, url, Some(&body)).await
    }
}

/// True when the update list carries the non-standard metadata-location action.
fn has_metadata_location_action(updates: &[serde_json::Value]) -> bool {
    updates
        .iter()
        .any(|u| u.get("action").and_then(|a| a.as_str()) == Some(METADATA_LOCATION_ACTION))
}

/// Remove the non-standard metadata-location action from an update list.
fn drop_metadata_location_action(updates: &mut Vec<serde_json::Value>) {
    updates.retain(|u| u.get("action").and_then(|a| a.as_str()) != Some(METADATA_LOCATION_ACTION));
}

#[async_trait]
impl Catalog for RestCatalogClient {
    async fn create_namespace(&self, namespace: &str) -> Result<()> {
        self.ensure_config_discovered().await?;
        let url = self.build_url("/namespaces").await;
        let req = CreateNamespaceRequest {
            namespace: vec![namespace.to_string()],
            properties: None,
        };
        let body = serde_json::to_value(&req)?;
        let resp = self.send_request(Method::POST, &url, Some(&body)).await?;
        if !resp.status().is_success() {
            return Err(error_from_response(resp, "create_namespace").await);
        }
        Ok(())
    }

    async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        self.ensure_config_discovered().await?;
        let url = self
            .build_url(&format!("/namespaces/{}", encode_segment(namespace)))
            .await;

        let resp = self.send_request(Method::HEAD, &url, None).await?;
        let resp = if matches!(
            resp.status(),
            reqwest::StatusCode::METHOD_NOT_ALLOWED | reqwest::StatusCode::NOT_IMPLEMENTED
        ) {
            self.send_request(Method::GET, &url, None).await?
        } else {
            resp
        };

        if resp.status().is_success() {
            Ok(true)
        } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(false)
        } else {
            Err(error_from_response(resp, "namespace_exists").await)
        }
    }

    async fn drop_namespace(&self, namespace: &str) -> Result<()> {
        self.ensure_config_discovered().await?;
        let url = self
            .build_url(&format!("/namespaces/{}", encode_segment(namespace)))
            .await;
        let resp = self.send_request(Method::DELETE, &url, None).await?;
        if !resp.status().is_success() {
            return Err(error_from_response(resp, "drop_namespace").await);
        }
        Ok(())
    }

    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        self.ensure_config_discovered().await?;
        let url = self
            .build_url(&format!("/namespaces/{}/tables", encode_segment(namespace)))
            .await;

        // Convert schema to JSON for REST API
        let manifest_schema = crate::core::manifest::Schema::from_arrow(&schema, 1);
        let schema_json = serde_json::to_value(&manifest_schema)?;

        let req = CreateTableRequest {
            name: table_name.to_string(),
            location: location.map(|s| s.to_string()),
            schema: schema_json,
            partition_spec: None,
            write_order: None,
            properties: None,
        };
        let body = serde_json::to_value(&req)?;

        let resp = self.send_request(Method::POST, &url, Some(&body)).await?;
        if !resp.status().is_success() {
            return Err(error_from_response(resp, "create_table").await);
        }

        Ok(())
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        self.ensure_config_discovered().await?;
        let url = self
            .build_url(&format!(
                "/namespaces/{}/tables/{}",
                encode_segment(namespace),
                encode_segment(table_name)
            ))
            .await;

        let resp = self.send_request(Method::GET, &url, None).await?;
        if !resp.status().is_success() {
            return Err(error_from_response(resp, "load_table").await);
        }

        let mut table_resp: LoadTableResponse = resp.json().await?;
        if table_resp.metadata.location.is_empty() && !table_resp.metadata_location.is_empty() {
            table_resp.metadata.location =
                table_root_from_metadata_location(&table_resp.metadata_location);
        }
        Ok(table_resp.metadata)
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, None)
            .await
    }

    async fn commit_table_with_requirements(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        let reqs = if requirements.is_empty() {
            None
        } else {
            Some(requirements)
        };
        self.commit_table_internal(namespace, table_name, updates, reqs)
            .await
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        // REST Catalog doesn't support branches (Nessie-specific feature)
        Err(anyhow!(
            "REST Catalog does not support branching. Use Nessie for Git-like branching."
        ))
    }

    async fn drop_table(&self, namespace: &str, table_name: &str) -> Result<()> {
        self.ensure_config_discovered().await?;
        let url = self
            .build_url(&format!(
                "/namespaces/{}/tables/{}",
                encode_segment(namespace),
                encode_segment(table_name)
            ))
            .await;
        let resp = self.send_request(Method::DELETE, &url, None).await?;
        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::NOT_FOUND {
            return Err(error_from_response(resp, "drop_table").await);
        }
        Ok(())
    }

    async fn list_tables(&self, namespace: &str) -> Result<Vec<String>> {
        self.ensure_config_discovered().await?;
        let base_list_url = self
            .build_url(&format!("/namespaces/{}/tables", encode_segment(namespace)))
            .await;

        let mut names = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let url = match &page_token {
                Some(tok) => format!("{base_list_url}?pageToken={}", encode_segment(tok)),
                None => base_list_url.clone(),
            };
            let resp = self.send_request(Method::GET, &url, None).await?;
            if !resp.status().is_success() {
                return Err(error_from_response(resp, "list_tables").await);
            }
            let list_resp: ListTablesResponse = resp.json().await?;
            names.extend(list_resp.identifiers.into_iter().map(|id| id.name));
            match list_resp.next_page_token.filter(|t| !t.is_empty()) {
                Some(tok) => page_token = Some(tok),
                None => break,
            }
        }

        Ok(names)
    }

    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        match self.load_table(namespace, table_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rest_catalog_bearer_token() -> Result<()> {
        let client = RestCatalogClient::with_bearer_token(
            "http://localhost:8181".to_string(),
            None,
            "static-secret-token".to_string(),
        );

        let token = client.get_token().await?;
        assert_eq!(token, Some("static-secret-token".to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn test_rest_catalog_token_caching() -> Result<()> {
        let client = RestCatalogClient::with_oauth2(
            "http://localhost:8181".to_string(),
            Some("polaris".to_string()),
            None,
            "principal_id".to_string(),
            "principal_secret".to_string(),
            Some("PRINCIPAL_ROLE:ALL".to_string()),
        );

        // Pre-populate valid cache
        {
            let mut cache = client.token_cache.write().await;
            *cache = Some(CachedToken {
                token: "cached-polaris-token".to_string(),
                expires_at: Instant::now() + Duration::from_secs(300),
            });
        }

        let token = client.get_token().await?;
        assert_eq!(token, Some("cached-polaris-token".to_string()));
        Ok(())
    }

    #[tokio::test]
    async fn test_create_catalog_async_oauth2() -> Result<()> {
        let mut config = HashMap::new();
        // Non-localhost host: the SSRF guard (always enabled) rejects
        // `localhost`, and `table_exists` maps the resulting network error to
        // `Ok(false)`, which is what this test asserts.
        config.insert(
            "url".to_string(),
            "http://catalog.example.com:8181".to_string(),
        );
        config.insert("prefix".to_string(), "polaris_warehouse".to_string());
        config.insert(
            "credential".to_string(),
            "polaris_client:polaris_secret".to_string(),
        );
        config.insert("scope".to_string(), "PRINCIPAL_ROLE:ALL".to_string());

        let catalog = crate::core::catalog::create_catalog_async(
            crate::core::catalog::CatalogType::Rest,
            config,
        )
        .await?;

        assert!(!catalog.table_exists("ns", "table").await.unwrap_or(true));

        Ok(())
    }

    #[test]
    fn test_compute_sigv4_headers_includes_required_headers() -> Result<()> {
        let creds = Credentials::new(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            Some("session-token-123".to_string()),
            None,
            "unit-test",
        );
        let ts = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let headers = compute_sigv4_headers(
            "POST",
            "https://s3tables.us-east-1.amazonaws.com/iceberg/v1/wh/namespaces/ns/tables",
            &[("content-type", "application/json")],
            br#"{"name":"events"}"#,
            &creds,
            "us-east-1",
            "s3tables",
            ts,
        )?;

        let map: HashMap<String, String> = headers
            .into_iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v))
            .collect();

        let auth = map
            .get("authorization")
            .ok_or_else(|| anyhow!("missing Authorization header"))?;
        assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/"));
        assert!(auth.contains("/us-east-1/s3tables/aws4_request"));
        assert!(auth.contains("SignedHeaders="));
        assert!(auth.contains("Signature="));
        assert!(map.contains_key("x-amz-date"));
        assert!(map.contains_key("x-amz-content-sha256"));
        assert_eq!(
            map.get("x-amz-security-token").map(String::as_str),
            Some("session-token-123")
        );
        Ok(())
    }
}
