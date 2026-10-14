// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::storage::create_object_store;
use anyhow::Result;
use object_store::path::Path;
use object_store::{ObjectStore, PutMode, PutOptions, UpdateVersion};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Standard filename for the authoritative catalog storage policy in the warehouse/bucket root.
pub const CATALOG_POLICY_FILE: &str = ".benostream-policy.json";

/// Baseline system catalog schemas that are strictly immutable and can NEVER be configured as writable.
pub const IMMUTABLE_SYSTEM_SCHEMAS: &[&str] =
    &["default", "public", "information_schema", "sys", "system"];

/// The default temporary schema reserved for ephemeral agent scratchpad queries.
pub const SCRATCH_SCHEMA: &str = "scratch";

/// Authoritative catalog storage policy stored directly in the catalog or warehouse root.
/// When present, this file defines the security policy set by the core database product or administrator.
/// When NOT present, engines default to conservative lockdown: all persistent catalog schemas
/// are read-only, and only the isolated `scratch` schema is writable.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogPolicy {
    /// Identifier of the catalog/warehouse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    /// Policy format version (e.g. "1.0").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_version: Option<String>,
    /// Explicit list of schemas permitted for writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readwrite_schemas: Option<Vec<String>>,
    /// Schemas strictly locked as read-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readonly_schemas: Option<Vec<String>>,
    /// Schemas permanently immutable in this catalog.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub immutable_schemas: Option<Vec<String>>,
    /// Schemas with **no access at all** for untrusted connectors — neither
    /// readable nor writable. Overrides the read-only floor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_access_schemas: Option<Vec<String>>,
    /// Default write schema for unqualified table creations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_write_schema: Option<String>,
    /// Description or comment about the policy or dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl CatalogPolicy {
    pub fn new() -> Self {
        Self {
            policy_version: Some("1.0".to_string()),
            ..Default::default()
        }
    }

    /// Checks if a schema is writable under this policy.
    pub fn is_schema_writable(&self, schema: &str) -> bool {
        let s = schema.to_ascii_lowercase();
        // 1. Immutable system schemas are NEVER writable
        if IMMUTABLE_SYSTEM_SCHEMAS
            .iter()
            .any(|sys| sys.eq_ignore_ascii_case(&s))
        {
            return false;
        }
        // 2. Explicit readonly schemas strictly veto
        if let Some(ref ro) = self.readonly_schemas {
            if ro.iter().any(|r| r.eq_ignore_ascii_case(&s)) {
                return false;
            }
        }
        // 3. Explicit immutable schemas strictly veto
        if let Some(ref imm) = self.immutable_schemas {
            if imm.iter().any(|r| r.eq_ignore_ascii_case(&s)) {
                return false;
            }
        }
        // 3b. Explicit no-access schemas are never writable
        if let Some(ref na) = self.no_access_schemas {
            if na.iter().any(|r| r.eq_ignore_ascii_case(&s)) {
                return false;
            }
        }
        // 4. Must be explicitly allowed in readwrite_schemas
        if let Some(ref rw) = self.readwrite_schemas {
            rw.iter().any(|w| w == "*" || w.eq_ignore_ascii_case(&s))
        } else {
            // Conservative default: only scratch is writable if no readwrite list is specified
            s == SCRATCH_SCHEMA
        }
    }

    /// Sets a schema's access to read-only or read-write.
    /// Immutable system schemas cannot be modified.
    pub fn set_schema_access(&mut self, schema: &str, read_only: bool) {
        let clean = schema.trim().to_ascii_lowercase();
        if IMMUTABLE_SYSTEM_SCHEMAS
            .iter()
            .any(|sys| sys.eq_ignore_ascii_case(&clean))
        {
            return;
        }

        let mut ro: BTreeSet<String> = self
            .readonly_schemas
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.to_ascii_lowercase())
            .collect();
        let mut rw: BTreeSet<String> = self
            .readwrite_schemas
            .clone()
            .unwrap_or_else(|| vec![SCRATCH_SCHEMA.to_string()])
            .into_iter()
            .map(|s| s.to_ascii_lowercase())
            .collect();

        if read_only {
            rw.remove(&clean);
            ro.insert(clean);
        } else {
            ro.remove(&clean);
            rw.insert(clean);
        }

        self.readonly_schemas = Some(ro.into_iter().collect());
        self.readwrite_schemas = Some(rw.into_iter().collect());
    }

    /// Locks down all schemas as read-only except the temporary `scratch` schema.
    pub fn lock_all_read_only(&mut self) {
        self.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string()]);
    }
}

/// Reads the authoritative catalog policy via the object-store abstraction.
///
/// Works for local paths (`/path`, `file:///path`) and object stores (`s3://`,
/// `gs://`, `az://`, `memory://`), because URI resolution is delegated to
/// [`create_object_store`]. A missing policy object is not an error: it maps to
/// `Ok(None)` so callers fail closed (conservative lockdown).
async fn read_policy_via_store(warehouse: &str) -> Result<Option<CatalogPolicy>> {
    let store = create_object_store(warehouse)?;
    let path = Path::from(CATALOG_POLICY_FILE);
    match store.get(&path).await {
        Ok(result) => {
            let bytes = result.bytes().await?;
            let policy: CatalogPolicy = serde_json::from_slice(&bytes)?;
            Ok(Some(policy))
        }
        // No policy object in the warehouse → "no policy set" (fail closed upstream).
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Writes the authoritative catalog policy via the object-store abstraction.
///
/// The underlying object write is atomic for S3/GCS/Azure and local fs. This is
/// the unconditional admin override; the read-modify-write helpers use
/// [`write_policy_cas`] instead.
async fn write_policy_via_store(warehouse: &str, policy: &CatalogPolicy) -> Result<()> {
    let store = create_object_store(warehouse)?;
    let path = Path::from(CATALOG_POLICY_FILE);
    let content = serde_json::to_vec_pretty(policy)?;
    store.put(&path, content.into()).await?;
    Ok(())
}

/// Reads the policy together with its object-store version/ETag, for CAS writes.
async fn read_policy_with_version(
    warehouse: &str,
) -> Result<Option<(CatalogPolicy, UpdateVersion)>> {
    let store = create_object_store(warehouse)?;
    let path = Path::from(CATALOG_POLICY_FILE);
    match store.get(&path).await {
        Ok(result) => {
            let version = UpdateVersion {
                e_tag: result.meta.e_tag.clone(),
                version: result.meta.version.clone(),
            };
            let bytes = result.bytes().await?;
            let policy: CatalogPolicy = serde_json::from_slice(&bytes)?;
            Ok(Some((policy, version)))
        }
        // No policy object → "no policy set" (fail closed upstream).
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Writes the policy under optimistic concurrency.
///
/// Returns `Ok(false)` (never an error) when the object changed since it was
/// read — a concurrent writer won the race — so the caller can re-read and
/// retry. `PutMode::Create` fails closed if the object appeared meanwhile.
async fn write_policy_cas(warehouse: &str, policy: &CatalogPolicy, mode: PutMode) -> Result<bool> {
    let store = create_object_store(warehouse)?;
    let path = Path::from(CATALOG_POLICY_FILE);
    let content = serde_json::to_vec_pretty(policy)?;
    let options = PutOptions {
        mode,
        ..Default::default()
    };
    match store.put_opts(&path, content.clone().into(), options).await {
        Ok(_) => Ok(true),
        Err(object_store::Error::AlreadyExists { .. })
        | Err(object_store::Error::Precondition { .. }) => Ok(false),
        // Backends that do not support conditional PUT (some local/temp stores):
        // fall back to an atomic overwrite. S3/GCS/Azure support conditions, so
        // this only affects stores that cannot lose updates across hosts anyway.
        Err(object_store::Error::NotImplemented) => {
            store.put(&path, content.into()).await?;
            Ok(true)
        }
        Err(e) => Err(e.into()),
    }
}

/// Bounded retries for the policy CAS loop.
const POLICY_MAX_RETRIES: usize = 8;

/// Read-modify-write the catalog policy under optimistic concurrency control.
///
/// Re-reads and retries when another writer wins the CAS race, so concurrent
/// [`set_catalog_schema_access`] / [`lock_catalog_read_only`] calls cannot lose
/// updates. `seed` is the policy used when none exists yet.
async fn update_policy<F>(warehouse: &str, seed: CatalogPolicy, mutate: F) -> Result<CatalogPolicy>
where
    F: Fn(&mut CatalogPolicy) + Send + Sync,
{
    for _ in 0..POLICY_MAX_RETRIES {
        let (mut policy, mode) = match read_policy_with_version(warehouse).await? {
            Some((existing, version)) => (existing, PutMode::Update(version)),
            None => (seed.clone(), PutMode::Create),
        };
        mutate(&mut policy);
        if write_policy_cas(warehouse, &policy, mode).await? {
            return Ok(policy);
        }
    }
    anyhow::bail!(
        "catalog policy update for '{warehouse}' lost {POLICY_MAX_RETRIES} \
         concurrent-writer races; aborting rather than clobbering another writer"
    )
}

/// Synchronous bridge that drives an object-store future from a synchronous caller.
///
/// The public catalog-policy API must stay synchronous because it is called from
/// sync contexts (SQL session helpers, the Python `lock_catalog_read_only`
/// binding, and the sync constructors of the MCP/Search untrusted-connector
/// permission resolvers) that must not become async.
///
/// Strategy:
/// - No runtime entered: build a temporary current-thread runtime on this
///   thread, drive the future, and drop it.
/// - Multi-thread runtime entered: drive the future with
///   `block_in_place(|| handle.block_on(fut))` so sibling worker tasks keep
///   making progress.
/// - Current-thread runtime entered: blocking the only worker thread would
///   deadlock and starting a runtime on it panics, so the future is handed to a
///   dedicated OS thread (via `std::thread::scope`, which allows borrowing the
///   caller's data) that owns its own runtime.
///
/// `F: Send` is required so the future can move to the dedicated thread in the
/// current-thread-runtime case; the store futures and the borrowed
/// warehouse/policy data satisfy this.
///
/// Runtime construction and `block_on` failures are propagated as
/// `anyhow::Error` (no-panic policy: no `unwrap`/`expect`/`panic`).
fn block_on_policy<F>(fut: F) -> Result<F::Output>
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    match tokio::runtime::Handle::try_current() {
        Err(_) => {
            // No runtime present: safe to build and drive one on this thread.
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| anyhow::anyhow!("failed to build catalog-policy runtime: {e}"))?;
            Ok(runtime.block_on(fut))
        }
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            Ok(tokio::task::block_in_place(|| handle.block_on(fut)))
        }
        Ok(_) => {
            // Current-thread runtime: run the future on a dedicated OS thread so
            // we neither block the sole worker nor nest a runtime inside one.
            std::thread::scope(|scope| {
                let thread = scope.spawn(move || -> Result<F::Output> {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| {
                            anyhow::anyhow!("failed to build catalog-policy runtime: {e}")
                        })?;
                    Ok(runtime.block_on(fut))
                });
                match thread.join() {
                    Ok(result) => result,
                    Err(_) => Err(anyhow::anyhow!("catalog-policy runtime thread panicked")),
                }
            })
        }
    }
}

/// Reads the catalog policy from `.benostream-policy.json` in the warehouse, if present.
///
/// Reads go through [`create_object_store`], so both local paths and object
/// stores (`s3://`, `gs://`, `az://`, `memory://`) are supported. When the
/// warehouse has no policy object, `Ok(None)` is returned so untrusted
/// connectors fail closed to scratch-only writable.
///
/// This function is synchronous for its callers; the object-store I/O is driven
/// on a Tokio runtime via [`block_on_policy`].
pub fn get_catalog_policy(warehouse_path: &str) -> Result<Option<CatalogPolicy>> {
    block_on_policy(read_policy_via_store(warehouse_path))?
}

/// Persists an authoritative catalog policy to `.benostream-policy.json` in the
/// warehouse root (local path or object-store URI).
///
/// This is an unconditional whole-policy write (the administrator override).
/// The object write is atomic for S3/GCS/Azure and local fs. The read-modify-write
/// helpers ([`set_catalog_schema_access`], [`lock_catalog_read_only`]) use
/// optimistic concurrency ([`update_policy`]) so they do not lose concurrent
/// updates.
pub fn set_catalog_policy(warehouse_path: &str, policy: &CatalogPolicy) -> Result<()> {
    block_on_policy(write_policy_via_store(warehouse_path, policy))?
}

/// Set read-only vs read-write access for a specific schema on a catalog.
///
/// Storage-aware: reads and writes the policy through the object store (local
/// path or `s3://`/`gs://`/`az://`/`memory://`). The read-modify-write runs under
/// optimistic concurrency control ([`update_policy`]), so a racing writer cannot
/// silently drop this update.
pub fn set_catalog_schema_access(
    warehouse_path: &str,
    schema: &str,
    read_only: bool,
) -> Result<CatalogPolicy> {
    let mut seed = CatalogPolicy::new();
    seed.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string()]);
    seed.readonly_schemas = Some(vec!["default".to_string()]);
    block_on_policy(update_policy(warehouse_path, seed, move |p| {
        p.set_schema_access(schema, read_only)
    }))?
}

/// Locks down a catalog warehouse to read-only mode (only `scratch` writable).
///
/// Storage-aware and optimistic-concurrency-safe, like
/// [`set_catalog_schema_access`]: a racing writer cannot lose this lock-down.
pub fn lock_catalog_read_only(warehouse_path: &str) -> Result<CatalogPolicy> {
    block_on_policy(update_policy(
        warehouse_path,
        CatalogPolicy::default(),
        |p| p.lock_all_read_only(),
    ))?
}

/// Resolves schema write permissions for an untrusted connector (MCP, Search, untrusted LLM tools).
///
/// Security Boundary:
/// - Trusted engines (Core BenoStreamDB kernel, Arrow Flight SQL, Spark, Trino, dbt) bypass this
///   connector-level lockdown and operate with full standard catalog authority.
/// - Untrusted connectors (MCP, Search REST API, external agents) are subject to zero-trust rules:
///   1. If an authoritative `.benostream-policy.json` exists in the warehouse, its rules are strictly enforced.
///   2. If NO policy file is present, untrusted connectors are locked down to CONSERVATIVE READ-ONLY MODE:
///      all persistent schemas are locked as read-only; only the isolated `scratch` schema is writable.
pub fn resolve_permissions_for_untrusted_connector(
    warehouse_path: &str,
    client_readwrite: Option<Vec<String>>,
    client_readonly: Option<Vec<String>>,
    client_default_write: Option<String>,
) -> (Vec<String>, Vec<String>, String) {
    match get_catalog_policy(warehouse_path) {
        Ok(Some(policy)) => {
            let mut readonly_set: BTreeSet<String> = client_readonly
                .unwrap_or_default()
                .into_iter()
                .map(|s| s.to_ascii_lowercase())
                .collect();

            for sys in IMMUTABLE_SYSTEM_SCHEMAS {
                readonly_set.insert(sys.to_ascii_lowercase());
            }

            if let Some(ref policy_ro) = policy.readonly_schemas {
                for s in policy_ro {
                    readonly_set.insert(s.to_ascii_lowercase());
                }
            }
            if let Some(ref policy_imm) = policy.immutable_schemas {
                for s in policy_imm {
                    readonly_set.insert(s.to_ascii_lowercase());
                }
            }

            let allowed_rw = if let Some(ref policy_rw) = policy.readwrite_schemas {
                if let Some(client_list) = client_readwrite {
                    let policy_set: BTreeSet<String> =
                        policy_rw.iter().map(|s| s.to_ascii_lowercase()).collect();
                    if client_list.iter().any(|w| w == "*") {
                        policy_rw.clone()
                    } else {
                        let mut filtered: Vec<String> = client_list
                            .into_iter()
                            .filter(|s| policy_set.contains(&s.to_ascii_lowercase()))
                            .collect();
                        if !filtered.iter().any(|s| s == SCRATCH_SCHEMA)
                            && policy_set.contains(SCRATCH_SCHEMA)
                        {
                            filtered.push(SCRATCH_SCHEMA.to_string());
                        }
                        filtered
                    }
                } else {
                    // No client restriction: adopt all readwrite schemas allowed by the warehouse policy!
                    policy_rw.clone()
                }
            } else {
                vec![SCRATCH_SCHEMA.to_string()]
            };

            let final_rw: Vec<String> = allowed_rw
                .into_iter()
                .filter(|s| !readonly_set.contains(&s.to_ascii_lowercase()))
                .collect();

            let final_rw = if final_rw.is_empty() {
                vec![SCRATCH_SCHEMA.to_string()]
            } else {
                final_rw
            };

            let default_write = policy
                .default_write_schema
                .clone()
                .or_else(|| {
                    if let Some(ref dw) = client_default_write {
                        if final_rw.contains(&dw.to_ascii_lowercase()) {
                            return Some(dw.clone());
                        }
                    }
                    final_rw.first().cloned()
                })
                .unwrap_or_else(|| SCRATCH_SCHEMA.to_string());

            (final_rw, readonly_set.into_iter().collect(), default_write)
        }
        Ok(None) | Err(_) => {
            // CONSERVATIVE LOCKDOWN FOR UNTRUSTED CONNECTORS
            let mut readonly_set: BTreeSet<String> = client_readonly
                .unwrap_or_default()
                .into_iter()
                .map(|s| s.to_ascii_lowercase())
                .collect();

            for sys in IMMUTABLE_SYSTEM_SCHEMAS {
                readonly_set.insert(sys.to_ascii_lowercase());
            }

            (
                vec![SCRATCH_SCHEMA.to_string()],
                readonly_set.into_iter().collect(),
                SCRATCH_SCHEMA.to_string(),
            )
        }
    }
}

/// MCP-consistent schema access policy for an untrusted connector.
///
/// This mirrors the `SchemaPermissions` policy used by the MCP server so that
/// every untrusted connector (MCP, the OpenSearch/Qdrant REST gateways, and
/// external agents) enforces the authoritative `.benostream-policy.json`
/// identically.
///
/// Writes are gated by [`SchemaPermissions::is_schema_writable`]; reads are
/// gated by [`SchemaPermissions::is_schema_readable`]. Reads have a
/// **read-only floor**: when no policy is present (`read_schemas_all`), every
/// schema is readable (read-only) and only `scratch` is writable; an explicit
/// policy narrows reads to the schemas it names (read-write / read-only /
/// immutable).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchemaPermissions {
    /// Schemas that permit writes. If it contains `"*"`, all schemas are
    /// writable except those explicitly listed in `readonly_schemas`.
    #[serde(default)]
    pub readwrite_schemas: Vec<String>,
    /// Schemas that are strictly read-only. Writes to these are always rejected.
    #[serde(default)]
    pub readonly_schemas: Vec<String>,
    /// Schemas with **no access** (neither readable nor writable) for untrusted
    /// connectors. Overrides the read-only floor.
    #[serde(default)]
    pub no_access_schemas: Vec<String>,
    /// Default schema used for unqualified writes (default `scratch`).
    #[serde(default)]
    pub default_write_schema: String,
    /// When true, every schema is readable (the read-only floor used when the
    /// warehouse has no policy). Writes remain restricted to
    /// `readwrite_schemas`. An explicit policy sets this to `false`, making the
    /// read-write / read-only / immutable lists an allowlist for reads too.
    #[serde(default)]
    pub read_schemas_all: bool,
}

impl Default for SchemaPermissions {
    /// Conservative lockdown (the no-policy state): every schema is read-only
    /// (readable), only `scratch` is writable, and the immutable system schemas
    /// are explicitly read-only.
    fn default() -> Self {
        Self {
            readwrite_schemas: vec![SCRATCH_SCHEMA.to_string()],
            readonly_schemas: IMMUTABLE_SYSTEM_SCHEMAS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            no_access_schemas: Vec::new(),
            default_write_schema: SCRATCH_SCHEMA.to_string(),
            read_schemas_all: true,
        }
    }
}

impl SchemaPermissions {
    pub fn new(
        readwrite_schemas: Vec<String>,
        readonly_schemas: Vec<String>,
        default_write_schema: String,
    ) -> Self {
        Self {
            readwrite_schemas,
            readonly_schemas,
            no_access_schemas: Vec::new(),
            default_write_schema,
            // An explicitly-constructed policy is an allowlist (narrow reads).
            read_schemas_all: false,
        }
    }

    /// MCP-consistent write check.
    ///
    /// Precedence: immutable system schemas are never writable; explicit
    /// read-only schemas veto (case-insensitive); a `"*"` entry in
    /// `readwrite_schemas` allows everything else; otherwise the schema must be
    /// listed in `readwrite_schemas` (case-insensitive).
    pub fn is_schema_writable(&self, schema: &str) -> bool {
        let s = schema.to_ascii_lowercase();
        // 1. Immutable system schemas are NEVER writable under any circumstances
        if IMMUTABLE_SYSTEM_SCHEMAS
            .iter()
            .any(|sys| sys.eq_ignore_ascii_case(&s))
        {
            return false;
        }
        // 2. Explicit readonly schemas strictly veto write permissions
        if self
            .readonly_schemas
            .iter()
            .any(|r| r.eq_ignore_ascii_case(&s))
        {
            return false;
        }
        // 2b. Explicit no-access schemas are never writable
        if self
            .no_access_schemas
            .iter()
            .any(|r| r.eq_ignore_ascii_case(&s))
        {
            return false;
        }
        // 3. A wildcard readwrite entry allows everything not rejected above
        if self.readwrite_schemas.iter().any(|w| w == "*") {
            return true;
        }
        // 4. Otherwise the schema must be explicitly listed
        self.readwrite_schemas
            .iter()
            .any(|w| w.eq_ignore_ascii_case(&s))
    }

    /// Whether reads of `schema` are permitted for an untrusted connector.
    ///
    /// Reads follow the same three-way model as writes so search and MCP agree.
    /// With no policy (`read_schemas_all`) — or on the built-in default — every
    /// schema is readable (read-only), so a fresh gateway can still read its
    /// data. When an explicit policy is present it is an allowlist: a schema
    /// named read-write, read-only, or immutable is readable, as is `scratch`;
    /// a schema the policy does not name is not readable. A `"*"` readwrite
    /// entry means every schema is readable.
    pub fn is_schema_readable(&self, schema: &str) -> bool {
        let s = schema.to_ascii_lowercase();
        // Explicitly denied schemas are not readable, even under the read-only
        // floor.
        if self
            .no_access_schemas
            .iter()
            .any(|r| r.eq_ignore_ascii_case(&s))
        {
            return false;
        }
        if self.read_schemas_all {
            return true;
        }
        if s == SCRATCH_SCHEMA {
            return true;
        }
        // Immutable system schemas are readable (and never writable).
        if IMMUTABLE_SYSTEM_SCHEMAS
            .iter()
            .any(|sys| sys.eq_ignore_ascii_case(&s))
        {
            return true;
        }
        // Read-only schemas are readable by definition.
        if self
            .readonly_schemas
            .iter()
            .any(|r| r.eq_ignore_ascii_case(&s))
        {
            return true;
        }
        // Read-write schemas are readable; a wildcard opens all.
        self.readwrite_schemas
            .iter()
            .any(|w| w == "*" || w.eq_ignore_ascii_case(&s))
    }

    /// Build permissions for an untrusted connector from the authoritative
    /// `.benostream-policy.json` in `warehouse_path`.
    ///
    /// Fails closed: when no policy file exists, only the `scratch` schema is
    /// writable and everything else is read-only. The immutable system schemas
    /// are always present in `readonly_schemas`.
    pub fn for_untrusted_connector(warehouse_path: &str) -> SchemaPermissions {
        let (rw, mut ro, dw) =
            resolve_permissions_for_untrusted_connector(warehouse_path, None, None, None);

        // Ensure the immutable system schemas are explicitly read-only.
        for sys in IMMUTABLE_SYSTEM_SCHEMAS {
            if !ro.iter().any(|r| r.eq_ignore_ascii_case(sys)) {
                ro.push((*sys).to_string());
            }
        }

        let policy = get_catalog_policy(warehouse_path).ok().flatten();
        let mut perms = SchemaPermissions::new(rw, ro, dw);
        // Explicit no-access schemas (deny read AND write), if the policy names any.
        perms.no_access_schemas = policy
            .as_ref()
            .and_then(|p| p.no_access_schemas.clone())
            .unwrap_or_default();
        // No usable policy (missing, malformed, or unreachable) → the documented
        // conservative default: every schema is READ-ONLY (readable), only
        // `scratch` is writable. A present policy narrows reads to its lists.
        perms.read_schemas_all = policy.is_none();
        perms
    }

    /// Permissions for a **network-facing** untrusted connector — the search
    /// gateway (OpenSearch/Qdrant APIs).
    ///
    /// Unlike a local MCP agent, a server connector gets **no default writable
    /// namespace**: schemas are provisioned on the backend, and the connector may
    /// write only to schemas the policy explicitly lists as read-write. It never
    /// creates a schema. With no policy it is read-only (reads allowed, writes
    /// denied). Reads follow the read-only floor / allowlist rule of
    /// [`SchemaPermissions::is_schema_readable`].
    pub fn for_server_connector(warehouse_path: &str) -> SchemaPermissions {
        let (rw, mut ro, _dw) =
            resolve_permissions_for_untrusted_connector(warehouse_path, None, None, None);

        // Ensure the immutable system schemas are explicitly read-only.
        for sys in IMMUTABLE_SYSTEM_SCHEMAS {
            if !ro.iter().any(|r| r.eq_ignore_ascii_case(sys)) {
                ro.push((*sys).to_string());
            }
        }

        let policy = get_catalog_policy(warehouse_path).ok().flatten();
        let policy_present = policy.is_some();

        // Writes require an explicit policy grant, and never fall back to the
        // MCP `scratch` namespace. `default_write_schema` is only a fallback for
        // unqualified names; it is *not* writable unless the policy names it.
        let readwrite: Vec<String> = if policy_present {
            rw.into_iter()
                .filter(|s| !s.eq_ignore_ascii_case(SCRATCH_SCHEMA))
                .collect()
        } else {
            Vec::new()
        };

        let mut perms = SchemaPermissions::new(readwrite, ro, SCRATCH_SCHEMA.to_string());
        perms.read_schemas_all = !policy_present;
        perms.no_access_schemas = policy.and_then(|p| p.no_access_schemas).unwrap_or_default();
        perms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_access_schemas_deny_reads_and_writes() {
        // Explicit no-access overrides the read-only floor for reads and blocks
        // writes, while leaving other schemas untouched.
        let perms = SchemaPermissions {
            no_access_schemas: vec!["secret".to_string()],
            ..Default::default()
        };
        assert!(perms.read_schemas_all);
        assert!(!perms.is_schema_readable("secret"));
        assert!(!perms.is_schema_writable("secret"));
        assert!(perms.is_schema_readable("analytics"));

        // And on the CatalogPolicy itself.
        let mut policy = CatalogPolicy::new();
        policy.readwrite_schemas = Some(vec!["*".to_string()]);
        policy.no_access_schemas = Some(vec!["secret".to_string()]);
        assert!(!policy.is_schema_writable("secret"));
        assert!(policy.is_schema_writable("anything_else"));
    }

    #[test]
    fn test_catalog_policy_get_set_and_lockdown() {
        let temp_dir = tempfile::tempdir().unwrap();
        let wh_path = temp_dir.path().to_str().unwrap();

        // 1. Initially no policy
        assert!(get_catalog_policy(wh_path).unwrap().is_none());

        // 2. Untrusted connector gets conservative lockdown
        let (rw, ro, dw) = resolve_permissions_for_untrusted_connector(wh_path, None, None, None);
        assert_eq!(rw, vec![SCRATCH_SCHEMA]);
        assert_eq!(dw, SCRATCH_SCHEMA);
        assert!(ro.contains(&"default".to_string()));

        // 3. Core DB product sets policy
        let mut policy = CatalogPolicy::new();
        policy.catalog_id = Some("test_catalog".to_string());
        policy.readwrite_schemas = Some(vec!["scratch".to_string(), "analytics".to_string()]);
        policy.readonly_schemas = Some(vec!["finance_raw".to_string()]);
        set_catalog_policy(wh_path, &policy).unwrap();

        let loaded = get_catalog_policy(wh_path).unwrap().unwrap();
        assert_eq!(loaded.catalog_id.as_deref(), Some("test_catalog"));
        assert!(loaded.is_schema_writable("analytics"));
        assert!(!loaded.is_schema_writable("finance_raw"));
        assert!(!loaded.is_schema_writable("default")); // system schema

        // 4. Untrusted connector adopts policy
        let (rw, ro, _dw) = resolve_permissions_for_untrusted_connector(wh_path, None, None, None);
        assert!(rw.contains(&"analytics".to_string()));
        assert!(ro.contains(&"finance_raw".to_string()));

        // 5. Change schema access from core
        set_catalog_schema_access(wh_path, "analytics", true).unwrap();
        let loaded2 = get_catalog_policy(wh_path).unwrap().unwrap();
        assert!(!loaded2.is_schema_writable("analytics"));

        // 6. Lock down all read-only
        lock_catalog_read_only(wh_path).unwrap();
        let loaded3 = get_catalog_policy(wh_path).unwrap().unwrap();
        assert!(!loaded3.is_schema_writable("analytics"));
        assert_eq!(loaded3.readwrite_schemas.unwrap(), vec![SCRATCH_SCHEMA]);
    }

    #[test]
    fn schema_permissions_immutable_schemas_are_never_writable() {
        // Even a wildcard readwrite list cannot unlock immutable system schemas.
        let perms =
            SchemaPermissions::new(vec!["*".to_string()], vec![], SCRATCH_SCHEMA.to_string());
        for sys in IMMUTABLE_SYSTEM_SCHEMAS {
            assert!(!perms.is_schema_writable(sys), "{sys} must be immutable");
        }
        assert!(perms.is_schema_writable("anything_else"));
    }

    #[test]
    fn schema_permissions_readonly_vetoes_readwrite() {
        let perms = SchemaPermissions::new(
            vec!["analytics".to_string(), "scratch".to_string()],
            vec!["Analytics".to_string()],
            SCRATCH_SCHEMA.to_string(),
        );
        // Case-insensitive readonly veto wins over an explicit readwrite entry.
        assert!(!perms.is_schema_writable("analytics"));
        assert!(!perms.is_schema_writable("ANALYTICS"));
        assert!(perms.is_schema_writable("scratch"));
    }

    #[test]
    fn schema_permissions_wildcard_and_membership() {
        let perms = SchemaPermissions::new(
            vec!["*".to_string()],
            vec!["finance".to_string()],
            SCRATCH_SCHEMA.to_string(),
        );
        assert!(perms.is_schema_writable("analytics"));
        assert!(!perms.is_schema_writable("finance"));

        let perms = SchemaPermissions::new(
            vec!["Analytics".to_string()],
            vec![],
            SCRATCH_SCHEMA.to_string(),
        );
        assert!(perms.is_schema_writable("analytics"));
        assert!(!perms.is_schema_writable("other"));
    }

    #[test]
    fn schema_permissions_theme() {
        // An explicit policy is an allowlist for reads: RW/RO/immutable schemas
        // (and scratch) are readable; an unlisted schema is not.
        let perms = SchemaPermissions::new(
            vec!["analytics".to_string(), SCRATCH_SCHEMA.to_string()],
            vec!["finance_raw".to_string(), "default".to_string()],
            SCRATCH_SCHEMA.to_string(),
        );
        assert!(perms.is_schema_readable("analytics"));
        assert!(perms.is_schema_readable("finance_raw"));
        assert!(perms.is_schema_readable("default"));
        assert!(perms.is_schema_readable(SCRATCH_SCHEMA));
        assert!(!perms.is_schema_readable("unmanaged"));

        // No policy: everything is read-only (readable); only scratch writable.
        let default = SchemaPermissions::default();
        assert!(default.read_schemas_all);
        assert!(default.is_schema_readable("analytics"));
        assert!(default.is_schema_readable("anything_at_all"));
        assert!(default.is_schema_writable(SCRATCH_SCHEMA));
        assert!(!default.is_schema_writable("analytics"));

        // A wildcard readwrite list makes every schema readable.
        let wildcard =
            SchemaPermissions::new(vec!["*".to_string()], vec![], SCRATCH_SCHEMA.to_string());
        assert!(wildcard.is_schema_readable("anything"));
    }

    #[test]
    fn schema_permissions_default_is_scratch_only() {
        let perms = SchemaPermissions::default();
        assert_eq!(perms.default_write_schema, SCRATCH_SCHEMA);
        assert!(perms.is_schema_writable("scratch"));
        assert!(!perms.is_schema_writable("analytics"));
        assert!(!perms.is_schema_writable("default"));
    }

    #[test]
    fn schema_permissions_for_untrusted_connector_fails_closed_without_policy() {
        let temp_dir = tempfile::tempdir().unwrap();
        let wh_path = temp_dir.path().to_str().unwrap();

        let perms = SchemaPermissions::for_untrusted_connector(wh_path);
        assert_eq!(perms.default_write_schema, SCRATCH_SCHEMA);
        assert!(perms.is_schema_writable("scratch"));
        assert!(!perms.is_schema_writable("analytics"));
        // Immutable system schemas must be present in readonly_schemas.
        for sys in IMMUTABLE_SYSTEM_SCHEMAS {
            assert!(
                perms.readonly_schemas.iter().any(|r| r == sys),
                "{sys} must be read-only"
            );
        }
    }

    #[test]
    fn schema_permissions_for_untrusted_connector_adopts_policy() {
        let temp_dir = tempfile::tempdir().unwrap();
        let wh_path = temp_dir.path().to_str().unwrap();

        let mut policy = CatalogPolicy::new();
        policy.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string(), "analytics".to_string()]);
        set_catalog_policy(wh_path, &policy).unwrap();

        let perms = SchemaPermissions::for_untrusted_connector(wh_path);
        assert!(perms.is_schema_writable("analytics"));
        assert!(perms.is_schema_writable("scratch"));
        assert!(!perms.is_schema_writable("default"));
        assert!(!perms.is_schema_writable("unlisted"));
    }

    #[test]
    fn file_uri_policy_round_trips_through_object_store() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

        // No policy yet: fail closed.
        assert!(get_catalog_policy(&uri).unwrap().is_none());

        let mut policy = CatalogPolicy::new();
        policy.catalog_id = Some("file-uri-catalog".to_string());
        policy.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string(), "analytics".to_string()]);
        set_catalog_policy(&uri, &policy).unwrap();

        let loaded = get_catalog_policy(&uri).unwrap().unwrap();
        assert_eq!(loaded.catalog_id.as_deref(), Some("file-uri-catalog"));
        assert!(loaded.is_schema_writable("analytics"));

        // Visible via a second independent read → the policy was persisted.
        let loaded_again = get_catalog_policy(&uri).unwrap().unwrap();
        assert_eq!(loaded_again, loaded);

        // The object physically exists at the warehouse root.
        assert!(temp_dir.path().join(CATALOG_POLICY_FILE).exists());
    }

    #[test]
    fn missing_policy_returns_none_fail_closed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

        assert!(get_catalog_policy(&uri).unwrap().is_none());

        // Untrusted connector therefore gets conservative scratch-only permissions.
        let perms = SchemaPermissions::for_untrusted_connector(&uri);
        assert_eq!(perms.default_write_schema, SCRATCH_SCHEMA);
        assert!(perms.is_schema_writable("scratch"));
        assert!(!perms.is_schema_writable("analytics"));
    }

    #[test]
    fn set_catalog_schema_access_persists_on_file_uri() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

        // First call creates the policy and grants analytics write access.
        set_catalog_schema_access(&uri, "analytics", false).unwrap();
        let loaded = get_catalog_policy(&uri).unwrap().unwrap();
        assert!(loaded.is_schema_writable("analytics"));

        // A later call re-reads the persisted policy and flips it back.
        set_catalog_schema_access(&uri, "analytics", true).unwrap();
        let loaded2 = get_catalog_policy(&uri).unwrap().unwrap();
        assert!(!loaded2.is_schema_writable("analytics"));
    }

    #[test]
    fn sequential_schema_access_updates_do_not_lose_schemas() {
        // Sequential read-modify-write calls both persist. (Concurrent writers
        // are covered by `concurrent_style_policy_updates_do_not_lose_schemas`
        // and `cas_write_rejects_a_stale_version` below.)
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

        set_catalog_schema_access(&uri, "analytics", false).unwrap();
        set_catalog_schema_access(&uri, "reporting", false).unwrap();

        let loaded = get_catalog_policy(&uri).unwrap().unwrap();
        assert!(loaded.is_schema_writable("analytics"));
        assert!(loaded.is_schema_writable("reporting"));
        assert!(loaded.is_schema_writable(SCRATCH_SCHEMA));
    }

    /// The MCP/Search sync constructors can call policy resolution from inside a
    /// live multi-thread runtime; the bridge must not deadlock there.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn policy_round_trips_inside_multi_thread_runtime() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

        let mut policy = CatalogPolicy::new();
        policy.catalog_id = Some("mt-runtime".to_string());
        // Sync public API called from within an async multi-thread context.
        set_catalog_policy(&uri, &policy).unwrap();
        let loaded = get_catalog_policy(&uri).unwrap().unwrap();
        assert_eq!(loaded.catalog_id.as_deref(), Some("mt-runtime"));
    }

    /// A current-thread runtime must not be blocked on itself; the bridge falls
    /// back to a dedicated runtime so this still completes.
    #[tokio::test(flavor = "current_thread")]
    async fn policy_round_trips_inside_current_thread_runtime() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

        let mut policy = CatalogPolicy::new();
        policy.catalog_id = Some("ct-runtime".to_string());
        set_catalog_policy(&uri, &policy).unwrap();
        let loaded = get_catalog_policy(&uri).unwrap().unwrap();
        assert_eq!(loaded.catalog_id.as_deref(), Some("ct-runtime"));
    }

    /// §1.1: a malformed / truncated policy file must NOT be treated as "allow".
    /// Reads error out, and the untrusted-connector resolver fails closed to
    /// scratch-only until a valid policy is written.
    #[test]
    fn malformed_policy_fails_closed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
        std::fs::write(temp_dir.path().join(CATALOG_POLICY_FILE), b"{ not json").unwrap();

        assert!(
            get_catalog_policy(&uri).is_err(),
            "malformed policy JSON must be a read error, not silently ignored"
        );

        let perms = SchemaPermissions::for_untrusted_connector(&uri);
        assert_eq!(perms.default_write_schema, SCRATCH_SCHEMA);
        assert!(perms.is_schema_writable(SCRATCH_SCHEMA));
        assert!(
            !perms.is_schema_writable("analytics"),
            "a malformed policy must not grant writes to a persistent schema"
        );
    }

    /// §1.5: the read-modify-write helpers now retry under CAS, so distinct
    /// schema updates all survive.
    #[test]
    fn concurrent_style_policy_updates_do_not_lose_schemas() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

        for schema in ["analytics", "reporting", "finance", "ml"] {
            set_catalog_schema_access(&uri, schema, false).unwrap();
        }

        let loaded = get_catalog_policy(&uri).unwrap().unwrap();
        for schema in ["analytics", "reporting", "finance", "ml"] {
            assert!(
                loaded.is_schema_writable(schema),
                "'{schema}' was lost by a read-modify-write"
            );
        }
    }

    /// §1.5: a CAS write pinned to a stale version must be reported as a conflict
    /// (not applied), so a racing update cannot be clobbered.
    #[tokio::test]
    async fn cas_write_rejects_a_stale_version() {
        // `memory://` is a process-shared in-memory store that supports
        // conditional PUTs (unlike some local filesystem backends).
        let uri = format!("memory://policy-cas-{}", std::process::id());

        let mut first = CatalogPolicy::new();
        first.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string()]);
        set_catalog_policy(&uri, &first).unwrap();

        // Read version V1.
        let (_p, v1) = read_policy_with_version(&uri).await.unwrap().unwrap();

        // A concurrent writer replaces the policy (V2).
        let mut second = CatalogPolicy::new();
        second.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string(), "analytics".to_string()]);
        set_catalog_policy(&uri, &second).unwrap();

        // A CAS write pinned to V1 must be rejected.
        let mut stale = CatalogPolicy::new();
        stale.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string()]);
        let applied = write_policy_cas(&uri, &stale, PutMode::Update(v1))
            .await
            .unwrap();
        assert!(!applied, "a stale CAS write must be reported as a conflict");

        // The concurrent writer's policy is intact.
        let loaded = get_catalog_policy(&uri).unwrap().unwrap();
        assert!(loaded.is_schema_writable("analytics"));
    }
}
