// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

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

/// Reads the catalog policy from `.benostream-policy.json` in the warehouse directory, if present.
pub fn get_catalog_policy(warehouse_path: &str) -> Result<Option<CatalogPolicy>> {
    let p = Path::new(warehouse_path).join(CATALOG_POLICY_FILE);
    if p.exists() {
        let content = std::fs::read_to_string(&p)?;
        let policy: CatalogPolicy = serde_json::from_str(&content)?;
        Ok(Some(policy))
    } else {
        Ok(None)
    }
}

/// Persists an authoritative catalog policy to `.benostream-policy.json` in the warehouse root.
pub fn set_catalog_policy(warehouse_path: &str, policy: &CatalogPolicy) -> Result<()> {
    std::fs::create_dir_all(warehouse_path)?;
    let p = Path::new(warehouse_path).join(CATALOG_POLICY_FILE);
    let content = serde_json::to_string_pretty(policy)?;
    std::fs::write(&p, content)?;
    Ok(())
}

/// Helper function to set read-only vs read-write access for a specific schema on a catalog.
pub fn set_catalog_schema_access(
    warehouse_path: &str,
    schema: &str,
    read_only: bool,
) -> Result<CatalogPolicy> {
    let mut policy = get_catalog_policy(warehouse_path)?.unwrap_or_else(|| {
        let mut p = CatalogPolicy::new();
        p.readwrite_schemas = Some(vec![SCRATCH_SCHEMA.to_string()]);
        p.readonly_schemas = Some(vec!["default".to_string()]);
        p
    });
    policy.set_schema_access(schema, read_only);
    set_catalog_policy(warehouse_path, &policy)?;
    Ok(policy)
}

/// Locks down a catalog warehouse to read-only mode (only scratch schema writable).
pub fn lock_catalog_read_only(warehouse_path: &str) -> Result<CatalogPolicy> {
    let mut policy = get_catalog_policy(warehouse_path)?.unwrap_or_default();
    policy.lock_all_read_only();
    set_catalog_policy(warehouse_path, &policy)?;
    Ok(policy)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
