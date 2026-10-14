use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::sync::RwLock;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{debug, error, info};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use benostreamdb::core::sql::session::BenoStreamSession;

/// The default schema the MCP server creates objects in if unqualified.
pub const SCRATCH_SCHEMA: &str = "scratch";

/// Derives the default warehouse directory path.
/// If `BSDB_WAREHOUSE` is already set, returns its value.
/// If `BSDB_EPHEMERAL` is set or `--ephemeral` argument is present, generates a fresh directory under the system temp directory.
/// In unit tests (`cfg!(test)`), defaults to ephemeral unless `BSDB_WAREHOUSE` is explicitly configured.
/// Otherwise, returns a persistent warehouse path under `$XDG_DATA_HOME/benostreamdb/mcp_warehouse` or `~/.benostreamdb/mcp_warehouse`.
pub fn resolve_default_warehouse() -> String {
    let is_test = cfg!(test);
    let is_ephemeral = is_test
        || std::env::var("BSDB_EPHEMERAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        || std::env::args().any(|arg| arg == "--ephemeral");

    if !is_test {
        if let Ok(w) = std::env::var("BSDB_WAREHOUSE") {
            if !w.trim().is_empty() {
                if validate_local_path(&w, "BSDB_WAREHOUSE").is_ok() {
                    let _ = std::fs::create_dir_all(&w);
                    return w;
                } else {
                    error!("Ignoring remote or non-local BSDB_WAREHOUSE: {}", w);
                }
            }
        }
    }

    let warehouse_path = if is_ephemeral {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_dir = std::env::temp_dir().join(format!(
            "benostreamdb_mcp_{}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            id
        ));
        let _ = std::fs::create_dir_all(&temp_dir);
        temp_dir
    } else {
        let base_dir = std::env::var("XDG_DATA_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| {
                std::env::var("HOME").map(|h| std::path::PathBuf::from(h).join(".benostreamdb"))
            })
            .unwrap_or_else(|_| std::env::temp_dir().join(".benostreamdb"));
        let warehouse_dir = base_dir.join("mcp_warehouse");
        let _ = std::fs::create_dir_all(&warehouse_dir);
        warehouse_dir
    };

    warehouse_path.to_string_lossy().to_string()
}

/// Validates that a path is strictly a local filesystem path.
/// Rejects remote URI schemes, network protocols, and hosted object store prefixes
/// (e.g. s3://, minio://, rustfs://, http://, https://, gcs://, gs://, abfs://, rest://, etc.).
pub fn validate_local_path(path: &str, field_name: &str) -> anyhow::Result<()> {
    let p = path.trim();
    if p.is_empty() {
        anyhow::bail!("{field_name} cannot be empty");
    }

    // Check for explicit URI schemes (e.g. s3://, minio://, rustfs://, http://, etc.)
    if let Some(idx) = p.find("://") {
        let scheme = &p[..idx].to_ascii_lowercase();
        if scheme != "file" {
            anyhow::bail!(
                "MCP storage is locked to local filesystem only. Remote/hosted object store or catalog URI scheme '{scheme}://' is strictly prohibited for {field_name} (e.g. MinIO, S3, RustFS, Iceberg REST, GCS)",
            );
        }
    }

    // Check for known cloud/object-store prefix schemes without :// (e.g. s3:, gs:, minio:)
    for prefix in &[
        "s3:", "gs:", "gcs:", "minio:", "rustfs:", "iceberg:", "rest:", "http:", "https:",
        "azure:", "abfs:", "abfss:", "oss:", "cos:",
    ] {
        if p.to_ascii_lowercase().starts_with(prefix) {
            anyhow::bail!(
                "MCP storage is locked to local filesystem only. Remote/hosted object store prefix '{prefix}' is strictly prohibited for {field_name}",
            );
        }
    }

    Ok(())
}

fn default_readwrite_schemas() -> Vec<String> {
    vec![SCRATCH_SCHEMA.to_string()]
}

fn default_readonly_schemas() -> Vec<String> {
    vec!["default".to_string()]
}

fn default_write_schema_val() -> String {
    SCRATCH_SCHEMA.to_string()
}

/// Helper deserializer that accepts either a JSON array of strings or a comma-separated string.
fn deserialize_optional_schema_list<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct SchemaListVisitor;

    impl<'de> serde::de::Visitor<'de> for SchemaListVisitor {
        type Value = Option<Vec<String>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a list of strings, a comma-separated string, or null")
        }

        fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_any(self)
        }

        fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            let list: Vec<String> = v
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            Ok(Some(list))
        }

        fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut items = Vec::new();
            while let Some(item) = seq.next_element::<String>()? {
                let trimmed = item.trim().to_ascii_lowercase();
                if !trimmed.is_empty() {
                    items.push(trimmed);
                }
            }
            Ok(Some(items))
        }
    }

    deserializer.deserialize_option(SchemaListVisitor)
}

/// Baseline system catalog schemas that are strictly immutable and can NEVER be configured as writable.
pub const IMMUTABLE_SYSTEM_SCHEMAS: &[&str] =
    &["default", "public", "information_schema", "sys", "system"];

/// Schema write permissions policy for the MCP server.
/// Governs which schemas can be mutated (tables created, records inserted, etc.)
/// and which schemas are locked as read-only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaPermissions {
    /// Schemas that permit writes (CREATE, INSERT, UPDATE, DELETE, ALTER, DROP, TRUNCATE, MERGE).
    /// If it contains "*", all schemas are writable except those explicitly in `readonly_schemas`.
    #[serde(default = "default_readwrite_schemas")]
    pub readwrite_schemas: Vec<String>,
    /// Schemas that are strictly read-only. Writes to these schemas are always rejected.
    #[serde(default = "default_readonly_schemas")]
    pub readonly_schemas: Vec<String>,
    /// The default schema used when creating scratch/user tables without an explicit schema qualifier.
    #[serde(default = "default_write_schema_val")]
    pub default_write_schema: String,
}

impl Default for SchemaPermissions {
    fn default() -> Self {
        Self {
            readwrite_schemas: default_readwrite_schemas(),
            readonly_schemas: default_readonly_schemas(),
            default_write_schema: default_write_schema_val(),
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
            default_write_schema,
        }
    }

    pub fn from_env() -> Self {
        let readwrite_schemas = if let Ok(val) = std::env::var("BSDB_READWRITE_SCHEMAS") {
            let list: Vec<String> = val
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            if list.is_empty() {
                vec![SCRATCH_SCHEMA.to_string()]
            } else {
                list
            }
        } else {
            vec![SCRATCH_SCHEMA.to_string()]
        };

        let readonly_schemas = if let Ok(val) = std::env::var("BSDB_READONLY_SCHEMAS") {
            let list: Vec<String> = val
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            if list.is_empty() {
                vec!["default".to_string()]
            } else {
                list
            }
        } else {
            vec!["default".to_string()]
        };

        let default_write_schema = std::env::var("BSDB_DEFAULT_WRITE_SCHEMA")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                readwrite_schemas
                    .iter()
                    .find(|s| *s != "*")
                    .cloned()
                    .unwrap_or_else(|| SCRATCH_SCHEMA.to_string())
            });

        Self {
            readwrite_schemas,
            readonly_schemas,
            default_write_schema,
        }
    }

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
        if self.readwrite_schemas.iter().any(|w| w == "*") {
            return true;
        }
        self.readwrite_schemas
            .iter()
            .any(|w| w.eq_ignore_ascii_case(&s))
    }
}

/// Global configuration for the BenoStreamDB MCP server.
/// Supports loading from JSON files, inline JSON strings (e.g. from client `settings.json`),
/// environment variables, and CLI arguments.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Warehouse directory path for table storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warehouse: Option<String>,
    /// Schemas that permit writes. Accepts a JSON array or a comma-separated string.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_schema_list"
    )]
    pub readwrite_schemas: Option<Vec<String>>,
    /// Schemas that are strictly read-only. Accepts a JSON array or a comma-separated string.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_schema_list"
    )]
    pub readonly_schemas: Option<Vec<String>>,
    /// Default schema for unqualified table creations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_write_schema: Option<String>,
    /// Whether to force ephemeral temporary directories instead of the persistent warehouse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephemeral: Option<bool>,
}

impl McpServerConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(json_str) = std::env::var("BSDB_MCP_CONFIG") {
            if let Ok(parsed) = serde_json::from_str::<Self>(&json_str) {
                config = parsed;
            }
        }
        if let Ok(w) = std::env::var("BSDB_WAREHOUSE") {
            if !w.trim().is_empty() {
                config.warehouse = Some(w);
            }
        }
        if let Ok(rw) = std::env::var("BSDB_READWRITE_SCHEMAS") {
            let list: Vec<String> = rw
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            if !list.is_empty() {
                config.readwrite_schemas = Some(list);
            }
        }
        if let Ok(ro) = std::env::var("BSDB_READONLY_SCHEMAS") {
            let list: Vec<String> = ro
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            if !list.is_empty() {
                config.readonly_schemas = Some(list);
            }
        }
        if let Ok(dw) = std::env::var("BSDB_DEFAULT_WRITE_SCHEMA") {
            if !dw.trim().is_empty() {
                config.default_write_schema = Some(dw.trim().to_ascii_lowercase());
            }
        }
        if let Ok(e) = std::env::var("BSDB_EPHEMERAL") {
            config.ephemeral = Some(e == "1" || e.eq_ignore_ascii_case("true"));
        }
        config
    }

    pub fn from_args(args: &[String]) -> anyhow::Result<Self> {
        let mut config = Self::default();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--warehouse" => {
                    i += 1;
                    if i < args.len() {
                        config.warehouse = Some(args[i].clone());
                    }
                }
                "--readwrite-schemas" => {
                    i += 1;
                    if i < args.len() {
                        let list: Vec<String> = args[i]
                            .split(',')
                            .map(|s| s.trim().to_ascii_lowercase())
                            .filter(|s| !s.is_empty())
                            .collect();
                        config.readwrite_schemas = Some(list);
                    }
                }
                "--readonly-schemas" => {
                    i += 1;
                    if i < args.len() {
                        let list: Vec<String> = args[i]
                            .split(',')
                            .map(|s| s.trim().to_ascii_lowercase())
                            .filter(|s| !s.is_empty())
                            .collect();
                        config.readonly_schemas = Some(list);
                    }
                }
                "--default-write-schema" => {
                    i += 1;
                    if i < args.len() {
                        config.default_write_schema = Some(args[i].trim().to_ascii_lowercase());
                    }
                }
                "--ephemeral" => {
                    config.ephemeral = Some(true);
                }
                "--config" => {
                    i += 1;
                    if i < args.len() {
                        let parsed: Self = serde_json::from_str(&args[i])
                            .map_err(|e| anyhow::anyhow!("Invalid --config JSON: {}", e))?;
                        config = config.merge(parsed);
                    }
                }
                "--config-file" => {
                    i += 1;
                    if i < args.len() {
                        let content = std::fs::read_to_string(&args[i]).map_err(|e| {
                            anyhow::anyhow!("Failed reading config file {}: {}", args[i], e)
                        })?;
                        let parsed: Self = serde_json::from_str(&content).map_err(|e| {
                            anyhow::anyhow!("Invalid JSON in config file {}: {}", args[i], e)
                        })?;
                        config = config.merge(parsed);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        Ok(config)
    }

    pub fn resolve_from_args_and_env(args: &[String]) -> anyhow::Result<Self> {
        let env_cfg = Self::from_env();
        let args_cfg = Self::from_args(args)?;
        Ok(env_cfg.merge(args_cfg))
    }

    pub fn merge(self, overlay: Self) -> Self {
        Self {
            warehouse: overlay.warehouse.or(self.warehouse),
            readwrite_schemas: overlay.readwrite_schemas.or(self.readwrite_schemas),
            readonly_schemas: overlay.readonly_schemas.or(self.readonly_schemas),
            default_write_schema: overlay.default_write_schema.or(self.default_write_schema),
            ephemeral: overlay.ephemeral.or(self.ephemeral),
        }
    }

    pub fn to_permissions(&self) -> SchemaPermissions {
        let readwrite_schemas = self
            .readwrite_schemas
            .clone()
            .unwrap_or_else(default_readwrite_schemas);
        let mut readonly_schemas = self
            .readonly_schemas
            .clone()
            .unwrap_or_else(default_readonly_schemas);
        for sys in IMMUTABLE_SYSTEM_SCHEMAS {
            let sys_str = sys.to_string();
            if !readonly_schemas
                .iter()
                .any(|r| r.eq_ignore_ascii_case(&sys_str))
            {
                readonly_schemas.push(sys_str);
            }
        }
        let default_write_schema = self.default_write_schema.clone().unwrap_or_else(|| {
            readwrite_schemas
                .iter()
                .find(|s| *s != "*")
                .cloned()
                .unwrap_or_else(default_write_schema_val)
        });

        SchemaPermissions {
            readwrite_schemas,
            readonly_schemas,
            default_write_schema,
        }
    }
}

/// Enforce the MCP schema write policy:
/// Writes (CREATE, INSERT, UPDATE, DELETE, ALTER, DROP, TRUNCATE, MERGE)
/// are permitted ONLY to schemas marked as read/write in `permissions`
/// and NEVER to schemas marked as read-only.
/// Reads (SELECT, COPY ... TO, DESCRIBE, EXPLAIN, …) are unrestricted across all schemas.
fn guard_schema_writes(query: &str, permissions: &SchemaPermissions) -> anyhow::Result<()> {
    use sqlparser::ast::{ObjectName, ObjectType, Statement, TableFactor};
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    let query_upper = query.to_ascii_uppercase();
    if query_upper.contains("LOCATION") {
        anyhow::bail!(
            "MCP write denied: LOCATION clauses are strictly prohibited in MCP SQL statements. Tables are managed strictly within the local warehouse"
        );
    }
    if query_upper.contains("CREATE EXTERNAL TABLE") {
        anyhow::bail!(
            "MCP write denied: CREATE EXTERNAL TABLE is strictly prohibited in MCP. Remote/external tables are not permitted; tables must reside in the local warehouse"
        );
    }
    if query_upper.contains("CREATE CATALOG")
        || query_upper.contains("ATTACH CATALOG")
        || query_upper.contains("ATTACH DATABASE")
        || query_upper.contains("MOUNT")
    {
        anyhow::bail!(
            "MCP write denied: external catalog/database mounting is strictly prohibited. Remote Iceberg, MinIO, S3, or RustFS catalogs cannot be attached via MCP"
        );
    }
    if query_upper.contains("SET BENOSTREAM.") || query_upper.contains("SET WAREHOUSE") {
        anyhow::bail!(
            "MCP write denied: mutating session settings via SQL SET is prohibited in MCP"
        );
    }

    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, query)
        .map_err(|e| anyhow::anyhow!("SQL parse error: {}", e))?;

    let check_table_writable = |name: &ObjectName| -> anyhow::Result<()> {
        if name.0.is_empty() {
            return Ok(());
        }
        if name.0.len() >= 3 {
            if let Some(cat_ident) = name.0[name.0.len() - 3].as_ident() {
                let cat = cat_ident.value.to_ascii_lowercase();
                if cat != "benostream" && cat != "datafusion" && cat != "default" {
                    anyhow::bail!(
                        "MCP write denied: writes to external catalog '{}' are prohibited (table '{}'). Only local schemas are writable",
                        cat,
                        name
                    );
                }
            }
        }
        let schema = if name.0.len() >= 2 {
            if let Some(ident) = name.0[name.0.len() - 2].as_ident() {
                ident.value.to_ascii_lowercase()
            } else {
                "".to_string()
            }
        } else {
            "".to_string()
        };

        if schema.is_empty() {
            anyhow::bail!(
                "MCP write statements must qualify the table with a writable schema (e.g. '{}.{}'). Allowed writable schemas: {:?}, locked schemas: {:?}",
                permissions.default_write_schema,
                name,
                permissions.readwrite_schemas,
                permissions.readonly_schemas
            );
        }

        if !permissions.is_schema_writable(&schema) {
            anyhow::bail!(
                "MCP write denied: schema '{}' is read-only or not in writable schemas (table '{}'). Allowed writable schemas: {:?}, locked schemas: {:?}",
                schema,
                name,
                permissions.readwrite_schemas,
                permissions.readonly_schemas
            );
        }
        Ok(())
    };

    let check_table_factor = |tf: &TableFactor| -> anyhow::Result<()> {
        if let TableFactor::Table { name, .. } = tf {
            check_table_writable(name)?;
        }
        Ok(())
    };

    for stmt in statements {
        match stmt {
            Statement::CreateSchema { schema_name, .. } => {
                let name = match schema_name {
                    sqlparser::ast::SchemaName::Simple(n) => n.to_string().to_ascii_lowercase(),
                    other => {
                        anyhow::bail!("Unsupported schema name: {:?}", other);
                    }
                };
                if name.is_empty() || !permissions.is_schema_writable(&name) {
                    anyhow::bail!(
                        "MCP may not create schema '{}': schema is not in writable schemas {:?}",
                        name,
                        permissions.readwrite_schemas
                    );
                }
            }
            Statement::CreateDatabase { .. } => {
                anyhow::bail!("MCP may not create databases");
            }
            Statement::CreateTable(cmd) => {
                if cmd.location.is_some() {
                    anyhow::bail!("MCP may not use LOCATION in CREATE TABLE");
                }
                check_table_writable(&cmd.name)?;
            }
            Statement::CreateView { name, .. } => check_table_writable(&name)?,
            Statement::Copy { .. } => {
                anyhow::bail!("MCP may not use COPY");
            }
            Statement::CreateIndex(cmd) => check_table_writable(&cmd.table_name)?,
            Statement::AlterTable { name, .. } => check_table_writable(&name)?,
            Statement::AlterIndex { name, .. } => check_table_writable(&name)?,
            Statement::Truncate { table_names, .. } => {
                for target in &table_names {
                    check_table_writable(&target.name)?;
                }
            }
            Statement::Insert(cmd) => {
                if let sqlparser::ast::TableObject::TableName(name) = &cmd.table {
                    check_table_writable(name)?;
                }
            }
            Statement::Update { table, .. } => check_table_factor(&table.relation)?,
            Statement::Delete(cmd) => {
                for name in &cmd.tables {
                    check_table_writable(name)?;
                }
                match &cmd.from {
                    sqlparser::ast::FromTable::WithFromKeyword(tables) => {
                        for t in tables {
                            check_table_factor(&t.relation)?;
                        }
                    }
                    sqlparser::ast::FromTable::WithoutKeyword(tables) => {
                        for t in tables {
                            check_table_factor(&t.relation)?;
                        }
                    }
                }
            }
            Statement::Merge { table, .. } => check_table_factor(&table)?,
            Statement::Drop {
                object_type, names, ..
            } => {
                if object_type == ObjectType::Index
                    || object_type == ObjectType::Table
                    || object_type == ObjectType::View
                {
                    for name in &names {
                        check_table_writable(name)?;
                    }
                } else if object_type == ObjectType::Schema {
                    for name in &names {
                        if let Some(ident) = name.0.last().and_then(|id| id.as_ident()) {
                            let schema = ident.value.to_ascii_lowercase();
                            if !permissions.is_schema_writable(&schema) {
                                anyhow::bail!("MCP may not drop read-only schema '{}'", schema);
                            }
                        }
                    }
                } else if object_type == ObjectType::Database {
                    anyhow::bail!("MCP may not drop databases");
                }
            }
            _ => {
                // other statements (like SELECT, DESCRIBE) are unrestricted reads
            }
        }
    }
    Ok(())
}

/// Enforce the default MCP write policy (scratch only).
#[allow(dead_code)]
pub fn guard_scratch_only(query: &str) -> anyhow::Result<()> {
    guard_schema_writes(query, &SchemaPermissions::default())
}

/// Validates that an identifier (table name, column name, or schema.table)
/// contains only alphanumeric characters and underscores, preventing SQL injection.
fn validate_sql_identifier(ident: &str) -> anyhow::Result<()> {
    let trimmed = ident.trim();
    if trimmed.is_empty() {
        anyhow::bail!("Identifier cannot be empty");
    }
    for part in trimmed.split('.') {
        if part.is_empty() {
            anyhow::bail!("Invalid identifier: empty component in '{}'", ident);
        }
        if !part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            anyhow::bail!(
                "Invalid identifier '{}': contains non-alphanumeric/underscore characters",
                ident
            );
        }
        if part.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            anyhow::bail!(
                "Invalid identifier '{}': component cannot start with a digit",
                ident
            );
        }
    }
    Ok(())
}

/// Escapes a string literal for safe interpolation into single-quoted SQL strings.
fn escape_sql_literal(s: &str) -> String {
    s.replace('\'', "''")
}

/// Sanitizes an identifier component (alphanumerics and underscores only).
fn sanitize_ident(name_part: &str) -> String {
    let mut sanitized = String::with_capacity(name_part.len());
    let mut last_was_underscore = false;

    for c in name_part.chars() {
        if c.is_ascii_alphanumeric() {
            sanitized.push(c.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !last_was_underscore && !sanitized.is_empty() {
            sanitized.push('_');
            last_was_underscore = true;
        }
    }

    if sanitized.ends_with('_') {
        sanitized.pop();
    }

    if sanitized.is_empty() {
        sanitized = "table".to_string();
    } else if sanitized.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        sanitized = format!("t_{}", sanitized);
    }

    sanitized
}

/// Sanitizes a table name or filename so no SQL commands, shell injection,
/// or invalid identifier characters can be formed.
fn sanitize_table_name(raw: &str) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return "table".to_string();
    }

    // Preserve scratch. schema prefix if explicitly present; other dots (like in filenames .xlsx) belong to name_part
    let (schema_prefix, name_part) = if raw.to_ascii_lowercase().starts_with("scratch.") {
        (format!("{}.", SCRATCH_SCHEMA), &raw[8..])
    } else {
        ("".to_string(), raw)
    };

    let sanitized = sanitize_ident(name_part);
    format!("{}{}", schema_prefix, sanitized)
}

/// Resolves a raw table name into `(schema, sanitized_table_name, full_qualified_name)`.
/// If `raw` includes a schema qualifier (e.g. `analytics.sales`), checks if the schema is writable.
/// If `raw` is unqualified (e.g. `sales`), prepends `permissions.default_write_schema`.
fn resolve_table_name_and_schema(
    raw: &str,
    permissions: &SchemaPermissions,
) -> anyhow::Result<(String, String, String)> {
    let raw = raw.trim();
    if raw.is_empty() {
        anyhow::bail!("Table name cannot be empty");
    }

    if let Some(dot_idx) = raw.find('.') {
        let prefix = &raw[..dot_idx];
        let rest = &raw[dot_idx + 1..];
        // If prefix is in immutable system schemas or readonly_schemas, reject immediately
        if IMMUTABLE_SYSTEM_SCHEMAS
            .iter()
            .any(|sys| sys.eq_ignore_ascii_case(prefix))
            || permissions
                .readonly_schemas
                .iter()
                .any(|r| r.eq_ignore_ascii_case(prefix))
        {
            anyhow::bail!(
                "MCP write denied: schema '{}' is locked as read-only. Allowed writable schemas: {:?}, locked schemas: {:?}",
                prefix,
                permissions.readwrite_schemas,
                permissions.readonly_schemas
            );
        }
        if permissions.is_schema_writable(prefix) {
            let clean_table = sanitize_ident(rest);
            validate_sql_identifier(prefix)?;
            validate_sql_identifier(&clean_table)?;
            let schema_lower = prefix.to_ascii_lowercase();
            let full_name = format!("{}.{}", schema_lower, clean_table);
            return Ok((schema_lower, clean_table, full_name));
        } else {
            anyhow::bail!(
                "MCP write denied: schema '{}' is not in allowed writable schemas. Allowed writable schemas: {:?}, locked schemas: {:?}",
                prefix,
                permissions.readwrite_schemas,
                permissions.readonly_schemas
            );
        }
    }

    let clean_table = sanitize_ident(raw);
    let schema = permissions.default_write_schema.to_ascii_lowercase();
    validate_sql_identifier(&schema)?;
    validate_sql_identifier(&clean_table)?;
    if !permissions.is_schema_writable(&schema) {
        anyhow::bail!(
            "MCP write denied: default schema '{}' is not writable. Allowed writable schemas: {:?}",
            schema,
            permissions.readwrite_schemas
        );
    }
    let full_name = format!("{}.{}", schema, clean_table);
    Ok((schema, clean_table, full_name))
}

fn split_schema_and_table(full_name: &str) -> (String, String) {
    if let Some(dot_idx) = full_name.find('.') {
        (
            full_name[..dot_idx].to_string(),
            full_name[dot_idx + 1..].to_string(),
        )
    } else {
        (SCRATCH_SCHEMA.to_string(), full_name.to_string())
    }
}

/// Validates that a schema DDL string contains only valid column specifications
/// and does not attempt to smuggle chained statements or commands.
fn validate_schema_ddl(schema_ddl: &str) -> anyhow::Result<()> {
    use sqlparser::ast::Statement;
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    let dialect = GenericDialect {};
    let probe = format!("CREATE TABLE scratch._probe_val ({})", schema_ddl);
    let statements = Parser::parse_sql(&dialect, &probe)
        .map_err(|e| anyhow::anyhow!("Invalid schema DDL: {}", e))?;
    if statements.len() != 1 {
        anyhow::bail!(
            "Schema DDL must contain only column specifications without chained statements"
        );
    }
    match &statements[0] {
        Statement::CreateTable(_) => Ok(()),
        _ => anyhow::bail!("Schema DDL must be column specifications"),
    }
}

/// Function signature for domain-specific schema resolvers that return default DDL for a given table_type.
pub type CustomSchemaResolver = fn(table_type: &str) -> Option<&'static str>;

/// Optional extension configuration for domain-specific MCP servers
/// (Finance, Healthcare, Legal, etc.) built on the BenoStreamDB MCP engine.
#[derive(Clone, Default)]
pub struct McpExtension {
    pub server_name: Option<String>,
    pub server_version: Option<String>,
    pub schema_resolver: Option<CustomSchemaResolver>,
    pub file_extractor: Option<crate::ingest_formats::CustomFileExtractor>,
    pub additional_tools: Vec<Value>,
    pub permissions: Option<SchemaPermissions>,
    pub warehouse: Option<String>,
    pub config: Option<McpServerConfig>,
}

pub struct McpServer {
    session: RwLock<BenoStreamSession>,
    extension: McpExtension,
    permissions: RwLock<SchemaPermissions>,
    warehouse: RwLock<String>,
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServer {
    pub fn new() -> Self {
        Self::with_extension(McpExtension::default())
    }

    pub fn with_config(config: McpServerConfig) -> Self {
        let extension = McpExtension {
            config: Some(config),
            ..Default::default()
        };
        Self::with_extension(extension)
    }

    pub fn with_extension(extension: McpExtension) -> Self {
        let config = extension
            .config
            .clone()
            .unwrap_or_else(McpServerConfig::from_env);

        let warehouse = extension
            .warehouse
            .clone()
            .or_else(|| config.warehouse.clone())
            .unwrap_or_else(resolve_default_warehouse);

        let _ = std::fs::create_dir_all(&warehouse);

        let client_rw = extension
            .permissions
            .as_ref()
            .map(|p| p.readwrite_schemas.clone())
            .or_else(|| config.readwrite_schemas.clone());
        let client_ro = extension
            .permissions
            .as_ref()
            .map(|p| p.readonly_schemas.clone())
            .or_else(|| config.readonly_schemas.clone());
        let client_dw = extension
            .permissions
            .as_ref()
            .map(|p| p.default_write_schema.clone())
            .or_else(|| config.default_write_schema.clone());

        // Untrusted connector policy enforcement:
        // If an authoritative catalog policy exists in the warehouse, enforce it.
        // If NO policy exists, untrusted connectors are conservatively locked to read-only on all catalog schemas.
        let (final_rw, final_ro, default_w) =
            benostreamdb::core::catalog::resolve_permissions_for_untrusted_connector(
                &warehouse, client_rw, client_ro, client_dw,
            );

        let permissions = SchemaPermissions {
            readwrite_schemas: final_rw,
            readonly_schemas: final_ro,
            default_write_schema: default_w,
        };

        let mut session = BenoStreamSession::default();
        session.set_warehouse(Some(warehouse.clone()));
        let _ = benostreamdb::core::index::gpu::apply_gpu_context_from_env();
        Self {
            session: RwLock::new(session),
            extension,
            permissions: RwLock::new(permissions),
            warehouse: RwLock::new(warehouse),
        }
    }

    fn session(&self) -> BenoStreamSession {
        self.session.read().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn permissions(&self) -> SchemaPermissions {
        self.permissions
            .read()
            .map(|p| p.clone())
            .unwrap_or_default()
    }

    pub fn warehouse(&self) -> String {
        self.warehouse.read().map(|w| w.clone()).unwrap_or_default()
    }

    pub async fn apply_config(&self, config: &McpServerConfig) {
        if let Some(w) = &config.warehouse {
            if !w.trim().is_empty() {
                let _ = std::fs::create_dir_all(w);
                if let Ok(mut wh) = self.warehouse.write() {
                    *wh = w.clone();
                }
                if let Ok(mut sess) = self.session.write() {
                    sess.set_warehouse(Some(w.clone()));
                }
                self.load_warehouse_tables().await;
            }
        }
        let warehouse = self.warehouse();
        let (current_rw, current_ro, current_dw) = {
            let perms = self.permissions();
            (
                config
                    .readwrite_schemas
                    .clone()
                    .unwrap_or(perms.readwrite_schemas),
                config
                    .readonly_schemas
                    .clone()
                    .unwrap_or(perms.readonly_schemas),
                config
                    .default_write_schema
                    .clone()
                    .unwrap_or(perms.default_write_schema),
            )
        };
        let (final_rw, final_ro, default_w) =
            benostreamdb::core::catalog::resolve_permissions_for_untrusted_connector(
                &warehouse,
                Some(current_rw),
                Some(current_ro),
                Some(current_dw),
            );
        if let Ok(mut perms) = self.permissions.write() {
            perms.readwrite_schemas = final_rw;
            perms.readonly_schemas = final_ro;
            perms.default_write_schema = default_w;
        }
    }

    /// Discovers and registers existing tables in the warehouse into the DataFusion session.
    pub async fn load_warehouse_tables(&self) {
        let warehouse = self.warehouse();
        let p = std::path::Path::new(&warehouse);
        if !p.is_dir() {
            return;
        }
        let Ok(schema_entries) = std::fs::read_dir(p) else {
            return;
        };
        for schema_entry in schema_entries.flatten() {
            let schema_path = schema_entry.path();
            if !schema_path.is_dir() {
                continue;
            }
            let schema_name = schema_entry.file_name().to_string_lossy().to_string();
            if schema_name.starts_with('.') {
                continue;
            }

            let Ok(table_entries) = std::fs::read_dir(&schema_path) else {
                continue;
            };
            for table_entry in table_entries.flatten() {
                let table_path = table_entry.path();
                if !table_path.is_dir() {
                    continue;
                }
                let table_name = table_entry.file_name().to_string_lossy().to_string();
                if table_name.starts_with('.') {
                    continue;
                }

                if table_path.join("metadata").is_dir() {
                    let uri = table_path.to_string_lossy().to_string();
                    if let Ok(table) = benostreamdb::core::table::TableBuilder::new(uri)
                        .with_index_all(false)
                        .build_async()
                        .await
                    {
                        let _ = self
                            .session()
                            .sql(&format!("CREATE SCHEMA IF NOT EXISTS {}", schema_name))
                            .await;
                        let _ = self.session().register_table(
                            &format!("{}.{}", schema_name, table_name),
                            std::sync::Arc::new(table),
                        );
                        debug!(
                            "Loaded existing warehouse table into session: {}.{}",
                            schema_name, table_name
                        );
                    }
                }
            }
        }
    }

    pub async fn run(&self) -> Result<()> {
        self.load_warehouse_tables().await;
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin).lines();

        loop {
            match reader.next_line().await {
                Ok(Some(line)) => {
                    debug!("Received line: {}", line);
                    if let Ok(req) = serde_json::from_str::<JsonRpcRequest>(&line) {
                        if let Some(id) = req.id {
                            let response = self
                                .handle_request(&req.method, req.params.unwrap_or(json!({})))
                                .await;
                            let send_result = match response {
                                Ok(res) => {
                                    self.send_response(JsonRpcResponse::success(id.clone(), res))
                                }
                                Err(err) => self.send_response(JsonRpcResponse::error(
                                    id.clone(),
                                    -32603,
                                    err.to_string(),
                                )),
                            };
                            if let Err(e) = send_result {
                                error!("Stdout closed or write failed, exiting loop: {}", e);
                                break;
                            }
                        } else {
                            // It's a notification, handle without response
                            let _ = self
                                .handle_notification(&req.method, req.params.unwrap_or(json!({})))
                                .await;
                        }
                    } else {
                        error!("Failed to parse JSON-RPC request");
                        if let Err(e) = self.send_response(JsonRpcResponse::error(
                            Value::Null,
                            -32700,
                            "Parse error".to_string(),
                        )) {
                            error!("Stdout closed or write failed, exiting loop: {}", e);
                            break;
                        }
                    }
                }
                Ok(None) => {
                    info!("Stdin closed by client, exiting event loop.");
                    break;
                }
                Err(e) => {
                    error!("Error reading from stdin, exiting: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }

    async fn query_to_json(&self, query: &str) -> Result<Vec<Value>> {
        let (batches, _) = self.session().sql(query).await?;
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = Vec::new();
        {
            let mut writer = arrow_json::LineDelimitedWriter::new(&mut buf);
            for batch in &batches {
                writer.write(batch)?;
            }
            writer.finish()?;
        }
        let text = String::from_utf8(buf)?;
        let mut rows = Vec::new();
        for line in text.lines() {
            if !line.trim().is_empty() {
                if let Ok(val) = serde_json::from_str::<Value>(line) {
                    rows.push(val);
                }
            }
        }
        Ok(rows)
    }

    async fn fetch_embedding(&self, text: &str) -> Option<Vec<f32>> {
        let url = std::env::var("BSDB_EMBEDDING_URL")
            .ok()
            .filter(|s| !s.is_empty())?;
        let model =
            std::env::var("BSDB_EMBEDDING_MODEL").unwrap_or_else(|_| "all-minilm".to_string());
        let api_key = std::env::var("BSDB_EMBEDDING_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .ok()?;

        let mut req = client.post(&url);
        if let Some(key) = api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let is_ollama = url.contains("ollama") || url.ends_with("/api/embeddings");
        let body = if is_ollama {
            json!({
                "model": model,
                "prompt": text
            })
        } else {
            json!({
                "model": model,
                "input": text
            })
        };

        let resp = req.json(&body).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let json_val: Value = resp.json().await.ok()?;

        if let Some(arr) = json_val.get("embedding").and_then(|v| v.as_array()) {
            let vec: Vec<f32> = arr
                .iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect();
            if !vec.is_empty() {
                return Some(vec);
            }
        }
        if let Some(data) = json_val.get("data").and_then(|v| v.as_array()) {
            if let Some(first) = data.first() {
                if let Some(arr) = first.get("embedding").and_then(|v| v.as_array()) {
                    let vec: Vec<f32> = arr
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    if !vec.is_empty() {
                        return Some(vec);
                    }
                }
            }
        }

        None
    }

    async fn fetch_image_embedding(&self, image_base64: &str) -> Option<Vec<f32>> {
        let url = std::env::var("BSDB_EMBEDDING_URL")
            .ok()
            .filter(|s| !s.is_empty())?;
        let model =
            std::env::var("BSDB_EMBEDDING_MODEL").unwrap_or_else(|_| "all-minilm".to_string());
        let api_key = std::env::var("BSDB_EMBEDDING_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .ok()?;

        let mut req = client.post(&url);
        if let Some(key) = api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let is_ollama = url.contains("ollama") || url.ends_with("/api/embeddings");
        let body = if is_ollama {
            json!({
                "model": model,
                "prompt": "",
                "images": [image_base64]
            })
        } else {
            json!({
                "model": model,
                "input": image_base64
            })
        };

        let resp = req.json(&body).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let json_val: Value = resp.json().await.ok()?;

        if let Some(arr) = json_val.get("embedding").and_then(|v| v.as_array()) {
            let vec: Vec<f32> = arr
                .iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect();
            if !vec.is_empty() {
                return Some(vec);
            }
        }
        if let Some(data) = json_val.get("data").and_then(|v| v.as_array()) {
            if let Some(first) = data.first() {
                if let Some(arr) = first.get("embedding").and_then(|v| v.as_array()) {
                    let vec: Vec<f32> = arr
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    if !vec.is_empty() {
                        return Some(vec);
                    }
                }
            }
        }

        None
    }

    async fn insert_records_internal(
        &self,
        full_table_name: &str,
        records: &[Value],
    ) -> Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }
        let (schema, _) = split_schema_and_table(full_table_name);
        let _ = self
            .session()
            .sql(&format!("CREATE SCHEMA IF NOT EXISTS {}", schema))
            .await;

        use std::io::Write;
        let temp_file = tempfile::Builder::new().suffix(".json").tempfile()?;
        {
            let mut writer = std::io::BufWriter::new(temp_file.as_file());
            for rec in records {
                let _ = serde_json::to_writer(&mut writer, rec);
                let _ = writer.write_all(b"\n");
            }
            let _ = writer.flush();
        }

        let path = temp_file.path().to_str().unwrap_or("").replace("\\", "/");
        let temp_import_name = format!(
            "{}.temp_import_{}",
            schema,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        let query1 = format!(
            "CREATE EXTERNAL TABLE {} STORED AS JSON LOCATION '{}'",
            temp_import_name, path
        );
        self.session().sql(&query1).await?;

        let table_check = self
            .session()
            .sql(&format!("SELECT * FROM {} LIMIT 0", full_table_name))
            .await;
        let res = match table_check {
            Ok((_, schema)) => {
                let cols_str = schema
                    .fields()
                    .iter()
                    .map(|f| format!("\"{}\"", f.name()))
                    .collect::<Vec<_>>()
                    .join(", ");
                let query2 = if !cols_str.is_empty() {
                    format!(
                        "INSERT INTO {} ({}) SELECT {} FROM {}",
                        full_table_name, cols_str, cols_str, temp_import_name
                    )
                } else {
                    format!(
                        "INSERT INTO {} SELECT * FROM {}",
                        full_table_name, temp_import_name
                    )
                };
                self.session().sql(&query2).await
            }
            Err(_) => {
                let query_ctas = format!(
                    "CREATE TABLE {} AS SELECT * FROM {}",
                    full_table_name, temp_import_name
                );
                self.session().sql(&query_ctas).await
            }
        };

        let _ = self
            .session()
            .sql(&format!("DROP TABLE {}", temp_import_name))
            .await;
        res.map(|_| records.len())
    }

    pub async fn handle_request(&self, method: &str, params: Value) -> Result<Value> {
        info!("Handling request method: {}", method);
        match method {
            "initialize" => {
                if let Some(opts) = params.get("initializationOptions") {
                    if let Ok(cfg) = serde_json::from_value::<McpServerConfig>(opts.clone()) {
                        self.apply_config(&cfg).await;
                    }
                }
                self.load_warehouse_tables().await;
                let name = self
                    .extension
                    .server_name
                    .as_deref()
                    .unwrap_or("benostreamdb-mcp");
                let version = self.extension.server_version.as_deref().unwrap_or("0.12.0");
                Ok(json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": name,
                        "version": version
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    }
                }))
            }
            "prompts/list" => Ok(json!({
                "prompts": [
                    {
                        "name": "benostreamdb_sql_expert",
                        "description": "Guidance on writing performant BenoStreamDB SQL queries (Vector distance, HNSW indexes, Graph table functions, JSON extraction).",
                        "arguments": [
                            {
                                "name": "topic",
                                "description": "Specific SQL capability (e.g. 'vector', 'graph', 'json', 'hybrid')",
                                "required": false
                            }
                        ]
                    },
                    {
                        "name": "agent_memory_guide",
                        "description": "Best practices for agent long-term memory management using store_memory and recall_memory.",
                        "arguments": []
                    },
                    {
                        "name": "code_rag_strategy",
                        "description": "Hierarchical Code RAG and repository exploration strategy combining hybrid_search and graph_rag_search (DRIFT RAG).",
                        "arguments": [
                            {
                                "name": "task",
                                "description": "The coding or exploration task being performed",
                                "required": false
                            }
                        ]
                    }
                ]
            })),
            "prompts/get" => {
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                match name {
                    "benostreamdb_sql_expert" => {
                        let topic = params
                            .get("arguments")
                            .and_then(|a| a.get("topic"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("general");
                        let prompt_text = format!(
                            "You are an expert AI assistant operating with BenoStreamDB.\n\
                            Key Capabilities:\n\
                            - Vector Search: Use `l2_distance(col, '[0.1, ...]')` or `cosine_distance(...)` in ORDER BY ... LIMIT k.\n\
                            - Vector Indexing: Accelerate scans using `CREATE INDEX idx ON scratch.table (col) USING hnsw`.\n\
                            - Graph Traversal: Use `SELECT * FROM graph_neighbors('edges', '101', 2, 'auto')` and `graph_shortest_path`.\n\
                            - Ephemeral Writes: All writes, temporary tables, and index creations MUST target the `scratch` schema (e.g. `scratch.my_table`).\n\
                            - JSON Path: Use `json_extract_path(metadata, '$.key')`.\n\
                            Focus topic: {}",
                            topic
                        );
                        Ok(json!({
                            "description": "BenoStreamDB SQL Expert Guide",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": {
                                        "type": "text",
                                        "text": prompt_text
                                    }
                                }
                            ]
                        }))
                    }
                    "agent_memory_guide" => {
                        let prompt_text = "BenoStreamDB Agent Memory Workflow:\n\
                        1. Whenever you discover key facts, architectural decisions, or user preferences, call `store_memory` with a descriptive `key`, `content`, and relevant `tags`.\n\
                        2. At the start of complex tasks or when unsure about context, call `recall_memory` with your current task or query.\n\
                        3. Memories are stored in `scratch.agent_memories` and are indexed with dense vectors (if BSDB_EMBEDDING_URL is active) and BM25 text search.";
                        Ok(json!({
                            "description": "Agent Memory Guide",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": {
                                        "type": "text",
                                        "text": prompt_text
                                    }
                                }
                            ]
                        }))
                    }
                    "code_rag_strategy" => {
                        let task = params
                            .get("arguments")
                            .and_then(|a| a.get("task"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("repository analysis");
                        let prompt_text = format!(
                            "BenoStreamDB Code RAG Strategy for: {}\n\
                            1. Dual-Path Retrieval:\n\
                               - Use `hybrid_search` on code chunks combining exact symbol matches (BM25) and conceptual similarity (Vector).\n\
                            2. Graph DRIFT Traversal:\n\
                               - Once key files or function symbols are found, use `graph_rag_search` with mode 'neighbors' or 'subgraph' on call/import graphs to find callers, callees, and dependencies.\n\
                            3. Synthesis & Ephemeral Prototyping:\n\
                               - Store intermediate candidate findings into `scratch.findings` using `ingest_records` and join them with repository tables.",
                            task
                        );
                        Ok(json!({
                            "description": "Code RAG Strategy",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": {
                                        "type": "text",
                                        "text": prompt_text
                                    }
                                }
                            ]
                        }))
                    }
                    _ => Err(anyhow::anyhow!("Unknown prompt: {}", name)),
                }
            }
            "resources/list" => {
                let mut resources = Vec::new();
                if let Ok(rows) = self.query_to_json("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema != 'information_schema'").await {
                    for row in rows {
                        let schema = row.get("table_schema").and_then(|v| v.as_str()).unwrap_or("datafusion");
                        let name = row.get("table_name").and_then(|v| v.as_str()).unwrap_or("");
                        if !name.is_empty() {
                            resources.push(json!({
                                "uri": format!("benostreamdb://tables/{}/{}", schema, name),
                                "name": format!("{}.{}", schema, name),
                                "description": format!("Table {}.{}", schema, name)
                            }));
                        }
                    }
                }
                resources.push(json!({
                    "uri": "benostreamdb://docs/sql_manual.md",
                    "name": "BenoStreamDB SQL Manual",
                    "description": "Comprehensive reference manual for BenoStreamDB's SQL extensions, including Vector Search, Graph Traversal, and JSON processing."
                }));
                Ok(json!({
                    "resources": resources
                }))
            }
            "resources/read" => {
                let uri = params.get("uri").and_then(|u| u.as_str()).unwrap_or("");

                if uri == "benostreamdb://docs/sql_manual.md" {
                    let md_content = "# BenoStreamDB SQL Reference Manual\n\n\
                    BenoStreamDB extends standard Apache DataFusion SQL with advanced analytical capabilities designed for AI, graph, and document workloads.\n\n\
                    ## 1. Vector Search & Embeddings\n\
                    BenoStreamDB provides native first-class support for high-dimensional vectors and similarity search.\n\
                    - **Usage Example**: \n\
                      ```sql\n\
                      SELECT id, text_chunk \n\
                      FROM embeddings_table \n\
                      ORDER BY vector_distance(embedding, [0.1, 0.2, 0.3]) ASC \n\
                      LIMIT 10;\n\
                      ```\n\n\
                    ## 2. Advanced Indexing\n\
                    Indexes in BenoStreamDB act as overlay accelerators. If an index exists, the engine automatically uses it for point lookups and nearest-neighbor scans.\n\
                    - **Create an index**: `CREATE INDEX idx_name ON scratch.my_table (column) USING algorithm;`\n\
                      or `ALTER TABLE scratch.my_table ADD INDEX (column);`\n\
                    - **Supported algorithms**: `hnsw`, `hnsw_pq`, `hnsw_tq4`, `hnsw_tq8` (vector), `bm25` (lexical), `bloom`, `bitmap`, `composite_bitmap` (scalar), `csr_graph` (graph), `json_path` (JSON).\n\
                    - **Primary key** (enables merge-on-read upserts): `ALTER TABLE scratch.my_table ADD PRIMARY KEY (id);`\n\
                    - **Note**: indexes and primary keys may only be added to `scratch` tables; the main catalog is read-only.\n\n\
                    ## 3. Ephemeral (Scratch) Tables\n\
                    When prototyping or executing multi-step LLM operations, you should use ephemeral tables.\n\
                    - Scratch tables are dynamically placed in the isolated `scratch` schema.\n\
                    - Example: `SELECT * FROM scratch.my_scratch_table`\n\
                    - You can seamlessly `JOIN` an ephemeral table in the `scratch` schema with persistent tables in the `default` schema.\n\n\
                    ## 4. Graph Traversals & Topology\n\
                    BenoStreamDB supports hybrid property graph queries using standard SQL.\n\
                    - **Schema Definition**: Declare a table's role and endpoints with table metadata:\n\
                      ```sql\n\
                      ALTER TABLE edges SET TBLPROPERTIES ('table_type' = 'edge', 'src_col' = 'src', 'dst_col' = 'dst');\n\
                      ALTER TABLE nodes SET TBLPROPERTIES ('table_type' = 'node', 'id_col' = 'id', 'label_col' = 'name');\n\
                      ```\n\
                      The graph functions then resolve the endpoints from the metadata, so you never name the columns.\n\
                      Declaring an edge table also auto-configures the forward (source) and reverse (target) CSR graph indexes.\n\
                    - **Traversal**: Use the graph table functions directly in `FROM`:\n\
                      ```sql\n\
                      SELECT * FROM graph_neighbors('edges', '101', 2, 'auto');\n\
                      SELECT * FROM graph_shortest_path('edges', 101, 205, 'auto');\n\
                      ```\n\
                      or the graph UDAFs (`graph_pagerank(src, dst)`, `graph_louvain_communities(src, dst)`, …) over a `JOIN`.\n\
                    - **Discovery**: Call the `list_graph_tables` tool to enumerate node/edge tables and their endpoints.\n\n\
                    ## 5. JSON & Document Processing\n\
                    For schemaless data, JSON paths can be traversed natively.\n\
                    - **Function**: `json_extract_path(json_column, '$.path.to.field')`\n\
                    - **Example**: `SELECT json_extract_path(metadata, '$.author.name') FROM documents;`\n\n\
                    ## 6. Complete Custom Function Signatures\n\
                    \n\
                    ### Vector Distance Functions (Dense & Sparse)\n\
                    - `vector_distance(vec1, vec2)` / `sparse_l2_distance`: Computes L2 (Euclidean) distance.\n\
                    - `vector_cosine_distance(vec1, vec2)` / `sparse_cosine_distance`: Computes Cosine distance.\n\
                    - `vector_inner_product(vec1, vec2)` / `sparse_inner_product_distance`: Computes Inner Product.\n\
                    - `vector_l1_distance(vec1, vec2)`: Computes L1 (Manhattan) distance.\n\
                    - `vector_hamming_distance(vec1, vec2)`: Computes Hamming distance.\n\
                    - `vector_jaccard_distance(vec1, vec2)`: Computes Jaccard distance.\n\
                    \n\
                    ### Vector Math & Quantization\n\
                    - `vector_add`, `vector_sub`, `vector_mul`: Element-wise arithmetic.\n\
                    - `vector_norm(vec)`, `vector_normalize(vec)`, `vector_dims(vec)`: Vector topology.\n\
                    - `binary_quantize(vec)`, `vector_to_binary(vec)`: Quantization to bit-vectors.\n\
                    - `subvector(vec, offset, len)`: Slice a vector.\n\
                    - `dense_to_sparse(vec)`, `sparse_to_dense(vec)`, `sparsevec_nnz(vec)`: Sparse operations.\n\
                    \n\
                    ### Vector Aggregations & Stats (UDAFs)\n\
                    - `vector_avg(vec)`, `vector_sum(vec)`: Aggregate math.\n\
                    - `centroid(vec)`: Computes the geometric center of vectors.\n\
                    - `vector_max(vec)`, `vector_min(vec)`, `vector_median(vec)`, `vector_stddev(vec)`: Aggregate stats.\n\
                    \n\
                    ### JSON Functions\n\
                    - `json_contains(json, target)`: Checks if the JSON document contains the target.\n\
                    - `json_exists(json, path)`: Checks if a top-level key/element exists.\n\
                    - `json_extract_path(json, path)`: Extracts JSON sub-element at the specified path.\n\
                    - `json_path_exists(json, jsonpath)`: Evaluates if a subset matches the jsonpath.\n\
                    - `json_path_query(json, jsonpath)`: Extracts all elements matching the jsonpath.\n\
                    - `json_typeof(json)`: Returns the type of the JSON value as a string.\n\
                    \n\
                    ### Text & Lexical Functions\n\
                    - `bm25_score(text, query)`: Computes BM25 relevance score.\n\
                    - `tf_idf(text, query)`: Computes TF-IDF relevance score.\n\
                    \n\
                    ### Graph Analysis & Traversals (UDAFs)\n\
                    - **Centrality**:\n\
                      - `graph_pagerank(src, dst)`: Computes PageRank.\n\
                      - `graph_personalized_pagerank(src, dst, node)`: Computes Personalized PageRank.\n\
                      - `graph_degree_centrality(src, dst)`: Computes degree centrality.\n\
                      - `graph_closeness_centrality(src, dst)`: Computes closeness centrality.\n\
                      - `graph_betweenness_centrality(src, dst)`: Computes betweenness centrality.\n\
                    - **Communities**:\n\
                      - `graph_connected_components(src, dst)`: Weakly connected components.\n\
                      - `graph_strongly_connected_components(src, dst)`: Strongly connected components.\n\
                      - `graph_louvain_communities(src, dst)`: Louvain community detection.\n\
                      - `graph_leiden_communities(src, dst)`: Leiden community detection.\n\
                      - `graph_label_propagation(src, dst)`: Label propagation.\n\
                      - `graph_modularity(src, dst, community_id)`: Graph modularity.\n\
                    - **Paths & Topology**:\n\
                      - `graph_shortest_path(src, dst, start, end)`: Shortest path between two nodes.\n\
                      - `graph_all_shortest_paths(src, dst, start, end)`: All shortest paths.\n\
                      - `graph_connecting_paths(src, dst, start, end)`: Connecting paths.\n\
                      - `graph_triangle_count(src, dst)`: Triangle counting.\n\
                      - `graph_preferential_attachment(src, dst)`: Preferential attachment.\n\
                      - `graph_jaccard_coefficient(src, dst)`: Jaccard coefficient.\n\
                    - **Extraction**:\n\
                      - `graph_neighbors(src, dst, node)`: Find all neighbors of a node.\n\
                      - `graph_subgraph(src, dst, node)`: Extract subgraph around a node.\n\
                    \n\
                    ## 7. Live Subscriptions (Change Feed)\n\
                    Subscribe to a table's committed changes (in-process).\n\
                    - **SQL**: `SELECT * FROM subscribe_events('scratch.my_table', 'id > 10', 100, 1000);` returns one row per event (`event_type` = 'batch' | 'commit', `rows`).\n\
                    - **Tool**: call the `subscribe_events` tool with `{ table, filter?, max_events?, timeout_ms? }`.\n\
                    - **Note**: the change feed is in-process; it only observes commits made by writers in the same process.";

                    return Ok(json!({
                        "contents": [{
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": md_content
                        }]
                    }));
                }

                if uri.starts_with("benostreamdb://tables/") {
                    let parts: Vec<&str> = uri
                        .trim_start_matches("benostreamdb://tables/")
                        .split('/')
                        .collect();
                    if parts.len() == 2 {
                        let schema = parts[0];
                        let table = parts[1];
                        let query = format!("DESCRIBE {}.{}", schema, table);
                        match self.session().sql(&query).await {
                            Ok((batches, _)) => {
                                let mut buf = Vec::new();
                                {
                                    let mut writer = arrow_json::LineDelimitedWriter::new(&mut buf);
                                    for batch in &batches {
                                        let _ = writer.write(batch);
                                    }
                                    let _ = writer.finish();
                                }
                                let text_output = String::from_utf8_lossy(&buf).to_string();
                                return Ok(json!({
                                    "contents": [
                                        {
                                            "uri": uri,
                                            "mimeType": "application/json",
                                            "text": text_output
                                        }
                                    ]
                                }));
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!("Failed to read resource: {}", e));
                            }
                        }
                    }
                }
                Err(anyhow::anyhow!("Resource not found: {}", uri))
            }
            "tools/list" => {
                let execute_sql_desc = "Execute a SQL query against the BenoStreamDB engine. Uses Apache DataFusion SQL dialect.\n\n\
                Available custom functions and syntax:\n\
                - Vector Search: Use `l2_distance(embedding, [1.0, 2.0, 3.0])` / `cosine_distance(...)` in SELECT or ORDER BY clauses.\n\
                - Indexes: Create indexes using `CREATE INDEX idx_name ON table_name (col) USING hnsw` (Supports: hnsw, hnsw_tq8, hnsw_tq4, bm25, bitmap).\n\
                - Table Functions: Query graph walks with `SELECT * FROM graph_neighbors('edges', '101', 2, 'auto')`.\n\
                - JSON Path: Extract JSON fields natively using `json_extract_path(metadata, '$.path')`.\n\
                - Graph Queries: Declare node/edge tables with `ALTER TABLE t SET TBLPROPERTIES ('table_type'='edge', 'src_col'='src', 'dst_col'='dst')`, then use the graph functions.\n\
                - Ephemeral Tables: Scratch tables are placed in the `scratch` schema (e.g. `scratch.my_table`).\n\
                - Discovery: Query `information_schema.tables` and `information_schema.columns`, or call the `list_graph_tables` tool.\n\n\
                NOTE: For full syntax examples, read the `benostreamdb://docs/sql_manual.md` resource!";

                let mut tools = json!([
                    {
                        "name": "execute_sql",
                        "description": execute_sql_desc,
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string" }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "create_scratch_table",
                        "description": "Creates a temporary, ephemeral table in the local scratch catalog. table_type can specify standard formats (e.g. 'node', 'edge'). If a recognized table_type is specified, schema_ddl can be omitted or set to 'default'.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "table_name": { "type": "string" },
                                "schema_ddl": { "type": "string", "description": "The column definitions, e.g., 'id INT, name VARCHAR', or 'default' if using a recognized table_type" },
                                "table_type": { "type": "string", "description": "Optional type, e.g. 'node' or 'edge'" }
                            },
                            "required": ["table_name"]
                        }
                    },
                    {
                        "name": "export_table",
                        "description": "Exports any table (scratch or main catalog) to local disk as Parquet.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "table_name": { "type": "string" },
                                "export_path": { "type": "string" }
                            },
                            "required": ["table_name", "export_path"]
                        }
                    },
                    {
                        "name": "list_graph_tables",
                        "description": "List the graph tables (node/edge) and their endpoint columns. Use this to discover the graph before traversing it.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "subscribe_events",
                        "description": "Drain the next committed-change events for a table (a bounded live tail). Returns one row per event: event_type ('batch'|'commit') and rows. Only observes commits made in this process.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "table": { "type": "string" },
                                "filter": { "type": "string", "description": "Optional SQL predicate; only matching rows are counted." },
                                "max_events": { "type": "integer", "description": "Max events to drain (default 100)." },
                                "timeout_ms": { "type": "integer", "description": "Wait budget in ms (default 1000)." }
                            },
                            "required": ["table"]
                        }
                    },
                    {
                        "name": "hybrid_search",
                        "description": "Performs hybrid search combining BM25 lexical search and dense vector similarity using Reciprocal Rank Fusion (RRF). If query_vector is omitted and BSDB_EMBEDDING_URL is configured, dense embeddings are auto-generated from query_text.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "table": { "type": "string", "description": "Target table name (e.g. 'code_chunks' or 'scratch.agent_memories')" },
                                "query_text": { "type": "string", "description": "Search query text for BM25 and vector embedding" },
                                "query_vector": { "type": "array", "items": { "type": "number" }, "description": "Optional precomputed embedding vector" },
                                "vector_col": { "type": "string", "description": "Vector column name (default 'embedding')" },
                                "text_col": { "type": "string", "description": "Text column name (default 'content')" },
                                "top_k": { "type": "integer", "description": "Number of top results to return (default 10)" },
                                "rrf_k": { "type": "integer", "description": "RRF damping constant (default 60)" }
                            },
                            "required": ["table", "query_text"]
                        }
                    },
                    {
                        "name": "graph_rag_search",
                        "description": "Performs regional graph retrieval (Graph RAG / DRIFT) around seed nodes using graph table functions (graph_neighbors, graph_subgraph, or graph_shortest_path).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "table": { "type": "string", "description": "Edge table name" },
                                "seeds": { "type": "array", "items": { "type": "string" }, "description": "Seed node identifiers" },
                                "hops": { "type": "integer", "description": "Traversal depth (default 2)" },
                                "mode": { "type": "string", "enum": ["neighbors", "shortest_path", "subgraph"], "description": "Graph retrieval mode (default 'neighbors')" },
                                "target": { "type": "string", "description": "Target node for shortest_path mode" },
                                "top_k": { "type": "integer", "description": "Max rows to return (default 50)" }
                            },
                            "required": ["table", "seeds"]
                        }
                    },
                    {
                        "name": "store_memory",
                        "description": "Stores an agent memory, context note, or code insight into the persistent scratch table 'scratch.agent_memories'. Embeddings are auto-generated if BSDB_EMBEDDING_URL is set.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "key": { "type": "string", "description": "Unique key or topic label" },
                                "content": { "type": "string", "description": "Memory content or insight" },
                                "tags": { "type": "array", "items": { "type": "string" }, "description": "Optional list of tags or categories" },
                                "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional precomputed embedding vector" }
                            },
                            "required": ["key", "content"]
                        }
                    },
                    {
                        "name": "recall_memory",
                        "description": "Recalls agent memories from 'scratch.agent_memories' using vector similarity or lexical search.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string", "description": "Text query to recall memories for" },
                                "tag": { "type": "string", "description": "Optional tag filter" },
                                "top_k": { "type": "integer", "description": "Number of memories to recall (default 5)" },
                                "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional precomputed query vector" }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "ingest_records",
                        "description": "Ingests structured JSON records into a scratch table. Auto-creates the table if it does not exist, and optionally generates embeddings for a specified text column.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "table_name": { "type": "string", "description": "Destination table name (will be placed in scratch schema)" },
                                "records": { "type": "array", "items": { "type": "object" }, "description": "Array of JSON objects" },
                                "embed_col": { "type": "string", "description": "Optional text column to auto-embed into 'embedding' column" }
                            },
                            "required": ["table_name", "records"]
                        }
                    },
                    {
                        "name": "ingest_file",
                        "description": "Ingests a local file (Word .docx, PowerPoint .pptx, Excel .xlsx/.xls/.ods, XML, Zstandard .zst/.zstd, Zip archives .zip, PDF documents, Images .png/.jpg/.webp, code, markdown, text, or JSON) into a scratch table. Text/PDF/Office are chunked with line/slide/paragraph metadata; Excel/XML are converted to structured tabular rows; Images are ingested as base64 Data URIs with optional multimodal embeddings.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string", "description": "Absolute path to local file" },
                                "table_name": { "type": "string", "description": "Target scratch table name (optional, defaults to sanitized file stem)" },
                                "chunk_size": { "type": "integer", "description": "Character chunk size for text/code/PDF (default 500)" },
                                "embed_chunks": { "type": "boolean", "description": "Whether to auto-generate embeddings for chunks or images (default false)" }
                            },
                            "required": ["file_path"]
                        }
                    }
                ]);
                if let Some(arr) = tools.as_array_mut() {
                    arr.extend(self.extension.additional_tools.clone());
                }
                Ok(json!({ "tools": tools }))
            }
            "tools/call" => {
                let permissions = self.permissions();
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");

                match name {
                    "execute_sql" => {
                        let query = params
                            .get("arguments")
                            .and_then(|a| a.get("query"))
                            .and_then(|q| q.as_str())
                            .unwrap_or("");
                        if let Err(e) = guard_schema_writes(query, &permissions) {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": e.to_string() } ],
                                "isError": true
                            }));
                        }
                        match self.session().sql(query).await {
                            Ok((batches, _schema)) => {
                                let mut text_output = String::new();
                                if !batches.is_empty() {
                                    let mut buf = Vec::new();
                                    {
                                        let mut writer =
                                            arrow_json::LineDelimitedWriter::new(&mut buf);
                                        for batch in &batches {
                                            let _ = writer.write(batch);
                                        }
                                        let _ = writer.finish();
                                    }
                                    text_output = String::from_utf8(buf).unwrap_or_default();
                                }
                                if text_output.is_empty() {
                                    text_output = "Success (no rows returned).".to_string();
                                }
                                Ok(json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": text_output
                                        }
                                    ]
                                }))
                            }
                            Err(e) => Ok(json!({
                                "content": [
                                    {
                                        "type": "text",
                                        "text": format!("Error executing SQL: {}", e)
                                    }
                                ],
                                "isError": true
                            })),
                        }
                    }
                    "create_scratch_table" => {
                        let table_name = params
                            .get("arguments")
                            .and_then(|a| a.get("table_name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let schema_ddl = params
                            .get("arguments")
                            .and_then(|a| a.get("schema_ddl"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let table_type = params
                            .get("arguments")
                            .and_then(|a| a.get("table_type"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        let clean_table_type = table_type.trim().to_lowercase();
                        let ddl_to_use = if schema_ddl.trim().is_empty()
                            || schema_ddl.trim().eq_ignore_ascii_case("default")
                        {
                            if let Some(resolver) = self.extension.schema_resolver {
                                if let Some(resolved) = resolver(&clean_table_type) {
                                    resolved.to_string()
                                } else {
                                    schema_ddl.to_string()
                                }
                            } else {
                                schema_ddl.to_string()
                            }
                        } else {
                            schema_ddl.to_string()
                        };

                        if ddl_to_use.trim().is_empty() {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": "create_scratch_table requires 'schema_ddl' unless a supported 'table_type' with a default schema is provided." } ],
                                "isError": true
                            }));
                        }

                        if table_name.trim().is_empty() {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": "create_scratch_table requires 'table_name'." } ],
                                "isError": true
                            }));
                        }

                        let (schema, _clean_table, full_table_name) =
                            match resolve_table_name_and_schema(table_name, &permissions) {
                                Ok(res) => res,
                                Err(e) => {
                                    return Ok(json!({
                                        "content": [ { "type": "text", "text": e.to_string() } ],
                                        "isError": true
                                    }));
                                }
                            };

                        if let Err(e) = validate_schema_ddl(&ddl_to_use) {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": format!("Invalid schema DDL: {}", e) } ],
                                "isError": true
                            }));
                        }

                        let _ = self
                            .session()
                            .sql(&format!("CREATE SCHEMA IF NOT EXISTS {}", schema))
                            .await;

                        let query = format!("CREATE TABLE {} ({})", full_table_name, ddl_to_use);
                        match self.session().sql(&query).await {
                            Ok(_) => {
                                // Declare the graph role via table metadata (the
                                // engine ignores `WITH (type = ...)`).
                                if !table_type.is_empty() {
                                    let alter = format!(
                                        "ALTER TABLE {} SET TBLPROPERTIES ('table_type' = '{}')",
                                        full_table_name, table_type
                                    );
                                    if let Err(e) = self.session().sql(&alter).await {
                                        error!(
                                            "Failed to set table_type on {}: {}",
                                            full_table_name, e
                                        );
                                    }
                                }
                                Ok(json!({
                                    "content": [
                                        { "type": "text", "text": format!("Table {} created successfully.", full_table_name) }
                                    ]
                                }))
                            }
                            Err(e) => Ok(json!({
                                "content": [
                                    { "type": "text", "text": format!("Error creating table: {}", e) }
                                ],
                                "isError": true
                            })),
                        }
                    }
                    "export_table" => {
                        let table_name = params
                            .get("arguments")
                            .and_then(|a| a.get("table_name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let export_path = params
                            .get("arguments")
                            .and_then(|a| a.get("export_path"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        if let Err(e) = validate_local_path(export_path, "export_path") {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": e.to_string() } ],
                                "isError": true
                            }));
                        }

                        let clean_table = sanitize_table_name(table_name);
                        let full_table_name = if clean_table.contains('.') {
                            clean_table
                        } else {
                            format!("{}.{}", permissions.default_write_schema, clean_table)
                        };

                        let esc_path = escape_sql_literal(&export_path.replace("\\", "/"));
                        let query = format!(
                            "COPY {} TO '{}' STORED AS PARQUET",
                            full_table_name, esc_path
                        );
                        match self.session().sql(&query).await {
                            Ok(_) => Ok(json!({
                                "content": [
                                    { "type": "text", "text": format!("Successfully exported {} to {}", table_name, export_path) }
                                ]
                            })),
                            Err(e) => Ok(json!({
                                "content": [
                                    { "type": "text", "text": format!("Error exporting table: {}", e) }
                                ],
                                "isError": true
                            })),
                        }
                    }
                    "list_graph_tables" => match self.session().list_graph_tables().await {
                        Ok(infos) => {
                            let text = if infos.is_empty() {
                                "No graph tables found. Declare one with \
                                     ALTER TABLE t SET TBLPROPERTIES ('table_type'='edge', 'src_col'='src', 'dst_col'='dst')."
                                        .to_string()
                            } else {
                                serde_json::to_string_pretty(&infos).unwrap_or_default()
                            };
                            Ok(json!({
                                "content": [ { "type": "text", "text": text } ]
                            }))
                        }
                        Err(e) => Ok(json!({
                            "content": [ { "type": "text", "text": format!("Error listing graph tables: {}", e) } ],
                            "isError": true
                        })),
                    },
                    "subscribe_events" => {
                        let table = params
                            .get("arguments")
                            .and_then(|a| a.get("table"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let filter = params
                            .get("arguments")
                            .and_then(|a| a.get("filter"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let max_events = params
                            .get("arguments")
                            .and_then(|a| a.get("max_events"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(100);
                        let timeout_ms = params
                            .get("arguments")
                            .and_then(|a| a.get("timeout_ms"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1000);
                        if table.is_empty() {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": "subscribe_events requires a 'table' argument." } ],
                                "isError": true
                            }));
                        }
                        let sanitized_table = sanitize_table_name(table);
                        let query = format!(
                            "SELECT * FROM subscribe_events('{}', '{}', {}, {})",
                            sanitized_table,
                            escape_sql_literal(filter),
                            max_events,
                            timeout_ms
                        );
                        match self.session().sql(&query).await {
                            Ok((batches, _schema)) => {
                                let mut text_output = String::new();
                                if !batches.is_empty() {
                                    let mut buf = Vec::new();
                                    {
                                        let mut writer =
                                            arrow_json::LineDelimitedWriter::new(&mut buf);
                                        for batch in &batches {
                                            let _ = writer.write(batch);
                                        }
                                        let _ = writer.finish();
                                    }
                                    text_output = String::from_utf8(buf).unwrap_or_default();
                                }
                                if text_output.is_empty() {
                                    text_output = "No events (nothing committed in this process during the wait window).".to_string();
                                }
                                Ok(json!({
                                    "content": [ { "type": "text", "text": text_output } ]
                                }))
                            }
                            Err(e) => Ok(json!({
                                "content": [ { "type": "text", "text": format!("Error draining events: {}", e) } ],
                                "isError": true
                            })),
                        }
                    }
                    "hybrid_search" => {
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let table = args.get("table").and_then(|v| v.as_str()).unwrap_or("");
                        let query_text = args
                            .get("query_text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let vector_col = args
                            .get("vector_col")
                            .and_then(|v| v.as_str())
                            .unwrap_or("embedding");
                        let text_col = args
                            .get("text_col")
                            .and_then(|v| v.as_str())
                            .unwrap_or("content");
                        let top_k =
                            args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                        let rrf_k = args.get("rrf_k").and_then(|v| v.as_u64()).unwrap_or(60);

                        if table.is_empty() || query_text.is_empty() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "hybrid_search requires 'table' and 'query_text'." }],
                                "isError": true
                            }));
                        }

                        let sanitized_table = sanitize_table_name(table);
                        if let Err(e) = validate_sql_identifier(text_col) {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": format!("Invalid text column: {}", e) }],
                                "isError": true
                            }));
                        }
                        if let Err(e) = validate_sql_identifier(vector_col) {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": format!("Invalid vector column: {}", e) }],
                                "isError": true
                            }));
                        }

                        let query_vector: Option<Vec<f32>> = if let Some(arr) =
                            args.get("query_vector").and_then(|v| v.as_array())
                        {
                            Some(
                                arr.iter()
                                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                                    .collect(),
                            )
                        } else {
                            self.fetch_embedding(query_text).await
                        };

                        let esc_text = escape_sql_literal(query_text);
                        let candidate_k = (top_k * 3).max(20);

                        // 1. BM25 search
                        let bm25_sql = format!(
                            "SELECT *, bm25_score({}, '{}') AS _bm25_score FROM {} ORDER BY _bm25_score DESC LIMIT {}",
                            text_col, esc_text, sanitized_table, candidate_k
                        );
                        let bm25_rows = self.query_to_json(&bm25_sql).await.unwrap_or_default();

                        // 2. Vector search (if vector available)
                        let vec_rows = if let Some(ref v) = query_vector {
                            let vec_str = format!(
                                "[{}]",
                                v.iter()
                                    .map(|f| f.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            );
                            let vec_sql = format!(
                                "SELECT *, l2_distance({}, '{}') AS _vec_dist FROM {} ORDER BY _vec_dist ASC LIMIT {}",
                                vector_col, vec_str, sanitized_table, candidate_k
                            );
                            self.query_to_json(&vec_sql).await.unwrap_or_default()
                        } else {
                            Vec::new()
                        };

                        // If no vector search occurred, return BM25 results up to top_k
                        if vec_rows.is_empty() {
                            let results: Vec<Value> = bm25_rows.into_iter().take(top_k).collect();
                            return Ok(json!({
                                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&results).unwrap_or_default() }]
                            }));
                        }

                        // 3. Merge with RRF
                        let doc_key = |val: &Value| -> String {
                            if let Some(id) = val.get("id").and_then(|v| {
                                v.as_str()
                                    .map(String::from)
                                    .or_else(|| v.as_i64().map(|n| n.to_string()))
                            }) {
                                return id;
                            }
                            if let Some(key) = val.get("key").and_then(|v| v.as_str()) {
                                return key.to_string();
                            }
                            if let (Some(path), Some(chunk)) = (
                                val.get("path").and_then(|v| v.as_str()),
                                val.get("chunk_index"),
                            ) {
                                return format!("{}:{}", path, chunk);
                            }
                            serde_json::to_string(val).unwrap_or_default()
                        };

                        let mut doc_map: std::collections::HashMap<
                            String,
                            (Value, Option<usize>, Option<usize>),
                        > = std::collections::HashMap::new();

                        for (i, row) in bm25_rows.into_iter().enumerate() {
                            let k = doc_key(&row);
                            doc_map.entry(k).or_insert((row, None, None)).1 = Some(i + 1);
                        }
                        for (i, row) in vec_rows.into_iter().enumerate() {
                            let k = doc_key(&row);
                            let entry = doc_map.entry(k).or_insert((row, None, None));
                            entry.2 = Some(i + 1);
                        }

                        let rrf_k_f64 = rrf_k as f64;
                        let mut fused: Vec<(Value, f64)> = doc_map
                            .into_values()
                            .map(|(mut row, bm_rank, v_rank)| {
                                let mut score = 0.0f64;
                                if let Some(r) = bm_rank {
                                    score += 1.0 / (rrf_k_f64 + r as f64);
                                }
                                if let Some(r) = v_rank {
                                    score += 1.0 / (rrf_k_f64 + r as f64);
                                }
                                if let Some(obj) = row.as_object_mut() {
                                    obj.insert("_rrf_score".to_string(), json!(score));
                                    if let Some(r) = bm_rank {
                                        obj.insert("_bm25_rank".to_string(), json!(r));
                                    }
                                    if let Some(r) = v_rank {
                                        obj.insert("_vec_rank".to_string(), json!(r));
                                    }
                                }
                                (row, score)
                            })
                            .collect();

                        fused.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let results: Vec<Value> =
                            fused.into_iter().take(top_k).map(|(r, _)| r).collect();

                        Ok(json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&results).unwrap_or_default() }]
                        }))
                    }
                    "graph_rag_search" => {
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let table = args.get("table").and_then(|v| v.as_str()).unwrap_or("");
                        let seeds = args.get("seeds").and_then(|v| v.as_array());
                        let hops = args.get("hops").and_then(|v| v.as_u64()).unwrap_or(2);
                        let mode = args
                            .get("mode")
                            .and_then(|v| v.as_str())
                            .unwrap_or("neighbors");
                        let target = args.get("target").and_then(|v| {
                            v.as_str()
                                .map(String::from)
                                .or_else(|| v.as_i64().map(|n| n.to_string()))
                        });
                        let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(50);

                        let Some(seeds) = seeds.filter(|s| !s.is_empty()) else {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "graph_rag_search requires 'table' and non-empty 'seeds' array." }],
                                "isError": true
                            }));
                        };
                        if table.is_empty() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "graph_rag_search requires 'table' and non-empty 'seeds' array." }],
                                "isError": true
                            }));
                        }
                        let sanitized_table = sanitize_table_name(table);
                        let mut all_rows = Vec::new();

                        for seed_val in seeds {
                            let seed_str = if let Some(s) = seed_val.as_str() {
                                s.to_string()
                            } else if let Some(n) = seed_val.as_i64() {
                                n.to_string()
                            } else {
                                seed_val.to_string()
                            };

                            let query = match mode {
                                "shortest_path" => {
                                    let target_str =
                                        escape_sql_literal(&target.clone().unwrap_or_default());
                                    format!(
                                        "SELECT * FROM graph_shortest_path('{}', '{}', '{}', 'auto') LIMIT {}",
                                        sanitized_table, escape_sql_literal(&seed_str), target_str, top_k
                                    )
                                }
                                "subgraph" => {
                                    format!(
                                        "SELECT * FROM graph_subgraph('{}', '{}', {}, 'auto') LIMIT {}",
                                        sanitized_table, escape_sql_literal(&seed_str), hops, top_k
                                    )
                                }
                                _ => {
                                    format!(
                                        "SELECT * FROM graph_neighbors('{}', '{}', {}, 'auto') LIMIT {}",
                                        sanitized_table, escape_sql_literal(&seed_str), hops, top_k
                                    )
                                }
                            };

                            if let Ok(rows) = self.query_to_json(&query).await {
                                all_rows.extend(rows);
                                if all_rows.len() >= top_k as usize {
                                    all_rows.truncate(top_k as usize);
                                    break;
                                }
                            }
                        }

                        Ok(json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&all_rows).unwrap_or_default() }]
                        }))
                    }
                    "store_memory" => {
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("");
                        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let tags = args.get("tags").and_then(|v| v.as_array());

                        if key.is_empty() || content.is_empty() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "store_memory requires 'key' and 'content'." }],
                                "isError": true
                            }));
                        }

                        let memory_table =
                            format!("{}.agent_memories", permissions.default_write_schema);
                        let (mem_schema, _) = split_schema_and_table(&memory_table);

                        // Auto-create memory table if needed
                        let table_check = self
                            .session()
                            .sql(&format!("SELECT * FROM {} LIMIT 0", memory_table))
                            .await;
                        if table_check.is_err() {
                            let _ = self
                                .session()
                                .sql(&format!("CREATE SCHEMA IF NOT EXISTS {}", mem_schema))
                                .await;
                            let create_ddl = format!(
                                "CREATE TABLE {} (key VARCHAR, content VARCHAR, tags VARCHAR, created_at VARCHAR, embedding VARCHAR)",
                                memory_table
                            );
                            let _ = self.session().sql(&create_ddl).await;
                        }

                        let embedding: Option<Vec<f32>> =
                            if let Some(arr) = args.get("embedding").and_then(|v| v.as_array()) {
                                Some(
                                    arr.iter()
                                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                                        .collect(),
                                )
                            } else {
                                self.fetch_embedding(content).await
                            };

                        let tags_str = tags
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .unwrap_or_default();

                        let created_at = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs()
                            .to_string();

                        let embedding_str = embedding
                            .as_ref()
                            .map(|v| {
                                format!(
                                    "[{}]",
                                    v.iter()
                                        .map(|f| f.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )
                            })
                            .unwrap_or_default();

                        let record = json!({
                            "key": key,
                            "content": content,
                            "tags": tags_str,
                            "created_at": created_at,
                            "embedding": embedding_str
                        });

                        match self.insert_records_internal(&memory_table, &[record]).await {
                            Ok(_) => Ok(json!({
                                "content": [{
                                    "type": "text",
                                    "text": format!("Memory stored successfully with key: '{}' (embedding: {}).", key, embedding.is_some())
                                }]
                            })),
                            Err(e) => Ok(json!({
                                "content": [{ "type": "text", "text": format!("Error storing memory: {}", e) }],
                                "isError": true
                            })),
                        }
                    }
                    "recall_memory" => {
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                        let tag = args.get("tag").and_then(|v| v.as_str());
                        let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5);

                        if query.is_empty() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "recall_memory requires 'query'." }],
                                "isError": true
                            }));
                        }

                        let memory_table =
                            format!("{}.agent_memories", permissions.default_write_schema);
                        if self
                            .session()
                            .sql(&format!("SELECT * FROM {} LIMIT 0", memory_table))
                            .await
                            .is_err()
                        {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "[]" }]
                            }));
                        }

                        let query_vector: Option<Vec<f32>> =
                            if let Some(arr) = args.get("embedding").and_then(|v| v.as_array()) {
                                Some(
                                    arr.iter()
                                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                                        .collect(),
                                )
                            } else {
                                self.fetch_embedding(query).await
                            };

                        let esc_query = query.replace('\'', "''");
                        let tag_filter = tag
                            .map(|t| format!("AND tags LIKE '%{}%'", t.replace('\'', "''")))
                            .unwrap_or_default();

                        let sql = if let Some(v) = query_vector {
                            let vec_str = format!(
                                "[{}]",
                                v.iter()
                                    .map(|f| f.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            );
                            format!(
                                "SELECT key, content, tags, created_at, l2_distance(embedding, '{}') AS distance FROM {} WHERE length(embedding) > 0 {} ORDER BY distance ASC LIMIT {}",
                                vec_str, memory_table, tag_filter, top_k
                            )
                        } else {
                            format!(
                                "SELECT key, content, tags, created_at, bm25_score(content, '{}') AS score FROM {} WHERE 1=1 {} ORDER BY score DESC LIMIT {}",
                                esc_query, memory_table, tag_filter, top_k
                            )
                        };

                        match self.query_to_json(&sql).await {
                            Ok(rows) => Ok(json!({
                                "content": [{ "type": "text", "text": serde_json::to_string_pretty(&rows).unwrap_or_default() }]
                            })),
                            Err(e) => Ok(json!({
                                "content": [{ "type": "text", "text": format!("Error recalling memory: {}", e) }],
                                "isError": true
                            })),
                        }
                    }
                    "ingest_records" => {
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let table_name = args
                            .get("table_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let records = args.get("records").and_then(|v| v.as_array());
                        let embed_col = args.get("embed_col").and_then(|v| v.as_str());

                        let Some(records_slice) = records else {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "ingest_records requires 'table_name' and 'records' array." }],
                                "isError": true
                            }));
                        };
                        if table_name.is_empty() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "ingest_records requires 'table_name' and 'records' array." }],
                                "isError": true
                            }));
                        }
                        let mut records = records_slice.clone();
                        if records.is_empty() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "No records to ingest." }]
                            }));
                        }

                        let (_schema, _clean_table, full_table_name) =
                            match resolve_table_name_and_schema(table_name, &permissions) {
                                Ok(res) => res,
                                Err(e) => {
                                    return Ok(json!({
                                        "content": [{ "type": "text", "text": e.to_string() }],
                                        "isError": true
                                    }));
                                }
                            };

                        if let Some(col) = embed_col {
                            if let Err(e) = validate_sql_identifier(col) {
                                return Ok(json!({
                                    "content": [{ "type": "text", "text": format!("Invalid embed column: {}", e) }],
                                    "isError": true
                                }));
                            }
                            for rec in &mut records {
                                if let Some(obj) = rec.as_object_mut() {
                                    if !obj.contains_key("embedding") {
                                        if let Some(text) = obj.get(col).and_then(|v| v.as_str()) {
                                            if let Some(vec) = self.fetch_embedding(text).await {
                                                let vec_str = format!(
                                                    "[{}]",
                                                    vec.iter()
                                                        .map(|f| f.to_string())
                                                        .collect::<Vec<_>>()
                                                        .join(", ")
                                                );
                                                obj.insert("embedding".to_string(), json!(vec_str));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        match self
                            .insert_records_internal(&full_table_name, &records)
                            .await
                        {
                            Ok(count) => Ok(json!({
                                "content": [{ "type": "text", "text": format!("Ingested {} records into {}.", count, full_table_name) }]
                            })),
                            Err(e) => Ok(json!({
                                "content": [{ "type": "text", "text": format!("Error ingesting records into {}: {}", full_table_name, e) }],
                                "isError": true
                            })),
                        }
                    }
                    "ingest_file" => {
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let file_path =
                            args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                        let user_table_name = args
                            .get("table_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let chunk_size = args
                            .get("chunk_size")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(500) as usize;
                        let embed_chunks = args
                            .get("embed_chunks")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);

                        if file_path.is_empty() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": "ingest_file requires 'file_path'." }],
                                "isError": true
                            }));
                        }

                        if let Err(e) = validate_local_path(file_path, "file_path") {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": e.to_string() }],
                                "isError": true
                            }));
                        }

                        let p = std::path::Path::new(file_path);
                        if !p.exists() {
                            return Ok(json!({
                                "content": [{ "type": "text", "text": format!("File not found: {}", file_path) }],
                                "isError": true
                            }));
                        }

                        let raw_table_name = if !user_table_name.trim().is_empty() {
                            user_table_name.trim()
                        } else {
                            p.file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("ingested_file")
                        };

                        let (_schema, _clean_table, full_table_name) =
                            match resolve_table_name_and_schema(raw_table_name, &permissions) {
                                Ok(res) => res,
                                Err(e) => {
                                    return Ok(json!({
                                        "content": [{ "type": "text", "text": e.to_string() }],
                                        "isError": true
                                    }));
                                }
                            };

                        let ext = p
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        let mut records = Vec::new();

                        if ext == "json" {
                            let content = std::fs::read_to_string(p)?;
                            if let Ok(val) = serde_json::from_str::<Value>(&content) {
                                if let Some(arr) = val.as_array() {
                                    records = arr.clone();
                                } else if val.is_object() {
                                    records.push(val);
                                }
                            }
                        } else if ext == "jsonl" || ext == "ndjson" {
                            let content = std::fs::read_to_string(p)?;
                            for line in content.lines() {
                                if !line.trim().is_empty() {
                                    if let Ok(val) = serde_json::from_str::<Value>(line) {
                                        records.push(val);
                                    }
                                }
                            }
                        } else if ext == "pdf" {
                            match pdf_extract::extract_text(p) {
                                Ok(content) => {
                                    let lines: Vec<&str> = content.lines().collect();
                                    let mut cur_chunk = String::new();
                                    let mut start_line = 1;
                                    let mut chunk_idx = 0;

                                    for (line_idx, line) in lines.iter().enumerate() {
                                        if cur_chunk.len() + line.len() > chunk_size
                                            && !cur_chunk.is_empty()
                                        {
                                            let mut rec = json!({
                                                "path": file_path,
                                                "chunk_index": chunk_idx,
                                                "start_line": start_line,
                                                "end_line": line_idx,
                                                "content": cur_chunk.clone(),
                                                "format": "pdf"
                                            });
                                            if embed_chunks {
                                                if let Some(vec) =
                                                    self.fetch_embedding(&cur_chunk).await
                                                {
                                                    let vec_str = format!(
                                                        "[{}]",
                                                        vec.iter()
                                                            .map(|f| f.to_string())
                                                            .collect::<Vec<_>>()
                                                            .join(", ")
                                                    );
                                                    rec["embedding"] = json!(vec_str);
                                                }
                                            }
                                            records.push(rec);
                                            chunk_idx += 1;
                                            cur_chunk.clear();
                                            start_line = line_idx + 1;
                                        }
                                        cur_chunk.push_str(line);
                                        cur_chunk.push('\n');
                                    }

                                    if !cur_chunk.is_empty() {
                                        let mut rec = json!({
                                            "path": file_path,
                                            "chunk_index": chunk_idx,
                                            "start_line": start_line,
                                            "end_line": lines.len(),
                                            "content": cur_chunk.clone(),
                                            "format": "pdf"
                                        });
                                        if embed_chunks {
                                            if let Some(vec) =
                                                self.fetch_embedding(&cur_chunk).await
                                            {
                                                let vec_str = format!(
                                                    "[{}]",
                                                    vec.iter()
                                                        .map(|f| f.to_string())
                                                        .collect::<Vec<_>>()
                                                        .join(", ")
                                                );
                                                rec["embedding"] = json!(vec_str);
                                            }
                                        }
                                        records.push(rec);
                                    }
                                }
                                Err(e) => {
                                    return Ok(json!({
                                        "content": [{ "type": "text", "text": format!("Error extracting text from PDF '{}': {}", file_path, e) }],
                                        "isError": true
                                    }));
                                }
                            }
                        } else if matches!(ext.as_str(), "xlsx" | "xls" | "xlsb" | "ods") {
                            use calamine::Reader;
                            match calamine::open_workbook_auto(p) {
                                Ok(mut workbook) => {
                                    let sheet_names = workbook.sheet_names().to_vec();
                                    for sheet in sheet_names {
                                        if let Ok(range) = workbook.worksheet_range(&sheet) {
                                            let mut rows_iter = range.rows();
                                            if let Some(header_row) = rows_iter.next() {
                                                let headers: Vec<String> = header_row
                                                    .iter()
                                                    .enumerate()
                                                    .map(|(idx, cell)| match cell {
                                                        calamine::Data::String(s)
                                                            if !s.trim().is_empty() =>
                                                        {
                                                            s.trim().to_string()
                                                        }
                                                        _ => format!("col_{}", idx + 1),
                                                    })
                                                    .collect();

                                                for (row_idx, row) in rows_iter.enumerate() {
                                                    let mut row_map = serde_json::Map::new();
                                                    row_map
                                                        .insert("sheet".to_string(), json!(sheet));
                                                    row_map.insert(
                                                        "row_index".to_string(),
                                                        json!(row_idx + 2),
                                                    );
                                                    let mut row_text_buf = String::new();
                                                    for (col_idx, cell) in row.iter().enumerate() {
                                                        let col_name = headers
                                                            .get(col_idx)
                                                            .cloned()
                                                            .unwrap_or_else(|| {
                                                                format!("col_{}", col_idx + 1)
                                                            });
                                                        let val = match cell {
                                                            calamine::Data::Empty => Value::Null,
                                                            calamine::Data::String(s) => {
                                                                row_text_buf.push_str(s);
                                                                row_text_buf.push(' ');
                                                                json!(s)
                                                            }
                                                            calamine::Data::Float(f) => json!(f),
                                                            calamine::Data::Int(i) => json!(i),
                                                            calamine::Data::Bool(b) => json!(b),
                                                            calamine::Data::DateTime(d) => {
                                                                json!(d.to_string())
                                                            }
                                                            calamine::Data::DateTimeIso(s) => {
                                                                json!(s)
                                                            }
                                                            calamine::Data::DurationIso(s) => {
                                                                json!(s)
                                                            }
                                                            calamine::Data::Error(e) => {
                                                                json!(format!("{:?}", e))
                                                            }
                                                        };
                                                        row_map.insert(col_name, val);
                                                    }
                                                    if embed_chunks
                                                        && !row_text_buf.trim().is_empty()
                                                    {
                                                        if let Some(vec) = self
                                                            .fetch_embedding(row_text_buf.trim())
                                                            .await
                                                        {
                                                            let vec_str = format!(
                                                                "[{}]",
                                                                vec.iter()
                                                                    .map(|f| f.to_string())
                                                                    .collect::<Vec<_>>()
                                                                    .join(", ")
                                                            );
                                                            row_map.insert(
                                                                "embedding".to_string(),
                                                                json!(vec_str),
                                                            );
                                                        }
                                                    }
                                                    records.push(Value::Object(row_map));
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    return Ok(json!({
                                        "content": [{ "type": "text", "text": format!("Error opening spreadsheet '{}': {}", file_path, e) }],
                                        "isError": true
                                    }));
                                }
                            }
                        } else if matches!(
                            ext.as_str(),
                            "png" | "jpg" | "jpeg" | "webp" | "gif" | "svg"
                        ) {
                            use base64::Engine;
                            match std::fs::read(p) {
                                Ok(img_bytes) => {
                                    let b64 = base64::prelude::BASE64_STANDARD.encode(&img_bytes);
                                    let mime_type = match ext.as_str() {
                                        "jpg" | "jpeg" => "image/jpeg",
                                        "png" => "image/png",
                                        "webp" => "image/webp",
                                        "gif" => "image/gif",
                                        "svg" => "image/svg+xml",
                                        _ => "image/png",
                                    };
                                    let data_uri = format!("data:{};base64,{}", mime_type, b64);
                                    let mut rec = json!({
                                        "path": file_path,
                                        "format": ext,
                                        "content": data_uri,
                                        "size_bytes": img_bytes.len(),
                                    });
                                    if embed_chunks {
                                        if let Some(vec) = self.fetch_image_embedding(&b64).await {
                                            let vec_str = format!(
                                                "[{}]",
                                                vec.iter()
                                                    .map(|f| f.to_string())
                                                    .collect::<Vec<_>>()
                                                    .join(", ")
                                            );
                                            rec["embedding"] = json!(vec_str);
                                        }
                                    }
                                    records.push(rec);
                                }
                                Err(e) => {
                                    return Ok(json!({
                                        "content": [{ "type": "text", "text": format!("Error reading image '{}': {}", file_path, e) }],
                                        "isError": true
                                    }));
                                }
                            }
                        } else if matches!(
                            ext.as_str(),
                            "csv"
                                | "tsv"
                                | "psv"
                                | "docx"
                                | "pptx"
                                | "xml"
                                | "zip"
                                | "zst"
                                | "zstd"
                        ) || self.extension.file_extractor.is_some()
                        {
                            let raw_bytes = std::fs::read(p)?;
                            match crate::ingest_formats::parse_file_bytes_with_custom(
                                &raw_bytes,
                                file_path,
                                &ext,
                                chunk_size,
                                self.extension.file_extractor,
                            ) {
                                Ok(recs) => {
                                    records = recs;
                                    if embed_chunks {
                                        for rec in &mut records {
                                            let text_to_embed = if let Some(content_str) =
                                                rec.get("content").and_then(|c| c.as_str())
                                            {
                                                Some(content_str.to_string())
                                            } else if let Some(obj) = rec.as_object() {
                                                let vals: Vec<String> = obj
                                                    .iter()
                                                    .filter(|(k, _)| {
                                                        k.as_str() != "path"
                                                            && k.as_str() != "embedding"
                                                    })
                                                    .filter_map(|(_, v)| match v {
                                                        Value::String(s)
                                                            if !s.trim().is_empty() =>
                                                        {
                                                            Some(s.clone())
                                                        }
                                                        Value::Number(n) => Some(n.to_string()),
                                                        _ => None,
                                                    })
                                                    .collect();
                                                if vals.is_empty() {
                                                    None
                                                } else {
                                                    Some(vals.join(" "))
                                                }
                                            } else {
                                                None
                                            };

                                            if let Some(txt) = text_to_embed {
                                                if !txt.trim().is_empty() {
                                                    if let Some(vec) =
                                                        self.fetch_embedding(txt.trim()).await
                                                    {
                                                        let vec_str = format!(
                                                            "[{}]",
                                                            vec.iter()
                                                                .map(|f| f.to_string())
                                                                .collect::<Vec<_>>()
                                                                .join(", ")
                                                        );
                                                        rec["embedding"] = json!(vec_str);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    return Ok(json!({
                                        "content": [{ "type": "text", "text": format!("Error extracting records from '{}': {}", file_path, e) }],
                                        "isError": true
                                    }));
                                }
                            }
                        } else {
                            // Text / code chunking
                            let content = std::fs::read_to_string(p)?;
                            let lines: Vec<&str> = content.lines().collect();
                            let mut cur_chunk = String::new();
                            let mut start_line = 1;
                            let mut chunk_idx = 0;

                            for (line_idx, line) in lines.iter().enumerate() {
                                if cur_chunk.len() + line.len() > chunk_size
                                    && !cur_chunk.is_empty()
                                {
                                    let mut rec = json!({
                                        "path": file_path,
                                        "chunk_index": chunk_idx,
                                        "start_line": start_line,
                                        "end_line": line_idx,
                                        "content": cur_chunk.clone()
                                    });
                                    if embed_chunks {
                                        if let Some(vec) = self.fetch_embedding(&cur_chunk).await {
                                            let vec_str = format!(
                                                "[{}]",
                                                vec.iter()
                                                    .map(|f| f.to_string())
                                                    .collect::<Vec<_>>()
                                                    .join(", ")
                                            );
                                            rec["embedding"] = json!(vec_str);
                                        }
                                    }
                                    records.push(rec);
                                    chunk_idx += 1;
                                    cur_chunk.clear();
                                    start_line = line_idx + 1;
                                }
                                cur_chunk.push_str(line);
                                cur_chunk.push('\n');
                            }

                            if !cur_chunk.is_empty() {
                                let mut rec = json!({
                                    "path": file_path,
                                    "chunk_index": chunk_idx,
                                    "start_line": start_line,
                                    "end_line": lines.len(),
                                    "content": cur_chunk.clone()
                                });
                                if embed_chunks {
                                    if let Some(vec) = self.fetch_embedding(&cur_chunk).await {
                                        let vec_str = format!(
                                            "[{}]",
                                            vec.iter()
                                                .map(|f| f.to_string())
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        );
                                        rec["embedding"] = json!(vec_str);
                                    }
                                }
                                records.push(rec);
                            }
                        }

                        match self
                            .insert_records_internal(&full_table_name, &records)
                            .await
                        {
                            Ok(count) => Ok(json!({
                                "content": [{ "type": "text", "text": format!("Ingested {} records from '{}' into {}.", count, file_path, full_table_name) }]
                            })),
                            Err(e) => Ok(json!({
                                "content": [{ "type": "text", "text": format!("Error ingesting file into {}: {}", full_table_name, e) }],
                                "isError": true
                            })),
                        }
                    }
                    _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
                }
            }
            _ => Err(anyhow::anyhow!("Method not found: {}", method)),
        }
    }

    async fn handle_notification(&self, method: &str, _params: Value) -> Result<()> {
        info!("Handling notification method: {}", method);
        Ok(())
    }

    fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        if let Ok(json) = serde_json::to_string(&response) {
            let mut stdout = std::io::stdout();
            writeln!(stdout, "{}", json)?;
            stdout.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_mcp_initialize() {
        let server = McpServer::new();
        let res = server
            .handle_request("initialize", json!({}))
            .await
            .unwrap();
        assert_eq!(res["protocolVersion"], "2024-11-05");
        assert_eq!(res["serverInfo"]["name"], "benostreamdb-mcp");
    }

    #[tokio::test]
    async fn test_mcp_tools_list() {
        let server = McpServer::new();
        let res = server
            .handle_request("tools/list", json!({}))
            .await
            .unwrap();
        let tools = res["tools"].as_array().unwrap();
        assert!(tools.iter().any(|t| t["name"] == "execute_sql"));
        assert!(tools.iter().any(|t| t["name"] == "create_scratch_table"));
        assert!(tools.iter().any(|t| t["name"] == "export_table"));
        assert!(tools.iter().any(|t| t["name"] == "hybrid_search"));
        assert!(tools.iter().any(|t| t["name"] == "graph_rag_search"));
        assert!(tools.iter().any(|t| t["name"] == "store_memory"));
        assert!(tools.iter().any(|t| t["name"] == "recall_memory"));
        assert!(tools.iter().any(|t| t["name"] == "ingest_records"));
        assert!(tools.iter().any(|t| t["name"] == "ingest_file"));
    }

    #[tokio::test]
    async fn test_mcp_execute_sql() {
        let server = McpServer::new();
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT 1 as x"
                    }
                }),
            )
            .await
            .unwrap();

        let content = res["content"][0]["text"].as_str().unwrap();
        assert!(content.contains("\"x\":1") || content.contains("\"x\": 1"));
    }

    /// R1 + R5 reachable through MCP: the graph UDAFs carry non-strict
    /// (`variadic_any`) signatures inherited from the core session, so an
    /// `Int64`-columned edge table can be queried with bare integer/float
    /// literals (no `arrow_cast(...)`); and `session.sql` tolerates a trailing
    /// `;` plus `--` line comments.
    #[tokio::test]
    async fn test_mcp_graph_udaf_without_casts() {
        let server = McpServer::new();

        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": { "table_name": "edges_i64", "schema_ddl": "source BIGINT, target BIGINT" }
                }),
            )
            .await
            .unwrap();
        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_records",
                    "arguments": {
                        "table_name": "edges_i64",
                        "records": [
                            {"source": 0, "target": 1},
                            {"source": 1, "target": 2},
                            {"source": 2, "target": 0}
                        ]
                    }
                }),
            )
            .await
            .unwrap();

        // Non-strict graph UDAF: Int64 columns + bare literals, no arrow_cast.
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT graph_pagerank(source, target, 0.85, 20) AS pr FROM scratch.edges_i64"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            res.get("isError").is_none(),
            "graph UDAF must run via MCP without casts: {res}"
        );
        assert!(!res["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .is_empty());

        // R5: trailing semicolon + line comments are tolerated via MCP too.
        let res2 = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "-- pagerank over Int64 edges\nSELECT graph_pagerank(source, target, 0.85, 20) AS pr FROM scratch.edges_i64;"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            res2.get("isError").is_none(),
            "trailing ';' and comments must be tolerated via MCP: {res2}"
        );
    }

    #[tokio::test]
    async fn test_mcp_scratch_table_workflow() {
        let server = McpServer::new();

        // 1. Create table
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "test_table",
                        "schema_ddl": "id INT, val VARCHAR"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res.get("isError").is_none());

        // 2. Insert records via ingest_records
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_records",
                    "arguments": {
                        "table_name": "test_table",
                        "records": [
                            {"id": 1, "val": "hello"},
                            {"id": 2, "val": "world"}
                        ]
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res.get("isError").is_none());

        // 3. Select back
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT * FROM scratch.test_table ORDER BY id"
                    }
                }),
            )
            .await
            .unwrap();
        let content = res["content"][0]["text"].as_str().unwrap();
        assert!(content.contains("hello"));
        assert!(content.contains("world"));
    }

    #[tokio::test]
    async fn test_mcp_export_workflow() {
        let server = McpServer::new();

        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": { "table_name": "export_test", "schema_ddl": "id INT" }
                }),
            )
            .await
            .unwrap();

        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_records",
                    "arguments": { "table_name": "export_test", "records": [{"id": 42}] }
                }),
            )
            .await
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let export_path = temp_dir.join(format!(
            "export_test_{}.parquet",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&export_path);
        let _ = std::fs::remove_file(&export_path);

        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "export_table",
                    "arguments": {
                        "table_name": "export_test",
                        "export_path": export_path.to_str().unwrap()
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res.get("isError").is_none());
        assert!(export_path.exists());
    }

    #[tokio::test]
    async fn test_mcp_scratch_only_guard() {
        let server = McpServer::new();
        // Writes to the main catalog are rejected.
        for q in [
            "CREATE TABLE default.evil (id INT)",
            "INSERT INTO default.evil VALUES (1)",
            "UPDATE default.evil SET id = 2",
            "DELETE FROM default.evil",
            "DROP TABLE default.evil",
            "CREATE INDEX idx ON default.evil (id)",
            "ALTER TABLE default.evil ADD INDEX (id)",
        ] {
            let res = server
                .handle_request(
                    "tools/call",
                    json!({ "name": "execute_sql", "arguments": { "query": q } }),
                )
                .await
                .unwrap();
            assert_eq!(res["isError"], true, "expected rejection for: {q}");
        }
        // Scratch writes (including index creation) are allowed.
        for q in [
            "CREATE TABLE scratch.ok (id INT)",
            "CREATE INDEX idx ON scratch.ok (id)",
        ] {
            let res = server
                .handle_request(
                    "tools/call",
                    json!({ "name": "execute_sql", "arguments": { "query": q } }),
                )
                .await
                .unwrap();
            assert!(
                res.get("isError").is_none(),
                "expected success for {q}: {res}"
            );
        }
    }

    #[tokio::test]
    async fn test_mcp_list_graph_tables() {
        let server = McpServer::new();
        // Declare an edge table via the scratch tool (table_type metadata).
        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": { "table_name": "g_edges", "schema_ddl": "src INT, dst INT", "table_type": "edge" }
                }),
            )
            .await
            .unwrap();
        let res = server
            .handle_request(
                "tools/call",
                json!({ "name": "list_graph_tables", "arguments": {} }),
            )
            .await
            .unwrap();
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("g_edges"), "expected g_edges in: {text}");
        assert!(text.contains("edge"), "expected table_type edge in: {text}");
    }

    #[tokio::test]
    async fn test_mcp_subscribe_events_tool() {
        let server = McpServer::new();
        // The tool must be advertised.
        let tools = server
            .handle_request("tools/list", json!({}))
            .await
            .unwrap();
        let names: Vec<&str> = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(names.contains(&"subscribe_events"), "tools: {names:?}");

        // Create a scratch table, then drain its (empty) change feed.
        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": { "table_name": "sub_t", "schema_ddl": "id INT" }
                }),
            )
            .await
            .unwrap();
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "subscribe_events",
                    "arguments": { "table": "scratch.sub_t", "max_events": 5, "timeout_ms": 50 }
                }),
            )
            .await
            .unwrap();
        assert!(
            res.get("isError").is_none(),
            "subscribe_events errored: {res}"
        );
    }

    #[tokio::test]
    async fn test_mcp_resources() {
        let server = McpServer::new();

        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": { "table_name": "resource_test", "schema_ddl": "id INT" }
                }),
            )
            .await
            .unwrap();

        let res = server
            .handle_request("resources/list", json!({}))
            .await
            .unwrap();
        let resources = res["resources"].as_array().unwrap();
        println!("Resources: {:?}", resources);
        assert!(resources
            .iter()
            .any(|r| r["name"].as_str().unwrap().contains("resource_test")));

        let res = server
            .handle_request(
                "resources/read",
                json!({
                    "uri": "benostreamdb://tables/scratch/resource_test"
                }),
            )
            .await
            .unwrap();

        let content = res["contents"][0]["text"].as_str().unwrap();
        assert!(content.contains("id"));
    }

    #[tokio::test]
    async fn test_mcp_prompts() {
        let server = McpServer::new();
        let list_res = server
            .handle_request("prompts/list", json!({}))
            .await
            .unwrap();
        let prompts = list_res["prompts"].as_array().unwrap();
        assert!(prompts
            .iter()
            .any(|p| p["name"] == "benostreamdb_sql_expert"));
        assert!(prompts.iter().any(|p| p["name"] == "agent_memory_guide"));
        assert!(prompts.iter().any(|p| p["name"] == "code_rag_strategy"));

        let get_res = server
            .handle_request(
                "prompts/get",
                json!({
                    "name": "benostreamdb_sql_expert",
                    "arguments": { "topic": "vector" }
                }),
            )
            .await
            .unwrap();
        let msg = get_res["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(msg.contains("Vector Search"));
        assert!(msg.contains("vector"));
    }

    #[tokio::test]
    async fn test_mcp_agent_memory() {
        let server = McpServer::new();

        // 1. Store memory
        let store_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "store_memory",
                    "arguments": {
                        "key": "user_pref_theme",
                        "content": "User prefers dark mode and high contrast charts.",
                        "tags": ["preferences", "ui"],
                        "embedding": [0.1, 0.2, 0.3]
                    }
                }),
            )
            .await
            .unwrap();
        assert!(store_res.get("isError").is_none());

        // 2. Recall memory
        let recall_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "recall_memory",
                    "arguments": {
                        "query": "dark mode preferences",
                        "tag": "preferences",
                        "embedding": [0.1, 0.2, 0.3]
                    }
                }),
            )
            .await
            .unwrap();
        let text = recall_res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("user_pref_theme"));
        assert!(text.contains("dark mode"));
    }

    #[tokio::test]
    async fn test_mcp_hybrid_search() {
        let server = McpServer::new();

        // Ingest sample records
        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_records",
                    "arguments": {
                        "table_name": "search_docs",
                        "records": [
                            {
                                "id": "doc1",
                                "content": "Fast HNSW vector indexing and distance search",
                                "embedding": "[0.9, 0.1, 0.0]"
                            },
                            {
                                "id": "doc2",
                                "content": "Property graph traversal and shortest paths",
                                "embedding": "[0.0, 0.8, 0.1]"
                            }
                        ]
                    }
                }),
            )
            .await
            .unwrap();

        // Run hybrid search
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "hybrid_search",
                    "arguments": {
                        "table": "scratch.search_docs",
                        "query_text": "vector search",
                        "query_vector": [0.9, 0.1, 0.0],
                        "top_k": 2
                    }
                }),
            )
            .await
            .unwrap();
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("doc1"));
        assert!(text.contains("_rrf_score"));
    }

    #[tokio::test]
    async fn test_mcp_ingest_records_and_file() {
        let server = McpServer::new();

        // 1. Ingest JSON file
        let mut temp = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        use std::io::Write;
        let data = json!([
            {"id": "c1", "content": "function authenticate() { return true; }"},
            {"id": "c2", "content": "function authorize() { return false; }"}
        ]);
        let _ = serde_json::to_writer(&mut temp, &data);
        let _ = temp.flush();

        let path = temp.path().to_str().unwrap().to_string();
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": path,
                        "table_name": "file_code_chunks"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res.get("isError").is_none());

        // Verify via execute_sql
        let sql_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT count(*) AS total FROM scratch.file_code_chunks"
                    }
                }),
            )
            .await
            .unwrap();
        let text = sql_res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"total\":2") || text.contains("\"total\": 2"));
    }

    #[tokio::test]
    async fn test_mcp_ingest_image_files() {
        let server = McpServer::new();
        use std::io::Write;

        // 1. PNG Image test
        let mut png_temp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        let png_bytes: [u8; 67] = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
            0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, // IDAT
            0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82, // IEND
        ];
        png_temp.write_all(&png_bytes).unwrap();
        png_temp.flush().unwrap();

        let path = png_temp.path().to_str().unwrap().to_string();
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": path,
                        "table_name": "png_assets"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res.get("isError").is_none(), "png ingest failed: {res}");

        // Verify via execute_sql
        let sql_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT format, size_bytes, substr(content, 1, 22) AS header FROM scratch.png_assets"
                    }
                }),
            )
            .await
            .unwrap();
        let text = sql_res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("png"));
        assert!(text.contains("data:image/png;base64,"));

        // 2. SVG Image test
        let mut svg_temp = tempfile::Builder::new().suffix(".svg").tempfile().unwrap();
        let svg_data = br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><circle cx="5" cy="5" r="5" fill="blue"/></svg>"#;
        svg_temp.write_all(svg_data).unwrap();
        svg_temp.flush().unwrap();

        let svg_path = svg_temp.path().to_str().unwrap().to_string();
        let res2 = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": svg_path,
                        "table_name": "svg_assets"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res2.get("isError").is_none(), "svg ingest failed: {res2}");

        let sql_res2 = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT format, substr(content, 1, 26) AS header FROM scratch.svg_assets"
                    }
                }),
            )
            .await
            .unwrap();
        let text2 = sql_res2["content"][0]["text"].as_str().unwrap();
        assert!(text2.contains("svg"));
        assert!(text2.contains("data:image/svg+xml;base64,"));
    }

    #[tokio::test]
    async fn test_mcp_ingest_text_and_code_chunking() {
        let server = McpServer::new();
        use std::io::Write;

        let mut code_temp = tempfile::Builder::new().suffix(".rs").tempfile().unwrap();
        let code_data = "fn compute_similarity(a: &[f32], b: &[f32]) -> f32 {\n    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();\n    dot\n}\n\nfn normalize(v: &mut [f32]) {\n    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();\n    for x in v.iter_mut() { *x /= norm; }\n}\n";
        code_temp.write_all(code_data.as_bytes()).unwrap();
        code_temp.flush().unwrap();

        let path = code_temp.path().to_str().unwrap().to_string();
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": path,
                        "table_name": "rust_code_chunks",
                        "chunk_size": 100
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res.get("isError").is_none(), "code chunking failed: {res}");

        let sql_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT chunk_index, start_line, end_line, length(content) AS len FROM scratch.rust_code_chunks ORDER BY chunk_index"
                    }
                }),
            )
            .await
            .unwrap();
        let text = sql_res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"chunk_index\":0") || text.contains("\"chunk_index\": 0"));
    }

    #[tokio::test]
    async fn test_mcp_ingest_pdf_and_spreadsheet_error_handling() {
        let server = McpServer::new();
        use std::io::Write;

        // Ingest invalid/corrupted PDF returns a clean MCP error
        let mut corrupt_pdf = tempfile::Builder::new().suffix(".pdf").tempfile().unwrap();
        corrupt_pdf.write_all(b"not a valid pdf content").unwrap();
        corrupt_pdf.flush().unwrap();

        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": corrupt_pdf.path().to_str().unwrap(),
                        "table_name": "bad_pdf"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(res["isError"], true);
        let err_msg = res["content"][0]["text"].as_str().unwrap();
        assert!(err_msg.contains("Error extracting text from PDF"));

        // Ingest invalid spreadsheet returns a clean MCP error
        let mut corrupt_xlsx = tempfile::Builder::new().suffix(".xlsx").tempfile().unwrap();
        corrupt_xlsx.write_all(b"not a valid xlsx archive").unwrap();
        corrupt_xlsx.flush().unwrap();

        let res_xlsx = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": corrupt_xlsx.path().to_str().unwrap(),
                        "table_name": "bad_xlsx"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(res_xlsx["isError"], true);
        let err_msg_xlsx = res_xlsx["content"][0]["text"].as_str().unwrap();
        assert!(err_msg_xlsx.contains("Error opening spreadsheet"));
    }

    #[tokio::test]
    async fn test_mcp_ingest_file_default_table_name() {
        let server = McpServer::new();

        // Create a temp json file with a specific stem e.g. "audit_metrics.json"
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("audit_metrics.json");
        let data = json!([
            {"metric": "latency_p99", "value": 12.5},
            {"metric": "error_rate", "value": 0.001}
        ]);
        std::fs::write(&file_path, serde_json::to_string(&data).unwrap()).unwrap();

        // Call ingest_file WITHOUT table_name
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": file_path.to_str().unwrap()
                    }
                }),
            )
            .await
            .unwrap();
        assert!(res.get("isError").is_none(), "ingest failed: {res}");

        // Verify table scratch.audit_metrics was created and populated
        let sql_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT count(*) AS total FROM scratch.audit_metrics"
                    }
                }),
            )
            .await
            .unwrap();
        let text = sql_res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"total\":2") || text.contains("\"total\": 2"));
    }

    #[tokio::test]
    async fn test_mcp_server_fuzz_resilience() {
        let server = McpServer::new();

        // 1. Unknown / malformed methods should return error without panicking
        let bad_method_res = server
            .handle_request("non_existent_method_xyz", json!({}))
            .await;
        assert!(bad_method_res.is_err());
        assert!(bad_method_res
            .unwrap_err()
            .to_string()
            .contains("Method not found"));

        // 2. Tool calls with completely invalid arguments should return clean errors
        let bad_tool_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "unknown_tool",
                    "arguments": { "foo": "bar" }
                }),
            )
            .await;
        assert!(bad_tool_res.is_err());

        // 3. Fuzzing execute_sql with garbage SQL strings
        for bad_sql in [
            "",
            "   ",
            ";;;",
            "SELECT FROM WHERE",
            "DROP SYSTEM EVERYTHING -- injection",
            "1234567890",
            "\0\0\0",
        ] {
            let res = server
                .handle_request(
                    "tools/call",
                    json!({
                        "name": "execute_sql",
                        "arguments": { "query": bad_sql }
                    }),
                )
                .await
                .unwrap();
            assert_eq!(res["isError"], true, "expected error for: {bad_sql}");
        }

        // 4. Fuzzing parse_file_bytes with corrupted / random binary inputs across all formats
        let fuzz_payloads: Vec<&[u8]> = vec![
            b"",
            b"\0",
            b"\0\0\0\0\0\0\0\0",
            b"\"unclosed quote,col2,col3",
            b"'unclosed single quote\tcol2",
            b"|pipe|unclosed|quote\"without\"end|",
            b"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"",
            b"\\\\\\\\\\\\\\\\\\\\\\\\\\\\",
            b"<not<an<xml>tag",
            b"<?xml?><><><>",
            b"PK\x03\x04corrupted_truncated_zip_header",
            b"\x28\xb5\x2f\xfdcorrupted_zstd_magic",
            b"\xff\xfe\xfd\xfc\xfb\xfa\x00\x01\x02",
        ];

        for payload in &fuzz_payloads {
            for ext in [
                "csv", "tsv", "psv", "xml", "docx", "pptx", "zip", "zst", "txt",
            ] {
                // Must not panic under any corrupted payload
                let _ = crate::ingest_formats::parse_file_bytes(payload, "fuzz_input", ext, 500);
            }
        }

        // 5. Fuzzing all MCP tools with null, empty, or mismatched parameter types
        let all_tools = [
            "execute_sql",
            "create_scratch_table",
            "export_table",
            "subscribe_events",
            "hybrid_search",
            "graph_rag_search",
            "store_memory",
            "recall_memory",
            "ingest_records",
            "ingest_file",
        ];

        let garbage_args = [
            json!({}),
            json!(null),
            json!([]),
            json!("string_instead_of_object"),
            json!({ "query": null, "table": null, "seeds": null, "records": null }),
            json!({ "query": 12345, "table": true, "seeds": "not_an_array", "records": "not_array" }),
            json!({ "query": "\0\0\0", "table_name": "", "file_path": "/nonexistent/path/\0" }),
            json!({ "seeds": [], "records": [] }),
            json!({ "top_k": -999, "hops": -10, "rrf_k": -1 }),
        ];

        for tool_name in all_tools {
            for args in &garbage_args {
                let call_res = server
                    .handle_request(
                        "tools/call",
                        json!({
                            "name": tool_name,
                            "arguments": args
                        }),
                    )
                    .await;
                // Calls either gracefully return an isError response or Err, but NEVER panic
                if let Ok(res) = call_res {
                    // Valid response structure
                    assert!(res.is_object());
                }
            }
        }

        // 6. Verify server is still completely healthy and operational after all fuzzing
        let ok_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": { "query": "SELECT 1337 AS alive" }
                }),
            )
            .await
            .unwrap();
        assert!(ok_res.get("isError").is_none());
        assert!(ok_res["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("1337"));
    }

    #[test]
    fn test_mcp_sanitize_table_name() {
        assert_eq!(sanitize_table_name("my report 2024"), "my_report_2024");
        assert_eq!(
            sanitize_table_name("Q3-Audit (Final).xlsx"),
            "q3_audit_final_xlsx"
        );
        assert_eq!(sanitize_table_name("2024_financials"), "t_2024_financials");
        assert_eq!(
            sanitize_table_name("audit; DROP TABLE scratch.audit; --"),
            "audit_drop_table_scratch_audit"
        );
        assert_eq!(
            sanitize_table_name("scratch.sales report 2024"),
            "scratch.sales_report_2024"
        );
        assert_eq!(
            sanitize_table_name("default.malicious"),
            "default_malicious"
        );
        assert_eq!(sanitize_table_name("   "), "table");
        assert_eq!(sanitize_table_name("___"), "table");
    }

    #[tokio::test]
    async fn test_mcp_injection_prevention() {
        let server = McpServer::new();

        // 1. Injected schema_ddl with chained DROP TABLE
        let bad_ddl_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "clean_tab",
                        "schema_ddl": "id INT); DROP TABLE scratch.important; --"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_ddl_res["isError"], true,
            "expected rejection of malicious schema_ddl: {bad_ddl_res}"
        );

        // 2. Injected table name with spaces and commands is sanitized into safe table
        let space_table_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "my financial statement 2024; SELECT 1",
                        "schema_ddl": "id INT, note VARCHAR"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            space_table_res.get("isError").is_none(),
            "table creation failed: {space_table_res}"
        );
        assert!(
            space_table_res["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("scratch.my_financial_statement_2024_select_1"),
            "expected sanitized name with spaces stripped to underscore"
        );

        // 3. Injected text_col in hybrid_search
        let bad_search_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "hybrid_search",
                    "arguments": {
                        "table": "scratch.my_financial_statement_2024_select_1",
                        "query_text": "audit",
                        "text_col": "note) FROM scratch.x; DROP TABLE ...; --"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_search_res["isError"], true,
            "expected rejection of malicious text_col: {bad_search_res}"
        );

        // 4. Ingest file with spaces in filename derives sanitized table name
        let temp_dir = tempfile::tempdir().unwrap();
        let file_with_spaces = temp_dir
            .path()
            .join("Company Q3 Earnings Report (Draft).txt");
        std::fs::write(&file_with_spaces, "Revenue was up 25% year over year.").unwrap();

        let ingest_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": file_with_spaces.to_str().unwrap()
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            ingest_res.get("isError").is_none(),
            "ingest failed: {ingest_res}"
        );
        let ingest_text = ingest_res["content"][0]["text"].as_str().unwrap();
        assert!(
            ingest_text.contains("scratch.company_q3_earnings_report_draft"),
            "expected spaces and parentheses replaced with underscores, got: {ingest_text}"
        );
    }

    #[tokio::test]
    async fn test_mcp_custom_extension_and_zstd() {
        // Define a custom vertical extension (e.g. specialized domain table schema and format)
        let custom_schema_resolver: CustomSchemaResolver = |table_type| match table_type {
            "custom_vertical" => Some("item_id VARCHAR, score DOUBLE, label VARCHAR"),
            _ => None,
        };

        let custom_file_extractor: crate::ingest_formats::CustomFileExtractor =
            |raw_bytes, file_path, ext, _chunk_size| {
                if ext == "custom" {
                    let content = String::from_utf8_lossy(raw_bytes);
                    let mut recs = Vec::new();
                    for line in content.lines() {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() == 3 {
                            recs.push(json!({
                                "path": file_path,
                                "item_id": parts[0].trim(),
                                "score": parts[1].trim().parse::<f64>().unwrap_or(0.0),
                                "label": parts[2].trim(),
                                "format": "custom"
                            }));
                        }
                    }
                    Some(Ok(recs))
                } else {
                    None
                }
            };

        let extension = McpExtension {
            server_name: Some("benostreamdb-custom-mcp".to_string()),
            server_version: Some("1.0.0".to_string()),
            schema_resolver: Some(custom_schema_resolver),
            file_extractor: Some(custom_file_extractor),
            additional_tools: vec![json!({
                "name": "custom_domain_ping",
                "description": "Test vertical ping",
                "inputSchema": { "type": "object", "properties": {} }
            })],
            permissions: None,
            ..Default::default()
        };

        let server = McpServer::with_extension(extension);

        // 1. Verify custom server info
        let init_res = server
            .handle_request("initialize", json!({}))
            .await
            .unwrap();
        assert_eq!(init_res["serverInfo"]["name"], "benostreamdb-custom-mcp");

        // 2. Verify additional tool appears in tools/list
        let tools_res = server
            .handle_request("tools/list", json!({}))
            .await
            .unwrap();
        let tools = tools_res["tools"].as_array().unwrap();
        assert!(tools.iter().any(|t| t["name"] == "custom_domain_ping"));

        // 3. Verify auto-generated schema from custom table_type
        let create_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "my_custom_table",
                        "table_type": "custom_vertical"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            create_res.get("isError").is_none(),
            "create table failed: {create_res}"
        );

        let temp_dir = tempfile::tempdir().unwrap();

        // 4. Ingest file using custom extractor
        let custom_file = temp_dir.path().join("data.custom");
        std::fs::write(&custom_file, "item_1, 98.5, ALPHA\nitem_2, 72.1, BETA\n").unwrap();

        let ingest_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": custom_file.to_str().unwrap(),
                        "table_name": "my_custom_table"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            ingest_res.get("isError").is_none(),
            "ingest failed: {ingest_res}"
        );

        let query_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT item_id, score, label FROM scratch.my_custom_table ORDER BY score DESC"
                    }
                }),
            )
            .await
            .unwrap();
        let q_text = query_res["content"][0]["text"].as_str().unwrap();
        assert!(q_text.contains("item_1") && q_text.contains("ALPHA"));

        // 2. Test Zstd-compressed text ingestion
        let zst_path = temp_dir.path().join("data.txt.zst");
        let plain_data = b"Line 1: Q1 revenue\nLine 2: Q2 revenue\nLine 3: Q3 revenue\n";
        let compressed = zstd::encode_all(&plain_data[..], 3).unwrap();
        std::fs::write(&zst_path, &compressed).unwrap();

        let zst_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": zst_path.to_str().unwrap(),
                        "table_name": "acme_zst"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            zst_res.get("isError").is_none(),
            "Zstd ingest failed: {zst_res}"
        );
    }

    #[tokio::test]
    async fn test_mcp_ingest_csv_tsv_psv_tables() {
        let server = McpServer::new();
        server
            .handle_request("initialize", json!({}))
            .await
            .unwrap();

        let temp_dir = tempfile::tempdir().unwrap();

        // 1. Comma-Separated Values (.csv)
        let csv_path = temp_dir.path().join("inventory.csv");
        let csv_data = "item_id,item_name,unit_price,in_stock\n1,Mechanical Keyboard,129.99,true\n2,Gaming Mouse,59.95,false\n";
        std::fs::write(&csv_path, csv_data).unwrap();

        let csv_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": csv_path.to_str().unwrap(),
                        "table_name": "inventory_csv"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            csv_res.get("isError").is_none(),
            "CSV ingest failed: {csv_res}"
        );

        let q_csv = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT item_id, item_name, unit_price, in_stock FROM scratch.inventory_csv WHERE unit_price > 100.0"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_csv.get("isError").is_none());
        let csv_text = q_csv["content"][0]["text"].as_str().unwrap();
        assert!(csv_text.contains("Mechanical Keyboard") && csv_text.contains("129.99"));

        // 2. Tab-Separated Values (.tsv)
        let tsv_path = temp_dir.path().join("users.tsv");
        let tsv_data = "user_id\tuser_handle\trank_score\n101\talice\t98.7\n102\tbob\t85.2\n";
        std::fs::write(&tsv_path, tsv_data).unwrap();

        let tsv_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": tsv_path.to_str().unwrap(),
                        "table_name": "users_tsv"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            tsv_res.get("isError").is_none(),
            "TSV ingest failed: {tsv_res}"
        );

        let q_tsv = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT user_id, user_handle, rank_score FROM scratch.users_tsv WHERE rank_score >= 90.0"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_tsv.get("isError").is_none());
        let tsv_text = q_tsv["content"][0]["text"].as_str().unwrap();
        assert!(tsv_text.contains("alice") && tsv_text.contains("98.7"));

        // 3. Pipe-Separated Values (.psv)
        let psv_path = temp_dir.path().join("regions.psv");
        let psv_data = "region_code|region_name|population|active\nNA|North America|580000000|true\nEU|Europe|740000000|true\n";
        std::fs::write(&psv_path, psv_data).unwrap();

        let psv_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": psv_path.to_str().unwrap(),
                        "table_name": "regions_psv"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            psv_res.get("isError").is_none(),
            "PSV ingest failed: {psv_res}"
        );

        let q_psv = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT region_code, region_name, population FROM scratch.regions_psv ORDER BY population DESC"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_psv.get("isError").is_none());
        let psv_text = q_psv["content"][0]["text"].as_str().unwrap();
        assert!(psv_text.contains("Europe") && psv_text.contains("North America"));

        // 4. Quoted text with "" and '' and backslash escaping
        let quoted_csv_path = temp_dir.path().join("quoted.csv");
        let quoted_csv_data = "\
id,title,author,comment
1,\"He said \"\"Hello, world!\"\"\",'O''Reilly',\"Multi-line \n comment with , comma\"
2,\"Escaped \\\"quote\\\" inside\",'Escaped \\'single\\' quote',\"Regular text\"
";
        std::fs::write(&quoted_csv_path, quoted_csv_data).unwrap();

        let quoted_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": quoted_csv_path.to_str().unwrap(),
                        "table_name": "quoted_books"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            quoted_res.get("isError").is_none(),
            "Quoted CSV ingest failed: {quoted_res}"
        );

        let q_quoted = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT id, title, author FROM scratch.quoted_books ORDER BY id"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_quoted.get("isError").is_none());
        let quoted_text = q_quoted["content"][0]["text"].as_str().unwrap();
        assert!(quoted_text.contains(r#"He said \"Hello, world!\""#));
        assert!(quoted_text.contains("O'Reilly"));
        assert!(quoted_text.contains(r#"Escaped \"quote\" inside"#));
        assert!(quoted_text.contains("Escaped 'single' quote"));
    }

    #[tokio::test]
    async fn test_mcp_ingest_docx_and_pptx_files() {
        use std::io::Write;
        let server = McpServer::new();
        server
            .handle_request("initialize", json!({}))
            .await
            .unwrap();

        let temp_dir = tempfile::tempdir().unwrap();

        // 1. Create and ingest .docx file
        let docx_path = temp_dir.path().join("document.docx");
        {
            let file = std::fs::File::create(&docx_path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            zip.start_file("word/document.xml", options).unwrap();
            let docx_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        <w:p><w:r><w:t>First paragraph of executive review.</w:t></w:r></w:p>
        <w:p><w:r><w:t>Second paragraph detailing margin expansion.</w:t></w:r></w:p>
    </w:body>
</w:document>"#;
            zip.write_all(docx_xml.as_bytes()).unwrap();
            zip.finish().unwrap();
        }

        let docx_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": docx_path.to_str().unwrap(),
                        "table_name": "exec_docx"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            docx_res.get("isError").is_none(),
            "docx ingest failed: {docx_res}"
        );

        let q_docx = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT content, format FROM scratch.exec_docx"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_docx.get("isError").is_none());
        let q_docx_text = q_docx["content"][0]["text"].as_str().unwrap();
        assert!(q_docx_text.contains("executive review"));

        // 2. Create and ingest .pptx file
        let pptx_path = temp_dir.path().join("slides.pptx");
        {
            let file = std::fs::File::create(&pptx_path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            zip.start_file("ppt/slides/slide1.xml", options).unwrap();
            let slide1_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide 1: Q3 Strategic Update</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"#;
            zip.write_all(slide1_xml.as_bytes()).unwrap();

            zip.start_file("ppt/slides/slide2.xml", options).unwrap();
            let slide2_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide 2: Key Milestones Delivered</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"#;
            zip.write_all(slide2_xml.as_bytes()).unwrap();
            zip.finish().unwrap();
        }

        let pptx_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": pptx_path.to_str().unwrap(),
                        "table_name": "deck_pptx"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            pptx_res.get("isError").is_none(),
            "pptx ingest failed: {pptx_res}"
        );

        let q_pptx = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT slide_index, content, format FROM scratch.deck_pptx ORDER BY slide_index"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_pptx.get("isError").is_none());
        let q_pptx_text = q_pptx["content"][0]["text"].as_str().unwrap();
        assert!(q_pptx_text.contains("Q3 Strategic Update"));
        assert!(q_pptx_text.contains("Key Milestones"));
    }

    #[tokio::test]
    async fn test_mcp_ingest_xml_and_zip() {
        use std::io::Write;
        let server = McpServer::new();
        server
            .handle_request("initialize", json!({}))
            .await
            .unwrap();

        let temp_dir = tempfile::tempdir().unwrap();

        // 1. Structured XML ingestion
        let xml_path = temp_dir.path().join("inventory.xml");
        let xml_content = r#"<?xml version="1.0" encoding="UTF-8"?>
<inventory>
    <item>
        <sku>DB-101</sku>
        <name>BenoStreamDB Server</name>
        <qty>50</qty>
    </item>
    <item>
        <sku>DB-102</sku>
        <name>Vector Acceleration Kit</name>
        <qty>25</qty>
    </item>
</inventory>"#;
        std::fs::write(&xml_path, xml_content).unwrap();

        let xml_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": xml_path.to_str().unwrap(),
                        "table_name": "warehouse_xml"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            xml_res.get("isError").is_none(),
            "xml ingest failed: {xml_res}"
        );

        let q_xml = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT sku, name, qty FROM scratch.warehouse_xml ORDER BY sku"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_xml.get("isError").is_none());
        let q_xml_text = q_xml["content"][0]["text"].as_str().unwrap();
        assert!(q_xml_text.contains("DB-101") && q_xml_text.contains("BenoStreamDB Server"));

        // 2. ZIP archive ingestion
        let zip_path = temp_dir.path().join("bundle.zip");
        {
            let file = std::fs::File::create(&zip_path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();

            zip.start_file("notes/readme.txt", options).unwrap();
            zip.write_all(b"Welcome to the BenoStream archive bundle.\n")
                .unwrap();

            zip.start_file("config/manifest.json", options).unwrap();
            zip.write_all(b"{\"version\": \"1.0.0\", \"target\": \"lakehouse\"}\n")
                .unwrap();

            zip.finish().unwrap();
        }

        let zip_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": zip_path.to_str().unwrap(),
                        "table_name": "bundle_zip"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            zip_res.get("isError").is_none(),
            "zip ingest failed: {zip_res}"
        );

        let q_zip = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT entry_name, archive_path FROM scratch.bundle_zip ORDER BY entry_name"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_zip.get("isError").is_none());
        let q_zip_text = q_zip["content"][0]["text"].as_str().unwrap();
        assert!(
            q_zip_text.contains("notes/readme.txt") || q_zip_text.contains("config/manifest.json")
        );
    }

    #[tokio::test]
    async fn test_mcp_configurable_schema_permissions() {
        let temp_wh = tempfile::tempdir().unwrap();
        let wh_path = temp_wh.path().to_str().unwrap().to_string();

        // 1. Without policy in warehouse: conservative lockdown for untrusted connectors
        // Even if client requests readwrite on analytics, without policy it is denied
        let unprov_server = McpServer::with_extension(McpExtension {
            warehouse: Some(wh_path.clone()),
            permissions: Some(SchemaPermissions {
                readwrite_schemas: vec!["scratch".to_string(), "analytics".to_string()],
                readonly_schemas: vec!["default".to_string()],
                default_write_schema: "scratch".to_string(),
            }),
            ..Default::default()
        });
        let unprov_write = unprov_server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "analytics.kpis",
                        "schema_ddl": "metric VARCHAR, val DOUBLE"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            unprov_write["isError"], true,
            "expected write rejection under conservative lockdown when warehouse policy is missing"
        );

        // 2. Authoritative catalog policy stamped by core product:
        // Core product grants write access to analytics, locks finance_raw read-only
        let mut policy = benostreamdb::core::catalog::CatalogPolicy::new();
        policy.readwrite_schemas = Some(vec!["scratch".to_string(), "analytics".to_string()]);
        policy.readonly_schemas = Some(vec!["default".to_string(), "finance_raw".to_string()]);
        benostreamdb::core::catalog::set_catalog_policy(&wh_path, &policy).unwrap();

        let server = McpServer::with_extension(McpExtension {
            warehouse: Some(wh_path.clone()),
            ..Default::default()
        });

        // 3. Create table in analytics (allowed by authoritative catalog policy)
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "analytics.kpis",
                        "schema_ddl": "metric VARCHAR, val DOUBLE"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            res.get("isError").is_none(),
            "expected successful creation in analytics: {res}"
        );

        // 2. Ingest records into analytics.kpis
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_records",
                    "arguments": {
                        "table_name": "analytics.kpis",
                        "records": [
                            {"metric": "mrr", "val": 125000.0},
                            {"metric": "churn", "val": 0.02}
                        ]
                    }
                }),
            )
            .await
            .unwrap();
        assert!(
            res.get("isError").is_none(),
            "expected successful ingest into analytics: {res}"
        );

        // 3. Query analytics.kpis
        let q_res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "SELECT metric, val FROM analytics.kpis WHERE metric = 'mrr'"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(q_res.get("isError").is_none());
        assert!(q_res["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("125000"));

        // 4. Writes to locked read-only schemas are rejected
        let bad_write = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "finance_raw.txns",
                        "schema_ddl": "id INT"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_write["isError"], true,
            "expected write rejection for locked finance_raw schema"
        );

        let bad_sql = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "INSERT INTO default.orders VALUES (1, 2)"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_sql["isError"], true,
            "expected SQL write rejection for locked default schema"
        );

        // 5. Even if a user attempts to configure default in readwrite_schemas, immutable system schemas block it
        let bypass_config = McpServerConfig {
            readwrite_schemas: Some(vec!["default".to_string(), "scratch".to_string()]),
            readonly_schemas: Some(vec![]),
            ..Default::default()
        };
        let bypass_server = McpServer::with_config(bypass_config);
        let bypass_res = bypass_server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": {
                        "table_name": "default.hacked",
                        "schema_ddl": "id INT"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bypass_res["isError"], true,
            "expected write rejection because default is an immutable system schema"
        );
    }

    #[tokio::test]
    async fn test_mcp_warehouse_persistence_across_sessions() {
        let temp_wh = tempfile::tempdir().unwrap();
        let wh_path = temp_wh.path().to_str().unwrap().to_string();

        // Session 1: Create a table and ingest records
        {
            let server1 = McpServer::with_extension(McpExtension {
                warehouse: Some(wh_path.clone()),
                ..Default::default()
            });

            let create_res = server1
                .handle_request(
                    "tools/call",
                    json!({
                        "name": "create_scratch_table",
                        "arguments": {
                            "table_name": "scratch.session_persist",
                            "schema_ddl": "id INT, note VARCHAR"
                        }
                    }),
                )
                .await
                .unwrap();
            assert!(
                create_res.get("isError").is_none(),
                "failed creating table: {create_res}"
            );

            let ingest_res = server1
                .handle_request(
                    "tools/call",
                    json!({
                        "name": "ingest_records",
                        "arguments": {
                            "table_name": "scratch.session_persist",
                            "records": [
                                {"id": 1001, "note": "survives restart"}
                            ]
                        }
                    }),
                )
                .await
                .unwrap();
            assert!(
                ingest_res.get("isError").is_none(),
                "failed ingesting: {ingest_res}"
            );
        }

        // Session 2: Fresh server pointing at the same warehouse path
        {
            let server2 = McpServer::with_extension(McpExtension {
                warehouse: Some(wh_path.clone()),
                ..Default::default()
            });
            server2.load_warehouse_tables().await;

            let query_res = server2
                .handle_request(
                    "tools/call",
                    json!({
                        "name": "execute_sql",
                        "arguments": {
                            "query": "SELECT id, note FROM scratch.session_persist WHERE id = 1001"
                        }
                    }),
                )
                .await
                .unwrap();
            assert!(
                query_res.get("isError").is_none(),
                "expected successful query in session 2: {query_res}"
            );
            let text = query_res["content"][0]["text"].as_str().unwrap();
            assert!(
                text.contains("survives restart"),
                "expected persisted content across sessions, got: {text}"
            );
        }
    }

    #[tokio::test]
    async fn test_mcp_json_configuration_and_initialization_options() {
        // 1. Deserialization from JSON string (array or comma-separated schemas)
        let json_config = r#"{
            "warehouse": "/tmp/custom_wh",
            "readwrite_schemas": ["scratch", "sandbox"],
            "readonly_schemas": "default, prod_archive",
            "default_write_schema": "sandbox"
        }"#;
        let config: McpServerConfig = serde_json::from_str(json_config).unwrap();
        assert_eq!(config.warehouse.as_deref(), Some("/tmp/custom_wh"));
        let perms = config.to_permissions();
        assert_eq!(perms.default_write_schema, "sandbox");
        assert!(perms.is_schema_writable("sandbox"));
        assert!(perms.is_schema_writable("scratch"));
        assert!(!perms.is_schema_writable("prod_archive"));

        // 2. CLI Argument parsing
        let args = vec![
            "--warehouse".to_string(),
            "/tmp/cli_wh".to_string(),
            "--readwrite-schemas".to_string(),
            "scratch,dev".to_string(),
            "--readonly-schemas".to_string(),
            "default,finance".to_string(),
        ];
        let cli_cfg = McpServerConfig::from_args(&args).unwrap();
        assert_eq!(cli_cfg.warehouse.as_deref(), Some("/tmp/cli_wh"));
        let cli_perms = cli_cfg.to_permissions();
        assert!(cli_perms.is_schema_writable("dev"));
        assert!(!cli_perms.is_schema_writable("finance"));

        // 3. Dynamic initializationOptions handshake with catalog policy
        let temp_wh = tempfile::tempdir().unwrap();
        let wh_path = temp_wh.path().to_str().unwrap().to_string();
        let mut policy = benostreamdb::core::catalog::CatalogPolicy::new();
        policy.readwrite_schemas = Some(vec!["scratch".to_string(), "agent_output".to_string()]);
        policy.readonly_schemas = Some(vec!["default".to_string(), "legacy".to_string()]);
        benostreamdb::core::catalog::set_catalog_policy(&wh_path, &policy).unwrap();

        let server = McpServer::with_extension(McpExtension {
            warehouse: Some(wh_path),
            ..Default::default()
        });
        let init_res = server
            .handle_request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "Antigravity", "version": "1.0.0" },
                    "initializationOptions": {
                        "readwrite_schemas": ["scratch", "agent_output"],
                        "readonly_schemas": ["default", "legacy"],
                        "default_write_schema": "agent_output"
                    }
                }),
            )
            .await
            .unwrap();
        assert!(init_res.get("serverInfo").is_some());
        let updated_perms = server.permissions();
        assert_eq!(updated_perms.default_write_schema, "agent_output");
        assert!(updated_perms.is_schema_writable("agent_output"));
        assert!(!updated_perms.is_schema_writable("legacy"));
    }

    #[tokio::test]
    async fn test_mcp_storage_lockdown_blocks_remote_uris_and_locations() {
        let server = McpServer::new();

        // 1. validate_local_path rejects remote object store schemes (MinIO, S3, RustFS, Iceberg REST, GCS)
        assert!(validate_local_path("s3://bucket/path", "test").is_err());
        assert!(validate_local_path("minio://host:9000/bucket", "test").is_err());
        assert!(validate_local_path("rustfs://cluster/data", "test").is_err());
        assert!(validate_local_path("http://localhost:9000/s3", "test").is_err());
        assert!(validate_local_path("https://storage.googleapis.com/bucket", "test").is_err());
        assert!(validate_local_path("gcs://bucket/path", "test").is_err());
        assert!(validate_local_path("iceberg://catalog.corp:8181", "test").is_err());

        // Local paths are allowed
        assert!(validate_local_path("/tmp/local_warehouse", "test").is_ok());
        assert!(validate_local_path("./relative_dir", "test").is_ok());
        assert!(validate_local_path("file:///tmp/local", "test").is_ok());

        // 2. Reject LOCATION in CREATE TABLE
        let bad_location = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "CREATE TABLE scratch.s3_table (id INT) LOCATION 's3://my-bucket/data'"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_location["isError"], true,
            "expected LOCATION rejection: {bad_location}"
        );

        // 3. Reject CREATE EXTERNAL TABLE
        let bad_ext = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "CREATE EXTERNAL TABLE scratch.ext_table STORED AS PARQUET LOCATION '/tmp/test'"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_ext["isError"], true,
            "expected CREATE EXTERNAL TABLE rejection: {bad_ext}"
        );

        // 4. Reject ATTACH / MOUNT CATALOG
        let bad_cat = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "execute_sql",
                    "arguments": {
                        "query": "ATTACH CATALOG iceberg_prod"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_cat["isError"], true,
            "expected CATALOG ATTACH rejection: {bad_cat}"
        );

        // 5. Reject export_table to remote object stores
        let bad_export = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "export_table",
                    "arguments": {
                        "table_name": "scratch.my_table",
                        "export_path": "s3://remote-bucket/export.parquet"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_export["isError"], true,
            "expected export_table remote rejection: {bad_export}"
        );

        // 6. Reject ingest_file with remote object store URL
        let bad_ingest = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "ingest_file",
                    "arguments": {
                        "file_path": "https://raw.githubusercontent.com/data.csv"
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(
            bad_ingest["isError"], true,
            "expected ingest_file remote URL rejection: {bad_ingest}"
        );
    }
}
