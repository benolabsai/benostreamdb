// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! SQL DDL / maintenance interception.
//!
//! DataFusion has no logical plan for `CREATE INDEX`, `OPTIMIZE`, `VACUUM`,
//! `DELETE` (for custom providers), or `ALTER TABLE ... EXECUTE`. Following the
//! [`merge_into`](crate::core::sql::merge_into) pattern, we parse the statement
//! with DataFusion's re-exported `sqlparser` and dispatch to the core `Table`
//! API *before* planning.
//!
//! The parser uses [`GenericDialect`] rather than the session's PostgreSQL
//! dialect so that `OPTIMIZE TABLE` (ClickHouse/Generic only) is recognized.
//!
//! Statements that sqlparser does not model (`COMPACT`, `MSCK REPAIR TABLE`,
//! `ALTER TABLE ... EXECUTE`, `SET benostream.*`, `SHOW benostream.*`,
//! `DELETE FROM`) are handled by a small hand-rolled pre-parser.

use anyhow::{bail, Context, Result};
use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::sql::parser::{DFParserBuilder, Statement as DFStatement};
use futures::StreamExt;
use datafusion::sql::sqlparser::ast::{
    AlterColumnOperation, AlterTableOperation, ColumnDef, ColumnOption, CreateIndex, CreateTable,
    Expr, IndexType, ObjectType, SchemaName, SqlOption, Statement, TableConstraint, Value,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::TableReference;
use std::sync::Arc;

use crate::core::manifest::IndexAlgorithm;
use crate::core::sql::session::BenoStreamSession;
use crate::core::sql::BenoStreamTableProvider;
use crate::core::table::Table;

/// Collapse runs of whitespace outside quoted strings into single spaces, so
/// the hand-rolled prefix matching is robust to extra spaces, tabs, and
/// newlines (e.g. `COMPACT\tt`, `ALTER  TABLE  t  EXECUTE  x`).
fn collapse_ws(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_ws = false;
    for c in sql.chars() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                out.push(c);
                prev_ws = false;
            }
            '"' if !in_single => {
                in_double = !in_double;
                out.push(c);
                prev_ws = false;
            }
            c if c.is_whitespace() && !in_single && !in_double => {
                if !prev_ws {
                    out.push(' ');
                    prev_ws = true;
                }
            }
            c => {
                out.push(c);
                prev_ws = false;
            }
        }
    }
    out.trim().to_string()
}

/// How this module classifies a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handled {
    /// Not ours; let the normal DataFusion pipeline run.
    No,
    /// Ours, with no result set (DDL / maintenance).
    Ddl,
    /// Ours, with a result set (e.g. `SHOW benostream.*`).
    Query,
}

/// The DDL / maintenance statement kinds the engine intercepts and executes
/// itself (before DataFusion planning). Single source of truth for the
/// connector surface-parity test.
pub fn handled_ddl_statements() -> Vec<String> {
    [
        "CREATE DATABASE",
        "CREATE SCHEMA",
        "CREATE TABLE",
        "CREATE INDEX",
        "DROP TABLE",
        "DROP SCHEMA",
        "DROP DATABASE",
        "ALTER TABLE",
        "OPTIMIZE TABLE",
        "TRUNCATE TABLE",
        "VACUUM",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// The `ALTER TABLE ... EXECUTE <action>` procedure names the engine supports.
/// Single source of truth for the connector surface-parity test.
pub fn table_action_names() -> Vec<String> {
    [
        "remove_orphan_files",
        "recover_indexes",
        "rollback",
        "rollback_to_snapshot",
        "preload_indexes",
        "verify_integrity",
        "checkpoint",
        "rewrite_data_files",
        "compact",
        "expire_snapshots",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Classify `sql` for the interception layer.
pub fn classify(sql: &str) -> Handled {
    let trimmed = collapse_ws(sql.trim().trim_end_matches(';'));
    let upper = trimmed.to_ascii_uppercase();

    if upper.starts_with("SHOW ") {
        let rest = trimmed
            .split_once(' ')
            .map(|x| x.1)
            .unwrap_or("")
            .trim_start();
        return if rest.to_ascii_lowercase().starts_with("benostream.") {
            Handled::Query
        } else {
            Handled::No
        };
    }
    if upper.starts_with("SET ") {
        let rest = trimmed
            .split_once(' ')
            .map(|x| x.1)
            .unwrap_or("")
            .trim_start();
        return if rest.to_ascii_lowercase().starts_with("benostream.") {
            Handled::Ddl
        } else {
            Handled::No
        };
    }
    if upper.starts_with("COMPACT ")
        || upper.starts_with("MSCK REPAIR TABLE ")
        || upper.starts_with("DELETE FROM ")
    {
        return Handled::Ddl;
    }
    if upper.starts_with("ALTER TABLE ") && upper.contains(" EXECUTE ") {
        return Handled::Ddl;
    }
    match parse_single(&trimmed) {
        Some(stmt) => match stmt {
            Statement::CreateDatabase { .. }
            | Statement::CreateSchema { .. }
            | Statement::CreateTable(_)
            | Statement::CreateIndex(_)
            | Statement::AlterTable { .. }
            | Statement::OptimizeTable { .. }
            | Statement::Truncate { .. }
            | Statement::Vacuum(_) => Handled::Ddl,
            Statement::Drop { object_type, .. } => {
                if matches!(
                    object_type,
                    ObjectType::Table | ObjectType::Schema | ObjectType::Database
                ) {
                    Handled::Ddl
                } else {
                    Handled::No
                }
            }
            _ => Handled::No,
        },
        None => Handled::No,
    }
}

/// Returns `true` if this module owns `sql`.
pub fn is_handled(sql: &str) -> bool {
    classify(sql) != Handled::No
}

/// The result schema for a handled query statement, if it has one.
pub fn result_schema(sql: &str) -> Option<SchemaRef> {
    match classify(sql) {
        Handled::Query => Some(Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            true,
        )]))),
        _ => None,
    }
}

/// Parse `sql` and, if it is a statement this module owns, execute it and return
/// the result batch. Returns `Ok(None)` for anything else so the normal
/// DataFusion pipeline runs.
pub async fn try_parse_and_execute(
    session: &BenoStreamSession,
    sql: &str,
) -> Result<Option<RecordBatch>> {
    // 1. Statements sqlparser does not model.
    if let Some(batch) = try_hand_rolled(session, sql).await? {
        return Ok(Some(batch));
    }

    // 2. Statements sqlparser models but DataFusion cannot plan.
    let stmt = match parse_single(sql) {
        Some(s) => s,
        None => return Ok(None),
    };

    match &stmt {
        Statement::CreateDatabase {
            db_name,
            if_not_exists,
            ..
        } => {
            create_database(session, &db_name.to_string(), *if_not_exists).await?;
            Ok(Some(empty_batch()))
        }
        Statement::CreateSchema {
            schema_name,
            if_not_exists,
            ..
        } => {
            let name = schema_name_to_string(schema_name)?;
            create_schema(session, &name, *if_not_exists).await?;
            Ok(Some(empty_batch()))
        }
        Statement::CreateTable(ct) => {
            create_table(session, ct, sql).await?;
            Ok(Some(empty_batch()))
        }
        Statement::CreateIndex(ci) => {
            create_index(session, ci).await?;
            Ok(Some(empty_batch()))
        }
        Statement::Drop {
            object_type,
            names,
            if_exists,
            ..
        } => {
            drop_objects(session, object_type, names, *if_exists).await?;
            Ok(Some(empty_batch()))
        }
        Statement::AlterTable {
            name, operations, ..
        } => {
            alter_table(session, &name.to_string(), operations).await?;
            Ok(Some(empty_batch()))
        }
        Statement::OptimizeTable { name, .. } => {
            let table = resolve_table(session, &name.to_string()).await?;
            table.rewrite_data_files_async(None).await?;
            Ok(Some(empty_batch()))
        }
        Statement::Truncate { table_names, .. } => {
            for target in table_names {
                let table = resolve_table(session, &target.name.to_string()).await?;
                table.truncate_async().await?;
            }
            Ok(Some(empty_batch()))
        }
        Statement::Vacuum(v) => {
            let name = v
                .table_name
                .as_ref()
                .map(|n| n.to_string())
                .ok_or_else(|| anyhow::anyhow!("VACUUM requires a table name"))?;
            let table = resolve_table(session, &name).await?;
            table.vacuum_async(1).await?;
            Ok(Some(empty_batch()))
        }
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Hand-rolled statements
// ---------------------------------------------------------------------------

async fn try_hand_rolled(session: &BenoStreamSession, sql: &str) -> Result<Option<RecordBatch>> {
    let trimmed = collapse_ws(sql.trim().trim_end_matches(';'));
    let upper = trimmed.to_ascii_uppercase();

    // SET benostream.<key> = <value>
    if upper.starts_with("SET ") {
        if let Some(rest) = trimmed.get(4..) {
            if rest
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("benostream.")
            {
                return set_session_setting(session, rest.trim()).await.map(Some);
            }
        }
        return Ok(None);
    }

    // SHOW benostream.<key>
    if upper.starts_with("SHOW ") {
        if let Some(rest) = trimmed.get(5..) {
            if rest
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("benostream.")
            {
                return show_session_setting(session, rest.trim()).await.map(Some);
            }
        }
        return Ok(None);
    }

    // COMPACT <table>
    if upper.starts_with("COMPACT ") {
        let name = trimmed[8..].trim();
        let table = resolve_table(session, name).await?;
        table.rewrite_data_files_async(None).await?;
        return Ok(Some(empty_batch()));
    }

    // MSCK REPAIR TABLE <table>
    if upper.starts_with("MSCK REPAIR TABLE ") {
        let name = trimmed[18..].trim();
        let table = resolve_table(session, name).await?;
        table.recover_indexes_async().await?;
        return Ok(Some(empty_batch()));
    }

    // ALTER TABLE <table> EXECUTE <action> [args]
    if upper.starts_with("ALTER TABLE ") {
        if let Some(idx) = upper.find(" EXECUTE ") {
            let table_part = trimmed[12..idx].trim();
            let action_part = trimmed[idx + 9..].trim();
            return execute_action(session, table_part, action_part)
                .await
                .map(Some);
        }
        return Ok(None);
    }

    // DELETE FROM <table> [WHERE <filter>]
    if upper.starts_with("DELETE FROM ") {
        return delete_from(session, &trimmed).await.map(Some);
    }

    Ok(None)
}

async fn set_session_setting(session: &BenoStreamSession, rest: &str) -> Result<RecordBatch> {
    let (key, value) = rest
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("SET requires '<key> = <value>'"))?;
    let key = key.trim();
    let value = value.trim().trim_matches('\'').trim_matches('"');
    session.set_setting(key, value).await?;
    Ok(empty_batch())
}

async fn show_session_setting(session: &BenoStreamSession, rest: &str) -> Result<RecordBatch> {
    let key = rest.trim();
    let value = session.get_setting(key).await.unwrap_or_default();
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, true)]));
    let array: ArrayRef = Arc::new(StringArray::from(vec![value]));
    Ok(RecordBatch::try_new(schema, vec![array])?)
}

async fn delete_from(session: &BenoStreamSession, sql: &str) -> Result<RecordBatch> {
    let rest = sql[12..].trim();
    let (table_name, filter) = match rest.to_ascii_uppercase().find(" WHERE ") {
        Some(idx) => (&rest[..idx], rest[idx + 7..].trim()),
        None => (rest, "TRUE"),
    };
    let table = resolve_table(session, table_name.trim()).await?;
    table.delete_async(filter).await?;
    Ok(empty_batch())
}

/// Dispatch `ALTER TABLE <t> EXECUTE <action> [args]`.
async fn execute_action(
    session: &BenoStreamSession,
    table_name: &str,
    action: &str,
) -> Result<RecordBatch> {
    let table = resolve_table(session, table_name).await?;
    let (name, args) = match action.find('(') {
        Some(idx) => {
            let name = action[..idx].trim();
            let args = action[idx..]
                .trim()
                .trim_start_matches('(')
                .trim_end_matches(')');
            (name, args)
        }
        None => (action.trim(), ""),
    };
    let name_lower = name.to_ascii_lowercase();
    match name_lower.as_str() {
        "remove_orphan_files" => {
            let ms = parse_named_i64(args, "older_than_ms").unwrap_or(0);
            table.remove_orphan_files_async(ms).await?;
        }
        "recover_indexes" => {
            table.recover_indexes_async().await?;
        }
        "rollback" | "rollback_to_snapshot" => {
            let id = parse_named_i64(args, "snapshot_id")
                .or_else(|| args.trim().parse::<i64>().ok())
                .ok_or_else(|| anyhow::anyhow!("rollback requires a snapshot id"))?;
            table.rollback_to_snapshot(id).await?;
        }
        "preload_indexes" => {
            let opts = crate::core::table::PreloadOptions::from_env();
            table.preload_indexes_async(opts).await?;
        }
        "verify_integrity" => {
            table.verify_integrity_async().await?;
        }
        "checkpoint" => {
            table.checkpoint_async().await?;
        }
        "rewrite_data_files" | "compact" => {
            table.rewrite_data_files_async(None).await?;
        }
        "expire_snapshots" => {
            let retention = parse_named_i64(args, "retention").unwrap_or(1).max(1) as usize;
            table.vacuum_async(retention).await?;
        }
        other => bail!("unsupported EXECUTE action: {}", other),
    }
    Ok(empty_batch())
}

/// Parse `key => value` or `key = value` from an action argument list.
fn parse_named_i64(args: &str, key: &str) -> Option<i64> {
    let lower = args.to_ascii_lowercase();
    let key_lower = key.to_ascii_lowercase();
    let idx = lower.find(&key_lower)?;
    let after = &args[idx + key.len()..];
    let after = after
        .trim_start()
        .trim_start_matches("=>")
        .trim_start_matches('=');
    let value: String = after
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    value.parse::<i64>().ok()
}

// ---------------------------------------------------------------------------
// Catalog / namespace
// ---------------------------------------------------------------------------

async fn create_database(
    session: &BenoStreamSession,
    name: &str,
    if_not_exists: bool,
) -> Result<()> {
    let ctx = session.get_ctx();
    if ctx.catalog(name).is_some() {
        if if_not_exists {
            return Ok(());
        }
        bail!("database '{}' already exists", name);
    }
    ctx.register_catalog(
        name,
        Arc::new(datafusion::catalog::MemoryCatalogProvider::new()),
    );
    Ok(())
}

async fn create_schema(session: &BenoStreamSession, name: &str, if_not_exists: bool) -> Result<()> {
    let (db, schema) = split_db_schema(session, name);
    let ctx = session.get_ctx();
    let catalog = ctx
        .catalog(&db)
        .ok_or_else(|| anyhow::anyhow!("database '{}' does not exist", db))?;
    if catalog.schema(&schema).is_some() {
        if if_not_exists {
            return Ok(());
        }
        bail!("schema '{}.{}' already exists", db, schema);
    }
    catalog.register_schema(
        &schema,
        Arc::new(datafusion::catalog::memory::MemorySchemaProvider::new()),
    )?;
    // Mirror into the external catalog, if one is bound to this database.
    if let Some(ext) = session.catalog_for(&db).await {
        ext.create_namespace(&schema).await?;
    }
    Ok(())
}

async fn drop_objects(
    session: &BenoStreamSession,
    object_type: &ObjectType,
    names: &[datafusion::sql::sqlparser::ast::ObjectName],
    if_exists: bool,
) -> Result<()> {
    let ctx = session.get_ctx();
    for name in names {
        let full = name.to_string();
        match object_type {
            ObjectType::Table => {
                let (db, schema, table) = split_three(session, &full);
                let reference = TableReference::full(db.clone(), schema.clone(), table.clone());
                // `IF EXISTS` must tolerate an unresolvable catalog path (fresh
                // session, schema never registered) — DataFusion errors instead
                // of returning None.
                let removed = match ctx.deregister_table(reference) {
                    Ok(removed) => removed,
                    Err(e) => {
                        if if_exists {
                            tracing::debug!("DROP TABLE IF EXISTS '{}': {}", full, e);
                            None
                        } else {
                            return Err(e).with_context(|| format!("dropping table '{}'", full));
                        }
                    }
                };
                if removed.is_none() && !if_exists {
                    bail!("table '{}' does not exist", full);
                }
                // Drop from the external catalog too.
                if let Some(ext) = session.catalog_for(&db).await {
                    let _ = ext.drop_table(&schema, &table).await;
                }
                // Delete the underlying data so a re-created table starts from
                // an empty location (mirrors the FFI `dropTable` used by the
                // Spark/Trino connectors; dbt's drop-then-create cycle depends
                // on this).
                if let Some(w) = session.warehouse() {
                    let uri = format!("{}/{}/{}", w.trim_end_matches('/'), schema, table);
                    match crate::core::storage::create_object_store(&uri) {
                        Ok(store) => {
                            let prefix = object_store::path::Path::from("");
                            let mut stream = store.list(Some(&prefix));
                            let mut failed = false;
                            while let Some(item) = stream.next().await {
                                match item {
                                    Ok(meta) => {
                                        if store.delete(&meta.location).await.is_err() {
                                            failed = true;
                                            break;
                                        }
                                    }
                                    Err(_) => {
                                        failed = true;
                                        break;
                                    }
                                }
                            }
                            if !failed {
                                let manager =
                                    crate::core::manifest::ManifestManager::new(store, "", &uri);
                                manager.invalidate_caches().await;
                            }
                        }
                        Err(e) => tracing::warn!("DROP TABLE '{}': could not open store: {}", uri, e),
                    }
                }
            }
            ObjectType::Schema => {
                let (db, schema) = split_db_schema(session, &full);
                if let Some(catalog) = ctx.catalog(&db) {
                    let _ = catalog.deregister_schema(&schema, true);
                }
            }
            ObjectType::Database => {
                // DataFusion's catalog list exposes no deregistration API, so
                // DROP DATABASE is a best-effort no-op.
                tracing::warn!(
                    "DROP DATABASE '{}' is a no-op (DataFusion has no catalog deregistration)",
                    full
                );
            }
            other => bail!("DROP {:?} is not supported", other),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

async fn create_table(session: &BenoStreamSession, ct: &CreateTable, raw_sql: &str) -> Result<()> {
    let full = ct.name.to_string();
    let (db, schema, table) = split_three(session, &full);

    // CREATE TABLE AS SELECT (dbt's `table` materialization): derive the
    // target schema from the source query, create the table, then populate it
    // with `INSERT INTO <name> <query>` after registration. The query text is
    // sliced from the original statement because sqlparser's AST re-rendering
    // mangles parenthesized WITH queries (`as (with a as (...) select ...)`)
    // into invalid SQL.
    let ctas_sql: Option<String> = if ct.query.is_some() {
        extract_ctas_query(raw_sql)
    } else {
        None
    };
    let arrow_schema: SchemaRef = match &ctas_sql {
        Some(q) => {
            let df = session
                .get_ctx()
                .sql(q)
                .await
                .with_context(|| "CREATE TABLE AS SELECT: planning source query")?;
            Arc::new(df.schema().as_arrow().clone())
        }
        None => columns_to_arrow_schema(&ct.columns)?,
    };

    // `GenericDialect` does not populate `CreateTable::location`, so fall back
    // to scanning the raw SQL for a `LOCATION '...'` clause.
    let location = ct
        .location
        .clone()
        .or_else(|| extract_location(raw_sql))
        .or_else(|| {
            session
                .warehouse()
                .map(|w| format!("{}/{}/{}", w.trim_end_matches('/'), schema, table))
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no location for '{}': set BSDB_WAREHOUSE or use CREATE TABLE ... LOCATION",
                full
            )
        })?;

    ensure_catalog_schema(session, &db, &schema).await?;

    let ext = session.catalog_for(&db).await;
    if let Some(cat) = &ext {
        cat.create_table(&schema, &table, arrow_schema.clone(), Some(&location))
            .await?;
    }

    let created = if let Some(cat) = ext {
        Table::create_with_catalog_async(location, arrow_schema, cat, &schema, &table).await?
    } else {
        Table::create_async(location, arrow_schema).await?
    };

    // Apply WITH options.
    apply_table_options(&created, ct)?;

    let provider = Arc::new(BenoStreamTableProvider::new(Arc::new(created)));
    session
        .get_ctx()
        .register_table(TableReference::full(db, schema, table), provider)?;

    if let Some(q) = ctas_sql {
        // Populate the new table. Executed through the DataFusion context
        // directly (not `session.sql`) to keep the async call graph acyclic;
        // the session's custom INSERT plan node is registered on the context,
        // so the write path is identical.
        let insert_sql = format!("INSERT INTO {} {}", full, q);
        session
            .get_ctx()
            .sql(&insert_sql)
            .await
            .with_context(|| "CREATE TABLE AS SELECT: populating new table")?
            .collect()
            .await
            .with_context(|| "CREATE TABLE AS SELECT: executing insert")?;
    }
    Ok(())
}

/// Slice the source query out of a `CREATE TABLE ... AS <query>` statement.
/// Finds the top-level (quote-aware) `AS` keyword and strips one balanced
/// layer of wrapping parentheses, if present.
fn extract_ctas_query(raw_sql: &str) -> Option<String> {
    // Byte-level scan with an ASCII-lowercased copy (same byte length), so the
    // returned offset is a valid byte index into `raw_sql` even with multibyte
    // UTF-8 earlier in the statement.
    let bytes = raw_sql.as_bytes();
    let lower = raw_sql.to_ascii_lowercase();
    let lb = lower.as_bytes();
    let is_ws = |b: u8| (b as char).is_ascii_whitespace();
    let mut in_single = false;
    let mut in_double = false;
    let mut i = 0usize;
    let mut as_end: Option<usize> = None;
    while i + 2 < bytes.len() {
        let c = bytes[i];
        if in_single {
            if c == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if c == b'"' {
                in_double = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'a' | b'A' => {
                // Top-level `AS` keyword: whitespace-delimited on both sides
                // (dbt renders it on its own line).
                if lb[i + 1] == b's' && i > 0 && is_ws(bytes[i - 1]) && is_ws(bytes[i + 2]) {
                    as_end = Some(i + 2);
                    break;
                }
                i += 1;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    let mut query = raw_sql[as_end?..].trim().to_string();
    // Strip one balanced layer of outer parentheses: `(with ... select ...)`.
    if query.starts_with('(') && query.ends_with(')') {
        let inner = &query[1..query.len() - 1];
        let mut depth = 0i32;
        let mut balanced = true;
        for c in inner.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        balanced = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if balanced && depth == 0 {
            query = inner.trim().to_string();
        }
    }
    if query.is_empty() {
        None
    } else {
        Some(query)
    }
}

fn apply_table_options(table: &Table, ct: &CreateTable) -> Result<()> {
    use datafusion::sql::sqlparser::ast::CreateTableOptions;
    let options = match &ct.table_options {
        CreateTableOptions::With(opts) | CreateTableOptions::Options(opts) => opts,
        _ => return Ok(()),
    };
    for opt in options {
        let (key, value) = match opt {
            datafusion::sql::sqlparser::ast::SqlOption::KeyValue { key, value } => (
                key.value.to_ascii_lowercase(),
                value.to_string().trim_matches('\'').to_string(),
            ),
            _ => continue,
        };
        match key.as_str() {
            "format_version" => {
                if let Ok(v) = value.parse::<i32>() {
                    table.set_format_version(v);
                }
            }
            "sort_order" => {
                let cols: Vec<&str> = value.split(',').map(|s| s.trim()).collect();
                let ascending = vec![true; cols.len()];
                table.replace_sort_order(&cols, &ascending)?;
            }
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Indexes
// ---------------------------------------------------------------------------

async fn create_index(session: &BenoStreamSession, ci: &CreateIndex) -> Result<()> {
    let table = resolve_table(session, &ci.table_name.to_string()).await?;
    let cols: Vec<String> = ci.columns.iter().map(|c| c.column.to_string()).collect();
    let alg = index_algorithm_from_using(ci.using.as_ref());
    add_index_columns(&table, cols, alg).await
}

async fn add_index_columns(table: &Table, cols: Vec<String>, alg: IndexAlgorithm) -> Result<()> {
    match cols.len() {
        0 => bail!("index requires at least one column"),
        1 => table.add_index(cols[0].clone(), alg).await,
        _ => table.add_composite_index_async(cols).await,
    }
}

/// Map a `CREATE INDEX ... USING <name>` clause to an [`IndexAlgorithm`].
///
/// Uses the shared [`IndexAlgorithm::from_name`] mapping, so every algorithm
/// the engine understands is reachable from DDL (`hnsw`, `hnsw_pq`,
/// `hnsw_tq4`, `hnsw_tq8`, `bm25`, `bloom`, `bitmap`, `composite_bitmap`,
/// `csr_graph`, `json_path`) — previously only `hnsw*`/`bm25` were recognized
/// and everything else silently became a Bitmap. An absent or unknown `USING`
/// defaults to Bitmap (the scalar default).
fn index_algorithm_from_using(using: Option<&IndexType>) -> IndexAlgorithm {
    match using {
        Some(IndexType::Custom(ident)) => {
            IndexAlgorithm::from_name(&ident.value).unwrap_or(IndexAlgorithm::Bitmap)
        }
        _ => IndexAlgorithm::Bitmap,
    }
}

// ---------------------------------------------------------------------------
// ALTER TABLE
// ---------------------------------------------------------------------------

async fn alter_table(
    session: &BenoStreamSession,
    name: &str,
    operations: &[AlterTableOperation],
) -> Result<()> {
    let table = resolve_table(session, name).await?;
    for op in operations {
        match op {
            AlterTableOperation::AddConstraint { constraint, .. } => match constraint {
                TableConstraint::PrimaryKey { columns, .. } => {
                    let cols: Vec<String> = columns.iter().map(|c| c.column.to_string()).collect();
                    table.set_primary_key_async(cols).await?;
                }
                TableConstraint::Index { columns, .. }
                | TableConstraint::Unique { columns, .. } => {
                    let cols: Vec<String> = columns.iter().map(|c| c.column.to_string()).collect();
                    add_index_columns(&table, cols, IndexAlgorithm::Bitmap).await?;
                }
                other => bail!("unsupported ADD constraint: {:?}", other),
            },
            AlterTableOperation::DropPrimaryKey { .. } => {
                table.set_primary_key_async(vec![]).await?;
            }
            AlterTableOperation::DropIndex { name } => {
                table.drop_index(name.value.clone()).await?;
            }
            AlterTableOperation::AddColumn { column_def, .. } => {
                let dt = sql_type_to_arrow(&column_def.data_type)?;
                table.add_column(&column_def.name.value, dt).await?;
            }
            AlterTableOperation::DropColumn { column_names, .. } => {
                for c in column_names {
                    table.drop_column(&c.value).await?;
                }
            }
            AlterTableOperation::RenameColumn {
                old_column_name,
                new_column_name,
            } => {
                table
                    .rename_column(&old_column_name.value, &new_column_name.value)
                    .await?;
            }
            AlterTableOperation::AlterColumn { column_name, op } => match op {
                AlterColumnOperation::SetDataType { data_type, .. } => {
                    let ty = sql_type_to_arrow(data_type)?;
                    table
                        .update_column_type(&column_name.value, &ty.to_string())
                        .await?;
                }
                other => bail!("unsupported ALTER COLUMN operation: {:?}", other),
            },
            AlterTableOperation::SetTblProperties { table_properties } => {
                let mut props = std::collections::HashMap::new();
                for opt in table_properties {
                    if let Some((k, v)) = sql_option_to_pair(opt) {
                        props.insert(k, v);
                    }
                }
                table.set_properties_async(props).await?;
                // Declaring `table_type = 'edge'` (with `src_col`/`dst_col`)
                // materialises the forward/reverse CSR overlays the graph fast
                // paths need. Idempotent and a no-op for non-edge tables.
                table.ensure_edge_indexes_async().await?;
            }
            other => bail!("unsupported ALTER TABLE operation: {:?}", other),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `SET TBLPROPERTIES` option into a `(key, value)` pair.
fn sql_option_to_pair(opt: &SqlOption) -> Option<(String, String)> {
    match opt {
        SqlOption::KeyValue { key, value } => {
            let v = match value {
                Expr::Value(val) => match &val.value {
                    Value::SingleQuotedString(s)
                    | Value::DoubleQuotedString(s) => s.clone(),
                    Value::Number(n, _) => n.clone(),
                    Value::Boolean(b) => b.to_string(),
                    _ => val.to_string(),
                },
                other => other.to_string(),
            };
            Some((key.value.clone(), v))
        }
        SqlOption::Ident(ident) => Some((ident.value.clone(), String::new())),
        _ => None,
    }
}

fn parse_single(sql: &str) -> Option<Statement> {
    let dialect = GenericDialect {};
    let mut parser = DFParserBuilder::new(sql)
        .with_dialect(&dialect)
        .build()
        .ok()?;
    let statements = parser.parse_statements().ok()?;
    if statements.len() != 1 {
        return None;
    }
    match &statements[0] {
        DFStatement::Statement(inner) => Some(inner.as_ref().clone()),
        _ => None,
    }
}

fn empty_batch() -> RecordBatch {
    RecordBatch::new_empty(Arc::new(Schema::empty()))
}

/// Extract a `LOCATION '...'` / `LOCATION "..."` clause from raw SQL.
fn extract_location(sql: &str) -> Option<String> {
    let upper = sql.to_ascii_uppercase();
    let idx = upper.find("LOCATION")?;
    let after = sql[idx + "LOCATION".len()..].trim_start();
    let quote = after.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &after[1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn schema_name_to_string(name: &SchemaName) -> Result<String> {
    match name {
        SchemaName::Simple(n) => Ok(n.to_string()),
        other => bail!("unsupported schema name: {:?}", other),
    }
}

/// Resolve a registered BenoStreamDB table by (possibly qualified) name.
async fn resolve_table(session: &BenoStreamSession, name: &str) -> Result<Arc<Table>> {
    let reference = TableReference::from(name.trim());
    let provider = session
        .get_ctx()
        .table_provider(reference)
        .await
        .with_context(|| format!("table '{}' is not registered", name))?;
    let provider = provider
        .as_any()
        .downcast_ref::<BenoStreamTableProvider>()
        .context("table is not a BenoStreamDB table")?;
    Ok(provider.table.clone())
}

/// Ensure a DataFusion catalog + schema exist so a table can be registered.
async fn ensure_catalog_schema(session: &BenoStreamSession, db: &str, schema: &str) -> Result<()> {
    let ctx = session.get_ctx();
    if ctx.catalog(db).is_none() {
        ctx.register_catalog(
            db,
            Arc::new(datafusion::catalog::MemoryCatalogProvider::new()),
        );
    }
    let catalog = ctx
        .catalog(db)
        .ok_or_else(|| anyhow::anyhow!("catalog '{}' missing", db))?;
    if catalog.schema(schema).is_none() {
        catalog.register_schema(
            schema,
            Arc::new(datafusion::catalog::memory::MemorySchemaProvider::new()),
        )?;
    }
    Ok(())
}

/// The session's configured default catalog name.
fn default_catalog(session: &BenoStreamSession) -> String {
    session
        .get_ctx()
        .state()
        .config()
        .options()
        .catalog
        .default_catalog
        .clone()
}

/// Strip identifier quoting (`"..."` / backticks) from a dotted name part.
/// Engines like dbt render fully-quoted identifiers; keeping the quotes would
/// leak them into warehouse paths and DataFusion catalog lookups.
fn unquote(s: &str) -> String {
    s.trim().trim_matches('"').trim_matches('`').to_string()
}

/// The unified default schema across every connector surface (Spark's V2
/// default namespace, Trino's default schema, the dbt adapter's profile).
pub const DEFAULT_SCHEMA: &str = "default";

/// Split `db.schema.table` into its parts, defaulting to the session's default
/// catalog/schema when fewer parts are given.
fn split_three(session: &BenoStreamSession, name: &str) -> (String, String, String) {
    let parts: Vec<String> = name.split('.').map(unquote).collect();
    let default_catalog = default_catalog(session);
    match parts.as_slice() {
        [t] => (default_catalog, DEFAULT_SCHEMA.to_string(), t.clone()),
        [s, t] => (default_catalog, s.clone(), t.clone()),
        [d, s, t] => (d.clone(), s.clone(), t.clone()),
        _ => (
            default_catalog,
            DEFAULT_SCHEMA.to_string(),
            parts.last().cloned().unwrap_or_default(),
        ),
    }
}

/// Split `db.schema` (or just `schema`) into (database, schema).
fn split_db_schema(session: &BenoStreamSession, name: &str) -> (String, String) {
    let parts: Vec<String> = name.split('.').map(unquote).collect();
    let default_catalog = default_catalog(session);
    match parts.as_slice() {
        [s] => (default_catalog, s.clone()),
        [d, s] => (d.clone(), s.clone()),
        _ => (
            default_catalog,
            parts.last().cloned().unwrap_or_default(),
        ),
    }
}

fn columns_to_arrow_schema(columns: &[ColumnDef]) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(columns.len());
    for c in columns {
        let dt = sql_type_to_arrow(&c.data_type)?;
        let nullable = !c
            .options
            .iter()
            .any(|o| matches!(o.option, ColumnOption::NotNull));
        fields.push(Field::new(c.name.value.clone(), dt, nullable));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Map a sqlparser SQL type to an Arrow type. Unknown types fall back to Utf8.
fn sql_type_to_arrow(dt: &datafusion::sql::sqlparser::ast::DataType) -> Result<DataType> {
    use datafusion::sql::sqlparser::ast::DataType as SqlType;
    Ok(match dt {
        SqlType::Boolean | SqlType::Bool => DataType::Boolean,
        SqlType::TinyInt(_) => DataType::Int8,
        SqlType::SmallInt(_) => DataType::Int16,
        SqlType::Int(_) | SqlType::Integer(_) => DataType::Int32,
        SqlType::BigInt(_) => DataType::Int64,
        SqlType::Float(_) | SqlType::Real => DataType::Float32,
        SqlType::Double(_) | SqlType::DoublePrecision => DataType::Float64,
        SqlType::Varchar(_) | SqlType::Text | SqlType::String(_) | SqlType::Char(_) => {
            DataType::Utf8
        }
        SqlType::Binary(_) | SqlType::Varbinary(_) | SqlType::Blob(_) => DataType::Binary,
        SqlType::Date => DataType::Date32,
        SqlType::Timestamp(_, _) | SqlType::Datetime(_) => {
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
        }
        _ => DataType::Utf8,
    })
}

/// Build a single-column `Int64` result batch (used by count-style statements).
#[allow(dead_code)]
fn count_batch(value: i64) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::Int64,
        false,
    )]));
    let array: ArrayRef = Arc::new(Int64Array::from(vec![value]));
    Ok(RecordBatch::try_new(schema, vec![array])?)
}

#[cfg(test)]
mod ctas_tests {
    use super::*;

    #[test]
    fn slices_bare_query() {
        let sql = "create table t as select 1 as id";
        assert_eq!(extract_ctas_query(sql).as_deref(), Some("select 1 as id"));
    }

    #[test]
    fn slices_parenthesized_cte_query() {
        // dbt renders `as (\n with ... select ... )` — the AST re-render of
        // this form is invalid SQL, so the raw text must be sliced.
        let sql = "create table \"db\".\"default\".\"t\"\n\nas (\n\nwith a as (select 1 as i)\nselect * from a\n)";
        let q = extract_ctas_query(sql).expect("query");
        assert!(q.starts_with("with a as"), "got: {q}");
        assert!(q.ends_with("select * from a"), "got: {q}");
    }

    #[test]
    fn ignores_as_inside_identifiers_and_strings() {
        let sql = "create table \"as\".\"t\" as select 'as' as v";
        assert_eq!(
            extract_ctas_query(sql).as_deref(),
            Some("select 'as' as v")
        );
    }

    #[test]
    fn keeps_inner_parentheses() {
        let sql = "create table t as (select (1 + 2) as v)";
        assert_eq!(
            extract_ctas_query(sql).as_deref(),
            Some("select (1 + 2) as v")
        );
    }

    #[test]
    fn returns_none_without_query() {
        assert_eq!(extract_ctas_query("create table t (a int)"), None);
    }

    #[test]
    fn using_clause_maps_every_algorithm() {
        use datafusion::sql::sqlparser::ast::{Ident, IndexType};
        let alg = |name: &str| {
            index_algorithm_from_using(Some(&IndexType::Custom(Ident::new(name))))
        };
        // Every engine algorithm must be reachable from `USING <name>`.
        for name in IndexAlgorithm::all_names() {
            assert_eq!(alg(&name).name(), name, "USING {name} did not map");
        }
        // Unknown / absent -> the scalar default (Bitmap).
        assert_eq!(alg("nonsense").name(), "bitmap");
        assert_eq!(index_algorithm_from_using(None).name(), "bitmap");
    }
}
