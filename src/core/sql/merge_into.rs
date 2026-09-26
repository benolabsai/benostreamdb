// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Native `MERGE INTO` support for the DataFusion SQL layer.
//!
//! DataFusion's planner has no logical plan for `MERGE`, so a `MERGE INTO`
//! statement is rejected after parsing. We intercept the parsed
//! [`Statement::Merge`] *before* planning and translate it into the existing
//! key-based upsert ([`Table::merge`]) and delete ([`Table::delete_async`])
//! primitives.
//!
//! Supported form:
//!
//! ```sql
//! MERGE INTO target [AS t]
//! USING source [AS s]
//! ON t.key = s.key
//! WHEN MATCHED [AND <pred>] THEN UPDATE SET col = <expr>, ...
//! WHEN MATCHED [AND <pred>] THEN DELETE
//! WHEN NOT MATCHED [AND <pred>] THEN INSERT (col, ...) VALUES (<expr>, ...)
//! WHEN NOT MATCHED [AND <pred>] THEN INSERT ROW
//! WHEN NOT MATCHED BY SOURCE [AND <pred>] THEN DELETE
//! WHEN NOT MATCHED BY SOURCE [AND <pred>] THEN UPDATE SET col = <expr>, ...
//! ```
//!
//! The `ON` condition must be an equi-join between the target's key column(s)
//! and the source's columns. Clauses are evaluated in order with first-match-wins
//! semantics: each clause's query excludes rows already claimed by an earlier
//! clause of the same kind.
//!
//! Matched rows are computed as the target row with the `UPDATE` assignments
//! applied; not-matched rows from the `INSERT` values. Both are handed to
//! [`Table::merge`], which replaces rows whose key exists and inserts rows whose
//! key does not. `DELETE` clauses are applied via [`Table::delete_async`] using a
//! key filter built from the matched target keys.

use anyhow::{bail, Context, Result};
use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion::sql::parser::{DFParserBuilder, Statement as DFStatement};
use datafusion::sql::sqlparser::ast::{
    Assignment, AssignmentTarget, BinaryOperator, Expr as SqlExpr, MergeAction, MergeClause,
    MergeClauseKind, MergeInsertExpr, MergeInsertKind, Statement, TableFactor,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::TableReference;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::sql::BenoStreamTableProvider;
use crate::core::table::{MergeMode, Table};

/// Parse `sql` and, if it is a single `MERGE INTO` statement, execute it and
/// return the affected-rows batch. Returns `Ok(None)` for anything else (or if
/// the statement does not parse, so DataFusion can report its own error).
pub async fn try_parse_and_execute(ctx: &SessionContext, sql: &str) -> Result<Option<RecordBatch>> {
    // `GenericDialect` (not PostgreSQL) so `INSERT ROW` parses; the session's
    // own planner still uses the PostgreSQL dialect for non-MERGE statements.
    let dialect = GenericDialect {};
    let mut parser = match DFParserBuilder::new(sql).with_dialect(&dialect).build() {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    let statements = match parser.parse_statements() {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    if statements.len() != 1 {
        return Ok(None);
    }
    // Unwrap DataFusion's statement wrapper to the underlying sqlparser AST.
    let stmt = match &statements[0] {
        DFStatement::Statement(inner) => inner.as_ref(),
        _ => return Ok(None),
    };
    try_execute_merge(ctx, stmt).await
}

/// Execute a `MERGE INTO` statement if `stmt` is one; otherwise return `None`.
pub async fn try_execute_merge(
    ctx: &SessionContext,
    stmt: &Statement,
) -> Result<Option<RecordBatch>> {
    let Statement::Merge {
        table,
        source,
        on,
        clauses,
        ..
    } = stmt
    else {
        return Ok(None);
    };
    Ok(Some(execute_merge(ctx, table, source, on, clauses).await?))
}

async fn execute_merge(
    ctx: &SessionContext,
    table: &TableFactor,
    source: &TableFactor,
    on: &SqlExpr,
    clauses: &[MergeClause],
) -> Result<RecordBatch> {
    // --- 1. Resolve the target table -------------------------------------
    let (target_name, target_alias) = table_factor_ident(table, "MERGE target")?;
    let bare_name = target_name
        .rsplit('.')
        .next()
        .unwrap_or(&target_name)
        .to_string();
    let provider = ctx
        .table_provider(TableReference::from(bare_name.as_str()))
        .await
        .with_context(|| format!("MERGE target table '{target_name}' is not registered"))?;
    let provider = provider
        .as_any()
        .downcast_ref::<BenoStreamTableProvider>()
        .context("MERGE target is not a BenoStreamDB table")?;
    let target_table: Arc<Table> = provider.table.clone();
    let target_schema = target_table.arrow_schema();

    // --- 2. Source SQL + aliases -----------------------------------------
    let (source_sql, source_alias) = source_sql_and_alias(source);
    let target_ref = target_alias.clone().unwrap_or_else(|| bare_name.clone());

    // --- 3. Extract the equi-join key columns (target side) --------------
    let keys = extract_equi_keys(on, &target_ref, &source_alias)?;
    if keys.is_empty() {
        bail!(
            "MERGE ... ON must be an equality between a target column and a source column \
             (e.g. `ON t.id = s.id`)"
        );
    }
    let key0 = &keys[0];

    // --- 4. Partition clauses by kind, preserving order ------------------
    let matched: Vec<&MergeClause> = clauses
        .iter()
        .filter(|c| c.clause_kind == MergeClauseKind::Matched)
        .collect();
    let not_matched: Vec<&MergeClause> = clauses
        .iter()
        .filter(|c| {
            matches!(
                c.clause_kind,
                MergeClauseKind::NotMatched | MergeClauseKind::NotMatchedByTarget
            )
        })
        .collect();
    let not_matched_by_source: Vec<&MergeClause> = clauses
        .iter()
        .filter(|c| c.clause_kind == MergeClauseKind::NotMatchedBySource)
        .collect();
    if matched.is_empty() && not_matched.is_empty() && not_matched_by_source.is_empty() {
        bail!("MERGE statement has no WHEN clause");
    }

    let on_sql = on.to_string();
    let target_sql = table.to_string();

    // Source schema, needed for `INSERT ROW`.
    let source_schema = {
        let df = ctx
            .sql(&format!("SELECT * FROM {source_sql} LIMIT 0"))
            .await
            .with_context(|| {
                format!("MERGE: failed to resolve source schema for `{source_sql}`")
            })?;
        Arc::new(df.schema().as_arrow().clone())
    };

    let mut upsert_batches: Vec<RecordBatch> = Vec::new();
    let mut delete_filters: Vec<String> = Vec::new();
    let mut rows_affected: i64 = 0;

    // --- 5. WHEN MATCHED ------------------------------------------------
    for (i, clause) in matched.iter().enumerate() {
        let where_clause = clause_where(&matched, i);
        match &clause.action {
            MergeAction::Update { assignments } => {
                let select = matched_select_list(&target_schema, assignments, &target_ref);
                let sql = format!(
                    "SELECT {select} FROM {target_sql} JOIN {source_sql} ON {on_sql} \
                     WHERE {where_clause}"
                );
                let batches = run_sql(ctx, &sql).await?;
                upsert_batches.extend(cast_to_schema(batches, &target_schema)?);
            }
            MergeAction::Delete => {
                let select = key_select_list(&keys, &target_ref);
                let sql = format!(
                    "SELECT {select} FROM {target_sql} JOIN {source_sql} ON {on_sql} \
                     WHERE {where_clause}"
                );
                let batches = run_sql(ctx, &sql).await?;
                let (filter, count) = build_delete_filter(&keys, &batches)?;
                rows_affected += count as i64;
                if let Some(f) = filter {
                    delete_filters.push(f);
                }
            }
            MergeAction::Insert(_) => {
                bail!("MERGE ... WHEN MATCHED THEN INSERT is not valid")
            }
        }
    }

    // --- 6. WHEN NOT MATCHED (INSERT) -----------------------------------
    for (i, clause) in not_matched.iter().enumerate() {
        let where_clause = clause_where(&not_matched, i);
        match &clause.action {
            MergeAction::Insert(ins) => {
                let select =
                    insert_select_list(&target_schema, ins, &source_schema, &source_alias)?;
                let sql = format!(
                    "SELECT {select} FROM {source_sql} LEFT JOIN {target_sql} ON {on_sql} \
                     WHERE {target_ref}.{} IS NULL AND {where_clause}",
                    quote_ident(key0)
                );
                let batches = run_sql(ctx, &sql).await?;
                upsert_batches.extend(cast_to_schema(batches, &target_schema)?);
            }
            _ => bail!("MERGE ... WHEN NOT MATCHED only supports INSERT"),
        }
    }

    // --- 7. WHEN NOT MATCHED BY SOURCE (DELETE / UPDATE) ----------------
    for (i, clause) in not_matched_by_source.iter().enumerate() {
        let where_clause = clause_where(&not_matched_by_source, i);
        let anti_join = format!(
            "FROM {target_sql} LEFT JOIN {source_sql} ON {on_sql} \
             WHERE {source_alias}.{} IS NULL AND {where_clause}",
            quote_ident(key0)
        );
        match &clause.action {
            MergeAction::Delete => {
                let select = key_select_list(&keys, &target_ref);
                let sql = format!("SELECT {select} {anti_join}");
                let batches = run_sql(ctx, &sql).await?;
                let (filter, count) = build_delete_filter(&keys, &batches)?;
                rows_affected += count as i64;
                if let Some(f) = filter {
                    delete_filters.push(f);
                }
            }
            MergeAction::Update { assignments } => {
                let select = matched_select_list(&target_schema, assignments, &target_ref);
                let sql = format!("SELECT {select} {anti_join}");
                let batches = run_sql(ctx, &sql).await?;
                upsert_batches.extend(cast_to_schema(batches, &target_schema)?);
            }
            MergeAction::Insert(_) => {
                bail!("MERGE ... WHEN NOT MATCHED BY SOURCE only supports DELETE or UPDATE")
            }
        }
    }

    // --- 8. Apply upserts via the key-based merge primitive -------------
    rows_affected += upsert_batches
        .iter()
        .map(|b| b.num_rows() as i64)
        .sum::<i64>();
    if !upsert_batches.is_empty() {
        let key = keys.join(",");
        let table_for_merge = target_table.clone();
        // `Table::merge` drives its own runtime via `block_on`, which panics
        // inside an async context — run it on a blocking thread.
        tokio::task::spawn_blocking(move || {
            table_for_merge.merge(upsert_batches, &key, MergeMode::MergeOnRead)
        })
        .await
        .context("MERGE task panicked")??;
    }

    // --- 9. Apply deletes ------------------------------------------------
    for filter in delete_filters {
        target_table.delete_async(&filter).await?;
    }

    // --- 10. Return an affected-rows result ------------------------------
    let schema = Arc::new(Schema::new(vec![Field::new(
        "rows_affected",
        DataType::Int64,
        false,
    )]));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![rows_affected]))],
    )?)
}

async fn run_sql(ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
    let df = ctx
        .sql(sql)
        .await
        .with_context(|| format!("MERGE: failed to plan `{sql}`"))?;
    df.collect()
        .await
        .with_context(|| format!("MERGE: failed to execute `{sql}`"))
}

/// The `WHERE` predicate for clause `i` in `group`: its own predicate AND the
/// negation of every earlier clause's predicate (first-match-wins).
fn clause_where(group: &[&MergeClause], i: usize) -> String {
    let mut parts = vec![clause_predicate(group[i])];
    for earlier in &group[..i] {
        parts.push(format!("NOT {}", clause_predicate(earlier)));
    }
    parts.join(" AND ")
}

fn clause_predicate(clause: &MergeClause) -> String {
    match &clause.predicate {
        Some(p) => format!("({p})"),
        None => "TRUE".to_string(),
    }
}

/// Build the `SELECT` list for the matched (UPDATE) rows: each target column is
/// either the assignment expression or the existing target value.
fn matched_select_list(target: &SchemaRef, assignments: &[Assignment], target_ref: &str) -> String {
    let map = assignment_map(assignments);
    target
        .fields()
        .iter()
        .map(|f| {
            let name = f.name();
            if let Some(expr) = map.get(name) {
                format!("{expr} AS {}", quote_ident(name))
            } else {
                format!(
                    "{target_ref}.{} AS {}",
                    quote_ident(name),
                    quote_ident(name)
                )
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build the `SELECT` list for the not-matched (INSERT) rows: each target column
/// is either the supplied insert value or `NULL`.
fn insert_select_list(
    target: &SchemaRef,
    ins: &MergeInsertExpr,
    source_schema: &SchemaRef,
    source_alias: &str,
) -> Result<String> {
    match &ins.kind {
        MergeInsertKind::Row => {
            // Insert the source row as-is: map source columns to target columns
            // by name; target columns absent from the source become NULL.
            Ok(target
                .fields()
                .iter()
                .map(|f| {
                    let name = f.name();
                    if source_schema.field_with_name(name).is_ok() {
                        format!(
                            "{source_alias}.{} AS {}",
                            quote_ident(name),
                            quote_ident(name)
                        )
                    } else {
                        format!("NULL AS {}", quote_ident(name))
                    }
                })
                .collect::<Vec<_>>()
                .join(", "))
        }
        MergeInsertKind::Values(values) => {
            if values.rows.len() != 1 {
                bail!(
                    "MERGE ... INSERT VALUES must have exactly one row (got {})",
                    values.rows.len()
                );
            }
            let row = &values.rows[0];

            let mut map: HashMap<String, String> = HashMap::new();
            if ins.columns.is_empty() {
                // Positional: values map to target columns in schema order.
                if row.len() != target.fields().len() {
                    bail!(
                        "MERGE ... INSERT VALUES has {} values but the target has {} columns",
                        row.len(),
                        target.fields().len()
                    );
                }
                for (f, v) in target.fields().iter().zip(row.iter()) {
                    map.insert(f.name().clone(), v.to_string());
                }
            } else {
                if ins.columns.len() != row.len() {
                    bail!(
                        "MERGE ... INSERT has {} columns but {} values",
                        ins.columns.len(),
                        row.len()
                    );
                }
                for (c, v) in ins.columns.iter().zip(row.iter()) {
                    map.insert(c.value.clone(), v.to_string());
                }
            }

            Ok(target
                .fields()
                .iter()
                .map(|f| {
                    let expr = map
                        .get(f.name())
                        .cloned()
                        .unwrap_or_else(|| "NULL".to_string());
                    format!("{expr} AS {}", quote_ident(f.name()))
                })
                .collect::<Vec<_>>()
                .join(", "))
        }
    }
}

/// `SELECT` list of the target key columns, used to build a delete filter.
fn key_select_list(keys: &[String], target_ref: &str) -> String {
    keys.iter()
        .map(|k| format!("{target_ref}.{}", quote_ident(k)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build a `WHERE` filter that matches the key values in `batches`, plus the
/// number of rows it covers. Returns `(None, 0)` when there are no rows.
fn build_delete_filter(
    keys: &[String],
    batches: &[RecordBatch],
) -> Result<(Option<String>, usize)> {
    let mut clauses = Vec::new();
    let mut count = 0usize;
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut parts = Vec::new();
            let mut has_null = false;
            for (i, key) in keys.iter().enumerate() {
                let col = batch.column(i);
                if col.is_null(row) {
                    has_null = true;
                    break;
                }
                parts.push(format!(
                    "{} = {}",
                    quote_ident(key),
                    sql_literal(col.as_ref(), row)?
                ));
            }
            if has_null {
                continue;
            }
            count += 1;
            if keys.len() == 1 {
                clauses.push(parts[0].clone());
            } else {
                clauses.push(format!("({})", parts.join(" AND ")));
            }
        }
    }
    if clauses.is_empty() {
        return Ok((None, 0));
    }
    Ok((Some(clauses.join(" OR ")), count))
}

/// Render a single array value as a SQL literal.
fn sql_literal(array: &dyn Array, row: usize) -> Result<String> {
    let s = arrow::util::display::array_value_to_string(array, row)
        .map_err(|e| anyhow::anyhow!("failed to render key value: {e}"))?;
    match array.data_type() {
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Time32(_)
        | DataType::Time64(_) => Ok(format!("'{}'", s.replace('\'', "''"))),
        _ => Ok(s),
    }
}

fn assignment_map(assignments: &[Assignment]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for a in assignments {
        if let AssignmentTarget::ColumnName(name) = &a.target {
            let full = name.to_string();
            let col = full.rsplit('.').next().unwrap_or(&full).to_string();
            map.insert(col, a.value.to_string());
        }
    }
    map
}

/// Extract the target-side key columns from an `ON` condition that is an
/// equality (or an `AND` of equalities) between a target column and a source
/// column.
fn extract_equi_keys(on: &SqlExpr, target_ref: &str, source_ref: &str) -> Result<Vec<String>> {
    let mut keys = Vec::new();
    collect_equi_keys(on, target_ref, source_ref, &mut keys)?;
    Ok(keys)
}

fn collect_equi_keys(
    expr: &SqlExpr,
    target_ref: &str,
    source_ref: &str,
    out: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_equi_keys(left, target_ref, source_ref, out)?;
            collect_equi_keys(right, target_ref, source_ref, out)?;
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            if let Some(t) = qualified_column(left, target_ref) {
                if qualified_column(right, source_ref).is_some() {
                    out.push(t);
                    return Ok(());
                }
            }
            if let Some(t) = qualified_column(right, target_ref) {
                if qualified_column(left, source_ref).is_some() {
                    out.push(t);
                    return Ok(());
                }
            }
            bail!(
                "MERGE ... ON equality must compare a target column with a source column \
                 (e.g. `{target_ref}.id = {source_ref}.id`)"
            );
        }
        _ => bail!("MERGE ... ON must be an equality (or an AND of equalities)"),
    }
    Ok(())
}

/// If `expr` is a column reference qualified by `qualifier`, return the bare
/// column name.
fn qualified_column(expr: &SqlExpr, qualifier: &str) -> Option<String> {
    if let SqlExpr::CompoundIdentifier(parts) = expr {
        if parts.len() == 2 && parts[0].value == qualifier {
            return Some(parts[1].value.clone());
        }
    }
    None
}

fn table_factor_ident(tf: &TableFactor, what: &str) -> Result<(String, Option<String>)> {
    match tf {
        TableFactor::Table { name, alias, .. } => {
            let n = name.to_string();
            Ok((n, alias.as_ref().map(|a| a.name.value.clone())))
        }
        _ => bail!("{what} must be a table name"),
    }
}

/// The source's SQL text and the alias it can be referenced by. A subquery
/// without an alias is given one so the generated anti-join can reference it.
fn source_sql_and_alias(source: &TableFactor) -> (String, String) {
    match source {
        TableFactor::Table { name, alias, .. } => {
            let a = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .unwrap_or_else(|| {
                    name.to_string()
                        .rsplit('.')
                        .next()
                        .unwrap_or("__source")
                        .to_string()
                });
            (source.to_string(), a)
        }
        TableFactor::Derived { alias, .. } => match alias.as_ref().map(|a| a.name.value.clone()) {
            Some(a) => (source.to_string(), a),
            None => (format!("{source} AS __source"), "__source".to_string()),
        },
        _ => (source.to_string(), "__source".to_string()),
    }
}

/// Cast each result batch to the target schema (column order and types).
fn cast_to_schema(batches: Vec<RecordBatch>, target: &SchemaRef) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        if batch.num_columns() != target.fields().len() {
            bail!(
                "MERGE: result has {} columns but the target has {}",
                batch.num_columns(),
                target.fields().len()
            );
        }
        let mut cols: Vec<Arc<dyn Array>> = Vec::with_capacity(target.fields().len());
        for (i, field) in target.fields().iter().enumerate() {
            let col = batch.column(i);
            let casted = if col.data_type() == field.data_type() {
                col.clone()
            } else {
                arrow::compute::cast(col, field.data_type()).with_context(|| {
                    format!(
                        "MERGE: cannot cast column '{}' from {} to {}",
                        field.name(),
                        col.data_type(),
                        field.data_type()
                    )
                })?
            };
            cols.push(casted);
        }
        out.push(RecordBatch::try_new(target.clone(), cols)?);
    }
    Ok(out)
}

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sql::session::BenoStreamSession;
    use arrow::array::{Int32Array, StringArray};
    use tempfile::tempdir;

    async fn make_table(uri: &str, rows: &[(i32, &str)]) -> Result<Arc<Table>> {
        let table = Table::new_async(uri.to_string()).await?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: Vec<i32> = rows.iter().map(|(i, _)| *i).collect();
        let names: Vec<&str> = rows.iter().map(|(_, n)| *n).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )?;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
        Ok(Arc::new(table))
    }

    async fn read_pairs(table: &Table) -> Result<Vec<(i32, String)>> {
        let batches = table.read_async(None, None, None).await?;
        let mut out = Vec::new();
        for b in batches {
            let ids = b
                .column_by_name("id")
                .context("id column")?
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("id is Int32")?;
            let names = b
                .column_by_name("name")
                .context("name column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("name is Utf8")?;
            for i in 0..b.num_rows() {
                out.push((ids.value(i), names.value(i).to_string()));
            }
        }
        out.sort();
        Ok(out)
    }

    fn affected(batches: &[RecordBatch]) -> i64 {
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("rows_affected is Int64")
            .value(0)
    }

    #[tokio::test]
    async fn merge_into_updates_and_inserts() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path().to_str().unwrap();
        let target = make_table(&format!("file://{base}/target"), &[(1, "a"), (2, "b")]).await?;
        let source = make_table(&format!("file://{base}/source"), &[(2, "B"), (3, "c")]).await?;

        let session = BenoStreamSession::new(None);
        session.register_table("target", target.clone())?;
        session.register_table("source", source.clone())?;

        let (batches, _) = session
            .sql(
                "MERGE INTO target t USING source s ON t.id = s.id \
                 WHEN MATCHED THEN UPDATE SET name = s.name \
                 WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)",
            )
            .await?;
        assert_eq!(affected(&batches), 2, "one update + one insert");

        assert_eq!(
            read_pairs(&target).await?,
            vec![
                (1, "a".to_string()),
                (2, "B".to_string()),
                (3, "c".to_string()),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn merge_into_insert_only() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path().to_str().unwrap();
        let target = make_table(&format!("file://{base}/target"), &[(1, "a")]).await?;
        let source = make_table(&format!("file://{base}/source"), &[(5, "e"), (6, "f")]).await?;

        let session = BenoStreamSession::new(None);
        session.register_table("target", target.clone())?;
        session.register_table("source", source.clone())?;

        let (batches, _) = session
            .sql(
                "MERGE INTO target t USING source s ON t.id = s.id \
                 WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)",
            )
            .await?;
        assert_eq!(affected(&batches), 2);
        assert_eq!(
            read_pairs(&target).await?,
            vec![
                (1, "a".to_string()),
                (5, "e".to_string()),
                (6, "f".to_string()),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn merge_into_matched_delete() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path().to_str().unwrap();
        let target = make_table(
            &format!("file://{base}/target"),
            &[(1, "a"), (2, "b"), (3, "c")],
        )
        .await?;
        let source = make_table(&format!("file://{base}/source"), &[(2, "x")]).await?;

        let session = BenoStreamSession::new(None);
        session.register_table("target", target.clone())?;
        session.register_table("source", source.clone())?;

        let (batches, _) = session
            .sql(
                "MERGE INTO target t USING source s ON t.id = s.id \
                 WHEN MATCHED THEN DELETE",
            )
            .await?;
        assert_eq!(affected(&batches), 1);
        assert_eq!(
            read_pairs(&target).await?,
            vec![(1, "a".to_string()), (3, "c".to_string())]
        );
        Ok(())
    }

    #[tokio::test]
    async fn merge_into_not_matched_by_source_delete() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path().to_str().unwrap();
        let target = make_table(
            &format!("file://{base}/target"),
            &[(1, "a"), (2, "b"), (3, "c")],
        )
        .await?;
        let source = make_table(&format!("file://{base}/source"), &[(2, "x")]).await?;

        let session = BenoStreamSession::new(None);
        session.register_table("target", target.clone())?;
        session.register_table("source", source.clone())?;

        let (batches, _) = session
            .sql(
                "MERGE INTO target t USING source s ON t.id = s.id \
                 WHEN NOT MATCHED BY SOURCE THEN DELETE",
            )
            .await?;
        assert_eq!(affected(&batches), 2, "ids 1 and 3 have no source match");
        assert_eq!(read_pairs(&target).await?, vec![(2, "b".to_string())]);
        Ok(())
    }

    #[tokio::test]
    async fn merge_into_insert_row() -> Result<()> {
        let dir = tempdir()?;
        let base = dir.path().to_str().unwrap();
        let target = make_table(&format!("file://{base}/target"), &[(1, "a")]).await?;
        let source = make_table(&format!("file://{base}/source"), &[(7, "g"), (8, "h")]).await?;

        let session = BenoStreamSession::new(None);
        session.register_table("target", target.clone())?;
        session.register_table("source", source.clone())?;

        let (batches, _) = session
            .sql(
                "MERGE INTO target t USING source s ON t.id = s.id \
                 WHEN NOT MATCHED THEN INSERT ROW",
            )
            .await?;
        assert_eq!(affected(&batches), 2);
        assert_eq!(
            read_pairs(&target).await?,
            vec![
                (1, "a".to_string()),
                (7, "g".to_string()),
                (8, "h".to_string()),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn merge_into_first_match_wins() -> Result<()> {
        // Two matched clauses: the first (predicate id = 2) updates, the second
        // deletes everything else. First-match-wins must not delete id = 2.
        let dir = tempdir()?;
        let base = dir.path().to_str().unwrap();
        let target = make_table(
            &format!("file://{base}/target"),
            &[(1, "a"), (2, "b"), (3, "c")],
        )
        .await?;
        let source = make_table(
            &format!("file://{base}/source"),
            &[(1, "x"), (2, "y"), (3, "z")],
        )
        .await?;

        let session = BenoStreamSession::new(None);
        session.register_table("target", target.clone())?;
        session.register_table("source", source.clone())?;

        let (batches, _) = session
            .sql(
                "MERGE INTO target t USING source s ON t.id = s.id \
                 WHEN MATCHED AND t.id = 2 THEN UPDATE SET name = s.name \
                 WHEN MATCHED THEN DELETE",
            )
            .await?;
        assert_eq!(affected(&batches), 3, "one update + two deletes");
        assert_eq!(read_pairs(&target).await?, vec![(2, "y".to_string())]);
        Ok(())
    }
}
