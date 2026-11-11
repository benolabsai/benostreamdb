// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{anyhow, Context, Result};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use sqlx::{any::AnyPoolOptions, AnyPool, Row};
use std::future::Future;
use std::sync::Once;
use std::time::Duration;

use super::common::{
    catalog_conflict_error, catalog_not_found_error, ensure_storage_table_initialized,
    extract_metadata_location_from_updates, load_table_metadata_from_location,
    persist_updates_to_storage, table_root_from_metadata_location,
    validate_requirements_against_metadata,
};
use super::{Catalog, TableMetadata};

static SQLX_DRIVERS_INIT: Once = Once::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JdbcDialect {
    Postgres,
    SqliteOrMySql,
}

impl JdbcDialect {
    fn from_uri(uri: &str) -> Self {
        let lower = uri.trim().to_ascii_lowercase();
        if lower.starts_with("postgres://") || lower.starts_with("postgresql://") {
            Self::Postgres
        } else {
            Self::SqliteOrMySql
        }
    }
}

fn is_sqlite_busy_or_locked(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("database is locked") || msg.contains("database schema is locked")
}

fn is_unique_constraint_violation(err: &sqlx::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("unique constraint")
        || msg.contains("duplicate key")
        || msg.contains("23505")
        || msg.contains("constraint failed")
}

async fn run_sqlite_retry<T, F, Fut>(mut op: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    for attempt in 0..7usize {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) if is_sqlite_busy_or_locked(&e) => {
                tokio::time::sleep(Duration::from_millis(5 * (attempt as u64 + 1))).await;
            }
            Err(e) => return Err(e),
        }
    }
    op().await
}

/// JDBC-style SQL Iceberg Catalog supporting PostgreSQL and SQLite backends via `sqlx::AnyPool`.
pub struct JdbcCatalogClient {
    pool: AnyPool,
    warehouse: String,
    catalog_name: String,
    dialect: JdbcDialect,
    is_sqlite: bool,
    is_in_memory: bool,
}

impl JdbcCatalogClient {
    pub async fn new(uri: String, warehouse: Option<String>, catalog_name: String) -> Result<Self> {
        SQLX_DRIVERS_INIT.call_once(sqlx::any::install_default_drivers);

        let lower = uri.trim().to_ascii_lowercase();
        let dialect = JdbcDialect::from_uri(&uri);
        let is_sqlite = lower.starts_with("sqlite:") || lower.starts_with("sqlite://");
        let is_in_memory = lower == "sqlite::memory:"
            || lower.contains(":memory:")
            || lower.contains("mode=memory");

        let connect_uri = if is_sqlite && lower.starts_with("sqlite://") && !is_in_memory {
            let without_scheme = uri.trim_start_matches("sqlite://");
            let file_part = without_scheme.split('?').next().unwrap_or(without_scheme);
            if !file_part.is_empty() && file_part != ":memory:" {
                if let Some(parent) = std::path::Path::new(file_part).parent() {
                    if !parent.as_os_str().is_empty() && !parent.exists() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                }
            }
            if !lower.contains("mode=") {
                if uri.contains('?') {
                    format!("{uri}&mode=rwc")
                } else {
                    format!("{uri}?mode=rwc")
                }
            } else {
                uri.clone()
            }
        } else {
            uri.clone()
        };

        let max_conns = if is_in_memory { 1 } else { 5 };
        let pool = AnyPoolOptions::new()
            .max_connections(max_conns)
            .connect(&connect_uri)
            .await
            .with_context(|| format!("Failed to connect to JDBC catalog database at {uri}"))?;

        let client = Self {
            pool,
            warehouse: warehouse
                .unwrap_or_else(|| {
                    std::env::temp_dir()
                        .join("benostream_warehouse")
                        .to_string_lossy()
                        .into_owned()
                })
                .trim_end_matches('/')
                .to_string(),
            catalog_name,
            dialect,
            is_sqlite,
            is_in_memory,
        };

        client.setup().await?;
        Ok(client)
    }

    /// Return the current committed `metadata_location` pointer for diagnostics and tests.
    pub async fn metadata_location(
        &self,
        namespace: &str,
        table_name: &str,
    ) -> Result<Option<String>> {
        let sql = match self.dialect {
            JdbcDialect::Postgres => {
                "SELECT metadata_location FROM iceberg_tables \
                 WHERE catalog_name = $1 AND table_namespace = $2 AND table_name = $3"
            }
            JdbcDialect::SqliteOrMySql => {
                "SELECT metadata_location FROM iceberg_tables \
                 WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
            }
        };
        let row = sqlx::query(sql)
            .bind(&self.catalog_name)
            .bind(namespace)
            .bind(table_name)
            .fetch_optional(&self.pool)
            .await?;
        match row {
            Some(r) => Ok(r.try_get::<Option<String>, _>(0)?),
            None => Ok(None),
        }
    }

    async fn setup(&self) -> Result<()> {
        if self.is_sqlite {
            let _ = sqlx::query("PRAGMA busy_timeout = 5000")
                .execute(&self.pool)
                .await;
            if !self.is_in_memory {
                let _ = sqlx::query("PRAGMA journal_mode = WAL")
                    .execute(&self.pool)
                    .await;
            }
        }

        run_sqlite_retry(|| async {
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS iceberg_tables (
                    catalog_name VARCHAR(255) NOT NULL,
                    table_namespace VARCHAR(255) NOT NULL,
                    table_name VARCHAR(255) NOT NULL,
                    metadata_location VARCHAR(1024),
                    previous_metadata_location VARCHAR(1024),
                    PRIMARY KEY (catalog_name, table_namespace, table_name)
                )",
            )
            .execute(&self.pool)
            .await?;

            sqlx::query(
                "CREATE TABLE IF NOT EXISTS namespaces (
                    catalog_name VARCHAR(255) NOT NULL,
                    namespace VARCHAR(255) NOT NULL,
                    PRIMARY KEY (catalog_name, namespace)
                )",
            )
            .execute(&self.pool)
            .await?;

            Ok(())
        })
        .await
    }

    async fn ensure_namespace_row(&self, namespace: &str) -> Result<()> {
        let sql = match self.dialect {
            JdbcDialect::Postgres => {
                "INSERT INTO namespaces (catalog_name, namespace) VALUES ($1, $2) \
                 ON CONFLICT (catalog_name, namespace) DO NOTHING"
            }
            JdbcDialect::SqliteOrMySql => {
                "INSERT INTO namespaces (catalog_name, namespace) VALUES (?, ?) \
                 ON CONFLICT (catalog_name, namespace) DO NOTHING"
            }
        };
        run_sqlite_retry(|| async {
            sqlx::query(sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .execute(&self.pool)
                .await?;
            Ok(())
        })
        .await
    }

    async fn commit_table_internal(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        let select_sql = match self.dialect {
            JdbcDialect::Postgres => {
                "SELECT metadata_location FROM iceberg_tables \
                 WHERE catalog_name = $1 AND table_namespace = $2 AND table_name = $3"
            }
            JdbcDialect::SqliteOrMySql => {
                "SELECT metadata_location FROM iceberg_tables \
                 WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
            }
        };

        let existing_row = run_sqlite_retry(|| async {
            Ok(sqlx::query(select_sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .bind(table_name)
                .fetch_optional(&self.pool)
                .await?)
        })
        .await?;

        let row = existing_row.ok_or_else(|| {
            catalog_not_found_error(
                "jdbc",
                format!("Table '{namespace}.{table_name}' not found in JDBC catalog"),
            )
        })?;
        let current_loc: Option<String> = row.try_get(0)?;

        let existing_meta = match current_loc.as_deref() {
            Some(loc) if !loc.is_empty() => Some(load_table_metadata_from_location(loc).await?),
            _ => None,
        };

        validate_requirements_against_metadata(&requirements, existing_meta.as_ref(), "jdbc")?;

        let new_loc = if let Some(loc) = extract_metadata_location_from_updates(&updates) {
            loc
        } else {
            let table_root = current_loc
                .as_deref()
                .map(table_root_from_metadata_location)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("{}/{}/{}", self.warehouse, namespace, table_name));
            persist_updates_to_storage(&table_root, &updates).await?
        };

        let rows_affected = run_sqlite_retry(|| async {
            let res = match (&current_loc, self.dialect) {
                (Some(prev), JdbcDialect::Postgres) => {
                    sqlx::query(
                        "UPDATE iceberg_tables \
                         SET previous_metadata_location = metadata_location, \
                             metadata_location = $1 \
                         WHERE catalog_name = $2 AND table_namespace = $3 \
                           AND table_name = $4 AND metadata_location = $5",
                    )
                    .bind(&new_loc)
                    .bind(&self.catalog_name)
                    .bind(namespace)
                    .bind(table_name)
                    .bind(prev)
                    .execute(&self.pool)
                    .await?
                }
                (Some(prev), JdbcDialect::SqliteOrMySql) => {
                    sqlx::query(
                        "UPDATE iceberg_tables \
                         SET previous_metadata_location = metadata_location, \
                             metadata_location = ? \
                         WHERE catalog_name = ? AND table_namespace = ? \
                           AND table_name = ? AND metadata_location = ?",
                    )
                    .bind(&new_loc)
                    .bind(&self.catalog_name)
                    .bind(namespace)
                    .bind(table_name)
                    .bind(prev)
                    .execute(&self.pool)
                    .await?
                }
                (None, JdbcDialect::Postgres) => {
                    sqlx::query(
                        "UPDATE iceberg_tables \
                         SET previous_metadata_location = metadata_location, \
                             metadata_location = $1 \
                         WHERE catalog_name = $2 AND table_namespace = $3 \
                           AND table_name = $4 AND metadata_location IS NULL",
                    )
                    .bind(&new_loc)
                    .bind(&self.catalog_name)
                    .bind(namespace)
                    .bind(table_name)
                    .execute(&self.pool)
                    .await?
                }
                (None, JdbcDialect::SqliteOrMySql) => {
                    sqlx::query(
                        "UPDATE iceberg_tables \
                         SET previous_metadata_location = metadata_location, \
                             metadata_location = ? \
                         WHERE catalog_name = ? AND table_namespace = ? \
                           AND table_name = ? AND metadata_location IS NULL",
                    )
                    .bind(&new_loc)
                    .bind(&self.catalog_name)
                    .bind(namespace)
                    .bind(table_name)
                    .execute(&self.pool)
                    .await?
                }
            };
            Ok(res.rows_affected())
        })
        .await?;

        if rows_affected == 0 {
            return Err(catalog_conflict_error(
                "jdbc",
                format!(
                    "CAS conflict committing '{namespace}.{table_name}': metadata_location changed concurrently"
                ),
            ));
        }

        Ok(())
    }
}

#[async_trait]
impl Catalog for JdbcCatalogClient {
    async fn create_namespace(&self, namespace: &str) -> Result<()> {
        self.ensure_namespace_row(namespace).await
    }

    async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        let sql = match self.dialect {
            JdbcDialect::Postgres => {
                "SELECT 1 FROM namespaces WHERE catalog_name = $1 AND namespace = $2 LIMIT 1"
            }
            JdbcDialect::SqliteOrMySql => {
                "SELECT 1 FROM namespaces WHERE catalog_name = ? AND namespace = ? LIMIT 1"
            }
        };
        let row = run_sqlite_retry(|| async {
            Ok(sqlx::query(sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .fetch_optional(&self.pool)
                .await?)
        })
        .await?;
        Ok(row.is_some())
    }

    async fn drop_namespace(&self, namespace: &str) -> Result<()> {
        let count_sql = match self.dialect {
            JdbcDialect::Postgres => {
                "SELECT COUNT(*) FROM iceberg_tables WHERE catalog_name = $1 AND table_namespace = $2"
            }
            JdbcDialect::SqliteOrMySql => {
                "SELECT COUNT(*) FROM iceberg_tables WHERE catalog_name = ? AND table_namespace = ?"
            }
        };
        let count: i64 = run_sqlite_retry(|| async {
            let row = sqlx::query(count_sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .fetch_one(&self.pool)
                .await?;
            let count = match row.try_get::<i64, _>(0) {
                Ok(c) => c,
                Err(_) => row.try_get::<i32, _>(0).unwrap_or(0) as i64,
            };
            Ok(count)
        })
        .await?;

        if count > 0 {
            return Err(catalog_conflict_error(
                "jdbc",
                format!("Namespace '{namespace}' is not empty"),
            ));
        }

        let del_sql = match self.dialect {
            JdbcDialect::Postgres => {
                "DELETE FROM namespaces WHERE catalog_name = $1 AND namespace = $2"
            }
            JdbcDialect::SqliteOrMySql => {
                "DELETE FROM namespaces WHERE catalog_name = ? AND namespace = ?"
            }
        };
        run_sqlite_retry(|| async {
            sqlx::query(del_sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .execute(&self.pool)
                .await?;
            Ok(())
        })
        .await
    }

    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        if self.table_exists(namespace, table_name).await? {
            return Err(catalog_conflict_error(
                "jdbc",
                format!("Table '{namespace}.{table_name}' already exists"),
            ));
        }

        let table_location = location
            .map(|s| s.trim_end_matches('/').to_string())
            .unwrap_or_else(|| format!("{}/{}/{}", self.warehouse, namespace, table_name));

        let metadata_location = ensure_storage_table_initialized(&table_location, &schema).await?;
        self.ensure_namespace_row(namespace).await?;

        let insert_sql = match self.dialect {
            JdbcDialect::Postgres => {
                "INSERT INTO iceberg_tables (catalog_name, table_namespace, table_name, metadata_location) \
                 VALUES ($1, $2, $3, $4)"
            }
            JdbcDialect::SqliteOrMySql => {
                "INSERT INTO iceberg_tables (catalog_name, table_namespace, table_name, metadata_location) \
                 VALUES (?, ?, ?, ?)"
            }
        };

        run_sqlite_retry(|| async {
            match sqlx::query(insert_sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .bind(table_name)
                .bind(&metadata_location)
                .execute(&self.pool)
                .await
            {
                Ok(_) => Ok(()),
                Err(e) if is_unique_constraint_violation(&e) => Err(catalog_conflict_error(
                    "jdbc",
                    format!("Table '{namespace}.{table_name}' already exists"),
                )),
                Err(e) => Err(e.into()),
            }
        })
        .await
    }

    async fn drop_table(&self, namespace: &str, table_name: &str) -> Result<()> {
        let sql = match self.dialect {
            JdbcDialect::Postgres => {
                "DELETE FROM iceberg_tables WHERE catalog_name = $1 AND table_namespace = $2 AND table_name = $3"
            }
            JdbcDialect::SqliteOrMySql => {
                "DELETE FROM iceberg_tables WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
            }
        };
        run_sqlite_retry(|| async {
            sqlx::query(sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .bind(table_name)
                .execute(&self.pool)
                .await?;
            Ok(())
        })
        .await
    }

    async fn list_tables(&self, namespace: &str) -> Result<Vec<String>> {
        let sql = match self.dialect {
            JdbcDialect::Postgres => {
                "SELECT table_name FROM iceberg_tables \
                 WHERE catalog_name = $1 AND table_namespace = $2 \
                 ORDER BY table_name ASC"
            }
            JdbcDialect::SqliteOrMySql => {
                "SELECT table_name FROM iceberg_tables \
                 WHERE catalog_name = ? AND table_namespace = ? \
                 ORDER BY table_name ASC"
            }
        };
        run_sqlite_retry(|| async {
            let rows = sqlx::query(sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .fetch_all(&self.pool)
                .await?;
            let mut names = Vec::with_capacity(rows.len());
            for row in rows {
                names.push(row.try_get::<String, _>(0)?);
            }
            Ok(names)
        })
        .await
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let sql = match self.dialect {
            JdbcDialect::Postgres => {
                "SELECT metadata_location FROM iceberg_tables \
                 WHERE catalog_name = $1 AND table_namespace = $2 AND table_name = $3"
            }
            JdbcDialect::SqliteOrMySql => {
                "SELECT metadata_location FROM iceberg_tables \
                 WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
            }
        };
        let row = run_sqlite_retry(|| async {
            Ok(sqlx::query(sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .bind(table_name)
                .fetch_optional(&self.pool)
                .await?)
        })
        .await?;

        let row = row.ok_or_else(|| {
            catalog_not_found_error(
                "jdbc",
                format!("Table '{namespace}.{table_name}' not found in JDBC catalog"),
            )
        })?;

        let metadata_location: Option<String> = row.try_get(0)?;
        let loc = metadata_location.filter(|s| !s.is_empty()).ok_or_else(|| {
            catalog_not_found_error(
                "jdbc",
                format!("Table '{namespace}.{table_name}' has missing metadata_location"),
            )
        })?;

        load_table_metadata_from_location(&loc).await
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Err(anyhow!("JDBC Catalog does not support branching."))
    }

    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        let sql = match self.dialect {
            JdbcDialect::Postgres => {
                "SELECT COUNT(*) FROM iceberg_tables \
                 WHERE catalog_name = $1 AND table_namespace = $2 AND table_name = $3"
            }
            JdbcDialect::SqliteOrMySql => {
                "SELECT COUNT(*) FROM iceberg_tables \
                 WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?"
            }
        };
        let count: i64 = run_sqlite_retry(|| async {
            let row = sqlx::query(sql)
                .bind(&self.catalog_name)
                .bind(namespace)
                .bind(table_name)
                .fetch_one(&self.pool)
                .await?;
            let count = match row.try_get::<i64, _>(0) {
                Ok(c) => c,
                Err(_) => row.try_get::<i32, _>(0).unwrap_or(0) as i64,
            };
            Ok(count)
        })
        .await?;

        Ok(count > 0)
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, Vec::new())
            .await
    }

    async fn commit_table_with_requirements(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, requirements)
            .await
    }
}
