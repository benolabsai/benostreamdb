// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Extensive end-to-end tests for the SQL DDL / maintenance surface
//! (`src/core/sql/catalog_ddl.rs`).

use benostreamdb::core::sql::session::BenoStreamSession;
use benostreamdb::core::sql::BenoStreamTableProvider;
use datafusion::sql::TableReference;
use tempfile::tempdir;

fn session_with_warehouse(warehouse: &str) -> BenoStreamSession {
    let mut session = BenoStreamSession::new(None);
    session.set_warehouse(Some(warehouse.to_string()));
    session
}

async fn table_of(
    session: &BenoStreamSession,
    db: &str,
    schema: &str,
    table: &str,
) -> anyhow::Result<std::sync::Arc<benostreamdb::Table>> {
    let provider = session
        .get_ctx()
        .table_provider(TableReference::full(db, schema, table))
        .await?;
    let provider = provider
        .as_any()
        .downcast_ref::<BenoStreamTableProvider>()
        .expect("BenoStreamTableProvider");
    Ok(provider.table.clone())
}

async fn row_count(session: &BenoStreamSession, table: &str) -> anyhow::Result<usize> {
    let (batches, _) = session.sql(&format!("SELECT * FROM {};", table)).await?;
    Ok(batches.iter().map(|b| b.num_rows()).sum())
}

/// Create a fresh `mydb.myschema.t (id INT, name VARCHAR)` and return the session.
async fn setup(dir: &tempfile::TempDir) -> anyhow::Result<BenoStreamSession> {
    let session = session_with_warehouse(dir.path().to_str().unwrap());
    session.sql("CREATE DATABASE mydb;").await?;
    session.sql("CREATE SCHEMA mydb.myschema;").await?;
    session
        .sql("CREATE TABLE mydb.myschema.t (id INT, name VARCHAR);")
        .await?;
    Ok(session)
}

// ---------------------------------------------------------------------------
// Catalog / namespace
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_database_and_schema() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = session_with_warehouse(dir.path().to_str().unwrap());

    session.sql("CREATE DATABASE mydb;").await?;
    assert!(session.get_ctx().catalog("mydb").is_some());

    session.sql("CREATE SCHEMA mydb.myschema;").await?;
    assert!(session
        .get_ctx()
        .catalog("mydb")
        .unwrap()
        .schema("myschema")
        .is_some());

    // IF NOT EXISTS is idempotent.
    session.sql("CREATE DATABASE IF NOT EXISTS mydb;").await?;
    session
        .sql("CREATE SCHEMA IF NOT EXISTS mydb.myschema;")
        .await?;
    Ok(())
}

#[tokio::test]
async fn create_database_duplicate_errors() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = session_with_warehouse(dir.path().to_str().unwrap());
    session.sql("CREATE DATABASE mydb;").await?;
    assert!(
        session.sql("CREATE DATABASE mydb;").await.is_err(),
        "duplicate CREATE DATABASE must error without IF NOT EXISTS"
    );
    Ok(())
}

#[tokio::test]
async fn create_schema_without_database_errors() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = session_with_warehouse(dir.path().to_str().unwrap());
    assert!(
        session.sql("CREATE SCHEMA nodb.noschema;").await.is_err(),
        "CREATE SCHEMA in a missing database must error"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Table lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_table_registers_at_db_schema_table() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;

    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert_eq!(table.arrow_schema().fields().len(), 2);
    assert_eq!(table.arrow_schema().field(0).name(), "id");
    assert_eq!(table.arrow_schema().field(1).name(), "name");

    let (batches, _) = session
        .sql("SELECT table_name FROM information_schema.tables WHERE table_name = 't';")
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    Ok(())
}

#[tokio::test]
async fn create_table_without_warehouse_errors() -> anyhow::Result<()> {
    let session = BenoStreamSession::new(None);
    session.sql("CREATE DATABASE mydb;").await?;
    session.sql("CREATE SCHEMA mydb.myschema;").await?;
    assert!(
        session
            .sql("CREATE TABLE mydb.myschema.t (id INT);")
            .await
            .is_err(),
        "CREATE TABLE without a warehouse or LOCATION must error"
    );
    Ok(())
}

#[tokio::test]
async fn create_table_with_explicit_location() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = BenoStreamSession::new(None);
    session.sql("CREATE DATABASE mydb;").await?;
    session.sql("CREATE SCHEMA mydb.myschema;").await?;
    let loc = dir.path().join("explicit");
    session
        .sql(&format!(
            "CREATE TABLE mydb.myschema.t (id INT) LOCATION '{}';",
            loc.to_str().unwrap()
        ))
        .await?;
    assert!(table_of(&session, "mydb", "myschema", "t").await.is_ok());
    Ok(())
}

#[tokio::test]
async fn create_table_with_format_version_option() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = session_with_warehouse(dir.path().to_str().unwrap());
    session.sql("CREATE DATABASE mydb;").await?;
    session.sql("CREATE SCHEMA mydb.myschema;").await?;
    session
        .sql("CREATE TABLE mydb.myschema.t (id INT) WITH (format_version = 3);")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert_eq!(table.get_format_version(), 3);
    Ok(())
}

#[tokio::test]
async fn drop_table_deregisters() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session.sql("DROP TABLE mydb.myschema.t;").await?;
    assert!(session
        .get_ctx()
        .table_provider(TableReference::full("mydb", "myschema", "t"))
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn drop_missing_table_errors_without_if_exists() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    assert!(session.sql("DROP TABLE mydb.myschema.nope;").await.is_err());
    session
        .sql("DROP TABLE IF EXISTS mydb.myschema.nope;")
        .await?;
    Ok(())
}

#[tokio::test]
async fn truncate_table_clears_rows() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("INSERT INTO mydb.myschema.t VALUES (1, 'a'), (2, 'b');")
        .await?;
    assert_eq!(row_count(&session, "mydb.myschema.t").await?, 2);
    session.sql("TRUNCATE TABLE mydb.myschema.t;").await?;
    assert_eq!(row_count(&session, "mydb.myschema.t").await?, 0);
    Ok(())
}

// ---------------------------------------------------------------------------
// INSERT / SELECT
// ---------------------------------------------------------------------------

#[tokio::test]
async fn insert_and_select_roundtrip() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("INSERT INTO mydb.myschema.t VALUES (1, 'Alice'), (2, 'Bob');")
        .await?;
    assert_eq!(row_count(&session, "mydb.myschema.t").await?, 2);
    Ok(())
}

// ---------------------------------------------------------------------------
// Indexes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_index_and_drop_index() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;

    session.sql("CREATE INDEX ON mydb.myschema.t (id);").await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(table.get_index_columns().contains(&"id".to_string()));

    session
        .sql("ALTER TABLE mydb.myschema.t DROP INDEX id;")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(!table.get_index_columns().contains(&"id".to_string()));
    Ok(())
}

#[tokio::test]
async fn create_composite_index() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("CREATE INDEX ON mydb.myschema.t (id, name);")
        .await?;
    // Composite indexes are registered under a synthesized column name.
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(
        !table.get_index_columns().is_empty(),
        "composite index should register at least one index column"
    );
    Ok(())
}

#[tokio::test]
async fn alter_table_add_index() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("ALTER TABLE mydb.myschema.t ADD INDEX (name);")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(table.get_index_columns().contains(&"name".to_string()));
    Ok(())
}

// ---------------------------------------------------------------------------
// Primary key
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_and_drop_primary_key() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;

    session
        .sql("ALTER TABLE mydb.myschema.t ADD PRIMARY KEY (id);")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert_eq!(table.get_primary_key(), vec!["id".to_string()]);

    session
        .sql("ALTER TABLE mydb.myschema.t DROP PRIMARY KEY;")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(table.get_primary_key().is_empty());
    Ok(())
}

#[tokio::test]
async fn add_composite_primary_key() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("ALTER TABLE mydb.myschema.t ADD PRIMARY KEY (id, name);")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert_eq!(
        table.get_primary_key(),
        vec!["id".to_string(), "name".to_string()]
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Schema evolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn schema_evolution_add_and_rename_column() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;

    session
        .sql("ALTER TABLE mydb.myschema.t ADD COLUMN age INT;")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(table.arrow_schema().field_with_name("age").is_ok());

    session
        .sql("ALTER TABLE mydb.myschema.t RENAME COLUMN name TO full_name;")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(table.arrow_schema().field_with_name("full_name").is_ok());
    assert!(table.arrow_schema().field_with_name("name").is_err());
    Ok(())
}

#[tokio::test]
async fn schema_evolution_drop_column() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("ALTER TABLE mydb.myschema.t DROP COLUMN name;")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    assert!(table.arrow_schema().field_with_name("name").is_err());
    Ok(())
}

#[tokio::test]
async fn schema_evolution_alter_column_type() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("ALTER TABLE mydb.myschema.t ALTER COLUMN id TYPE BIGINT;")
        .await?;
    let table = table_of(&session, "mydb", "myschema", "t").await?;
    let field = table.arrow_schema().field_with_name("id")?.clone();
    assert_eq!(
        field.data_type(),
        &arrow::datatypes::DataType::Int64,
        "id should be widened to Int64"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Maintenance
// ---------------------------------------------------------------------------

#[tokio::test]
async fn optimize_compact_vacuum_msck() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("INSERT INTO mydb.myschema.t VALUES (1, 'a'), (2, 'b');")
        .await?;

    session.sql("OPTIMIZE TABLE mydb.myschema.t;").await?;
    session.sql("COMPACT mydb.myschema.t;").await?;
    session.sql("VACUUM mydb.myschema.t;").await?;
    session.sql("MSCK REPAIR TABLE mydb.myschema.t;").await?;

    // Maintenance must not lose committed rows.
    assert_eq!(row_count(&session, "mydb.myschema.t").await?, 2);
    Ok(())
}

#[tokio::test]
async fn delete_from_intercepted() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("INSERT INTO mydb.myschema.t VALUES (1, 'a'), (2, 'b'), (3, 'c');")
        .await?;
    session
        .sql("DELETE FROM mydb.myschema.t WHERE id = 2;")
        .await?;
    assert_eq!(row_count(&session, "mydb.myschema.t").await?, 2);
    Ok(())
}

// ---------------------------------------------------------------------------
// EXECUTE actions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn execute_actions() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;

    for action in [
        "verify_integrity",
        "recover_indexes",
        "checkpoint",
        "rewrite_data_files",
    ] {
        session
            .sql(&format!("ALTER TABLE mydb.myschema.t EXECUTE {};", action))
            .await?;
    }
    session
        .sql("ALTER TABLE mydb.myschema.t EXECUTE remove_orphan_files(older_than_ms => 0);")
        .await?;
    session
        .sql("ALTER TABLE mydb.myschema.t EXECUTE expire_snapshots(retention => 1);")
        .await?;
    Ok(())
}

#[tokio::test]
async fn udfs_and_pgvector_syntax_work() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = session_with_warehouse(dir.path().to_str().unwrap());

    // Vector UDF callable as a plain function.
    let (batches, _) = session
        .sql("SELECT dist_l2(ARRAY[1.0,2.0,3.0], ARRAY[4.0,5.0,6.0]) AS d;")
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    // `::vector` cast is stripped by the rewriter.
    let (batches, _) = session
        .sql("SELECT dist_l2(ARRAY[1.0,2.0,3.0]::vector, ARRAY[4.0,5.0,6.0]::vector) AS d;")
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    Ok(())
}

#[tokio::test]
async fn extra_whitespace_is_tolerated() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    session
        .sql("INSERT INTO mydb.myschema.t VALUES (1, 'a');")
        .await?;

    // Extra spaces, tabs, and newlines must not break the hand-rolled parser.
    session.sql("OPTIMIZE   TABLE\n  mydb.myschema.t ;").await?;
    session.sql("COMPACT\tmydb.myschema.t;").await?;
    session
        .sql("ALTER  TABLE  mydb.myschema.t  EXECUTE  verify_integrity ;")
        .await?;
    session
        .sql("DELETE   FROM   mydb.myschema.t   WHERE   id = 1 ;")
        .await?;
    assert_eq!(row_count(&session, "mydb.myschema.t").await?, 0);
    Ok(())
}

#[tokio::test]
async fn execute_unknown_action_errors() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = setup(&dir).await?;
    assert!(session
        .sql("ALTER TABLE mydb.myschema.t EXECUTE not_a_real_action;")
        .await
        .is_err());
    Ok(())
}

// ---------------------------------------------------------------------------
// Session settings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn session_settings_are_scoped_and_allowlisted() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = session_with_warehouse(dir.path().to_str().unwrap());

    session
        .sql("SET benostream.warehouse = 's3://bucket/warehouse';")
        .await?;
    assert_eq!(
        session.get_setting("benostream.warehouse").await.as_deref(),
        Some("s3://bucket/warehouse")
    );

    let (batches, _) = session.sql("SHOW benostream.warehouse;").await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    assert!(
        session
            .sql("SET benostream.aws_secret = 'x';")
            .await
            .is_err(),
        "disallowed setting must be rejected"
    );
    Ok(())
}

#[tokio::test]
async fn set_does_not_touch_process_env() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let session = session_with_warehouse(dir.path().to_str().unwrap());
    session
        .sql("SET benostream.warehouse = 's3://bucket/warehouse';")
        .await?;
    assert!(
        std::env::var("BSDB_WAREHOUSE").is_err(),
        "SQL SET must never write process environment variables"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[tokio::test]
async fn is_handled_classification() {
    use benostreamdb::core::sql::catalog_ddl::{classify, is_handled, Handled};
    assert!(is_handled("CREATE DATABASE db;"));
    assert!(is_handled("CREATE SCHEMA db.s;"));
    assert!(is_handled("CREATE TABLE db.s.t (id INT);"));
    assert!(is_handled("CREATE INDEX ON t (id);"));
    assert!(is_handled("ALTER TABLE t ADD PRIMARY KEY (id);"));
    assert!(is_handled("OPTIMIZE TABLE t;"));
    assert!(is_handled("VACUUM t;"));
    assert!(is_handled("COMPACT t;"));
    assert!(is_handled("MSCK REPAIR TABLE t;"));
    assert!(is_handled("TRUNCATE TABLE t;"));
    assert!(is_handled("DELETE FROM t WHERE id = 1;"));
    assert!(is_handled("ALTER TABLE t EXECUTE verify_integrity;"));
    assert!(is_handled("SET benostream.warehouse = 'x';"));
    assert!(is_handled("SHOW benostream.warehouse;"));

    // SHOW has a result set; DDL does not.
    assert_eq!(classify("SHOW benostream.warehouse;"), Handled::Query);
    assert_eq!(classify("CREATE TABLE t (id INT);"), Handled::Ddl);

    // Not ours.
    assert!(!is_handled("SELECT * FROM t;"));
    assert!(!is_handled("INSERT INTO t VALUES (1);"));
    assert!(!is_handled("SET datafusion.execution.batch_size = 1;"));
    assert!(!is_handled("SHOW datafusion.execution.batch_size;"));
}
