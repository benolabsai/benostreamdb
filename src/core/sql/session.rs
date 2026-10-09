// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::core::sql::BenoStreamTableProvider;
use crate::core::table::Table;

/// DataFusion catalog name -> external (Iceberg) catalog.
type CatalogRegistry = std::sync::Arc<
    tokio::sync::RwLock<
        std::collections::HashMap<String, std::sync::Arc<dyn crate::core::catalog::Catalog>>,
    >,
>;

/// Session-scoped settings that SQL `SET benostream.<key>` may write.
///
/// Deliberately an allowlist: SQL must never be able to rewrite process
/// environment variables (credentials, endpoints) or affect other connections.
pub const ALLOWED_SESSION_SETTINGS: &[&str] = &["warehouse", "default_index_algorithm"];

#[derive(Clone)]
pub struct BenoStreamSession {
    ctx: SessionContext,
    /// External catalogs keyed by their DataFusion catalog name, so SQL DDL can
    /// mirror `CREATE DATABASE` / `CREATE SCHEMA` / `CREATE TABLE` into the
    /// instance's catalog.
    catalogs: CatalogRegistry,
    /// Base location used to derive a table URI when the catalog does not
    /// assign one (e.g. `s3://warehouse`).
    warehouse: Option<String>,
    /// Session-scoped settings (`SET benostream.<key> = <value>`). Never written
    /// to `std::env`; scoped to this session only.
    settings: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, String>>>,
}

/// A graph table discovered in a session (node/edge + endpoints).
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphTableInfo {
    pub name: String,
    pub table_type: String,
    pub source_column: Option<String>,
    pub target_column: Option<String>,
    pub id_column: Option<String>,
    pub label_column: Option<String>,
}

use datafusion::prelude::{SessionConfig, SessionContext};
// use datafusion::execution::context::SessionState; // Unused
use crate::core::sql::optimizer::IndexJoinOptimizerRule;
use crate::core::sql::udf;
use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use datafusion::execution::session_state::SessionStateBuilder;

impl BenoStreamSession {
    pub fn get_ctx(&self) -> SessionContext {
        self.ctx.clone()
    }

    pub fn new(memory_limit_bytes: Option<usize>) -> Self {
        let mut config = SessionConfig::new();
        config = config.set_str("datafusion.sql_parser.dialect", "PostgreSQL");
        config = config.with_information_schema(true);
        // Align DataFusion's unqualified-name search path with the engine's
        // unified default schema (see `catalog_ddl::DEFAULT_SCHEMA`), so DDL
        // (`CREATE TABLE t`) and DML (`INSERT INTO t`) resolve to the same
        // catalog/schema without explicit qualification.
        config = config.with_default_catalog_and_schema(
            "datafusion",
            crate::core::sql::catalog_ddl::DEFAULT_SCHEMA,
        );
        // Scale query parallelism to the effective CPU budget (respects cgroup
        // limits) instead of leaving DataFusion's default or hard-coding it.
        config = config.with_target_partitions(crate::core::sql::effective_target_partitions());

        // Session creation is on the request path for the Python/FFI bindings
        // (`PySession::new`), so a failure here must degrade rather than panic.
        let runtime = {
            let builder = RuntimeEnvBuilder::new();
            // Default the query memory limit from the effective memory so SQL
            // sorts/joins/aggregations spill to disk instead of OOMing. A caller
            // that passes an explicit limit (e.g. `PySession`) still wins;
            // `BSDB_DATAFUSION_MEMORY_GB` overrides the derived default.
            let limit = memory_limit_bytes.unwrap_or_else(|| {
                crate::core::resources::default_datafusion_memory_bytes() as usize
            });
            let builder = builder.with_memory_limit(limit, 1.0);
            match builder.build() {
                Ok(rt) => Arc::new(rt),
                Err(e) => {
                    // Only reachable with an invalid memory limit; a default
                    // runtime keeps queries working instead of aborting them.
                    tracing::error!(
                        error = %e,
                        "failed to build DataFusion RuntimeEnv with the requested memory limit; using defaults"
                    );
                    Arc::new(RuntimeEnv::default())
                }
            }
        };

        let state_builder = SessionStateBuilder::new()
            .with_config(config)
            .with_runtime_env(runtime)
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(IndexJoinOptimizerRule::default()))
            .with_physical_optimizer_rule(Arc::new(
                crate::core::sql::optimizer::VectorSearchOptimizerRule::default(),
            ));

        let state = state_builder.build();
        let mut ctx = SessionContext::new_with_state(state);

        // Registration only fails on a duplicate definition, which cannot happen
        // in a fresh context; log rather than abort the query if it ever does.
        if let Err(e) = datafusion_functions::register_all(&mut ctx) {
            tracing::error!(error = %e, "failed to register standard DataFusion functions");
        }
        if let Err(e) = datafusion_functions_aggregate::register_all(&mut ctx) {
            tracing::error!(error = %e, "failed to register standard DataFusion aggregates");
        }

        // Register the full custom function surface (vector scalar UDFs, JSON
        // path functions, vector aggregates, and graph UDAFs) from the single
        // source of truth, so this session matches the ad-hoc `Table::sql()`
        // context and the connectors.
        udf::register_all_custom_udfs(&mut ctx);

        // Register graph traversal table functions (`FROM graph_neighbors(...)`).
        crate::core::sql::graph_udf::register_graph_table_functions(&mut ctx);
        crate::core::sql::subscribe_table_function::register_subscribe_table_functions(&mut ctx);

        // Register vector operators (validates UDFs are present)
        if let Err(e) = crate::core::sql::vector_operators::register_vector_operators(&mut ctx) {
            tracing::error!(error = %e, "failed to register vector operators");
        }

        Self {
            ctx,
            catalogs: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            warehouse: None,
            settings: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    /// Set a session-scoped setting. Returns an error for keys outside the
    /// allowlist so SQL cannot mutate process-wide state.
    pub async fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let key = key.strip_prefix("benostream.").unwrap_or(key);
        if !ALLOWED_SESSION_SETTINGS.contains(&key) {
            anyhow::bail!(
                "unknown or disallowed session setting 'benostream.{}'; allowed: {}",
                key,
                ALLOWED_SESSION_SETTINGS.join(", ")
            );
        }
        self.settings
            .write()
            .await
            .insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Read a session-scoped setting.
    pub async fn get_setting(&self, key: &str) -> Option<String> {
        let key = key.strip_prefix("benostream.").unwrap_or(key);
        self.settings.read().await.get(key).cloned()
    }

    /// Snapshot of all session-scoped settings.
    pub async fn settings_snapshot(&self) -> std::collections::HashMap<String, String> {
        self.settings.read().await.clone()
    }

    /// Attach an external catalog under a DataFusion catalog name.
    pub async fn register_catalog(
        &self,
        name: &str,
        catalog: std::sync::Arc<dyn crate::core::catalog::Catalog>,
    ) {
        self.catalogs
            .write()
            .await
            .insert(name.to_string(), catalog);
    }

    /// Look up the external catalog bound to a DataFusion catalog name.
    pub async fn catalog_for(
        &self,
        name: &str,
    ) -> Option<std::sync::Arc<dyn crate::core::catalog::Catalog>> {
        self.catalogs.read().await.get(name).cloned()
    }

    /// Set the warehouse base location used to derive table URIs.
    pub fn set_warehouse(&mut self, warehouse: Option<String>) {
        self.warehouse = warehouse;
    }

    /// The configured warehouse base location, if any.
    pub fn warehouse(&self) -> Option<&str> {
        self.warehouse.as_deref()
    }

    pub fn register_table(&self, name: &str, table: Arc<Table>) -> Result<()> {
        let provider = Arc::new(BenoStreamTableProvider::new(table));
        self.ctx.register_table(name, provider)?;
        Ok(())
    }

    /// Returns `true` if `query` is a DDL statement (no result set).
    pub async fn is_ddl(&self, query: &str) -> Result<bool> {
        if crate::core::sql::catalog_ddl::classify(query)
            == crate::core::sql::catalog_ddl::Handled::Ddl
        {
            return Ok(true);
        }
        let query_processed = crate::core::sql::pgvector_rewriter::rewrite_sql_string(query);
        let query_processed =
            crate::core::sql::partition_rewriter::strip_partitioned_by(&query_processed);
        let plan = self
            .ctx
            .state()
            .create_logical_plan(&query_processed)
            .await?;
        Ok(matches!(
            plan,
            datafusion::logical_expr::LogicalPlan::Ddl(_)
        ))
    }

    /// Returns `true` if `query` is a DML statement (`INSERT`/`UPDATE`/`DELETE`).
    ///
    /// The Flight SQL gateway must execute these inside `GetFlightInfo` because
    /// ADBC cancels the follow-up `DoGet` for DML, so the statement would
    /// otherwise never run.
    pub async fn is_dml(&self, query: &str) -> Result<bool> {
        let query_processed = crate::core::sql::pgvector_rewriter::rewrite_sql_string(query);
        let query_processed =
            crate::core::sql::partition_rewriter::strip_partitioned_by(&query_processed);
        let plan = self
            .ctx
            .state()
            .create_logical_plan(&query_processed)
            .await?;
        Ok(matches!(
            plan,
            datafusion::logical_expr::LogicalPlan::Dml(_)
        ))
    }

    /// List the graph tables (node/edge) registered in the session.
    ///
    /// The discovery primitive an MCP agent uses to find the graph without being
    /// told the column names. Returns each table's declared type and endpoints.
    pub async fn list_graph_tables(&self) -> Result<Vec<GraphTableInfo>> {
        let mut out = Vec::new();
        let list = self.ctx.state().catalog_list().clone();
        for cat_name in list.catalog_names() {
            let Some(cat) = list.catalog(&cat_name) else {
                continue;
            };
            for sch_name in cat.schema_names() {
                let Some(sch) = cat.schema(&sch_name) else {
                    continue;
                };
                for tbl_name in sch.table_names() {
                    let Ok(Some(provider)) = sch.table(&tbl_name).await else {
                        continue;
                    };
                    let Some(bs) = provider.as_any().downcast_ref::<BenoStreamTableProvider>()
                    else {
                        continue;
                    };
                    let meta = bs.table.graph_metadata_async().await.unwrap_or_default();
                    if meta.is_edge() || meta.is_node() {
                        out.push(GraphTableInfo {
                            name: format!("{cat_name}.{sch_name}.{tbl_name}"),
                            table_type: meta.table_type.as_str().to_string(),
                            source_column: meta.source_column,
                            target_column: meta.target_column,
                            id_column: meta.id_column,
                            label_column: meta.label_column,
                        });
                    }
                }
            }
        }
        Ok(out)
    }

    /// Return the primary-key column names for a registered table, if any.
    ///
    /// Used by the Flight SQL `GetPrimaryKeys` metadata endpoint.
    pub async fn get_primary_keys(&self, catalog: &str, schema: &str, table: &str) -> Vec<String> {
        let Some(cat) = self.ctx.catalog(catalog) else {
            return Vec::new();
        };
        let Some(sch) = cat.schema(schema) else {
            return Vec::new();
        };
        let Ok(Some(provider)) = sch.table(table).await else {
            return Vec::new();
        };
        provider
            .as_any()
            .downcast_ref::<BenoStreamTableProvider>()
            .map(|p| p.table.get_primary_key())
            .unwrap_or_default()
    }

    pub async fn sql_to_df(&self, query: &str) -> Result<datafusion::dataframe::DataFrame> {
        // Pre-process string to handle pgvector syntax not supported by DataFusion parser natively
        let query_processed = crate::core::sql::pgvector_rewriter::rewrite_sql_string(query);
        // Strip PARTITIONED BY to bypass DataFusion's lack of Hive distribution support on memory tables
        let query_processed =
            crate::core::sql::partition_rewriter::strip_partitioned_by(&query_processed);

        // Catalog DDL / maintenance: DataFusion has no logical plan for these,
        // so intercept the parsed statement before planning and dispatch to the
        // core `Table` API.
        if let Some(batch) =
            crate::core::sql::catalog_ddl::try_parse_and_execute(self, &query_processed).await?
        {
            return self.ctx.read_batch(batch).map_err(Into::into);
        }

        // Native MERGE INTO: DataFusion has no logical plan for it, so intercept
        // the parsed statement before planning and execute it via the key-based
        // merge primitive.
        if let Some(batch) =
            crate::core::sql::merge_into::try_parse_and_execute(&self.ctx, &query_processed).await?
        {
            return self.ctx.read_batch(batch).map_err(Into::into);
        }

        // Parse the SQL query to get a logical plan
        let plan = self
            .ctx
            .state()
            .create_logical_plan(&query_processed)
            .await?;

        // Rewrite the plan to convert pgvector syntax to UDF calls
        let rewritten_plan = crate::core::sql::pgvector_rewriter::rewrite_pgvector_plan(plan)?;

        // Execute the rewritten logical plan
        let df = self.ctx.execute_logical_plan(rewritten_plan).await?;
        Ok(df)
    }

    pub async fn get_schema(&self, query: &str) -> Result<arrow::datatypes::SchemaRef> {
        if let Some(schema) = crate::core::sql::catalog_ddl::result_schema(query) {
            return Ok(schema);
        }
        if crate::core::sql::catalog_ddl::classify(query)
            == crate::core::sql::catalog_ddl::Handled::Ddl
        {
            return Ok(std::sync::Arc::new(arrow::datatypes::Schema::empty()));
        }
        let query_processed = crate::core::sql::pgvector_rewriter::rewrite_sql_string(query);
        let query_processed =
            crate::core::sql::partition_rewriter::strip_partitioned_by(&query_processed);
        let plan = self
            .ctx
            .state()
            .create_logical_plan(&query_processed)
            .await?;
        let rewritten_plan = crate::core::sql::pgvector_rewriter::rewrite_pgvector_plan(plan)?;
        if matches!(
            rewritten_plan,
            datafusion::logical_expr::LogicalPlan::Ddl(_)
        ) {
            return Ok(std::sync::Arc::new(arrow::datatypes::Schema::empty()));
        }
        // We can create a DataFrame without executing the plan to get the schema
        let df = datafusion::dataframe::DataFrame::new(self.ctx.state(), rewritten_plan);
        Ok(std::sync::Arc::new(df.schema().as_arrow().clone()))
    }

    pub async fn sql(
        &self,
        query: &str,
    ) -> Result<(Vec<RecordBatch>, arrow::datatypes::SchemaRef)> {
        let df = self.sql_to_df(query).await?;
        let schema: arrow::datatypes::SchemaRef =
            std::sync::Arc::new(df.schema().as_arrow().clone());
        let batches = df.collect().await?;
        Ok((batches, schema))
    }
}

impl Default for BenoStreamSession {
    fn default() -> Self {
        Self::new(None)
    }
}
