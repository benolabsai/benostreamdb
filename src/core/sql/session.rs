// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::core::sql::HyperStreamTableProvider;
use crate::core::table::Table;

#[derive(Clone)]
pub struct HyperStreamSession {
    ctx: SessionContext,
}

use datafusion::prelude::{SessionConfig, SessionContext};
// use datafusion::execution::context::SessionState; // Unused
use crate::core::sql::optimizer::IndexJoinOptimizerRule;
use crate::core::sql::vector_udf;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;

impl HyperStreamSession {
    pub fn get_ctx(&self) -> SessionContext {
        self.ctx.clone()
    }

    pub fn new(memory_limit_bytes: Option<usize>) -> Self {
        let mut config = SessionConfig::new();
        config = config.set_str("datafusion.sql_parser.dialect", "PostgreSQL");
        config = config.with_information_schema(true);

        let runtime = Arc::new({
            let builder = RuntimeEnvBuilder::new();
            let builder = if let Some(limit) = memory_limit_bytes {
                builder.with_memory_limit(limit, 1.0)
            } else {
                builder
            };
            builder.build().expect("Failed to build RuntimeEnv")
        });

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

        // Register standard functions (now that we've added the crates to Cargo.toml)
        datafusion_functions::register_all(&mut ctx)
            .expect("Failed to register standard functions");
        datafusion_functions_aggregate::register_all(&mut ctx)
            .expect("Failed to register standard aggregates");

        // Add Vector Scalar Functions (Additive registration)
        for udf in vector_udf::all_vector_udfs() {
            ctx.register_udf(udf);
        }

        // Add Vector Aggregate Functions (Additive registration via register_udaf in DF 52)
        for udf in vector_udf::all_vector_aggregates() {
            ctx.register_udaf(udf);
        }

        // Register vector operators (validates UDFs are present)
        crate::core::sql::vector_operators::register_vector_operators(&mut ctx)
            .expect("Failed to register vector operators");

        Self { ctx }
    }

    pub fn register_table(&self, name: &str, table: Arc<Table>) -> Result<()> {
        let provider = Arc::new(HyperStreamTableProvider::new(table));
        self.ctx.register_table(name, provider)?;
        Ok(())
    }

    pub async fn sql_to_df(&self, query: &str) -> Result<datafusion::dataframe::DataFrame> {
        // Pre-process string to handle pgvector syntax not supported by DataFusion parser natively
        let query_processed = crate::core::sql::pgvector_rewriter::rewrite_sql_string(query);
        // Strip PARTITIONED BY to bypass DataFusion's lack of Hive distribution support on memory tables
        let query_processed =
            crate::core::sql::partition_rewriter::strip_partitioned_by(&query_processed);

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

impl Default for HyperStreamSession {
    fn default() -> Self {
        Self::new(None)
    }
}
