use std::sync::Arc;
use arrow::record_batch::RecordBatch;
use anyhow::Result;

use crate::core::table::Table;
use crate::core::sql::HyperStreamTableProvider;

#[derive(Clone)]
pub struct HyperStreamSession {
    ctx: SessionContext,
}

use datafusion::prelude::{SessionConfig, SessionContext};
// use datafusion::execution::context::SessionState; // Unused
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::session_state::SessionStateBuilder;
use crate::core::sql::optimizer::IndexJoinOptimizerRule;
use crate::core::sql::vector_udf;

impl HyperStreamSession {
    pub fn new(_memory_limit_bytes: Option<usize>) -> Self {
        let mut config = SessionConfig::new();
        config = config.set_str("datafusion.sql_parser.dialect", "PostgreSQL");
        
        let runtime = Arc::new(RuntimeEnv::default());
        
        let mut state_builder = SessionStateBuilder::new()
            .with_config(config)
            .with_runtime_env(runtime)
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(IndexJoinOptimizerRule::default()))
            .with_physical_optimizer_rule(Arc::new(crate::core::sql::optimizer::VectorSearchOptimizerRule::default()));
            
        // Register Vector UDFs
        let udfs = vector_udf::all_vector_udfs();
        state_builder = state_builder.with_scalar_functions(
            udfs.into_iter().map(Arc::new).collect()
        );
        
        // Register Vector Aggregates
        let aggregates = vector_udf::all_vector_aggregates();
        state_builder = state_builder.with_aggregate_functions(
            aggregates.into_iter().map(Arc::new).collect()
        );

        let state = state_builder.build();
        
        let mut ctx = SessionContext::new_with_state(state);
        
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

    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        // Parse the SQL query and execute directly
        // Note: pgvector rewriter temporarily disabled due to schema mismatch issues
        let df = self.ctx.sql(query).await?;
        let batches = df.collect().await?;
        Ok(batches)
    }
}

impl Default for HyperStreamSession {
    fn default() -> Self {
        Self::new(None)
    }
}
