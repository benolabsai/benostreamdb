// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use crate::core::planner::VectorSearchParams;
use crate::core::index::gpu::{set_global_gpu_context, ComputeContext};

use super::Table;

pub struct TableQuery<'a> {
    pub table: &'a Table,
    pub filter_str: Option<String>,
    pub vector_filter: Option<VectorSearchParams>,
    pub columns: Option<Vec<String>>,
    pub context: Option<ComputeContext>,
}

impl<'a> TableQuery<'a> {
    pub fn new(table: &'a Table) -> Self {
        Self {
            table,
            filter_str: None,
            vector_filter: None,
            columns: None,
            context: None,
        }
    }

    pub fn filter(mut self, expr: &str) -> Self {
        if let Some(ref mut f) = self.filter_str {
            *f = format!("({}) AND ({})", f, expr);
        } else {
            self.filter_str = Some(expr.to_string());
        }
        self
    }

    pub fn vector_search(mut self, column: &str, query: crate::core::index::VectorValue, k: usize) -> Self {
        self.vector_filter = Some(VectorSearchParams::new(column, query, k));
        self
    }

    pub fn select(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    pub fn with_context(mut self, context: ComputeContext) -> Self {
        self.context = Some(context);
        self
    }

    pub async fn to_batches(self) -> Result<Vec<RecordBatch>> {
        let cols_refs: Option<Vec<&str>> = self.columns.as_ref().map(|c| c.iter().map(|s| s.as_str()).collect());
        let cols_slice: Option<&[&str]> = cols_refs.as_deref();
        
        // Inject context if provided
        if let Some(ctx) = self.context {
            set_global_gpu_context(Some(ctx));
        }
        
        self.table.read_async(self.filter_str.as_deref(), self.vector_filter, cols_slice).await
    }
}
