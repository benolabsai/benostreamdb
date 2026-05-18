// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Query optimizer rules for HyperStreamDB.
//!
//! Provides physical optimizer rules for DataFusion that detect and rewrite
//! common patterns like vector search KNN queries and index joins.

mod config;
mod index_join;
mod vector_search;

pub use config::*;
pub use index_join::*;
pub use vector_search::*;
