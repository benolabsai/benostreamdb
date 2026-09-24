// Copyright (c) 2026 Richard Albright. All rights reserved.

#![deny(warnings)]
// No-panic policy for production paths (see NO_PANIC_POLICY.md). Staged behind
// the `no-panic` feature; see the root crate for the rationale. `#[cfg(test)]`
// code is exempt.
#![cfg_attr(
    all(not(test), feature = "no-panic"),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

//! BenoStreamDB Search — an OpenSearch / Elasticsearch 7.10-compatible REST
//! API served on top of the BenoStreamDB core engine.
//!
//! The server binds to `127.0.0.1:9200` by default (override with
//! `BENOSEARCH_BIND` / `BENOSEARCH_PORT`) and stores index data under
//! `BENOSEARCH_STORAGE_URI` (default `file://~/.benostreamdb/search`).

pub mod es_types;
pub mod handlers;
pub mod index_cache;
pub mod infer;
pub mod qdrant_types;
pub mod state;

/// ES 7.10 cluster version reported by `GET /` and health endpoints.
pub const ES_VERSION: &str = "7.10.2";

/// OpenSearch-compatible tagline reported by `GET /`.
pub const TAGLINE: &str = "You know, you search";
