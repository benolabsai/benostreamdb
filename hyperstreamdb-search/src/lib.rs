// Copyright (c) 2026 Richard Albright. All rights reserved.

#![deny(warnings)]

//! HyperStreamDB Search — an OpenSearch / Elasticsearch 7.10-compatible REST
//! API served on top of the HyperStreamDB core engine.
//!
//! The server binds to `127.0.0.1:9200` by default (override with
//! `HYPERSEARCH_BIND` / `HYPERSEARCH_PORT`) and stores index data under
//! `HYPERSEARCH_STORAGE_URI` (default `file://~/.hyperstreamdb/search`).

pub mod es_types;
pub mod handlers;
pub mod state;

/// ES 7.10 cluster version reported by `GET /` and health endpoints.
pub const ES_VERSION: &str = "7.10.2";

/// OpenSearch-compatible tagline reported by `GET /`.
pub const TAGLINE: &str = "You know, you search";
