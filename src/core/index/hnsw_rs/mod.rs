// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Internalized third-party HNSW implementation.
//!
//! Vendored from `hnsw_rs` (dual MIT / Apache-2.0 — see `LICENSE-MIT` and
//! `LICENSE-APACHE` in this directory), which is no longer a crates.io
//! dependency.
//!
//! # No-panic policy
//!
//! This subtree is deliberately excluded from the production no-panic gate.
//! The `unwrap()`/`expect()` calls here assert graph invariants that hold by
//! construction inside the builder and the search loop — an entry point exists
//! once a layer has a point, a candidate heap is non-empty while the loop runs,
//! a point id resolves in the index that produced it. Threading `Result` through
//! them would put a branch inside the innermost distance comparison, which is
//! the hot path of every vector search.
//!
//! The one place a *user* action can trip them is the file I/O in `hnswio`
//! (dump/load of an index), where a truncated or corrupt file could fail a
//! `read_exact`. Converting those paths to `Result` is tracked as the remaining
//! item for this module in `NO_PANIC_POLICY.md`; it is not done here because it
//! changes the signatures of the upstream load/dump entry points.
#![allow(clippy::empty_docs)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//#![feature(portable_simd)]
// prededing line to uncomment to get std::simd by using
// packed_simd_2 = { version = "0.3", optional = true}
// and compile with cargo [test|build] --features "stdsimd" ...

// for logging (debug mostly, switched at compile time in cargo.toml)
use lazy_static::lazy_static;

pub mod dist;
pub mod hnsw;
pub use dist::*;
pub mod api;
pub mod flatten;
pub mod hnswio;
pub mod prelude;

lazy_static! {
    static ref LOG: u64 = init_log();
}

// install a logger facility
#[allow(dead_code)]
fn init_log() -> u64 {
    let mut builder = env_logger::Builder::from_default_env();
    let _ = builder.try_init();
    1
}
pub mod arrow_hnsw;
pub mod arrow_ipc;
