// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

pub mod algorithms;
pub mod cache;
pub mod catalog;
pub mod clustering;
pub mod compaction;
pub mod error;
#[cfg(feature = "java")]
pub mod ffi;
pub mod iceberg;
pub mod index;
pub mod jni_util;
pub mod maintenance;
pub mod manifest;
pub mod memory;
pub mod merge;
pub mod metadata;
pub mod nessie;
pub mod planner;
pub mod puffin;
pub mod query;
pub mod reader;
pub mod resources;
pub mod segment;
pub mod sql;
pub mod storage;
pub mod table;
pub mod wal;
// pub mod parquet_filter;
pub mod auth;
pub mod embeddings;
pub mod fault_injection;
pub mod lock;
pub mod search;
pub mod telemetry;

/// Run `f` on the current thread, using `tokio::task::block_in_place` only when
/// the ambient Tokio runtime is multi-threaded. `block_in_place` panics on a
/// current-thread runtime, so callers that may run under either flavor must use
/// this instead of calling it directly.
pub(crate) fn run_blocking<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}
