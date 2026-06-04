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
pub mod libext;
pub mod prelude;

lazy_static! {
    static ref LOG: u64 = { init_log() };
}

// install a logger facility
#[allow(dead_code)]
fn init_log() -> u64 {
    let mut builder = env_logger::Builder::from_default_env();
    let _ = builder.try_init();
    1
}
