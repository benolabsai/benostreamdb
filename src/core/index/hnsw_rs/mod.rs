//#![feature(portable_simd)]
// prededing line to uncomment to get std::simd by using
// packed_simd_2 = { version = "0.3", optional = true}
// and compile with cargo [test|build] --features "stdsimd" ...

// for logging (debug mostly, switched at compile time in cargo.toml)
use lazy_static::lazy_static;

pub mod hnsw;
pub mod dist;
pub mod hnswio;
pub mod prelude;
pub mod api;
pub mod libext;
pub mod flatten;

lazy_static! {
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
#[allow(dead_code)]
fn init_log() -> u64 {
    let mut builder = env_logger::Builder::from_default_env();
    let _ = builder.try_init();
    return 1;
}
