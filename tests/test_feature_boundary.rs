// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Locks down the feature boundary between the supported CPU-only core and the
//! experimental / deployment surfaces.
//!
//! The default build must stay CPU-only and free of the optional surfaces
//! (Python, JVM, GPU backends, metrics exporter). Those surfaces are **opt-in
//! feature flags only**. This test fails if an experimental feature leaks into
//! `default`, or if a documented feature is renamed/removed — keeping
//! `docs/SUPPORT_MATRIX.md` honest rather than aspirational.
//!
//! If this fails because a change is *intended*, update this test and
//! `docs/SUPPORT_MATRIX.md` together.

use std::fs;

#[test]
fn default_features_stay_cpu_only() {
    let manifest = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("read Cargo.toml");

    let default_line = manifest
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("default ="))
        .expect("[features] must declare a `default` list");

    assert_eq!(
        default_line, "default = []",
        "the default feature set must stay empty (CPU-only); experimental and \
         deployment surfaces must be opt-in feature flags"
    );

    // Every experimental / deployment surface must be declared as an opt-in
    // feature (so it cannot silently become a default dependency).
    for feat in [
        "python",
        "java",
        "cuda",
        "wgpu",
        "observability",
        "no-panic",
        "proptest",
    ] {
        let needle = format!("\n{feat} = [");
        assert!(
            manifest.contains(&needle),
            "expected opt-in feature '{feat}' to be declared in [features]; \
             if it was renamed, update this test and docs/SUPPORT_MATRIX.md"
        );
    }
}
