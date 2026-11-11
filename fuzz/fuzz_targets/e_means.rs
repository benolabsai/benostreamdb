// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Fuzz the e-means coarse centroid quantizer against degenerate, singular,
//! or adversarial vector inputs (NaNs, Infs, extreme norms, all zeros, identical vectors).
//!
//! Must never panic, divide by zero, or crash regardless of input bytes.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    // Extract dim (3..=32) and k (2..=16) from the first bytes
    let dim = (data[0] as usize % 30) + 3;
    let k = (data[1] as usize % 15) + 2;
    let raw_payload = &data[2..];

    // Ensure 4-byte chunks for f32 slices
    let f32_count = raw_payload.len() / 4;
    let n = f32_count / dim;
    if n < k || n < 8 {
        return;
    }

    let mut vectors = Vec::with_capacity(n * dim);
    for chunk in raw_payload[..n * dim * 4].chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        vectors.push(val);
    }

    let config = benostreamdb::core::index::e_means::EMeansConfig {
        k,
        nepochs: 2,
        batch_law: benostreamdb::core::index::e_means::BatchLaw::Fixed(32),
        fixed_alpha0: Some(0.5),
        unused_threshold: Some(0.1),
        seed: 42,
    };

    // Must never panic, even if vectors contain NaNs, infinities, or extreme floats
    let _ = benostreamdb::core::index::e_means::train_emeans(&vectors, dim, &config);
});
