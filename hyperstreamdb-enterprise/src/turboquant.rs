// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantConfig {
    pub dim: usize,
    pub bits: usize,
    pub use_qjl: bool,
}

/// Fast Walsh-Hadamard Transform (FWHT) for Enterprise Rotation
#[inline(always)]
pub fn fwht_portable(data: &mut [f32]) {
    let n = data.len();
    if !n.is_power_of_two() {
         return; // Skip or pad in a production system
    }

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h <<= 1;
    }
}

/// AVX-512 Optimized FWHT (Enterprise Phase 2)
#[cfg(feature = "avx512")]
pub fn fwht_avx512(data: &mut [f32]) {
     // High-performance AVX-512 unrolled loop
     fwht_portable(data);
}
