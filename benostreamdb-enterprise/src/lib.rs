// Copyright (c) 2026 Richard Albright. All rights reserved.
// BenoStreamDB Enterprise Edition

/// TurboQuant (FWHT + Scalar Quantization) is part of the free community core crate.
pub mod turboquant;
pub use benostreamdb::core::index::turboquant::{fwht, TurboQuantEncoder};
pub use benostreamdb::core::index::Quantizer;

pub mod continuous_indexing {
    pub use benostreamdb::enterprise::continuous_indexing::*;
}

pub use benostreamdb::enterprise::*;
