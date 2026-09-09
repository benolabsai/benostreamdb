// Copyright (c) 2026 Richard Albright. All rights reserved.
// HyperStreamDB Enterprise Edition

/// TurboQuant (FWHT + Scalar Quantization) is part of the free community core crate.
pub mod turboquant;
pub use hyperstreamdb::core::index::turboquant::{fwht, TurboQuantEncoder};
pub use hyperstreamdb::core::index::Quantizer;

pub mod continuous_indexing {
    pub use hyperstreamdb::enterprise::continuous_indexing::*;
}

pub use hyperstreamdb::enterprise::*;
