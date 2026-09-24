// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod continuous_indexing;

pub use crate::core::license::{verify_license as validate_license, LicensePayload as License};
pub use continuous_indexing::ContinuousIndexBuilder;
