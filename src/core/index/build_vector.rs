// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Vector-index build for a flushed segment.
//!
//! A declared vector index must be built for every vector column, regardless of
//! whether the column arrived as `Float32` or `Float64`. `Float64` is narrowed
//! to the engine's canonical `Float32` element type. Previously any inner type
//! other than `Float32` caused a **silent** no-op: a table declared with an
//! index on an empty schema would store `list<double>`, never build the index,
//! and silently fall back to a brute-force scan at query time.

use anyhow::{Context, Result};
use arrow::array::{Array, FixedSizeListArray, Float32Array, Float64Array, ListArray};
use arrow::datatypes::DataType;
use rayon::prelude::*;
use std::sync::Arc;

/// Extract one vector row as `f32`.
///
/// Accepts `Float32` (copied) and `Float64` (narrowed); any other inner type
/// yields an empty vector, which the caller treats as "not a vector column".
fn row_to_f32(item: &dyn Array) -> Vec<f32> {
    if let Some(a) = item.as_any().downcast_ref::<Float32Array>() {
        a.values().to_vec()
    } else if let Some(a) = item.as_any().downcast_ref::<Float64Array>() {
        a.values().iter().map(|&x| x as f32).collect()
    } else {
        Vec::new()
    }
}

impl crate::core::segment::HybridSegmentWriter {
    pub(crate) fn build_vector_index(
        &self,
        col_name: &str,
        col_array: &Arc<dyn Array>,
        _row_offset: usize,
        local_staging_dir: &std::path::Path,
    ) -> Result<()> {
        let _config = self.index_configs.get(col_name);

        let inner_type = match col_array.data_type() {
            DataType::List(inner) => inner.data_type().clone(),
            DataType::FixedSizeList(inner, _) => inner.data_type().clone(),
            _ => return Ok(()),
        };

        if inner_type != DataType::Float32 && inner_type != DataType::Float64 {
            return Ok(());
        }
        if inner_type == DataType::Float64 {
            tracing::warn!(
                "vector column '{}' has Float64 elements; indexing as Float32. \
                 Declare vector columns as float32 to avoid the per-build conversion.",
                col_name
            );
        }

        tracing::info!(
            "Indexing Vector column: {} (type={:?})",
            col_name,
            col_array.data_type()
        );

        let vectors: Vec<Vec<f32>> = match col_array.data_type() {
            DataType::FixedSizeList(_, _) => {
                let list_array = col_array
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .context("Invalid cast")?;
                (0..list_array.len())
                    .into_par_iter()
                    .map(|i| row_to_f32(list_array.value(i).as_ref()))
                    .collect()
            }
            DataType::List(_) => {
                let list_array = col_array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .context("Invalid cast")?;
                (0..list_array.len())
                    .into_par_iter()
                    .map(|i| row_to_f32(list_array.value(i).as_ref()))
                    .collect()
            }
            _ => unreachable!(),
        };

        if vectors.is_empty() {
            return Ok(());
        }
        let _dim = vectors[0].len();

        // Build vector index ONLY if configured for immediate indexing
        let in_config = self
            .config
            .columns_to_index
            .as_ref()
            .map(|cols| cols.iter().any(|c| c == col_name))
            .unwrap_or(false);
        if self.config.index_all || in_config {
            let tmp_path = local_staging_dir.join(format!(
                "{}.{}.tmp.vec.bin",
                self.config.segment_id, col_name
            ));

            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&tmp_path)
                .context("Failed to open vector temp file")?;

            use std::io::Write;
            let dim = vectors[0].len() as u32;
            // Self-describing header, written once when the file is created.
            // The out-of-core builder derives the vector count from the file
            // length, so it no longer has to re-read the whole file just to
            // count vectors (a multi-GB read per segment).
            if file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
                file.write_all(&crate::core::index::hnsw_ivf::VEC_TMP_MAGIC.to_le_bytes())?;
                file.write_all(&dim.to_le_bytes())?;
            }
            for (i, vec) in vectors.iter().enumerate() {
                let global_row_id = (_row_offset + i) as u32;
                file.write_all(&global_row_id.to_le_bytes())?;
                file.write_all(&dim.to_le_bytes())?;
                let vec_bytes = bytemuck::cast_slice(vec);
                file.write_all(vec_bytes)?;
            }

            {
                let tmp_path_str = tmp_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("temporary vector path is not valid UTF-8"))?;
                let mut v_data = self.vector_data.lock();
                v_data.insert(col_name.to_string(), tmp_path_str.to_string());
            }
        } else {
            tracing::info!(
                "Skipping vector indexing for column {} (delayed/background mode)",
                col_name
            );
        }

        Ok(())
    }
}
