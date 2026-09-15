
use anyhow::{Context, Result};
use arrow::array::Array;
use rayon::prelude::*;
use std::sync::Arc;

impl crate::core::segment::HybridSegmentWriter {
    pub(crate) fn build_vector_index(
        &self,
        col_name: &str,
        col_array: &Arc<dyn Array>,
        _row_offset: usize,
        local_staging_dir: &std::path::Path,
    ) -> Result<()> {
        let _config = self.index_configs.get(col_name);
        let inner = match col_array.data_type() {
            arrow::datatypes::DataType::List(inner) => inner,
            arrow::datatypes::DataType::FixedSizeList(inner, _) => inner,
            _ => return Ok(()),
        };
        if *inner.data_type() == arrow::datatypes::DataType::Float32 {
            tracing::info!(
                "Indexing Vector column: {} (type={:?})",
                col_name,
                col_array.data_type()
            );

            let vectors: Vec<Vec<f32>> = match col_array.data_type() {
                arrow::datatypes::DataType::FixedSizeList(_, _) => {
                    let list_array = col_array
                        .as_any()
                        .downcast_ref::<arrow::array::FixedSizeListArray>()
                        .context("Invalid cast")?;
                    (0..list_array.len())
                        .into_par_iter()
                        .map(|i| {
                            let item = list_array.value(i);
                            let Some(float_array) =
                                item.as_any().downcast_ref::<arrow::array::Float32Array>()
                            else {
                                return vec![];
                            };
                            float_array.values().to_vec()
                        })
                        .collect()
                }
                arrow::datatypes::DataType::List(_) => {
                    let list_array = col_array
                        .as_any()
                        .downcast_ref::<arrow::array::ListArray>()
                        .context("Invalid cast")?;
                    (0..list_array.len())
                        .into_par_iter()
                        .map(|i| {
                            let item = list_array.value(i);
                            let Some(float_array) =
                                item.as_any().downcast_ref::<arrow::array::Float32Array>()
                            else {
                                return vec![];
                            };
                            float_array.values().to_vec()
                        })
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
                let tmp_path = local_staging_dir.join(format!("{}.{}.tmp.vec.bin", self.config.segment_id, col_name));
                
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&tmp_path)
                    .context("Failed to open vector temp file")?;
                    
                use std::io::Write;
                let dim = vectors[0].len() as u32;
                for (i, vec) in vectors.iter().enumerate() {
                    let global_row_id = (_row_offset + i) as u32;
                    file.write_all(&global_row_id.to_le_bytes())?;
                    file.write_all(&dim.to_le_bytes())?;
                    let vec_bytes = bytemuck::cast_slice(vec);
                    file.write_all(vec_bytes)?;
                }
                
                {
                    let mut v_data = self.vector_data.lock();
                    v_data.insert(col_name.to_string(), tmp_path.to_str().unwrap().to_string());
                }
            } else {
                tracing::info!(
                    "Skipping vector indexing for column {} (delayed/background mode)",
                    col_name
                );
            }
        }
        Ok(())
    }
}
