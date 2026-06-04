use crate::core::index::hnsw_ivf::HnswIvfIndex;
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
                let mut algos = self
                    .index_configs
                    .get(col_name)
                    .map(|c| c.algorithms.clone())
                    .unwrap_or_else(|| {
                        self.config
                            .column_algorithms
                            .get(col_name)
                            .cloned()
                            .unwrap_or_default()
                    });

                // If it's a vector column but no specific algorithms were provided, use the global default (TurboQuant 8-bit)
                if algos.is_empty() {
                    tracing::info!(
                        "No index algorithm specified for {}; defaulting to hnsw_tq8",
                        col_name
                    );
                    algos.push(crate::core::manifest::IndexAlgorithm::default());
                }

                for (idx, algo) in algos.iter().enumerate() {
                    let hnsw_ivf_index = HnswIvfIndex::build(
                        vectors.clone(),
                        crate::core::index::VectorMetric::L2,
                        None,
                        None,
                        algo,
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW-IVF build failed: {}", e))?;

                    let algo_id = match algo {
                        crate::core::manifest::IndexAlgorithm::Hnsw { .. } => "hnsw",
                        crate::core::manifest::IndexAlgorithm::HnswPq { .. } => "pq",
                        crate::core::manifest::IndexAlgorithm::HnswTq4 { .. } => "tq4",
                        crate::core::manifest::IndexAlgorithm::HnswTq8 { .. } => "tq8",
                        _ => "idx",
                    };

                    let suffix = if algos.len() > 1 {
                        format!("{}.{}.{}", col_name, algo_id, idx)
                    } else {
                        format!("{}.{}", col_name, algo_id)
                    };
                    let local_base_path =
                        local_staging_dir.join(format!("{}.{}", self.config.segment_id, suffix));

                    let saved_files = hnsw_ivf_index
                        .save(local_base_path.to_str().context("Invalid UTF-8 in path")?)
                        .map_err(|e| anyhow::anyhow!("HNSW-IVF save failed: {}", e))?;

                    {
                        let mut meta = self.index_metadata.lock();
                        meta.insert(
                            format!("{}.{}", self.config.segment_id, suffix),
                            algo.to_string(),
                        );
                    }

                    {
                        let mut files = self.generated_files.lock();
                        files.extend(saved_files);
                    }
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
