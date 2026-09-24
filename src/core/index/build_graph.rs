use anyhow::{Context, Result};
use arrow::array::Array;

impl crate::core::segment::HybridSegmentWriter {
    pub(crate) fn build_graph_index(
        &self,
        col_name: &str,
        batch: &arrow::record_batch::RecordBatch,
        src_col: &str,
        dst_col: &str,
        row_offset: usize,
        local_staging_dir: &std::path::Path,
    ) -> Result<()> {
        let src_array = batch.column_by_name(src_col).context("Missing src_col")?;
        let dst_array = batch.column_by_name(dst_col).context("Missing dst_col")?;

        let num_rows = batch.num_rows();

        let mut edges = Vec::with_capacity(num_rows);

        // For simplicity we assume src and dst are u64 or i64 integers for now
        // A production version would hash strings or use dictionary encodings
        if matches!(
            *src_array.data_type(),
            arrow::datatypes::DataType::UInt64 | arrow::datatypes::DataType::Int64
        ) && matches!(
            *dst_array.data_type(),
            arrow::datatypes::DataType::UInt64 | arrow::datatypes::DataType::Int64
        ) {
            for i in 0..num_rows {
                if src_array.is_null(i) || dst_array.is_null(i) {
                    continue;
                }

                // The `data_type()` checks above guarantee the downcast; skip the
                // row instead of panicking if the invariant is ever violated.
                let src_id = if *src_array.data_type() == arrow::datatypes::DataType::UInt64 {
                    match src_array
                        .as_any()
                        .downcast_ref::<arrow::array::UInt64Array>()
                    {
                        Some(a) => a.value(i),
                        None => continue,
                    }
                } else {
                    match src_array
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                    {
                        Some(a) => a.value(i) as u64,
                        None => continue,
                    }
                };

                let dst_id = if *dst_array.data_type() == arrow::datatypes::DataType::UInt64 {
                    match dst_array
                        .as_any()
                        .downcast_ref::<arrow::array::UInt64Array>()
                    {
                        Some(a) => a.value(i),
                        None => continue,
                    }
                } else {
                    match dst_array
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                    {
                        Some(a) => a.value(i) as u64,
                        None => continue,
                    }
                };

                edges.push((src_id, dst_id, (row_offset + i) as u32));
            }
        } else {
            // Very naive fallback for strings (hash)
            for i in 0..num_rows {
                let src_val =
                    crate::core::manifest::ManifestValue::from_array(src_array, i).to_string();
                let dst_val =
                    crate::core::manifest::ManifestValue::from_array(dst_array, i).to_string();

                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                src_val.hash(&mut hasher);
                let src_id = hasher.finish();

                let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
                dst_val.hash(&mut hasher2);
                let dst_id = hasher2.finish();

                edges.push((src_id, dst_id, (row_offset + i) as u32));
            }
        }

        if edges.is_empty() {
            return Ok(());
        }

        let tmp_path = local_staging_dir.join(format!(
            "{}.{}.tmp.graph.bin",
            self.config.segment_id, col_name
        ));

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&tmp_path)
            .context("Failed to open graph temp file")?;

        use std::io::Write;
        for (src_id, dst_id, row_id) in edges {
            file.write_all(&src_id.to_le_bytes())?;
            file.write_all(&dst_id.to_le_bytes())?;
            file.write_all(&row_id.to_le_bytes())?;
        }

        self.graph_data
            .lock()
            .insert(col_name.to_string(), tmp_path.to_string_lossy().to_string());

        Ok(())
    }
}
