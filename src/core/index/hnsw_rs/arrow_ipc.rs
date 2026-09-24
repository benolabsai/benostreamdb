use arrow::array::{
    BinaryBuilder, Float32Builder, ListBuilder, StructBuilder, UInt32Builder, UInt64Builder,
    UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::core::index::hnsw_rs::dist::Distance;
use crate::core::index::hnsw_rs::hnsw::Hnsw;

/// Serialization for the value type stored in an [`ArrowHnsw`](super::arrow_hnsw::ArrowHnsw).
///
/// The conversion is *owned* rather than a zero-copy reinterpretation: a
/// variable-length type such as [`SparseVector`](crate::core::index::SparseVector)
/// stores its elements in separate heap allocations, so there is no contiguous
/// `&[Self]` view over raw bytes to hand back. Fixed-width types simply
/// `bytemuck`-cast; sparse vectors use a self-describing encoding.
pub trait ArrowType: Clone + Send + Sync + 'static {
    /// Encode a slice of values into a byte blob.
    fn to_bytes(slice: &[Self]) -> Vec<u8>;
    /// Decode a blob produced by [`ArrowType::to_bytes`].
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;
}

impl ArrowType for f32 {
    fn to_bytes(slice: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(slice).to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice(bytes).to_vec()
    }
}

impl ArrowType for u8 {
    fn to_bytes(slice: &[u8]) -> Vec<u8> {
        slice.to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Vec<u8> {
        bytes.to_vec()
    }
}

impl ArrowType for crate::core::index::SparseVector {
    /// Layout: `[u64 count]` then, per vector, `[u64 dim][u64 nnz]` followed by
    /// `nnz` × (`[u32 index][f32 value]`), all little-endian.
    fn to_bytes(slice: &[Self]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(slice.len() as u64).to_le_bytes());
        for v in slice {
            out.extend_from_slice(&(v.dim as u64).to_le_bytes());
            out.extend_from_slice(&(v.indices.len() as u64).to_le_bytes());
            for (idx, val) in v.indices.iter().zip(v.values.iter()) {
                out.extend_from_slice(&idx.to_le_bytes());
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
        out
    }

    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        struct Reader<'a> {
            b: &'a [u8],
            p: usize,
        }
        impl Reader<'_> {
            fn u64(&mut self) -> Option<u64> {
                let s = self.b.get(self.p..self.p + 8)?;
                self.p += 8;
                Some(u64::from_le_bytes(s.try_into().ok()?))
            }
            fn u32(&mut self) -> Option<u32> {
                let s = self.b.get(self.p..self.p + 4)?;
                self.p += 4;
                Some(u32::from_le_bytes(s.try_into().ok()?))
            }
            fn f32(&mut self) -> Option<f32> {
                let s = self.b.get(self.p..self.p + 4)?;
                self.p += 4;
                Some(f32::from_le_bytes(s.try_into().ok()?))
            }
        }

        let mut r = Reader { b: bytes, p: 0 };
        let n = match r.u64() {
            Some(n) => n as usize,
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let (dim, nnz) = match (r.u64(), r.u64()) {
                (Some(d), Some(k)) => (d as usize, k as usize),
                _ => break,
            };
            let mut indices = Vec::with_capacity(nnz);
            let mut values = Vec::with_capacity(nnz);
            for _ in 0..nnz {
                match (r.u32(), r.f32()) {
                    (Some(i), Some(v)) => {
                        indices.push(i);
                        values.push(v);
                    }
                    _ => break,
                }
            }
            out.push(crate::core::index::SparseVector {
                indices,
                values,
                dim,
            });
        }
        out
    }
}

pub fn dump_arrow_ipc<T: ArrowType, D: Distance<T>>(hnsw: &Hnsw<T, D>) -> Result<Vec<u8>, String> {
    // 1. Define Schema
    let data_id_field = Field::new("data_id", DataType::UInt64, false);

    let vector_field = Field::new("vector", DataType::Binary, false);

    let max_layer_field = Field::new("max_layer", DataType::UInt8, false);

    let neighbor_fields = vec![
        Field::new("neighbor_idx", DataType::UInt32, false),
        Field::new("distance", DataType::Float32, false),
    ];
    let neighbor_struct_field = Field::new(
        "item",
        DataType::Struct(neighbor_fields.clone().into()),
        true,
    );

    let inner_list_field = Field::new(
        "item",
        DataType::List(Arc::new(neighbor_struct_field.clone())),
        true,
    );
    let neighbors_field = Field::new(
        "neighbors",
        DataType::List(Arc::new(inner_list_field.clone())),
        false,
    );

    let schema = Arc::new(Schema::new(vec![
        data_id_field,
        vector_field,
        max_layer_field,
        neighbors_field,
    ]));

    // 2. Initialize Builders
    let mut data_id_builder = UInt64Builder::new();
    let mut vector_builder = BinaryBuilder::new();
    let mut max_layer_builder = UInt8Builder::new();

    // The neighbors builder is a List of List of Structs
    let struct_builder = StructBuilder::new(
        neighbor_fields,
        vec![
            Box::new(UInt32Builder::new()),
            Box::new(Float32Builder::new()),
        ],
    );
    let inner_list_builder = ListBuilder::new(struct_builder);
    let mut neighbors_builder = ListBuilder::new(inner_list_builder);

    // 3. Iterate through all points
    let mut point_id_to_idx = std::collections::HashMap::new();
    let mut all_points = Vec::new();
    for point in hnsw.layer_indexed_points.into_iter() {
        if point_id_to_idx.contains_key(&point.get_point_id()) {
            continue;
        }
        point_id_to_idx.insert(point.get_point_id(), all_points.len() as u32);
        all_points.push(point);
    }

    if all_points.is_empty() {
        return Ok(Vec::new());
    }

    for point in all_points.iter() {
        let origin_id = point.get_origin_id() as u64;
        data_id_builder.append_value(origin_id);

        // Vector
        let v = point.get_v();
        vector_builder.append_value(T::to_bytes(v));

        // Max layer
        let max_layer_for_point = point.get_point_id().0;
        let ref_neighbors = point.neighbours.read();
        max_layer_builder.append_value(max_layer_for_point);

        // Neighbors (List of Layers -> List of Structs)
        for i in 0..=max_layer_for_point as usize {
            let layer_neighbors = &ref_neighbors[i];

            for neighbor in layer_neighbors.iter() {
                // Struct has 2 fields: idx, distance
                let idx = *point_id_to_idx
                    .get(&neighbor.point_ref.get_point_id())
                    .ok_or_else(|| {
                        format!(
                            "Neighbor point ID not found: {:?}",
                            neighbor.point_ref.get_point_id()
                        )
                    })?;
                let dist = neighbor.dist_to_ref;

                let sb = neighbors_builder.values().values();
                sb.field_builder::<UInt32Builder>(0)
                    .ok_or("Failed to get UInt32Builder for neighbor_idx")?
                    .append_value(idx);
                sb.field_builder::<Float32Builder>(1)
                    .ok_or("Failed to get Float32Builder for distance")?
                    .append_value(dist);
                sb.append(true);
            }
            neighbors_builder.values().append(true);
        }
        neighbors_builder.append(true);
    }

    // 4. Build RecordBatch
    let data_id_array = Arc::new(data_id_builder.finish());
    let vector_array = Arc::new(vector_builder.finish());
    let max_layer_array = Arc::new(max_layer_builder.finish());
    let neighbors_array = Arc::new(neighbors_builder.finish());

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            data_id_array,
            vector_array,
            max_layer_array,
            neighbors_array,
        ],
    )
    .map_err(|e| e.to_string())?;

    // 5. Write to IPC Buffer
    let mut buffer = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buffer, &schema).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }

    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::index::SparseVector;

    fn sv(indices: Vec<u32>, values: Vec<f32>, dim: usize) -> SparseVector {
        SparseVector {
            indices,
            values,
            dim,
        }
    }

    #[test]
    fn f32_round_trip() {
        let v = vec![1.0f32, 2.5, -3.0];
        let bytes = <f32 as ArrowType>::to_bytes(&v);
        assert_eq!(<f32 as ArrowType>::from_bytes(&bytes), v);
    }

    #[test]
    fn sparse_vector_round_trip() {
        let original = vec![
            sv(vec![0, 5, 10], vec![1.0, 2.0, 3.0], 100),
            sv(vec![], vec![], 50),
            sv(vec![1], vec![-0.5], 7),
        ];
        let bytes = <SparseVector as ArrowType>::to_bytes(&original);
        let decoded = <SparseVector as ArrowType>::from_bytes(&bytes);

        assert_eq!(decoded.len(), original.len());
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert_eq!(a.dim, b.dim);
            assert_eq!(a.indices, b.indices);
            assert_eq!(a.values, b.values);
        }
    }

    #[test]
    fn sparse_vector_truncated_blob_does_not_panic() {
        let original = vec![sv(vec![0, 1], vec![1.0, 2.0], 10)];
        let bytes = <SparseVector as ArrowType>::to_bytes(&original);
        // Decoding a truncated blob must stop cleanly rather than panic.
        let decoded = <SparseVector as ArrowType>::from_bytes(&bytes[..bytes.len() - 3]);
        assert!(decoded.iter().all(|v| v.indices.len() < 2));
    }
}
