use std::sync::Arc;
use arrow::array::{
    BinaryBuilder, ListBuilder, StructBuilder,
    UInt32Builder, UInt64Builder, UInt8Builder,
    Float32Builder
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;

use crate::core::index::hnsw_rs::dist::Distance;
use crate::core::index::hnsw_rs::hnsw::Hnsw;

pub trait ArrowType: Clone + Send + Sync + 'static {
    fn as_bytes(slice: &[Self]) -> &[u8];
    fn from_bytes(bytes: &[u8]) -> &[Self];
}

impl ArrowType for f32 {
    fn as_bytes(slice: &[f32]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[f32] {
        bytemuck::cast_slice(bytes)
    }
}

impl ArrowType for u8 {
    fn as_bytes(slice: &[u8]) -> &[u8] { slice }
    fn from_bytes(bytes: &[u8]) -> &[u8] { bytes }
}

impl ArrowType for crate::core::index::SparseVector {
    fn as_bytes(_slice: &[Self]) -> &[u8] {
        unimplemented!("SparseVector Arrow IPC serialization is not yet supported")
    }
    fn from_bytes(_bytes: &[u8]) -> &[Self] {
        unimplemented!("SparseVector Arrow IPC deserialization is not yet supported")
    }
}

pub fn dump_arrow_ipc<T: ArrowType, D: Distance<T>>(
    hnsw: &Hnsw<T, D>,
) -> Result<Vec<u8>, String> {

    // 1. Define Schema
    let data_id_field = Field::new("data_id", DataType::UInt64, false);
    
    let vector_field = Field::new("vector", DataType::Binary, false);
    
    let max_layer_field = Field::new("max_layer", DataType::UInt8, false);

    let neighbor_fields = vec![
        Field::new("neighbor_idx", DataType::UInt32, false),
        Field::new("distance", DataType::Float32, false),
    ];
    let neighbor_struct_field = Field::new("item", DataType::Struct(neighbor_fields.clone().into()), true);
    
    let inner_list_field = Field::new("item", DataType::List(Arc::new(neighbor_struct_field.clone())), true);
    let neighbors_field = Field::new("neighbors", DataType::List(Arc::new(inner_list_field.clone())), false);

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
        ]
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
        vector_builder.append_value(T::as_bytes(v));

        // Max layer
        let max_layer_for_point = point.get_point_id().0;
        let ref_neighbors = point.neighbours.read();
        max_layer_builder.append_value(max_layer_for_point);

        // Neighbors (List of Layers -> List of Structs)
        for i in 0..=max_layer_for_point as usize {
            let layer_neighbors = &ref_neighbors[i];
            
            for neighbor in layer_neighbors.iter() {
                // Struct has 2 fields: idx, distance
                let idx = *point_id_to_idx.get(&neighbor.point_ref.get_point_id()).ok_or_else(|| format!("Neighbor point ID not found: {:?}", neighbor.point_ref.get_point_id()))?;
                let dist = neighbor.dist_to_ref;
                
                let sb = neighbors_builder.values().values();
                sb.field_builder::<UInt32Builder>(0).ok_or("Failed to get UInt32Builder for neighbor_idx")?.append_value(idx);
                sb.field_builder::<Float32Builder>(1).ok_or("Failed to get Float32Builder for distance")?.append_value(dist);
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
    ).map_err(|e| e.to_string())?;

    // 5. Write to IPC Buffer
    let mut buffer = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buffer, &schema).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }

    Ok(buffer)
}
