#![allow(unused)]
#![allow(dead_code)]

use std::fs::File;
use std::sync::Arc;
use arrow::array::{Array, ListArray, StructArray, UInt32Array, UInt64Array, UInt8Array, BinaryArray};
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;

use crate::core::index::hnsw_rs::dist::Distance;
use crate::core::index::hnsw_rs::hnsw::Neighbour;
use crate::core::index::hnsw_rs::arrow_ipc::ArrowType;

pub struct ArrowHnsw<T: ArrowType, D: Distance<T>> {
    batch: RecordBatch,
    distance: D,
    // Cached typed arrays for fast zero-copy access
    data_id_array: Arc<UInt64Array>,
    vector_array: Arc<BinaryArray>,
    max_layer_array: Arc<UInt8Array>,
    neighbors_array: Arc<ListArray>, // List of Layers
    dimension: usize,
    entry_point: usize,
    max_layer: u8,
    pub ipc_bytes: Vec<u8>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: ArrowType, D: Distance<T>> ArrowHnsw<T, D> {
    pub fn load_from_bytes(bytes: &[u8], distance: D) -> Result<Self, String> {
        let cursor = std::io::Cursor::new(bytes);
        let mut reader = FileReader::try_new(cursor, None).map_err(|e| e.to_string())?;
        
        let batch = reader.next().ok_or("No batches in IPC file")?.map_err(|e| e.to_string())?;
        
        println!("Schema of loaded batch: {:#?}", batch.schema());
        let data_id_array = Arc::new(batch.column(0).as_any().downcast_ref::<UInt64Array>().ok_or("Failed to downcast data_id column to UInt64Array")?.clone());
        let vector_array = Arc::new(batch.column(1).as_any().downcast_ref::<BinaryArray>().ok_or_else(|| format!("expected BinaryArray for vector, got {:?}", batch.column(1).data_type()))?.clone());
        let max_layer_array = Arc::new(batch.column(2).as_any().downcast_ref::<UInt8Array>().ok_or("Failed to downcast max_layer column to UInt8Array")?.clone());
        let neighbors_array = Arc::new(batch.column(3).as_any().downcast_ref::<ListArray>().ok_or("Failed to downcast neighbors column to ListArray")?.clone());
        
        // Assume all vectors have the same dimension
        let dimension = if vector_array.len() > 0 {
            let b = vector_array.value(0);
            b.len() / std::mem::size_of::<T>()
        } else {
            0
        };

        // Find entry point (point with highest layer)
        let mut entry_point = 0;
        let mut max_layer = 0;
        for i in 0..max_layer_array.len() {
            let l = max_layer_array.value(i);
            if l > max_layer {
                max_layer = l;
                entry_point = i;
            }
        }

        Ok(Self {
            batch,
            distance,
            data_id_array,
            vector_array,
            max_layer_array,
            neighbors_array,
            dimension,
            entry_point,
            max_layer,
            ipc_bytes: bytes.to_vec(),
            _marker: std::marker::PhantomData,
        })
    }
    
    // Internal helper to get a vector slice safely using ArrowType trait
    pub fn get_vector(&self, idx: usize) -> &[T] {
        let bytes = self.vector_array.value(idx);
        T::from_bytes(bytes)
    }
    
    fn get_neighbors(&self, point_idx: usize, layer: usize) -> Vec<u32> {
        let layers_list = self.neighbors_array.value(point_idx);
        let layers_list = layers_list.as_any().downcast_ref::<ListArray>().unwrap();
        
        if layer >= layers_list.len() {
            return Vec::new();
        }
        
        let neighbors_structs = layers_list.value(layer);
        let neighbors_structs = neighbors_structs.as_any().downcast_ref::<StructArray>().unwrap();
        let idx_array = neighbors_structs.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        
        let mut result = Vec::with_capacity(neighbors_structs.len());
        for i in 0..neighbors_structs.len() {
            result.push(idx_array.value(i));
        }
        result
    }
    
    fn search_layer(
        &self,
        query: &[T],
        entry_point: usize,
        ef: usize,
        layer: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> std::collections::BinaryHeap<std::sync::Arc<crate::core::index::hnsw_rs::hnsw::PointWithOrder<T>>> {
        let mut return_points = std::collections::BinaryHeap::new();
        if self.neighbors_array.len() == 0 {
            return return_points;
        }

        let dist_to_entry = self.distance.eval(query, self.get_vector(entry_point));
        
        // visited points
        let mut visited = std::collections::HashSet::new();
        visited.insert(entry_point);

        // Min-heap for candidates (using negative distance)
        let mut candidate_points = std::collections::BinaryHeap::new();
        
        // We need dummy points to reuse hnsw::PointWithOrder for binary heap
        let dummy_pt = |idx: usize, dist: f32| {
            let p = crate::core::index::hnsw_rs::hnsw::Point::new(
                &[], 
                self.data_id_array.value(idx) as usize, 
                crate::core::index::hnsw_rs::hnsw::PointId(0, idx as i32)
            );
            std::sync::Arc::new(crate::core::index::hnsw_rs::hnsw::PointWithOrder::new(
                &std::sync::Arc::new(p), 
                dist
            ))
        };

        candidate_points.push(dummy_pt(entry_point, -dist_to_entry));
        
        let mut entry_valid = true;
        if let Some(f) = filter {
            if !f.contains(self.data_id_array.value(entry_point) as u32) {
                entry_valid = false;
            }
        }
        if entry_valid {
            return_points.push(dummy_pt(entry_point, dist_to_entry));
        }

        while !candidate_points.is_empty() {
            let c = candidate_points.pop().unwrap();
            
            if let Some(f) = return_points.peek() {
                if return_points.len() >= ef && -(c.dist_to_ref) > f.dist_to_ref {
                    break;
                }
            }

            let neighbors = self.get_neighbors(c.point_ref.get_point_id().1 as usize, layer);
            for e_idx in neighbors {
                let e_idx = e_idx as usize;
                if !visited.contains(&e_idx) {
                    visited.insert(e_idx);
                    let v = self.get_vector(e_idx);
                    let e_dist = self.distance.eval(query, v);

                    let mut enters_return = false;
                    if let Some(f) = filter {
                        if !f.contains(self.data_id_array.value(e_idx) as u32) {
                            enters_return = false;
                        } else if return_points.len() < ef {
                            enters_return = true;
                        } else if let Some(f_pt) = return_points.peek() {
                            if e_dist < f_pt.dist_to_ref {
                                enters_return = true;
                            }
                        }
                    } else if return_points.len() < ef {
                        enters_return = true;
                    } else if let Some(f_pt) = return_points.peek() {
                        if e_dist < f_pt.dist_to_ref {
                            enters_return = true;
                        }
                    }

                    let is_promising = if return_points.len() < ef {
                        true
                    } else if let Some(f_pt) = return_points.peek() {
                        e_dist < f_pt.dist_to_ref
                    } else {
                        true
                    };

                    if is_promising || enters_return {
                        candidate_points.push(dummy_pt(e_idx, -e_dist));
                        if enters_return {
                            return_points.push(dummy_pt(e_idx, e_dist));
                            if return_points.len() > ef {
                                return_points.pop();
                            }
                        }
                    }
                }
            }
        }
        return_points
    }

    pub fn search(&self, query: &[T], knbn: usize, ef_s: usize, filter: Option<&roaring::RoaringBitmap>) -> Vec<crate::core::index::hnsw_rs::hnsw::Neighbour> {
        if self.neighbors_array.len() == 0 {
            return Vec::new();
        }
        
        let mut pivot = self.entry_point;
        let mut dist_to_entry = self.distance.eval(query, self.get_vector(pivot));
        let mut new_pivot = None;

        for layer in (1..=self.max_layer as usize).rev() {
            loop {
                let mut has_changed = false;
                let neighbors = self.get_neighbors(pivot, layer);
                for n_idx in neighbors {
                    let n_idx = n_idx as usize;
                    let tmp_dist = self.distance.eval(query, self.get_vector(n_idx));
                    if tmp_dist < dist_to_entry {
                        new_pivot = Some(n_idx);
                        has_changed = true;
                        dist_to_entry = tmp_dist;
                    }
                }
                if has_changed {
                    pivot = new_pivot.unwrap();
                } else {
                    break;
                }
            }
        }

        let ef = ef_s.max(knbn);
        let mut neighbours_heap = self.search_layer(query, pivot, ef, 0, filter);
        
        // Heap is a max-heap of distances. We want a sorted vector of increasing distances.
        let mut neighbours = Vec::with_capacity(neighbours_heap.len());
        while let Some(p) = neighbours_heap.pop() {
            neighbours.push(p);
        }
        neighbours.reverse();

        let last = knbn.min(ef).min(neighbours.len());
        let mut results = Vec::with_capacity(last);
        for i in 0..last {
            let p = &neighbours[i];
            results.push(Neighbour {
                d_id: p.point_ref.get_origin_id(),
                distance: p.dist_to_ref,
                p_id: p.point_ref.get_point_id(),
            });
        }
        results
    }
}
