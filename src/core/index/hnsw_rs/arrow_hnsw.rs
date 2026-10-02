#![allow(unused)]
#![allow(dead_code)]

use arrow::array::{
    Array, BinaryArray, ListArray, StructArray, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::Buffer;
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::{read_footer_length, FileDecoder};
use arrow_ipc::{convert::fb_to_schema, root_as_footer};
use std::sync::Arc;

use crate::core::index::hnsw_rs::arrow_ipc::ArrowType;
use crate::core::index::hnsw_rs::dist::Distance;
use crate::core::index::hnsw_rs::hnsw::Neighbour;
use ahash::AHashSet;

#[derive(Clone)]
struct Candidate {
    idx: usize,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.idx == other.idx
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub struct ArrowHnsw<T: ArrowType, D: Distance<T>> {
    batch: RecordBatch,
    distance: D,
    // Cached typed arrays for fast zero-copy access
    data_id_array: Arc<UInt64Array>,
    vector_array: Arc<BinaryArray>,
    max_layer_array: Arc<UInt8Array>,
    neighbors_array: Arc<ListArray>, // List of Layers
    l0_offsets: arrow::buffer::OffsetBuffer<i32>,
    l1_offsets: arrow::buffer::OffsetBuffer<i32>,
    neighbors_flat: Arc<UInt32Array>,
    dimension: usize,
    entry_point: usize,
    max_layer: u8,
    _marker: std::marker::PhantomData<T>,
}

impl<T: ArrowType, D: Distance<T>> ArrowHnsw<T, D> {
    pub fn load_from_mmap(mmap: Arc<memmap2::Mmap>, distance: D) -> Result<Self, String> {
        let buffer = unsafe {
            Buffer::from_custom_allocation(
                std::ptr::NonNull::new_unchecked(mmap.as_ptr() as *mut u8),
                mmap.len(),
                mmap.clone(),
            )
        };
        Self::load_from_buffer(buffer, distance)
    }

    pub fn load_from_bytes(bytes: &[u8], distance: D) -> Result<Self, String> {
        let buffer = Buffer::from(bytes);
        Self::load_from_buffer(buffer, distance)
    }

    fn load_from_buffer(buffer: Buffer, distance: D) -> Result<Self, String> {
        if buffer.len() < 10 {
            return Err("IPC file too small".to_string());
        }
        let trailer_start = buffer.len() - 10;
        let footer_len = read_footer_length(buffer[trailer_start..].try_into().unwrap())
            .map_err(|e: arrow::error::ArrowError| e.to_string())?;

        let footer = root_as_footer(&buffer[trailer_start - footer_len..trailer_start])
            .map_err(|e| e.to_string())?;

        let schema = fb_to_schema(footer.schema().ok_or("No schema")?);

        let mut decoder = FileDecoder::new(Arc::new(schema), footer.version());

        if let Some(dicts) = footer.dictionaries() {
            for block in dicts.iter() {
                let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
                let data = buffer.slice_with_length(block.offset() as _, block_len);
                decoder
                    .read_dictionary(block, &data)
                    .map_err(|e| e.to_string())?;
            }
        }

        let batches: Vec<arrow_ipc::Block> = if let Some(rb) = footer.recordBatches() {
            rb.iter().copied().collect()
        } else {
            Vec::new()
        };

        if batches.is_empty() {
            return Err("No batches in IPC file".to_string());
        }

        let block = &batches[0];
        let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
        let data = buffer.slice_with_length(block.offset() as _, block_len);
        let batch = decoder
            .read_record_batch(block, &data)
            .map_err(|e| e.to_string())?
            .ok_or("Failed to read record batch")?;

        tracing::debug!("Schema of loaded batch: {:#?}", batch.schema());
        let data_id_array = Arc::new(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or("Failed to downcast data_id column to UInt64Array")?
                .clone(),
        );
        let mut vector_array = Arc::new(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| {
                    format!(
                        "expected BinaryArray for vector, got {:?}",
                        batch.column(1).data_type()
                    )
                })?
                .clone(),
        );

        // `get_vector` casts a binary-array value slice to `&[T]` via
        // `bytemuck::cast_slice`, which requires `align_of::<T>()` alignment.
        // Arrow IPC buffers are only guaranteed aligned relative to the
        // message body, so a value slice can land on a misaligned offset
        // (intermittently, depending on data sizes / allocation history),
        // causing a `TargetAlignmentGreaterAndInputNotAligned` panic.
        // Repack once into an arrow-allocated (aligned) buffer if needed;
        // the hot search path stays zero-copy.
        let align = std::mem::align_of::<T>();
        if align > 1
            && !vector_array.is_empty()
            && !(vector_array.value(0).as_ptr() as usize).is_multiple_of(align)
        {
            let total_bytes = (*vector_array.value_offsets().last().unwrap_or(&0)) as usize;
            let mut builder =
                arrow::array::BinaryBuilder::with_capacity(vector_array.len(), total_bytes);
            for i in 0..vector_array.len() {
                builder.append_value(vector_array.value(i));
            }
            vector_array = Arc::new(builder.finish());
        }

        let max_layer_array = Arc::new(
            batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or("Failed to downcast max_layer column to UInt8Array")?
                .clone(),
        );
        let neighbors_array = Arc::new(
            batch
                .column(3)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or("Failed to downcast neighbors column to ListArray")?
                .clone(),
        );

        let l0_offsets = neighbors_array.offsets().clone();
        let l1 = neighbors_array.values();
        let l1 = l1.as_any().downcast_ref::<ListArray>().unwrap();
        let l1_offsets = l1.offsets().clone();
        let s = l1.values();
        let s = s
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .unwrap();
        let neighbors_flat = Arc::new(
            s.column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .clone(),
        );

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
            l0_offsets,
            l1_offsets,
            neighbors_flat,
            dimension,
            entry_point,
            max_layer,
            _marker: std::marker::PhantomData,
        })
    }

    /// Decode the value stored for `idx`.
    ///
    /// Returns an owned `Vec<T>`: the byte blob in the Arrow array is only a
    /// contiguous `&[T]` for fixed-width types, so variable-length values (e.g.
    /// `SparseVector`) must be decoded.
    pub fn get_vector(&self, idx: usize) -> Vec<T> {
        let bytes = self.vector_array.value(idx);
        T::from_bytes(bytes)
    }

    pub fn get_vector_slice<'a>(&'a self, idx: usize, scratch: &'a mut Vec<T>) -> &'a [T] {
        let bytes = self.vector_array.value(idx);
        if let Some(s) = T::slice_from_bytes(bytes) {
            s
        } else {
            *scratch = T::from_bytes(bytes);
            scratch.as_slice()
        }
    }

    fn with_neighbors<F>(&self, point_idx: usize, layer: usize, mut f: F)
    where
        F: FnMut(&[u32]),
    {
        let l1_start = self.l0_offsets[point_idx] as usize;
        let l1_end = self.l0_offsets[point_idx + 1] as usize;
        let l1_len = l1_end - l1_start;

        if layer >= l1_len {
            f(&[]);
            return;
        }

        let l1_idx = l1_start + layer;
        let u_start = self.l1_offsets[l1_idx] as usize;
        let u_end = self.l1_offsets[l1_idx + 1] as usize;

        let flat_values = self.neighbors_flat.values();
        f(&flat_values[u_start..u_end])
    }

    fn search_layer(
        &self,
        query: &[T],
        entry_point: usize,
        ef: usize,
        layer: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> std::collections::BinaryHeap<Candidate> {
        let mut return_points = std::collections::BinaryHeap::with_capacity(ef);
        if self.neighbors_array.len() == 0 {
            return return_points;
        }

        let dist_to_entry = self.distance.eval(query, &self.get_vector(entry_point));

        // visited points
        let mut visited = AHashSet::with_capacity(ef * 4);
        visited.insert(entry_point);

        // Min-heap for candidates (using negative distance)
        let mut candidate_points = std::collections::BinaryHeap::with_capacity(ef);

        candidate_points.push(Candidate {
            idx: entry_point,
            dist: -dist_to_entry,
        });

        let mut entry_valid = true;
        if let Some(f) = filter {
            if !f.contains(self.data_id_array.value(entry_point) as u32) {
                entry_valid = false;
            }
        }
        if entry_valid {
            return_points.push(Candidate {
                idx: entry_point,
                dist: dist_to_entry,
            });
        }

        while !candidate_points.is_empty() {
            let c = candidate_points.pop().unwrap();

            if let Some(f) = return_points.peek() {
                if return_points.len() >= ef && -(c.dist) > f.dist {
                    break;
                }
            }

            let mut scratch = Vec::new();
            self.with_neighbors(c.idx, layer, |neighbors| {
                for &e_idx in neighbors {
                    let e_idx = e_idx as usize;
                    if !visited.contains(&e_idx) {
                        visited.insert(e_idx);
                        let v_slice = self.get_vector_slice(e_idx, &mut scratch);
                        let e_dist = self.distance.eval(query, v_slice);

                        let is_promising = if return_points.len() < ef {
                            true
                        } else if let Some(f_pt) = return_points.peek() {
                            e_dist < f_pt.dist
                        } else {
                            true
                        };

                        let mut enters_return = is_promising;
                        if let Some(f) = filter {
                            if !f.contains(self.data_id_array.value(e_idx) as u32) {
                                enters_return = false;
                            }
                        }

                        if is_promising || enters_return {
                            candidate_points.push(Candidate {
                                idx: e_idx,
                                dist: -e_dist,
                            });
                            if enters_return {
                                return_points.push(Candidate {
                                    idx: e_idx,
                                    dist: e_dist,
                                });
                                if return_points.len() > ef {
                                    return_points.pop();
                                }
                            }
                        }
                    }
                }
            });
        }
        return_points
    }

    pub fn search(
        &self,
        query: &[T],
        knbn: usize,
        ef_s: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Vec<crate::core::index::hnsw_rs::hnsw::Neighbour> {
        if self.neighbors_array.len() == 0 {
            return Vec::new();
        }

        let mut pivot = self.entry_point;
        let mut scratch = Vec::new();
        let v_slice = self.get_vector_slice(pivot, &mut scratch);
        let mut dist_to_entry = self.distance.eval(query, v_slice);
        let mut new_pivot = None;

        for layer in (1..=self.max_layer as usize).rev() {
            loop {
                let mut has_changed = false;
                self.with_neighbors(pivot, layer, |neighbors| {
                    for &n_idx in neighbors {
                        let n_idx = n_idx as usize;
                        let tmp_slice = self.get_vector_slice(n_idx, &mut scratch);
                        let tmp_dist = self.distance.eval(query, tmp_slice);
                        if tmp_dist < dist_to_entry {
                            new_pivot = Some(n_idx);
                            has_changed = true;
                            dist_to_entry = tmp_dist;
                        }
                    }
                });
                if has_changed {
                    pivot = new_pivot.unwrap();
                } else {
                    break;
                }
            }
        }

        let ef = ef_s.max(knbn);
        let neighbours_heap = self.search_layer(query, pivot, ef, 0, filter);

        let neighbours = neighbours_heap.into_sorted_vec();

        let last = knbn.min(ef).min(neighbours.len());
        let mut results = Vec::with_capacity(last);
        for i in 0..last {
            let p = &neighbours[i];
            results.push(Neighbour {
                d_id: self.data_id_array.value(p.idx) as usize,
                distance: p.dist,
                p_id: crate::core::index::hnsw_rs::hnsw::PointId(0, p.idx as i32),
            });
        }
        results
    }
}
