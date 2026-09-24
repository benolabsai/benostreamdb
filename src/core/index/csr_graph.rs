use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CsrEdge {
    pub dst_id: u64,
    pub row_id: u32,
}

unsafe impl Zeroable for CsrEdge {}
unsafe impl Pod for CsrEdge {}

pub struct MmapCsrGraph {
    /// Mmap of the offsets array (u64 array of size num_nodes + 1)
    offsets_mmap: Arc<Mmap>,
    /// Mmap of the edges array (CsrEdge array)
    edges_mmap: Arc<Mmap>,
    /// Mmap of the dictionary mapping dense_id to original_id (u64 array)
    dict_mmap: Arc<Mmap>,
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
}

impl MmapCsrGraph {
    pub fn load(offsets_path: &Path, edges_path: &Path, dict_path: &Path) -> Result<Self> {
        let offsets_file = File::open(offsets_path)?;
        let edges_file = File::open(edges_path)?;
        let dict_file = File::open(dict_path)?;

        let offsets_mmap = unsafe { Mmap::map(&offsets_file)? };
        let edges_mmap = unsafe { Mmap::map(&edges_file)? };
        let dict_mmap = unsafe { Mmap::map(&dict_file)? };

        Ok(Self::from_mmaps(
            Arc::new(offsets_mmap),
            Arc::new(edges_mmap),
            Arc::new(dict_mmap),
        ))
    }

    pub fn from_mmaps(
        offsets_mmap: Arc<Mmap>,
        edges_mmap: Arc<Mmap>,
        dict_mmap: Arc<Mmap>,
    ) -> Self {
        let num_nodes = offsets_mmap.len() / std::mem::size_of::<u64>() - 1;
        let num_edges = edges_mmap.len() / std::mem::size_of::<CsrEdge>();
        Self {
            offsets_mmap,
            edges_mmap,
            dict_mmap,
            num_nodes,
            num_edges,
        }
    }

    /// Binary search dictionary for dense ID
    pub fn to_dense(&self, original_id: u64) -> Option<usize> {
        let dict = bytemuck::cast_slice::<u8, u64>(&self.dict_mmap);
        dict.binary_search(&original_id).ok()
    }

    /// Lookup original ID by dense ID
    pub fn to_original(&self, dense_id: usize) -> u64 {
        let dict = bytemuck::cast_slice::<u8, u64>(&self.dict_mmap);
        dict[dense_id]
    }

    pub fn get_neighbors_raw(&self, dense_node_id: usize) -> &[CsrEdge] {
        if dense_node_id >= self.num_nodes {
            return &[];
        }
        let offsets = bytemuck::cast_slice::<u8, u64>(&self.offsets_mmap);
        let start = offsets[dense_node_id] as usize;
        let end = offsets[dense_node_id + 1] as usize;

        let edges = bytemuck::cast_slice::<u8, CsrEdge>(&self.edges_mmap);
        &edges[start..end]
    }

    pub fn build_from_file(tmp_path: &Path, local_base_path: &Path) -> Result<Vec<String>> {
        use std::io::Read;

        let mut file = File::open(tmp_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut raw_edges: Vec<(u64, u64, u32)> = Vec::new();
        let chunk_size = 8 + 8 + 4;
        let mut unique_ids = Vec::new();

        for chunk in buffer.chunks_exact(chunk_size) {
            // `chunks_exact` guarantees exactly `chunk_size` bytes, so these
            // fixed-size reads are infallible without `try_into().unwrap()`.
            let src_id = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            let dst_id = u64::from_le_bytes([
                chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14],
                chunk[15],
            ]);
            let row_id = u32::from_le_bytes([chunk[16], chunk[17], chunk[18], chunk[19]]);
            raw_edges.push((src_id, dst_id, row_id));
            unique_ids.push(src_id);
            unique_ids.push(dst_id);
        }

        // Build dictionary: sort and dedup
        unique_ids.sort_unstable();
        unique_ids.dedup();

        // Rewrite raw edges to use dense IDs
        for edge in raw_edges.iter_mut() {
            // Every endpoint was pushed into `unique_ids`, so the search always
            // succeeds; on the impossible miss, keep the raw id rather than panic.
            if let Ok(i) = unique_ids.binary_search(&edge.0) {
                edge.0 = i as u64;
            }
            if let Ok(i) = unique_ids.binary_search(&edge.1) {
                edge.1 = i as u64;
            }
        }

        // Sort by dense src_id
        raw_edges.sort_unstable_by_key(|e| e.0);

        let max_src_id = unique_ids.len().saturating_sub(1) as u64;

        let mut offsets = vec![0u64; (max_src_id + 2) as usize];
        let mut edges = Vec::with_capacity(raw_edges.len());

        let mut current_src = 0;
        let mut current_offset = 0;
        for (src_id, dst_id, row_id) in raw_edges {
            while current_src < src_id {
                offsets[(current_src + 1) as usize] = current_offset;
                current_src += 1;
            }
            edges.push(CsrEdge { dst_id, row_id });
            current_offset += 1;
        }
        while current_src <= max_src_id {
            offsets[(current_src + 1) as usize] = current_offset;
            current_src += 1;
        }

        let offsets_path = std::path::PathBuf::from(format!(
            "{}.graph.csr.offsets",
            local_base_path.to_string_lossy()
        ));
        let edges_path = std::path::PathBuf::from(format!(
            "{}.graph.csr.edges",
            local_base_path.to_string_lossy()
        ));
        let dict_path = std::path::PathBuf::from(format!(
            "{}.graph.csr.dict",
            local_base_path.to_string_lossy()
        ));

        std::fs::write(&offsets_path, bytemuck::cast_slice(&offsets))?;
        std::fs::write(&edges_path, bytemuck::cast_slice(&edges))?;
        std::fs::write(&dict_path, bytemuck::cast_slice(&unique_ids))?;

        Ok(vec![
            offsets_path.to_string_lossy().to_string(),
            edges_path.to_string_lossy().to_string(),
            dict_path.to_string_lossy().to_string(),
        ])
    }
}

use crate::core::sql::graph_udf::drift_search::DriftGraph;

impl DriftGraph for MmapCsrGraph {
    fn get_neighbors(&self, node: u64) -> Vec<u64> {
        if let Some(dense_node) = self.to_dense(node) {
            self.get_neighbors_raw(dense_node)
                .iter()
                .map(|e| self.to_original(e.dst_id as usize))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn get_degree(&self, node: u64) -> usize {
        if let Some(dense_node) = self.to_dense(node) {
            if dense_node >= self.num_nodes {
                return 0;
            }
            let offsets = bytemuck::cast_slice::<u8, u64>(&self.offsets_mmap);
            let start = offsets[dense_node] as usize;
            let end = offsets[dense_node + 1] as usize;
            end - start
        } else {
            0
        }
    }
}

pub struct MultiSegmentCsrGraph {
    pub segments: Vec<MmapCsrGraph>,
}

impl MultiSegmentCsrGraph {
    pub fn new(segments: Vec<MmapCsrGraph>) -> Self {
        Self { segments }
    }
}

impl DriftGraph for MultiSegmentCsrGraph {
    fn get_neighbors(&self, node: u64) -> Vec<u64> {
        let mut all_neighbors = Vec::new();
        for seg in &self.segments {
            all_neighbors.extend(seg.get_neighbors(node));
        }
        all_neighbors
    }

    fn get_degree(&self, node: u64) -> usize {
        self.segments.iter().map(|seg| seg.get_degree(node)).sum()
    }
}
