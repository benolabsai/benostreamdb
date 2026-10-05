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

/// Backing storage for a CSR array: either a memory-mapped file (loose sidecar
/// layout) or an owned byte buffer (Puffin compound-bundle layout, where the
/// blob is read into memory).
#[derive(Clone)]
enum GraphBytes {
    Mmap(Arc<Mmap>),
    Owned(Arc<Vec<u8>>),
}

impl GraphBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            GraphBytes::Mmap(m) => m.as_ref(),
            GraphBytes::Owned(v) => v.as_slice(),
        }
    }
}

#[derive(Clone)]
pub struct MmapCsrGraph {
    /// Offsets array (u64 array of size num_nodes + 1)
    offsets: GraphBytes,
    /// Edges array (CsrEdge array)
    edges: GraphBytes,
    /// Dictionary mapping dense_id to original_id (u64 array)
    dict: GraphBytes,
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
            offsets: GraphBytes::Mmap(offsets_mmap),
            edges: GraphBytes::Mmap(edges_mmap),
            dict: GraphBytes::Mmap(dict_mmap),
            num_nodes,
            num_edges,
        }
    }

    /// Build a CSR graph from owned byte buffers (Puffin blob payloads).
    pub fn from_bytes(offsets: Vec<u8>, edges: Vec<u8>, dict: Vec<u8>) -> Self {
        let num_nodes = offsets.len() / std::mem::size_of::<u64>() - 1;
        let num_edges = edges.len() / std::mem::size_of::<CsrEdge>();
        Self {
            offsets: GraphBytes::Owned(Arc::new(offsets)),
            edges: GraphBytes::Owned(Arc::new(edges)),
            dict: GraphBytes::Owned(Arc::new(dict)),
            num_nodes,
            num_edges,
        }
    }

    /// Total resident bytes of the three CSR arrays (offsets + edges + dict).
    ///
    /// Used as the cache weigher so the graph cache respects the global
    /// `BSDB_CACHE_GB` budget.
    pub fn size_in_bytes(&self) -> usize {
        self.offsets.as_slice().len() + self.edges.as_slice().len() + self.dict.as_slice().len()
    }

    /// Binary search dictionary for dense ID
    pub fn to_dense(&self, original_id: u64) -> Option<usize> {
        let dict = bytemuck::cast_slice::<u8, u64>(self.dict.as_slice());
        dict.binary_search(&original_id).ok()
    }

    /// Lookup original ID by dense ID
    pub fn to_original(&self, dense_id: usize) -> u64 {
        let dict = bytemuck::cast_slice::<u8, u64>(self.dict.as_slice());
        dict[dense_id]
    }

    /// Slice of the dictionary mapping dense_id to original_id
    pub fn dict(&self) -> &[u64] {
        bytemuck::cast_slice::<u8, u64>(self.dict.as_slice())
    }

    pub fn get_neighbors_raw(&self, dense_node_id: usize) -> &[CsrEdge] {
        if dense_node_id >= self.num_nodes {
            return &[];
        }
        let offsets = bytemuck::cast_slice::<u8, u64>(self.offsets.as_slice());
        if dense_node_id + 1 >= offsets.len() {
            return &[];
        }
        let start = offsets[dense_node_id] as usize;
        let end = offsets[dense_node_id + 1] as usize;

        let edges = bytemuck::cast_slice::<u8, CsrEdge>(self.edges.as_slice());
        let len = edges.len();
        let start = start.min(len);
        let end = end.min(len);
        if start >= end {
            return &[];
        }
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

        // The `graph_v2` suffix is the on-disk format version. Graph indexes
        // are keyed solely by `src_column`.
        let offsets_path = std::path::PathBuf::from(format!(
            "{}.graph_v2.csr.offsets",
            local_base_path.to_string_lossy()
        ));
        let edges_path = std::path::PathBuf::from(format!(
            "{}.graph_v2.csr.edges",
            local_base_path.to_string_lossy()
        ));
        let dict_path = std::path::PathBuf::from(format!(
            "{}.graph_v2.csr.dict",
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

use crate::core::sql::graph_udf::graph_view::GraphView;

impl GraphView for MmapCsrGraph {
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

    fn get_neighbors_into(&self, node: u64, out: &mut Vec<u64>) {
        if let Some(dense_node) = self.to_dense(node) {
            out.extend(
                self.get_neighbors_raw(dense_node)
                    .iter()
                    .map(|e| self.to_original(e.dst_id as usize)),
            );
        }
    }

    fn get_degree(&self, node: u64) -> usize {
        if let Some(dense_node) = self.to_dense(node) {
            if dense_node >= self.num_nodes {
                return 0;
            }
            let offsets = bytemuck::cast_slice::<u8, u64>(self.offsets.as_slice());
            let start = offsets[dense_node] as usize;
            let end = offsets[dense_node + 1] as usize;
            end - start
        } else {
            0
        }
    }

    /// Nodes with at least one outgoing edge, matching [`SimpleGraph`]'s
    /// source-only semantics so global algorithms see the same node set in
    /// every mode.
    ///
    /// [`SimpleGraph`]: crate::core::sql::graph_udf::graph_view::SimpleGraph
    fn all_nodes(&self) -> Vec<u64> {
        let mut nodes = Vec::new();
        for dense in 0..self.num_nodes {
            if !self.get_neighbors_raw(dense).is_empty() {
                nodes.push(self.to_original(dense));
            }
        }
        nodes.sort_unstable();
        nodes
    }

    fn num_edges(&self) -> usize {
        self.num_edges
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

impl GraphView for MultiSegmentCsrGraph {
    fn get_neighbors(&self, node: u64) -> Vec<u64> {
        let mut all_neighbors = Vec::new();
        for seg in &self.segments {
            all_neighbors.extend(seg.get_neighbors(node));
        }
        all_neighbors
    }

    fn get_neighbors_into(&self, node: u64, out: &mut Vec<u64>) {
        for seg in &self.segments {
            seg.get_neighbors_into(node, out);
        }
    }

    fn get_degree(&self, node: u64) -> usize {
        self.segments.iter().map(|seg| seg.get_degree(node)).sum()
    }

    fn all_nodes(&self) -> Vec<u64> {
        let mut nodes: Vec<u64> = self
            .segments
            .iter()
            .flat_map(|seg| seg.all_nodes())
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    fn num_edges(&self) -> usize {
        self.segments.iter().map(|seg| seg.num_edges()).sum()
    }
}
