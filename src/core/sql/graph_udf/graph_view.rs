// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Shared graph-view abstraction for the graph algorithms.
//!
//! Every graph algorithm (DRIFT, communities, PageRank, BFS, …) needs only
//! `get_neighbors` and `get_degree`, so they are written against [`GraphView`]
//! and can run on either an in-memory adjacency map ([`SimpleGraph`]) or the
//! mmap-backed CSR ([`crate::core::index::csr_graph`]).
//!
//! [`GraphMode`] is the shared "in-memory vs out-of-core" selector, and
//! [`graph_memory_budget_bytes`] is the single budget knob
//! (`BSDB_DRIFT_MEMORY_MB`, else the engine's derived memory budget).
//!
//! [`GraphAccumulatorBase`] is the shared state that every graph-algorithm UDAF
//! accumulator embeds: it collects `(source, target)` edge rows from Arrow
//! batches and resolves them into a [`GraphView`] — either in-memory from the
//! rows, or out-of-core from a CSR index when `graph_uri` is supplied.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ListArray, ListBuilder, StringArray, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::scalar::ScalarValue;

use crate::core::table::Table;

/// The minimal graph surface every graph algorithm needs.
///
/// Default methods provide optional extensions for algorithms that need
/// weighted edges, the full node set, or edge counts. The defaults are
/// safe no-ops so existing impls (CSR, `SubgraphView`, `CachingGraph`)
/// keep working without changes.
pub trait GraphView: Send + Sync {
    fn get_neighbors(&self, node: u64) -> Vec<u64>;
    fn get_degree(&self, node: u64) -> usize;

    /// Append `node`'s neighbors to `out` (which is **not** cleared).
    ///
    /// This is the allocation-free traversal primitive. Hot algorithms keep one
    /// scratch `Vec<u64>` and reuse it across every hop instead of receiving a
    /// freshly heap-allocated `Vec` from [`Self::get_neighbors`] per visit. Backings
    /// that own a contiguous adjacency override this to copy straight into
    /// `out`; the default forwards to `get_neighbors`. The method takes no
    /// generic parameters, so it stays object-safe for `dyn GraphView`.
    fn get_neighbors_into(&self, node: u64, out: &mut Vec<u64>) {
        out.extend(self.get_neighbors(node));
    }

    /// A shared, reference-counted neighbor list — the zero-copy accessor.
    ///
    /// Backings that own a contiguous adjacency (`SimpleGraph`) or cache
    /// neighbor lists (`CachingGraph`) return an `Arc` clone, so a traversal can
    /// hold the list without copying it. The default allocates once from
    /// [`Self::get_neighbors`]. Object-safe (no generics), so `dyn GraphView` works.
    fn get_neighbors_shared(&self, node: u64) -> Arc<[u64]> {
        Arc::from(self.get_neighbors(node))
    }

    /// Weighted neighbor list. Default: every edge has weight 1.0.
    fn get_weighted_neighbors(&self, node: u64) -> Vec<(u64, f32)> {
        self.get_neighbors(node)
            .into_iter()
            .map(|n| (n, 1.0))
            .collect()
    }

    /// All node ids in the graph (sorted). Required by global algorithms
    /// (PageRank, connected components). Returns empty by default.
    fn all_nodes(&self) -> Vec<u64> {
        Vec::new()
    }

    /// Total number of directed edges. Returns 0 by default.
    fn num_edges(&self) -> usize {
        0
    }

    /// All edges in the graph as a vector of (source, target) pairs.
    fn all_edges(&self) -> Vec<(u64, u64)> {
        let mut edges = Vec::new();
        for u in self.all_nodes() {
            for v in self.get_neighbors(u) {
                edges.push((u, v));
            }
        }
        edges
    }
}

/// How a graph algorithm sources its graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphMode {
    /// Materialize the working subgraph in RAM (`SimpleGraph`). Fastest per
    /// hop, but the whole adjacency is resident.
    InMemory,
    /// Walk the mmap-backed CSR directly. No materialization; each hop reads
    /// from the mapped file.
    OutOfCore,
    /// Out-of-core backing with an in-memory LRU cache of hot neighbor lists,
    /// bounded by the graph memory budget. The adaptive middle tier.
    Cached,
    /// Pick `InMemory` when the estimated subgraph fits the graph memory
    /// budget, else `OutOfCore`.
    Auto,
}

/// Byte budget for the in-memory graph.
///
/// `BSDB_DRIFT_MEMORY_MB` overrides; otherwise the engine's derived memory
/// budget (cgroup limit or host RAM × 0.8) applies — the same knob the ingest
/// and DataFusion budgets use.
pub fn graph_memory_budget_bytes() -> u64 {
    if let Some(mb) = std::env::var("BSDB_DRIFT_MEMORY_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|mb| *mb > 0)
    {
        return mb * 1_000_000;
    }
    crate::core::resources::default_memory_budget_bytes()
}

/// Estimate the resident bytes of a `SimpleGraph` over `nodes` with `avg_degree`.
///
/// Each adjacency entry is a `Vec<u64>` (8 bytes/element + 24-byte header) plus
/// the `HashMap` slot overhead. The estimate only needs to be order-of-magnitude
/// to choose a mode.
pub fn estimate_subgraph_bytes(nodes: usize, avg_degree: usize) -> usize {
    nodes.saturating_mul(
        avg_degree
            .saturating_mul(8)
            .saturating_add(24)
            .saturating_add(16),
    )
}

/// In-memory adjacency map.
///
/// Neighbor lists are stored as reference-counted slices, so
/// [`GraphView::get_neighbors_shared`] is a pointer clone — the zero-copy path
/// for in-memory traversal.
#[derive(Default)]
pub struct SimpleGraph {
    pub adjacency: HashMap<u64, Arc<[u64]>>,
}

impl SimpleGraph {
    /// Sort/dedup each neighbor list and freeze it into an `Arc<[u64]>`.
    fn finalize(adj: HashMap<u64, Vec<u64>>) -> Self {
        Self {
            adjacency: adj
                .into_iter()
                .map(|(k, mut v)| {
                    v.sort_unstable();
                    v.dedup();
                    (k, Arc::from(v))
                })
                .collect(),
        }
    }

    /// Build a directed adjacency from `(source, target)` pairs.
    /// Deduplicates and sorts neighbor lists.
    pub fn from_directed_edges(edges: &[(u64, u64)]) -> Self {
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for &(u, v) in edges {
            adj.entry(u).or_default().push(v);
        }
        Self::finalize(adj)
    }

    /// Build an undirected adjacency from `(source, target)` pairs.
    /// Each edge is inserted in both directions. Deduplicates and sorts.
    pub fn from_undirected_edges(edges: &[(u64, u64)]) -> Self {
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for &(u, v) in edges {
            adj.entry(u).or_default().push(v);
            adj.entry(v).or_default().push(u);
        }
        Self::finalize(adj)
    }

    /// Build a directed adjacency from parallel source/target slices.
    /// Convenience for accumulators that store edges in two `Vec<u64>`s.
    pub fn from_edge_vecs(sources: &[u64], targets: &[u64]) -> Self {
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&s, &t) in sources.iter().zip(targets.iter()) {
            adj.entry(s).or_default().push(t);
        }
        Self::finalize(adj)
    }

    /// Build a directed adjacency from any `(source, target)` iterator —
    /// e.g. the zero-copy [`GraphAccumulatorBase::edges`] stream.
    pub fn from_edges_iter<I: IntoIterator<Item = (u64, u64)>>(edges: I) -> Self {
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for (u, v) in edges {
            adj.entry(u).or_default().push(v);
        }
        Self::finalize(adj)
    }
}

impl GraphView for SimpleGraph {
    fn get_neighbors(&self, node: u64) -> Vec<u64> {
        self.adjacency
            .get(&node)
            .map(|a| a.to_vec())
            .unwrap_or_default()
    }

    fn get_neighbors_shared(&self, node: u64) -> Arc<[u64]> {
        self.adjacency
            .get(&node)
            .cloned()
            .unwrap_or_else(|| Arc::from(Vec::<u64>::new()))
    }

    fn get_neighbors_into(&self, node: u64, out: &mut Vec<u64>) {
        if let Some(ns) = self.adjacency.get(&node) {
            out.extend_from_slice(ns);
        }
    }

    fn get_degree(&self, node: u64) -> usize {
        self.adjacency.get(&node).map(|v| v.len()).unwrap_or(0)
    }

    fn all_nodes(&self) -> Vec<u64> {
        let mut nodes: Vec<u64> = self.adjacency.keys().copied().collect();
        nodes.sort_unstable();
        nodes
    }

    fn num_edges(&self) -> usize {
        self.adjacency.values().map(|v| v.len()).sum()
    }
}

/// A `GraphView` that caches neighbor lists from an out-of-core backing graph,
/// evicting when the cache exceeds a byte budget.
///
/// This is the adaptive tier: hot frontier nodes are served from RAM, the rest
/// from the mmap CSR, with a hard memory ceiling. A spill that cannot be bounded
/// is just a slower OOM, so the budget is enforced on every insert.
pub struct CachingGraph<G: GraphView> {
    backing: G,
    cache: parking_lot::Mutex<CacheState>,
    budget_bytes: usize,
}

struct CacheState {
    map: HashMap<u64, Arc<[u64]>>,
    order: VecDeque<u64>,
    used_bytes: usize,
}

impl<G: GraphView> CachingGraph<G> {
    pub fn new(backing: G, budget_bytes: usize) -> Self {
        Self {
            backing,
            cache: parking_lot::Mutex::new(CacheState {
                map: HashMap::new(),
                order: VecDeque::new(),
                used_bytes: 0,
            }),
            budget_bytes,
        }
    }
}

fn neighbor_bytes(len: usize) -> usize {
    len.saturating_mul(8).saturating_add(24)
}

impl<G: GraphView> GraphView for CachingGraph<G> {
    fn get_neighbors(&self, node: u64) -> Vec<u64> {
        self.get_neighbors_shared(node).to_vec()
    }

    fn get_neighbors_shared(&self, node: u64) -> Arc<[u64]> {
        {
            let state = self.cache.lock();
            if let Some(v) = state.map.get(&node) {
                // Zero-copy cache hit: hand back the shared pointer.
                return Arc::clone(v);
            }
        }
        let neighbors = self.backing.get_neighbors_shared(node);
        let bytes = neighbor_bytes(neighbors.len());
        let mut state = self.cache.lock();
        // Evict (FIFO) until the new entry fits within the budget.
        while state.used_bytes + bytes > self.budget_bytes && !state.order.is_empty() {
            if let Some(old) = state.order.pop_front() {
                if let Some(v) = state.map.remove(&old) {
                    state.used_bytes = state.used_bytes.saturating_sub(neighbor_bytes(v.len()));
                }
            }
        }
        state.map.insert(node, Arc::clone(&neighbors));
        state.order.push_back(node);
        state.used_bytes += bytes;
        neighbors
    }

    fn get_neighbors_into(&self, node: u64, out: &mut Vec<u64>) {
        {
            let state = self.cache.lock();
            if let Some(v) = state.map.get(&node) {
                out.extend_from_slice(v);
                return;
            }
        }
        // `out` is an append buffer that callers reuse across hops, so cache
        // only the slice this call appended — not any pre-existing prefix.
        let start = out.len();
        self.backing.get_neighbors_into(node, out);
        let appended = &out[start..];
        let bytes = neighbor_bytes(appended.len());
        let mut state = self.cache.lock();
        // Evict (FIFO) until the new entry fits within the budget.
        while state.used_bytes + bytes > self.budget_bytes && !state.order.is_empty() {
            if let Some(old) = state.order.pop_front() {
                if let Some(v) = state.map.remove(&old) {
                    state.used_bytes = state.used_bytes.saturating_sub(neighbor_bytes(v.len()));
                }
            }
        }
        state.map.insert(node, Arc::from(appended));
        state.order.push_back(node);
        state.used_bytes += bytes;
    }

    fn get_degree(&self, node: u64) -> usize {
        self.backing.get_degree(node)
    }

    fn all_nodes(&self) -> Vec<u64> {
        self.backing.all_nodes()
    }

    fn num_edges(&self) -> usize {
        self.backing.num_edges()
    }

    fn all_edges(&self) -> Vec<(u64, u64)> {
        self.backing.all_edges()
    }
}

/// A [`GraphView`] that restricts another graph to a fixed node set.
///
/// Neighbors outside the set are hidden and degrees count only in-set edges, so
/// an out-of-core CSR can be made to present exactly the same induced subgraph
/// as an in-memory [`SimpleGraph`] built over the same region. This is what makes
/// `InMemory` and `OutOfCore` return identical results: the algorithm traverses
/// the same logical graph either way.
pub struct SubgraphView {
    backing: Box<dyn GraphView>,
    nodes: HashSet<u64>,
}

impl SubgraphView {
    pub fn new(backing: Box<dyn GraphView>, nodes: HashSet<u64>) -> Self {
        Self { backing, nodes }
    }
}

impl GraphView for SubgraphView {
    fn get_neighbors(&self, node: u64) -> Vec<u64> {
        if !self.nodes.contains(&node) {
            return Vec::new();
        }
        self.backing
            .get_neighbors(node)
            .into_iter()
            .filter(|n| self.nodes.contains(n))
            .collect()
    }

    fn get_neighbors_into(&self, node: u64, out: &mut Vec<u64>) {
        if !self.nodes.contains(&node) {
            return;
        }
        let start = out.len();
        self.backing.get_neighbors_into(node, out);
        // Keep only in-set neighbors, compacting the tail in place (no
        // second allocation). Slices have no `retain`, so do it by hand.
        let mut write = start;
        for read in start..out.len() {
            if self.nodes.contains(&out[read]) {
                out[write] = out[read];
                write += 1;
            }
        }
        out.truncate(write);
    }

    fn get_neighbors_shared(&self, node: u64) -> Arc<[u64]> {
        // The filtered view cannot borrow the backing's shared list (it must
        // drop out-of-set nodes), so build the filtered list once into an Arc.
        let mut v = Vec::new();
        self.get_neighbors_into(node, &mut v);
        Arc::from(v)
    }

    fn get_degree(&self, node: u64) -> usize {
        if !self.nodes.contains(&node) {
            return 0;
        }
        let mut scratch = Vec::new();
        self.get_neighbors_into(node, &mut scratch);
        scratch.len()
    }

    /// Nodes with at least one in-set outgoing edge.
    ///
    /// This mirrors [`SimpleGraph::all_nodes`] (which only records source
    /// nodes), so the in-memory and CSR-backed modes report the *same* node
    /// set to global algorithms. Returning the whole region would include
    /// target-only nodes and break mode invariance.
    fn all_nodes(&self) -> Vec<u64> {
        let mut nodes: Vec<u64> = self
            .nodes
            .iter()
            .copied()
            .filter(|&n| self.get_degree(n) > 0)
            .collect();
        nodes.sort_unstable();
        nodes
    }

    fn num_edges(&self) -> usize {
        self.nodes.iter().map(|&n| self.get_degree(n)).sum()
    }
}

/// Parse a SQL `mode` string into a [`GraphMode`].
///
/// Unknown values fall back to `Auto` so a typo degrades to the safe adaptive
/// path rather than erroring the query.
pub fn parse_graph_mode(s: &str) -> GraphMode {
    match s.trim().to_ascii_lowercase().as_str() {
        "in_memory" | "inmemory" | "in-memory" => GraphMode::InMemory,
        "out_of_core" | "outofcore" | "out-of-core" => GraphMode::OutOfCore,
        "cached" => GraphMode::Cached,
        _ => GraphMode::Auto,
    }
}

/// Resolve a graph view for `uri` under `mode`, blocking until it is ready.
///
/// The graph UDAFs run inside DataFusion's async context, where a nested
/// `block_on` panics with "Cannot start a runtime from within a runtime".
///
/// When the ambient reactor is **multi-threaded** (DataFusion's default) we
/// reuse it: `block_in_place` parks the current worker so the rest of the pool
/// keeps running, and `Handle::block_on` drives the future on that *shared*
/// runtime. This avoids building a throwaway runtime and spawning a thread on
/// every evaluation. Only a current-thread ambient runtime (where
/// `block_in_place` panics) falls back to a dedicated thread, and only with no
/// ambient runtime at all do we build a temporary one.
#[allow(clippy::expect_used)]
pub fn load_graph_view(
    uri: &str,
    mode: GraphMode,
    seeds: &[u64],
    hops: u32,
) -> anyhow::Result<Box<dyn GraphView>> {
    let uri = uri.to_string();
    let seeds = seeds.to_vec();
    tracing::debug!(
        uri = %uri,
        mode = ?mode,
        seeds = seeds.len(),
        hops,
        "load_graph_view: resolving graph view"
    );
    let future = async move {
        let table = Table::new_async(uri).await?;
        table.graph_view(mode, &seeds, hops).await
    };

    match tokio::runtime::Handle::try_current() {
        // Fast path: reuse the ambient multi-threaded runtime in place.
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(future))
        }
        // A current-thread runtime would panic on `block_in_place`; offload the
        // future to its own single-threaded runtime on a dedicated thread.
        Ok(_) => std::thread::scope(|s| {
            s.spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create Tokio runtime for graph view")
                    .block_on(future)
            })
            .join()
            .expect("graph view runtime thread panicked")
        }),
        // Cold path: no ambient runtime.
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime for graph view")
            .block_on(future),
    }
}

// ---------------------------------------------------------------------------
// GraphAccumulatorBase — shared edge accumulation for all graph UDAFs
// ---------------------------------------------------------------------------

/// Common state that every graph-algorithm UDAF accumulator embeds.
///
/// Collects `(source, target)` edge rows from Arrow batches and resolves them
/// into a [`GraphView`] — either in-memory from the rows, or out-of-core from a
/// CSR index when `graph_uri` is supplied. Also handles the Arrow
/// `state()`/`update_batch()`/`merge_batch()` boilerplate so individual UDAFs
/// only implement `evaluate()`.
#[derive(Debug)]
pub struct GraphAccumulatorBase {
    /// Source-column chunks, retained **zero-copy** from the Arrow batches.
    source_chunks: Vec<ArrayRef>,
    /// Target-column chunks, parallel to `source_chunks`.
    target_chunks: Vec<ArrayRef>,
    pub graph_uri: Option<String>,
    pub mode: Option<String>,
}

/// Sequential `(edge_index, source, target)` iterator over the retained chunks.
///
/// Reads straight out of the Arrow buffers — no intermediate `Vec<u64>`.
pub struct EdgeIter<'a> {
    sources: &'a [ArrayRef],
    targets: &'a [ArrayRef],
    chunk: usize,
    within: usize,
    global: usize,
}

impl Iterator for EdgeIter<'_> {
    type Item = (usize, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.chunk >= self.sources.len() {
                return None;
            }
            let s = self.sources[self.chunk]
                .as_any()
                .downcast_ref::<UInt64Array>()?;
            let t = self.targets[self.chunk]
                .as_any()
                .downcast_ref::<UInt64Array>()?;
            if self.within >= s.len() {
                self.chunk += 1;
                self.within = 0;
                continue;
            }
            let i = self.within;
            self.within += 1;
            let item = (self.global, s.value(i), t.value(i));
            self.global += 1;
            return Some(item);
        }
    }
}

impl Default for GraphAccumulatorBase {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAccumulatorBase {
    pub fn new() -> Self {
        Self {
            source_chunks: Vec::new(),
            target_chunks: Vec::new(),
            graph_uri: None,
            mode: None,
        }
    }

    /// Number of retained edges.
    pub fn edge_count(&self) -> usize {
        self.source_chunks.iter().map(|c| c.len()).sum()
    }

    /// True when no edges have been retained.
    pub fn is_empty(&self) -> bool {
        self.edge_count() == 0
    }

    /// Sequential `(edge_index, source, target)` over all retained chunks.
    pub fn edges(&self) -> EdgeIter<'_> {
        EdgeIter {
            sources: &self.source_chunks,
            targets: &self.target_chunks,
            chunk: 0,
            within: 0,
            global: 0,
        }
    }

    /// Test/utility helper: retain two `u64` slices as a single chunk.
    pub fn set_edges(&mut self, sources: &[u64], targets: &[u64]) {
        self.source_chunks.clear();
        self.target_chunks.clear();
        self.source_chunks
            .push(Arc::new(UInt64Array::from(sources.to_vec())));
        self.target_chunks
            .push(Arc::new(UInt64Array::from(targets.to_vec())));
    }

    /// Build the graph: CSR when `graph_uri` is set, else in-memory directly
    /// from the retained edge chunks (no intermediate `Vec<u64>`).
    ///
    /// `seeds` and `hops` are used for regional restriction when loading from
    /// the CSR. Pass `(&[], 0)` for global (unrestricted) algorithms.
    pub fn resolve_graph(&self, seeds: &[u64], hops: u32) -> Result<Box<dyn GraphView>> {
        if let Some(uri) = self.graph_uri.as_deref().filter(|u| !u.is_empty()) {
            let mode = parse_graph_mode(self.mode.as_deref().unwrap_or("auto"));
            return load_graph_view(uri, mode, seeds, hops)
                .map_err(|e| DataFusionError::Execution(format!("graph_view failed: {e}")));
        }
        Ok(Box::new(SimpleGraph::from_edges_iter(
            self.edges().map(|(_, u, v)| (u, v)),
        )))
    }

    /// Extract `(source, target)` columns from positions 0 and 1 of `values`.
    /// Optionally reads `graph_uri` and `mode` from positions `uri_idx` and
    /// `mode_idx` (pass `None` to skip).
    pub fn update_edge_batch(
        &mut self,
        values: &[ArrayRef],
        uri_idx: Option<usize>,
        mode_idx: Option<usize>,
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let sources = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected UInt64Array for sources".to_string())
            })?;
        let targets = values[1]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected UInt64Array for targets".to_string())
            })?;

        if sources.null_count() == 0 && targets.null_count() == 0 {
            // Zero-copy: retain the batch's own columns (Arc bumps only).
            self.source_chunks.push(values[0].clone());
            self.target_chunks.push(values[1].clone());
        } else {
            // Null-bearing input: retain only valid rows (one compact copy).
            let keep = |i: usize| sources.is_valid(i) && targets.is_valid(i);
            let s: Vec<u64> = (0..sources.len())
                .filter(|&i| keep(i))
                .map(|i| sources.value(i))
                .collect();
            let t: Vec<u64> = (0..targets.len())
                .filter(|&i| keep(i))
                .map(|i| targets.value(i))
                .collect();
            self.source_chunks.push(Arc::new(UInt64Array::from(s)));
            self.target_chunks.push(Arc::new(UInt64Array::from(t)));
        }

        if let Some(idx) = uri_idx {
            if self.graph_uri.is_none() && values.len() > idx && !values[idx].is_empty() {
                if let Some(arr) = values[idx].as_any().downcast_ref::<StringArray>() {
                    if arr.is_valid(0) {
                        self.graph_uri = Some(arr.value(0).to_string());
                    }
                }
            }
        }

        if let Some(idx) = mode_idx {
            if self.mode.is_none() && values.len() > idx && !values[idx].is_empty() {
                if let Some(arr) = values[idx].as_any().downcast_ref::<StringArray>() {
                    if arr.is_valid(0) {
                        self.mode = Some(arr.value(0).to_string());
                    }
                }
            }
        }

        Ok(())
    }

    /// True if a partial-aggregation state carries at least one edge row.
    ///
    /// DataFusion fans the aggregate out over `target_partitions` input
    /// partitions; every partition that holds no rows still produces a partial
    /// state whose scalar arguments are the accumulator defaults. Callers must
    /// not adopt those defaults, or the result becomes dependent on the
    /// non-deterministic order in which partials are merged.
    pub fn state_has_edges(states: &[ArrayRef]) -> bool {
        states
            .first()
            .and_then(|s| s.as_any().downcast_ref::<ListArray>())
            .map(|l| (0..l.len()).any(|i| l.is_valid(i) && !l.value(i).is_empty()))
            .unwrap_or(false)
    }

    /// Merge edge state from partial aggregation results.
    /// `uri_idx` / `mode_idx` are the positions in the state array for the
    /// optional `graph_uri` and `mode` fields.
    pub fn merge_edge_state(
        &mut self,
        states: &[ArrayRef],
        uri_idx: Option<usize>,
        mode_idx: Option<usize>,
    ) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }

        let sources_list = states[0]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected ListArray for sources state".to_string())
            })?;
        let targets_list = states[1]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected ListArray for targets state".to_string())
            })?;

        for i in 0..sources_list.len() {
            if sources_list.is_valid(i) {
                let inner = sources_list.value(i);
                // Retain the state's inner array directly (zero-copy).
                if inner.as_any().is::<UInt64Array>() {
                    self.source_chunks.push(inner);
                }
            }
            if targets_list.is_valid(i) {
                let inner = targets_list.value(i);
                if inner.as_any().is::<UInt64Array>() {
                    self.target_chunks.push(inner);
                }
            }
        }

        if let Some(idx) = uri_idx {
            if self.graph_uri.is_none() && states.len() > idx {
                if let Some(arr) = states[idx].as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        if arr.is_valid(i) {
                            self.graph_uri = Some(arr.value(i).to_string());
                            break;
                        }
                    }
                }
            }
        }

        if let Some(idx) = mode_idx {
            if self.mode.is_none() && states.len() > idx {
                if let Some(arr) = states[idx].as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        if arr.is_valid(i) {
                            self.mode = Some(arr.value(i).to_string());
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Serialize edge state for partial aggregation.
    pub fn edge_state(&self) -> Result<Vec<ScalarValue>> {
        let src = concat_u64_chunks(&self.source_chunks)?;
        let tgt = concat_u64_chunks(&self.target_chunks)?;

        let mut sources_builder = ListBuilder::new(UInt64Builder::new());
        sources_builder.values().append_slice(&src);
        sources_builder.append(true);

        let mut targets_builder = ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&tgt);
        targets_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::Utf8(self.graph_uri.clone()),
            ScalarValue::Utf8(self.mode.clone()),
        ])
    }

    /// The standard state fields every graph UDAF needs for edge data.
    pub fn state_fields() -> Vec<Arc<Field>> {
        vec![
            Arc::new(Field::new(
                "sources",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new(
                "targets",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("graph_uri", DataType::Utf8, true)),
            Arc::new(Field::new("mode", DataType::Utf8, true)),
        ]
    }

    /// Byte size estimate for memory tracking.
    pub fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self
                .source_chunks
                .iter()
                .map(|c| c.get_array_memory_size())
                .sum::<usize>()
            + self
                .target_chunks
                .iter()
                .map(|c| c.get_array_memory_size())
                .sum::<usize>()
            + self.graph_uri.as_ref().map(|s| s.capacity()).unwrap_or(0)
            + self.mode.as_ref().map(|s| s.capacity()).unwrap_or(0)
    }
}

/// Concatenate `UInt64Array` chunks into one `Vec<u64>` (used only at the
/// partial-aggregation boundary, not on the hot traversal path).
fn concat_u64_chunks(chunks: &[ArrayRef]) -> Result<Vec<u64>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    if chunks.len() == 1 {
        let a = chunks[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Execution("expected UInt64 edge chunk".to_string()))?;
        return Ok(a.values().to_vec());
    }
    let refs: Vec<&dyn Array> = chunks.iter().map(|c| c.as_ref()).collect();
    let arr = arrow::compute::concat(&refs)
        .map_err(|e| DataFusionError::Execution(format!("edge state concat: {e}")))?;
    let a = arr
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Execution("expected UInt64 edge concat".to_string()))?;
    Ok(a.values().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_graph(n: u64) -> SimpleGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            if i + 1 < n {
                edges.push((i, i + 1));
            }
        }
        SimpleGraph::from_undirected_edges(&edges)
    }

    #[test]
    fn simple_graph_neighbors_and_degree() {
        let g = line_graph(4);
        assert_eq!(g.get_neighbors(0), vec![1]);
        assert_eq!(g.get_neighbors(1), vec![0, 2]);
        assert_eq!(g.get_degree(1), 2);
        assert_eq!(g.get_degree(99), 0);
    }

    /// `get_neighbors_into` must equal `get_neighbors` and **append** (so a
    /// caller can reuse one scratch buffer across hops without reallocating).
    #[test]
    fn get_neighbors_into_matches_and_appends() {
        let g = SimpleGraph::from_undirected_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)]);

        let mut buf = vec![999u64]; // pre-filled: must survive
        g.get_neighbors_into(1, &mut buf);
        let mut expect = vec![999u64];
        expect.extend(g.get_neighbors(1));
        assert_eq!(buf, expect, "get_neighbors_into must append, not clobber");

        // Reuse: the caller clears, then refills.
        buf.clear();
        g.get_neighbors_into(2, &mut buf);
        assert_eq!(buf, g.get_neighbors(2));

        // Unknown node appends nothing.
        buf.clear();
        g.get_neighbors_into(42, &mut buf);
        assert!(buf.is_empty());
    }

    /// `SubgraphView` filters the tail in place; the result must equal the
    /// filtered `get_neighbors`, and degree must count only in-set edges.
    #[test]
    fn subgraph_view_into_matches_filtered_neighbors() {
        let backing = Box::new(SimpleGraph::from_undirected_edges(&[
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
        ])) as Box<dyn GraphView>;
        let nodes: HashSet<u64> = [0u64, 1, 2].into_iter().collect();
        let sub = SubgraphView::new(backing, nodes);

        let mut buf = Vec::new();
        sub.get_neighbors_into(0, &mut buf);
        assert_eq!(buf, sub.get_neighbors(0));
        assert_eq!(sub.get_degree(0), 2);

        // Node 2's neighbor 3 is outside the set and must be filtered out.
        buf.clear();
        sub.get_neighbors_into(2, &mut buf);
        assert_eq!(buf, sub.get_neighbors(2));
        assert_eq!(sub.get_degree(2), 2);
    }

    /// `CachingGraph` must agree on the cold path (backing) and warm path (cache).
    #[test]
    fn caching_graph_into_matches_backing() {
        let backing = SimpleGraph::from_undirected_edges(&[(0, 1), (1, 2), (2, 3)]);
        let cached = CachingGraph::new(backing, 1 << 20);
        let mut buf = Vec::new();
        for n in 0u64..4 {
            buf.clear();
            cached.get_neighbors_into(n, &mut buf);
            assert_eq!(buf, cached.get_neighbors(n));
        }
        // Second pass is served from the populated cache.
        buf.clear();
        cached.get_neighbors_into(1, &mut buf);
        assert_eq!(buf, vec![0, 2]);
    }

    #[test]
    fn from_directed_edges_deduplicates() {
        let g = SimpleGraph::from_directed_edges(&[(1, 2), (1, 2), (2, 3)]);
        assert_eq!(g.get_neighbors(1), vec![2]);
        assert_eq!(g.get_degree(1), 1);
        assert_eq!(g.get_neighbors(2), vec![3]);
    }

    #[test]
    fn from_undirected_edges_symmetrizes() {
        let g = SimpleGraph::from_undirected_edges(&[(1, 2), (2, 3)]);
        assert_eq!(g.get_neighbors(1), vec![2]);
        assert_eq!(g.get_neighbors(2), vec![1, 3]);
        assert_eq!(g.get_neighbors(3), vec![2]);
    }

    #[test]
    fn from_edge_vecs_matches_pairs() {
        let g1 = SimpleGraph::from_directed_edges(&[(1, 2), (2, 3)]);
        let g2 = SimpleGraph::from_edge_vecs(&[1, 2], &[2, 3]);
        assert_eq!(g1.get_neighbors(1), g2.get_neighbors(1));
        assert_eq!(g1.get_neighbors(2), g2.get_neighbors(2));
    }

    #[test]
    fn all_nodes_returns_sorted() {
        let g = SimpleGraph::from_directed_edges(&[(3, 1), (1, 2)]);
        assert_eq!(g.all_nodes(), vec![1, 3]);
    }

    #[test]
    fn caching_graph_matches_backing() {
        let backing = line_graph(10);
        let cached = CachingGraph::new(backing, 1024);
        for i in 0..10 {
            assert_eq!(cached.get_neighbors(i), cached.get_neighbors(i));
        }
        assert_eq!(cached.get_degree(5), 2);
    }

    /// Zero-copy: `SimpleGraph` hands back the same shared neighbor list.
    #[test]
    fn simple_graph_shared_is_zero_copy() {
        let g = SimpleGraph::from_undirected_edges(&[(0, 1), (1, 2), (2, 3)]);
        let a = g.get_neighbors_shared(1);
        let b = g.get_neighbors_shared(1);
        assert!(
            Arc::ptr_eq(&a, &b),
            "SimpleGraph::get_neighbors_shared must return the same Arc"
        );
    }

    /// Zero-copy: a `CachingGraph` cache hit returns the shared pointer.
    #[test]
    fn caching_graph_shared_is_zero_copy_on_hit() {
        let backing = SimpleGraph::from_undirected_edges(&[(0, 1), (1, 2), (2, 3)]);
        let cached = CachingGraph::new(backing, 1 << 20);
        let _ = cached.get_neighbors_shared(1); // populate the cache
        let a = cached.get_neighbors_shared(1);
        let b = cached.get_neighbors_shared(1);
        assert!(
            Arc::ptr_eq(&a, &b),
            "CachingGraph cache hit must be zero-copy"
        );
        assert_eq!(&*a, &[0u64, 2][..]);
    }

    #[test]
    fn estimate_scales_with_nodes() {
        assert!(estimate_subgraph_bytes(1000, 16) > estimate_subgraph_bytes(100, 16));
        assert_eq!(estimate_subgraph_bytes(0, 16), 0);
    }

    #[test]
    fn accumulator_base_resolve_in_memory() -> Result<()> {
        let mut base = GraphAccumulatorBase::new();
        base.set_edges(&[1, 2, 3], &[2, 3, 1]);
        let graph = base.resolve_graph(&[], 0)?;
        assert_eq!(graph.get_degree(1), 1);
        assert_eq!(graph.get_neighbors(1), vec![2]);
        Ok(())
    }

    /// Edges are retained as (borrowed) Arrow chunks and read back directly;
    /// partial-aggregation state round-trips through `edge_state`/`merge_edge_state`.
    #[test]
    fn accumulator_retains_arrow_chunks_and_round_trips_state() -> Result<()> {
        let mut base = GraphAccumulatorBase::new();
        let s: ArrayRef = Arc::new(UInt64Array::from(vec![1u64, 2]));
        let t: ArrayRef = Arc::new(UInt64Array::from(vec![3u64, 4]));
        base.update_edge_batch(&[s, t], None, None)?;

        assert_eq!(base.edge_count(), 2);
        assert_eq!(base.edges().collect::<Vec<_>>(), vec![(0, 1, 3), (1, 2, 4)]);

        // State serialization then merge preserves the edges. DataFusion passes
        // state as arrays; convert the ScalarValues the same way.
        let state: Vec<ArrayRef> = base
            .edge_state()?
            .iter()
            .map(|s| s.to_array())
            .collect::<Result<Vec<_>>>()?;
        let mut merged = GraphAccumulatorBase::new();
        merged.merge_edge_state(&state, None, None)?;
        assert_eq!(merged.edge_count(), 2);
        assert_eq!(
            merged.edges().collect::<Vec<_>>(),
            vec![(0, 1, 3), (1, 2, 4)]
        );
        Ok(())
    }

    /// An empty partial-aggregation state must be distinguishable from a
    /// non-empty one, so UDAFs can avoid adopting default scalar arguments
    /// from empty input partitions (which would make results depend on the
    /// non-deterministic merge order).
    #[test]
    fn state_has_edges_distinguishes_empty_partials() -> Result<()> {
        let empty = GraphAccumulatorBase::new();
        let empty_state: Vec<ArrayRef> = empty
            .edge_state()?
            .iter()
            .map(|s| s.to_array())
            .collect::<Result<Vec<_>>>()?;
        assert!(!GraphAccumulatorBase::state_has_edges(&empty_state));

        let mut non_empty = GraphAccumulatorBase::new();
        let s: ArrayRef = Arc::new(UInt64Array::from(vec![1u64]));
        let t: ArrayRef = Arc::new(UInt64Array::from(vec![2u64]));
        non_empty.update_edge_batch(&[s, t], None, None)?;
        let non_empty_state: Vec<ArrayRef> = non_empty
            .edge_state()?
            .iter()
            .map(|s| s.to_array())
            .collect::<Result<Vec<_>>>()?;
        assert!(GraphAccumulatorBase::state_has_edges(&non_empty_state));

        Ok(())
    }
}
