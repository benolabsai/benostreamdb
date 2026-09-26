// Copyright (c) 2026 Richard Albright. All rights reserved.

//! CSR-backed community detection (Louvain / Leiden).
//!
//! These read adjacency from the memory-mapped CSR instead of an in-RAM edge
//! accumulator, so only **O(V)** state is resident (`community`, `degree`,
//! `comm_tot`). The CSR is unweighted — every edge weight is 1.0 — matching the
//! SQL `louvain_communities` default. On a 6M-node graph the resident state is
//! ~72 MB regardless of the edge count, versus ~7.7 GB for the 383M-edge
//! accumulator path.
//!
//! Both algorithms share the greedy modularity local-move phase. Leiden adds a
//! refinement phase that splits each community into the connected components of
//! its induced subgraph, so every returned community is internally connected.
//!
//! ## Incremental updates
//!
//! [`csr_louvain_seeded`] / [`csr_leiden_seeded`] warm-start from a previous
//! partition (a map of node → community id). Unchanged communities keep their
//! ids, so a graph that grows by a few edges does not renumber every community.

use crate::core::index::csr_graph::MultiSegmentCsrGraph;
use crate::core::sql::graph_udf::drift_search::DriftGraph;
use std::collections::{HashMap, HashSet};

/// A community: a stable id plus its member node ids (sorted).
pub type Community = (u32, Vec<u64>);

/// CSR adjacency with a global dense node index and O(V) degree state.
struct CsrAdj<'a> {
    forward: &'a MultiSegmentCsrGraph,
    reverse: Option<&'a MultiSegmentCsrGraph>,
    node_map: HashMap<u64, usize>,
    originals: Vec<u64>,
    degree: Vec<f32>,
    total_weight: f32,
}

impl<'a> CsrAdj<'a> {
    fn build(forward: &'a MultiSegmentCsrGraph, reverse: Option<&'a MultiSegmentCsrGraph>) -> Self {
        // Enumerate every node once, assigning a dense index.
        let mut node_map: HashMap<u64, usize> = HashMap::new();
        let mut originals: Vec<u64> = Vec::new();
        for seg in &forward.segments {
            for d in 0..seg.num_nodes {
                let orig = seg.to_original(d);
                if let std::collections::hash_map::Entry::Vacant(e) = node_map.entry(orig) {
                    e.insert(originals.len());
                    originals.push(orig);
                }
            }
        }

        let n = originals.len();
        let mut degree = vec![0.0f32; n];
        let mut total_weight = 0.0f32;

        // Each edge appears once in `forward`. Undirected (a reverse CSR is
        // present) adds it to both endpoints' degrees; directed counts
        // out-degree only. O(E) time, O(V) memory.
        let undirected = reverse.is_some();
        for seg in &forward.segments {
            for d in 0..seg.num_nodes {
                let u = match node_map.get(&seg.to_original(d)) {
                    Some(&u) => u,
                    None => continue,
                };
                for e in seg.get_neighbors_raw(d) {
                    let v_orig = seg.to_original(e.dst_id as usize);
                    let Some(&v) = node_map.get(&v_orig) else {
                        continue;
                    };
                    degree[u] += 1.0;
                    if undirected {
                        degree[v] += 1.0;
                    }
                    total_weight += 1.0;
                }
            }
        }

        Self {
            forward,
            reverse,
            node_map,
            originals,
            degree,
            total_weight,
        }
    }

    fn num_nodes(&self) -> usize {
        self.originals.len()
    }

    /// Dense indices of `u`'s neighbours: out-neighbours, plus in-neighbours
    /// when a reverse CSR is present (undirected semantics).
    fn neighbors(&self, u: usize) -> Vec<usize> {
        let orig = self.originals[u];
        let mut out = Vec::new();
        for n in self.forward.get_neighbors(orig) {
            if let Some(&d) = self.node_map.get(&n) {
                out.push(d);
            }
        }
        if let Some(rev) = self.reverse {
            for n in rev.get_neighbors(orig) {
                if let Some(&d) = self.node_map.get(&n) {
                    out.push(d);
                }
            }
        }
        out
    }
}

/// Split each community into the connected components of its induced subgraph,
/// preserving the community id for the first component so ids stay stable.
fn refine_communities(adj: &CsrAdj<'_>, community: &[u32], next_id: &mut u32) -> Vec<u32> {
    let n = community.len();
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for (u, &c) in community.iter().enumerate() {
        members.entry(c).or_default().push(u);
    }

    let mut refined = community.to_vec();
    let mut seen = vec![false; n];

    for (&c, nodes) in &members {
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let mut first_component = true;
        for &start in nodes {
            if seen[start] {
                continue;
            }
            // The first component keeps the original id; splits get fresh ids.
            let id = if first_component {
                first_component = false;
                c
            } else {
                let id = *next_id;
                *next_id += 1;
                id
            };
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(x) = stack.pop() {
                refined[x] = id;
                for v in adj.neighbors(x) {
                    if node_set.contains(&v) && !seen[v] {
                        seen[v] = true;
                        stack.push(v);
                    }
                }
            }
        }
    }

    refined
}

fn partition(
    adj: &CsrAdj<'_>,
    resolution: f32,
    refine: bool,
    seed: Option<&HashMap<u64, u32>>,
) -> Vec<Community> {
    let n = adj.num_nodes();
    if n == 0 {
        return Vec::new();
    }
    let two_m = (adj.total_weight * 2.0).max(1e-6);

    // Warm start from the seed partition (stable ids), else singletons.
    let mut community: Vec<u32> = vec![0; n];
    let mut next_id: u32 = 0;
    match seed {
        Some(seed) => {
            for u in 0..n {
                match seed.get(&adj.originals[u]) {
                    Some(&c) => {
                        community[u] = c;
                        next_id = next_id.max(c.saturating_add(1));
                    }
                    None => {
                        community[u] = next_id;
                        next_id += 1;
                    }
                }
            }
        }
        None => {
            for (u, c) in community.iter_mut().enumerate() {
                *c = u as u32;
            }
            next_id = n as u32;
        }
    }

    // Per-community total degree, from the initial assignment.
    let mut comm_tot: Vec<f32> = vec![0.0; next_id as usize + n + 1];
    for u in 0..n {
        comm_tot[community[u] as usize] += adj.degree[u];
    }

    for _pass in 0..15 {
        let mut moved = false;

        for u in 0..n {
            let c_u = community[u] as usize;
            let k_u = adj.degree[u];
            comm_tot[c_u] -= k_u;

            let mut comm_weights: HashMap<u32, f32> = HashMap::new();
            for v in adj.neighbors(u) {
                if v != u {
                    *comm_weights.entry(community[v]).or_default() += 1.0;
                }
            }

            let mut best_c = c_u as u32;
            let k_in_curr = *comm_weights.get(&(c_u as u32)).unwrap_or(&0.0);
            let mut best_delta = k_in_curr - resolution * (comm_tot[c_u] * k_u) / two_m;
            for (&c, &k_in) in &comm_weights {
                let delta = k_in - resolution * (comm_tot[c as usize] * k_u) / two_m;
                if delta > best_delta {
                    best_delta = delta;
                    best_c = c;
                }
            }

            if best_c != c_u as u32 {
                moved = true;
            }
            community[u] = best_c;
            comm_tot[best_c as usize] += k_u;
        }

        if !moved {
            break;
        }
    }

    let final_ids = if refine {
        refine_communities(adj, &community, &mut next_id)
    } else {
        community
    };

    let mut groups: HashMap<u32, Vec<u64>> = HashMap::new();
    for (u, &c) in final_ids.iter().enumerate() {
        groups.entry(c).or_default().push(adj.originals[u]);
    }

    let mut out: Vec<Community> = groups
        .into_iter()
        .map(|(c, mut g)| {
            g.sort_unstable();
            (c, g)
        })
        .collect();
    out.sort_by_key(|(c, _)| *c);
    out
}

/// Louvain communities over the CSR (O(V) resident state).
pub fn csr_louvain(
    forward: &MultiSegmentCsrGraph,
    reverse: Option<&MultiSegmentCsrGraph>,
    resolution: f32,
) -> Vec<Community> {
    let adj = CsrAdj::build(forward, reverse);
    partition(&adj, resolution, false, None)
}

/// Leiden communities over the CSR: local move + connectivity refinement.
pub fn csr_leiden(
    forward: &MultiSegmentCsrGraph,
    reverse: Option<&MultiSegmentCsrGraph>,
    resolution: f32,
) -> Vec<Community> {
    let adj = CsrAdj::build(forward, reverse);
    partition(&adj, resolution, true, None)
}

/// Louvain, warm-started from a previous partition so unchanged communities keep
/// their ids.
pub fn csr_louvain_seeded(
    forward: &MultiSegmentCsrGraph,
    reverse: Option<&MultiSegmentCsrGraph>,
    resolution: f32,
    seed: &HashMap<u64, u32>,
) -> Vec<Community> {
    let adj = CsrAdj::build(forward, reverse);
    partition(&adj, resolution, false, Some(seed))
}

/// Leiden, warm-started from a previous partition so unchanged communities keep
/// their ids.
pub fn csr_leiden_seeded(
    forward: &MultiSegmentCsrGraph,
    reverse: Option<&MultiSegmentCsrGraph>,
    resolution: f32,
    seed: &HashMap<u64, u32>,
) -> Vec<Community> {
    let adj = CsrAdj::build(forward, reverse);
    partition(&adj, resolution, true, Some(seed))
}
