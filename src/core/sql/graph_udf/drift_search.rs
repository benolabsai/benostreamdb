// Copyright (c) 2026 Richard Albright. All rights reserved.
//
// Portions of this file are adapted from the Microsoft GraphRAG project
// (https://github.com/microsoft/graphrag), which is licensed under the
// MIT License. Copyright (c) Microsoft Corporation.
//
// The DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) search
// algorithm, including the multi-phase primer/follow-up/reduction architecture,
// the DriftAction search tree, and the DriftQueryState traversal management,
// has been adapted to use HyperStreamDB-native graph primitives.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub trait DriftGraph: Send + Sync {
    fn get_neighbors(&self, node: u64) -> Vec<u64>;
    fn get_degree(&self, node: u64) -> usize;
}

pub struct SimpleGraph {
    pub adjacency: HashMap<u64, Vec<u64>>,
}

impl DriftGraph for SimpleGraph {
    fn get_neighbors(&self, node: u64) -> Vec<u64> {
        self.adjacency.get(&node).cloned().unwrap_or_default()
    }

    fn get_degree(&self, node: u64) -> usize {
        self.adjacency.get(&node).map(|v| v.len()).unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAction {
    pub action_id: u64,
    pub query: String,
    pub query_seeds: Vec<u64>,
    pub score: f64,
    pub nodes_discovered: Vec<u64>,
    pub is_complete: bool,
    pub parent_id: Option<u64>,
    pub round_num: u32,
    pub children: Vec<u64>,
}

pub struct DriftQueryState {
    pub actions: HashMap<u64, DriftAction>,
    pub next_id: u64,
}

impl Default for DriftQueryState {
    fn default() -> Self {
        Self::new()
    }
}

impl DriftQueryState {
    pub fn new() -> Self {
        Self {
            actions: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn add_action(
        &mut self,
        query: String,
        query_seeds: Vec<u64>,
        score: f64,
        parent_id: Option<u64>,
        round_num: u32,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.actions.insert(
            id,
            DriftAction {
                action_id: id,
                query,
                query_seeds,
                score,
                nodes_discovered: Vec::new(),
                is_complete: false,
                parent_id,
                round_num,
                children: Vec::new(),
            },
        );

        if let Some(pid) = parent_id {
            if let Some(parent) = self.actions.get_mut(&pid) {
                parent.children.push(id);
            }
        }

        id
    }

    pub fn mark_complete(&mut self, action_id: u64, nodes_discovered: Vec<u64>) {
        if let Some(action) = self.actions.get_mut(&action_id) {
            action.nodes_discovered = nodes_discovered;
            action.is_complete = true;
        }
    }

    pub fn rank_incomplete_actions(&self) -> Vec<&DriftAction> {
        let mut incomplete: Vec<&DriftAction> =
            self.actions.values().filter(|a| !a.is_complete).collect();

        incomplete.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        incomplete
    }
}

pub trait DriftFollowUpGenerator {
    /// Generate initial primer actions from the query and top communities.
    fn generate_primer(&self, query: &str, top_communities: &[u64])
        -> Vec<(String, f64, Vec<u64>)>;

    /// Generate follow-up actions from the newly discovered nodes in an epoch.
    fn generate_follow_ups(
        &self,
        query: &str,
        discovered_nodes: &[u64],
        round_num: u32,
    ) -> Vec<(String, f64, Vec<u64>)>;
}

pub struct HeuristicFollowUpGenerator<'a> {
    pub graph: &'a dyn DriftGraph,
    pub community_map: &'a HashMap<u64, u64>,
}

impl<'a> DriftFollowUpGenerator for HeuristicFollowUpGenerator<'a> {
    fn generate_primer(
        &self,
        query: &str,
        top_communities: &[u64],
    ) -> Vec<(String, f64, Vec<u64>)> {
        // Collect nodes from top communities
        let mut seeds = Vec::new();
        for (node, comm) in self.community_map.iter() {
            if top_communities.contains(comm) {
                // Select high-degree nodes in these communities as seeds
                seeds.push(*node);
            }
        }
        // Limit seeds
        seeds.sort_by_key(|&n| std::cmp::Reverse(self.graph.get_degree(n)));
        seeds.truncate(10);

        vec![(query.to_string(), 1.0, seeds)]
    }

    fn generate_follow_ups(
        &self,
        _query: &str,
        discovered_nodes: &[u64],
        _round_num: u32,
    ) -> Vec<(String, f64, Vec<u64>)> {
        // Heuristic: Find bridge nodes connected to the discovered set that lead to new communities
        let mut border_nodes = HashMap::new();
        let discovered_set: HashSet<u64> = discovered_nodes.iter().copied().collect();

        for &node in discovered_nodes {
            for neighbor in self.graph.get_neighbors(node) {
                if !discovered_set.contains(&neighbor) {
                    *border_nodes.entry(neighbor).or_insert(0) += 1;
                }
            }
        }

        let mut border_vec: Vec<_> = border_nodes.into_iter().collect();
        border_vec.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        border_vec.truncate(5);

        let seeds: Vec<u64> = border_vec.into_iter().map(|(n, _)| n).collect();

        if seeds.is_empty() {
            Vec::new()
        } else {
            vec![("Heuristic Follow-up".to_string(), 0.8, seeds)]
        }
    }
}

pub struct DriftSearchParams {
    pub n_depth: u32,
    pub k_followups: usize,
    pub top_k: usize,
    pub hops: u32,
    pub alpha: f64, // PPR damping
    pub confidence_threshold: f64,
}

pub struct DriftSearchResult {
    pub all_discovered_nodes: Vec<u64>,
    pub actions: Vec<DriftAction>,
}

/// Executes Personalized PageRank from the seed nodes
fn local_search_ppr(graph: &dyn DriftGraph, seeds: &[u64], params: &DriftSearchParams) -> Vec<u64> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    let num_seeds = seeds.len() as f32;
    if num_seeds == 0.0 {
        return Vec::new();
    }

    // Initialize seed scores
    for &seed in seeds {
        scores.insert(seed, 1.0 / num_seeds);
    }

    let iterations = 30; // standard PPR iterations
    let damping = params.alpha as f32;

    for _ in 0..iterations {
        let mut new_scores: HashMap<u64, f32> = HashMap::new();
        // Add random jump probability to seeds
        for &seed in seeds {
            new_scores.insert(seed, (1.0 - damping) / num_seeds);
        }

        // Distribute PageRank along edges
        for (&u, &score) in &scores {
            let degree = graph.get_degree(u) as f32;

            if degree > 0.0 {
                let transfer = (damping * score) / degree;
                for v in graph.get_neighbors(u) {
                    *new_scores.entry(v).or_insert(0.0) += transfer;
                }
            } else {
                // Dangling node: distribute its score among seeds
                let transfer = (damping * score) / num_seeds;
                for &seed in seeds {
                    *new_scores.entry(seed).or_insert(0.0) += transfer;
                }
            }
        }
        scores = new_scores;
    }

    let mut ranked_nodes: Vec<(u64, f32)> = scores.into_iter().collect();
    ranked_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked_nodes.truncate(params.top_k);

    ranked_nodes.into_iter().map(|(n, _)| n).collect()
}

pub fn execute_drift_search(
    query: &str,
    graph: &dyn DriftGraph,
    top_communities: &[u64],
    generator: &dyn DriftFollowUpGenerator,
    params: &DriftSearchParams,
) -> DriftSearchResult {
    let mut state = DriftQueryState::new();

    // 1. Primer Phase
    let primers = generator.generate_primer(query, top_communities);
    for (q, score, seeds) in primers {
        state.add_action(q, seeds, score, None, 0);
    }

    // 2. Epoch Loop
    let mut all_discovered: HashSet<u64> = HashSet::new();

    for epoch in 0..params.n_depth {
        let incomplete = state.rank_incomplete_actions();
        if incomplete.is_empty() {
            break;
        }

        // Take top k_followups actions
        let actions_to_run: Vec<u64> = incomplete
            .into_iter()
            .take(params.k_followups)
            .filter(|a| a.score >= params.confidence_threshold)
            .map(|a| a.action_id)
            .collect();

        if actions_to_run.is_empty() {
            break; // All remaining are below threshold
        }

        let mut newly_discovered_epoch = Vec::new();

        for action_id in actions_to_run {
            let seeds = state.actions.get(&action_id).unwrap().query_seeds.clone();

            // Execute local search (PPR)
            let discovered = local_search_ppr(graph, &seeds, params);

            for &node in &discovered {
                all_discovered.insert(node);
                newly_discovered_epoch.push(node);
            }

            state.mark_complete(action_id, discovered);
        }

        // Generate follow-ups for the next epoch based on discoveries
        let follow_ups = generator.generate_follow_ups(query, &newly_discovered_epoch, epoch + 1);
        for (q, score, seeds) in follow_ups {
            // we link it to the first action we ran this epoch as a proxy,
            // or we could just link it as a root. For now, root.
            state.add_action(q, seeds, score, None, epoch + 1);
        }
    }

    DriftSearchResult {
        all_discovered_nodes: all_discovered.into_iter().collect(),
        actions: state.actions.into_values().collect(),
    }
}
