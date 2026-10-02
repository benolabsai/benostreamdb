// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
// Portions of this file are adapted from the Microsoft GraphRAG project
// (https://github.com/microsoft/graphrag), which is licensed under the
// MIT License. Copyright (c) Microsoft Corporation.
//
// The DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) search
// algorithm, including the multi-phase primer/follow-up/reduction architecture,
// the DriftAction search tree, and the DriftQueryState traversal management,
// has been adapted to use BenoStreamDB-native graph primitives.

use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use arrow::array::{
    Array, ArrayRef, ListArray, ListBuilder, StringArray, UInt32Array, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field};
use datafusion::error::Result;
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::sync::Arc;

// The graph-view abstraction (trait, mode, in-memory + caching impls, budget)
// lives in `graph_view` so every graph algorithm shares it.
pub use super::graph_view::{
    estimate_subgraph_bytes, graph_memory_budget_bytes, load_graph_view, parse_graph_mode,
    CachingGraph, GraphAccumulatorBase, GraphMode, GraphView, SimpleGraph,
};

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

        incomplete.sort_by(|a, b| match (a.score.is_nan(), b.score.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.score.total_cmp(&a.score),
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
    pub graph: &'a dyn GraphView,
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
        round_num: u32,
    ) -> Vec<(String, f64, Vec<u64>)> {
        // Heuristic: Find bridge nodes connected to the discovered set that
        // lead to new communities. Score by cross-community novelty.
        let mut border_nodes = HashMap::new();
        let discovered_set: HashSet<u64> = discovered_nodes.iter().copied().collect();

        let mut neighbors: Vec<u64> = Vec::new();
        for &node in discovered_nodes {
            neighbors.clear();
            self.graph.get_neighbors_into(node, &mut neighbors);
            for &neighbor in &neighbors {
                if !discovered_set.contains(&neighbor) {
                    *border_nodes.entry(neighbor).or_insert(0usize) += 1;
                }
            }
        }

        if border_nodes.is_empty() {
            return Vec::new();
        }

        // Group border nodes by community for cross-community diversity
        let mut community_groups: HashMap<u64, Vec<(u64, usize)>> = HashMap::new();
        for (&node, &count) in &border_nodes {
            let comm = self.community_map.get(&node).copied().unwrap_or(u64::MAX);
            community_groups
                .entry(comm)
                .or_default()
                .push((node, count));
        }

        // Decay follow-up scores by round so deeper exploration tapers
        let base_score = 0.8 * 0.7_f64.powi(round_num as i32);

        // Generate one action per new community (up to 5), picking the
        // highest-connectivity border node from each
        let mut sorted_comms: Vec<(u64, Vec<(u64, usize)>)> =
            community_groups.into_iter().collect();
        sorted_comms.sort_by_key(|(_, nodes)| {
            std::cmp::Reverse(nodes.iter().map(|(_, c)| *c).sum::<usize>())
        });
        sorted_comms.truncate(5);

        sorted_comms
            .into_iter()
            .map(|(_comm, mut nodes)| {
                nodes.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
                nodes.truncate(3);
                let seeds: Vec<u64> = nodes.into_iter().map(|(n, _)| n).collect();
                (format!("Follow-up round {}", round_num), base_score, seeds)
            })
            .collect()
    }
}

pub struct DriftSearchParams {
    pub n_depth: u32,
    pub k_followups: usize,
    pub top_k: usize,
    pub hops: u32,
    pub alpha: f64, // PPR damping
    pub confidence_threshold: f64,
    /// Maximum PPR iterations (convergence may exit earlier).
    pub max_ppr_iterations: u32,
    /// L1-norm convergence tolerance for PPR early exit.
    pub ppr_tolerance: f64,
    /// Follow-up score decay per round: `base_score * decay^round_num`.
    pub followup_decay: f64,
}

impl Default for DriftSearchParams {
    /// Canonical defaults. All surfaces should use this to stay consistent.
    fn default() -> Self {
        Self {
            n_depth: 2,
            k_followups: 3,
            top_k: 5,
            hops: 2,
            alpha: 0.85,
            confidence_threshold: 0.0,
            max_ppr_iterations: 30,
            ppr_tolerance: 1e-6,
            followup_decay: 0.7,
        }
    }
}

pub struct DriftSearchResult {
    pub all_discovered_nodes: Vec<u64>,
    pub actions: Vec<DriftAction>,
    pub ppr_scores: HashMap<u64, f64>,
    pub community_assignments: HashMap<u64, u64>,
}

/// Executes Personalized PageRank from the seed nodes with early convergence.
fn local_search_ppr(
    graph: &dyn GraphView,
    seeds: &[u64],
    params: &DriftSearchParams,
) -> (Vec<u64>, HashMap<u64, f64>) {
    let mut scores: HashMap<u64, f64> = HashMap::new();
    let mut new_scores: HashMap<u64, f64> = HashMap::new();
    let num_seeds = seeds.len() as f64;
    if num_seeds == 0.0 {
        return (Vec::new(), HashMap::new());
    }

    // Initialize seed scores
    for &seed in seeds {
        scores.insert(seed, 1.0 / num_seeds);
    }

    let damping = params.alpha;
    let mut ppr_neighbors: Vec<u64> = Vec::new();

    for _ in 0..params.max_ppr_iterations {
        new_scores.clear();
        // Add random jump probability to seeds
        for &seed in seeds {
            new_scores.insert(seed, (1.0 - damping) / num_seeds);
        }

        // Distribute PageRank along edges
        for (&u, &score) in &scores {
            let degree = graph.get_degree(u) as f64;

            if degree > 0.0 {
                let transfer = (damping * score) / degree;
                ppr_neighbors.clear();
                graph.get_neighbors_into(u, &mut ppr_neighbors);
                for &v in &ppr_neighbors {
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

        // L1-norm convergence check: early exit when scores stabilize
        let mut l1_diff = 0.0;
        for (node, &new_val) in &new_scores {
            let old_val = scores.get(node).copied().unwrap_or(0.0);
            l1_diff += (new_val - old_val).abs();
        }
        // Also account for nodes in old scores that dropped out
        for (node, &old_val) in &scores {
            if !new_scores.contains_key(node) {
                l1_diff += old_val;
            }
        }

        std::mem::swap(&mut scores, &mut new_scores);

        if l1_diff < params.ppr_tolerance {
            break;
        }
    }

    let mut ranked_nodes: Vec<(u64, f64)> = scores.clone().into_iter().collect();
    ranked_nodes.sort_by(|a, b| match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.1.total_cmp(&a.1),
    });
    ranked_nodes.truncate(params.top_k);

    let ranked = ranked_nodes.into_iter().map(|(n, _)| n).collect();
    (ranked, scores)
}

/// Lightweight label-propagation community detection for DRIFT's primer phase.
///
/// On a connected subgraph, BFS/CC produces a **single** community, making the
/// primer phase meaningless. Label propagation naturally finds dense clusters
/// within a connected component, giving DRIFT multiple communities to seed from.
///
/// Runs at most 15 iterations; convergence on small regional subgraphs is
/// typically 3-5 rounds.
fn label_propagation_communities(graph: &dyn GraphView, nodes: &[u64]) -> HashMap<u64, u64> {
    if nodes.is_empty() {
        return HashMap::new();
    }

    // Initialize: every node is its own community (using node id as label).
    let mut labels: HashMap<u64, u64> = nodes.iter().map(|&n| (n, n)).collect();
    let mut neighbors: Vec<u64> = Vec::new();

    for _ in 0..15 {
        let mut changed = false;
        for &node in nodes {
            // Count neighbor labels
            let mut label_counts: HashMap<u64, usize> = HashMap::new();
            neighbors.clear();
            graph.get_neighbors_into(node, &mut neighbors);
            for &neighbor in &neighbors {
                if let Some(&lbl) = labels.get(&neighbor) {
                    *label_counts.entry(lbl).or_default() += 1;
                }
            }
            if label_counts.is_empty() {
                continue;
            }
            // Pick the most frequent label (ties broken by smallest label id
            // for determinism).
            let best = label_counts
                .into_iter()
                .max_by_key(|&(lbl, count)| (count, std::cmp::Reverse(lbl)))
                .map(|(lbl, _)| lbl)
                .unwrap_or(node);
            if labels.get(&node) != Some(&best) {
                labels.insert(node, best);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    labels
}

pub fn execute_drift_search(
    query: &str,
    graph: &dyn GraphView,
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
    let mut all_ppr_scores: HashMap<u64, f64> = HashMap::new();

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
            // `action_id` came from `state.actions`, so the lookup is normally
            // present; skip defensively if the map changed under us.
            let seeds = match state.actions.get(&action_id) {
                Some(action) => action.query_seeds.clone(),
                None => continue,
            };

            // Execute local search (PPR)
            let (discovered, scores) = local_search_ppr(graph, &seeds, params);

            for (&node, &score) in &scores {
                *all_ppr_scores.entry(node).or_insert(0.0) += score;
            }

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

    // We don't have access to community_map inside execute_drift_search,
    // so we just return empty for community_assignments here.
    DriftSearchResult {
        all_discovered_nodes: all_discovered.into_iter().collect(),
        actions: state.actions.into_values().collect(),
        ppr_scores: all_ppr_scores,
        community_assignments: HashMap::new(),
    }
}

macro_rules! impl_dyn_traits {
    ($name:ident) => {
        impl PartialEq for $name {
            fn eq(&self, _other: &Self) -> bool {
                true
            }
        }
        impl Eq for $name {}
        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                std::any::type_name::<Self>().hash(state);
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct DriftSearchUDF {
    signature: Signature,
}
impl_dyn_traits!(DriftSearchUDF);

impl Default for DriftSearchUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl DriftSearchUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64,                                                     // source
                    DataType::UInt64,                                                     // target
                    DataType::Utf8,                                                       // query
                    DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))), // top_communities
                    DataType::UInt32,                                                     // n_depth
                    DataType::Utf8, // graph_uri
                    DataType::Utf8, // mode
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for DriftSearchUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "drift_search"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::UInt64,
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(DriftSearchAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.extend(vec![
            Arc::new(Field::new("query", DataType::Utf8, true)),
            Arc::new(Field::new(
                "top_communities",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("n_depth", DataType::UInt32, true)),
        ]);
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct DriftSearchAccumulator {
    base: GraphAccumulatorBase,
    query: Option<String>,
    top_communities: Option<Vec<u64>>,
    n_depth: Option<u32>,
}

impl Default for DriftSearchAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl DriftSearchAccumulator {
    pub fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            query: None,
            top_communities: None,
            n_depth: None,
        }
    }
}

impl Accumulator for DriftSearchAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut edge_state = self.base.edge_state()?;

        let mut comm_builder = ListBuilder::new(UInt64Builder::new());
        if let Some(ref comms) = self.top_communities {
            comm_builder.values().append_slice(comms);
            comm_builder.append(true);
        } else {
            comm_builder.append(false);
        }

        edge_state.push(ScalarValue::Utf8(self.query.clone()));
        edge_state.push(ScalarValue::List(Arc::new(comm_builder.finish())));
        edge_state.push(ScalarValue::UInt32(self.n_depth));

        Ok(edge_state)
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        // Columns 0, 1 = source, target; 5, 6 = graph_uri, mode
        self.base.update_edge_batch(values, Some(5), Some(6))?;

        if values.len() > 2 && !values[2].is_empty() && self.query.is_none() {
            if let Some(q_arr) = values[2].as_any().downcast_ref::<StringArray>() {
                if q_arr.is_valid(0) {
                    self.query = Some(q_arr.value(0).to_string());
                }
            }
        }

        if values.len() > 3 && !values[3].is_empty() && self.top_communities.is_none() {
            if let Some(list_arr) = values[3].as_any().downcast_ref::<ListArray>() {
                if list_arr.is_valid(0) {
                    let inner = list_arr.value(0);
                    if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                        self.top_communities = Some(u64_inner.values().to_vec());
                    }
                }
            }
        }

        if values.len() > 4 && !values[4].is_empty() && self.n_depth.is_none() {
            if let Some(depth_arr) = values[4].as_any().downcast_ref::<UInt32Array>() {
                if depth_arr.is_valid(0) {
                    self.n_depth = Some(depth_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // State layout: [0]=sources, [1]=targets, [2]=graph_uri, [3]=mode,
        //               [4]=query, [5]=top_communities, [6]=n_depth
        self.base.merge_edge_state(states, Some(2), Some(3))?;

        if states.len() > 4 && self.query.is_none() {
            if let Some(q_arr) = states[4].as_any().downcast_ref::<StringArray>() {
                for i in 0..q_arr.len() {
                    if q_arr.is_valid(i) {
                        self.query = Some(q_arr.value(i).to_string());
                        break;
                    }
                }
            }
        }

        if states.len() > 5 && self.top_communities.is_none() {
            if let Some(list_arr) = states[5].as_any().downcast_ref::<ListArray>() {
                for i in 0..list_arr.len() {
                    if list_arr.is_valid(i) {
                        let inner = list_arr.value(i);
                        if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                            self.top_communities = Some(u64_inner.values().to_vec());
                            break;
                        }
                    }
                }
            }
        }

        if states.len() > 6 && self.n_depth.is_none() {
            if let Some(depth_arr) = states[6].as_any().downcast_ref::<UInt32Array>() {
                for i in 0..depth_arr.len() {
                    if depth_arr.is_valid(i) {
                        self.n_depth = Some(depth_arr.value(i));
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut list_builder = ListBuilder::new(UInt64Builder::new());

        if self.base.is_empty() {
            list_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(list_builder.finish())));
        }
        let Some(query) = self.query.as_ref() else {
            list_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(list_builder.finish())));
        };

        let graph = self.base.resolve_graph(&[], 0)?;

        // Label-propagation community detection (much better than BFS/CC on
        // connected subgraphs where CC produces a single giant community).
        let all_nodes: Vec<u64> = {
            let mut ns: HashSet<u64> = HashSet::new();
            for (_, u, v) in self.base.edges() {
                ns.insert(u);
                ns.insert(v);
            }
            let mut v: Vec<u64> = ns.into_iter().collect();
            v.sort_unstable();
            v
        };
        let community_map = label_propagation_communities(graph.as_ref(), &all_nodes);

        let top_comm = if let Some(ref comms) = self.top_communities {
            comms.clone()
        } else {
            let mut comm_counts: HashMap<u64, usize> = HashMap::new();
            for &c in community_map.values() {
                *comm_counts.entry(c).or_default() += 1;
            }
            let mut comm_vec: Vec<(u64, usize)> = comm_counts.into_iter().collect();
            comm_vec.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
            comm_vec.into_iter().take(5).map(|(c, _)| c).collect()
        };

        let generator = HeuristicFollowUpGenerator {
            graph: graph.as_ref(),
            community_map: &community_map,
        };

        let params = DriftSearchParams {
            n_depth: self.n_depth.unwrap_or_default(),
            ..DriftSearchParams::default()
        };

        let result = execute_drift_search(query, graph.as_ref(), &top_comm, &generator, &params);
        let mut nodes = result.all_discovered_nodes;
        nodes.sort_unstable();

        list_builder.values().append_slice(&nodes);
        list_builder.append(true);

        Ok(ScalarValue::List(Arc::new(list_builder.finish())))
    }

    fn size(&self) -> usize {
        self.base.size() + self.query.as_ref().map(|q| q.capacity()).unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct RegionalDriftUDF {
    signature: Signature,
}
impl_dyn_traits!(RegionalDriftUDF);

impl Default for RegionalDriftUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionalDriftUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64,                                                     // source
                    DataType::UInt64,                                                     // target
                    DataType::Utf8,                                                       // query
                    DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))), // seeds
                    DataType::UInt32,                                                     // hops
                    DataType::UInt32,                                                     // n_depth
                    DataType::Utf8, // graph_uri
                    DataType::Utf8, // mode
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for RegionalDriftUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "regional_drift"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::UInt64,
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(RegionalDriftAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.extend(vec![
            Arc::new(Field::new("query", DataType::Utf8, true)),
            Arc::new(Field::new(
                "seeds",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("hops", DataType::UInt32, true)),
            Arc::new(Field::new("n_depth", DataType::UInt32, true)),
        ]);
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct RegionalDriftAccumulator {
    base: GraphAccumulatorBase,
    query: Option<String>,
    seeds: Option<Vec<u64>>,
    hops: Option<u32>,
    n_depth: Option<u32>,
}

impl Default for RegionalDriftAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionalDriftAccumulator {
    pub fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            query: None,
            seeds: None,
            hops: None,
            n_depth: None,
        }
    }
}

impl Accumulator for RegionalDriftAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut edge_state = self.base.edge_state()?;

        let mut seeds_builder = ListBuilder::new(UInt64Builder::new());
        if let Some(ref s) = self.seeds {
            seeds_builder.values().append_slice(s);
            seeds_builder.append(true);
        } else {
            seeds_builder.append(false);
        }

        edge_state.push(ScalarValue::Utf8(self.query.clone()));
        edge_state.push(ScalarValue::List(Arc::new(seeds_builder.finish())));
        edge_state.push(ScalarValue::UInt32(self.hops));
        edge_state.push(ScalarValue::UInt32(self.n_depth));

        Ok(edge_state)
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        // Columns 0, 1 = source, target; 6, 7 = graph_uri, mode
        self.base.update_edge_batch(values, Some(6), Some(7))?;

        if values.len() > 2 && !values[2].is_empty() && self.query.is_none() {
            if let Some(q_arr) = values[2].as_any().downcast_ref::<StringArray>() {
                if q_arr.is_valid(0) {
                    self.query = Some(q_arr.value(0).to_string());
                }
            }
        }

        if values.len() > 3 && !values[3].is_empty() && self.seeds.is_none() {
            if let Some(list_arr) = values[3].as_any().downcast_ref::<ListArray>() {
                if list_arr.is_valid(0) {
                    let inner = list_arr.value(0);
                    if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                        self.seeds = Some(u64_inner.values().to_vec());
                    }
                }
            }
        }

        if values.len() > 4 && !values[4].is_empty() && self.hops.is_none() {
            if let Some(arr) = values[4].as_any().downcast_ref::<UInt32Array>() {
                if arr.is_valid(0) {
                    self.hops = Some(arr.value(0));
                }
            }
        }

        if values.len() > 5 && !values[5].is_empty() && self.n_depth.is_none() {
            if let Some(arr) = values[5].as_any().downcast_ref::<UInt32Array>() {
                if arr.is_valid(0) {
                    self.n_depth = Some(arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // State layout: [0]=sources, [1]=targets, [2]=graph_uri, [3]=mode,
        //               [4]=query, [5]=seeds, [6]=hops, [7]=n_depth
        self.base.merge_edge_state(states, Some(2), Some(3))?;

        if states.len() > 4 && self.query.is_none() {
            if let Some(q_arr) = states[4].as_any().downcast_ref::<StringArray>() {
                for i in 0..q_arr.len() {
                    if q_arr.is_valid(i) {
                        self.query = Some(q_arr.value(i).to_string());
                        break;
                    }
                }
            }
        }

        if states.len() > 5 && self.seeds.is_none() {
            if let Some(list_arr) = states[5].as_any().downcast_ref::<ListArray>() {
                for i in 0..list_arr.len() {
                    if list_arr.is_valid(i) {
                        let inner = list_arr.value(i);
                        if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                            self.seeds = Some(u64_inner.values().to_vec());
                            break;
                        }
                    }
                }
            }
        }

        if states.len() > 6 && self.hops.is_none() {
            if let Some(arr) = states[6].as_any().downcast_ref::<UInt32Array>() {
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        self.hops = Some(arr.value(i));
                        break;
                    }
                }
            }
        }

        if states.len() > 7 && self.n_depth.is_none() {
            if let Some(arr) = states[7].as_any().downcast_ref::<UInt32Array>() {
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        self.n_depth = Some(arr.value(i));
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut list_builder = ListBuilder::new(UInt64Builder::new());

        if self.base.is_empty() {
            list_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(list_builder.finish())));
        }
        let (Some(query), Some(seeds)) = (
            self.query.as_ref(),
            self.seeds.as_ref().filter(|s| !s.is_empty()),
        ) else {
            list_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(list_builder.finish())));
        };
        let hops = self.hops.unwrap_or(1);

        let graph = self.base.resolve_graph(seeds, hops)?;

        // BFS expansion from seeds for `hops` over the resolved graph.
        let mut visited: HashSet<u64> = seeds.iter().copied().collect();
        let mut current_frontier: Vec<u64> = seeds.clone();

        let mut neighbors: Vec<u64> = Vec::new();
        for _ in 0..hops {
            let mut next_frontier = Vec::new();
            for &node in &current_frontier {
                neighbors.clear();
                graph.get_neighbors_into(node, &mut neighbors);
                for &nbr in &neighbors {
                    if visited.insert(nbr) {
                        next_frontier.push(nbr);
                        if visited.len() >= 50_000 {
                            break;
                        }
                    }
                }
                if visited.len() >= 50_000 {
                    break;
                }
            }
            current_frontier = next_frontier;
            if current_frontier.is_empty() || visited.len() >= 50_000 {
                break;
            }
        }

        let visited_vec: Vec<u64> = visited.iter().copied().collect();
        let community_map = label_propagation_communities(graph.as_ref(), &visited_vec);

        // Prioritize top communities by seed presence and degree
        let mut comm_scores: HashMap<u64, usize> = HashMap::new();
        for &s in seeds {
            if let Some(&c) = community_map.get(&s) {
                *comm_scores.entry(c).or_default() += 10;
            }
        }
        for (&node, &c) in &community_map {
            *comm_scores.entry(c).or_default() += graph.get_degree(node);
        }

        let mut sorted_comms: Vec<(u64, usize)> = comm_scores.into_iter().collect();
        sorted_comms.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        let top_communities: Vec<u64> = sorted_comms.into_iter().take(5).map(|(c, _)| c).collect();

        let generator = HeuristicFollowUpGenerator {
            graph: graph.as_ref(),
            community_map: &community_map,
        };

        let params = DriftSearchParams {
            n_depth: self.n_depth.unwrap_or(1),
            hops,
            ..DriftSearchParams::default()
        };

        let result =
            execute_drift_search(query, graph.as_ref(), &top_communities, &generator, &params);
        let mut nodes = result.all_discovered_nodes;
        nodes.sort_unstable();

        list_builder.values().append_slice(&nodes);
        list_builder.append(true);

        Ok(ScalarValue::List(Arc::new(list_builder.finish())))
    }

    fn size(&self) -> usize {
        self.base.size()
            + self.query.as_ref().map(|q| q.capacity()).unwrap_or(0)
            + self.seeds.as_ref().map(|s| s.capacity() * 8).unwrap_or(0)
    }
}
