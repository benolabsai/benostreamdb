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

use arrow::array::{
    Array, ArrayRef, ListArray, ListBuilder, StringArray, UInt32Array, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
    ranked_nodes.sort_by(|a, b| match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.1.total_cmp(&a.1),
    });
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
            // `action_id` came from `state.actions`, so the lookup is normally
            // present; skip defensively if the map changed under us.
            let seeds = match state.actions.get(&action_id) {
                Some(action) => action.query_seeds.clone(),
                None => continue,
            };

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
        Ok(vec![
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
            Arc::new(Field::new("query", DataType::Utf8, true)),
            Arc::new(Field::new(
                "top_communities",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("n_depth", DataType::UInt32, true)),
        ])
    }
}

#[derive(Debug)]
pub struct DriftSearchAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
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
            sources: Vec::new(),
            targets: Vec::new(),
            query: None,
            top_communities: None,
            n_depth: None,
        }
    }
}

impl Accumulator for DriftSearchAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut sources_builder = ListBuilder::new(UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);

        let mut targets_builder = ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);

        let mut comm_builder = ListBuilder::new(UInt64Builder::new());
        if let Some(ref comms) = self.top_communities {
            comm_builder.values().append_slice(comms);
            comm_builder.append(true);
        } else {
            comm_builder.append(false);
        }

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::Utf8(self.query.clone()),
            ScalarValue::List(Arc::new(comm_builder.finish())),
            ScalarValue::UInt32(self.n_depth),
        ])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
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

        for i in 0..sources.len() {
            if sources.is_valid(i) && targets.is_valid(i) {
                self.sources.push(sources.value(i));
                self.targets.push(targets.value(i));
            }
        }

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
                if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                    self.sources.extend_from_slice(u64_inner.values());
                }
            }
            if targets_list.is_valid(i) {
                let inner = targets_list.value(i);
                if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                    self.targets.extend_from_slice(u64_inner.values());
                }
            }
        }

        if states.len() > 2 && self.query.is_none() {
            if let Some(q_arr) = states[2].as_any().downcast_ref::<StringArray>() {
                for i in 0..q_arr.len() {
                    if q_arr.is_valid(i) {
                        self.query = Some(q_arr.value(i).to_string());
                        break;
                    }
                }
            }
        }

        if states.len() > 3 && self.top_communities.is_none() {
            if let Some(list_arr) = states[3].as_any().downcast_ref::<ListArray>() {
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

        if states.len() > 4 && self.n_depth.is_none() {
            if let Some(depth_arr) = states[4].as_any().downcast_ref::<UInt32Array>() {
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

        if self.sources.is_empty() {
            list_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(list_builder.finish())));
        }
        let Some(query) = self.query.as_ref() else {
            list_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(list_builder.finish())));
        };

        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&s, &t) in self.sources.iter().zip(self.targets.iter()) {
            adj.entry(s).or_default().push(t);
            adj.entry(t).or_default().push(s);
        }
        let graph = SimpleGraph { adjacency: adj };

        // Connected component community mapping
        let mut community_map: HashMap<u64, u64> = HashMap::new();
        let mut current_comm = 0u64;

        for &node in self.sources.iter().chain(self.targets.iter()) {
            if community_map.contains_key(&node) {
                continue;
            }
            let mut q = std::collections::VecDeque::new();
            q.push_back(node);
            community_map.insert(node, current_comm);

            while let Some(curr) = q.pop_front() {
                for neighbor in graph.get_neighbors(curr) {
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        community_map.entry(neighbor)
                    {
                        e.insert(current_comm);
                        q.push_back(neighbor);
                    }
                }
            }
            current_comm += 1;
        }

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
            graph: &graph,
            community_map: &community_map,
        };

        let params = DriftSearchParams {
            n_depth: self.n_depth.unwrap_or(2),
            k_followups: 3,
            top_k: 5,
            hops: 2,
            alpha: 0.85,
            confidence_threshold: 0.0,
        };

        let result = execute_drift_search(query, &graph, &top_comm, &generator, &params);
        let mut nodes = result.all_discovered_nodes;
        nodes.sort_unstable();

        list_builder.values().append_slice(&nodes);
        list_builder.append(true);

        Ok(ScalarValue::List(Arc::new(list_builder.finish())))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.sources.capacity() * 8
            + self.targets.capacity() * 8
            + self.query.as_ref().map(|q| q.capacity()).unwrap_or(0)
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
        Ok(vec![
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
            Arc::new(Field::new("query", DataType::Utf8, true)),
            Arc::new(Field::new(
                "seeds",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("hops", DataType::UInt32, true)),
            Arc::new(Field::new("n_depth", DataType::UInt32, true)),
        ])
    }
}

#[derive(Debug)]
pub struct RegionalDriftAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
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
            sources: Vec::new(),
            targets: Vec::new(),
            query: None,
            seeds: None,
            hops: None,
            n_depth: None,
        }
    }
}

impl Accumulator for RegionalDriftAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut sources_builder = ListBuilder::new(UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);

        let mut targets_builder = ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);

        let mut seeds_builder = ListBuilder::new(UInt64Builder::new());
        if let Some(ref s) = self.seeds {
            seeds_builder.values().append_slice(s);
            seeds_builder.append(true);
        } else {
            seeds_builder.append(false);
        }

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::Utf8(self.query.clone()),
            ScalarValue::List(Arc::new(seeds_builder.finish())),
            ScalarValue::UInt32(self.hops),
            ScalarValue::UInt32(self.n_depth),
        ])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
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

        for i in 0..sources.len() {
            if sources.is_valid(i) && targets.is_valid(i) {
                self.sources.push(sources.value(i));
                self.targets.push(targets.value(i));
            }
        }

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
                if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                    self.sources.extend_from_slice(u64_inner.values());
                }
            }
            if targets_list.is_valid(i) {
                let inner = targets_list.value(i);
                if let Some(u64_inner) = inner.as_any().downcast_ref::<UInt64Array>() {
                    self.targets.extend_from_slice(u64_inner.values());
                }
            }
        }

        if states.len() > 2 && self.query.is_none() {
            if let Some(q_arr) = states[2].as_any().downcast_ref::<StringArray>() {
                for i in 0..q_arr.len() {
                    if q_arr.is_valid(i) {
                        self.query = Some(q_arr.value(i).to_string());
                        break;
                    }
                }
            }
        }

        if states.len() > 3 && self.seeds.is_none() {
            if let Some(list_arr) = states[3].as_any().downcast_ref::<ListArray>() {
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

        if states.len() > 4 && self.hops.is_none() {
            if let Some(arr) = states[4].as_any().downcast_ref::<UInt32Array>() {
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        self.hops = Some(arr.value(i));
                        break;
                    }
                }
            }
        }

        if states.len() > 5 && self.n_depth.is_none() {
            if let Some(arr) = states[5].as_any().downcast_ref::<UInt32Array>() {
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

        if self.sources.is_empty() {
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

        // Build full adjacency
        let mut full_adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&s, &t) in self.sources.iter().zip(self.targets.iter()) {
            full_adj.entry(s).or_default().push(t);
            full_adj.entry(t).or_default().push(s);
        }

        // BFS expansion from seeds for hops
        let mut visited: HashSet<u64> = HashSet::new();
        let mut current_frontier: Vec<u64> = seeds.clone();
        for &s in seeds {
            visited.insert(s);
        }

        for _ in 0..hops {
            let mut next_frontier = Vec::new();
            for &node in &current_frontier {
                if let Some(nbrs) = full_adj.get(&node) {
                    for &nbr in nbrs {
                        if visited.insert(nbr) {
                            next_frontier.push(nbr);
                            if visited.len() >= 50_000 {
                                break;
                            }
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

        // Induce regional subgraph
        let mut regional_adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for &u in &visited {
            if let Some(nbrs) = full_adj.get(&u) {
                for &v in nbrs {
                    if visited.contains(&v) {
                        regional_adj.entry(u).or_default().push(v);
                    }
                }
            }
        }

        let regional_graph = SimpleGraph {
            adjacency: regional_adj,
        };

        // Community partitioning over regional subgraph
        let mut community_map: HashMap<u64, u64> = HashMap::new();
        let mut current_comm = 0u64;

        for &node in &visited {
            if community_map.contains_key(&node) {
                continue;
            }
            let mut q = std::collections::VecDeque::new();
            q.push_back(node);
            community_map.insert(node, current_comm);

            while let Some(curr) = q.pop_front() {
                for neighbor in regional_graph.get_neighbors(curr) {
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        community_map.entry(neighbor)
                    {
                        e.insert(current_comm);
                        q.push_back(neighbor);
                    }
                }
            }
            current_comm += 1;
        }

        // Prioritize top communities by seed presence and degree
        let mut comm_scores: HashMap<u64, usize> = HashMap::new();
        for &s in seeds {
            if let Some(&c) = community_map.get(&s) {
                *comm_scores.entry(c).or_default() += 10;
            }
        }
        for (&node, &c) in &community_map {
            *comm_scores.entry(c).or_default() += regional_graph.get_degree(node);
        }

        let mut sorted_comms: Vec<(u64, usize)> = comm_scores.into_iter().collect();
        sorted_comms.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        let top_communities: Vec<u64> = sorted_comms.into_iter().take(5).map(|(c, _)| c).collect();

        let generator = HeuristicFollowUpGenerator {
            graph: &regional_graph,
            community_map: &community_map,
        };

        let params = DriftSearchParams {
            n_depth: self.n_depth.unwrap_or(1),
            k_followups: 2,
            top_k: 5,
            hops,
            alpha: 0.85,
            confidence_threshold: 0.0,
        };

        let result = execute_drift_search(
            query,
            &regional_graph,
            &top_communities,
            &generator,
            &params,
        );
        let mut nodes = result.all_discovered_nodes;
        nodes.sort_unstable();

        list_builder.values().append_slice(&nodes);
        list_builder.append(true);

        Ok(ScalarValue::List(Arc::new(list_builder.finish())))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.sources.capacity() * 8
            + self.targets.capacity() * 8
            + self.query.as_ref().map(|q| q.capacity()).unwrap_or(0)
            + self.seeds.as_ref().map(|s| s.capacity() * 8).unwrap_or(0)
    }
}
