// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Graph traversal and neighborhood analysis primitives for BenoStreamDB tables.
//!
//! Provides first-class graph querying over tables:
//! - High-speed zero-copy CSR (Compressed Sparse Row) index path (`.graph_v2.csr.*`)
//! - General table-scan BFS fallback for unindexed edge tables and relation-filtered queries

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{Context, Result};
use arrow::array::{Array, Int32Array, Int64Array, UInt32Array, UInt64Array};

use crate::core::cache::DiskCache;
use crate::core::index::csr_graph::{MmapCsrGraph, MultiSegmentCsrGraph};
use crate::core::manifest::ManifestManager;
use crate::core::table::Table;

/// Configuration options for graph neighborhood traversal.
#[derive(Debug, Clone, Default)]
pub struct GraphNeighborhoodOptions {
    /// Initial seed node IDs to expand from.
    pub seeds: Vec<u64>,
    /// Maximum traversal depth (number of hops from seeds).
    pub hops: u32,
    /// If true, traverse edges only in the forward (source -> target) direction.
    /// If false, traverse edges symmetrically (source <-> target).
    pub directed: bool,
    /// Optional whitelist of allowed edge relation/predicate strings.
    pub allowed_relations: Option<Vec<String>>,
    /// Name of the source column with an associated CSR graph index (if any).
    pub graph_column: Option<String>,
    /// Optional max-degree limit for combinatorial explosion control (super-node truncation).
    pub max_degree: Option<usize>,
    /// Optional token budget cap: stop expanding once visited set reaches this size.
    pub max_nodes: Option<usize>,
    /// Explicit source column name (defaults to auto-detect: "source", "src", "src_id").
    pub source_column: Option<String>,
    /// Explicit target column name (defaults to auto-detect: "target", "dst", "dst_id").
    pub target_column: Option<String>,
    /// Explicit relation column name (defaults to auto-detect: "relation", "predicate", "type", "rel", "edge_type").
    pub relation_column: Option<String>,
}

impl Table {
    /// Load memory-mapped [`MultiSegmentCsrGraph`] for `graph_column` from the table's
    /// index files (`{file}.graph_v2.csr.{offsets,edges,dict}`).
    ///
    /// Returns `Ok(None)` if no CSR graph index exists for that column.
    /// Load memory-mapped [`MultiSegmentCsrGraph`] for multiple columns in a single manifest pass.
    pub async fn load_graph_indices(
        &self,
        columns: &[&str],
    ) -> Result<HashMap<String, MultiSegmentCsrGraph>> {
        let manifest = self.manifest().await?;
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let entries = manifest_manager.load_all_entries(&manifest).await?;
        self.load_graph_indices_from_entries(&entries, columns)
            .await
    }

    async fn load_graph_indices_from_entries(
        &self,
        entries: &[crate::core::manifest::ManifestEntry],
        columns: &[&str],
    ) -> Result<HashMap<String, MultiSegmentCsrGraph>> {
        let mut col_segments: HashMap<String, Vec<MmapCsrGraph>> = HashMap::new();
        let cache = DiskCache::new(self.store.clone());

        for entry in entries {
            for idx in &entry.index_files {
                if idx.index_type == "graph_v2" {
                    if let Some(col) = idx.column_name.as_deref() {
                        if columns.contains(&col) {
                            let offsets_str = format!("{}.graph_v2.csr.offsets", idx.file_path);
                            let edges_str = format!("{}.graph_v2.csr.edges", idx.file_path);
                            let dict_str = format!("{}.graph_v2.csr.dict", idx.file_path);

                            if let (Ok(offsets_mmap), Ok(edges_mmap), Ok(dict_mmap)) = (
                                cache.get_mmap(&offsets_str).await,
                                cache.get_mmap(&edges_str).await,
                                cache.get_mmap(&dict_str).await,
                            ) {
                                col_segments.entry(col.to_string()).or_default().push(
                                    MmapCsrGraph::from_mmaps(offsets_mmap, edges_mmap, dict_mmap),
                                );
                            }
                        }
                    }
                }
            }
        }

        let mut res = HashMap::new();
        for (col, segs) in col_segments {
            if !segs.is_empty() {
                res.insert(col, MultiSegmentCsrGraph::new(segs));
            }
        }
        Ok(res)
    }

    /// Load memory-mapped [`MultiSegmentCsrGraph`] for `graph_column` from the table's
    /// index files (`{file}.graph_v2.csr.{offsets,edges,dict}`).
    ///
    /// Returns `Ok(None)` if no CSR graph index exists for that column.
    pub async fn load_graph_index(
        &self,
        graph_column: &str,
    ) -> Result<Option<MultiSegmentCsrGraph>> {
        let mut map = self.load_graph_indices(&[graph_column]).await?;
        Ok(map.remove(graph_column))
    }

    /// Compute the k-hop neighborhood from `options.seeds` over this table.
    ///
    /// Execution Strategy:
    /// 1. If `allowed_relations` is None and a CSR graph index exists for `graph_column`,
    ///    uses the zero-copy memory-mapped CSR fast path (`csr_bfs_visited`).
    /// 2. For undirected traversals on directed CSR graphs, the engine automatically checks for
    ///    a reverse index named `{graph_column}_rev` (e.g. `source_rev`). If not found,
    ///    it falls back to relational scan so results remain complete.
    /// 3. Otherwise, falls back to the relational table scan with optional predicate pushdown,
    ///    extracting source/target node IDs into an in-memory graph representation.
    pub async fn graph_neighborhood(&self, options: &GraphNeighborhoodOptions) -> Result<Vec<u64>> {
        if options.seeds.is_empty() {
            return Ok(Vec::new());
        }

        // Fast path: CSR index if available and no relation filtering is requested
        if options.allowed_relations.is_none() {
            if let Some(ref col) = options.graph_column {
                let rev_col = format!("{col}_rev");
                let cols_to_load: Vec<&str> = if !options.directed {
                    vec![col.as_str(), rev_col.as_str()]
                } else {
                    vec![col.as_str()]
                };

                let mut indices = self
                    .load_graph_indices(&cols_to_load)
                    .await
                    .unwrap_or_default();
                if let Some(forward) = indices.remove(col) {
                    let reverse = if !options.directed {
                        indices.remove(&rev_col)
                    } else {
                        None
                    };

                    // If undirected traversal was requested but no reverse CSR exists,
                    // fall through to relational path so traversal is complete.
                    if options.directed || reverse.is_some() {
                        return Ok(csr_bfs_visited(
                            &forward,
                            reverse.as_ref(),
                            &options.seeds,
                            options.hops,
                            options.directed,
                            options.max_degree,
                            options.max_nodes,
                        ));
                    }
                }
            }
        }

        // Relational table-scan path (supports allowed_relations filtering and unindexed tables)
        self.graph_neighborhood_scan(options).await
    }

    async fn graph_neighborhood_scan(
        &self,
        options: &GraphNeighborhoodOptions,
    ) -> Result<Vec<u64>> {
        let schema = self.arrow_schema();

        // 1. Resolve source and target columns
        let src_col_name = options.source_column.clone().unwrap_or_else(|| {
            ["source", "src", "src_id", "from", "u"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "source".to_string())
        });

        let tgt_col_name = options.target_column.clone().unwrap_or_else(|| {
            ["target", "dst", "dst_id", "to", "v"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "target".to_string())
        });

        // 2. Build relation filter if requested
        let rel_filter = if let Some(ref rels) = options.allowed_relations {
            let rel_col = options.relation_column.clone().or_else(|| {
                ["relation", "predicate", "type", "rel", "edge_type"]
                    .iter()
                    .find(|c| schema.column_with_name(c).is_some())
                    .map(|s| s.to_string())
            });

            if let Some(col) = rel_col {
                let vals: Vec<String> = rels
                    .iter()
                    .map(|r| format!("'{}'", r.replace('\'', "''")))
                    .collect();
                Some(format!("{col} IN ({})", vals.join(", ")))
            } else {
                None
            }
        } else {
            None
        };

        // 3. Scan the table with predicate pushdown
        let batches = self
            .read_async(rel_filter.as_deref(), None, None)
            .await
            .context("failed to read table batches for graph neighborhood scan")?;

        // 4. Construct adjacency list
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for batch in &batches {
            let b_schema = batch.schema();
            let src_idx = match b_schema.index_of(&src_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let tgt_idx = match b_schema.index_of(&tgt_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };

            let src_col = batch.column(src_idx);
            let tgt_col = batch.column(tgt_idx);

            let sources = extract_u64_values(src_col.as_ref());
            let targets = extract_u64_values(tgt_col.as_ref());

            for i in 0..batch.num_rows() {
                if let (Some(Some(s)), Some(Some(t))) =
                    (sources.get(i).copied(), targets.get(i).copied())
                {
                    adj.entry(s).or_default().push(t);
                    if !options.directed {
                        adj.entry(t).or_default().push(s);
                    }
                }
            }
        }

        // 5. Breadth-First Search (BFS)
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<(u64, u32)> = VecDeque::new();

        for &seed in &options.seeds {
            if visited.insert(seed) {
                queue.push_back((seed, 0));
            }
        }

        let budget_exhausted =
            |visited: &HashSet<u64>| options.max_nodes.is_some_and(|cap| visited.len() >= cap);

        while let Some((node, depth)) = queue.pop_front() {
            if budget_exhausted(&visited) {
                break;
            }
            if depth >= options.hops {
                continue;
            }

            if let Some(neighbors) = adj.get(&node) {
                if let Some(limit) = options.max_degree {
                    if neighbors.len() > limit {
                        continue;
                    }
                }

                for &nbr in neighbors {
                    if budget_exhausted(&visited) {
                        break;
                    }
                    if visited.insert(nbr) {
                        queue.push_back((nbr, depth + 1));
                    }
                }
            }
        }

        let mut out: Vec<u64> = visited.into_iter().collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Compute the shortest path between `start_node` and `end_node`.
    ///
    /// Fast path: If a CSR graph index exists, executes BFS over the CSR graph.
    /// For undirected traversals on directed graphs, the engine automatically checks for
    /// a reverse index named `{graph_column}_rev` (e.g. `source_rev`). If `graph_column` is `None`,
    /// defaults to `"source"`.
    ///
    /// Fallback path: If unindexed, performs table scan with relation/endpoint detection.
    /// Both paths are bounded by `MAX_SHORTEST_PATH_NODES` to avoid runaway memory on super-nodes.
    pub async fn shortest_path(
        &self,
        start_node: u64,
        end_node: u64,
        directed: bool,
        graph_column: Option<&str>,
    ) -> Result<Vec<u64>> {
        if start_node == end_node {
            return Ok(vec![start_node]);
        }

        const MAX_SHORTEST_PATH_NODES: usize = 100_000;
        const MAX_BFS_DEGREE: usize = 10_000;

        use crate::core::sql::graph_udf::drift_search::DriftGraph;

        // Fast path: CSR index
        let col_to_check = graph_column.unwrap_or("source");
        let rev_col = format!("{col_to_check}_rev");
        let cols_to_load: Vec<&str> = if !directed {
            vec![col_to_check, rev_col.as_str()]
        } else {
            vec![col_to_check]
        };

        let mut indices = self
            .load_graph_indices(&cols_to_load)
            .await
            .unwrap_or_default();
        if let Some(forward) = indices.remove(col_to_check) {
            let reverse = if !directed {
                indices.remove(&rev_col)
            } else {
                None
            };

            if directed || reverse.is_some() {
                let mut queue = VecDeque::new();
                let mut visited = HashMap::new();

                queue.push_back(start_node);
                visited.insert(start_node, start_node);

                let mut found = false;
                while let Some(current) = queue.pop_front() {
                    if current == end_node {
                        found = true;
                        break;
                    }

                    if visited.len() >= MAX_SHORTEST_PATH_NODES {
                        tracing::warn!(
                            "shortest_path exceeded maximum node expansion limit ({} nodes); aborting search",
                            MAX_SHORTEST_PATH_NODES
                        );
                        break;
                    }

                    for neighbor in forward
                        .get_neighbors(current)
                        .into_iter()
                        .take(MAX_BFS_DEGREE)
                    {
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            visited.entry(neighbor)
                        {
                            entry.insert(current);
                            queue.push_back(neighbor);
                        }
                    }

                    if let Some(ref rev) = reverse {
                        for neighbor in rev.get_neighbors(current).into_iter().take(MAX_BFS_DEGREE)
                        {
                            if let std::collections::hash_map::Entry::Vacant(entry) =
                                visited.entry(neighbor)
                            {
                                entry.insert(current);
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }

                if found {
                    let mut path = Vec::new();
                    let mut curr = end_node;
                    while curr != start_node {
                        path.push(curr);
                        curr = visited[&curr];
                    }
                    path.push(start_node);
                    path.reverse();
                    return Ok(path);
                } else {
                    return Ok(Vec::new());
                }
            }
        }

        // Fallback path: Table scan BFS
        let batches = self.read_async(None, None, None).await?;
        let schema = self.arrow_schema();
        let src_col_name = ["source", "src", "src_id", "from", "u"]
            .iter()
            .find(|c| schema.column_with_name(c).is_some())
            .unwrap_or(&"source");
        let tgt_col_name = ["target", "dst", "dst_id", "to", "v"]
            .iter()
            .find(|c| schema.column_with_name(c).is_some())
            .unwrap_or(&"target");

        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for batch in &batches {
            let b_schema = batch.schema();
            let src_idx = match b_schema.index_of(src_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let tgt_idx = match b_schema.index_of(tgt_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };

            let sources = extract_u64_values(batch.column(src_idx).as_ref());
            let targets = extract_u64_values(batch.column(tgt_idx).as_ref());

            for i in 0..batch.num_rows() {
                if let (Some(Some(s)), Some(Some(t))) =
                    (sources.get(i).copied(), targets.get(i).copied())
                {
                    adj.entry(s).or_default().push(t);
                    if !directed {
                        adj.entry(t).or_default().push(s);
                    }
                }
            }
        }

        let mut queue = VecDeque::new();
        let mut visited = HashMap::new();

        queue.push_back(start_node);
        visited.insert(start_node, start_node);

        let mut found = false;
        while let Some(current) = queue.pop_front() {
            if current == end_node {
                found = true;
                break;
            }

            if visited.len() >= MAX_SHORTEST_PATH_NODES {
                tracing::warn!(
                    "shortest_path scan fallback exceeded maximum node expansion limit ({} nodes); aborting search",
                    MAX_SHORTEST_PATH_NODES
                );
                break;
            }

            if let Some(neighbors) = adj.get(&current) {
                for &neighbor in neighbors.iter().take(MAX_BFS_DEGREE) {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        visited.entry(neighbor)
                    {
                        entry.insert(current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if found {
            let mut path = Vec::new();
            let mut curr = end_node;
            while curr != start_node {
                path.push(curr);
                curr = visited[&curr];
            }
            path.push(start_node);
            path.reverse();
            Ok(path)
        } else {
            Ok(Vec::new())
        }
    }

    /// Extract pairwise shortest connecting paths between seed nodes.
    pub async fn connecting_paths(
        &self,
        seeds: &[u64],
        directed: bool,
        graph_column: Option<&str>,
    ) -> Result<Vec<(u64, u64)>> {
        if seeds.len() < 2 {
            return Ok(Vec::new());
        }

        let mut edges = HashSet::new();

        for i in 0..seeds.len() {
            for j in (i + 1)..seeds.len() {
                let u = seeds[i];
                let v = seeds[j];

                let forward_path = self.shortest_path(u, v, directed, graph_column).await?;
                if forward_path.len() >= 2 {
                    for k in 0..forward_path.len() - 1 {
                        edges.insert((forward_path[k], forward_path[k + 1]));
                    }
                }

                if directed && forward_path.is_empty() {
                    let rev_path = self.shortest_path(v, u, directed, graph_column).await?;
                    if rev_path.len() >= 2 {
                        for k in 0..rev_path.len() - 1 {
                            edges.insert((rev_path[k], rev_path[k + 1]));
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u64, u64)> = edges.into_iter().collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Find reachable neighbors from `node` within `hops` steps.
    pub async fn graph_neighbors(
        &self,
        node: u64,
        hops: u32,
        graph_column: Option<&str>,
    ) -> Result<Vec<u64>> {
        let opts = GraphNeighborhoodOptions {
            seeds: vec![node],
            hops,
            directed: true,
            graph_column: graph_column.map(String::from),
            max_nodes: Some(100_000),
            max_degree: Some(10_000),
            ..Default::default()
        };
        let mut visited = self.graph_neighborhood(&opts).await?;
        visited.retain(|&x| x != node);
        visited.sort_unstable();
        visited.dedup();
        Ok(visited)
    }

    /// Extract induced subgraph edges within `hops` steps of `seeds`.
    pub async fn subgraph_edges(
        &self,
        seeds: &[u64],
        hops: u32,
        directed: bool,
        max_degree: Option<usize>,
        max_nodes: Option<usize>,
        graph_column: Option<&str>,
    ) -> Result<Vec<(u64, u64)>> {
        use crate::core::sql::graph_udf::drift_search::DriftGraph;

        let opts = GraphNeighborhoodOptions {
            seeds: seeds.to_vec(),
            hops,
            directed,
            max_degree,
            max_nodes,
            graph_column: graph_column.map(String::from),
            ..Default::default()
        };
        let visited_nodes = self.graph_neighborhood(&opts).await?;
        let node_set: HashSet<u64> = visited_nodes.into_iter().collect();

        if node_set.is_empty() {
            return Ok(Vec::new());
        }

        let mut edges = HashSet::new();
        let col_to_check = graph_column.unwrap_or("source");

        if let Ok(Some(forward)) = self.load_graph_index(col_to_check).await {
            for &u in &node_set {
                for v in forward.get_neighbors(u) {
                    if node_set.contains(&v) {
                        edges.insert((u, v));
                    }
                }
            }
        } else {
            let batches = self.read_async(None, None, None).await?;
            let schema = self.arrow_schema();
            let src_col_name = ["source", "src", "src_id", "from", "u"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .unwrap_or(&"source");
            let tgt_col_name = ["target", "dst", "dst_id", "to", "v"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .unwrap_or(&"target");

            for batch in &batches {
                let b_schema = batch.schema();
                let src_idx = match b_schema.index_of(src_col_name) {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                let tgt_idx = match b_schema.index_of(tgt_col_name) {
                    Ok(i) => i,
                    Err(_) => continue,
                };

                let sources = extract_u64_values(batch.column(src_idx).as_ref());
                let targets = extract_u64_values(batch.column(tgt_idx).as_ref());

                for i in 0..batch.num_rows() {
                    if let (Some(Some(s)), Some(Some(t))) =
                        (sources.get(i).copied(), targets.get(i).copied())
                    {
                        if node_set.contains(&s) && node_set.contains(&t) {
                            edges.insert((s, t));
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u64, u64)> = edges.into_iter().collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Execute regional DRIFT search around query `seeds` over this edge table.
    pub async fn regional_drift(
        &self,
        query: &str,
        seeds: &[u64],
        top_k: usize,
        hops: u32,
        n_depth: u32,
        k_followups: usize,
    ) -> Result<crate::core::sql::graph_udf::drift_search::DriftSearchResult> {
        use crate::core::sql::graph_udf::drift_search::{
            execute_drift_search, DriftGraph, DriftSearchParams, HeuristicFollowUpGenerator,
            SimpleGraph,
        };

        let edges = self
            .subgraph_edges(seeds, hops, false, Some(1000), Some(50000), None)
            .await?;

        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for (u, v) in &edges {
            adj.entry(*u).or_default().push(*v);
            adj.entry(*v).or_default().push(*u);
        }

        let graph = SimpleGraph { adjacency: adj };

        // Community partitioning: connected components over regional subgraph
        let mut community_map = HashMap::new();
        let mut current_comm = 0u64;

        let mut all_nodes = seeds.to_vec();
        for (u, v) in &edges {
            all_nodes.push(*u);
            all_nodes.push(*v);
        }

        for &node in &all_nodes {
            if community_map.contains_key(&node) {
                continue;
            }
            let mut q = VecDeque::new();
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

        // Top communities prioritized by seed presence and degree
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
        let top_communities: Vec<u64> = sorted_comms
            .into_iter()
            .take(top_k)
            .map(|(c, _)| c)
            .collect();

        let generator = HeuristicFollowUpGenerator {
            graph: &graph,
            community_map: &community_map,
        };

        let params = DriftSearchParams {
            n_depth,
            k_followups,
            top_k,
            hops,
            alpha: 0.85,
            confidence_threshold: 0.0,
        };

        Ok(execute_drift_search(
            query,
            &graph,
            &top_communities,
            &generator,
            &params,
        ))
    }
}

/// Multi-hop BFS over a CSR graph, returning the sorted visited node set (seeds included).
pub fn csr_bfs_visited(
    forward: &MultiSegmentCsrGraph,
    reverse: Option<&MultiSegmentCsrGraph>,
    seeds: &[u64],
    hops: u32,
    directed: bool,
    max_degree: Option<usize>,
    max_nodes: Option<usize>,
) -> Vec<u64> {
    use crate::core::sql::graph_udf::drift_search::DriftGraph;

    let mut visited: HashSet<u64> = seeds.iter().copied().collect();
    let mut frontier: Vec<u64> = visited.iter().copied().collect();

    let budget_exhausted = |visited: &HashSet<u64>| max_nodes.is_some_and(|l| visited.len() >= l);

    for _ in 0..hops {
        let mut next: Vec<u64> = Vec::new();
        for &node in &frontier {
            if budget_exhausted(&visited) {
                break;
            }
            if let Some(limit) = max_degree {
                let mut degree = forward.get_degree(node);
                if !directed {
                    if let Some(rev) = reverse {
                        degree += rev.get_degree(node);
                    }
                }
                if degree > limit {
                    continue;
                }
            }
            for n in forward.get_neighbors(node) {
                if budget_exhausted(&visited) {
                    break;
                }
                if visited.insert(n) {
                    next.push(n);
                }
            }
            if !directed {
                if let Some(rev) = reverse {
                    for n in rev.get_neighbors(node) {
                        if budget_exhausted(&visited) {
                            break;
                        }
                        if visited.insert(n) {
                            next.push(n);
                        }
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    let mut out: Vec<u64> = visited.into_iter().collect();
    out.sort_unstable();
    out
}

/// Extract numeric IDs into `Option<u64>` (supporting UInt64, Int64, UInt32, Int32).
fn extract_u64_values(col: &dyn Array) -> Vec<Option<u64>> {
    let n = col.len();
    let mut out = Vec::with_capacity(n);

    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else {
        out.resize(n, None);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_core_graph_neighborhood_traversal() -> Result<()> {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());
        let table = Table::new_async(uri).await?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::UInt64, false),
            Field::new("target", DataType::UInt64, false),
            Field::new("relation", DataType::Utf8, false),
        ]));

        // Graph edges:
        // 1 -> 2 (rel1)
        // 2 -> 3 (rel2)
        // 4 -> 5 (rel1)
        // 1 -> 4 (rel2)
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 4, 1])),
                Arc::new(UInt64Array::from(vec![2, 3, 5, 4])),
                Arc::new(StringArray::from(vec!["rel1", "rel2", "rel1", "rel2"])),
            ],
        )?;

        table.write_async(vec![batch]).await?;
        table.commit_async().await?;

        // 1. hops = 1, directed = true from seed 1 -> {1, 2, 4}
        let opts = GraphNeighborhoodOptions {
            seeds: vec![1],
            hops: 1,
            directed: true,
            ..Default::default()
        };
        let res = table.graph_neighborhood(&opts).await?;
        assert_eq!(res, vec![1, 2, 4]);

        // 2. hops = 1, directed = true, allowed_relations = ["rel1"] from seed 1 -> {1, 2}
        let opts_filtered = GraphNeighborhoodOptions {
            seeds: vec![1],
            hops: 1,
            directed: true,
            allowed_relations: Some(vec!["rel1".to_string()]),
            ..Default::default()
        };
        let res_filtered = table.graph_neighborhood(&opts_filtered).await?;
        assert_eq!(res_filtered, vec![1, 2]);

        // 3. hops = 2, directed = true from seed 1 -> {1, 2, 3, 4, 5}
        let opts_2hop = GraphNeighborhoodOptions {
            seeds: vec![1],
            hops: 2,
            directed: true,
            ..Default::default()
        };
        let res_2hop = table.graph_neighborhood(&opts_2hop).await?;
        assert_eq!(res_2hop, vec![1, 2, 3, 4, 5]);

        // 4. Undirected 1-hop from node 3 -> {2, 3} (2 -> 3 in edges)
        let opts_undirected = GraphNeighborhoodOptions {
            seeds: vec![3],
            hops: 1,
            directed: false,
            ..Default::default()
        };
        let res_undirected = table.graph_neighborhood(&opts_undirected).await?;
        assert_eq!(res_undirected, vec![2, 3]);

        Ok(())
    }
}
